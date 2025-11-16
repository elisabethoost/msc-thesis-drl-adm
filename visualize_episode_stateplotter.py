#!/usr/bin/env python3
"""
Episode Visualization using StatePlotter - Based on notebooks and visualizations.py
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.patches as patches
import re

# parse_time_with_day_offset function copied from scripts.utils to avoid torch dependency
def parse_time_with_day_offset(time_str, reference_date):
    """
    Parses time and adds a day offset if '+1' is present, or if the arrival time 
    is earlier than the departure time (indicating a flight crosses midnight).
    
    Args:
        time_str: Either a string in 'HH:MM' format or a datetime object
        reference_date: Reference date for parsing
    """
    from datetime import datetime, timedelta
    
    # If time_str is already a datetime object, return it directly
    if isinstance(time_str, datetime):
        return time_str
    
    # Check if '+1' exists in the time string
    if '+1' in time_str:
        # Remove the '+1' and strip any whitespace
        time_str = time_str.replace('+1', '').strip()
        time_obj = datetime.strptime(time_str, '%H:%M')
        # Add 1 day to the time
        return datetime.combine(reference_date, time_obj.time()) + timedelta(days=1)
    else:
        # No '+1', parse the time normally
        time_obj = datetime.strptime(time_str, '%H:%M')
        parsed_time = datetime.combine(reference_date, time_obj.time())
        
        # If the parsed time is earlier than the reference time, it's the next day
        if parsed_time < reference_date:
            parsed_time += timedelta(days=1)
            
        return parsed_time

class EpisodeStatePlotter:
    """Episode visualizer using StatePlotter-style plotting"""
    
    def __init__(self, detailed_episode_data, env_type, seed, episode_number, scenario_folder):
        """Initialize the episode state plotter"""
        self.detailed_episode_data = detailed_episode_data
        self.env_type = env_type
        self.seed = seed
        self.episode_number = episode_number
        self.scenario_folder = scenario_folder
        
        # Extract data from the episode
        episode_data = detailed_episode_data[episode_number]
        scenario_data = episode_data["scenarios"][scenario_folder]
        
        # Extract initial state data
        self.initial_state = scenario_data["initial_state"]
        self.steps = scenario_data["steps"]
        
        # Extract dictionaries
        self.aircraft_dict = self.initial_state["aircraft_dict"]
        self.flights_dict = self.initial_state["flights_dict"]
        self.rotations_dict = self.initial_state["rotations_dict"]
        self.alt_aircraft_dict = self.initial_state["alt_aircraft_dict"]
        
        # Extract time information from config_dict
        config_dict = self.initial_state["config_dict"]
        self.start_datetime = datetime.strptime(
            config_dict['RecoveryPeriod']['StartDate'] + ' ' + config_dict['RecoveryPeriod']['StartTime'], 
            '%d/%m/%y %H:%M'
        )
        self.end_datetime = datetime.strptime(
            config_dict['RecoveryPeriod']['EndDate'] + ' ' + config_dict['RecoveryPeriod']['EndTime'], 
            '%d/%m/%y %H:%M'
        )
        
        # Set up offsets for visualization (matching StatePlotter defaults)
        self.offset_baseline = 0
        self.offset_delayed_flight = 0
        self.offset_marker_minutes = 4
        self.offset_id_number = -0.05
        
        # Calculate earliest and latest datetimes
        self.earliest_datetime = min(
            min(parse_time_with_day_offset(flight_info['DepTime'], self.start_datetime) for flight_info in self.flights_dict.values()),
            self.start_datetime
        )
        self.latest_datetime = max(
            max(parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime) for flight_info in self.flights_dict.values()),
            self.end_datetime
        )
    
    def get_aircraft_indices(self, swapped_flights=None, rotations_dict=None):
        """Get aircraft indices for plotting - matching StatePlotter logic"""
        if swapped_flights is None:
            swapped_flights = []
        if rotations_dict is None:
            rotations_dict = self.rotations_dict
            
        updated_rotations_dict = rotations_dict.copy()
        for swap in swapped_flights:
            flight_id, new_aircraft_id = swap
            if flight_id in updated_rotations_dict:
                updated_rotations_dict[flight_id]['Aircraft'] = new_aircraft_id

        # Define the sorting key function (matching StatePlotter)
        def extract_sort_key(aircraft_id):
            letters = ''.join(re.findall(r'[A-Za-z]+', aircraft_id))
            numbers = tuple(int(num) for num in re.findall(r'\d+', aircraft_id))
            return (letters, ) + numbers

        # Collect and sort aircraft IDs using the custom sort key
        all_aircraft_ids = sorted(list(set([rotation_info['Aircraft'] for rotation_info in updated_rotations_dict.values()]).union(set(self.aircraft_dict.keys()))))
        aircraft_indices = {aircraft_id: index + 1 for index, aircraft_id in enumerate(all_aircraft_ids)}
        
        return all_aircraft_ids, aircraft_indices, updated_rotations_dict
    
    def plot_state(self, ax, state_data, title_suffix="", show_plot=True):
        """
        Plot state using StatePlotter-style visualization
        """
        # Extract state information
        swapped_flights = state_data["swapped_flights"]
        environment_delayed_flights = state_data["environment_delayed_flights"]
        cancelled_flights = state_data["cancelled_flights"]
        current_datetime = state_data["current_datetime"]
        
        # Use updated flights_dict and rotations_dict from step if available, otherwise use initial
        flights_dict = state_data.get("flights_dict", self.flights_dict)
        rotations_dict = state_data.get("rotations_dict", self.rotations_dict)
        
        all_aircraft_ids, aircraft_indices, updated_rotations_dict = self.get_aircraft_indices(swapped_flights, rotations_dict)
        
        # Set up date formatter and locator
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        # Debug: Print time range for Step 4 (moved after earliest_time is defined)
        
        labels = {
            'Scheduled Flight': False,
            'Swapped Flight': False,
            'Environment Delayed Flight': False,
            'Cancelled Flight': False,
            'Aircraft Unavailable': False,
            'Aircraft Unavailable (Confirmed)': False,  # Added for prob == 1.0
            'Uncertain Breakdown': False,
            'Zero Probability': False,
            'Zero Probability (Resolved: Did Not Occur)': False,  # Added for prob == 0.0 after resolution
            'Current Action Flight': False,
            'Last Action Flight': False
        }

        earliest_time = self.earliest_datetime
        latest_time = self.latest_datetime

        # Plot flights using updated rotations (reflecting swaps)
        for rotation_id, rotation_info in updated_rotations_dict.items():
            flight_id = rotation_id
            aircraft_id = rotation_info['Aircraft']
            
            if flight_id in flights_dict:
                flight_info = flights_dict[flight_id]
                dep_datetime_str = flight_info['DepTime']
                arr_datetime_str = flight_info['ArrTime']
                
                dep_datetime = parse_time_with_day_offset(dep_datetime_str, self.start_datetime)
                arr_datetime = parse_time_with_day_offset(arr_datetime_str, dep_datetime)
                
                # Handle flights crossing midnight properly (matching StatePlotter)
                if '+1' in dep_datetime_str and '+1' in arr_datetime_str:
                    # Both departure and arrival are on next day
                    dep_datetime = self.start_datetime + timedelta(days=1)
                    dep_datetime = dep_datetime.replace(hour=int(dep_datetime_str.split(':')[0]), 
                                                     minute=int(dep_datetime_str.split(':')[1].split('+')[0]))
                    arr_datetime = dep_datetime + timedelta(days=0)  # Same day as departure
                    arr_datetime = arr_datetime.replace(hour=int(arr_datetime_str.split(':')[0]), 
                                                     minute=int(arr_datetime_str.split(':')[1].split('+')[0]))
                elif '+1' in dep_datetime_str:
                    # Only departure is next day
                    dep_datetime = self.start_datetime + timedelta(days=1)
                    dep_datetime = dep_datetime.replace(hour=int(dep_datetime_str.split(':')[0]), 
                                                     minute=int(dep_datetime_str.split(':')[1].split('+')[0]))
                    arr_datetime = dep_datetime
                    arr_datetime = arr_datetime.replace(hour=int(arr_datetime_str.split(':')[0]), 
                                                     minute=int(arr_datetime_str.split(':')[1]))
                elif '+1' in arr_datetime_str:
                    # Only arrival is next day relative to departure
                    dep_datetime = dep_datetime.replace(hour=int(dep_datetime_str.split(':')[0]), 
                                                     minute=int(dep_datetime_str.split(':')[1]))
                    arr_datetime = dep_datetime + timedelta(days=1)
                    arr_datetime = arr_datetime.replace(hour=int(arr_datetime_str.split(':')[0]), 
                                                     minute=int(arr_datetime_str.split(':')[1].split('+')[0]))
                
                # Recalculate earliest and latest times after fixing arr_datetime
                earliest_time = min(earliest_time, dep_datetime)
                latest_time = max(latest_time, arr_datetime)
                
                
                swapped = any(flight_id == swap[0] for swap in swapped_flights)
                delayed = flight_id in environment_delayed_flights
                cancelled = flight_id in cancelled_flights
                
                
                # Note: flights_dict already contains updated times from update_flight_times(),
                # so we don't need to apply delays again. The delay tracking is just for information.
                # However, if flights_dict is not available in step_info (old format), we apply delays.
                if "flights_dict" not in state_data and delayed:
                    # Old format: apply delays manually
                    delay_info = environment_delayed_flights[flight_id]
                    delay_hours = 0
                    
                    # Handle different delay formats
                    if isinstance(delay_info, (int, float)) and delay_info > 0:
                        # If delay is a reasonable number (likely minutes), convert to hours
                        if delay_info < 1000:  # Likely minutes
                            delay_hours = delay_info / 60.0
                        else:  # Likely hours
                            delay_hours = delay_info
                    elif isinstance(delay_info, dict):
                        if 'delay_hours' in delay_info and delay_info['delay_hours'] > 0:
                            delay_hours = delay_info['delay_hours']
                        elif 'delay_minutes' in delay_info and delay_info['delay_minutes'] > 0:
                            delay_hours = delay_info['delay_minutes'] / 60.0
                        elif 'delay' in delay_info and delay_info['delay'] > 0:
                            # Assume delay is in minutes if it's a reasonable number
                            if delay_info['delay'] < 1000:  # Likely minutes
                                delay_hours = delay_info['delay'] / 60.0
                            else:  # Likely hours
                                delay_hours = delay_info['delay']
                    
                    if delay_hours > 0:
                        dep_datetime += timedelta(hours=delay_hours)
                        arr_datetime += timedelta(hours=delay_hours)
                    else:
                        # If delay is 0 or negative, don't treat as delayed
                        delayed = False
                elif "flights_dict" in state_data:
                    # New format: flights_dict already has updated times, so no need to apply delays
                    # The dep_datetime and arr_datetime from flights_dict are already correct
                    pass
                
                if cancelled:
                    plot_color = 'red'
                    plot_label = 'Cancelled Flight'
                elif swapped:
                    plot_color = 'blue'
                    plot_label = 'Swapped Flight'
                elif delayed:
                    plot_color = 'orange'
                    plot_label = 'Environment Delayed Flight'
                else:
                    plot_color = 'blue'
                    plot_label = 'Scheduled Flight'
                
                y_offset = aircraft_indices[aircraft_id] + self.offset_baseline
                if delayed:
                    y_offset += self.offset_delayed_flight

                ax.plot([dep_datetime, arr_datetime], [y_offset, y_offset], color=plot_color, 
                       label=plot_label if not labels[plot_label] else None, linewidth=2)
                
                marker_offset = timedelta(minutes=self.offset_marker_minutes)
                dep_marker = dep_datetime + marker_offset
                arr_marker = arr_datetime - marker_offset

                ax.plot(dep_marker, y_offset, color=plot_color, marker='>', markersize=6, markeredgewidth=0)
                ax.plot(arr_marker, y_offset, color=plot_color, marker='<', markersize=6, markeredgewidth=0)

                if delayed:
                    ax.vlines([dep_datetime, arr_datetime], y_offset - self.offset_delayed_flight, y_offset, 
                             color='orange', linestyle='-', linewidth=2)

                labels[plot_label] = True
                
                mid_datetime = dep_datetime + (arr_datetime - dep_datetime) / 2
                ax.text(mid_datetime, y_offset + self.offset_id_number, flight_id, 
                        ha='center', va='bottom', fontsize=10, color='black')

        # Plot disruptions (matching StatePlotter logic) - pass state_data for step-specific probabilities
        self.plot_disruptions(ax, aircraft_indices, labels, state_data)
        
        # Add recovery period markers (matching StatePlotter)
        ax.axvline(self.start_datetime, color='green', linestyle='--', linewidth=2, label='Start/End Recovery Period')
        ax.axvline(self.end_datetime, color='green', linestyle='--', linewidth=2)
        ax.axvspan(self.end_datetime, latest_time + timedelta(hours=1), color='lightgrey', alpha=0.3)
        ax.axvspan(earliest_time - timedelta(hours=1), self.start_datetime, color='lightgrey', alpha=0.3)
        
        # Add current time line
        ax.axvline(current_datetime, color='black', linestyle='-', linewidth=2, label='Current Time')
        
        # Set up plot
        buffer_time = timedelta(hours=1)
        ax.set_xlim(earliest_time - buffer_time, latest_time + buffer_time)
        
        ax.invert_yaxis()
        ytick_labels = [f"{index + 1}: {aircraft_id}" for index, aircraft_id in enumerate(all_aircraft_ids)]
        plt.yticks(range(1, len(all_aircraft_ids) + 1), ytick_labels)
        
        plt.xlabel('Time')
        plt.ylabel('Aircraft')
        
        # Create a more prominent title
        if "Step" in title_suffix:
            # Extract step number and action from title_suffix
            step_part = title_suffix.split(" - Step ")[1] if " - Step " in title_suffix else ""
            step_title = f"Step {step_part}" if step_part else title_suffix
            ax.set_title(step_title, fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Episode {self.episode_number + 1} ({self.env_type}){title_suffix}', fontsize=12)
        
        plt.grid(True)
        plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
    
    def plot_disruptions(self, ax, aircraft_indices, labels, state_data=None):
        """Plot aircraft disruptions - matching StatePlotter logic
        
        Args:
            state_data: If provided (from step_info), use updated probabilities from that step.
                       Otherwise, use initial probabilities from alt_aircraft_dict.
        """
        def get_height_in_data_units(ax, pixels):
            y0_display = ax.transData.transform((0, 0))[1]
            y1_display = ax.transData.transform((0, 1))[1]
            pixels_per_data_unit = abs(y1_display - y0_display)
            data_units_per_pixel = 1 / pixels_per_data_unit
            return data_units_per_pixel * pixels

        rect_height = get_height_in_data_units(ax, 150)
        
        # Get probabilities from state_data if available, otherwise use initial state
        unavailabilities_probabilities = None
        if state_data and "unavailabilities_probabilities" in state_data:
            unavailabilities_probabilities = state_data["unavailabilities_probabilities"]
        
        # Handle alt_aircraft_dict unavailabilities
        if self.alt_aircraft_dict:
            for aircraft_id, unavailability_info in self.alt_aircraft_dict.items():
                if not isinstance(unavailability_info, list):
                    unavailability_info = [unavailability_info]
                
                for unavail in unavailability_info:
                    start_date = unavail['StartDate']
                    start_time = unavail['StartTime']
                    end_date = unavail['EndDate']
                    end_time = unavail['EndTime']
                    
                    # Use step-specific probability if available, otherwise use initial probability
                    if unavailabilities_probabilities and aircraft_id in unavailabilities_probabilities:
                        prob_info = unavailabilities_probabilities[aircraft_id]
                        probability = prob_info.get('probability', 1.0)
                        # If probability is None (nan), treat as 0.0
                        if probability is None:
                            probability = 0.0
                    else:
                        probability = unavail.get('Probability', 1.0)
                    
                    # Convert to datetime objects
                    unavail_start = datetime.strptime(f"{start_date} {start_time}", '%d/%m/%y %H:%M')
                    unavail_end = datetime.strptime(f"{end_date} {end_time}", '%d/%m/%y %H:%M')
                    y_offset = aircraft_indices[aircraft_id]

                    if np.isnan(probability):
                        probability = 0.0
                    
                    # Set color based on probability - use updated probability from step
                    if probability == 0.0:
                        rect_color = 'lightgrey'
                        plot_label = 'Zero Probability (Resolved: Did Not Occur)'
                    elif probability < 1.0:
                        rect_color = 'orange'
                        plot_label = 'Uncertain Breakdown'
                    else:
                        rect_color = 'red'
                        plot_label = 'Aircraft Unavailable (Confirmed)'
                    
                    # Plot the unavailability period as a rectangle
                    rect = patches.Rectangle((unavail_start, y_offset - rect_height / 2),
                                            unavail_end - unavail_start,
                                            rect_height,
                                            linewidth=0,
                                            color=rect_color,
                                            alpha=0.3,
                                            zorder=0,
                                            label=plot_label if not labels[plot_label] else None)
                    ax.add_patch(rect)
                    labels[plot_label] = True

                    # Plot the probability below the rectangle - show evolution if different from initial
                    initial_prob = unavail.get('Probability', 1.0)
                    if np.isnan(initial_prob):
                        initial_prob = 0.0
                    
                    # Show probability if it's uncertain (< 1.0) or if it changed from initial
                    if probability < 1.0 or (unavailabilities_probabilities and abs(probability - initial_prob) > 0.01):
                        x_position = unavail_start + (unavail_end - unavail_start) / 2
                        y_position = y_offset - rect_height / 2 - get_height_in_data_units(ax, 10) + 0.2
                        prob_text = f"{probability:.2f}"
                        # If probability changed, show both initial and current
                        if unavailabilities_probabilities and abs(probability - initial_prob) > 0.01:
                            prob_text = f"{initial_prob:.2f}→{probability:.2f}"
                        ax.text(x_position, y_position + 0.1, prob_text, ha='center', va='top', fontsize=9, fontweight='bold')
    
    def create_episode_visualization(self, save_path=None, max_steps_per_plot=3):
        """Create episode visualization with multiple plots"""
        num_steps = len(self.steps)
        
        if num_steps == 0:
            print("No steps to visualize")
            return None
        
        # Create multiple plots with max_steps_per_plot steps each
        num_plots = (num_steps + max_steps_per_plot - 1) // max_steps_per_plot
        figures = []
        
        for plot_idx in range(num_plots):
            start_step = plot_idx * max_steps_per_plot
            end_step = min(start_step + max_steps_per_plot, num_steps)
            steps_in_plot = end_step - start_step
            
            # Create figure for this plot with reasonable size
            fig, axes = plt.subplots(steps_in_plot + 1, 1, figsize=(14, 4 * (steps_in_plot + 1)), dpi=100)
            
            if steps_in_plot == 0:
                axes = [axes]  # Make it a list for consistency
            
            # Plot initial state
            self.plot_state(axes[0], self.initial_state, " - Initial State")
            
            # Plot each step in this plot
            for i in range(start_step, end_step):
                step_info = self.steps[i]
                step_num = i + 1
                action_desc = self.get_action_description(step_info)
                
                # Add flight times info to title if flights_dict is available
                flights_dict = step_info.get("flights_dict")
                if flights_dict and i > 0:
                    # Show which flights were moved and their new times
                    prev_step = self.steps[i - 1] if i > 0 else None
                    flight_action = step_info.get("flight_action")
                    aircraft_action = step_info.get("aircraft_action")
                    if flight_action and aircraft_action and flight_action != 0 and aircraft_action != 0:
                        if flight_action in flights_dict:
                            new_dep = flights_dict[flight_action]['DepTime']
                            new_arr = flights_dict[flight_action]['ArrTime']
                            action_desc += f" | New: {new_dep}-{new_arr}"
                
                step_title = f" - Step {step_num}: {action_desc}"
                self.plot_state(axes[i - start_step + 1], step_info, step_title)
            
            plt.tight_layout()
            
            # Save individual plot
            if save_path:
                base_path = save_path.replace('.png', '')
                plot_save_path = f"{base_path}_plot_{plot_idx + 1}_of_{num_plots}.png"
                plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
                print(f"Plot {plot_idx + 1}/{num_plots} saved to: {plot_save_path}")
            
            figures.append(fig)
        
        return figures
    
    def print_episode_summary(self):
        """Print a summary of all actions taken in the episode"""
        print(f"\nEpisode {self.episode_number + 1} Action Summary:")
        print("=" * 50)
        
        for i, step_info in enumerate(self.steps):
            step_num = i + 1
            action_desc = self.get_action_description(step_info)
            reward = step_info["reward"]
            epsilon = step_info["epsilon"]
            print(f"Step {step_num}: {action_desc} | Reward: {reward:.2f} | Epsilon: {epsilon:.3f}")
        
        print("=" * 50)
    
    def get_action_description(self, step_info):
        """Get a human-readable description of the action taken"""
        action_reason = step_info.get("action_reason", "unknown")
        
        # Use decoded action if available, otherwise decode from index
        if "flight_action" in step_info and "aircraft_action" in step_info:
            flight_action = step_info["flight_action"]
            aircraft_action = step_info["aircraft_action"]
        else:
            # Fallback: decode from action index
            action_index = step_info["action"]
            flight_action, aircraft_action = self.map_index_to_action(action_index)
        
        # Create action description
        if flight_action == 0 and aircraft_action == 0:
            action_desc = "No Action (0, 0)"
        elif flight_action == 0:
            action_desc = f"Cancel on AC{aircraft_action} (0, {aircraft_action})"
        elif aircraft_action == 0:
            action_desc = f"Cancel Flight {flight_action} ({flight_action}, 0)"
        else:
            action_desc = f"Flight {flight_action} -> AC{aircraft_action} ({flight_action}, {aircraft_action})"
        
        return action_desc
    
    def map_index_to_action(self, index):
        """Map action index to (flight_action, aircraft_action) - simplified version"""
        # Get the number of aircraft from the aircraft_dict
        num_aircraft = len(self.aircraft_dict)
        flight_action = index // (num_aircraft + 1)
        aircraft_action = index % (num_aircraft + 1)
        return flight_action, aircraft_action

def load_and_visualize_episode(save_folder, env_type, seed, episode_number, scenario_folder=None):
    """Load detailed episode data and create visualization"""
    
    # Load detailed episode data
    detailed_episodes_path = f"{save_folder}/detailed_episodes/{env_type}_detailed_episodes_seed_{seed}.pkl"
    
    if not os.path.exists(detailed_episodes_path):
        print(f"ERROR: Detailed episodes file not found at {detailed_episodes_path}")
        return
    
    try:
        with open(detailed_episodes_path, 'rb') as f:
            detailed_episode_data = pickle.load(f)
        
        print(f"Successfully loaded detailed episode data")
        
        # Check if the requested episode exists
        if episode_number not in detailed_episode_data:
            print(f"ERROR: Episode {episode_number} not found")
            print(f"Available episodes: {list(detailed_episode_data.keys())}")
            return
        
        # Get scenario folder
        if scenario_folder is None:
            episode_data = detailed_episode_data[episode_number]
            available_scenarios = list(episode_data["scenarios"].keys())
            if not available_scenarios:
                print(f"ERROR: No scenarios found for episode {episode_number}")
                return
            scenario_folder = available_scenarios[0]
            print(f"Using first available scenario: {scenario_folder}")
        
        # Create visualizer
        visualizer = EpisodeStatePlotter(detailed_episode_data, env_type, seed, episode_number, scenario_folder)
        
        # Create output directory
        output_dir = f"{save_folder}/visualizations/episode1_{episode_number}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization
        save_path = f"{output_dir}/stateplotter_visualization_{env_type}_seed_{seed}.png"
        figures = visualizer.create_episode_visualization(save_path)
        
        # Print episode summary
        visualizer.print_episode_summary()
        
        print(f"Visualization completed successfully!")
        print(f"Episode {episode_number + 1} Statistics:")
        print(f"Environment Type: {env_type}")
        print(f"Seed: {seed}")
        print(f"Scenario: {scenario_folder}")
        print(f"Total Steps: {len(visualizer.steps)}")
        
        return figures
        
    except Exception as e:
        print(f"ERROR: Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    
    # Configuration - can be overridden via command-line arguments
    # Usage: python visualize_episode_stateplotter.py [save_folder] [env_type] [seed] [episode_number] [scenario_folder]
    
    if len(sys.argv) >= 5:
        save_folder = sys.argv[1]
        env_type = sys.argv[2]
        seed = int(sys.argv[3])
        episode_number = int(sys.argv[4])
        scenario_folder = sys.argv[5] if len(sys.argv) > 5 else None
    else:
        # Default values (update these for your current run)
        save_folder = "Save_Trained_Models40/3ac-130-green"
        env_type = "proactive"
        seed = 232323
        episode_number = 5  # 0-indexed
        scenario_folder = "Data/TRAINING/3ac-130-green/stochastic_Scenario_00088"  # Can be None to use first available
    
    print(f"Visualization Configuration:")
    print(f"   Save Folder: {save_folder}")
    print(f"   Environment Type: {env_type}")
    print(f"   Seed: {seed}")
    print(f"   Episode Number: {episode_number} (Episode {episode_number + 1} in display)")
    if scenario_folder:
        print(f"   Scenario: {scenario_folder}")
    else:
        print(f"   Scenario: (will use first available)")
    print()
    
    # Load and visualize
    load_and_visualize_episode(save_folder, env_type, seed, episode_number, scenario_folder)
