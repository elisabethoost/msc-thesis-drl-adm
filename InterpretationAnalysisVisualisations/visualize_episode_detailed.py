#!/usr/bin/env python3
"""
Detailed Episode and Scenario Visualization
Combines schedule visualization with step-by-step metrics analysis

This script provides:
1. Visual representation of the initial schedule and each step
2. Epsilon values and rewards at each step
3. Final statistics (delays, tail swaps, cancellations, etc.)

Usage:
    python InterpretationAnalysisVisualisations/visualize_episode_detailed.py [save_folder] [env_type] [seed] [episode_num] [scenario_folder]
    OR
    cd InterpretationAnalysisVisualisations
    python visualize_episode_detailed.py [save_folder] [env_type] [seed] [episode_num] [scenario_folder]
    
Example:
    python InterpretationAnalysisVisualisations/visualize_episode_detailed.py Final_Model_20_wPenalty10/3ac-130-green proactive 232323 0 "Data/TRAINING/3ac-130-green/stochastic_Scenario_00052"
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.patches as patches
import re

# Get the script's directory and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Parent of InterpretationAnalysisVisualisations

# Add parent directory to path so we can import from src
sys.path.insert(0, PROJECT_ROOT)

# Parse time function (same as in visualize_episode_stateplotter.py)
def parse_time_with_day_offset(time_str, reference_date):
    """Parse time and add day offset if needed."""
    if isinstance(time_str, datetime):
        return time_str
    
    if '+1' in time_str:
        time_str = time_str.replace('+1', '').strip()
        time_obj = datetime.strptime(time_str, '%H:%M')
        return datetime.combine(reference_date, time_obj.time()) + timedelta(days=1)
    else:
        time_obj = datetime.strptime(time_str, '%H:%M')
        parsed_time = datetime.combine(reference_date, time_obj.time())
        if parsed_time < reference_date:
            parsed_time += timedelta(days=1)
        return parsed_time


class DetailedEpisodeVisualizer:
    """Comprehensive episode visualizer combining schedule plots and metrics."""
    
    def __init__(self, detailed_episode_data, env_type, seed, episode_number, scenario_folder):
        """Initialize the detailed visualizer."""
        self.detailed_episode_data = detailed_episode_data
        self.env_type = env_type
        self.seed = seed
        self.episode_number = episode_number
        self.scenario_folder = scenario_folder
        
        # Extract data
        episode_data = detailed_episode_data[episode_number]
        scenario_data = episode_data["scenarios"][scenario_folder]
        
        # Extract initial state and steps
        self.initial_state = scenario_data["initial_state"]
        self.steps = scenario_data["steps"]
        self.final_metrics = scenario_data.get("final_scenario_metrics", {})
        
        # Extract dictionaries
        self.aircraft_dict = self.initial_state["aircraft_dict"]
        self.flights_dict = self.initial_state["flights_dict"]
        self.rotations_dict = self.initial_state["rotations_dict"]
        self.alt_aircraft_dict = self.initial_state["alt_aircraft_dict"]
        
        # Extract time information
        config_dict = self.initial_state["config_dict"]
        self.start_datetime = datetime.strptime(
            config_dict['RecoveryPeriod']['StartDate'] + ' ' + config_dict['RecoveryPeriod']['StartTime'], 
            '%d/%m/%y %H:%M'
        )
        self.end_datetime = datetime.strptime(
            config_dict['RecoveryPeriod']['EndDate'] + ' ' + config_dict['RecoveryPeriod']['EndTime'], 
            '%d/%m/%y %H:%M'
        )
        
        # Visualization offsets
        self.offset_baseline = 0
        self.offset_delayed_flight = 0
        self.offset_marker_minutes = 4
        self.offset_id_number = -0.05
        
        # Calculate time bounds
        self.earliest_datetime = min(
            min(parse_time_with_day_offset(flight_info['DepTime'], self.start_datetime) 
                for flight_info in self.flights_dict.values()),
            self.start_datetime
        )
        self.latest_datetime = max(
            max(parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime) 
                for flight_info in self.flights_dict.values()),
            self.end_datetime
        )
        
        # Get penalty flags
        self.penalty_flags = self.steps[0].get('penalty_flags', {}) if self.steps else {}
    
    def get_aircraft_indices(self, swapped_flights=None, rotations_dict=None):
        """Get aircraft indices for plotting."""
        if swapped_flights is None:
            swapped_flights = []
        if rotations_dict is None:
            rotations_dict = self.rotations_dict
            
        updated_rotations_dict = rotations_dict.copy()
        for swap in swapped_flights:
            flight_id, new_aircraft_id = swap
            if flight_id in updated_rotations_dict:
                updated_rotations_dict[flight_id]['Aircraft'] = new_aircraft_id

        def extract_sort_key(aircraft_id):
            letters = ''.join(re.findall(r'[A-Za-z]+', aircraft_id))
            numbers = tuple(int(num) for num in re.findall(r'\d+', aircraft_id))
            return (letters, ) + numbers

        all_aircraft_ids = sorted(list(set([rotation_info['Aircraft'] 
                                           for rotation_info in updated_rotations_dict.values()]).union(set(self.aircraft_dict.keys()))))
        aircraft_indices = {aircraft_id: index + 1 for index, aircraft_id in enumerate(all_aircraft_ids)}
        
        return all_aircraft_ids, aircraft_indices, updated_rotations_dict
    
    def plot_state(self, ax, state_data, title_suffix=""):
        """Plot state using StatePlotter-style visualization."""
        # Extract state information
        swapped_flights = state_data["swapped_flights"]
        environment_delayed_flights = state_data["environment_delayed_flights"]
        cancelled_flights = state_data["cancelled_flights"]
        current_datetime = state_data["current_datetime"]
        
        flights_dict = state_data.get("flights_dict", self.flights_dict)
        rotations_dict = state_data.get("rotations_dict", self.rotations_dict)
        
        all_aircraft_ids, aircraft_indices, updated_rotations_dict = self.get_aircraft_indices(swapped_flights, rotations_dict)
        
        # Set up date formatter
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        labels = {
            'Scheduled Flight': False,
            'Swapped Flight': False,
            'Environment Delayed Flight': False,
            'Cancelled Flight': False,
            'Aircraft Unavailable': False,
            'Aircraft Unavailable (Confirmed)': False,
            'Uncertain Breakdown': False,
            'Zero Probability': False,
            'Zero Probability (Resolved: Did Not Occur)': False,
            'Current Action Flight': False,
            'Last Action Flight': False
        }

        earliest_time = self.earliest_datetime
        latest_time = self.latest_datetime

        # Plot flights
        for rotation_id, rotation_info in updated_rotations_dict.items():
            flight_id = rotation_id
            aircraft_id = rotation_info['Aircraft']
            
            if flight_id in flights_dict:
                flight_info = flights_dict[flight_id]
                dep_datetime_str = flight_info['DepTime']
                arr_datetime_str = flight_info['ArrTime']
                
                dep_datetime = parse_time_with_day_offset(dep_datetime_str, self.start_datetime)
                arr_datetime = parse_time_with_day_offset(arr_datetime_str, dep_datetime)
                
                # Handle midnight crossings
                if '+1' in dep_datetime_str and '+1' in arr_datetime_str:
                    dep_datetime = self.start_datetime + timedelta(days=1)
                    dep_datetime = dep_datetime.replace(hour=int(dep_datetime_str.split(':')[0]), 
                                                     minute=int(dep_datetime_str.split(':')[1].split('+')[0]))
                    arr_datetime = dep_datetime + timedelta(days=0)
                    arr_datetime = arr_datetime.replace(hour=int(arr_datetime_str.split(':')[0]), 
                                                     minute=int(arr_datetime_str.split(':')[1].split('+')[0]))
                elif '+1' in dep_datetime_str:
                    dep_datetime = self.start_datetime + timedelta(days=1)
                    dep_datetime = dep_datetime.replace(hour=int(dep_datetime_str.split(':')[0]), 
                                                     minute=int(dep_datetime_str.split(':')[1].split('+')[0]))
                    arr_datetime = dep_datetime
                    arr_datetime = arr_datetime.replace(hour=int(arr_datetime_str.split(':')[0]), 
                                                     minute=int(arr_datetime_str.split(':')[1]))
                elif '+1' in arr_datetime_str:
                    dep_datetime = dep_datetime.replace(hour=int(dep_datetime_str.split(':')[0]), 
                                                     minute=int(dep_datetime_str.split(':')[1]))
                    arr_datetime = dep_datetime + timedelta(days=1)
                    arr_datetime = arr_datetime.replace(hour=int(arr_datetime_str.split(':')[0]), 
                                                     minute=int(arr_datetime_str.split(':')[1].split('+')[0]))
                
                earliest_time = min(earliest_time, dep_datetime)
                latest_time = max(latest_time, arr_datetime)
                
                swapped = any(flight_id == swap[0] for swap in swapped_flights)
                delayed = flight_id in environment_delayed_flights
                cancelled = flight_id in cancelled_flights
                
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
                        ha='center', va='bottom', fontsize=9, color='black')

        # Plot disruptions
        self.plot_disruptions(ax, aircraft_indices, labels, state_data)
        
        # Add recovery period markers
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
        ax.set_yticks(range(1, len(all_aircraft_ids) + 1))
        ax.set_yticklabels(ytick_labels)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Aircraft')
        
        if "Step" in title_suffix:
            step_part = title_suffix.split(" - Step ")[1] if " - Step " in title_suffix else ""
            step_title = f"Step {step_part}" if step_part else title_suffix
            ax.set_title(step_title, fontsize=11, fontweight='bold')
        else:
            ax.set_title(f'{title_suffix}', fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=8)
    
    def plot_disruptions(self, ax, aircraft_indices, labels, state_data=None):
        """Plot aircraft disruptions."""
        def get_height_in_data_units(ax, pixels):
            y0_display = ax.transData.transform((0, 0))[1]
            y1_display = ax.transData.transform((0, 1))[1]
            pixels_per_data_unit = abs(y1_display - y0_display)
            data_units_per_pixel = 1 / pixels_per_data_unit
            return data_units_per_pixel * pixels

        rect_height = get_height_in_data_units(ax, 150)
        
        unavailabilities_probabilities = None
        if state_data and "unavailabilities_probabilities" in state_data:
            unavailabilities_probabilities = state_data["unavailabilities_probabilities"]
        
        if self.alt_aircraft_dict:
            for aircraft_id, unavailability_info in self.alt_aircraft_dict.items():
                if not isinstance(unavailability_info, list):
                    unavailability_info = [unavailability_info]
                
                for unavail in unavailability_info:
                    start_date = unavail['StartDate']
                    start_time = unavail['StartTime']
                    end_date = unavail['EndDate']
                    end_time = unavail['EndTime']
                    
                    if unavailabilities_probabilities and aircraft_id in unavailabilities_probabilities:
                        prob_info = unavailabilities_probabilities[aircraft_id]
                        probability = prob_info.get('probability', 1.0)
                        if probability is None:
                            probability = 0.0
                    else:
                        probability = unavail.get('Probability', 1.0)
                    
                    unavail_start = datetime.strptime(f"{start_date} {start_time}", '%d/%m/%y %H:%M')
                    unavail_end = datetime.strptime(f"{end_date} {end_time}", '%d/%m/%y %H:%M')
                    y_offset = aircraft_indices[aircraft_id]

                    if np.isnan(probability):
                        probability = 0.0
                    
                    if probability == 0.0:
                        rect_color = 'lightgrey'
                        plot_label = 'Zero Probability (Resolved: Did Not Occur)'
                    elif probability < 1.0:
                        rect_color = 'orange'
                        plot_label = 'Uncertain Breakdown'
                    else:
                        rect_color = 'red'
                        plot_label = 'Aircraft Unavailable (Confirmed)'
                    
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

                    initial_prob = unavail.get('Probability', 1.0)
                    if np.isnan(initial_prob):
                        initial_prob = 0.0
                    
                    if probability < 1.0 or (unavailabilities_probabilities and abs(probability - initial_prob) > 0.01):
                        x_position = unavail_start + (unavail_end - unavail_start) / 2
                        y_position = y_offset - rect_height / 2 - get_height_in_data_units(ax, 10) + 0.2
                        prob_text = f"{probability:.2f}"
                        if unavailabilities_probabilities and abs(probability - initial_prob) > 0.01:
                            prob_text = f"{initial_prob:.2f}→{probability:.2f}"
                        ax.text(x_position, y_position + 0.1, prob_text, ha='center', va='top', fontsize=8, fontweight='bold')
    
    def get_action_description(self, step_info):
        """Get human-readable action description."""
        action_reason = step_info.get("action_reason", "unknown")
        
        if "flight_action" in step_info and "aircraft_action" in step_info:
            flight_action = step_info["flight_action"]
            aircraft_action = step_info["aircraft_action"]
        else:
            action_index = step_info["action"]
            flight_action, aircraft_action = self.map_index_to_action(action_index)
        
        if flight_action == 0 and aircraft_action == 0:
            action_desc = "No Action (0, 0)"
        elif flight_action == 0:
            action_desc = f"Cancel on AC{aircraft_action} (0, {aircraft_action})"
        elif aircraft_action == 0:
            action_desc = f"Cancel Flight {flight_action} ({flight_action}, 0)"
        else:
            action_desc = f"Flight {flight_action} -> AC{aircraft_action}"
        
        return action_desc
    
    def map_index_to_action(self, index):
        """Map action index to (flight_action, aircraft_action)."""
        num_aircraft = len(self.aircraft_dict)
        flight_action = index // (num_aircraft + 1)
        aircraft_action = index % (num_aircraft + 1)
        return flight_action, aircraft_action
    
    def print_step_metrics(self):
        """Print detailed metrics for each step."""
        print("\n" + "=" * 100)
        print(f"DETAILED STEP ANALYSIS")
        print(f"Episode: {self.episode_number} | Scenario: {self.scenario_folder}")
        print("=" * 100)
        
        total_reward = 0
        
        # Penalty labels without emojis
        penalty_labels = {
            "delay_penalty_total": "Delay penalty",
            "cancel_penalty": "Manual cancellation penalty",
            "inaction_penalty": "Inaction penalty",
            "automatic_cancellation_penalty": "Automatic cancellation penalty",
            "proactive_penalty": "Proactive penalty (last-minute)",
            "time_penalty": "Time penalty",
            "final_conflict_resolution_reward": "Final conflict resolution reward (LEGACY)",
            "unresolved_conflict_penalty": "Episode-End Unresolved Conflict Penalty",
            "probability_resolution_bonus": "Probability resolution bonus (Reward #8)",
            "low_confidence_action_penalty": "Low-confidence action penalty (Penalty #9)",
            "non_action_penalty": "Ineffective action penalty (Penalty #10)",
            "action_taking_bonus": "Action-taking bonus (LEGACY)",
            # Old format keys
            "delay": "Delay penalty",
            "cancellation": "Manual cancellation penalty",
            "inaction": "Inaction penalty",
            "automatic_cancellation": "Automatic cancellation penalty",
            "proactive": "Proactive penalty",
            "time": "Time penalty"
        }
        
        for i, step_info in enumerate(self.steps):
            step_num = i + 1
            reward = step_info.get('reward', 0)
            epsilon = step_info.get('epsilon', 0)
            action_desc = self.get_action_description(step_info)
            
            total_reward += reward
            
            print(f"\n{'─' * 100}")
            print(f"Step {step_num}: {action_desc}")
            print(f"{'─' * 100}")
            print(f"  Epsilon: {epsilon:.4f}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Total Reward So Far: {total_reward:.4f}")
            
            # Get delay information (show even if not penalized)
            delay_minutes_info = step_info.get('delay_penalty_minutes', None)
            environment_delayed = step_info.get('environment_delayed_flights', {})
            total_delay_minutes = 0
            
            if delay_minutes_info is not None:
                total_delay_minutes = delay_minutes_info
            elif environment_delayed:
                # Sum up delay minutes from delayed flights
                for flight_id, delay_info in environment_delayed.items():
                    if isinstance(delay_info, (int, float)):
                        total_delay_minutes += delay_info if delay_info < 1000 else delay_info * 60
                    elif isinstance(delay_info, dict):
                        if 'delay_minutes' in delay_info:
                            total_delay_minutes += delay_info['delay_minutes']
                        elif 'delay_hours' in delay_info:
                            total_delay_minutes += delay_info['delay_hours'] * 60
            
            # Always show delay information if there are delays
            if total_delay_minutes > 0:
                delay_hours = total_delay_minutes / 60
                num_delayed_flights = len(environment_delayed) if environment_delayed else 0
                print(f"  Delays: {total_delay_minutes:.1f} min ({delay_hours:.2f} hours) across {num_delayed_flights} flight(s)")
            
            # Print penalties breakdown
            penalties = step_info.get('penalties', {})
            if penalties:
                # Deduplicate penalties (prefer new format keys over old format)
                penalty_key_mapping = {
                    "delay": "delay_penalty_total",
                    "cancellation": "cancel_penalty",
                    "inaction": "inaction_penalty",
                    "automatic_cancellation": "automatic_cancellation_penalty",
                    "proactive": "proactive_penalty",
                    "time": "time_penalty"
                }
                
                processed_penalties = {}
                for penalty_name, penalty_value in penalties.items():
                    # Normalize to new format
                    normalized_name = penalty_key_mapping.get(penalty_name, penalty_name)
                    # Only keep if not already present, or if this is the new format key
                    if normalized_name not in processed_penalties:
                        processed_penalties[normalized_name] = penalty_value
                    elif penalty_name == normalized_name:  # Prefer new format
                        processed_penalties[normalized_name] = penalty_value
                
                print(f"  Penalties/Rewards:")
                for penalty_name, penalty_value in processed_penalties.items():
                    if abs(penalty_value) > 1e-6:
                        # Get readable label
                        display_name = penalty_labels.get(penalty_name, penalty_name)
                        
                        # Determine if this is a reward (positive) or penalty (negative)
                        # Rewards: probability_resolution_bonus, final_conflict_resolution_reward, action_taking_bonus
                        is_reward = penalty_name in ["probability_resolution_bonus", "final_conflict_resolution_reward", "action_taking_bonus"]
                        
                        # For probability resolution bonus, add probability information
                        if penalty_name == "probability_resolution_bonus":
                            # Try to get probability information from step_info
                            unavailabilities_probs = step_info.get('unavailabilities_probabilities', {})
                            prob_info = ""
                            
                            # Get the aircraft involved in the action
                            aircraft_action = step_info.get('aircraft_action', None)
                            
                            if unavailabilities_probs and aircraft_action is not None and aircraft_action != 0:
                                # Find the aircraft that was involved in the action
                                # aircraft_action is 1-based index, need to match with aircraft_id
                                for aircraft_id, prob_data in unavailabilities_probs.items():
                                    prob = prob_data.get('probability', None)
                                    
                                    # Check if this aircraft matches the action
                                    # Extract aircraft number from aircraft_id (e.g., "AC3" -> 3, "B737#3" -> 3)
                                    ac_num = None
                                    if aircraft_id.startswith('AC'):
                                        try:
                                            ac_num = int(aircraft_id[2:])
                                        except ValueError:
                                            pass
                                    elif '#' in aircraft_id:
                                        try:
                                            ac_num = int(aircraft_id.split('#')[-1])
                                        except ValueError:
                                            pass
                                    
                                    # If this aircraft matches the action and has a probability
                                    if ac_num == aircraft_action and prob is not None:
                                        # Calculate the probability that was resolved
                                        # The bonus is calculated based on probability reduction
                                        # If prob is now lower, that's what was resolved
                                        # Bonus = probability_reduced * PROBABILITY_RESOLUTION_BONUS_SCALE
                                        # So: probability_reduced = abs(penalty_value) / 50
                                        try:
                                            from src.config_rf import PROBABILITY_RESOLUTION_BONUS_SCALE
                                            prob_resolved = abs(penalty_value) / PROBABILITY_RESOLUTION_BONUS_SCALE
                                        except ImportError:
                                            prob_resolved = abs(penalty_value) / 50  # Fallback
                                        
                                        prob_info = f" [AC{aircraft_action} resolved prob={prob_resolved:.2f}]"
                                        break
                            
                            display_name = display_name + prob_info
                        
                        # Ensure correct sign: penalties should be negative, rewards positive
                        # If the value is stored with wrong sign, correct it
                        if is_reward:
                            display_value = abs(penalty_value)  # Rewards are positive
                        else:
                            display_value = -abs(penalty_value)  # Penalties are negative
                        
                        # Show value with proper sign
                        if abs(display_value) < 0.01:
                            print(f"    {display_name}: {display_value:.6f}")
                        else:
                            print(f"    {display_name}: {display_value:.4f}")
            
            if not penalties:
                print(f"  (No penalties/rewards)")
        
        print(f"\n{'═' * 100}")
        print(f"FINAL TOTAL REWARD: {total_reward:.4f}")
        print(f"{'═' * 100}")
    
    def print_final_statistics(self):
        """Print final statistics summary."""
        print("\n" + "=" * 100)
        print("FINAL STATISTICS SUMMARY")
        print("=" * 100)
        
        if not self.final_metrics:
            print("WARNING: No final metrics available (scenario may not have completed)")
            return
        
        # Extract metrics
        delay_count = self.final_metrics.get('delay_count', 0)
        delay_minutes = self.final_metrics.get('delay_minutes', 0)
        tail_swaps_total = self.final_metrics.get('tail_swaps_total', 0)
        tail_swaps_resolving = self.final_metrics.get('tail_swaps_resolving', 0)
        tail_swaps_inefficient = self.final_metrics.get('tail_swaps_inefficient', 0)
        manual_cancellations = self.final_metrics.get('cancelled_flights', 0)
        auto_cancellations = self.final_metrics.get('automatically_cancelled_count', 0)
        resolved_conflicts = self.final_metrics.get('resolved_initial_conflicts', 0)
        inaction_count = self.final_metrics.get('inaction_count', 0)
        steps = self.final_metrics.get('steps', len(self.steps))
        
        # Print summary
        print(f"\nSCENARIO METRICS:")
        print(f"  Total Steps: {steps}")
        print(f"  Active Actions: {steps - inaction_count}")
        print(f"  Inaction (0,0): {inaction_count}")
        
        print(f"\nFLIGHT OPERATIONS:")
        print(f"  Delays: {delay_count} flights, {delay_minutes:.0f} minutes ({delay_minutes/60:.2f} hours)")
        if delay_count > 0:
            print(f"    Avg delay per flight: {delay_minutes/delay_count:.1f} minutes")
        
        print(f"\nTAIL SWAPS:")
        print(f"  Total: {tail_swaps_total}")
        print(f"  Resolving Conflicts: {tail_swaps_resolving}")
        print(f"  Inefficient: {tail_swaps_inefficient}")
        if tail_swaps_total > 0:
            efficiency = (tail_swaps_resolving / tail_swaps_total) * 100
            print(f"  Efficiency: {efficiency:.1f}%")
        
        print(f"\nCANCELLATIONS:")
        print(f"  Manual (by agent): {manual_cancellations}")
        print(f"  Automatic (system): {auto_cancellations}")
        print(f"  Total: {manual_cancellations + auto_cancellations}")
        
        print(f"\nCONFLICT RESOLUTION:")
        print(f"  Initial Conflicts Resolved: {resolved_conflicts}")
        
        print("=" * 100)
    
    def create_visualization(self, save_path=None, max_steps_per_plot=3):
        """Create comprehensive visualization with schedule plots and metrics."""
        num_steps = len(self.steps)
        
        if num_steps == 0:
            print("No steps to visualize")
            return None
        
        # Print metrics first
        self.print_step_metrics()
        self.print_final_statistics()
        
        # Create schedule visualizations
        num_plots = (num_steps + max_steps_per_plot - 1) // max_steps_per_plot
        figures = []
        
        print(f"\n" + "=" * 100)
        print(f"Creating {num_plots} visualization plot(s)...")
        print("=" * 100)
        
        for plot_idx in range(num_plots):
            start_step = plot_idx * max_steps_per_plot
            end_step = min(start_step + max_steps_per_plot, num_steps)
            steps_in_plot = end_step - start_step
            
            # Create figure
            fig, axes = plt.subplots(steps_in_plot + 1, 1, figsize=(16, 4.5 * (steps_in_plot + 1)), dpi=100)
            
            if steps_in_plot == 0:
                axes = [axes]
            
            # Plot initial state
            self.plot_state(axes[0], self.initial_state, "Initial State")
            
            # Plot each step
            for i in range(start_step, end_step):
                step_info = self.steps[i]
                step_num = i + 1
                action_desc = self.get_action_description(step_info)
                epsilon = step_info.get('epsilon', 0)
                reward = step_info.get('reward', 0)
                
                step_title = f" - Step {step_num}: {action_desc} | ε={epsilon:.3f} | R={reward:.3f}"
                self.plot_state(axes[i - start_step + 1], step_info, step_title)
            
            plt.tight_layout()
            
            # Save plot
            if save_path:
                base_path = save_path.replace('.png', '')
                plot_save_path = f"{base_path}_plot_{plot_idx + 1}_of_{num_plots}.png"
                plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
                print(f"Plot {plot_idx + 1}/{num_plots} saved to: {plot_save_path}")
            
            figures.append(fig)
        
        return figures


def load_and_visualize(save_folder, env_type, seed, episode_number, scenario_folder):
    """Load data and create detailed visualization."""
    
    # If save_folder is relative, make it relative to PROJECT_ROOT
    if not os.path.isabs(save_folder):
        save_folder = os.path.join(PROJECT_ROOT, save_folder)
    
    # Load detailed episode data
    detailed_episodes_path = f"{save_folder}/detailed_episodes/{env_type}_detailed_episodes_seed_{seed}.pkl"
    
    if not os.path.exists(detailed_episodes_path):
        print(f"ERROR: Detailed episodes file not found at {detailed_episodes_path}")
        return None
    
    try:
        with open(detailed_episodes_path, 'rb') as f:
            detailed_episode_data = pickle.load(f)
        
        print(f"Loaded detailed episode data with {len(detailed_episode_data)} episodes")
        
        # Check if episode exists
        if episode_number not in detailed_episode_data:
            print(f"ERROR: Episode {episode_number} not found")
            print(f"Available episodes: {list(detailed_episode_data.keys())[:10]}...")
            return None
        
        # Get scenario
        episode_data = detailed_episode_data[episode_number]
        available_scenarios = list(episode_data["scenarios"].keys())
        
        if scenario_folder is None:
            if not available_scenarios:
                print(f"ERROR: No scenarios found for episode {episode_number}")
                return None
            scenario_folder = available_scenarios[0]
            print(f"Using first available scenario: {scenario_folder}")
        elif scenario_folder not in available_scenarios:
            # Try partial match
            matching = [s for s in available_scenarios if scenario_folder in s]
            if len(matching) == 1:
                scenario_folder = matching[0]
                print(f"Matched scenario: {scenario_folder}")
            else:
                print(f"ERROR: Scenario '{scenario_folder}' not found")
                print(f"Available scenarios: {available_scenarios[:5]}...")
                return None
        
        # Create visualizer
        print(f"\nAnalyzing Episode {episode_number}, Scenario: {scenario_folder}")
        visualizer = DetailedEpisodeVisualizer(detailed_episode_data, env_type, seed, 
                                              episode_number, scenario_folder)
        
        # Create output directory
        output_dir = f"{save_folder}/visualizations/episode_{episode_number}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization
        scenario_name = scenario_folder.split('/')[-1] if '/' in scenario_folder else scenario_folder
        save_path = f"{output_dir}/detailed_{env_type}_{scenario_name}.png"
        figures = visualizer.create_visualization(save_path)
        
        print(f"\nVisualization completed successfully!")
        print(f"Output saved to: {output_dir}")
        
        return figures
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Configuration
    if len(sys.argv) >= 5:
        save_folder = sys.argv[1]
        env_type = sys.argv[2]
        seed = int(sys.argv[3])
        episode_number = int(sys.argv[4])
        scenario_folder = sys.argv[5] if len(sys.argv) > 5 else None
    else:
        # Default values - UPDATE THESE FOR YOUR ANALYSIS
        save_folder = "Final_Model_20_wPenalty10/3ac-130-green"  # Relative to project root
        env_type = "proactive"
        seed = 232323
        episode_number = 0
        scenario_folder = None  # None = use first available scenario
    
    print("=" * 100)
    print("DETAILED EPISODE VISUALIZATION")
    print("=" * 100)
    print(f"Configuration:")
    print(f"  Save Folder: {save_folder}")
    print(f"  Environment Type: {env_type}")
    print(f"  Seed: {seed}")
    print(f"  Episode Number: {episode_number}")
    if scenario_folder:
        print(f"  Scenario: {scenario_folder}")
    else:
        print(f"  Scenario: (will use first available)")
    print("=" * 100)
    
    # Load and visualize
    load_and_visualize(save_folder, env_type, seed, episode_number, scenario_folder)
    
    plt.show()

