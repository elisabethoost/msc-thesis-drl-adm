"""
Quick script to analyze a random or specific episode and scenario from training results.

Usage:
    # Analyze a random episode/scenario (uses latest pickle file)
    python analyze_random_episode.py
    
    # Analyze a random episode/scenario from a specific file
    python analyze_random_episode.py Save_Trained_Models39/3ac-130-green/detailed_episodes/proactive_detailed_episodes_seed_232323.pkl
    
    # Analyze a specific episode (random scenario)
    python analyze_random_episode.py Save_Trained_Models38/3ac-130-green/detailed_episodes/proactive_detailed_episodes_seed_232323.pkl 46
    
    # Analyze a specific episode and scenario
    python analyze_random_episode.py Save_Trained_Models39/3ac-130-green/detailed_episodes/proactive_detailed_episodes_seed_232323.pkl 19 "Data/TRAINING/3ac-130-green/mixed_Scenario_00124"

    If no path is provided, it will look for the most recent detailed_episodes file.
"""

import pickle
import sys
import random
import os
import glob
from pathlib import Path

# Force unbuffered output to prevent display issues
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

def find_latest_pickle_file():
    """Find the most recent detailed_episodes pickle file."""
    # Look in common locations
    search_paths = [
        "Save_Trained_Models*/3ac-130-green/detailed_episodes/*.pkl",
        "**/detailed_episodes/*.pkl",
    ]
    
    all_files = []
    for pattern in search_paths:
        all_files.extend(glob.glob(pattern, recursive=True))
    
    if not all_files:
        return None
    
    # Get the most recent file
    latest_file = max(all_files, key=os.path.getmtime)
    return latest_file

def analyze_random_episode(pickle_path=None, episode_num=None, scenario_folder=None):
    """Analyze a random or specific episode and scenario."""
    
    # Track if episode/scenario were originally specified (for display purposes)
    episode_specified = episode_num is not None
    scenario_specified = scenario_folder is not None
    
    if pickle_path is None:
        pickle_path = find_latest_pickle_file()
        if pickle_path is None:
            print("Error: No pickle file found. Please specify a path.")
            print("Usage: python analyze_random_episode.py <path_to_pickle_file> [episode_num] [scenario_folder]")
            return
        print(f"Using latest file: {pickle_path}")
    
    if not os.path.exists(pickle_path):
        print(f"Error: File not found: {pickle_path}")
        return
    
    # Load data
    print(f"Loading data from: {pickle_path}")
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return
    
    # Get available episodes
    available_episodes = sorted([k for k in data.keys() if isinstance(k, int)])
    if not available_episodes:
        print("Error: No episodes found in data.")
        return
    
    # Select episode (specific or random)
    if episode_num is not None:
        try:
            episode_num = int(episode_num)
            if episode_num not in available_episodes:
                print(f"Error: Episode {episode_num} not found in data.")
                print(f"Available episodes: {available_episodes[:10]}{'...' if len(available_episodes) > 10 else ''}")
                return
        except ValueError:
            print(f"Error: Episode number must be an integer, got: {episode_num}")
            return
    else:
        episode_num = random.choice(available_episodes)
    
    episode_data = data[episode_num]
    
    # Get available scenarios
    available_scenarios = list(episode_data.get('scenarios', {}).keys())
    if not available_scenarios:
        print(f"Error: No scenarios found in episode {episode_num}")
        return
    
    # Select scenario (specific or random)
    if scenario_folder is not None:
        # Try exact match first
        if scenario_folder not in available_scenarios:
            # Try partial match (in case user provided just the scenario name without full path)
            matching_scenarios = [s for s in available_scenarios if scenario_folder in s or s.endswith(scenario_folder)]
            if len(matching_scenarios) == 1:
                scenario_folder = matching_scenarios[0]
                print(f"Matched scenario: {scenario_folder}")
            elif len(matching_scenarios) > 1:
                print(f"Error: Multiple scenarios match '{scenario_folder}':")
                for s in matching_scenarios:
                    print(f"  - {s}")
                return
            else:
                print(f"Error: Scenario '{scenario_folder}' not found in episode {episode_num}.")
                print(f"Available scenarios (first 10): {available_scenarios[:10]}")
                if len(available_scenarios) > 10:
                    print(f"... and {len(available_scenarios) - 10} more")
                return
    else:
        scenario_folder = random.choice(available_scenarios)
    
    scenario_data = episode_data['scenarios'][scenario_folder]
    
    # Get steps
    steps = scenario_data.get('steps', [])
    if not steps:
        print(f"Error: No steps found for scenario '{scenario_folder}'")
        return
    
    # Check for duplicate step numbers (for debugging)
    step_numbers = [s.get('step', i+1) for i, s in enumerate(steps)]
    if len(step_numbers) != len(set(step_numbers)):
        print(f"WARNING: Found duplicate step numbers in data! Step numbers: {step_numbers}")
    
    # Get penalty flags
    first_step = steps[0]
    penalty_flags = first_step.get('penalty_flags', {})
    
    # Print analysis
    print("\n" + "=" * 100)
    analysis_type = "SPECIFIC" if (episode_specified or scenario_specified) else "RANDOM"
    print(f"{analysis_type} ANALYSIS: Episode {episode_num}, Scenario {scenario_folder}")
    print("=" * 100)
    
    # Penalty configuration
    print("\nPenalty Configuration:")
    print("-" * 100)
    penalty_names = {
        "penalty_1_delay_enabled": "Penalty #1: Delay",
        "penalty_2_cancellation_enabled": "Penalty #2: Cancellation",
        "penalty_3_inaction_enabled": "Penalty #3: Inaction",
        "penalty_4_proactive_enabled": "Penalty #4: Proactive",
        "penalty_5_time_enabled": "Penalty #5: Time",
        "penalty_6_final_reward_enabled": "Penalty #6: Final Reward",
        "penalty_7_auto_cancellation_enabled": "Penalty #7: Auto Cancellation"
    }
    
    for penalty_key, penalty_label in penalty_names.items():
        enabled = penalty_flags.get(penalty_key, False)
        status = "[ENABLED]" if enabled else "[DISABLED]"
        print(f"  {penalty_label}: {status}")
    
    # Step-by-step analysis
    print("\n" + "=" * 100)
    print(f"Step-by-Step Analysis ({len(steps)} steps):")
    print("=" * 100)
    
    total_reward = 0
    penalty_totals = {
        "delay": 0,
        "cancellation": 0,
        "inaction": 0,
        "automatic_cancellation": 0,
        "proactive": 0,
        "time": 0,
        "final_conflict_resolution_reward": 0
    }
    
    for i, step_info in enumerate(steps):
        step_num = step_info.get('step', i + 1)
        reward = step_info.get('reward', 0)
        penalties = step_info.get('penalties', {})
        action = step_info.get('action', '?')
        invalid_action = step_info.get('invalid_action', False)
        invalid_action_reason = step_info.get('invalid_action_reason', None)
        
        # Get decoded action (flight, aircraft) if available, otherwise decode from action index
        flight_action = step_info.get('flight_action', None)
        aircraft_action = step_info.get('aircraft_action', None)
        if flight_action is None or aircraft_action is None:
            # Fallback: try to decode from action index (would need env, but for display we can show action)
            action_display = f"{action}"
        else:
            # Display as (flight, aircraft) or [flight, aircraft]
            if flight_action == 0 and aircraft_action == 0:
                action_display = "(0, 0) [No action]"
            elif flight_action == 0:
                action_display = f"(0, {aircraft_action}) [Cancel flight on aircraft {aircraft_action}]"
            elif aircraft_action == 0:
                action_display = f"({flight_action}, 0) [Cancel flight {flight_action}]"
            else:
                action_display = f"({flight_action}, {aircraft_action}) [Flight {flight_action} -> Aircraft {aircraft_action}]"
        
        action_reason = step_info.get('action_reason', '?')
        something_happened = step_info.get('something_happened', False)
        scenario_ended = step_info.get('scenario_ended', False)  # Get scenario_ended flag
        epsilon = step_info.get('epsilon', '?')
        current_datetime = step_info.get('current_datetime', '?')
        total_reward_so_far = step_info.get('total_reward_so_far', 0)
        
        total_reward += reward
        
        # Build status string
        status_parts = []
        if something_happened:
            status_parts.append("Changed state")
        if scenario_ended:
            status_parts.append("Scenario ENDED (final reward calculated)")
        status_str = " | ".join(status_parts) if status_parts else "No state change"
        
        print(f"\nStep {step_num}:")
        print(f"  Epsilon: {epsilon:.4f} | Action: {action_display} ({action_reason}) | {status_str}")
        print(f"  Time: {current_datetime} | Step reward: {reward:.2f} | Total: {total_reward_so_far:.2f}")

        # Explicitly report invalid action penalties
        if invalid_action:
            reason_str = f" Reason: {invalid_action_reason}" if invalid_action_reason else ""
            print(f"  INVALID ACTION detected -> Fixed penalty applied: -1000.{reason_str}")
        
        # Print penalties
        has_penalties = False
        for penalty_name, penalty_value in penalties.items():
            if penalty_value != 0:
                has_penalties = True
                penalty_totals[penalty_name] += abs(penalty_value)
                sign = "-" if penalty_value < 0 else "+"
                
                # Special handling for delay penalty to show minutes
                if penalty_name == "delay":
                    DELAY_MINUTE_PENALTY = 0.02
                    delay_minutes = abs(penalty_value) / DELAY_MINUTE_PENALTY
                    delay_hours = delay_minutes / 60
                    print(f"    {sign} {penalty_name}: {abs(penalty_value):.2f} ({delay_minutes:.1f} min = {delay_hours:.2f} hours)")
                else:
                    print(f"    {sign} {penalty_name}: {abs(penalty_value):.2f}")
        
        if not has_penalties:
            print(f"    (No penalties)")
        
        # Print state info
        cancelled = step_info.get('cancelled_flights', set())
        delayed = step_info.get('environment_delayed_flights', {})
        if cancelled:
            print(f"  Cancelled flights: {sorted(cancelled)}")
        if delayed:
            delayed_list = list(delayed.keys())[:3]
            if len(delayed) > 3:
                delayed_list.append(f"... (+{len(delayed)-3} more)")
            print(f"  Delayed flights: {delayed_list}")
    
    # Summary
    print("\n" + "=" * 100)
    print("Summary:")
    print("=" * 100)
    print(f"Total steps: {len(steps)}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward / len(steps):.2f}")
    
    print(f"\nPenalty breakdown:")
    for penalty_name, total in penalty_totals.items():
        if total != 0:
            print(f"  {penalty_name}: {total:.2f}")
    
    print(f"\nTo analyze a specific episode/scenario, run:")
    print(f"  python analyze_training_results.py {pickle_path} {episode_num} \"{scenario_folder}\"")
    
    # Extract save folder from pickle path for visualization
    if "Save_Trained_Models" in pickle_path:
        save_folder_match = pickle_path.split("Save_Trained_Models")[1]
        if save_folder_match:
            model_num = save_folder_match.split("/")[0] if "/" in save_folder_match else save_folder_match.split("\\")[0]
            scenario_name = pickle_path.split("3ac-130-green")[0] + "3ac-130-green" if "3ac-130-green" in pickle_path else ""
            if scenario_name:
                save_folder = f"Save_Trained_Models{model_num}/3ac-130-green"
                env_type = pickle_path.split("_")[0].split("/")[-1] if "_" in pickle_path.split("/")[-1] else pickle_path.split("_")[0].split("\\")[-1]
                seed = int(pickle_path.split("seed_")[1].split(".")[0]) if "seed_" in pickle_path else 232323
                print(f"\nTo visualize this episode/scenario, run:")
                print(f"  python visualize_episode_stateplotter.py {save_folder} {env_type} {seed} {episode_num} \"{scenario_folder}\"")

if __name__ == "__main__":
    pickle_path = sys.argv[1] if len(sys.argv) > 1 else None
    episode_num = sys.argv[2] if len(sys.argv) > 2 else None
    scenario_folder = sys.argv[3] if len(sys.argv) > 3 else None
    
    analyze_random_episode(pickle_path, episode_num, scenario_folder)

