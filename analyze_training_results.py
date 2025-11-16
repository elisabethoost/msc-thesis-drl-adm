"""
Script to analyze training results from detailed_episodes pickle files.
This allows you to analyze random episodes and scenarios, see actions taken, and penalties received.

Usage:
    python analyze_training_results.py <path_to_detailed_episodes.pkl> [episode_num] [scenario_folder]
    
    If episode_num and scenario_folder are not provided, a random one will be selected.

Example:
    python analyze_training_results.py Save_Trained_Models30/3ac-130-green/detailed_episodes/proactive_detailed_episodes_seed_232323.pkl
    python analyze_training_results.py Save_Trained_Models30/3ac-130-green/detailed_episodes/proactive_detailed_episodes_seed_232323.pkl 0 deterministic_Scenario_00001
"""

import pickle
import sys
import random
from datetime import datetime

def analyze_episode_scenario(pickle_path, episode_num=None, scenario_folder=None):
    """Analyze a specific episode and scenario from the training results."""
    
    # Load the detailed episodes data
    print(f"Loading data from: {pickle_path}")
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {pickle_path}")
        return
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return
    
    # Get available episodes
    available_episodes = sorted([k for k in data.keys() if isinstance(k, int)])
    
    if not available_episodes:
        print("Error: No episodes found in data.")
        return
    
    print(f"Available episodes: {available_episodes}")
    
    # Select episode
    if episode_num is None:
        episode_num = random.choice(available_episodes)
        print(f"\nRandomly selected episode: {episode_num}")
    else:
        if episode_num not in available_episodes:
            print(f"Error: Episode {episode_num} not found.")
            print(f"Available episodes: {available_episodes}")
            return
        print(f"\nAnalyzing episode: {episode_num}")
    
    episode_data = data[episode_num]
    
    if 'scenarios' not in episode_data:
        print(f"Error: 'scenarios' key not found in episode {episode_num}")
        return
    
    # Get available scenarios
    available_scenarios = list(episode_data['scenarios'].keys())
    
    if not available_scenarios:
        print(f"Error: No scenarios found in episode {episode_num}")
        return
    
    print(f"Available scenarios: {len(available_scenarios)} scenarios")
    
    # Select scenario
    if scenario_folder is None:
        scenario_folder = random.choice(available_scenarios)
        print(f"Randomly selected scenario: {scenario_folder}")
    else:
        if scenario_folder not in available_scenarios:
            print(f"Error: Scenario '{scenario_folder}' not found.")
            print(f"Available scenarios: {available_scenarios[:5]}... (showing first 5)")
            return
        print(f"Analyzing scenario: {scenario_folder}")
    
    scenario_data = episode_data['scenarios'][scenario_folder]
    
    # Get steps
    if 'steps' not in scenario_data or len(scenario_data['steps']) == 0:
        print(f"Error: No steps found for scenario '{scenario_folder}'")
        return
    
    steps = scenario_data['steps']
    
    # Get penalty flags from first step (they should be consistent across steps)
    first_step = steps[0]
    penalty_flags = first_step.get('penalty_flags', {})
    
    # Print header
    print("\n" + "=" * 100)
    print(f"ANALYSIS: Episode {episode_num}, Scenario {scenario_folder}")
    print("=" * 100)
    
    # Print penalty configuration
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
    
    # Print step-by-step analysis
    print("\n" + "=" * 100)
    print("Step-by-Step Analysis:")
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
    
    action_counts = {}
    action_reason_counts = {}
    
    for i, step_info in enumerate(steps):
        step_num = step_info.get('step', i + 1)
        reward = step_info.get('reward', 0)
        penalties = step_info.get('penalties', {})
        action = step_info.get('action', '?')
        
        # Get decoded action (flight, aircraft) if available
        flight_action = step_info.get('flight_action', None)
        aircraft_action = step_info.get('aircraft_action', None)
        if flight_action is None or aircraft_action is None:
            action_display = f"{action}"
            action_key = str(action)
        else:
            # Display as (flight, aircraft)
            if flight_action == 0 and aircraft_action == 0:
                action_display = "(0, 0) [No action]"
                action_key = "(0, 0)"
            elif flight_action == 0:
                action_display = f"(0, {aircraft_action}) [Inaction: Cancel on AC{aircraft_action}]"
                action_key = f"(0, {aircraft_action})"
            elif aircraft_action == 0:
                action_display = f"({flight_action}, 0) [Cancel F{flight_action}]"
                action_key = f"({flight_action}, 0)"
            else:
                action_display = f"({flight_action}, {aircraft_action}) [F{flight_action}->AC{aircraft_action}]"
                action_key = f"({flight_action}, {aircraft_action})"
        
        action_reason = step_info.get('action_reason', '?')
        something_happened = step_info.get('something_happened', False)
        scenario_ended = step_info.get('scenario_ended', False)  # Get scenario_ended flag
        epsilon = step_info.get('epsilon', '?')
        current_datetime = step_info.get('current_datetime', '?')
        total_reward_so_far = step_info.get('total_reward_so_far', 0)
        
        # Track actions
        action_counts[action_key] = action_counts.get(action_key, 0) + 1
        action_reason_counts[action_reason] = action_reason_counts.get(action_reason, 0) + 1
        
        total_reward += reward
        
        # Build status string
        status_parts = []
        if something_happened:
            status_parts.append("Changed state")
        if scenario_ended:
            status_parts.append("Scenario ENDED (final reward calculated)")
        status_str = " | ".join(status_parts) if status_parts else "No state change"
        
        print(f"\n{'-' * 100}")
        print(f"Step {step_num}:")
        print(f"  Epsilon: {epsilon:.4f} | Action: {action_display} ({action_reason}) | {status_str}")
        print(f"  Current datetime: {current_datetime}")
        print(f"  Step reward: {reward:.2f} | Total reward so far: {total_reward_so_far:.2f}")
        
        # Print penalties
        penalties_applied = False
        for penalty_name, penalty_value in penalties.items():
            if penalty_value != 0:
                penalties_applied = True
                penalty_totals[penalty_name] += abs(penalty_value)  # Sum absolute values for totals
                sign = "-" if penalty_value < 0 else "+"
                
                # Special handling for delay penalty to show minutes
                if penalty_name == "delay":
                    # Calculate delay minutes from penalty (penalty = minutes * DELAY_MINUTE_PENALTY)
                    # DELAY_MINUTE_PENALTY = 0.02 from config_rf.py
                    DELAY_MINUTE_PENALTY = 0.02
                    delay_minutes = abs(penalty_value) / DELAY_MINUTE_PENALTY
                    delay_hours = delay_minutes / 60
                    print(f"    {sign} {penalty_name}: {abs(penalty_value):.2f} ({delay_minutes:.1f} min = {delay_hours:.2f} hours)")
                else:
                    print(f"    {sign} {penalty_name}: {abs(penalty_value):.2f}")
        
        if not penalties_applied:
            print(f"    (No penalties applied)")
        
        # Print state information
        cancelled_flights = step_info.get('cancelled_flights', set())
        delayed_flights = step_info.get('environment_delayed_flights', {})
        swapped_flights = step_info.get('swapped_flights', [])
        
        if cancelled_flights:
            print(f"  Cancelled flights: {sorted(cancelled_flights)}")
        if delayed_flights:
            print(f"  Delayed flights: {list(delayed_flights.keys())[:5]}..." if len(delayed_flights) > 5 else f"  Delayed flights: {list(delayed_flights.keys())}")
        if swapped_flights:
            print(f"  Swapped flights: {swapped_flights}")
        
        # Print impact of action
        impact = step_info.get('impact_of_action', {})
        if impact:
            print(f"  Impact of action:")
            for key, value in impact.items():
                if value != 0:
                    print(f"    - {key}: {value}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("Summary:")
    print("=" * 100)
    print(f"Total steps: {len(steps)}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward / len(steps):.2f}")
    
    print(f"\nPenalty totals:")
    for penalty_name, total in penalty_totals.items():
        if total != 0:
            print(f"  {penalty_name}: {total:.2f}")
    
    print(f"\nAction statistics:")
    print(f"  Total actions taken: {len(steps)}")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(steps)) * 100
        print(f"    Action {action}: {count} times ({percentage:.1f}%)")
    
    print(f"\nAction reason statistics:")
    for reason, count in sorted(action_reason_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(steps)) * 100
        print(f"    {reason}: {count} times ({percentage:.1f}%)")
    
    # Verify reward calculation
    calculated_total = (
        - penalty_totals["delay"]
        - penalty_totals["cancellation"]
        - penalty_totals["inaction"]
        - penalty_totals["automatic_cancellation"]
        - penalty_totals["proactive"]
        - penalty_totals["time"]
        + penalty_totals["final_conflict_resolution_reward"]
    )
    
    print(f"\nVerification:")
    print(f"  Calculated total (sum of penalties): {calculated_total:.2f}")
    print(f"  Actual total reward: {total_reward:.2f}")
    if abs(calculated_total - total_reward) < 0.1:
        print("  [OK] Reward calculation is correct!")
    else:
        print(f"  [WARNING] Reward mismatch! Difference: {abs(calculated_total - total_reward):.2f}")
        print("    (Note: This might be due to rounding or other factors)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    pickle_path = sys.argv[1]
    episode_num = int(sys.argv[2]) if len(sys.argv) > 2 else None
    scenario_folder = sys.argv[3] if len(sys.argv) > 3 else None
    
    analyze_episode_scenario(pickle_path, episode_num, scenario_folder)

