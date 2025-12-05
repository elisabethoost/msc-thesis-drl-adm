"""
Quick script to analyze a random or specific episode and scenario from training results.

Usage:
    # Analyze a random episode/scenario (uses latest pickle file)
    python analyze_random_episode.py
    
    # Analyze a random episode/scenario from a specific file
    python analyze_random_episode.py Final_Model_17_wPenalty10/3ac-130-green/detailed_episodes/proactive_detailed_episodes_seed_232323.pkl
    
    # Analyze a specific episode (random scenario)
    python analyze_random_episode.py Save_Trained_Models38/3ac-130-green/detailed_episodes/proactive_detailed_episodes_seed_232323.pkl 46
    
    # Analyze a specific episode and scenario
    python analyze_random_episode.py Save_Trained_Models50/3ac-130-green/detailed_episodes/proactive_detailed_episodes_seed_232323.pkl 11 "Data/TRAINING/3ac-130-green/mixed_Scenario_00115"

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
        "penalty_7_auto_cancellation_enabled": "Penalty #7: Auto Cancellation",
        "penalty_8_probability_resolution_bonus_enabled": "Reward #8: Probability Resolution Bonus",
        "penalty_9_low_confidence_action_enabled": "Penalty #9: Low-Confidence Action"
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
    # Map penalty dict keys to display names (handle both old and new formats)
    penalty_totals = {
        "delay": 0,  # Old format
        "delay_penalty_total": 0,  # New format
        "cancellation": 0,  # Old format
        "cancel_penalty": 0,  # New format
        "inaction": 0,  # Old format
        "inaction_penalty": 0,  # New format
        "automatic_cancellation": 0,  # Old format
        "automatic_cancellation_penalty": 0,  # New format
        "proactive": 0,  # Old format
        "proactive_penalty": 0,  # New format
        "time": 0,  # Old format
        "time_penalty": 0,  # New format
        "final_conflict_resolution_reward": 0,
        "probability_resolution_bonus": 0,
        "low_confidence_action_penalty": 0
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
        
        # Check swapped_flights to verify the actual action taken (more reliable than decoded action)
        # swapped_flights stores (flight_id, aircraft_id) tuples and accumulates across steps
        swapped_flights = step_info.get('swapped_flights', [])
        action_discrepancy = False
        corrected_note = ""
        
        # Get new swaps for this step by comparing with previous step
        prev_swapped = steps[i-1].get('swapped_flights', []) if i > 0 else []
        new_swaps = [swap for swap in swapped_flights if swap not in prev_swapped]
        
        # Only check swapped_flights for reassignment actions (not cancellations, which have aircraft_action == 0)
        # If there's a new swap and it doesn't match the logged action, use swapped_flights as source of truth
        if (new_swaps and flight_action is not None and aircraft_action is not None 
            and aircraft_action != 0):  # Only for reassignments, not cancellations
            actual_flight, actual_aircraft_id = new_swaps[0]
            # swapped_flights stores aircraft_id (string like "AC1"), but action uses aircraft index (1-based int)
            # If the logged flight doesn't match the swapped flight, there's a discrepancy
            if flight_action != actual_flight:
                action_discrepancy = True
                logged_flight = step_info.get('flight_action', '?')
                logged_aircraft = step_info.get('aircraft_action', '?')
                # Use the actual flight from swapped_flights
                flight_action = actual_flight
                # Try to convert aircraft_id to aircraft index if possible
                # For now, we'll try to extract the number from aircraft_id (e.g., "AC1" -> 1)
                if isinstance(actual_aircraft_id, str) and actual_aircraft_id.startswith("AC"):
                    try:
                        aircraft_action = int(actual_aircraft_id[2:])
                    except ValueError:
                        pass  # Keep original aircraft_action if conversion fails
                elif isinstance(actual_aircraft_id, (int, float)):
                    aircraft_action = int(actual_aircraft_id)
                corrected_note = f" [CORRECTED: logged was ({logged_flight}, {logged_aircraft}), actual from swapped_flights: ({actual_flight}, {actual_aircraft_id})]"
        
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
            
            # Add correction note if there was a discrepancy
            if action_discrepancy:
                action_display += corrected_note
        
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
        
        # Get time information
        time_minutes = step_info.get('current_time_minutes', None)
        time_minutes_from_start = step_info.get('current_time_minutes_from_start', None)
        time_display = str(current_datetime)
        if time_minutes_from_start is not None:
            time_display += f" ({time_minutes_from_start:.1f} min from start)"
        elif time_minutes is not None:
            time_display += f" ({time_minutes:.1f} min)"
        
        print(f"\nStep {step_num}:")
        print(f"  Epsilon: {epsilon:.4f} | Action: {action_display} ({action_reason}) | {status_str}")
        print(f"  Time: {time_display} | Step reward: {reward:.2f} | Total: {total_reward_so_far:.2f}")

        # Explicitly report invalid action penalties
        if invalid_action:
            reason_str = f" Reason: {invalid_action_reason}" if invalid_action_reason else ""
            print(f"  INVALID ACTION detected -> Fixed penalty applied: -1000.{reason_str}")
        
        # Get delay information (even if penalty is 0)
        delay_minutes_info = step_info.get('delay_penalty_minutes', None)
        environment_delayed = step_info.get('environment_delayed_flights', {})
        total_delay_minutes = 0
        if delay_minutes_info is not None:
            total_delay_minutes = delay_minutes_info
        elif environment_delayed:
            total_delay_minutes = sum(environment_delayed.values())
        
        # Print penalties
        has_penalties = False
        
        # Map old format to new format for consistent handling
        penalty_key_mapping = {
            "delay": "delay_penalty_total",
            "cancellation": "cancel_penalty",
            "inaction": "inaction_penalty",
            "automatic_cancellation": "automatic_cancellation_penalty",
            "proactive": "proactive_penalty",
            "time": "time_penalty"
        }
        
        # Map penalty names to readable labels
        penalty_labels = {
            "delay_penalty_total": "Delay penalty",
            "cancel_penalty": "Cancellation penalty",
            "inaction_penalty": "Inaction penalty",
            "automatic_cancellation_penalty": "Automatic cancellation penalty",
            "proactive_penalty": "Proactive penalty",
            "time_penalty": "Time penalty",
            "final_conflict_resolution_reward": "Final conflict resolution reward",
            "probability_resolution_bonus": "Probability resolution bonus (Reward #8)",
            "low_confidence_action_penalty": "Low-confidence action penalty (Penalty #9)"
        }
        
        # Deduplicate penalties: prefer new format keys, fall back to old format
        processed_penalties = {}
        for penalty_name, penalty_value in penalties.items():
            # Normalize to new format
            normalized_name = penalty_key_mapping.get(penalty_name, penalty_name)
            # Only keep the new format key (or unique keys like rewards)
            if normalized_name not in processed_penalties:
                processed_penalties[normalized_name] = penalty_value
            elif penalty_name == normalized_name:  # If it's already the new format, prefer it
                processed_penalties[normalized_name] = penalty_value
        
        for penalty_name, penalty_value in processed_penalties.items():
            # Show penalty even if very small (to see actual values)
            if abs(penalty_value) > 1e-6:  # Show if greater than 0.000001
                has_penalties = True
                # Track in totals
                if penalty_name not in penalty_totals:
                    penalty_totals[penalty_name] = 0
                penalty_totals[penalty_name] += abs(penalty_value)
                sign = "-" if penalty_value < 0 else "+"
                display_name = penalty_labels.get(penalty_name, penalty_name)
                
                # Special handling for delay penalty to show minutes
                if penalty_name == "delay_penalty_total":
                    try:
                        from src.config_rf import DELAY_MINUTE_PENALTY
                        if total_delay_minutes > 0:
                            delay_hours = total_delay_minutes / 60
                            print(f"    {sign} {display_name}: {abs(penalty_value):.4f} (Delay: {total_delay_minutes:.1f} min = {delay_hours:.2f} hours)")
                        else:
                            print(f"    {sign} {display_name}: {abs(penalty_value):.4f}")
                    except (ImportError, ZeroDivisionError):
                        print(f"    {sign} {display_name}: {abs(penalty_value):.4f}")
                else:
                    # Show more precision for small values
                    if abs(penalty_value) < 0.01:
                        print(f"    {sign} {display_name}: {abs(penalty_value):.6f}")
                    else:
                        print(f"    {sign} {display_name}: {abs(penalty_value):.4f}")
        
        # Always show delay minutes if there are any delays, even if penalty is 0
        if total_delay_minutes > 0 and not has_penalties:
            delay_hours = total_delay_minutes / 60
            print(f"    Delay: {total_delay_minutes:.1f} min = {delay_hours:.2f} hours (no penalty, below threshold)")
        elif total_delay_minutes > 0:
            # Check if delay penalty was shown
            delay_shown = any(p in ["delay", "delay_penalty_total"] for p in penalties.keys() if penalties[p] != 0)
            if not delay_shown:
                delay_hours = total_delay_minutes / 60
                print(f"    Delay: {total_delay_minutes:.1f} min = {delay_hours:.2f} hours (no penalty, below threshold)")
        
        # Check for rewards #8 and #9 even if they're 0 or missing
        prob_bonus = penalties.get("probability_resolution_bonus", 0)
        low_conf_penalty = penalties.get("low_confidence_action_penalty", 0)
        
        if prob_bonus == 0 and "probability_resolution_bonus" not in penalties:
            # Check if it should have been there but wasn't logged
            resolved_conflicts = step_info.get('resolved_conflicts_count', 0)
            if resolved_conflicts > 0:
                print(f"    Note: {resolved_conflicts} conflicts resolved but Reward #8 not in penalties dict")
        
        if not has_penalties and total_delay_minutes == 0:
            print(f"    (No penalties/rewards)")
        
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
    
    print(f"\nPenalty/Reward breakdown:")
    # Map penalty names to readable labels for summary (handle both old and new formats)
    penalty_labels = {
        "delay": "Delay penalty",
        "delay_penalty_total": "Delay penalty",
        "cancellation": "Cancellation penalty",
        "cancel_penalty": "Cancellation penalty",
        "inaction": "Inaction penalty",
        "inaction_penalty": "Inaction penalty",
        "automatic_cancellation": "Automatic cancellation penalty",
        "automatic_cancellation_penalty": "Automatic cancellation penalty",
        "proactive": "Proactive penalty",
        "proactive_penalty": "Proactive penalty",
        "time": "Time penalty",
        "time_penalty": "Time penalty",
        "final_conflict_resolution_reward": "Final conflict resolution reward",
        "probability_resolution_bonus": "Probability resolution bonus (Reward #8)",
        "low_confidence_action_penalty": "Low-confidence action penalty (Penalty #9)"
    }
    
    # Combine old and new format totals
    combined_totals = {}
    penalty_key_mapping = {
        "delay": "delay_penalty_total",
        "cancellation": "cancel_penalty",
        "inaction": "inaction_penalty",
        "automatic_cancellation": "automatic_cancellation_penalty",
        "proactive": "proactive_penalty",
        "time": "time_penalty"
    }
    
    for penalty_name, total in penalty_totals.items():
        if total != 0:
            # Normalize to new format
            normalized = penalty_key_mapping.get(penalty_name, penalty_name)
            if normalized in combined_totals:
                combined_totals[normalized] += total
            else:
                combined_totals[normalized] = total
    
    for penalty_name, total in sorted(combined_totals.items()):
        display_name = penalty_labels.get(penalty_name, penalty_name)
        sign = "+" if penalty_name in ["final_conflict_resolution_reward", "probability_resolution_bonus"] else "-"
        print(f"  {display_name}: {sign}{total:.2f}")
    
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

