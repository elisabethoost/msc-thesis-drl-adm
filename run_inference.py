import os
import json
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from stable_baselines3 import DQN
from src.environment import AircraftDisruptionEnv
from scripts.utils import load_scenario_data

def run_inference_dqn_single(model_path, scenario_folder, env_type, seed):
    """
    Runs inference on a single scenario and returns the metrics.
    """
    start_time = time.time()
    
    # Load scenario data
    data_dict = load_scenario_data(scenario_folder)
    aircraft_dict = data_dict['aircraft']
    flights_dict = data_dict['flights']
    rotations_dict = data_dict['rotations']
    alt_aircraft_dict = data_dict['alt_aircraft']
    config_dict = data_dict['config']

    # Initialize environment
    env = AircraftDisruptionEnv(
        aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict, env_type=env_type
    )

    # Load trained model
    model = DQN.load(model_path)
    model.set_env(env)
    model.policy.set_training_mode(False)
    model.exploration_rate = 0.0  # No exploration during inference

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Run inference
    obs, _ = env.reset()
    done_flag = False
    total_reward = 0
    step_num = 0
    max_steps = 1000

    while not done_flag and step_num < max_steps:
        action_mask = obs['action_mask']
        obs = {key: np.array(value, dtype=np.float32) for key, value in obs.items()}
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        q_values = model.policy.q_net(obs_tensor).detach().cpu().numpy().squeeze()

        masked_q_values = q_values.copy()
        masked_q_values[action_mask == 0] = -np.inf

        # If no valid actions remain, break out
        if np.all(np.isinf(masked_q_values)):
            break

        action = np.argmax(masked_q_values)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done_flag = terminated or truncated
        step_num += 1

    # Collect metrics from environment
    total_delays = env.scenario_wide_delay_minutes
    total_cancelled_flights = env.scenario_wide_cancelled_flights
    end_time = time.time()
    scenario_time = end_time - start_time
    scenario_steps = env.scenario_wide_steps
    scenario_resolved_conflicts = env.scenario_wide_resolved_conflicts
    solution_slack = env.scenario_wide_solution_slack
    scenario_wide_tail_swaps = env.scenario_wide_tail_swaps
    scenario_wide_actual_disrupted_flights = env.scenario_wide_actual_disrupted_flights
    scenario_wide_reward_components = env.scenario_wide_reward_components

    return (total_reward, total_delays, total_cancelled_flights, scenario_time, 
            scenario_steps, scenario_resolved_conflicts, solution_slack, 
            scenario_wide_tail_swaps, scenario_wide_actual_disrupted_flights, 
            scenario_wide_reward_components)



def run_inference_all_models(model_paths, data_folder, seeds, output_file):
    """
    Runs inference on all scenarios found in 'data_folder', for each model in 'model_paths' and each seed in 'seeds'.
    
    Args:
        model_paths (list): List of tuples containing (model_path, env_type).
        data_folder (str): Path to the folder containing scenario subfolders.
        seeds (list): List of seeds for reproducibility.
        output_file (str): Path to save the results CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame containing scenario, model, seed, and metrics.
    """
    
    # Identify all scenario folders within data_folder
    scenario_folders = [
        os.path.join(data_folder, folder)
        for folder in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, folder))
    ]

    total_combinations = len(scenario_folders) * len(model_paths) * len(seeds)
    print(f"Total combinations: {total_combinations}")
    print(f"Estimated time: {total_combinations * 0.1:.1f} minutes")
    print("=" * 50)
    
    current_combination = 0
    start_time = time.time()
    results = []
    
    for scenario_folder in scenario_folders:
        scenario_name = os.path.basename(scenario_folder)
        for model_tuple in model_paths:
            model_path, env_type = model_tuple
            # Extract the actual environment type from the model path
            model_filename = os.path.basename(model_path)
            if model_filename.endswith('.zip'):
                # Extract env_type from filename like "myopic_232323.zip"
                actual_env_type = model_filename.split('_')[0]
            else:
                actual_env_type = env_type
            for seed in seeds:
                current_combination += 1
                elapsed_time = time.time() - start_time
                avg_time_per_run = elapsed_time / current_combination if current_combination > 0 else 0
                estimated_remaining = avg_time_per_run * (total_combinations - current_combination)
                
                print(f"[{current_combination:3d}/{total_combinations:3d}] "
                      f"({current_combination/total_combinations*100:5.1f}%) "
                      f"Elapsed: {elapsed_time/60:.1f}m | "
                      f"ETA: {estimated_remaining/60:.1f}m | "
                      f"{scenario_name} | {actual_env_type} | seed {seed}")

                try:
                    (total_reward, total_delays, total_cancelled_flights, scenario_time, 
                     scenario_steps, scenario_resolved_conflicts, solution_slack, 
                     scenario_wide_tail_swaps, scenario_wide_actual_disrupted_flights, 
                     scenario_wide_reward_components) = run_inference_dqn_single(model_path, scenario_folder, actual_env_type, seed)
                    
                    results.append({
                        "Scenario": scenario_name,
                        "Model": actual_env_type,
                        "Seed": seed,
                        "TotalReward": total_reward,
                        "TotalDelays": total_delays,
                        "TotalCancelledFlights": total_cancelled_flights,
                        "ScenarioTime": scenario_time,
                        "ScenarioSteps": scenario_steps,
                        "ScenarioResolvedConflicts": scenario_resolved_conflicts,
                        "SolutionSlack": solution_slack,
                        "TailSwaps": scenario_wide_tail_swaps,
                        "ActualDisruptedFlights": scenario_wide_actual_disrupted_flights,
                        "Reward_delay_penalty_total": scenario_wide_reward_components.get("delay_penalty_total", 0),
                        "Reward_cancel_penalty": scenario_wide_reward_components.get("cancel_penalty", 0),
                        "Reward_inaction_penalty": scenario_wide_reward_components.get("inaction_penalty", 0),
                        "Reward_proactive_bonus": scenario_wide_reward_components.get("proactive_bonus", 0),
                        "Reward_time_penalty": scenario_wide_reward_components.get("time_penalty", 0),
                        "Reward_final_conflict_resolution_reward": scenario_wide_reward_components.get("final_conflict_resolution_reward", 0)
                    })
                    
                except Exception as e:
                    print(f"ERROR in {scenario_name} | {env_type} | seed {seed}: {str(e)}")
                    continue

    total_time = time.time() - start_time
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"\n" + "=" * 50)
    print(f"INFERENCE COMPLETED!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per run: {total_time/total_combinations:.2f} seconds")
    print(f"Results saved to: {output_file}")
    
    return results_df

if __name__ == "__main__":
    # Configuration
    # For testing: use fewer seeds first
    seeds = list(range(1, 6))  # 5 seeds for testing (change to list(range(1, 101)) for full run)
    
    # Define your scenario folder (testing data)
    data_folder = "Data/TRAINING/6ac-26-lilac"  # Your training scenario folder
    data_folder_name = data_folder.split("/")[-1]
    
    # Define model paths - updated for your configuration (both training seeds)
    model_paths = [
        # Paths based on your main.py configuration - both training seeds
        (os.path.join("Save_Trained_Models", data_folder_name, "myopic_232323.zip"), "myopic_232323"),
        (os.path.join("Save_Trained_Models", data_folder_name, "proactive_232323.zip"), "proactive_232323"),
        (os.path.join("Save_Trained_Models", data_folder_name, "reactive_232323.zip"), "reactive_232323"),
        (os.path.join("Save_Trained_Models", data_folder_name, "myopic_242424.zip"), "myopic_242424"),
        (os.path.join("Save_Trained_Models", data_folder_name, "proactive_242424.zip"), "proactive_242424"),
        (os.path.join("Save_Trained_Models", data_folder_name, "reactive_242424.zip"), "reactive_242424"),
    ]
    
    # Output file - better naming structure
    output_file = f"logs/inference_metrics/{data_folder_name}_{len(seeds)}_seeds.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Run inference
    results_df = run_inference_all_models(model_paths, data_folder, seeds, output_file)
    
    print(f"Inference completed! Results saved to {output_file}")
    print(f"Total results: {len(results_df)}")
    print(f"Scenarios: {results_df['Scenario'].nunique()}")
    print(f"Models: {results_df['Model'].nunique()}")
    print(f"Seeds: {results_df['Seed'].nunique()}")
