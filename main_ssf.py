import argparse
import os
import time
import subprocess
import sys
import src.config_ssf as config
import pandas as pd
from train_dqn_modular_ssf import run_train_dqn_both_timesteps
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def get_config_variables(config_module):
    config_vars = {
        key: value for key, value in vars(config_module).items()
        if not key.startswith("__") and not callable(value)
    }
    return config_vars

def run_for_single_folder(training_folder, MAX_TOTAL_TIMESTEPS, single_seed, brute_force_flag, cross_val_flag, early_stopping_flag, CROSS_VAL_INTERVAL, printing_intermediate_results, save_folder, TESTING_FOLDERS_PATH, env_type, save_training_monitor):
    """
    Runs training for a single scenario folder and a single seed.
    Saves the results for each environment type immediately after training.
    """
    stripped_scenario_folder = training_folder.strip("/").split("/")[-1]  # handle trailing slash
    save_results_big_run = f"{save_folder}/{stripped_scenario_folder}"

    # Create the numpy directory if it doesn't exist
    os.makedirs(f"{save_results_big_run}/numpy", exist_ok=True)

    # Run training for the given seed - results will be saved during training for each env_type
    rewards_myopic, rewards_proactive, rewards_reactive, rewards_drl_greedy, \
    test_rewards_myopic, test_rewards_proactive, test_rewards_reactive, test_rewards_drl_greedy = run_train_dqn_both_timesteps(
        MAX_TOTAL_TIMESTEPS=MAX_TOTAL_TIMESTEPS,
        single_seed=single_seed,
        brute_force_flag=brute_force_flag,
        cross_val_flag=cross_val_flag,
        early_stopping_flag=early_stopping_flag,
        CROSS_VAL_INTERVAL=CROSS_VAL_INTERVAL,
        printing_intermediate_results=printing_intermediate_results,
        TRAINING_FOLDERS_PATH=training_folder,
        stripped_scenario_folder=stripped_scenario_folder,
        save_folder=save_folder,
        save_results_big_run=save_results_big_run,
        TESTING_FOLDERS_PATH=TESTING_FOLDERS_PATH,
        env_type=env_type,
        save_training_monitor=save_training_monitor
    )

    # Note: The rewards are already in the correct format (lists) and saved to disk
    # No need to process them further here since they're saved during training
    # Just return them for any subsequent processing if needed
    return

def aggregate_results_and_plot(SEEDS, MAX_TOTAL_TIMESTEPS, brute_force_flag, cross_val_flag, early_stopping_flag, CROSS_VAL_INTERVAL, printing_intermediate_results, save_folder, TESTING_FOLDERS_PATH):
    """
    After all seeds have finished, this function will:
    - Identify all scenario folders
    - Load the per-seed arrays
    - Combine them (mean, std) just like originally
    - Produce a single plot per scenario with mean and std shaded area
    """

    # Add debug logging at the start
    print(f"\nStarting aggregation in folder: {save_folder}")
    print(f"Looking for data from seeds: {SEEDS}")

    # Define environment types and their properties
    env_types = [
        {'name': 'myopic', 'label': 'DQN Proactive-N', 'color': 'blue'},
        {'name': 'proactive', 'label': 'DQN Proactive-U', 'color': 'orange'},
        {'name': 'reactive', 'label': 'DQN Reactive', 'color': 'green'},
        {'name': 'drl-greedy', 'label': 'DQN Greedy-Guided', 'color': 'red'}
    ]

    # Identify scenario folders that contain numpy data
    scenario_folders = []
    for d in os.listdir(save_folder):
        folder_path = os.path.join(save_folder, d)
        numpy_path = os.path.join(folder_path, "numpy")
        
        print(f"\nChecking folder: {folder_path}")
        print(f"Numpy path exists: {os.path.exists(numpy_path)}")
        
        if os.path.isdir(folder_path) and os.path.exists(numpy_path):
            # Check which env types have data
            available_env_types = []
            for env_type in env_types:
                expected_file = os.path.join(numpy_path, f"{env_type['name']}_runs_seed_{SEEDS[0]}.npy")
                print(f"Looking for: {expected_file}")
                if os.path.exists(expected_file):
                    print(f"✓ Found {env_type['name']} data")
                    available_env_types.append(env_type)
                else:
                    print(f"✗ Missing {env_type['name']} data")
            
            # If we have at least one environment type with data, include this folder
            if available_env_types:
                scenario_folders.append({
                    'path': folder_path,
                    'available_env_types': available_env_types
                })
                print(f"✓ Added {folder_path} to scenario folders with {len(available_env_types)} environment types")

    if not scenario_folders:
        print("\nERROR: No scenario folders with any data found. Skipping aggregation.")
        return

    def smooth(data, window=10):
        if window > 1 and len(data) >= window:
            return np.convolve(data, np.ones(window)/window, mode='valid')
        return data

    for scenario in scenario_folders:
        scenario_path = scenario['path']
        available_env_types = scenario['available_env_types']
        stripped_scenario_folder = os.path.basename(scenario_path)
        numpy_path = os.path.join(scenario_path, "numpy")
        print(f"\nProcessing scenario: {stripped_scenario_folder}")

        # Initialize data structures only for available environment types
        all_runs = {env_type['name']: [] for env_type in available_env_types}
        all_steps_runs = {env_type['name']: [] for env_type in available_env_types}
        all_timesteps_per_episode_runs = {env_type['name']: [] for env_type in available_env_types}
        all_test_rewards = {env_type['name']: [] for env_type in available_env_types}

        # Load data for each seed and available environment type
        for seed in SEEDS:
            for env_type in available_env_types:
                try:
                    runs = np.load(os.path.join(numpy_path, f"{env_type['name']}_runs_seed_{seed}.npy"), allow_pickle=True)
                    steps = np.load(os.path.join(numpy_path, f"{env_type['name']}_steps_runs_seed_{seed}.npy"), allow_pickle=True)
                    
                    # Try to load timesteps per episode data (new format)
                    try:
                        timesteps_per_episode = np.load(os.path.join(numpy_path, f"{env_type['name']}_timesteps_per_episode_seed_{seed}.npy"), allow_pickle=True)
                    except FileNotFoundError:
                        # Fallback: calculate from cumulative steps if new format doesn't exist
                        timesteps_per_episode = np.diff(steps, prepend=steps[0])
                    
                    all_runs[env_type['name']].append(runs)
                    all_steps_runs[env_type['name']].append(steps)
                    all_timesteps_per_episode_runs[env_type['name']].append(timesteps_per_episode)

                    if cross_val_flag:
                        test_rewards = np.load(os.path.join(numpy_path, f"test_rewards_{env_type['name']}_seed_{seed}.npy"), allow_pickle=True)
                        all_test_rewards[env_type['name']].append(test_rewards)

                except FileNotFoundError as e:
                    print(f"Warning: Files missing for seed {seed} and env_type {env_type['name']}: {e}")
                    continue

        # Process data only for available environment types
        processed_data = {}
        for env_type in available_env_types:
            name = env_type['name']
            if not all_runs[name]:
                continue

            # Find minimum length and truncate arrays
            min_length = min(len(run) for run in all_runs[name] if len(run) > 0)
            runs_array = np.array([run[:min_length] for run in all_runs[name]])
            steps_array = np.array([steps[:min_length] for steps in all_steps_runs[name]])
            timesteps_per_episode_array = np.array([tpe[:min_length] for tpe in all_timesteps_per_episode_runs[name]])

            # # Calculate statistics
            # mean = runs_array.mean(axis=0)
            # std = runs_array.std(axis=0)
            # Calculate statistics with minimal outlier clipping (reduced from 5-95th to 2-98th percentile)
            # This allows more data points to be visible while still removing extreme outliers
            clipped_runs = np.clip(runs_array, 
                                 np.percentile(runs_array, 2, axis=0), 
                                 np.percentile(runs_array, 98, axis=0))
            mean = clipped_runs.mean(axis=0)
            std = clipped_runs.std(axis=0)
            steps_mean = steps_array.mean(axis=0).astype(int)
            timesteps_per_episode_mean = timesteps_per_episode_array.mean(axis=0)
            timesteps_per_episode_std = timesteps_per_episode_array.std(axis=0)

            # Apply smoothing with reduced window to better see true trends
            smooth_window = 5  # Reduced from 20 to 5 for less smoothing, better trend visibility
            mean_sm = smooth(mean, smooth_window)
            std_sm = smooth(std, smooth_window)
            # Align steps_sm with smoothed data: smoothing with mode='valid' removes the first (window-1) elements
            # So we need to take steps_mean starting from index (window-1) to align properly
            if smooth_window > 1 and len(steps_mean) >= smooth_window:
                steps_sm = steps_mean[smooth_window - 1:smooth_window - 1 + len(mean_sm)]
            else:
                steps_sm = steps_mean[:len(mean_sm)]

            processed_data[name] = {
                'mean_sm': mean_sm,
                'std_sm': std_sm,
                'steps_sm': steps_sm,
                'timesteps_per_episode_mean': timesteps_per_episode_mean,
                'timesteps_per_episode_std': timesteps_per_episode_std
            }

            # Save combined arrays
            np.save(f'{numpy_path}/all_{name}_runs.npy', runs_array)
            np.save(f'{numpy_path}/all_{name}_steps_runs.npy', steps_array)

            if cross_val_flag and all_test_rewards[name]:
                np.save(f'{numpy_path}/all_test_rewards_{name}.npy', np.array(all_test_rewards[name]))

        # Create plots directory
        plots_dir = f"{scenario_path}/plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        # Plot 1: Episode Rewards over Environment Steps
        for env_type in available_env_types:
            name = env_type['name']
            if name in processed_data:
                data = processed_data[name]
                ax1.plot(data['steps_sm'], data['mean_sm'], label=env_type['label'], color=env_type['color'])
                ax1.fill_between(data['steps_sm'],
                               data['mean_sm'] - data['std_sm'],
                               data['mean_sm'] + data['std_sm'],
                               alpha=0.2, color=env_type['color'])

        ax1.set_xlabel("Environment Steps (Frames)")
        ax1.set_ylabel("Episode Reward")
        ax1.set_title(f"Episode Rewards over {len(SEEDS)} Seeds ({stripped_scenario_folder})" if len(SEEDS) > 1 else f"Episode Rewards ({stripped_scenario_folder})")
        ax1.legend(frameon=False)
        ax1.grid(True)
        
        # Plot 2: Timesteps per Episode
        for env_type in available_env_types:
            name = env_type['name']
            if name in processed_data:
                data = processed_data[name]
                # Use timesteps per episode data directly
                timesteps_mean = data['timesteps_per_episode_mean']
                timesteps_std = data['timesteps_per_episode_std']
                
                episodes = np.arange(len(timesteps_mean))
                ax2.plot(episodes, timesteps_mean, label=env_type['label'], color=env_type['color'])
                ax2.fill_between(episodes,
                               timesteps_mean - timesteps_std,
                               timesteps_mean + timesteps_std,
                               alpha=0.2, color=env_type['color'])

        ax2.set_xlabel("Episode Number")
        ax2.set_ylabel("Timesteps per Episode")
        ax2.set_title(f"Timesteps per Episode over {len(SEEDS)} Seeds ({stripped_scenario_folder})" if len(SEEDS) > 1 else f"Timesteps per Episode ({stripped_scenario_folder})")
        ax2.legend(frameon=False)
        ax2.grid(True)

        plot_file = os.path.join(plots_dir, f"averaged_rewards_and_timesteps_{stripped_scenario_folder}.png")
        plt.tight_layout()
        plt.savefig(plot_file)
        print(f"Combined plot saved for scenario {stripped_scenario_folder} at {plot_file}")

        # Plot cross validation results if enabled
        if cross_val_flag:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
            cv_steps = np.arange(len(next(iter(processed_data.values()))['mean_sm'])) * CROSS_VAL_INTERVAL

            # Plot 1: Training curves with cross validation
            for env_type in available_env_types:
                name = env_type['name']
                if name in processed_data:
                    data = processed_data[name]
                    ax1.plot(data['steps_sm'], data['mean_sm'], 
                            label=f"Train {env_type['label']}", color=env_type['color'])
                    ax1.fill_between(data['steps_sm'],
                                   data['mean_sm'] - data['std_sm'],
                                   data['mean_sm'] + data['std_sm'],
                                   alpha=0.2, color=env_type['color'])

                    # Plot cross validation points
                    if all_test_rewards[name]:
                        test_data = np.array(all_test_rewards[name])
                        test_mean = np.mean(test_data, axis=0)
                        test_std = np.std(test_data, axis=0)
                        ax1.plot(cv_steps, test_mean, 'o-', 
                                label=f"CV {env_type['label']}", 
                                color=env_type['color'], alpha=0.5, markersize=4)
                        ax1.fill_between(cv_steps,
                                       test_mean - test_std,
                                       test_mean + test_std,
                                       alpha=0.1, color=env_type['color'])

            ax1.set_xlabel("Environment Steps")
            ax1.set_ylabel("Episode Reward")
            ax1.set_title(f"Training and Cross Validation Rewards over {len(SEEDS)} Seeds ({stripped_scenario_folder})" if len(SEEDS) > 1 else f"Training and Cross Validation Rewards ({stripped_scenario_folder})")
            ax1.legend(frameon=False)
            ax1.grid(True)
            
            # Plot 2: Timesteps per Episode (same as before)
            for env_type in available_env_types:
                name = env_type['name']
                if name in processed_data:
                    data = processed_data[name]
                    # Use timesteps per episode data directly
                    timesteps_mean = data['timesteps_per_episode_mean']
                    timesteps_std = data['timesteps_per_episode_std']
                    
                    episodes = np.arange(len(timesteps_mean))
                    ax2.plot(episodes, timesteps_mean, label=env_type['label'], color=env_type['color'])
                    ax2.fill_between(episodes,
                                   timesteps_mean - timesteps_std,
                                   timesteps_mean + timesteps_std,
                                   alpha=0.2, color=env_type['color'])

            ax2.set_xlabel("Episode Number")
            ax2.set_ylabel("Timesteps per Episode")
            ax2.set_title(f"Timesteps per Episode over {len(SEEDS)} Seeds ({stripped_scenario_folder})" if len(SEEDS) > 1 else f"Timesteps per Episode ({stripped_scenario_folder})")
            ax2.legend(frameon=False)
            ax2.grid(True)

            combined_plot_file = os.path.join(plots_dir, f"averaged_rewards_and_timesteps_{stripped_scenario_folder}_combined.png")
            plt.tight_layout()
            plt.savefig(combined_plot_file)
            print(f"Combined training and cross validation plot saved for scenario {stripped_scenario_folder} at {combined_plot_file}")


def save_experiment_parameters(save_results_big_run, MAX_TOTAL_TIMESTEPS):
    """
    Save all hyperparameters and configuration parameters for reproducibility.
    This includes training parameters, environment config, and reward constants.
    Dynamically reads values from the actual source files.
    """
    import pickle
    from datetime import datetime
    import importlib.util
    
    # Dynamically read training parameters from train_dqn_modular_ssf.py
    training_params = {}
    try:
        # Read the file and extract parameter values using regex
        with open('train_dqn_modular_ssf.py', 'r') as f:
            content = f.read()
        
        # Extract training parameters using regex patterns
        import re
        
        # Define patterns for each parameter (ignore commented lines)
        patterns = {
            'LEARNING_RATE': r'^\s*LEARNING_RATE\s*=\s*([0-9.]+)',
            'GAMMA': r'^\s*GAMMA\s*=\s*([0-9.]+)',
            'BUFFER_SIZE': r'^\s*BUFFER_SIZE\s*=\s*([0-9]+)',
            'BATCH_SIZE': r'^\s*BATCH_SIZE\s*=\s*([0-9]+)',
            'TARGET_UPDATE_INTERVAL': r'^\s*TARGET_UPDATE_INTERVAL\s*=\s*([0-9]+)',
            'LEARNING_STARTS': r'^\s*LEARNING_STARTS\s*=\s*([0-9]+)',
            'TRAIN_FREQ': r'^\s*TRAIN_FREQ\s*=\s*([0-9]+)',
            'MAX_STEPS_PER_SCENARIO': r'^\s*MAX_STEPS_PER_SCENARIO\s*=\s*([0-9]+)',
            'EPSILON_START': r'^\s*EPSILON_START\s*=\s*([0-9.]+)',
            'EPSILON_MIN': r'^\s*EPSILON_MIN\s*=\s*([0-9.]+)',
            'PERCENTAGE_MIN': r'^\s*PERCENTAGE_MIN\s*=\s*([0-9]+)',
            'EPSILON_TYPE': r'^\s*EPSILON_TYPE\s*=\s*["\']([^"\']+)["\']',
        }
        
        for param, pattern in patterns.items():
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                value = match.group(1)
                # Convert to appropriate type
                if param in ['LEARNING_RATE', 'GAMMA', 'EPSILON_START', 'EPSILON_MIN']:
                    training_params[param] = float(value)
                elif param == 'EPSILON_TYPE':
                    training_params[param] = value
                else:
                    training_params[param] = int(value)
        
        # Extract NEURAL_NET_STRUCTURE (more complex pattern, ignore commented lines)
        net_arch_match = re.search(r'^\s*NEURAL_NET_STRUCTURE\s*=\s*dict\(net_arch=\[([^\]]+)\]\)', content, re.MULTILINE)
        if net_arch_match:
            arch_str = net_arch_match.group(1)
            # Parse the architecture
            arch_parts = [part.strip() for part in arch_str.split(',')]
            arch_values = []
            for part in arch_parts:
                if '*' in part:
                    # Handle expressions like "256*2"
                    base, multiplier = part.split('*')
                    arch_values.append(int(base) * int(multiplier))
                else:
                    arch_values.append(int(part))
            training_params['NEURAL_NET_STRUCTURE'] = {'net_arch': arch_values}
        
        # Extract stability parameters (ignore commented lines)
        stability_patterns = {
            'exploration_fraction': r'^\s*exploration_fraction=([0-9.]+)',
            'exploration_initial_eps': r'^\s*exploration_initial_eps=([0-9.]+)',
            'exploration_final_eps': r'^\s*exploration_final_eps=([0-9.]+)',
            'max_grad_norm': r'^\s*max_grad_norm=([0-9.]+)',
            'train_freq': r'^\s*train_freq=([A-Z_]+)',
            'gradient_steps': r'^\s*gradient_steps=([0-9]+)',
        }
        
        stability_params = {}
        for param, pattern in stability_patterns.items():
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                value = match.group(1)
                if param == 'train_freq':
                    # This will be the variable name, we need to get its value
                    var_match = re.search(rf'{value}\s*=\s*([0-9]+)', content)
                    if var_match:
                        stability_params[param] = int(var_match.group(1))
                elif param in ['exploration_fraction', 'exploration_initial_eps', 'exploration_final_eps', 'max_grad_norm']:
                    stability_params[param] = float(value)
                else:
                    stability_params[param] = int(value)
        
    except Exception as e:
        print(f"Warning: Could not extract training parameters: {e}")
        training_params = {}
        stability_params = {}
    
    # Dynamically read reward parameters from config_ssf.py
    reward_params = {}
    try:
        with open('src/config_ssf.py', 'r') as f:
            content = f.read()
        
        reward_patterns = {
            'RESOLVED_CONFLICT_REWARD': r'^\s*RESOLVED_CONFLICT_REWARD\s*=\s*([0-9]+)',
            'DELAY_MINUTE_PENALTY': r'^\s*DELAY_MINUTE_PENALTY\s*=\s*([0-9]+)',
            'CANCELLED_FLIGHT_PENALTY': r'^\s*CANCELLED_FLIGHT_PENALTY\s*=\s*([0-9]+)',
            'NO_ACTION_PENALTY': r'^\s*NO_ACTION_PENALTY\s*=\s*([0-9]+)',
            'AHEAD_PENALTY': r'^\s*AHEAD_PENALTY\s*=\s*([0-9]+)',
            'TIME_MINUTE_PENALTY': r'^\s*TIME_MINUTE_PENALTY\s*=\s*([0-9]+)',
            'AUTOMATIC_CANCELLATION_PENALTY': r'^\s*AUTOMATIC_CANCELLATION_PENALTY\s*=\s*([0-9]+)',
            'MAX_DELAY_PENALTY': r'^\s*MAX_DELAY_PENALTY\s*=\s*([0-9]+)',
        }
        
        for param, pattern in reward_patterns.items():
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                reward_params[param] = int(match.group(1))
        
    except Exception as e:
        print(f"Warning: Could not extract reward parameters: {e}")
        reward_params = {}
    
    # Main.py configuration (only MAX_TOTAL_TIMESTEPS)
    main_config = {
        'MAX_TOTAL_TIMESTEPS': MAX_TOTAL_TIMESTEPS
    }
    
    # Combine all parameters
    all_parameters = {
        'timestamp': datetime.now().isoformat(),
        'training_parameters': training_params,
        'stability_parameters': stability_params,
        'reward_parameters': reward_params,
        'main_configuration': main_config
    }
    
    # Create the parameters directory
    os.makedirs(save_parameters_folder, exist_ok=True)
    
    # Save to pickle file
    params_file = os.path.join(save_parameters_folder, 'experiment_parameters.pkl')
    with open(params_file, 'wb') as f:
        pickle.dump(all_parameters, f)
    
    print(f"Experiment parameters saved to: {params_file}")
    
    # Also save as a readable text file for quick reference
    txt_file = os.path.join(save_parameters_folder, 'experiment_parameters.txt')
    with open(txt_file, 'w') as f:
        f.write("EXPERIMENT PARAMETERS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {all_parameters['timestamp']}\n\n")
        
        f.write("TRAINING PARAMETERS\n")
        f.write("-" * 20 + "\n")
        for key, value in training_params.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nSTABILITY PARAMETERS\n")
        f.write("-" * 20 + "\n")
        for key, value in stability_params.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nREWARD PARAMETERS\n")
        f.write("-" * 20 + "\n")
        for key, value in reward_params.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nMAIN CONFIGURATION\n")
        f.write("-" * 20 + "\n")
        for key, value in main_config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Human-readable parameters saved to: {txt_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_folder", type=str, help="Path to a single training folder")
    parser.add_argument("--seed", type=int, help="Specific seed to run")
    parser.add_argument("--env_type", type=str, choices=['myopic', 'proactive', 'reactive', 'drl-greedy'], help="Environment type to run")
    args = parser.parse_args()

    # Common configuration
    MAX_TOTAL_TIMESTEPS = int(0.1e5)  #5e5 = 500000 timesteps for proper convergence (increased from 200k)
    SEEDS = [232323, 242424]
    brute_force_flag = False
    cross_val_flag = False
    early_stopping_flag = False
    CROSS_VAL_INTERVAL = 1
    printing_intermediate_results = False
    save_training_monitor = False  # Set to True to save training monitor checkpoints and logs (saves ~28GB per run)
    save_folder = "Final_SSFModel_1"
    TESTING_FOLDERS_PATH = "Data/TRAINING50/3ac-702-train/"

    # Define environment types
    env_types = ['myopic', 'proactive', 'reactive']
    # env_types = ['proactive']  # Test only proactive environment

    print(f"Temporal features enabled: {config.ENABLE_TEMPORAL_DERIVED_FEATURES} | "
          f"Derived features/aircraft: {config.DERIVED_FEATURES_PER_AIRCRAFT} | "
          f"Observation stack size: {config.OBS_STACK_SIZE}")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    all_folders_temp = [
        "Data/TRAINING/3ac-130-green/"
        # "Data/TRAINING/3ac-520-blue/"	
        # "Data/TRAINING/3ac-702-train/"
        #"Data/TRAINING/6ac-26-lilac/"
        # "Data/TRAINING/6ac-13-mauve"
        # "Data/TRAINING/6ac-20-lilac/" 
        # "Data/TRAINING/6ac-65-yellow"
        ]
            
    # Save all hyperparameters and configuration for reproducibility
    # Extract scenario folder name and create parameters directory
    # parameters_folder = all_folders_temp[0]
    # stripped_parameters_folder = parameters_folder.strip("/").split("/")[-1]
    # save_parameters_folder = f"{save_folder}/{stripped_parameters_folder}/parameters"
    
    # save_experiment_parameters(save_parameters_folder, MAX_TOTAL_TIMESTEPS)

    # main.py considers 3 different scenarios: 
    # 1. no seed or training folder, 2. seed and training folder, 3. seed and no training folder
    # Case 1/3
    if args.seed is None and args.training_folder is None:
        # Controller mode: Spawn multiple subprocesses, one per combination of seed and env_type
        config_values = get_config_variables(config)
        config_df = pd.DataFrame([config_values])
        config_df['MAX_TOTAL_TIMESTEPS'] = MAX_TOTAL_TIMESTEPS
        config_df['SEEDS'] = str([str(seed) for seed in SEEDS])
        config_df['brute_force_flag'] = brute_force_flag
        config_df['cross_val_flag'] = cross_val_flag
        config_df['early_stopping_flag'] = early_stopping_flag
        config_df['CROSS_VAL_INTERVAL'] = CROSS_VAL_INTERVAL
        config_df['printing_intermediate_results'] = printing_intermediate_results
        config_df['save_training_monitor'] = save_training_monitor
        config_df.to_csv(f"{save_folder}/config.csv", index=False)

        start_time = time.time()

        processes = []
        # Get the path to the virtual environment's Python interpreter
        venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Scripts", "python.exe")
        
        for seed in SEEDS:
            for env_type in env_types:
                cmd = [
                    venv_python,  # Use the virtual environment's Python
                    os.path.abspath(__file__),  # Use absolute path to main.py
                    "--seed", str(seed),
                    "--env_type", env_type,
                    "--training_folder", all_folders_temp[0] #the training folder is specified in all_folders_temp
                ]
                # Set up environment variables for the subprocess
                env = os.environ.copy()
                env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
                
                p = subprocess.Popen(cmd, env=env)
                processes.append(p)

        # Wait for all subprocesses to finish
        for p in processes:
            p.wait()

        # After all combinations are done, aggregate and plot the combined results
        aggregate_results_and_plot(
            SEEDS=SEEDS,
            MAX_TOTAL_TIMESTEPS=MAX_TOTAL_TIMESTEPS,
            brute_force_flag=brute_force_flag,
            cross_val_flag=cross_val_flag,
            early_stopping_flag=early_stopping_flag,
            CROSS_VAL_INTERVAL=CROSS_VAL_INTERVAL,
            printing_intermediate_results=printing_intermediate_results,
            save_folder=save_folder,
            TESTING_FOLDERS_PATH=TESTING_FOLDERS_PATH
        )

        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"\nTotal runtime: {total_runtime:.2f} seconds")

    # specify in terminal sth like: python main.py --seed 232323 --training_folder "Data/TRAINING/6ac-20-lilac/deterministic_Scenario_00001"
    # Case 2/3
    elif args.seed is not None and args.training_folder:
        # Worker mode: run a single scenario folder for a single seed and env_type
        run_for_single_folder(
            training_folder=args.training_folder,
            MAX_TOTAL_TIMESTEPS=MAX_TOTAL_TIMESTEPS,
            single_seed=args.seed,
            brute_force_flag=brute_force_flag,
            cross_val_flag=cross_val_flag,
            early_stopping_flag=early_stopping_flag,
            CROSS_VAL_INTERVAL=CROSS_VAL_INTERVAL,
            printing_intermediate_results=printing_intermediate_results,
            save_folder=save_folder,
            TESTING_FOLDERS_PATH=TESTING_FOLDERS_PATH,
            env_type=args.env_type,
            save_training_monitor=save_training_monitor
        )

    # Case 2/3
    # specify sth like python main.py --seed 232323 --training_folder "Data/TRAINING/6ac-20-lilac/"
    elif args.seed is not None and args.training_folder is None:
        # Worker mode: run all scenario folders for this single seed and env_type
        for folder in all_folders_temp:
            run_for_single_folder(
                training_folder=folder,
                MAX_TOTAL_TIMESTEPS=MAX_TOTAL_TIMESTEPS,
                single_seed=args.seed,
                brute_force_flag=brute_force_flag,
                cross_val_flag=cross_val_flag,
                early_stopping_flag=early_stopping_flag,
                CROSS_VAL_INTERVAL=CROSS_VAL_INTERVAL,
                printing_intermediate_results=printing_intermediate_results,
                save_folder=save_folder,
                TESTING_FOLDERS_PATH=TESTING_FOLDERS_PATH,
                env_type=args.env_type,
                save_training_monitor=save_training_monitor
            )
    else:
        pass