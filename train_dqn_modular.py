import sys
import torch as th
import os
import warnings
import datetime
import math
import numpy as np
import pandas as pd
import platform
import re
import subprocess
import pickle
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime
from scripts.utils import *
from scripts.visualizations import *
from src.config_rf import *
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import polyak_update, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from scripts.utils import NumpyEncoder
from scripts.logger import *
from stable_baselines3.common.prioritized_replay_buffer import PrioritizedReplayBuffer
from training_utils import TrainingMonitor
import json
import seaborn as sns
sns.set(style="darkgrid")
import time
import subprocess
import psutil
import logging



def run_train_dqn_both_timesteps(
    MAX_TOTAL_TIMESTEPS,
    single_seed,
    brute_force_flag,
    cross_val_flag,
    early_stopping_flag,
    CROSS_VAL_INTERVAL,
    printing_intermediate_results,
    TRAINING_FOLDERS_PATH,
    stripped_scenario_folder,
    save_folder,
    save_results_big_run,
    TESTING_FOLDERS_PATH,
    env_type):

    print(f"Training on {stripped_scenario_folder} with {env_type} environment")
    save_results_big_run = f"{save_folder}/{stripped_scenario_folder}"

    # Constants and Training Settings - FIXED FOR STABLE LEARNING
    LEARNING_RATE = 0.0001                    # the learning rate is the step size for the gradient descent algorithm
    GAMMA = 0.9999                            # Standard discount factor for future rewards.
    BUFFER_SIZE = 100000                     # Increased buffer size for better experience replay
    BATCH_SIZE = 128                         # Standard batch size for stability
    TARGET_UPDATE_INTERVAL = 100            # Less frequent target updates for stability
    NEURAL_NET_STRUCTURE = dict(net_arch=[256, 256*2, 256])
    

    LEARNING_STARTS = 0                # More steps before training starts
    TRAIN_FREQ = 4                          # More frequent training

    # Episode termination settings
    MAX_STEPS_PER_SCENARIO = 40             # Maximum steps per scenario (increased from 25 for more time to solve)

    # Exploration parameters - FIXED FOR STABLE LEARNING
    EPSILON_START = 1.0                      # Starts with 100% random exploration
    EPSILON_MIN = 0.025                        # Higher minimum for gradual transition (increased from 0.01)
    PERCENTAGE_MIN = 80                     # Decay over 80% of training (increased from 40)
    EPSILON_TYPE = "exponential"                 # Use linear decay for more predictable learning
    if EPSILON_TYPE == "linear":
        EPSILON_MIN = 0

    N_EPISODES = 10                         # Reduced from 50

    starting_time = time.time()              # gives the time elapsed since 1970-01-01 00:00:00 UTC aka Unix epoch 

    # extract number of scenarios in training and testing folders
    num_scenarios_training = len(os.listdir(TRAINING_FOLDERS_PATH))

    # Based on parameters, calculate EPSILON_DECAY_RATE as in original code
    EPSILON_DECAY_RATE = calculate_epsilon_decay_rate(
        MAX_TOTAL_TIMESTEPS, EPSILON_START, EPSILON_MIN, PERCENTAGE_MIN, EPSILON_TYPE
    )
    print("EPSILON DECAY RATE: ", EPSILON_DECAY_RATE)

    # Initialize device
    device = initialize_device()

    # Check device capabilities
    check_device_capabilities()

    # Get device-specific information
    device_info = get_device_info(device)
    print(f"Device info: {device_info}")

    # Verify training folders and gather training data
    # returns a list of folders inside the specified path
    training_folders = verify_training_folders(TRAINING_FOLDERS_PATH)

    # Calculate training days and model naming
    num_days_trained_on = calculate_training_days(N_EPISODES, training_folders)
    print(f"Training on {num_days_trained_on} days of data "
          f"({N_EPISODES} episodes of {len(training_folders)} scenarios)")

    formatted_days = format_days(num_days_trained_on)
    MODEL_SAVE_PATH = f'../trained_models/dqn/'  

    # Create results directory
    results_dir = create_results_directory(append_to_name='dqn')
    print(f"Results directory created at: {results_dir}")

    # create_new_id creates a new ID for the training run and adds it to the ids.json file
    # each time you run the main script, a new ID is created
    from scripts.logger import create_new_id, get_config_variables
    # import src.config_rf as config  # Use the same config as the environment LOLL
    import src.config_rf as config  # Use the same config as the environment
    all_logs = {}
    def train_dqn_agent(env_type, seed):
        log_data = {}  # Main dictionary to store all logs

        config_variables = get_config_variables(config)

        # Generate unique ID for training
        training_id = create_new_id("training").split("_")[1]
        runtime_start_in_seconds = time.time()

        # Construct model_path with required directory structure
        model_save_dir = f"{save_folder}/{stripped_scenario_folder}"
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = f"{model_save_dir}/{env_type}_{single_seed}.zip"

        print(f"Models will be saved to: {model_path}")

        training_metadata = {
            "myopic_or_proactive": env_type,
            "model_type": "dqn",
            "training_id": training_id,
            "MODEL_SAVE_PATH": model_path,
            "N_EPISODES": N_EPISODES,
            "num_scenarios_training": num_scenarios_training,
            "results_dir": results_dir,
            "CROSS_VAL_FLAG": cross_val_flag,
            "CROSS_VAL_INTERVAL": CROSS_VAL_INTERVAL,
            **config_variables,
            "LEARNING_RATE": LEARNING_RATE,
            "GAMMA": GAMMA,
            "BUFFER_SIZE": BUFFER_SIZE,
            "BATCH_SIZE": BATCH_SIZE,
            "TARGET_UPDATE_INTERVAL": TARGET_UPDATE_INTERVAL,
            "EPSILON_START": EPSILON_START,
            "EPSILON_MIN": EPSILON_MIN,
            "EPSILON_DECAY_RATE": EPSILON_DECAY_RATE,
            "LEARNING_STARTS": LEARNING_STARTS,
            "TRAIN_FREQ": TRAIN_FREQ,
            "NEURAL_NET_STRUCTURE": NEURAL_NET_STRUCTURE,
            "device_info": str(get_device_info(device)),
            "TRAINING_FOLDERS_PATH": TRAINING_FOLDERS_PATH,
            "TESTING_FOLDERS_PATH": TESTING_FOLDERS_PATH,
            "runtime_start": datetime.utcnow().isoformat() + "Z",
            "runtime_start_in_seconds": runtime_start_in_seconds,
        }

        # Initialize tracking variables
        log_data['metadata'] = {}
        log_data['episodes'] = {}
        log_data['cross_validation'] = {}

        best_reward_avg = float('-inf')

        rewards = {}
        good_rewards = {}
        test_rewards = []
        epsilon_values = []
        total_timesteps = 0  
        consecutive_drops = 0  # Track consecutive performance drops
        best_test_reward = float('-inf')  # Track best test performance
        
        # Initialize detailed episode tracking for visualization
        detailed_episode_data = {}

        def cross_validate_on_test_data(model, current_episode, log_data):
            cross_val_data = {
                "episode": current_episode,
                "scenarios": [],
                "avg_test_reward": 0,
            }

            test_scenario_folders = [
                os.path.join(TESTING_FOLDERS_PATH, folder)
                for folder in os.listdir(TESTING_FOLDERS_PATH)
                if os.path.isdir(os.path.join(TESTING_FOLDERS_PATH, folder))
            ]
            total_test_reward = 0
            for test_scenario_folder in test_scenario_folders:
                scenario_data = {
                    "scenario_folder": test_scenario_folder,
                    "total_reward": 0,
                }
                # Load data
                data_dict = load_scenario_data(test_scenario_folder)
                aircraft_dict = data_dict['aircraft']
                flights_dict = data_dict['flights']
                rotations_dict = data_dict['rotations']
                alt_aircraft_dict = data_dict['alt_aircraft']
                config_dict = data_dict['config']

                # from src.environment_simplified import AircraftDisruptionEnv LOLL
                # from src.original_environment import AircraftDisruptionEnv
                from src.environment_rf import AircraftDisruptionEnv
                env = AircraftDisruptionEnv(
                    aircraft_dict,
                    flights_dict,
                    rotations_dict,
                    alt_aircraft_dict,
                    config_dict,
                    env_type=env_type
                )
                # Set the environment, but handle observation space mismatch gracefully
                try:
                    model.set_env(env)  # Update the model's environment with the new instance
                except ValueError as e:
                    if "Observation spaces do not match" in str(e):
                        print(f"Warning: Observation space mismatch for test scenario {test_scenario_folder}. Skipping this scenario.")
                        continue
                    else:
                        raise e

                obs, _ = env.reset()

                done_flag = False
                total_reward_local = 0
                timesteps_local = 0

                while not done_flag:
                    # Get the action mask from the environment
                    action_mask = obs['action_mask']

                    # Convert observation to float32
                    obs = {key: np.array(value, dtype=np.float32) for key, value in obs.items()}

                    # Preprocess observation and get Q-values
                    obs_tensor = model.policy.obs_to_tensor(obs)[0]
                    q_values = model.policy.q_net(obs_tensor).detach().cpu().numpy().squeeze()

                    # Apply the action mask (set invalid actions to -np.inf)
                    masked_q_values = q_values.copy()
                    masked_q_values[action_mask == 0] = -np.inf

                    # Select the action with the highest masked Q-value
                    action = np.argmax(masked_q_values)

                    # Take the selected action in the environment
                    result = env.step(action)

                    obs_next, reward, terminated, truncated, info = result

                    done_flag = terminated or truncated
                    total_reward_local += reward
                    obs = obs_next

                    timesteps_local += 1
                    if done_flag:
                        break

                total_test_reward += total_reward_local
                scenario_data["total_reward"] = total_reward_local
                cross_val_data["scenarios"].append(scenario_data)

            avg_test_reward = total_test_reward / len(test_scenario_folders)
            cross_val_data["avg_test_reward"] = avg_test_reward
            test_rewards.append(avg_test_reward)
            print(f"cross-val done at episode {current_episode}")

            log_data['cross_validation'] = {}

            return avg_test_reward
        
        # load the training scenarios
        scenario_folders = [
            os.path.join(TRAINING_FOLDERS_PATH, folder)
            for folder in os.listdir(TRAINING_FOLDERS_PATH)
            if os.path.isdir(os.path.join(TRAINING_FOLDERS_PATH, folder))
        ]

        epsilon = EPSILON_START
        total_timesteps = 0

        # Initialize the DQN
        dummy_scenario_folder = scenario_folders[0]
        data_dict = load_scenario_data(dummy_scenario_folder)
        aircraft_dict = data_dict['aircraft']
        flights_dict = data_dict['flights']
        rotations_dict = data_dict['rotations']
        alt_aircraft_dict = data_dict['alt_aircraft']
        config_dict = data_dict['config']

        # initialize the environment
        # from src.environment_simplified import AircraftDisruptionEnv LOLL
        # from src.original_environment import AircraftDisruptionEnv
        from src.environment_rf import AircraftDisruptionEnv
        
        # Create a minimal environment first to get the correct observation space
        # This environment will be used to initialize the model with the correct observation space
        minimal_env = AircraftDisruptionEnv(
            aircraft_dict,
            flights_dict,
            rotations_dict,
            alt_aircraft_dict,
            config_dict,
            env_type=env_type
        )
        
        # Reset the environment to initialize the observation space
        minimal_env.reset()
        stacked_obs_length = minimal_env.observation_space['state'].shape[0]
        single_frame_length = getattr(minimal_env, "single_observation_length", stacked_obs_length)
        obs_stack_size = getattr(minimal_env, "obs_stack_size", 1)
        print(f"Observation vector (stacked): {stacked_obs_length} dims | single frame: {single_frame_length} | stack: {obs_stack_size}")
        training_metadata["stacked_observation_length"] = int(stacked_obs_length)
        training_metadata["single_frame_observation_length"] = int(single_frame_length)
        training_metadata["observation_stack_size"] = int(obs_stack_size)
        
        # Create the model with the correct observation space - OPTIMIZED FOR STABILITY
        model = DQN(
            policy='MultiInputPolicy',
            env=minimal_env,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            target_update_interval=TARGET_UPDATE_INTERVAL,
            verbose=0,
            policy_kwargs=NEURAL_NET_STRUCTURE,
            device=device,
            # Stability parameters
            exploration_fraction=PERCENTAGE_MIN/100,  # Use PERCENTAGE_MIN from top of file
            exploration_initial_eps=EPSILON_START,  # Use EPSILON_START from top of file
            exploration_final_eps=EPSILON_MIN,  # Use EPSILON_MIN from top of file
            max_grad_norm=1.0,  # Less aggressive gradient clipping
            train_freq=TRAIN_FREQ,
            gradient_steps=4  # More gradient steps for better learning
        )

        logger = configure()
        model._logger = logger

        # Confirm device being used by the model
        if th.cuda.is_available():
            print(f"[CONFIRMED] Model is using NVIDIA GPU: {th.cuda.get_device_name(0)}")
            # print(f"[CONFIRMED] All training operations will run on GPU")
        else:
            print("[WARNING] Model is using CPU (no GPU available)")

        print("Initializing training monitor...")
        # Initialize training monitor
        monitor = TrainingMonitor(save_folder, stripped_scenario_folder, env_type, single_seed)
        print(f"Monitor initialized. Log directory: {monitor.log_dir}")

        # Try to load checkpoint (disabled for new environment to avoid observation space mismatch)
        print("Checking for existing checkpoints...")
        checkpoint_data = None
        try:
            checkpoint_data = monitor.load_checkpoint(model)
            if checkpoint_data:
                total_timesteps, episode_start, rewards, test_rewards, epsilon_values, epsilon = checkpoint_data
                print(f"Loaded checkpoint: episode {episode_start}, timesteps {total_timesteps}")
            else:
                print("No checkpoint found, starting fresh training")
                total_timesteps = 0
                episode_start = 0
                rewards = {}
                test_rewards = []
                epsilon_values = []
                epsilon = EPSILON_START
        except Exception as e:
            print(f"Checkpoint loading failed (likely due to observation space mismatch): {e}")
            print("Starting fresh training...")
            total_timesteps = 0
            episode_start = 0
            rewards = {}
            test_rewards = []
            epsilon_values = []
            epsilon = EPSILON_START
        #########################################################

        episode = episode_start
        while total_timesteps < MAX_TOTAL_TIMESTEPS:
            rewards[episode] = {}
            episode_data = {
                "episode_number": episode + 1,
                "epsilon_start": epsilon,
                "scenarios": {},
            }
            
            # Initialize detailed episode tracking for visualization
            detailed_episode_data[episode] = {
                "episode_number": episode + 1,
                "epsilon_start": epsilon,
                "scenarios": {}
            }

            for scenario_folder in scenario_folders:
                scenario_data = {
                    "scenario_folder": scenario_folder,
                    "total_reward": 0,
                }
                rewards[episode][scenario_folder] = {}
                best_reward_local = float('-inf')
                
                # Initialize detailed scenario tracking for visualization
                detailed_episode_data[episode]["scenarios"][scenario_folder] = {
                    "scenario_folder": scenario_folder,
                    "steps": [],
                    "initial_state": None,
                    "total_reward": 0
                }
                data_dict = load_scenario_data(scenario_folder)
                aircraft_dict = data_dict['aircraft']
                flights_dict = data_dict['flights']
                rotations_dict = data_dict['rotations']
                alt_aircraft_dict = data_dict['alt_aircraft']
                config_dict = data_dict['config']

                env = AircraftDisruptionEnv(
                    aircraft_dict,
                    flights_dict,
                    rotations_dict,
                    alt_aircraft_dict,
                    config_dict,
                    env_type=env_type
                )
                model.set_env(env)

                # initializes the environment and returns the initial observation (obs). Underscore _ is used to ignore the second value returned by reset() 
                # obs (or env.reset) is a dictionary: its keys contain the state vector and the action mask i.e which actions are valid at the current step.
                obs, _ = env.reset()
                done_flag = False
                total_reward_local = 0
                timesteps_local = 0
                max_steps_reached = False
                
                # Store initial unavailability probabilities for tracking evolution
                initial_unavailabilities_probabilities = {}
                for aircraft_id in env.aircraft_ids:
                    prob = env.unavailabilities_dict[aircraft_id]['Probability']
                    start = env.unavailabilities_dict[aircraft_id]['StartTime']
                    end = env.unavailabilities_dict[aircraft_id]['EndTime']
                    initial_unavailabilities_probabilities[aircraft_id] = {
                        'probability': float(prob) if not np.isnan(prob) else None,
                        'start_minutes': float(start) if not np.isnan(start) else None,
                        'end_minutes': float(end) if not np.isnan(end) else None
                    }
                
                # Store initial state for visualization
                detailed_episode_data[episode]["scenarios"][scenario_folder]["initial_state"] = {
                    "flights_dict": flights_dict.copy(),
                    "rotations_dict": rotations_dict.copy(),
                    "alt_aircraft_dict": alt_aircraft_dict.copy(),
                    "aircraft_dict": aircraft_dict.copy(),
                    "config_dict": config_dict.copy(),
                    "current_datetime": env.current_datetime,
                    "swapped_flights": env.swapped_flights.copy(),
                    "environment_delayed_flights": env.environment_delayed_flights.copy(),
                    "cancelled_flights": env.cancelled_flights.copy(),
                    "unavailabilities_probabilities": initial_unavailabilities_probabilities  # Store initial probabilities
                }

                while not done_flag and not max_steps_reached:
                    num_cancelled_flights_before_step = len(env.cancelled_flights)             # immediately after reset this is 0 as there are no cancelled flights at the start of the episode
                    num_delayed_flights_before_step = len(env.environment_delayed_flights)     # empty dictionary
                    num_penalized_delays_before_step = len(env.penalized_delays)               # empty dictionary
                    num_penalized_cancelled_before_step = len(env.penalized_cancelled_flights)
                    num_automatically_cancelled_before_step = len(env.automatically_cancelled_flights)
                    num_penalized_automatically_cancelled_before_step = len(env.penalized_automatically_cancelled_flights) # empty set

                    model.exploration_rate = epsilon

                    action_mask = obs['action_mask']
                    obs = {key: np.array(value, dtype=np.float32) for key, value in obs.items()}

                    obs_tensor = model.policy.obs_to_tensor(obs)[0]
                    q_values = model.policy.q_net(obs_tensor).detach().cpu().numpy().squeeze()

                    masked_q_values = q_values.copy()
                    masked_q_values[action_mask == 0] = -np.inf

                    current_seed = int(time.time() * 1e9) % (2**32 - 1)
                    np.random.seed(current_seed)

                    action_reason = "None"
                    if env_type == "drl-greedy" or env_type == "myopic" or env_type == "proactive" or env_type == "reactive":
                        if np.random.rand() < epsilon or brute_force_flag:
                            # During exploration (50% the conflicted flights, 50% random)
                            if np.random.rand() < 0.5:
                                # Get current conflicts and use them to guide exploration
                                # Prefer actions that operate on conflicted flights (any aircraft); rely on env scheduling
                                current_conflicts = env.get_current_conflicts()
                                if current_conflicts:
                                    # Build set of conflicted flight IDs from conflict tuples: (aircraft_id, flight_id, dep, arr)
                                    conflicted_flight_ids = set()
                                    for conf in current_conflicts:
                                        if isinstance(conf, tuple) and len(conf) >= 2:
                                            conflicted_flight_ids.add(conf[1])

                                    # Get current action mask
                                    action_mask = env.get_action_mask()
                                    all_valid = np.where(action_mask == 1)[0]
                                    ac_count_plus_one = len(env.aircraft_ids) + 1

                                    # Restrict to actions whose flight component is conflicted (plus allow (0,0))
                                    restricted = [
                                        idx for idx in all_valid
                                        if idx == 0 or (idx // ac_count_plus_one) in conflicted_flight_ids
                                    ]

                                    # Use restricted actions if any exist, otherwise fall back to all valid
                                    candidates = restricted if len(restricted) > 0 else all_valid
                                    action = np.random.choice(candidates)
                                    action = np.array(action).reshape(1, -1)
                                    action_reason = "conflict-guided-random"
                                else:
                                    # If no conflicts, fall back to random exploration
                                    valid_actions = np.where(action_mask == 1)[0]
                                    action = np.random.choice(valid_actions)
                                    action = np.array(action).reshape(1, -1)
                                    action_reason = "exploration"
                            else:
                                # Random exploration from original action mask
                                valid_actions = np.where(action_mask == 1)[0]
                                action = np.random.choice(valid_actions)
                                action = np.array(action).reshape(1, -1)
                                action_reason = "exploration"
                        else:
                            # Exploitation: always use Q-values
                            action = np.argmax(masked_q_values)
                            action = np.array(action).reshape(1, -1)
                            action_reason = "exploitation"
                    else:
                        # For other environment types, use standard DQN logic
                        if np.random.rand() < epsilon or brute_force_flag:
                            valid_actions = np.where(action_mask == 1)[0]
                            action = np.random.choice(valid_actions)
                            action = np.array(action).reshape(1, -1)
                            action_reason = "exploration"
                        else:
                            action = np.argmax(masked_q_values)
                            action = np.array(action).reshape(1, -1)
                            action_reason = "exploitation"

                    result = env.step(action.item())  # Convert back to scalar for the environment
                    obs_next, reward, terminated, truncated, info = result
                    # print(f"    ****    Action: {action}, flight: {env.map_index_to_action(action)[0]}, aircraft: {env.map_index_to_action(action)[1]}")

                    done_flag = terminated or truncated     # done_flag is True if the episode is terminated or truncated

                    model.replay_buffer.add(
                        obs=obs,
                        next_obs=obs_next,
                        action=action,  # Now action is already in the correct format
                        reward=reward,
                        done=done_flag,
                        infos=[info]
                    )

                    obs = obs_next

                    # # OPTIMIZED: Use exponential epsilon decay for better exploration
                    # if EPSILON_TYPE == "linear":
                    #     progress = total_timesteps / (MAX_TOTAL_TIMESTEPS * PERCENTAGE_MIN / 100)
                    #     epsilon = max(EPSILON_MIN, EPSILON_START - progress * (EPSILON_START - EPSILON_MIN))
                    # else:
                    #     # Exponential decay with better rate
                    #     # epsilon = max(EPSILON_MIN, epsilon * (1 - EPSILON_DECAY_RATE))
                    #     decay_rate = 0.9995  # Slower decay for better exploration
                    #     epsilon = max(EPSILON_MIN, epsilon * decay_rate)
                    # epsilon_values.append((episode + 1, epsilon))

                    epsilon = max(EPSILON_MIN, epsilon * (1 - EPSILON_DECAY_RATE))
                    epsilon_values.append((episode + 1, epsilon))

                    timesteps_local += 1
                    total_timesteps += 1
                    
                    # Check if we've reached the maximum steps for this scenario
                    step_limit_penalty = 0.0  # Initialize penalty
                    if timesteps_local >= MAX_STEPS_PER_SCENARIO:
                        max_steps_reached = True
                        # Apply penalty for hitting step limit with unresolved conflicts
                        if env.check_flight_disruption_overlaps():
                            step_limit_penalty = -3000.0  # Large negative reward for failing to resolve conflicts (matches cancellation penalty after scaling)
                            # Add the penalty to the replay buffer
                            model.replay_buffer.add(
                                obs=obs,
                                next_obs=obs,  # Same state since we're terminating
                                action=action,
                                reward=step_limit_penalty,
                                done=True,
                                infos=[{"step_limit_penalty": True}]
                            )
                    
                    # Track reward for plotting (includes penalty if applied)
                    rewards[episode][scenario_folder][timesteps_local] = reward + step_limit_penalty
                    
                    # Removed frequent progress logging to avoid slowing down training

                    if total_timesteps > model.learning_starts and total_timesteps % TRAIN_FREQ == 0:
                        model.train(gradient_steps=1, batch_size=BATCH_SIZE)

                    if total_timesteps % model.target_update_interval == 0:
                        polyak_update(model.q_net.parameters(), model.q_net_target.parameters(), model.tau)
                        polyak_update(model.batch_norm_stats, model.batch_norm_stats_target, 1.0)

                    # Monitor system health every 10,000 timesteps
                    if total_timesteps % 10000 == 0:
                        monitor.check_system_health(
                            model, total_timesteps, episode, rewards, test_rewards, epsilon_values, epsilon
                        )

                    # Add checkpoint saving every 50,000 timesteps
                    if total_timesteps % 50000 == 0:
                        monitor.save_checkpoint(
                            model, total_timesteps, episode, rewards, test_rewards, epsilon_values, epsilon
                        )

                    num_cancelled_flights_after_step = len(env.cancelled_flights)
                    num_delayed_flights_after_step = len(env.environment_delayed_flights)
                    num_penalized_delays_after_step = len(env.penalized_delays)
                    num_penalized_cancelled_after_step = len(env.penalized_cancelled_flights)

                    impact_of_action = {
                        "num_cancelled_flights": num_cancelled_flights_after_step - num_cancelled_flights_before_step,
                        "num_delayed_flights": num_delayed_flights_after_step - num_delayed_flights_before_step,
                        "num_penalized_delays": num_penalized_delays_after_step - num_penalized_delays_before_step,
                        "num_penalized_cancelled": num_penalized_cancelled_after_step - num_penalized_cancelled_before_step,
                    }
                    
                    # Extract penalty details from info dict
                    penalties_dict = info.get("penalties", {})
                    delay_penalty_total = penalties_dict.get("delay_penalty_total", 0.0)
                    cancel_penalty = penalties_dict.get("cancel_penalty", 0.0)
                    inaction_penalty = penalties_dict.get("inaction_penalty", 0.0)
                    automatic_cancellation_penalty = penalties_dict.get("automatic_cancellation_penalty", 0.0)
                    proactive_penalty = penalties_dict.get("proactive_penalty", 0.0)
                    time_penalty = penalties_dict.get("time_penalty", 0.0)
                    final_conflict_resolution_reward = penalties_dict.get("final_conflict_resolution_reward", 0.0)
                    probability_resolution_bonus = penalties_dict.get("probability_resolution_bonus", 0.0)  # Reward #8
                    low_confidence_action_penalty = penalties_dict.get("low_confidence_action_penalty", 0.0)  # Penalty #9
                    something_happened = info.get("something_happened", False)
                    scenario_ended = info.get("scenario_ended", False)  # Get scenario_ended flag
                    penalty_flags = info.get("penalty_flags", {})  # Get penalty enable flags
                    delay_penalty_minutes = info.get("delay_penalty_minutes", 0)  # Get delay minutes for display
                    
                    # Get decoded action from info dict (stored by env.step) if available
                    # This ensures we use the actual action that was executed, not re-decode after flights may have been removed
                    flight_action = info.get("flight_action", None)
                    aircraft_action = info.get("aircraft_action", None)
                    action_index_from_info = info.get("action_index", None)
                    
                    # Fallback: decode from action index if not in info (for backward compatibility)
                    if flight_action is None or aircraft_action is None:
                        flight_action, aircraft_action = env.map_index_to_action(action.item())
                        action_index_from_info = action.item()
                    
                    # Extract unavailability probabilities from info
                    unavailabilities_probabilities = info.get("unavailabilities_probabilities", {})
                    
                    # Store detailed step information for visualization (after all variables are calculated)
                    step_info = {
                        "step": timesteps_local,
                        "action": action.item(),  # Original action index chosen by agent
                        "action_decoded": [flight_action, aircraft_action],  # Add decoded action as [flight, aircraft]
                        "flight_action": flight_action,  # Flight ID (0 = no action/cancellation, -1 = invalid)
                        "aircraft_action": aircraft_action,  # Aircraft ID (0 = no action/cancellation, -1 = invalid)
                        "action_reason": action_reason,
                        "epsilon": epsilon,
                        "reward": reward, # reward for this step
                        "something_happened": something_happened,  # Track if action actually changed something
                        "scenario_ended": scenario_ended,  # Track if scenario ended and final reward was calculated
                        "penalties": {
                            "delay": delay_penalty_total,  # Old format for backward compatibility
                            "delay_penalty_total": delay_penalty_total,  # New format
                            "cancellation": cancel_penalty,  # Old format
                            "cancel_penalty": cancel_penalty,  # New format
                            "inaction": inaction_penalty,  # Old format
                            "inaction_penalty": inaction_penalty,  # New format
                            "automatic_cancellation": automatic_cancellation_penalty,  # Old format
                            "automatic_cancellation_penalty": automatic_cancellation_penalty,  # New format
                            "proactive": proactive_penalty,  # Old format
                            "proactive_penalty": proactive_penalty,  # New format
                            "time": time_penalty,  # Old format
                            "time_penalty": time_penalty,  # New format
                            "final_conflict_resolution_reward": final_conflict_resolution_reward,
                            "probability_resolution_bonus": probability_resolution_bonus,  # Reward #8
                            "low_confidence_action_penalty": low_confidence_action_penalty,  # Penalty #9
                        },
                        "delay_penalty_minutes": delay_penalty_minutes,  # Delay minutes for display
                        "current_time_minutes": info.get("current_time_minutes", None),  # Current time in minutes
                        "current_time_minutes_from_start": info.get("current_time_minutes_from_start", None),  # Time from start
                        "total_reward_so_far": total_reward_local + reward, # total reward till now for this day/scenario/schedule in this episode
                        "current_datetime": env.current_datetime, # where we are in the recovery schedule
                        "swapped_flights": env.swapped_flights.copy(),
                        "environment_delayed_flights": env.environment_delayed_flights.copy(),
                        "cancelled_flights": env.cancelled_flights.copy(),
                        "penalized_delays": env.penalized_delays.copy(),
                        "penalized_cancelled_flights": env.penalized_cancelled_flights.copy(),
                        "impact_of_action": impact_of_action,
                        "penalty_flags": penalty_flags,  # Store penalty flags for analysis
                        "flights_dict": {k: v.copy() for k, v in env.flights_dict.items()},  # Store updated flight times
                        "rotations_dict": {k: v.copy() for k, v in env.rotations_dict.items()},  # Store updated rotations
                        "unavailabilities_probabilities": unavailabilities_probabilities  # Store probability evolution
                    }
                    detailed_episode_data[episode]["scenarios"][scenario_folder]["steps"].append(step_info)

                    # Break if max steps reached or episode naturally terminated
                    if max_steps_reached or done_flag:
                        break

                # total_reward_local = sum(rewards[episode][scenario_folder].values())
                # Sum only step rewards (numeric keys), excluding any metadata keys like "total"
                step_rewards = [v for k, v in rewards[episode][scenario_folder].items() if isinstance(k, (int, np.integer))]
                total_reward_local = sum(step_rewards) if step_rewards else 0
                rewards[episode][scenario_folder]["total"] = total_reward_local

                scenario_data["total_reward"] = total_reward_local
                scenario_data["episode_ended_reason"] = "max_steps_reached" if max_steps_reached else "natural_termination"
                episode_data["scenarios"][scenario_folder] = scenario_data # this returns a dictionary with the scenario folder as the key and the scenario data as the value
                
                # Update detailed episode data with final totals
                detailed_episode_data[episode]["scenarios"][scenario_folder]["total_reward"] = total_reward_local
                detailed_episode_data[episode]["scenarios"][scenario_folder]["total_steps"] = timesteps_local
                detailed_episode_data[episode]["scenarios"][scenario_folder]["episode_ended_reason"] = "max_steps_reached" if max_steps_reached else "natural_termination"

            # Perform cross-validation if enabled
            if cross_val_flag:
                if (episode + 1) % CROSS_VAL_INTERVAL == 0:
                    current_test_reward = cross_validate_on_test_data(model, episode + 1, log_data)
                    if not hasattr(train_dqn_agent, 'best_test_reward'):
                        train_dqn_agent.best_test_reward = current_test_reward
                    best_test_reward_local = train_dqn_agent.best_test_reward

                    # Early stopping logic
                    if current_test_reward < best_test_reward_local:
                        consecutive_drops += 1
                        print(f"Performance drop {consecutive_drops}/5 (current: {current_test_reward:.2f}, best: {best_test_reward_local:.2f})")
                        if consecutive_drops >= 500:
                            print(f"Early stopping triggered at episode {episode + 1} due to 5 consecutive drops in test performance")
                            break
                    else:
                        consecutive_drops = 0
                        train_dqn_agent.best_test_reward = current_test_reward
                        best_test_reward_local = current_test_reward

            # Calculate the average reward for this episode (so all scenarios for episode 0, or 1, or 2, etc.)
            avg_reward_for_this_batch = 0
            max_steps_hit_count = 0
            for i in range(len(scenario_folders)):
                avg_reward_for_this_batch += rewards[episode][scenario_folders[i]]["total"]
                # Count how many scenarios hit the step limit
                if episode_data["scenarios"][scenario_folders[i]]["episode_ended_reason"] == "max_steps_reached":
                    max_steps_hit_count += 1
            avg_reward_for_this_batch /= len(scenario_folders)

            rewards[episode]["avg_reward"] = avg_reward_for_this_batch
            rewards[episode]["total_timesteps"] = total_timesteps
            rewards[episode]["max_steps_hit_count"] = max_steps_hit_count

            current_time = time.time()
            percentage_complete = (total_timesteps / MAX_TOTAL_TIMESTEPS) * 100

            # Calculate time per 10k timesteps and store timesteps for this episode
            if episode > 0:
                time_this_episode = current_time - previous_episode_time
                timesteps_this_episode = total_timesteps - previous_timesteps
                time_per_10000 = (time_this_episode / timesteps_this_episode) * 10000
                
                # Store timesteps for this episode (not cumulative)
                rewards[episode]["timesteps_this_episode"] = timesteps_this_episode
                
                # Calculate remaining time based on recent timestep rate
                remaining_timesteps = MAX_TOTAL_TIMESTEPS - total_timesteps
                estimated_time_remaining = (time_per_10000 / 10000) * remaining_timesteps
                hours = int(estimated_time_remaining // 3600)
                minutes = int((estimated_time_remaining % 3600) // 60)
            else:
                # For the first episode, timesteps_this_episode is the same as total_timesteps
                rewards[episode]["timesteps_this_episode"] = total_timesteps
                time_per_10000 = 0
                hours = 0
                minutes = 0
            
            rewards[episode]["timestamp"] = current_time
            time_remaining_str = f"{hours}h{minutes}m" if hours > 0 else f"{minutes}m"
            step_limit_info = f" (max_steps_hit: {max_steps_hit_count}/{len(scenario_folders)})" if max_steps_hit_count > 0 else ""
            print(f"({total_timesteps:.0f}/{MAX_TOTAL_TIMESTEPS:.0f} - {percentage_complete:.0f}% - {time_remaining_str} remaining, {time_per_10000:.0f}s/10k steps) {env_type:<10} - episode {episode + 1} - epsilon {epsilon:.2f} - reward this episode: {avg_reward_for_this_batch:.2f}{step_limit_info}")

            previous_episode_time = current_time
            previous_timesteps = total_timesteps

            episode_data["avg_reward"] = avg_reward_for_this_batch
            log_data['episodes'] = {}
            episode += 1

        model.save(model_path)
        runtime_end_in_seconds = time.time()
        runtime_in_seconds = runtime_end_in_seconds - runtime_start_in_seconds
        actual_total_timesteps = total_timesteps

        # Return collected data
        return rewards, test_rewards, total_timesteps, epsilon_values, good_rewards, {}, model_path, detailed_episode_data

    # Run training for the specified environment type only
    rewards, test_rewards, total_timesteps, epsilon_values, good_rewards, action_sequences, model_path, detailed_episode_data = train_dqn_agent(env_type, single_seed)

            # Extract only the necessary data
    # here we save the average rewards for each episode (where episode reward = [(avg reward of schedule 1 for episode 0 + avg reward of schedule 2 for episode 0 + ... + avg reward of schedule n for episode 0)/n]
    # and the steps per episode 
    episode_rewards = [rewards[e]["avg_reward"] for e in sorted(rewards.keys()) if "avg_reward" in rewards[e]]
    episode_steps_cumulative = [rewards[e]["total_timesteps"] for e in sorted(rewards.keys()) if "total_timesteps" in rewards[e]]
    episode_steps_per_episode = [rewards[e]["timesteps_this_episode"] for e in sorted(rewards.keys()) if "timesteps_this_episode" in rewards[e]]
    nr_episodes = len(episode_rewards)

    # Save results for this environment type
    os.makedirs(f"{save_results_big_run}/numpy", exist_ok=True)
    np.save(f'{save_results_big_run}/numpy/{env_type}_runs_seed_{single_seed}.npy', np.array(episode_rewards))
    np.save(f'{save_results_big_run}/numpy/{env_type}_steps_runs_seed_{single_seed}.npy', np.array(episode_steps_cumulative))
    np.save(f'{save_results_big_run}/numpy/{env_type}_timesteps_per_episode_seed_{single_seed}.npy', np.array(episode_steps_per_episode))
    if test_rewards:  # Only save if we have test rewards
        np.save(f'{save_results_big_run}/numpy/test_rewards_{env_type}_seed_{single_seed}.npy', test_rewards)
    
    # Save detailed episode data for visualization
    os.makedirs(f"{save_results_big_run}/detailed_episodes", exist_ok=True)
    import pickle
    with open(f'{save_results_big_run}/detailed_episodes/{env_type}_detailed_episodes_seed_{single_seed}.pkl', 'wb') as f:
        pickle.dump(detailed_episode_data, f)

    # Return empty arrays for the other environment types
    empty_array = np.array([])
    if env_type == 'myopic':
        return episode_rewards, empty_array, empty_array, empty_array, test_rewards, [], [], []
    elif env_type == 'proactive':
        return empty_array, episode_rewards, empty_array, empty_array, [], test_rewards, [], []
    elif env_type == 'reactive':
        return empty_array, empty_array, episode_rewards, empty_array, [], [], test_rewards, []
    else:  # drl-greedy
        return empty_array, empty_array, empty_array, episode_rewards, [], [], [], test_rewards
