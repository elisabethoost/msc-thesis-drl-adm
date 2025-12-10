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
from src.config_rf import *
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import polyak_update, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from scripts.utils import NumpyEncoder
from scripts.logger import *
from stable_baselines3.common.prioritized_replay_buffer import PrioritizedReplayBuffer
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

    save_results_big_run = f"{save_folder}/{stripped_scenario_folder}"

    # Constants and Training Settings - OPTIMIZED FOR 50K TIMESTEPS
    LEARNING_RATE = 0.001                    # Increased from 0.0005 for faster learning with limited timesteps
    GAMMA = 0.995                            # Reduced from 0.9999 to reduce impact of distant penalties, focus on immediate rewards
    BUFFER_SIZE = 100000                    # Large buffer for diverse experience replay
    BATCH_SIZE = 256                        # Increased from 128: with ~840 steps/episode, larger batches improve stability
    TARGET_UPDATE_INTERVAL = 200            # Increased from 100: more stable target network updates
    NEURAL_NET_STRUCTURE = dict(net_arch=[256, 256*2, 256])
    

    LEARNING_STARTS = 1000                  # Start training after collecting some diverse experience
    TRAIN_FREQ = 8                          # Reduced from 20: train more frequently for faster learning
    GRADIENT_STEPS = 2                      # More gradient steps per training call for better learning

    # Episode termination settings
    MAX_STEPS_PER_SCENARIO = 40            

    # Exploration parameters - OPTIMIZED FOR 50K TIMESTEPS
    EPSILON_START = 1.0                      # Starts with 100% random exploration
    EPSILON_MIN = 0.05                       # Reduced from 0.15: less exploration late in training, more exploitation
    PERCENTAGE_MIN = 70                     # Reduced from 90: reach min epsilon faster (at 60% = 30k steps), more exploitation time
    EPSILON_TYPE = "exponential"             # Use exponential decay for gradual learning
    if EPSILON_TYPE == "linear":
        EPSILON_MIN = 0
    
    # Exploration-exploitation matching: probability of using Q-values (with noise) during exploration
    # This helps the buffer contain some greedy-like actions, reducing distribution mismatch
    EXPLORATION_Q_PROB = 0.3                # 30% of exploration steps use Q-values with noise instead of pure random
    EXPLORATION_CONFLICT_GUIDED_PROB = 0.3  # 30% conflict-guided exploration (reduced from 56% to allow more random exploration)
    EXPLORATION_NOISE_SCALE = 0.2            # Noise scale for Q-value exploration (relative to Q-value range)

    starting_time = time.time()              # gives the time elapsed since 1970-01-01 00:00:00 UTC aka Unix epoch 

    # extract number of scenarios in training and testing folders
    num_scenarios_training = len(os.listdir(TRAINING_FOLDERS_PATH))

    # Based on parameters, calculate EPSILON_DECAY_RATE
    if EPSILON_TYPE != "mixed":
        EPSILON_DECAY_RATE = calculate_epsilon_decay_rate(
            MAX_TOTAL_TIMESTEPS, EPSILON_START, EPSILON_MIN, PERCENTAGE_MIN, EPSILON_TYPE
        )
    if EPSILON_TYPE == "mixed":
        EPSILON_DECAY_RATE, EPSILON_DECAY_RATE_LINEAR = calculate_epsilon_decay_rate(
            MAX_TOTAL_TIMESTEPS, EPSILON_START, EPSILON_MIN, PERCENTAGE_MIN, EPSILON_TYPE
        )

    # Initialize device
    device = initialize_device()
    check_device_capabilities()
    device_info = get_device_info(device)

    # Create results directory
    results_dir = create_results_directory(append_to_name='dqn')
    print(f"Results directory created at: {results_dir}")

    # create_new_id creates a new ID for the training run and adds it to the ids.json file
    # each time you run the main script, a new ID is created
    from scripts.logger import create_new_id, get_config_variables

    import src.config_rf as config # LOLL
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
            "GRADIENT_STEPS": GRADIENT_STEPS,
            "NEURAL_NET_STRUCTURE": NEURAL_NET_STRUCTURE,
            "EXPLORATION_Q_PROB": EXPLORATION_Q_PROB,
            "EXPLORATION_CONFLICT_GUIDED_PROB": EXPLORATION_CONFLICT_GUIDED_PROB,
            "EXPLORATION_NOISE_SCALE": EXPLORATION_NOISE_SCALE,
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
        
        # Initialize metrics tracking for analysis
        episode_metrics = {
            'conflicts_resolved': [],           # Number of conflicts resolved per episode
            'conflicts_total': [],              # Total conflicts at start per episode
            'max_steps_hit': [],                # Number of scenarios hitting max steps per episode
            'resolution_bonus_total': [],       # Total resolution bonuses per episode
            'unresolved_penalty_total': [],     # Total unresolved penalties per episode
            'exploration_actions': [],          # Count of exploration actions per episode
            'exploitation_actions': [],         # Count of exploitation actions per episode
            'conflict_guided_actions': [],      # Count of conflict-guided actions per episode
            'q_value_exploration_actions': []   # Count of Q-value exploration actions per episode
        }

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

        # Sort scenarios by difficulty: deterministic (p=1.0) first, then stochastic (0<p<1)
        def get_scenario_difficulty(scenario_folder):
            """Return difficulty score: lower = easier (more deterministic)
            Score = stochastic_count - deterministic_count
            Negative scores = more deterministic (easier)
            Positive scores = more stochastic (harder)
            """
            try:
                data_dict = load_scenario_data(scenario_folder)
                alt_aircraft_dict = data_dict.get('alt_aircraft', {})
                
                deterministic_count = 0
                stochastic_count = 0
                
                for ac_id, unavail_info in alt_aircraft_dict.items():
                    # Handle both list and single dict formats
                    if not isinstance(unavail_info, list):
                        unavail_info = [unavail_info]
                    
                    for unavail in unavail_info:
                        prob = unavail.get('Probability', 1.0)
                        if prob == 1.0:
                            deterministic_count += 1
                        elif 0 < prob < 1:
                            stochastic_count += 1
                
                # Lower score = easier (more deterministic)
                return stochastic_count - deterministic_count
            except Exception as e:
                # If loading fails, assume medium difficulty
                print(f"Warning: Could not load scenario {scenario_folder} for difficulty sorting: {e}")
                return 0
        
        # Sort scenarios: deterministic first (negative scores), then stochastic (positive scores)
        scenario_folders.sort(key=get_scenario_difficulty)
        print(f"Sorted {len(scenario_folders)} scenarios by difficulty (deterministic first, stochastic last)")

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
        # Use PrioritizedReplayBuffer to emphasize positive reward experiences (like +5000 bonuses)
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
            # Prioritized Replay Buffer: samples high-reward transitions more often
            replay_buffer_class=PrioritizedReplayBuffer,
            replay_buffer_kwargs={
                'alpha': 0.6,  # Prioritization strength (0=uniform, 1=full priority)
                'beta': 0.4,   # Importance sampling correction (0=no correction, 1=full correction)
                'epsilon': 1e-6  # Small constant to ensure non-zero priorities
            },
            # Stability parameters
            exploration_fraction=PERCENTAGE_MIN/100,  # Use PERCENTAGE_MIN from top of file
            exploration_initial_eps=EPSILON_START,  # Use EPSILON_START from top of file
            exploration_final_eps=EPSILON_MIN,  # Use EPSILON_MIN from top of file
            max_grad_norm=1.0,  # Less aggressive gradient clipping
            train_freq=TRAIN_FREQ,
            gradient_steps=GRADIENT_STEPS  # Use GRADIENT_STEPS from top of file
        )

        logger = configure()
        model._logger = logger

        # Confirm device being used by the model
        if th.cuda.is_available():
            print(f"[CONFIRMED] Model is using NVIDIA GPU: {th.cuda.get_device_name(0)}")
            # print(f"[CONFIRMED] All training operations will run on GPU")
        else:
            print("[WARNING] Model is using CPU (no GPU available)")

        # Initialize training variables
        total_timesteps = 0
        episode_start = 0
        rewards = {}
        test_rewards = []
        epsilon_values = []
        epsilon = EPSILON_START

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
            
            # Initialize episode metrics counters
            episode_conflicts_resolved = 0
            episode_conflicts_total = 0
            episode_max_steps_hit = 0
            episode_resolution_bonus = 0.0
            episode_unresolved_penalty = 0.0
            episode_exploration_count = 0
            episode_exploitation_count = 0
            episode_conflict_guided_count = 0
            episode_q_exploration_count = 0

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
                
                # Track initial conflicts for this scenario
                initial_conflicts = env.get_current_conflicts()
                scenario_initial_conflicts = len(initial_conflicts) if initial_conflicts else 0
                episode_conflicts_total += scenario_initial_conflicts
                
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
                    model.exploration_rate = epsilon

                    action_mask = obs['action_mask']
                    obs = {key: np.array(value, dtype=np.float32) for key, value in obs.items()}

                    obs_tensor = model.policy.obs_to_tensor(obs)[0]
                    q_values = model.policy.q_net(obs_tensor).detach().cpu().numpy().squeeze()

                    masked_q_values = q_values.copy()
                    masked_q_values[action_mask == 0] = -np.inf

                    current_seed = int(time.time() * 1e9) % (2**32 - 1)
                    np.random.seed(current_seed)

                    if DEBUG_MODE_REWARD:
                        print("")
                        print(f"Current scenario: {scenario_folder}, epsilon: {epsilon}, episode: {episode}, local timesteps: {timesteps_local}, total timesteps: {total_timesteps}")

                    action_reason = "None"
                    if env_type == "drl-greedy" or env_type == "myopic" or env_type == "proactive" or env_type == "reactive":
                        if np.random.rand() < epsilon or brute_force_flag:
                            # During exploration: mix conflict-guided, Q-value-based, and pure random
                            exploration_choice = np.random.rand()
                            
                            # Option 1: Use Q-values with noise (30% of exploration) - matches exploitation distribution better
                            if exploration_choice < EXPLORATION_Q_PROB:
                                # Add noise to Q-values for exploration
                                q_noise = np.random.normal(0, EXPLORATION_NOISE_SCALE * np.std(masked_q_values[masked_q_values != -np.inf]), size=masked_q_values.shape)
                                noisy_q_values = masked_q_values + q_noise
                                noisy_q_values[action_mask == 0] = -np.inf  # Re-apply mask
                                action = np.argmax(noisy_q_values)
                                action = np.array(action).reshape(1, -1)
                                action_reason = "q-value-exploration"
                                episode_q_exploration_count += 1
                            
                            # Option 2: Conflict-guided exploration (30% of total exploration)
                            elif exploration_choice < EXPLORATION_Q_PROB + EXPLORATION_CONFLICT_GUIDED_PROB:
                                # Get current conflicts and use them to guide exploration
                                if DEBUG_MODE_REWARD:
                                    print(f"Current conflicts train_dqn_modular:")
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
                                    episode_conflict_guided_count += 1
                                else:
                                    # If no conflicts, fall back to random exploration
                                    valid_actions = np.where(action_mask == 1)[0]
                                    action = np.random.choice(valid_actions)
                                    action = np.array(action).reshape(1, -1)
                                    action_reason = "exploration"
                            else:
                                # Pure random exploration (remaining 40% of total exploration - ensures broad coverage)
                                valid_actions = np.where(action_mask == 1)[0]
                                action = np.random.choice(valid_actions)
                                action = np.array(action).reshape(1, -1)
                                action_reason = "exploration"
                            
                            episode_exploration_count += 1
                        else:
                            # Exploitation: always use Q-values
                            action = np.argmax(masked_q_values)
                            action = np.array(action).reshape(1, -1)
                            action_reason = "exploitation"
                            episode_exploitation_count += 1
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
                    
                    if DEBUG_MODE_REWARD:
                        print(f"Action: {action}, flight_action, aircraft_action = {env.map_index_to_action(action.item())}")
                        print(f"action reason: {action_reason}")

                    result = env.step(action.item())  # Convert back to scalar for the environment
                    obs_next, reward, terminated, truncated, info = result

                    done_flag = terminated or truncated 

                    model.replay_buffer.add(
                        obs=obs,
                        next_obs=obs_next,
                        action=action,  # Now action is already in the correct format
                        reward=reward,
                        done=done_flag,
                        infos=[info]
                    )

                    obs = obs_next

                    if EPSILON_TYPE != "mixed":
                        epsilon = max(EPSILON_MIN, epsilon * (1 - EPSILON_DECAY_RATE))
                    elif EPSILON_TYPE == "mixed":
                        if total_timesteps <= int(MAX_TOTAL_TIMESTEPS/2):
                            epsilon = max(EPSILON_MIN, epsilon * (1 - EPSILON_DECAY_RATE_LINEAR))
                        else:
                            epsilon = max(EPSILON_MIN, epsilon * (1 - EPSILON_DECAY_RATE))
                    epsilon_values.append((episode + 1, epsilon))

                    timesteps_local += 1
                    total_timesteps += 1
                    
                    # Check if we've reached the maximum steps for this scenario
                    step_limit_penalty = 0.0  # Initialize penalty
                    if timesteps_local >= MAX_STEPS_PER_SCENARIO:
                        max_steps_reached = True
                        episode_max_steps_hit += 1
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
                        model.train(gradient_steps=GRADIENT_STEPS, batch_size=BATCH_SIZE) #performs mini SGD (stochastic gradient descent)

                    if total_timesteps % model.target_update_interval == 0:
                        polyak_update(model.q_net.parameters(), model.q_net_target.parameters(), model.tau)
                        polyak_update(model.batch_norm_stats, model.batch_norm_stats_target, 1.0)


                    # Extract scenario-wide metrics from info dict (only present when scenario ends)
                    scenario_metrics = info.get("scenario_metrics", None)
                    
                    # Extract penalty details from info dict
                    penalties_dict = info.get("penalties", {})
                    delay_penalty_total = penalties_dict.get("delay_penalty_total", 0.0)
                    cancel_penalty = penalties_dict.get("cancel_penalty", 0.0)
                    inaction_penalty = penalties_dict.get("inaction_penalty", 0.0)
                    automatic_cancellation_penalty = penalties_dict.get("automatic_cancellation_penalty", 0.0)
                    proactive_penalty = penalties_dict.get("proactive_penalty", 0.0)
                    time_penalty = penalties_dict.get("time_penalty", 0.0)
                    unresolved_conflict_penalty = penalties_dict.get("unresolved_conflict_penalty", 0.0)
                    probability_resolution_bonus = penalties_dict.get("probability_resolution_bonus", 0.0)  # Reward #8
                    low_confidence_action_penalty = penalties_dict.get("low_confidence_action_penalty", 0.0)  # Penalty #9
                    something_happened = info.get("something_happened", False)
                    scenario_ended = info.get("scenario_ended", False)  # Get scenario_ended flag
                    penalty_flags = info.get("penalty_flags", {})  # Get penalty enable flags
                    delay_penalty_minutes = info.get("delay_penalty_minutes", 0)  # Get delay minutes for display
                    
                    # Track metrics when scenario ends
                    if scenario_ended:
                        episode_resolution_bonus += probability_resolution_bonus
                        episode_unresolved_penalty += unresolved_conflict_penalty
                        # Check if conflicts were resolved (simplified: if we got resolution bonus, conflicts were resolved)
                        if probability_resolution_bonus > 0:
                            episode_conflicts_resolved += int(probability_resolution_bonus / PROBABILITY_RESOLUTION_BONUS_SCALE)
                    
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
                    
                    # Extract conflict counters from state (after action, before observation processing)
                    # State has conflict counters in columns 4-5 of each aircraft row (rows 1-3)
                    conflict_counters = {}
                    state_after_action = env.state  # State after action but before process_observation
                    for idx, aircraft_id in enumerate(env.aircraft_ids):
                        if idx >= env.max_aircraft:
                            break
                        row_idx = idx + 1  # Aircraft rows start at index 1
                        initial_count = float(state_after_action[row_idx, 4]) if not np.isnan(state_after_action[row_idx, 4]) else 0.0
                        current_count = float(state_after_action[row_idx, 5]) if not np.isnan(state_after_action[row_idx, 5]) else 0.0
                        conflict_counters[aircraft_id] = {
                            "initial": initial_count,
                            "current": current_count
                        }
                    
                    # Get actual conflicts for verification (compare with counters)
                    actual_conflicts = env.get_current_conflicts()
                    # Count conflicts per aircraft for comparison
                    actual_conflicts_per_ac = {}
                    for aircraft_id in env.aircraft_ids:
                        count = 0
                        for conflict in actual_conflicts:
                            if isinstance(conflict, (tuple, list)) and len(conflict) >= 2:
                                if conflict[0] == aircraft_id:
                                    count += 1
                            elif conflict == aircraft_id:
                                count += 1
                        actual_conflicts_per_ac[aircraft_id] = count
                    
                    # Store Q-values for analysis (before action is taken)
                    # Get Q-values for the chosen action and top actions
                    chosen_action_q = float(masked_q_values[action.item()]) if action.item() < len(masked_q_values) else None
                    valid_q_values = masked_q_values[masked_q_values != -np.inf]
                    top_q_value = float(np.max(valid_q_values)) if len(valid_q_values) > 0 else None
                    mean_q_value = float(np.mean(valid_q_values)) if len(valid_q_values) > 0 else None
                    std_q_value = float(np.std(valid_q_values)) if len(valid_q_values) > 0 else None
                    
                    # Get top 5 Q-values and their actions for analysis
                    top_5_indices = np.argsort(masked_q_values)[-5:][::-1]  # Top 5, descending
                    top_5_q_values = [(int(idx), float(masked_q_values[idx])) for idx in top_5_indices if masked_q_values[idx] != -np.inf]
                    
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
                        "q_values": {
                            "chosen_action_q": chosen_action_q,  # Q-value of the action that was taken
                            "top_q_value": top_q_value,  # Highest Q-value among valid actions
                            "mean_q_value": mean_q_value,  # Mean Q-value of valid actions
                            "std_q_value": std_q_value,  # Std of Q-values of valid actions
                            "top_5_actions": top_5_q_values,  # Top 5 actions and their Q-values
                        },
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
                            "unresolved_conflict_penalty": unresolved_conflict_penalty,
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
                        "penalty_flags": penalty_flags,  # Store penalty flags for analysis
                        "flights_dict": {k: v.copy() for k, v in env.flights_dict.items()},  # Store updated flight times
                        "rotations_dict": {k: v.copy() for k, v in env.rotations_dict.items()},  # Store updated rotations
                        "unavailabilities_probabilities": unavailabilities_probabilities,  # Store probability evolution
                        "conflict_counters": conflict_counters,  # Store conflict counters per aircraft (initial and current from state)
                        "actual_conflicts_per_ac": actual_conflicts_per_ac  # Store actual conflict counts per aircraft (for verification)
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
                
                # Only store scenario metrics if they exist (i.e., scenario ended properly with penalty #6)
                if scenario_metrics is not None:
                    detailed_episode_data[episode]["scenarios"][scenario_folder]["final_scenario_metrics"] = scenario_metrics

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
            
            # Store episode metrics
            episode_metrics['conflicts_resolved'].append(episode_conflicts_resolved)
            episode_metrics['conflicts_total'].append(episode_conflicts_total)
            episode_metrics['max_steps_hit'].append(episode_max_steps_hit)
            episode_metrics['resolution_bonus_total'].append(episode_resolution_bonus)
            episode_metrics['unresolved_penalty_total'].append(episode_unresolved_penalty)
            episode_metrics['exploration_actions'].append(episode_exploration_count)
            episode_metrics['exploitation_actions'].append(episode_exploitation_count)
            episode_metrics['conflict_guided_actions'].append(episode_conflict_guided_count)
            episode_metrics['q_value_exploration_actions'].append(episode_q_exploration_count)
            
            # Calculate success rate (conflicts resolved / total conflicts)
            success_rate = (episode_conflicts_resolved / episode_conflicts_total * 100) if episode_conflicts_total > 0 else 0.0

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
            conflicts_info = f" | conflicts: {episode_conflicts_resolved}/{episode_conflicts_total} resolved ({success_rate:.1f}%)" if episode_conflicts_total > 0 else ""
            action_info = f" | actions: {episode_exploitation_count} exploit, {episode_exploration_count} explore ({episode_q_exploration_count} Q-explore, {episode_conflict_guided_count} conflict-guided)"
            print(f"({total_timesteps:.0f}/{MAX_TOTAL_TIMESTEPS:.0f} - {percentage_complete:.0f}% - {time_remaining_str} remaining, {time_per_10000:.0f}s/10k steps) {env_type:<10} - episode {episode + 1} - epsilon {epsilon:.2f} - reward: {avg_reward_for_this_batch:.2f}{step_limit_info}{conflicts_info}{action_info}")

            previous_episode_time = current_time
            previous_timesteps = total_timesteps

            episode_data["avg_reward"] = avg_reward_for_this_batch
            log_data['episodes'] = {}
            episode += 1

        model.save(model_path)
        runtime_end_in_seconds = time.time()
        runtime_in_seconds = runtime_end_in_seconds - runtime_start_in_seconds
        actual_total_timesteps = total_timesteps
        
        # Save episode metrics to file for analysis
        metrics_save_path = f"{save_results_big_run}/numpy/{env_type}_metrics_seed_{single_seed}.npz"
        np.savez(metrics_save_path, **episode_metrics)
        print(f"Episode metrics saved to: {metrics_save_path}")
        
        # Print summary statistics
        if episode_metrics['conflicts_total']:
            avg_conflicts_total = np.mean(episode_metrics['conflicts_total'])
            avg_conflicts_resolved = np.mean(episode_metrics['conflicts_resolved'])
            overall_success_rate = (np.sum(episode_metrics['conflicts_resolved']) / np.sum(episode_metrics['conflicts_total']) * 100) if np.sum(episode_metrics['conflicts_total']) > 0 else 0.0
            total_max_steps_hit = np.sum(episode_metrics['max_steps_hit'])
            avg_resolution_bonus = np.mean(episode_metrics['resolution_bonus_total'])
            avg_unresolved_penalty = np.mean(episode_metrics['unresolved_penalty_total'])
            print(f"\n=== Training Summary for {env_type} (seed {single_seed}) ===")
            print(f"Overall conflict resolution rate: {overall_success_rate:.1f}% ({np.sum(episode_metrics['conflicts_resolved'])}/{np.sum(episode_metrics['conflicts_total'])})")
            print(f"Average conflicts per episode: {avg_conflicts_total:.1f}")
            print(f"Average conflicts resolved per episode: {avg_conflicts_resolved:.1f}")
            print(f"Total max steps hit: {total_max_steps_hit}")
            print(f"Average resolution bonus per episode: {avg_resolution_bonus:.2f}")
            print(f"Average unresolved penalty per episode: {avg_unresolved_penalty:.2f}")
            print(f"Total exploration actions: {np.sum(episode_metrics['exploration_actions'])}")
            print(f"Total exploitation actions: {np.sum(episode_metrics['exploitation_actions'])}")
            print(f"Total Q-value exploration actions: {np.sum(episode_metrics['q_value_exploration_actions'])}")
            print(f"Total conflict-guided actions: {np.sum(episode_metrics['conflict_guided_actions'])}")
            print("=" * 60)

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