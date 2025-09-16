# This is a test file to check if the code is working
import os
import numpy as np


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
    env_type
):


    N_EPISODES = 10                         # Reduced from 50

    all_logs = {}
    def train_dqn_agent(env_type, seed):
        log_data = {}  # Main dictionary to store all logs

        # Initialize tracking variables
        log_data['metadata'] = {}
        log_data['episodes'] = {}
        log_data['cross_validation'] = {}

        episode_start = 0
        episode = episode_start
        while total_timesteps < MAX_TOTAL_TIMESTEPS:
            rewards[episode] = {}
            episode_data = {
                "episode_number": episode + 1,
                "epsilon_start": epsilon,
                "scenarios": {},
            } 

            for scenario_folder in scenario_folders:
                scenario_data = {
                    "scenario_folder": scenario_folder,
                    "total_reward": 0,
                }

                rewards[episode][scenario_folder] = {}
                best_reward_local = float('-inf')
                total_reward_local = 0
                timesteps_local = 0

                while not done_flag:
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
                                valid_actions = np.where(action_mask == 1)[0]
                                    action = np.random.choice(valid_actions)
                                    action = np.array(action).reshape(1, -1)
                                    action_reason = "..."
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
                    
                    result = env.step(action.item())  # Convert back to scalar for the environment
                    obs_next, reward, terminated, truncated, info = result

                    done_flag = terminated or truncated     # done_flag is True if the episode is terminated or truncated

                    model.replay_buffer.add(
                        obs=obs,
                        next_obs=obs_next,
                        action=action,  # Now action is already in the correct format
                        reward=reward,
                        done=done_flag,
                        infos=[info]
                    )
                    

'''
1. start with 0 TS
1.1 initialize episode_data dictionary
2. goes into Data Set and starts with days/schedules/scenarios 1
3. Takes 1 action. 
4*. adds REWARDS, updated SS, terminated or truncated to results LINE 474, 477
5. sets obs = obs_next, updates epsilon
6. timesteps_local += 1, total_timesteps += 1
7. num_cancelled_flights_after_step, num_delayed_flights_after_step, 
num_penalized_delays_after_step, num_penalized_cancelled_after_step
get updated after each step/action
8. if terminated or truncated is True, break => exit while loop, no actions are taken for this days/schedules/scenarios anymore

IF 8 = True & you break the while loop, you are STILL IN THE SAME days/schedules/scenarios:
1. sum the reward for each action in that days/schedules/scenarios (total local reward)
2. Save it to REWARDS: rewards[episode][scenario_folder]["total"] = total_reward_local
3. scenario_data["total_reward"] = total_reward_local
   episode_data["scenarios"][scenario_folder] = scenario_data
=> so you end up with a dictionary with the scenario folder as the key and the scenario data (total reward) as the value

When all days/schedules/scenarios are done: 
We have:
    1. !!episode_data!! dictionary with episode nr and a SEPARATE ENTRY FOR EACH days/schedules/scenarios and its total local reward
    2. !!rewards!! dictionary with episode nr and a SEPARATE ENTRY FOR EACH days/schedules/scenarios and its total local reward
    3. You are still in episode 0 becuase each days/schedules/scenarios has its own episode 0

Exiting Scenario loop:
1. calculate average reward (summ local rewards of each day and divide by the number of days/schedules/scenarios)



after each action (handle_no_conflicts or handle_flight_operations) check if episode is done by checking if terminated or truncated is True


We have:
*The duration of the episode:
- defined as the amount of time it takes to solve the complete all episode 0s for all days/schedules/scenarios
*The TS of the episode:
- timesteps updated after each action taken no matter what days/schedules/scenarios you are in


''' 
