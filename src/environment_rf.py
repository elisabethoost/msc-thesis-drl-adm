import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
from collections import deque
from src.config_rf import *
from scripts.utils import *
import time
import random
from typing import Dict, List, Tuple
import os
# old environment_copy.py

class AircraftDisruptionEnv(gym.Env):
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict, env_type):
        """Initializes the AircraftDisruptionEnv class.

        Args:
            aircraft_dict (dict): Dictionary containing aircraft information.
            flights_dict (dict): Dictionary containing flight information.
            rotations_dict (dict): Dictionary containing rotation information.
            alt_aircraft_dict (dict): Dictionary containing alternative aircraft information.
            config_dict (dict): Dictionary containing configuration information.
            env_type (str): Type of environment ('myopic' or 'proactive', 'reactive', 'drl-greedy').
        """
        super(AircraftDisruptionEnv, self).__init__()
        
        # Store the environment type ('myopic' or 'proactive', 'reactive', 'drl-greedy')
        self.env_type = env_type  
        
        # Constants for environment configuration
        self.max_aircraft = MAX_AIRCRAFT
        self.columns_state_space = COLUMNS_STATE_SPACE 
        self.rows_state_space = ROWS_STATE_SPACE

        self.config_dict = config_dict

        # Define the recovery period based on provided configuration
        start_date = config_dict['RecoveryPeriod']['StartDate']
        start_time = config_dict['RecoveryPeriod']['StartTime']
        end_date = config_dict['RecoveryPeriod']['EndDate']
        end_time = config_dict['RecoveryPeriod']['EndTime']
        self.start_datetime = datetime.strptime(f"{start_date} {start_time}", '%d/%m/%y %H:%M')
        self.end_datetime = datetime.strptime(f"{end_date} {end_time}", '%d/%m/%y %H:%M')
        self.timestep = timedelta(hours=TIMESTEP_HOURS)
        self.timestep_minutes = TIMESTEP_HOURS * 60
        self.recovery_period_minutes = max(1, (self.end_datetime - self.start_datetime).total_seconds() / 60)

        # Aircraft information and indexing
        self.aircraft_ids = list(aircraft_dict.keys()) # list of aircraft ids e.g. ['B737#1', 'A320#2', 'B737#2']
        self.aircraft_id_to_idx = {aircraft_id: idx for idx, aircraft_id in enumerate(self.aircraft_ids)} # dict of aircraft ids to indices e.g. {'B737#1': 0, 'A320#2': 1, 'B737#2': 2}

        self.conflicted_flights = {}  # Tracks flights in conflict due to past departure and prob == 1.0
    

        # Flight information and indexing
        # if flights_dict is empty, flights_dict is empty
        if flights_dict is None:
            flights_dict = {}  # Initialize as empty dict if None

        if flights_dict:
            self.flight_ids = list(flights_dict.keys())
            self.flight_id_to_idx = {flight_id: idx for idx, flight_id in enumerate(self.flight_ids)}
        else:
            self.flight_ids = []
            self.flight_id_to_idx = {}

        # Filter out flights with '+' in DepTime (next day flights) - ehhm okay why do I not use this? 
        this_day_flights = [flight_info for flight_info in flights_dict.values() if '+' not in flight_info['DepTime']]

        self.earliest_datetime = self.start_datetime

        # Define observation and action spaces
        self.action_space_size = ACTION_SPACE_SIZE
        self.action_space = spaces.Discrete(self.action_space_size)
        self.enable_temporal_features = ENABLE_TEMPORAL_DERIVED_FEATURES
        self.num_temporal_features = DERIVED_FEATURES_PER_AIRCRAFT if self.enable_temporal_features else 0
        self.obs_stack_size = max(1, OBS_STACK_SIZE)
        self.base_state_vector_length = self.rows_state_space * self.columns_state_space
        self.temporal_feature_length = self.max_aircraft * self.num_temporal_features
        self.single_observation_length = self.base_state_vector_length + self.temporal_feature_length
        self.stacked_state_length = self.single_observation_length * self.obs_stack_size

        self.observation_space = spaces.Dict({
            'state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.stacked_state_length,),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(self.action_space.n,),
                dtype=np.uint8
            )
        })

        # Store the dictionaries as class attributes
        self.alt_aircraft_dict = alt_aircraft_dict
        self.rotations_dict = rotations_dict
        self.flights_dict = flights_dict
        self.aircraft_dict = aircraft_dict

        # Deep copies of initial data to reset the environment later
        self.initial_aircraft_dict = copy.deepcopy(aircraft_dict)
        self.initial_flights_dict = copy.deepcopy(flights_dict)
        self.initial_rotations_dict = copy.deepcopy(rotations_dict)
        self.initial_alt_aircraft_dict = copy.deepcopy(alt_aircraft_dict)

        # Track environment state related to delays and conflicts
        self.environment_delayed_flights = {}       
        self.penalized_delays = {}           
        self.penalized_cancelled_flights = set()  
        self.cancelled_flights = set()
        self.automatically_cancelled_flights = set()
        self.penalized_automatically_cancelled_flights = set()
        
        self.penalized_conflicts = set()        
        self.resolved_conflicts = set()         
        

        # Initialize empty containers for breakdowns
        self.uncertain_breakdowns = {}
        self.current_breakdowns = {}

        # Initialize a dictionary to store unavailabilities
        self.unavailabilities_dict = {}

        self.info_after_step = {}
        self.last_action_probability = None
        self.last_action_aircraft_id = None
        self.pre_action_probabilities = {}

        # Initialize the environment state without generating probabilities
        self.current_datetime = self.start_datetime
        self.something_happened = False  # Track if current action actually changed something
        self.previous_probabilities = {aircraft_id: 0.0 for aircraft_id in self.aircraft_ids}
        self.state_history = deque(maxlen=self.obs_stack_size)
        self.state = self._get_initial_state()

        # Initialize eligible flights for conflict resolution bonus
        self.eligible_flights_for_resolved_bonus = self.get_initial_conflicts()
        self.eligible_flights_for_not_being_cancelled_when_disruption_happens = self.get_initial_conflicts_with_deptime_before_unavail_start()

        # Scenario-wide tracking metrics for analysis
        self.scenario_wide_delay_minutes = 0                    
        self.scenario_wide_delay_count = 0                      # Number of flights delayed
        self.scenario_wide_cancelled_flights = 0                # Number of manually cancelled flights
        self.scenario_wide_automatically_cancelled_count = 0    # Number of automatically cancelled flights
        self.scenario_wide_steps = 0                            # Total steps taken
        self.scenario_wide_resolved_initial_conflicts = 0       # Conflicts resolved at end
        self.scenario_wide_disruption_resolved_to_zero_count = 0
        self.scenario_wide_solution_slack = 0                   # Solution quality metric
        self.scenario_wide_tail_swaps = 0                       # Total tail swaps
        self.scenario_wide_tail_swaps_resolving = 0             # Tail swaps that resolved conflicts (got reward #8)
        self.scenario_wide_tail_swaps_inefficient = 0           # Tail swaps that didn't resolve conflicts
        self.scenario_wide_inaction_count = 0                   # Count of (0,0) actions
        if DEBUG_MODE_REWARD:
            print(f"[{self.env_type} environment initialized]")
            print(f"scenario wide initial disrupted flights length:")
        
        self.scenario_wide_initial_disrupted_flights_list = self.get_current_conflicts()
        self.scenario_wide_actual_disrupted_flights = len(self.get_current_conflicts())

        self.scenario_wide_reward_components = {
            "delay_penalty_total": 0,
            "cancel_penalty": 0,
            "inaction_penalty": 0,
            "proactive_penalty": 0,
            "time_penalty": 0,
            "unresolved_conflict_penalty": 0,
            "automatic_cancellation_penalty": 0,
            "probability_resolution_bonus": 0,
            "low_confidence_action_penalty": 0,
            "action_taking_bonus": 0
        }
 
        # Initialize tail swap tracking
        self.tail_swap_happened = False

    def _get_initial_state(self):
        """Initializes the state matrix for the environment.
        
        Returns:
            np.ndarray: The initial state matrix.
        """

        # Initialize state matrix with NaN values
        state = np.full((self.rows_state_space, self.columns_state_space), np.nan)

        # Calculate current time and remaining recovery period in minutes
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
        time_until_end_minutes = (self.end_datetime - self.current_datetime).total_seconds() / 60

        # Insert the current_time_minutes and time_until_end_minutes in the first row
        for i in range(0, 2):  
            if i + 1 < self.columns_state_space:  
                state[0, i] = current_time_minutes  
                state[0, i + 1] = time_until_end_minutes  

        # List to keep track of flights to remove from dictionaries
        flights_to_remove = set()

        # Set to collect actual flights in state space
        active_flights = set()

        # Populate state matrix with aircraft and flight information
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break  

            state[idx + 1, 0] = idx + 1  

            # Check for predefined unavailabilities and assign actual probability         
            if aircraft_id in self.alt_aircraft_dict:
                unavails = self.alt_aircraft_dict[aircraft_id] # list of unavailabilities for the aircraft e.g. [{'StartDate': '01/07/25', 'StartTime': '00:00', 'EndDate': '01/07/25', 'EndTime': '00:00', 'Probability': 1.0}]
                if not isinstance(unavails, list):
                    unavails = [unavails]
                breakdown_probability = unavails[0].get('Probability', 1.0) 

                # Get earliest start and latest end time
                start_times = []
                end_times = []
                for unavail_info in unavails:
                    unavail_start_time = datetime.strptime(unavail_info['StartDate'] + ' ' + unavail_info['StartTime'], '%d/%m/%y %H:%M')
                    unavail_end_time = datetime.strptime(unavail_info['EndDate'] + ' ' + unavail_info['EndTime'], '%d/%m/%y %H:%M')
                    start_times.append((unavail_start_time - self.earliest_datetime).total_seconds() / 60)
                    end_times.append((unavail_end_time - self.earliest_datetime).total_seconds() / 60)

                if start_times:
                    unavail_start_minutes = min(start_times)
                    unavail_end_minutes = max(end_times)

            # Check for uncertain breakdowns
            elif aircraft_id in self.uncertain_breakdowns:
                breakdown_info = self.uncertain_breakdowns[aircraft_id][0]  # Get first breakdown
                breakdown_probability = breakdown_info['Probability']  # Use existing probability
                unavail_start_minutes = (breakdown_info['StartTime'] - self.earliest_datetime).total_seconds() / 60
                unavail_end_minutes = (breakdown_info['EndTime'] - self.earliest_datetime).total_seconds() / 60

            else:
                # No unavailability, set default values
                breakdown_probability = 0.0
                unavail_start_minutes = np.nan
                unavail_end_minutes = np.nan

            # Store the unavailability information in the unavailabilities dictionary
            self.unavailabilities_dict[aircraft_id] = {
                'Probability': breakdown_probability,
                'StartTime': unavail_start_minutes,
                'EndTime': unavail_end_minutes
            }
            self.previous_probabilities[aircraft_id] = 0.0 if np.isnan(breakdown_probability) else breakdown_probability

            # # In the myopic env, the info for uncertain breakdowns is not shown
            # if breakdown_probability != 1.0 and self.env_type == 'myopic':
            #     breakdown_probability = np.nan  # Set to NaN if not 1.00
            #     unavail_start_minutes = np.nan
            #     unavail_end_minutes = np.nan

            # # In the proactive env, the info for unrealized breakdowns is also not shown
            # if np.isnan(breakdown_probability) or breakdown_probability == 0.00:  # Added check for 0.00
            #     breakdown_probability = np.nan  # Set to NaN if not 1.00
            #     unavail_start_minutes = np.nan
            #     unavail_end_minutes = np.nan


            # # In the reactive env, the info for unrealized breakdowns is also not shown
            # if self.env_type == 'reactive':  # Added check for 0.00
            #     breakdown_probability = np.nan  # Set to NaN if not 1.00
            #     unavail_start_minutes = np.nan
            #     unavail_end_minutes = np.nan

            # Store probability and unavailability times
            state[idx + 1, 1] = breakdown_probability
            state[idx + 1, 2] = unavail_start_minutes
            state[idx + 1, 3] = unavail_end_minutes

            # Gather and store flight times (starting from column 4)
            # we add every flight that has not yet departed and that is not in conflict with an unavailability period
            flight_times = []
            for flight_id, rotation_info in self.rotations_dict.items():
                if flight_id in self.flights_dict and rotation_info['Aircraft'] == aircraft_id:
                    flight_info = self.flights_dict[flight_id]
                    dep_time = parse_time_with_day_offset(flight_info['DepTime'], self.start_datetime)
                    arr_time = parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime)

                    dep_time_minutes = (dep_time - self.earliest_datetime).total_seconds() / 60
                    arr_time_minutes = (arr_time - self.earliest_datetime).total_seconds() / 60

                    # Include AUTOMATIC CANCELLATION LOGIC again
                    # Exclude flights that have already departed and are in conflict
                    if dep_time_minutes < current_time_minutes:
                        # Flight has already departed
                        if breakdown_probability == 1.00 and not np.isnan(unavail_start_minutes) and not np.isnan(unavail_end_minutes):
                            # There is an unavailability with prob == 1.00
                            # Check if the flight overlaps with the unavailability
                            if dep_time_minutes < unavail_end_minutes and arr_time_minutes > unavail_start_minutes:
                                if DEBUG_MODE_CANCELLED_FLIGHT:
                                    print(f"REMOVING FLIGHT {flight_id} DUE TO UNAVAILABILITY AND PAST DEPARTURE")
                                # Flight is in conflict with unavailability
                                flights_to_remove.add(flight_id)
                                continue


                    flight_times.append((flight_id, dep_time_minutes, arr_time_minutes))
                    active_flights.add(flight_id)  # Add to active flights set

            # Sort flights by departure time
            flight_times.sort(key=lambda x: x[1])

            # Store flight information starting from column 4
            for i, (flight_id, dep_time, arr_time) in enumerate(flight_times):
                col_start = 4 + (i * 3)
                if col_start + 2 < self.columns_state_space:
                    state[idx + 1, col_start] = flight_id
                    state[idx + 1, col_start + 1] = dep_time
                    state[idx + 1, col_start + 2] = arr_time

        # Update flight_id_to_idx with only the active flights
        self.flight_id_to_idx = {
            flight_id: idx for idx, flight_id in enumerate(sorted(active_flights))
        }

        # Remove past flights from dictionaries
        for flight_id in flights_to_remove:
            self.remove_flight(flight_id)

        return state

    def process_observation(self, state):
        """Processes the observation by applying env_type-specific masking.
        Does NOT modify internal state or unavailabilities_dict.
        Returns an observation suitable for the agent."""
        state_to_observe = state.copy()
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
        derived_feature_rows = []

        for idx in range(self.max_aircraft):
            if idx >= len(self.aircraft_ids):
                if self.num_temporal_features > 0:
                    derived_feature_rows.append(np.full(self.num_temporal_features, np.nan, dtype=np.float32))
                continue

            aircraft_id = self.aircraft_ids[idx]

            real_breakdown_probability = state[idx + 1, 1]
            real_unavail_start_minutes = state[idx + 1, 2]
            real_unavail_end_minutes = state[idx + 1, 3]

            # Make copies for observation only
            obs_breakdown_probability = real_breakdown_probability
            obs_unavail_start_minutes = real_unavail_start_minutes
            obs_unavail_end_minutes = real_unavail_end_minutes

            # Apply env_type logic to these observed values ONLY
            if self.env_type == 'reactive':
                # Reactive sees no info about disruptions
                obs_breakdown_probability = np.nan
                obs_unavail_start_minutes = np.nan
                obs_unavail_end_minutes = np.nan
            elif self.env_type == 'myopic':
                # Myopic only sees if probability == 1.00
                if obs_breakdown_probability != 1.0:
                    obs_breakdown_probability = np.nan
                    obs_unavail_start_minutes = np.nan
                    obs_unavail_end_minutes = np.nan
            elif self.env_type == 'proactive' or self.env_type == 'drl-greedy':
                # Proactive and drl-greedy see everything (no masking needed)
                pass

            # Assign the masked values back to the observation copy
            state_to_observe[idx + 1, 1] = obs_breakdown_probability
            state_to_observe[idx + 1, 2] = obs_unavail_start_minutes
            state_to_observe[idx + 1, 3] = obs_unavail_end_minutes

            if self.enable_temporal_features and self.num_temporal_features > 0:
                aircraft_features = self._get_aircraft_temporal_features(
                    aircraft_id,
                    real_breakdown_probability,
                    real_unavail_start_minutes,
                    real_unavail_end_minutes,
                    current_time_minutes
                )

                features_for_observation = aircraft_features.copy()
                if self.env_type == 'reactive':
                    features_for_observation[:] = np.nan
                elif self.env_type == 'myopic':
                    if np.isnan(obs_breakdown_probability) or obs_breakdown_probability != 1.0:
                        features_for_observation[:] = np.nan
                elif np.isnan(obs_breakdown_probability):
                    features_for_observation[:] = np.nan

                derived_feature_rows.append(features_for_observation)

            # Update previous probability tracker using real (unmasked) probability
            self.previous_probabilities[aircraft_id] = 0.0 if np.isnan(real_breakdown_probability) else real_breakdown_probability

        if self.enable_temporal_features and self.num_temporal_features > 0:
            # Pad derived features if fewer aircraft were processed
            while len(derived_feature_rows) < self.max_aircraft:
                derived_feature_rows.append(np.full(self.num_temporal_features, np.nan, dtype=np.float32))
            derived_features_vector = np.array(derived_feature_rows, dtype=np.float32).flatten()
        else:
            derived_features_vector = np.array([], dtype=np.float32)

        # Create a mask where 1 indicates valid values, 0 indicates NaN
        mask = np.where(np.isnan(state_to_observe), 0, 1)
        # Replace NaN with a dummy value
        state_to_observe = np.nan_to_num(state_to_observe, nan=DUMMY_VALUE)

        # Flatten both state and mask
        state_flat = state_to_observe.flatten()

        if derived_features_vector.size > 0:
            derived_features_vector = np.nan_to_num(derived_features_vector, nan=DUMMY_VALUE)
            single_state_vector = np.concatenate([state_flat, derived_features_vector]).astype(np.float32)
        else:
            single_state_vector = state_flat.astype(np.float32)

        stacked_state = self._stack_observation(single_state_vector)

        # Use get_action_mask to generate the action mask
        action_mask = self.get_action_mask()

        # Return the observation dictionary without modifying internal structures
        obs_with_mask = {
            'state': stacked_state,
            'action_mask': action_mask
        }
        return obs_with_mask, state_to_observe
        # obs_with_mask: A dictionary containing both the state vector and the action mask (= a vector indicating which actions are valid (1 = valid, 0 = invalid) in the current state)
        # state_to_observe: A copy of the state matrix that is used to update the state matrix after each step but with certain information hidden

    # YES: need to modify for new state-space system
    def fix_state(self, state):
        # Go over all starttimes and endtimes (columns 2 and 3 for unavailabilities and then for flights: 5, 6, 8, 9, 11, 12, ...)
        # If endtime is smaller than starttime, add 1440 minutes to endtime
        for i in range(1, self.rows_state_space):
            if not np.isnan(state[i, 2]) and not np.isnan(state[i, 3]) and state[i, 2] > state[i, 3]:
                state[i, 3] += 1440
                # Update unavailabilities_dict to keep it in sync
                aircraft_id = self.aircraft_ids[i - 1]
                if aircraft_id in self.unavailabilities_dict:
                    self.unavailabilities_dict[aircraft_id]['EndTime'] = state[i, 3]
            for j in range(4, self.columns_state_space - 2, 3):
                if not np.isnan(state[i, j + 1]) and not np.isnan(state[i, j + 2]) and state[i, j + 1] > state[i, j + 2]:
                    state[i, j + 2] += 1440

    def remove_flight(self, flight_id):
        """Removes the specified flight from the dictionaries. Adds it to the cancelled_flights set."""
        # Remove from flights_dict
        if flight_id in self.flights_dict:
            del self.flights_dict[flight_id]

        # Remove from rotations_dict
        if flight_id in self.rotations_dict:
            del self.rotations_dict[flight_id]

        # Mark the flight as canceled
        #self.cancelled_flights.add(flight_id)
        self.automatically_cancelled_flights.add(flight_id)


    def step(self, action_index):
        """Executes a step in the environment based on the provided action.

        This function processes the action taken by the agent, checks for conflicts, updates the environment state,
        and returns the new state, reward, termination status, truncation status, and additional info.

        Args:
            action (tuple or list): The action to be taken by the agent.

        Returns:
            tuple: A tuple containing the processed state, reward, terminated flag, truncated flag, and additional info.

        Notes:
            - Invalid actions: If the decoded action is invalid (i.e., (-1, -1)), the environment immediately returns
              a reward of -1000, advances time by one timestep, and sets the following in the info dict:
              "invalid_action": True and "invalid_action_reason". This should not occur if get_action_mask() is correct,
              but can happen due to race conditions or masking bugs. The early check prevents crashes and makes the
              penalty visible in analysis.
        """

        # Reset something_happened at the start of each step
        self.something_happened = False

        # Fix the state before processing the action
        self.fix_state(self.state)

        # Extract the action values from the action
        flight_action, aircraft_action = self.map_index_to_action(action_index)
        self.last_action_probability = None
        self.last_action_aircraft_id = None
        if aircraft_action > 0 and aircraft_action <= len(self.aircraft_ids):
            acted_aircraft_id = self.aircraft_ids[aircraft_action - 1]
            self.last_action_aircraft_id = acted_aircraft_id
            prob_snapshot = self.unavailabilities_dict.get(acted_aircraft_id, {}).get('Probability', np.nan)
            if not np.isnan(prob_snapshot):
                self.last_action_probability = float(prob_snapshot)
        
        if flight_action == -1 or aircraft_action == -1:
            # Invalid action detected early - return negative reward immediately
            # Store action info for debugging
            reward = -1000  # Large negative reward for invalid actions
            terminated = self.check_termination_criteria()
            truncated = False
            
            # Proceed to next timestep
            next_datetime = self.current_datetime + self.timestep
            self.current_datetime = next_datetime
            self.state = self._get_initial_state()
            
            processed_state, _ = self.process_observation(self.state)
            
            # Prepare info dict with action information for debugging
            info = {
                "flight_action": flight_action,
                "aircraft_action": aircraft_action,
                "action_index": action_index,
                "invalid_action": True,
                "invalid_action_reason": f"map_index_to_action({action_index}) returned (-1, -1) - flight not in flights_dict"
            }
            
            # Still update info_after_step for consistency
            self.info_after_step = {
                "total_reward": reward,
                "something_happened": False,
                "current_time_minutes": (self.current_datetime - self.earliest_datetime).total_seconds() / 60,
                "current_time_minutes_from_start": (self.current_datetime - self.start_datetime).total_seconds() / 60,
                "start_datetime": str(self.start_datetime),
                "earliest_datetime": str(self.earliest_datetime),
                "current_datetime": str(self.current_datetime),
                "flight_action": flight_action,
                "aircraft_action": aircraft_action,
                "action_index": action_index,
                "invalid_action": True,
                "invalid_action_reason": f"map_index_to_action({action_index}) returned (-1, -1)",
                "penalties": {
                    "delay_penalty_total": 0,
                    "cancel_penalty": 0,
                    "inaction_penalty": 0,
                    "automatic_cancellation_penalty": 0,
                    "proactive_penalty": 0,
                    "time_penalty": 0,
                    "unresolved_conflict_penalty": 0
                },
                "penalty_flags": {
                    "penalty_1_delay_enabled": PENALTY_1_DELAY_ENABLED,
                    "penalty_2_cancellation_enabled": PENALTY_2_CANCELLATION_ENABLED,
                    "penalty_3_inaction_enabled": PENALTY_3_INACTION_ENABLED,
                    "penalty_4_proactive_enabled": PENALTY_4_PROACTIVE_ENABLED,
                    "penalty_5_time_enabled": PENALTY_5_TIME_ENABLED,
                    "penalty_6_final_reward_enabled": PENALTY_6_FINAL_REWARD_ENABLED,
                    "penalty_7_auto_cancellation_enabled": PENALTY_7_AUTO_CANCELLATION_ENABLED
                }
            }
            info.update(self.info_after_step)
            
            return processed_state, reward, terminated, truncated, info

        # Check if the flight action is valid
        if DEBUG_MODE_ACTION:
            pass
        
        if flight_action != 0:
            # Check if the flight_action exists in our valid flight IDs
            if flight_action not in self.flight_id_to_idx.keys():
                raise ValueError(f"Invalid flight action: {flight_action}")
            else:
                pass
        else:
            pass

        # Validate the action
        self.validate_action(flight_action, aircraft_action)

        # Print the processed action and chosen action
        if DEBUG_MODE:
            print(f"Processed action: {action_index} of type: {type(action_index)}")
            print(f"Chosen action: flight {flight_action}, aircraft {aircraft_action}")

        # Get pre-action conflicts
        # Gets probabilities ac unavailitbilities that have not yet been resolved (they are neither 1 or 0 but 0<p<1)
        if DEBUG_MODE_REWARD:
            print(f"pre action conflicts are:")
        pre_action_conflicts = self.get_current_conflicts()
        
        # Keep track of which flights were actually in conflict before the action
        self.pre_action_conflict_flights = set()
        for conflict in pre_action_conflicts:
            if isinstance(conflict, (tuple, list)) and len(conflict) >= 2:
                self.pre_action_conflict_flights.add(conflict[1])
        # Snapshot probabilities BEFORE they are advanced, so reward #8 can use the pre-action value
        self.pre_action_probabilities = {
            aircraft_id: self.unavailabilities_dict.get(aircraft_id, {}).get('Probability', np.nan)
            for aircraft_id in self.aircraft_ids
        }
        unresolved_uncertainties = self.get_unresolved_uncertainties()

        # print(f"pre_action_conflicts: {pre_action_conflicts}")

        # Process uncertainties before handling flight operations
        self.process_uncertainties()
        self.scenario_wide_steps += 1

        if len(pre_action_conflicts) == 0 and len(unresolved_uncertainties) == 0:
            # Handle the case when there are no conflicts
            # print("-----2-----")
            processed_state, reward, terminated, truncated, info, stop_scenario = self.handle_no_conflicts(flight_action, aircraft_action)
        else:
            # print("-----3-----")
            # Resolve the conflict based on the action
            processed_state, reward, terminated, truncated, info, stop_scenario = self.handle_flight_operations(flight_action, aircraft_action, pre_action_conflicts)

        # Update the processed state after processing uncertainties
        processed_state, _ = self.process_observation(self.state)

        # Determine termination based on two conditions:
        # 1. Time limit reached → terminate regardless of conflict status
        # 2. No overlaps remain AND all probabilities resolved → terminate (successful completion)
        time_limit_reached = self.current_datetime >= self.end_datetime
        criteria_met = self.check_termination_criteria()
        
        if time_limit_reached:
            terminated = True  # Time's up - end scenario regardless of conflicts
        elif criteria_met:
            terminated = True  # Successfully completed - all resolved
        else:
            terminated = False  # Keep going - either conflicts remain or probabilities uncertain
        
        if DEBUG_MODE_STOPPING_CRITERIA and terminated:            
            # Debug: Show termination status details
            if stop_scenario or time_limit_reached or criteria_met:
                print(f"  [TERMINATION STATUS]")
                print(f"    - stop_scenario: {stop_scenario}")
                print(f"    - time_limit_reached: scenario ended because time limit reached {time_limit_reached}")
                print(f"    - check_termination_criteria(): {criteria_met}")
                print(f"    - final terminated: {terminated}")

        # Merge info_after_step into info dict to pass penalty details through
        if info is None:
            info = {}
        
        # Always store action_index in info_after_step for accurate action tracking
        # track the original action index before flights are removed
        self.info_after_step["action_index"] = action_index
        
        info.update(self.info_after_step)

        return processed_state, reward, terminated, truncated, info


    def check_termination_criteria(self):
        """
        Checks if the stopping criteria are met:

        Stopping criteria:
        1. There are no uncertainties in the system anymore.
           (All probabilities are either np.nan, 0.0, or 1.0.)
        2. There is no overlap of breakdowns (Probability == 1.0) and flights.

        Returns:
            bool: True if both criteria are met, False otherwise.
        """
        # Check that all probabilities are either nan, 0.0, or 1.0
        for aircraft_id in self.aircraft_ids:
            prob = self.unavailabilities_dict[aircraft_id]['Probability']
            if not (np.isnan(prob) or prob == 0.0 or prob == 1.0):
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"    prob: {prob} is not nan, 0.0, or 1.0, for aircraft {aircraft_id}, so termination = False")
                return False

        
        current_conflicts = self.get_current_conflicts_with_prob_1()
        
        if len(current_conflicts) > 0:
            if DEBUG_MODE_STOPPING_CRITERIA:
                print(f"    get_current_conflicts_with_prob_1() returns {current_conflicts}, so termination = False")
            return False
        
        

        return True


    def extract_action_value(self, action):
        """Extracts the flight and aircraft action values from the flattened action.

        Args:
            action (int): The flattened action index.

        Returns:
            tuple: The flight action and aircraft action values.
        """
        if action < 0 or action >= ACTION_SPACE_SIZE:
            raise ValueError("Invalid action index")

        flight_action = action // (len(self.aircraft_ids) + 1)  # Integer division to get flight action
        aircraft_action = action % (len(self.aircraft_ids) + 1)  # Modulus to get aircraft action

        return flight_action, aircraft_action

    # NO need to modify for new state-space formulation
    def validate_action(self, flight_action, aircraft_action): 
        """Validates the provided action values.

        Args:
            flight_action (int): The flight action value to be validated.
            aircraft_action (int): The aircraft action value to be validated.

        Raises:
            ValueError: If the action is not valid.
        """
        # Get valid actions
        valid_flight_actions = self.get_valid_flight_actions()
        valid_aircraft_actions = self.get_valid_aircraft_actions()

        # Check if flight_action is valid
        if flight_action not in valid_flight_actions:
            raise ValueError(f"Invalid flight action: {flight_action}")

        # Check if aircraft_action is valid
        if aircraft_action not in valid_aircraft_actions:
            raise ValueError(f"Invalid aircraft action: {aircraft_action}")

        # No action case
        if flight_action == 0:
            # Treat as 'no action'
            return


    def process_uncertainties(self):
        """Processes breakdown uncertainties directly from the state space.

        Probabilities evolve stochastically over time but are capped at [0.05, 0.95].
        When the current datetime + timestep reaches the breakdown start time,
        resolve the uncertainty fully to 0.00 or 1.00 by rolling the dice.

        The bias term pushes probabilities that are above 0.5 towards 1.0 and probabilities below 0.5 towards 0.0       
        """
        if DEBUG_MODE:
            print(f"Current datetime: {self.current_datetime}")

        # Iterate over each aircraft's row in the state space to check for unresolved breakdowns
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            # Get probability, start, and end time from the state space
            prob = self.unavailabilities_dict[aircraft_id]['Probability']
            start_minutes = self.unavailabilities_dict[aircraft_id]['StartTime']
            end_minutes = self.unavailabilities_dict[aircraft_id]['EndTime']

            # Only process unresolved breakdowns
            if prob != 0.00 and prob != 1.00:
                # Check for valid start and end times
                if not np.isnan(start_minutes) and not np.isnan(end_minutes) and not np.isnan(prob):
                    breakdown_start_time = self.earliest_datetime + timedelta(minutes=start_minutes)
                else:
                    # No start or end time, skip processing
                    continue

                # Apply random progression to probability
                random_variation = np.random.uniform(-0.05, 0.05)  # Random adjustment
                bias = 0.05 * (1 - prob) if prob > 0.5 else -0.05 * prob  # Bias toward extremes
                progression = random_variation + bias
                new_prob = prob + progression

                # Cap probabilities at [0.05, 0.95]
                new_prob = max(0.05, min(0.95, new_prob))
                self.unavailabilities_dict[aircraft_id]['Probability'] = new_prob
                self.alt_aircraft_dict[aircraft_id]['Probability'] = new_prob
                # print(f"*** new_prob: {new_prob}")

                if self.env_type == "proactive":
                    self.state[idx + 1, 1] = new_prob
                    # print(f"*** updated proactive state: {self.state[idx + 1, 1]}")

                if DEBUG_MODE:
                    print(f"Aircraft {aircraft_id}: Probability updated from {prob:.2f} to {new_prob:.2f}")

                if self.current_datetime + self.timestep >= breakdown_start_time:
                    
                    # Roll the dice
                    if np.random.rand() < new_prob:
                        if DEBUG_MODE_BREAKDOWN:
                            print(f"Breakdown confirmed for aircraft {aircraft_id} with probability {new_prob:.2f}")
                        self.state[idx + 1, 1] = 1.00  # Confirm the breakdown
                        self.state[idx + 1, 2] = start_minutes  # Update start time
                        self.state[idx + 1, 3] = end_minutes
                        self.unavailabilities_dict[aircraft_id]['Probability'] = 1.00

                    else:
                        if DEBUG_MODE_BREAKDOWN:
                            print(f"Breakdown not occurring for aircraft {aircraft_id}")
                        
                        # Count affected flights for this aircraft in the initial disrupted flights list
                        affected_flights = 0
                        for conflict in self.scenario_wide_initial_disrupted_flights_list:
                            # Extract just the flight ID and aircraft ID from the conflict tuple
                            if isinstance(conflict, tuple):
                                conflict_aircraft_id = conflict[0]  # First element is aircraft ID
                                if conflict_aircraft_id == aircraft_id:
                                    affected_flights += 1
                        
                        # Subtract the count of affected flights from the total
                        self.scenario_wide_actual_disrupted_flights -= affected_flights

                        # make the prop, start, end all np.nan for both env types
                        self.state[idx + 1, 1] = np.nan
                        self.state[idx + 1, 2] = np.nan
                        self.state[idx + 1, 3] = np.nan
                        self.unavailabilities_dict[aircraft_id]['Probability'] = 0.00

                    # Update alt_aircraft_dict if necessary
                    if aircraft_id in self.alt_aircraft_dict:
                        if isinstance(self.alt_aircraft_dict[aircraft_id], dict):
                            # print("*** alt_aircraft_dict[aircraft_id] is a dict")
                            self.alt_aircraft_dict[aircraft_id] = [self.alt_aircraft_dict[aircraft_id]]
                        elif isinstance(self.alt_aircraft_dict[aircraft_id], str):
                            # print("*** alt_aircraft_dict[aircraft_id] is a string")
                            # Handle case where entry is a string
                            self.alt_aircraft_dict[aircraft_id] = [{
                                'StartDate': breakdown_start_time.strftime('%d/%m/%y'),
                                'StartTime': breakdown_start_time.strftime('%H:%M'),
                                'EndDate': (breakdown_start_time + timedelta(minutes=end_minutes - start_minutes)).strftime('%d/%m/%y'),
                                'EndTime': (breakdown_start_time + timedelta(minutes=end_minutes - start_minutes)).strftime('%H:%M'),
                                'Probability': self.state[idx + 1, 1]  # Updated probability
                            }]
                        for breakdown_info in self.alt_aircraft_dict[aircraft_id]:
                            breakdown_info['Probability'] = self.state[idx + 1, 1]
                            # print(f"*** breakdown_info['Probability']: {breakdown_info['Probability']}")



    def handle_no_conflicts(self, flight_action, aircraft_action):
        """Handles the case when there are no conflicts in the current state.

        This function updates the current datetime, checks if the episode is terminated,
        updates the state, and returns the appropriate outputs.
        """

        if flight_action != 0:
            original_flight_action_departure_time = self.flights_dict[flight_action]['DepTime']
        else:
            original_flight_action_departure_time = None

        next_datetime = self.current_datetime + self.timestep
        if next_datetime >= self.end_datetime:
            terminated, reason = self._is_done()
            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Scenario ended: {reason}")
                processed_state, _ = self.process_observation(self.state)
                truncated = False
                reward = self._calculate_reward(set(), set(), flight_action, aircraft_action, original_flight_action_departure_time, terminated)
                return processed_state, reward, terminated, truncated, {}

        self.current_datetime = next_datetime
        self.state = self._get_initial_state()
        
        stop_scenario, reason = self.end_of_recovery_period_reached()
        terminated, reason = self._is_done()
        truncated = False
        done = terminated or truncated
        reward = self._calculate_reward(set(), set(), flight_action, aircraft_action, original_flight_action_departure_time, done)

        processed_state, _ = self.process_observation(self.state)

        if terminated:
            if DEBUG_MODE_STOPPING_CRITERIA:
                print(f"Scenario ended: {reason}")

        return processed_state, reward, terminated, truncated, {}, stop_scenario


    def handle_flight_operations(self, flight_action, aircraft_action, pre_action_conflicts):
        """
        Handles flight operation decisions and resolves conflicts.

        This method processes the agent's actions to either maintain the current state, cancel a flight, 
        or reschedule it to a different aircraft. It updates the system state accordingly, resolves 
        any conflicts, and computes the rewards based on the chosen action.

        Args:
            flight_action (int): The index of the flight action chosen by the agent. 
                                Use 0 to skip the flight operation.
            aircraft_action (int): The index of the aircraft action chosen by the agent. 
                                Use 0 to cancel the flight.
            pre_action_conflicts (set): The set of conflicts present before the action is taken.

        Returns:
            tuple: A tuple containing:
                - processed_state: The updated system state after the action.
                - reward (float): The reward value calculated based on the resolved conflicts.
                - terminated (bool): Whether the episode has ended.
                - truncated (bool): Whether the episode was prematurely stopped.
                - info (dict): Additional diagnostic information.
        """
        
        # Note: Invalid actions (-1, -1) are already handled in step() before this function is called
        # This ensures: never have invalid actions here, avoiding crashes in handle_no_conflicts()
        # which would try to access self.flights_dict[-1] at line 761

        # store the departure time of the flight that is being acted upon (before the action is taken)
        if flight_action != 0:
            original_flight_action_departure_time = self.flights_dict[flight_action]['DepTime']
        else:
            original_flight_action_departure_time = None

        terminated = self.check_termination_criteria()

        if flight_action == 0:
            # No action taken
            # Proceed to next timestep
            next_datetime = self.current_datetime + self.timestep
            self.current_datetime = next_datetime
            self.state = self._get_initial_state()

            if DEBUG_MODE_REWARD:
                print(f"")
                print(f"post action conflicts (flight action = 0):")
                
            post_action_conflicts = self.get_current_conflicts()
            

            resolved_conflicts = pre_action_conflicts - post_action_conflicts
            
            stop_scenario, reason = self.end_of_recovery_period_reached()
            terminated, reason = self._is_done()
            truncated = False
            done = terminated or truncated
            # print("*** 3")
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, done)

            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            processed_state, _ = self.process_observation(self.state)
            return processed_state, reward, terminated, truncated, {}, stop_scenario
        elif aircraft_action == 0:
            # Cancel the flight
            self.cancel_flight(flight_action)
            if DEBUG_MODE_CANCELLED_FLIGHT:
                print(f"Cancelled flight {flight_action}")

            # Proceed to next timestep
            next_datetime = self.current_datetime + self.timestep
            self.current_datetime = next_datetime
            self.state = self._get_initial_state()

            if DEBUG_MODE_REWARD:
                print(f"post action conflicts (aircraft action = 0): ")
            post_action_conflicts = self.get_current_conflicts()
            resolved_conflicts = pre_action_conflicts - post_action_conflicts

            stop_scenario, reason = self.end_of_recovery_period_reached()
            terminated, reason = self._is_done()
            truncated = False
            done = terminated or truncated
            # print("*** 4")
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, done)

            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            processed_state, _ = self.process_observation(self.state)
            return processed_state, reward, terminated, truncated, {}, stop_scenario
        else:
            # Reschedule the flight to the selected aircraft
            selected_flight_id = flight_action
            selected_aircraft_id = self.aircraft_ids[aircraft_action - 1]

            # Check if the flight is in rotations_dict
            if selected_flight_id not in self.rotations_dict:
                # Flight has been canceled or does not exist
                print(f"Flight {selected_flight_id} has been canceled or does not exist.")
                
                # Proceed to next timestep
                next_datetime = self.current_datetime + self.timestep
                self.current_datetime = next_datetime
                self.state = self._get_initial_state()
                
                stop_scenario, reason = self.end_of_recovery_period_reached()
                terminated, reason = self._is_done()
                truncated = False

                done = terminated or truncated
                # print("*** 5")
                reward = self._calculate_reward(pre_action_conflicts, pre_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, done)

                processed_state, _ = self.process_observation(self.state)
                return processed_state, reward, terminated, truncated, {}, stop_scenario

            current_aircraft_id = self.rotations_dict[selected_flight_id]['Aircraft']

            if selected_aircraft_id == current_aircraft_id:
                # Delay the flight by scheduling it on the same aircraft
                # Get unavailability end time for the aircraft
                aircraft_idx = self.aircraft_id_to_idx[current_aircraft_id] + 1  # Adjust for state index
                unavail_end = self.state[aircraft_idx, 3]  # Unavailability end time in minutes from earliest_datetime

                if np.isnan(unavail_end):
                    # No unavailability end time, cannot proceed
                    # In this case, set unavail_end to current time
                    unavail_end = (self.current_datetime - self.earliest_datetime).total_seconds() / 60

                unavail_end_datetime = self.earliest_datetime + timedelta(minutes=unavail_end)
                new_dep_time = unavail_end_datetime + timedelta(minutes=MIN_TURN_TIME)
                new_dep_time_minutes = (new_dep_time - self.earliest_datetime).total_seconds() / 60

                # Schedule the flight on the same aircraft starting from new_dep_time_minutes
                # The schedule_flight_on_aircraft method will handle adjusting subsequent flights
                self.schedule_flight_on_aircraft(
                    selected_aircraft_id, selected_flight_id, new_dep_time_minutes, current_aircraft_id, None
                )

            else:
                # Swap the flight to the selected aircraft
                # Update rotations_dict
                self.rotations_dict[selected_flight_id]['Aircraft'] = selected_aircraft_id
                self.scenario_wide_tail_swaps += 1  # Increment counter when flight is moved to new aircraft
                
                # Track the swap for visualization
                self.swapped_flights.append((selected_flight_id, selected_aircraft_id))

                # Remove flight from current aircraft's schedule
                current_aircraft_idx = self.aircraft_id_to_idx[current_aircraft_id] + 1
                for j in range(4, self.columns_state_space - 2, 3):
                    if self.state[current_aircraft_idx, j] == selected_flight_id:
                        self.state[current_aircraft_idx, j] = np.nan
                        self.state[current_aircraft_idx, j + 1] = np.nan
                        self.state[current_aircraft_idx, j + 2] = np.nan
                        break

                # Schedule flight on new aircraft
                # Get dep and arr times
                flight_info = self.flights_dict[selected_flight_id]
                dep_time_str = flight_info['DepTime']
                arr_time_str = flight_info['ArrTime']
                dep_time = parse_time_with_day_offset(dep_time_str, self.start_datetime)
                arr_time = parse_time_with_day_offset(arr_time_str, self.start_datetime)
                dep_time_minutes = (dep_time - self.earliest_datetime).total_seconds() / 60
                arr_time_minutes = (arr_time - self.earliest_datetime).total_seconds() / 60

                self.schedule_flight_on_aircraft(selected_aircraft_id, selected_flight_id, dep_time_minutes, current_aircraft_id, arr_time_minutes)

            # Proceed to next timestep
            next_datetime = self.current_datetime + self.timestep
            self.current_datetime = next_datetime
            self.state = self._get_initial_state()

            if DEBUG_MODE_REWARD:
                print(f"post action conflicts (swap flight):")
            post_action_conflicts = self.get_current_conflicts()
            
            resolved_conflicts = pre_action_conflicts - post_action_conflicts

            stop_scenario, reason = self.end_of_recovery_period_reached()
            terminated, reason = self._is_done()
            terminated = self.check_termination_criteria()
            truncated = False
            done = terminated or truncated
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, terminated=done)

            if terminated:
                # print("*** 6 - but with terminated = ", terminated)
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            processed_state, _ = self.process_observation(self.state)
            # print("*** returning: ", processed_state, reward, terminated, truncated, {})
            return processed_state, reward, terminated, truncated, {}, stop_scenario

    def schedule_flight_on_aircraft(self, aircraft_id, flight_id, dep_time, current_aircraft_id, arr_time=None, delayed_flights=None, secondary=False):
        """Schedules a flight on an aircraft.

        This function schedules a flight on an aircraft, taking into account unavailability periods and conflicts with existing flights.
        It updates the state and flights dictionary accordingly.
        
        Args:
            aircraft_id (str): The ID of the aircraft to schedule the flight on.
            flight_id (str): The ID of the flight to schedule.
            dep_time (float): The departure time of the flight in minutes from earliest_datetime.
            current_aircraft_id (str): The ID of the current aircraft.
            arr_time (float, optional): The arrival time of the flight in minutes from earliest_datetime. Defaults to None.
            delayed_flights (set, optional): A set of flight IDs that have already been delayed. Defaults to None.
        """
        if DEBUG_MODE_SCHEDULING:
            print("\n=== Starting schedule_flight_on_aircraft ===")
            print(f"Scheduling flight {flight_id} on aircraft {aircraft_id}")
            print(f"Initial dep_time: {dep_time}, arr_time: {arr_time}")

        if delayed_flights is None:
            delayed_flights = set()
        
        aircraft_idx = self.aircraft_id_to_idx[aircraft_id] + 1  # Adjust for state indexing

        # Get the original flight times and duration
        original_dep_time = parse_time_with_day_offset(
            self.flights_dict[flight_id]['DepTime'], self.start_datetime
        )
        original_arr_time = parse_time_with_day_offset(
            self.flights_dict[flight_id]['ArrTime'], self.start_datetime
        )
        original_dep_minutes = (original_dep_time - self.earliest_datetime).total_seconds() / 60
        flight_duration = (original_arr_time - original_dep_time).total_seconds() / 60
        original_arr_minutes = (original_arr_time - self.earliest_datetime).total_seconds() / 60

        if DEBUG_MODE_SCHEDULING:
            print(f"Original departure minutes: {original_dep_minutes}")
            print(f"Flight duration: {flight_duration}")

        # Ensure dep_time is not earlier than original departure time
        dep_time = max(dep_time, original_dep_minutes)

        if arr_time is None:
            arr_time = dep_time + flight_duration
        else:
            flight_duration = arr_time - dep_time

        # Check for unavailability conflicts
        # Read from state matrix (which has been adjusted by fix_state) rather than unavailabilities_dict
        aircraft_idx = self.aircraft_id_to_idx[aircraft_id] + 1
        unavail_start = self.state[aircraft_idx, 2]
        unavail_end = self.state[aircraft_idx, 3]
        unavail_prob = self.state[aircraft_idx, 1]

        if DEBUG_MODE_SCHEDULING:
            print(f"\nUnavailability check:")
            print(f"Current aircraft: {current_aircraft_id}, Target aircraft: {aircraft_id}")
            print(f"Unavailability - Start: {unavail_start}, End: {unavail_end}, Prob: {unavail_prob}")

        # Check if flight overlaps with unavailability
        has_unavail_overlap = False
        if (not np.isnan(unavail_start) and 
            not np.isnan(unavail_end) and 
            unavail_prob > 0.0):  # Only check for overlap if there's an actual unavailability
            
            # Convert times to ensure proper comparison
            flight_start = float(original_dep_minutes)
            flight_end = float(original_arr_minutes)
            unavail_start_time = float(unavail_start)
            unavail_end_time = float(unavail_end)
            
            # Check for any overlap between flight and unavailability period
            # A flight overlaps if it doesn't end before the disruption starts
            if flight_end > unavail_start_time:
                has_unavail_overlap = True
                
            if DEBUG_MODE_SCHEDULING:
                print(f"\nChecking overlap:")
                print(f"Flight: {flight_start} -> {flight_end}")
                print(f"Unavail: {unavail_start_time} -> {unavail_end_time}")
                print(f"Overlap detected: {has_unavail_overlap}")

        current_ac_is_same_as_target_ac = aircraft_id == current_aircraft_id
        if not current_ac_is_same_as_target_ac:
            self.something_happened = True
            self.tail_swap_happened = True  # Track that a tail swap occurred
            # Track the swap for visualization (same as in handle_no_conflicts)
            self.swapped_flights.append((flight_id, aircraft_id))

        if DEBUG_MODE_SCHEDULING:
            print(f"current_ac_is_same_as_target_ac: {current_ac_is_same_as_target_ac}") 

        # If flight is on current aircraft and completes before disruption, keep it there
        if current_ac_is_same_as_target_ac and not has_unavail_overlap:
            if DEBUG_MODE_SCHEDULING:
                print("Flight is on current aircraft and completes before disruption - keeping original schedule")
            self.something_happened = False
            return

        # Handle unavailability overlap
        if has_unavail_overlap:
            if DEBUG_MODE_SCHEDULING:
                print("\nFlight overlaps with unavailability period!")
                print(f"Flight times - Dep: {dep_time}, Arr: {arr_time}")
                print(f"Unavail times - Start: {unavail_start}, End: {unavail_end}")

            if aircraft_id == current_aircraft_id:
                if unavail_prob > 0.00: #move flight after unavailability
                    if DEBUG_MODE_SCHEDULING:
                        print("Case 1: Current aircraft with prob > 0.00 - Moving flight after unavailability")
                    dep_time = max(dep_time, unavail_end + MIN_TURN_TIME)
                    dep_time = max(dep_time, original_dep_minutes)
                    arr_time = dep_time + flight_duration
                    delay = dep_time - original_dep_minutes
                    
                    # Sanity check: delay should be reasonable (not more than a few days) LOLL
                    max_reasonable_delay_minutes = 1 * 24 * 60  # 1 day
                    if delay > max_reasonable_delay_minutes:
                        if DEBUG_MODE_SCHEDULING or DEBUG_MODE_REWARD:
                            print(f"  WARNING: Delay ({delay:.1f} minutes = {delay/60:.1f} hours) is unreasonably large for flight {flight_id}!")
                            print(f"    original_dep_minutes: {original_dep_minutes:.1f}")
                            print(f"    dep_time: {dep_time:.1f}")
                            print(f"    unavail_end: {unavail_end:.1f}")
                        delay = max_reasonable_delay_minutes
                    
                    # Store delay as absolute delay from original departure time (don't accumulate)
                    # If flight is moved multiple times, this always calculates from original
                    if delay > 0:
                        # self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + delay LOLL
                        self.environment_delayed_flights[flight_id] = delay
                    self.something_happened = True
                else: #unavail prob = 0.00, keep original schedule
                    if DEBUG_MODE_SCHEDULING:
                        print("Case 2: Current aircraft with prob = 0.00 - Keeping original schedule")
                    self.something_happened = False
            else: #aircraft_id != current_aircraft_id
                if unavail_prob == 1.00: #move flight after unavailability
                    if DEBUG_MODE_SCHEDULING:
                        print("Case 3: Different aircraft with prob = 1.00 - Moving flight after unavailability")
                    dep_time = max(dep_time, unavail_end + MIN_TURN_TIME)
                    dep_time = max(dep_time, original_dep_minutes)
                    arr_time = dep_time + flight_duration
                    delay = dep_time - original_dep_minutes
                    
                    # Sanity check: delay should be reasonable (not more than a few days) LOLL
                    max_reasonable_delay_minutes = 1 * 24 * 60  # 7 days
                    if delay > max_reasonable_delay_minutes:
                        if DEBUG_MODE_SCHEDULING:
                            print(f"  WARNING: Delay ({delay:.1f} minutes = {delay/60:.1f} hours) is unreasonably large for flight {flight_id}!")
                            print(f"    original_dep_minutes: {original_dep_minutes:.1f}")
                            print(f"    dep_time: {dep_time:.1f}")
                            print(f"    unavail_end: {unavail_end:.1f}")
                        delay = max_reasonable_delay_minutes
                    
                    # Store delay as absolute delay from original departure time (don't accumulate)
                    # If flight is moved multiple times, this always calculates from original
                    if delay > 0:
                        # self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + delay LOLL
                        self.environment_delayed_flights[flight_id] = delay
                    self.something_happened = True
                else: #unavail prob < 1.00, allow overlap
                    if DEBUG_MODE_SCHEDULING:
                        print("Case 4: Different aircraft with prob < 1.00 - Allowing overlap")
                    self.something_happened = True

        # Get all flights on this aircraft sorted by departure time
        scheduled_flights = []
        for j in range(4, self.columns_state_space - 2, 3):
            existing_flight_id = self.state[aircraft_idx, j]
            existing_dep_time = self.state[aircraft_idx, j + 1]
            existing_arr_time = self.state[aircraft_idx, j + 2]
            if not np.isnan(existing_flight_id) and not np.isnan(existing_dep_time) and not np.isnan(existing_arr_time):
                if existing_flight_id != flight_id:  # Don't include the flight we're scheduling
                    scheduled_flights.append((existing_flight_id, existing_dep_time, existing_arr_time))
        
        # Sort flights by departure time
        scheduled_flights.sort(key=lambda x: x[1])

        # Find optimal placement to minimize total delays
        optimal_dep_time, optimal_arr_time = self._find_optimal_placement(
            flight_id, dep_time, arr_time, scheduled_flights, original_dep_minutes
        )
        
        # Update our flight times with optimal placement
        dep_time = optimal_dep_time
        arr_time = optimal_arr_time

        # Insert our flight at the optimal position
        insert_idx = 0
        for i, (_, existing_dep_time, _) in enumerate(scheduled_flights):
            if dep_time < existing_dep_time:
                break
            insert_idx = i + 1

        scheduled_flights.insert(insert_idx, (flight_id, dep_time, arr_time))

        # Now process all flights in sequence to ensure proper spacing with minimal cascading
        self._optimize_schedule_with_minimal_cascading(scheduled_flights, aircraft_idx)

        #extract the new dep_time and arr_time for the flight_id of interest
        for flight_tuple in scheduled_flights:
            if flight_tuple[0] == flight_id:
                dep_time = flight_tuple[1]  # Use updated time
                arr_time = flight_tuple[2]  # Use updated time
                break

        # Finally, update the state for our flight
        flight_placed = False
        for j in range(4, self.columns_state_space - 2, 3):
            if self.state[aircraft_idx, j] == flight_id:
                self.state[aircraft_idx, j + 1] = dep_time
                self.state[aircraft_idx, j + 2] = arr_time
                flight_placed = True
                break
            elif np.isnan(self.state[aircraft_idx, j]) and not flight_placed:
                self.state[aircraft_idx, j] = flight_id
                self.state[aircraft_idx, j + 1] = dep_time
                self.state[aircraft_idx, j + 2] = arr_time
                flight_placed = True
                break

        # Update flights_dict
        self.update_flight_times(flight_id, dep_time, arr_time)

        if DEBUG_MODE_SCHEDULING:
            print(f"Final departure time for flight {flight_id}: {dep_time} minutes.")
            print(f"Final arrival time for flight {flight_id}: {arr_time} minutes.")

    def _find_optimal_placement(self, flight_id, dep_time, arr_time, scheduled_flights, original_dep_minutes):
        """
        Finds the optimal placement for a flight to minimize total delays.
        
        Args:
            flight_id (int): The flight to place
            dep_time (float): Initial departure time
            arr_time (float): Initial arrival time
            scheduled_flights (list): List of existing flights on the aircraft
            original_dep_minutes (float): Original departure time
            
        Returns:
            tuple: (optimal_dep_time, optimal_arr_time)
        """
        if not scheduled_flights:
            return dep_time, arr_time
            
        flight_duration = arr_time - dep_time
        min_total_delay = float('inf')
        optimal_dep = dep_time
        optimal_arr = arr_time
        
        # Try placing the flight at different positions
        for insert_pos in range(len(scheduled_flights) + 1):
            # Create a copy of scheduled flights for testing
            test_schedule = scheduled_flights.copy()
            test_schedule.insert(insert_pos, (flight_id, dep_time, arr_time))
            
            # Calculate total delay for this placement
            total_delay = 0
            current_time = 0
            
            for i, (test_flight_id, test_dep, test_arr) in enumerate(test_schedule):
                # Check for overlap with previous flight (no turn time needed)
                if test_dep < current_time:
                    new_dep = current_time
                    new_arr = new_dep + (test_arr - test_dep)
                    
                    # Calculate delay for this flight
                    if test_flight_id == flight_id:
                        delay = new_dep - original_dep_minutes
                    else:
                        # Get original time for other flights
                        orig_dep = parse_time_with_day_offset(
                            self.flights_dict[test_flight_id]['DepTime'], 
                            self.start_datetime
                        )
                        orig_dep_minutes = (orig_dep - self.earliest_datetime).total_seconds() / 60
                        delay = new_dep - orig_dep_minutes
                    
                    total_delay += max(0, delay)
                    current_time = new_arr
                else:
                    current_time = test_arr
                    
            # Update optimal placement if this has less total delay
            if total_delay < min_total_delay:
                min_total_delay = total_delay
                optimal_dep = dep_time
                optimal_arr = arr_time
                
        return optimal_dep, optimal_arr
    
    def _optimize_schedule_with_minimal_cascading(self, scheduled_flights, aircraft_idx):
        """
        Optimizes the schedule to minimize cascading delays.
        
        Args:
            scheduled_flights (list): List of flights to optimize
            aircraft_idx (int): Aircraft index in state
        """
        current_time = 0
        
        for i, (flight_id, dep_time, arr_time) in enumerate(scheduled_flights):
            # Check for overlap with previous flight (no turn time needed)
            if dep_time < current_time:
                new_dep_time = current_time
                new_arr_time = new_dep_time + (arr_time - dep_time)
                
                # Update the flight times in the list
                scheduled_flights[i] = (flight_id, new_dep_time, new_arr_time)
                
                # Update the flight times in the state
                for k in range(4, self.columns_state_space - 2, 3):
                    if self.state[aircraft_idx, k] == flight_id:
                        self.state[aircraft_idx, k + 1] = new_dep_time
                        self.state[aircraft_idx, k + 2] = new_arr_time
                        break
                
                # Update the flights_dict
                self.update_flight_times(flight_id, new_dep_time, new_arr_time)
                
                # Track the delay (only if there's an actual delay)
                original_dep = parse_time_with_day_offset(
                    self.flights_dict[flight_id]['DepTime'], 
                    self.start_datetime
                )
                original_dep_minutes = (original_dep - self.earliest_datetime).total_seconds() / 60
                delay = new_dep_time - original_dep_minutes
                
                # Sanity check: delay should be reasonable (not more than a few days) LOLL
                max_reasonable_delay_minutes = 1 * 24 * 60  # 7 days
                if delay > max_reasonable_delay_minutes:
                    if DEBUG_MODE_SCHEDULING:
                        print(f"  WARNING: Delay ({delay:.1f} minutes = {delay/60:.1f} hours) is unreasonably large!")
                        print(f"    flight_id: {flight_id}")
                        print(f"    original_dep_minutes: {original_dep_minutes:.1f}")
                        print(f"    new_dep_time: {new_dep_time:.1f}")
                    # Cap the delay to prevent massive penalties
                    delay = max_reasonable_delay_minutes
                
                # Only add to delayed flights if there's an actual delay
                if delay > 0:
                    self.environment_delayed_flights[flight_id] = delay
                
                current_time = new_arr_time
            else:
                current_time = arr_time
        
    def cancel_flight(self, flight_id):
        """Cancels the specified flight.

        This function removes the flight from the rotations dictionary, the flights dictionary, and the state.
        It also marks the flight as cancelled and removes it from the state.
        
        Args:
            flight_id (str): The ID of the flight to cancel.
        """
        # Remove the flight from rotations_dict
        if flight_id in self.rotations_dict:
            del self.rotations_dict[flight_id]

        # Remove the flight from flights_dict
        if flight_id in self.flights_dict:
            del self.flights_dict[flight_id]

        # Mark the flight as cancelled
        self.cancelled_flights.add(flight_id)

        # Remove the flight from the state
        for idx in range(1, self.rows_state_space):
            for j in range(4, self.columns_state_space - 2, 3):
                existing_flight_id = self.state[idx, j]
                if existing_flight_id == flight_id:
                    # Remove flight from state
                    self.state[idx, j] = np.nan
                    self.state[idx, j + 1] = np.nan
                    self.state[idx, j + 2] = np.nan

        self.something_happened = True


    def update_flight_times(self, flight_id, dep_time_minutes, arr_time_minutes):
        """Updates the flight times in the flights dictionary.

        This function converts the departure and arrival times from minutes to datetime format and updates the
        corresponding entries in the flights dictionary.

        Args:
            flight_id (str): The ID of the flight to update.
            dep_time_minutes (float): The new departure time in minutes.
            arr_time_minutes (float): The new arrival time in minutes.
        """
        # Convert minutes to datetime
        dep_time = self.earliest_datetime + timedelta(minutes=dep_time_minutes)
        arr_time = self.earliest_datetime + timedelta(minutes=arr_time_minutes)

        # Update flights_dict with new dates and times
        if DEBUG_MODE:
            print("Updating flight times for flight", flight_id)
            print(" - previous times:", self.flights_dict[flight_id]['DepTime'], self.flights_dict[flight_id]['ArrTime'])
            print(" - new times:", dep_time.strftime('%H:%M'), arr_time.strftime('%H:%M'))
        
        self.flights_dict[flight_id]['DepDate'] = dep_time.strftime('%d/%m/%y')
        self.flights_dict[flight_id]['DepTime'] = dep_time.strftime('%H:%M')
        self.flights_dict[flight_id]['ArrDate'] = arr_time.strftime('%d/%m/%y')
        self.flights_dict[flight_id]['ArrTime'] = arr_time.strftime('%H:%M')

    def _flight_has_conflict(self, flight_id):
        """Check if a specific flight has any conflicts with aircraft unavailability periods."""
        if flight_id not in self.flights_dict:
            return False
        
        flight_info = self.flights_dict[flight_id]
        aircraft_id = self.rotations_dict[flight_id]['Aircraft']
        
        # Check if this aircraft has any unavailability periods
        if aircraft_id not in self.unavailabilities_dict:
            return False
        
        unavail_info = self.unavailabilities_dict[aircraft_id]
        unavail_start = unavail_info['StartTime']
        unavail_end = unavail_info['EndTime']
        
        if np.isnan(unavail_start) or np.isnan(unavail_end):
            return False
        
        # Get flight times
        dep_time = parse_time_with_day_offset(flight_info['DepTime'], self.start_datetime)
        arr_time = parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime)
        
        dep_time_minutes = (dep_time - self.earliest_datetime).total_seconds() / 60
        arr_time_minutes = (arr_time - self.earliest_datetime).total_seconds() / 60
        
        # Check for overlap
        overlap_start = max(dep_time_minutes, unavail_start)
        overlap_end = min(arr_time_minutes, unavail_end)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        return overlap_duration > 0

    def _calculate_reward(self, resolved_conflicts, remaining_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, terminated):
        """Calculates the reward based on the current state of the environment.

        The reward consists of several components:
        1. Delay Penalty: Penalty for each minute of delay introduced
        2. Cancellation Penalty: Penalty for each newly cancelled flight
        3. Inaction Penalty: Penalty for taking no action when conflicts exist
        4. Proactive Bonus: Reward for taking actions well before flight departure
        5. Time Penalty: Small penalty for each minute of simulation time
        7. automatic cancellation penalty: penalty for a flight that was cancelled due to time passing its departure time
        6. Final Resolution Reward: Bonus for resolving real conflicts at scenario end
        8. Probability Resolution Bonus: Bonus for resolving conflicts with any probability above 0.00. Given immediately
        9. Low Confidence Action Penalty: Penalty for taking actions on low-confidence disruptions
        
        Args:
            resolved_conflicts (set): The set of conflicts that were resolved during the action
            remaining_conflicts (set): The set of conflicts that remain after the action
            flight_action (int): The flight action taken by the agent
            aircraft_action (int): The aircraft action taken by the agent
            original_flight_action_departure_time (str): The departure time of the flight being acted upon
            terminated (bool): Whether the scenario has ended

        Returns:
            float: The calculated reward for the action
        """
        reward = 0

        if DEBUG_MODE_REWARD:
            print("")
            print(f"Calculating reward for action: flight {flight_action}, aircraft {aircraft_action}")

        # 1. Delay Penalty: Penalize additional minutes of delay
        # NOTE: Delays solve conflicts, so they should be minimally penalized (or not at all)
        # Only penalize delays that exceed a threshold (e.g., >3 hours) to slightly discourage very long delays
        delay_penalty_minutes = sum(
            self.environment_delayed_flights[flight_id] - self.penalized_delays.get(flight_id, 0)
            for flight_id in self.environment_delayed_flights
        )
        
        # Track number of NEW delays (not cumulative delay minutes)
        new_delayed_flights = len([
            flight_id for flight_id in self.environment_delayed_flights
            if flight_id not in self.penalized_delays or self.environment_delayed_flights[flight_id] > self.penalized_delays.get(flight_id, 0)
        ])
        
        self.scenario_wide_delay_minutes += delay_penalty_minutes
        self.scenario_wide_delay_count += new_delayed_flights  # Track count of delayed flights
        
        if PENALTY_1_DELAY_ENABLED:
            delay_minutes_above_threshold = max(0, delay_penalty_minutes - DELAY_PENALTY_THRESHOLD_MINUTES)
            delay_penalty_total = min(delay_minutes_above_threshold * DELAY_MINUTE_PENALTY, MAX_DELAY_PENALTY)
        else:
            delay_penalty_total = 0

        if DEBUG_MODE_REWARD:
            status = "ENABLED" if PENALTY_1_DELAY_ENABLED else "DISABLED"
            if PENALTY_1_DELAY_ENABLED and delay_penalty_minutes > DELAY_PENALTY_THRESHOLD_MINUTES:
                print(f"  [Penalty #1: {status}] -{delay_penalty_total} penalty for {delay_minutes_above_threshold} minutes of delay above {DELAY_PENALTY_THRESHOLD_MINUTES}min threshold (total delay: {delay_penalty_minutes}min)")
            elif PENALTY_1_DELAY_ENABLED:
                print(f"  [Penalty #1: {status}] No delay penalty (delay {delay_penalty_minutes}min is below {DELAY_PENALTY_THRESHOLD_MINUTES}min threshold - delays solve conflicts!)")
            else:
                print(f"  [Penalty #1: {status}] Delay penalty disabled (delays solve conflicts, so no penalty)")

        # 2. Cancellation Penalty: Penalize newly cancelled flights
        # Always track cancellations, regardless of penalty flag
        new_cancellations = {
            flight_id for flight_id in self.cancelled_flights if flight_id not in self.penalized_cancelled_flights
        }
        cancellation_penalty_count = len(new_cancellations)
        
        # Always track the count
        self.scenario_wide_cancelled_flights += cancellation_penalty_count
        
        # Only apply penalty if enabled
        if PENALTY_2_CANCELLATION_ENABLED:
            cancel_penalty = cancellation_penalty_count * CANCELLED_FLIGHT_PENALTY
            self.penalized_cancelled_flights.update(new_cancellations)
        else:
            cancel_penalty = 0
            # Still update penalized set to avoid double-counting, even if penalty is disabled
            self.penalized_cancelled_flights.update(new_cancellations)

        if DEBUG_MODE_REWARD:
            status = "ENABLED" if PENALTY_2_CANCELLATION_ENABLED else "DISABLED"
            print(f"  [Penalty #2: {status}] -{cancel_penalty} penalty for {cancellation_penalty_count} new cancelled flights: {new_cancellations}")

        # 3. Inaction Penalty: Penalize doing nothing when conflicts exist
        # Always track inaction, regardless of penalty flag
        if flight_action == 0:
            self.scenario_wide_inaction_count += 1  # Track inaction occurrences
        
        # Only apply penalty if enabled
        if PENALTY_3_INACTION_ENABLED:
            if flight_action == 0 and remaining_conflicts:
                inaction_penalty = NO_ACTION_PENALTY  
            elif flight_action == 0 and not remaining_conflicts:
                inaction_penalty = NO_ACTION_PENALTY/2  
            else:
                inaction_penalty = 0
        else:
            inaction_penalty = 0

        if DEBUG_MODE_REWARD:
            status = "ENABLED" if PENALTY_3_INACTION_ENABLED else "DISABLED"
            if inaction_penalty > 0:
                print(f"  [Penalty #3: {status}] -{inaction_penalty} penalty for inaction (0,0) with {len(remaining_conflicts)} remaining conflicts")
            elif flight_action == 0 and not remaining_conflicts:
                print(f"  [Penalty #3: {status}] No inaction penalty (no conflicts remain - waiting for probabilities to resolve is fine)")
            else:
                print(f"  [Penalty #3: {status}] No inaction penalty (action was not 0,0)")

        # 4. Proactive Penalty: Fixed penalty for acting too close to departure
        proactive_penalty = 0
        time_to_departure = None
        if PENALTY_4_PROACTIVE_ENABLED and flight_action != 0 and self.something_happened:
            original_dep_time = parse_time_with_day_offset(original_flight_action_departure_time, self.start_datetime)
            current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60
            action_time = current_time_minutes - TIMESTEP_HOURS * 60
            time_to_departure = (original_dep_time - self.earliest_datetime).total_seconds() / 60 - action_time
            
            # Fixed penalty if action is taken less than 60 minutes before departure
            if time_to_departure < 60:
                proactive_penalty = AHEAD_PENALTY  # Fixed penalty for last-minute actions

        if DEBUG_MODE_REWARD:
            status = "ENABLED" if PENALTY_4_PROACTIVE_ENABLED else "DISABLED"
            if proactive_penalty > 0:
                print(f"  [Penalty #4: {status}] -{proactive_penalty} penalty for last-minute action ({time_to_departure:.1f} minutes before departure)")

        # 5. Time Penalty: Small penalty per timestep to encourage faster resolution
        if PENALTY_5_TIME_ENABLED:
            if remaining_conflicts:
                time_penalty = TIMESTEP_HOURS * 60 * TIME_MINUTE_PENALTY  # Penalty per timestep
            else:
                time_penalty = 0  # No penalty if no conflicts remain (waiting is fine)
        else:
            time_penalty = 0

        if DEBUG_MODE_REWARD:
            status = "ENABLED" if PENALTY_5_TIME_ENABLED else "DISABLED"
            if time_penalty > 0:
                print(f"  [Penalty #5: {status}] Time penalty calculation:")
                print(f"    Penalty per timestep: {TIMESTEP_HOURS * 60} minutes * {TIME_MINUTE_PENALTY} = {time_penalty:.4f}")
                print(f"  -{time_penalty} penalty for this timestep ({len(remaining_conflicts)} conflicts remain)")
            elif not remaining_conflicts:
                print(f"  [Penalty #5: {status}] No time penalty (no conflicts remain - waiting for probabilities to resolve is fine)")
            else:
                print(f"  [Penalty #5: {status}] Time penalty disabled")

        # 7. Automatic cancellation of flights that have already departed
        # Always track automatic cancellations, regardless of penalty flag
        new_automatic_cancellations = {
            flight_id for flight_id in self.automatically_cancelled_flights if flight_id not in self.penalized_automatically_cancelled_flights
        }
        automatic_cancellation_penalty_count = len(new_automatic_cancellations)
        
        # Always track the count
        self.scenario_wide_automatically_cancelled_count += automatic_cancellation_penalty_count  # Track auto-cancellations separately
        
        # Only apply penalty if enabled
        if PENALTY_7_AUTO_CANCELLATION_ENABLED:
            automatic_cancellation_penalty = automatic_cancellation_penalty_count * AUTOMATIC_CANCELLATION_PENALTY
            self.penalized_automatically_cancelled_flights.update(new_automatic_cancellations)
        else:
            automatic_cancellation_penalty = 0
            # Still update penalized set to avoid double-counting, even if penalty is disabled
            self.penalized_automatically_cancelled_flights.update(new_automatic_cancellations)
        
        if DEBUG_MODE_REWARD and automatic_cancellation_penalty_count > 0:
            status = "ENABLED" if PENALTY_7_AUTO_CANCELLATION_ENABLED else "DISABLED"
            print(f"  [Penalty #7: {status}] -{automatic_cancellation_penalty} penalty for {automatic_cancellation_penalty_count} automatic cancellations")

        # 6. Episode-End Penalty:for each initial conflict, check if it was properly resolved (prob=1.00, not cancelled)
        # If NOT properly resolved, apply penalty
        unresolved_conflict_penalty = 0
        scenario_ended_flag = False  # Track if scenario actually ended (for step_info)
        
        # Always calculate metrics when scenario terminates, regardless of penalty #6 being enabled
        # This ensures metrics are saved even when penalty #6 is disabled (e.g., Model 1 debug mode)
        if terminated:
            scenario_ended = self.check_termination_criteria()
            if scenario_ended:
                scenario_ended_flag = True  # Mark that scenario ended (for metrics saving)
                unresolved_count = 0
                resolved_count = 0
                disruption_resolved_to_zero_count = 0
                unresolved_flights = []
                resolved_flights = []
                
                for (aircraft_id, flight_id) in self.initial_conflict_combinations:
                    # Properly resolved = prob=1.00 AND not cancelled AND not auto-cancelled
                    if (self.unavailabilities_dict[aircraft_id]['Probability'] == 1.00 
                        and flight_id not in self.cancelled_flights 
                        and flight_id not in self.automatically_cancelled_flights):
                        resolved_count += 1
                        resolved_flights.append(flight_id)
                    elif (self.unavailabilities_dict[aircraft_id]['Probability'] == 0.00 
                          or self.unavailabilities_dict[aircraft_id]['Probability'] is None
                          or (isinstance(self.unavailabilities_dict[aircraft_id]['Probability'], (int, float)) 
                              and np.isnan(self.unavailabilities_dict[aircraft_id]['Probability']))):
                        disruption_resolved_to_zero_count += 1
                    else:
                        # Not properly resolved = prob=1.00 BUT cancelled or auto-cancelled (instead of swapped)
                        # Note: check_termination_criteria ensures all probabilities are resolved (0 or 1) and no conflicts exist
                        unresolved_count += 1
                        unresolved_flights.append(flight_id)

                # Always update metrics when scenario ends
                self.scenario_wide_resolved_initial_conflicts += resolved_count
                self.scenario_wide_disruption_resolved_to_zero_count += disruption_resolved_to_zero_count

                # Calculate scenario-wide solution slack
                self._calculate_scenario_wide_solution_slack()
                
                # Only apply penalty if penalty #6 is enabled
                if PENALTY_6_FINAL_REWARD_ENABLED:
                    unresolved_conflict_penalty = unresolved_count * UNRESOLVED_CONFLICT_PENALTY
                
                if DEBUG_MODE_REWARD:
                    status = "ENABLED" if PENALTY_6_FINAL_REWARD_ENABLED else "DISABLED"
                    print(f"  [Penalty #6: {status}] Scenario ended - Conflict Resolution Summary:")
                    print(f"    Properly resolved: {resolved_count} conflicts {resolved_flights}")
                    print(f"    Unresolved/Cancelled: {unresolved_count} conflicts {unresolved_flights}")
                    print(f"    Auto-cancelled flights at this point: {self.automatically_cancelled_flights}")
                    print(f"    Cancelled flights at this point: {self.cancelled_flights}")
                    if PENALTY_6_FINAL_REWARD_ENABLED:
                        print(f"    -{unresolved_conflict_penalty:.2f} penalty for unresolved conflicts ({unresolved_count} * {UNRESOLVED_CONFLICT_PENALTY})")
            elif DEBUG_MODE_REWARD:
                print(f"  [Penalty #6] Episode-end penalty NOT applied because scenario_ended=False (termination criteria not met)")
            
        # 8. Probability-aware shaping: reward resolving high-probability conflicts
        # Only give bonus if agent took an action that resolved conflicts (not if conflicts disappeared naturally)
        probability_resolution_bonus = 0
        resolved_probability_total = 0
        tail_swap_resolved_conflict = False  # Track if this tail swap resolved a conflict
        
        # Always check if tail swap resolved conflicts (for tracking), regardless of penalty flag
        if (resolved_conflicts 
            and flight_action != 0  # Agent must have taken an action
            and self.something_happened):  # Action must have changed the state
            for conflict in resolved_conflicts:
                if isinstance(conflict, (tuple, list)) and len(conflict) >= 2:
                    aircraft_id = conflict[0]
                    conflict_flight_id = conflict[1]
                else:
                    aircraft_id = conflict
                    conflict_flight_id = None

                # Only count conflicts directly resolved by the acted flight
                if conflict_flight_id is None or conflict_flight_id != flight_action:
                    continue
                # Do not count conflicts that were cancelled (manually or automatically) by the environment
                if conflict_flight_id in self.cancelled_flights or conflict_flight_id in self.automatically_cancelled_flights:
                    if DEBUG_MODE_REWARD:
                        print(f"  [Reward #8] Skipping conflict for flight {conflict_flight_id} - cancelled={conflict_flight_id in self.cancelled_flights}, auto_cancelled={conflict_flight_id in self.automatically_cancelled_flights}")
                    continue

                # If we got here, this action resolved a conflict
                if self.tail_swap_happened:
                    tail_swap_resolved_conflict = True
                
                # Calculate probability for bonus (only if penalty is enabled)
                if PENALTY_8_PROBABILITY_RESOLUTION_BONUS_ENABLED and PROBABILITY_RESOLUTION_BONUS_SCALE > 0:
                    # Use the probability snapshot from BEFORE uncertainties were processed this step
                    pre_prob = np.nan
                    if hasattr(self, "pre_action_probabilities"):
                        pre_prob = self.pre_action_probabilities.get(aircraft_id, np.nan)
                    if np.isnan(pre_prob):
                        pre_prob = self.unavailabilities_dict.get(aircraft_id, {}).get('Probability', np.nan)
                    if np.isnan(pre_prob):
                        pre_prob = 1.0  # fallback: treat as certain if unknown

                    resolved_probability_total += max(0.0, pre_prob)

        # Only apply bonus if penalty is enabled
        if PENALTY_8_PROBABILITY_RESOLUTION_BONUS_ENABLED and resolved_probability_total > 0:
            probability_resolution_bonus = resolved_probability_total * PROBABILITY_RESOLUTION_BONUS_SCALE
        
        # Always track tail swaps that resolved conflicts, regardless of penalty flag
        if tail_swap_resolved_conflict:
            self.scenario_wide_tail_swaps_resolving += 1

        if DEBUG_MODE_REWARD:
            status = "ENABLED" if PENALTY_8_PROBABILITY_RESOLUTION_BONUS_ENABLED else "DISABLED"
            if probability_resolution_bonus > 0:
                print(f"  [Reward #8: {status}] +{probability_resolution_bonus:.2f} bonus for resolving conflicts with total probability {resolved_probability_total:.2f}")

        # 9. Low-confidence action penalty: discourage acting on low-probability disruptions when nothing is resolved
        low_confidence_action_penalty = 0
        if PENALTY_9_LOW_CONFIDENCE_ACTION_ENABLED:
            if (
                flight_action != 0
                and self.something_happened
                and self.last_action_probability is not None
                and self.last_action_probability < LOW_CONFIDENCE_ACTION_THRESHOLD
                and hasattr(self, "pre_action_conflict_flights")
                and flight_action in self.pre_action_conflict_flights
                and len(resolved_conflicts) == 0
            ):
                low_confidence_action_penalty = LOW_CONFIDENCE_ACTION_PENALTY

        if DEBUG_MODE_REWARD:
            status = "ENABLED" if PENALTY_9_LOW_CONFIDENCE_ACTION_ENABLED else "DISABLED"
            if low_confidence_action_penalty > 0:
                print(f"  [Penalty #9: {status}] -{low_confidence_action_penalty} penalty for acting on low-confidence disruption (prob={self.last_action_probability:.2f} < {LOW_CONFIDENCE_ACTION_THRESHOLD}) without resolving conflicts")


        if self.tail_swap_happened and not tail_swap_resolved_conflict:
            self.scenario_wide_tail_swaps_inefficient += 1
        
        # Reset tail swap tracking for next step
        self.tail_swap_happened = False

        # Update penalized delays for next iteration
        for flight_id, delay in self.environment_delayed_flights.items():
            self.penalized_delays[flight_id] = delay

        reward = (
            - delay_penalty_total               
            - cancel_penalty                    
            - inaction_penalty                  
            - proactive_penalty                 
            - time_penalty                      
            - automatic_cancellation_penalty    
            - unresolved_conflict_penalty       
            - low_confidence_action_penalty     
            + probability_resolution_bonus      
            # + action_taking_bonus              
        )

        # Update scenario-wide reward components
        self.scenario_wide_reward_components.update({
            "delay_penalty_total": self.scenario_wide_reward_components["delay_penalty_total"] - delay_penalty_total,
            "cancel_penalty": self.scenario_wide_reward_components["cancel_penalty"] - cancel_penalty,
            "inaction_penalty": self.scenario_wide_reward_components["inaction_penalty"] - inaction_penalty,
            "proactive_penalty": self.scenario_wide_reward_components.get("proactive_penalty", 0) - proactive_penalty,
            "time_penalty": self.scenario_wide_reward_components["time_penalty"] - time_penalty,
            "unresolved_conflict_penalty": self.scenario_wide_reward_components.get("unresolved_conflict_penalty", 0) - unresolved_conflict_penalty,
            "automatic_cancellation_penalty": self.scenario_wide_reward_components["automatic_cancellation_penalty"] - automatic_cancellation_penalty,
            "probability_resolution_bonus": self.scenario_wide_reward_components.get("probability_resolution_bonus", 0) + probability_resolution_bonus,
            "low_confidence_action_penalty": self.scenario_wide_reward_components["low_confidence_action_penalty"] - low_confidence_action_penalty
            # "action_taking_bonus": self.scenario_wide_reward_components.get("action_taking_bonus", 0) + action_taking_bonus
        })

        # Store reward components in state
        self.state[0, 4] = reward
        self.state[0, 6] = delay_penalty_total
        self.state[0, 7] = cancel_penalty
        self.state[0, 8] = inaction_penalty
        self.state[0, 9] = proactive_penalty
        self.state[0, 10] = time_penalty
        self.state[0, 11] = automatic_cancellation_penalty
        self.state[0, 12] = unresolved_conflict_penalty
        self.state[0, 13] = probability_resolution_bonus
        self.state[0, 14] = low_confidence_action_penalty

        # Round final reward to 4 decimal places to preserve small penalty values (e.g., time penalty 0.003)
        # This ensures the DQN receives actual penalty values, not rounded zeros
        reward = round(reward, 4)

        if DEBUG_MODE_REWARD:
            print("--------------------------------")
            print(f"Total reward: {reward}")
            print("--------------------------------")

        # Store current unavailability probabilities for tracking evolution
        unavailabilities_probabilities = {}
        for aircraft_id in self.aircraft_ids:
            prob = self.unavailabilities_dict[aircraft_id]['Probability']
            start = self.unavailabilities_dict[aircraft_id]['StartTime']
            end = self.unavailabilities_dict[aircraft_id]['EndTime']
            unavailabilities_probabilities[aircraft_id] = {
                'probability': float(prob) if not np.isnan(prob) else None,
                'start_minutes': float(start) if not np.isnan(start) else None,
                'end_minutes': float(end) if not np.isnan(end) else None
            }
        
        # Store step information
        self.info_after_step = {
            "total_reward": reward,
            "something_happened": self.something_happened,
            "current_time_minutes": (self.current_datetime - self.earliest_datetime).total_seconds() / 60,
            "current_time_minutes_from_start": (self.current_datetime - self.start_datetime).total_seconds() / 60,
            "start_datetime": str(self.start_datetime),
            "earliest_datetime": str(self.earliest_datetime),
            "current_datetime": str(self.current_datetime),
            "resolved_conflicts_count": len(resolved_conflicts),
            "resolved_conflicts_entries": [tuple(conflict) for conflict in resolved_conflicts],
            "remaining_conflicts_count": len(remaining_conflicts),
            # Metadata: delay information (raw minutes and flags, not penalty values)
            "delay_penalty_minutes": delay_penalty_minutes,
            "delay_penalty_capped": delay_penalty_total == MAX_DELAY_PENALTY,
            # Metadata: cancellation information (count, not penalty value)
            "cancelled_flights_count": cancellation_penalty_count,
            # Metadata: time to departure
            "time_to_departure_minutes": time_to_departure,
            # Action information
            "action_index": None,  # Will be set by step() function
            "flight_action": flight_action,
            "aircraft_action": aircraft_action,
            "original_departure_time": original_flight_action_departure_time,
            # Scenario termination flag: indicates if scenario ended and final reward was calculated
            "scenario_ended": scenario_ended_flag,
            # Unavailability probabilities evolution (for visualization)
            "unavailabilities_probabilities": unavailabilities_probabilities,
            # All penalty values consolidated in penalties dict (no duplicates)
            "penalties": {
                "delay_penalty_total": delay_penalty_total,
                "cancel_penalty": cancel_penalty,
                "inaction_penalty": inaction_penalty,
                "automatic_cancellation_penalty": automatic_cancellation_penalty,
                "proactive_penalty": proactive_penalty,
                "time_penalty": time_penalty,
                "unresolved_conflict_penalty": unresolved_conflict_penalty,
                "probability_resolution_bonus": probability_resolution_bonus,
                "low_confidence_action_penalty": low_confidence_action_penalty
                # "action_taking_bonus": action_taking_bonus
            },
            # Penalty enable flags (for debugging/verification)
            "penalty_flags": {
                "penalty_1_delay_enabled": PENALTY_1_DELAY_ENABLED,
                "penalty_2_cancellation_enabled": PENALTY_2_CANCELLATION_ENABLED,
                "penalty_3_inaction_enabled": PENALTY_3_INACTION_ENABLED,
                "penalty_4_proactive_enabled": PENALTY_4_PROACTIVE_ENABLED,
                "penalty_5_time_enabled": PENALTY_5_TIME_ENABLED,
                "penalty_6_final_reward_enabled": PENALTY_6_FINAL_REWARD_ENABLED,
                "penalty_7_auto_cancellation_enabled": PENALTY_7_AUTO_CANCELLATION_ENABLED,
                "penalty_8_probability_resolution_bonus_enabled": PENALTY_8_PROBABILITY_RESOLUTION_BONUS_ENABLED,
                "penalty_9_low_confidence_action_enabled": PENALTY_9_LOW_CONFIDENCE_ACTION_ENABLED
            }
        }
        
        # Add scenario-wide metrics ONLY when scenario ends
        # This saves memory by not storing metrics at every step
        # Note: Metrics are saved regardless of whether penalty #6 is enabled
        if scenario_ended_flag:
            # Determine why scenario ended: time limit vs successful completion
            time_limit_reached = self.current_datetime >= self.end_datetime
            criteria_met = self.check_termination_criteria()
            ended_due_to_time_limit = time_limit_reached and not criteria_met
            
            self.info_after_step["scenario_metrics"] = {
                # Action counts
                "delay_minutes": self.scenario_wide_delay_minutes,
                "delay_count": self.scenario_wide_delay_count,
                "cancelled_flights": self.scenario_wide_cancelled_flights,
                "automatically_cancelled_count": self.scenario_wide_automatically_cancelled_count,
                "tail_swaps_total": self.scenario_wide_tail_swaps,
                "tail_swaps_resolving": self.scenario_wide_tail_swaps_resolving,
                "tail_swaps_inefficient": self.scenario_wide_tail_swaps_inefficient,
                "inaction_count": self.scenario_wide_inaction_count,
                "resolved_initial_conflicts": self.scenario_wide_resolved_initial_conflicts,
                "scenario_wide_disruption_resolved_to_zero_count": self.scenario_wide_disruption_resolved_to_zero_count,
                "steps": self.scenario_wide_steps,
                # Cumulative penalty/reward values (breakdown of total reward)
                "reward_components": {
                    "delay_penalty_total": self.scenario_wide_reward_components["delay_penalty_total"],
                    "cancel_penalty": self.scenario_wide_reward_components["cancel_penalty"],
                    "inaction_penalty": self.scenario_wide_reward_components["inaction_penalty"],
                    "proactive_penalty": self.scenario_wide_reward_components.get("proactive_penalty", 0),
                    "time_penalty": self.scenario_wide_reward_components["time_penalty"],
                    "unresolved_conflict_penalty": self.scenario_wide_reward_components.get("unresolved_conflict_penalty", 0),
                    "automatic_cancellation_penalty": self.scenario_wide_reward_components["automatic_cancellation_penalty"],
                    "probability_resolution_bonus": self.scenario_wide_reward_components.get("probability_resolution_bonus", 0),
                    "low_confidence_action_penalty": self.scenario_wide_reward_components["low_confidence_action_penalty"]
                }
            }

        return reward

    def _get_aircraft_temporal_features(self, aircraft_id, probability, start_minutes, end_minutes, current_time_minutes):
        """Derive engineered temporal features for a single aircraft timeline."""
        if not self.enable_temporal_features or self.num_temporal_features == 0:
            return np.array([], dtype=np.float32)

        features = np.full(self.num_temporal_features, np.nan, dtype=np.float32)

        time_to_start = np.nan if np.isnan(start_minutes) else start_minutes - current_time_minutes
        time_to_end = np.nan if np.isnan(end_minutes) else end_minutes - current_time_minutes

        previous_prob = self.previous_probabilities.get(aircraft_id, 0.0)
        if np.isnan(probability):
            probability_slope = 0.0
            probability_for_norm = 0.0
        else:
            probability_slope = float(probability) - float(previous_prob)
            probability_for_norm = float(probability)

        normalized_time_to_start = np.nan
        if not np.isnan(time_to_start):
            normalized_time_to_start = time_to_start / self.recovery_period_minutes

        if self.num_temporal_features >= 1:
            features[0] = time_to_start if not np.isnan(time_to_start) else np.nan
        if self.num_temporal_features >= 2:
            features[1] = time_to_end if not np.isnan(time_to_end) else np.nan
        if self.num_temporal_features >= 3:
            features[2] = probability_slope
        if self.num_temporal_features >= 4:
            features[3] = normalized_time_to_start
        if self.num_temporal_features >= 5:
            features[4] = probability_for_norm

        return features

    def _stack_observation(self, single_state_vector):
        """Return a stacked observation vector composed of the most recent frames."""
        if self.obs_stack_size == 1:
            return single_state_vector

        if len(self.state_history) == 0:
            padding = np.full_like(single_state_vector, STACKING_PADDING_VALUE, dtype=np.float32)
            for _ in range(self.obs_stack_size - 1):
                self.state_history.append(padding.copy())

        self.state_history.append(single_state_vector.copy())

        if len(self.state_history) < self.obs_stack_size:
            padding = np.full_like(single_state_vector, STACKING_PADDING_VALUE, dtype=np.float32)
            while len(self.state_history) < self.obs_stack_size:
                self.state_history.appendleft(padding.copy())

        stacked_vector = np.concatenate(list(self.state_history)).astype(np.float32)
        return stacked_vector
    
    #change this name to _does_not_create_new_conflicts
    def _is_proactive_move(self, flight_id, target_aircraft_id):
        """Check if a move action is proactive (prevents future conflicts).
        
        Args:
            flight_id (int): The flight being moved
            target_aircraft_id (int): The target aircraft
            
        Returns:
            bool: True if the move is proactive, & does not cause overlap with another unavailability period, False otherwise.
            Only true if both conditions are met.
        """
        if flight_id not in self.flights_dict:
            return False
            
        flight_info = self.flights_dict[flight_id]
        dep_time = parse_time_with_day_offset(flight_info['DepTime'], self.start_datetime)
        current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60
        dep_time_minutes = (dep_time - self.earliest_datetime).total_seconds() / 60
            
        # Check if the move prevents a future conflict
        # Look for unavailability periods on the target aircraft that would conflict
        if target_aircraft_id in self.unavailabilities_dict:
            unavail_start = self.unavailabilities_dict[target_aircraft_id]['StartTime']
            unavail_end = self.unavailabilities_dict[target_aircraft_id]['EndTime']
            unavail_prob = self.unavailabilities_dict[target_aircraft_id]['Probability']
            
            if (not np.isnan(unavail_start) and not np.isnan(unavail_end) and 
                unavail_prob > 0.0):  # Any probability > 0 indicates potential conflict
                
                arr_time = parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime)
                arr_time_minutes = (arr_time - self.earliest_datetime).total_seconds() / 60
                
                # Check if flight would overlap with unavailability period
                overlap_start = max(dep_time_minutes, unavail_start)
                overlap_end = min(arr_time_minutes, unavail_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration < 0:
                    return True 
                    
        return False

    def _calculate_scenario_wide_solution_slack(self):
        """Calculate the scenario-wide solution slack based on flight schedules."""
        # Extract all flights and their departure/arrival times
        all_dep_times = []
        all_arr_times = []
        for f_id, f_info in self.flights_dict.items():
            dep_dt = parse_time_with_day_offset(f_info['DepTime'], self.start_datetime)
            arr_dt = parse_time_with_day_offset(f_info['ArrTime'], self.start_datetime)
            all_dep_times.append(dep_dt)
            all_arr_times.append(arr_dt)

        if not all_dep_times:
            self.scenario_wide_solution_slack = 0.0
            return

        # Calculate time horizon
        earliest_dep = min(all_dep_times)
        latest_arr = max(all_arr_times)
        horizon = max(int((latest_arr - earliest_dep).total_seconds() / 60), 1)

        # Organize flights by aircraft
        aircraft_flights = {ac: [] for ac in self.aircraft_ids}
        for f_id, f_info in self.flights_dict.items():
            if f_id not in self.rotations_dict:
                continue

            ac_id = self.rotations_dict[f_id]['Aircraft']
            if ac_id not in aircraft_flights:
                continue

            dep_dt = parse_time_with_day_offset(f_info['DepTime'], self.start_datetime)
            arr_dt = parse_time_with_day_offset(f_info['ArrTime'], self.start_datetime)
            flight_duration = int((arr_dt - dep_dt).total_seconds() / 60)
            aircraft_flights[ac_id].append(flight_duration)

        # Calculate slack per aircraft
        aircraft_slacks = []
        for ac_id in self.aircraft_ids:
            if not aircraft_flights[ac_id]:
                ac_slack = 0.0
            else:
                total_flight_time = sum(aircraft_flights[ac_id])
                ac_slack = total_flight_time / horizon
            aircraft_slacks.append(ac_slack)

        # Calculate average slack across all aircraft
        self.scenario_wide_solution_slack = sum(aircraft_slacks) / len(aircraft_slacks) if aircraft_slacks else 0.0

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        # Generate a random seed based on current time if none provided
        if seed is None:
            seed = int(time.time() * 1000000) % (2**32 - 1)
        
        # Set random seeds for all random number generators
        random.seed(seed)
        np.random.seed(seed)
        
        # Rest of the reset method remains unchanged
        self.current_datetime = self.start_datetime
        self.actions_taken = set()
        self.state_history.clear()
        self.previous_probabilities = {aircraft_id: 0.0 for aircraft_id in self.aircraft_ids}
        self.last_action_probability = None
        self.last_action_aircraft_id = None

        self.something_happened = False

        # Deep copy the initial dictionaries
        self.aircraft_dict = copy.deepcopy(self.initial_aircraft_dict)
        self.flights_dict = copy.deepcopy(self.initial_flights_dict)
        self.rotations_dict = copy.deepcopy(self.initial_rotations_dict)
        self.alt_aircraft_dict = copy.deepcopy(self.initial_alt_aircraft_dict)

        # Clear and regenerate breakdowns
        self.uncertain_breakdowns = {}
        self.current_breakdowns = {}

        # Calculate total simulation minutes
        total_simulation_minutes = (self.end_datetime - self.start_datetime).total_seconds() / 60

        self.state = self._get_initial_state()

        self.initial_conflict_combinations = self.get_initial_conflicts()

        self.swapped_flights = []  # Reset the swapped flights list
        self.environment_delayed_flights = {}  # Reset the delayed flights list
        self.penalized_delays = {}  # Reset the penalized delays
        self.penalized_conflicts = set()
        self.resolved_conflicts = set()
        self.penalized_cancelled_flights = set()  # Reset penalized cancelled flights

        self.cancelled_flights = set()
        self.automatically_cancelled_flights = set()
        self.penalized_automatically_cancelled_flights = set()

        # Initialize eligible flights for conflict resolution bonus
        self.eligible_flights_for_resolved_bonus = self.get_initial_conflicts()

        # Process the state into an observation as a NumPy array
        processed_state, _ = self.process_observation(self.state)

        if DEBUG_MODE:
            print(f"State space shape: {self.state.shape}")
            print(f"Type of processed state: {type(processed_state)}")
            print(processed_state)
        return processed_state, {}

    def get_initial_conflicts(self):
        """Retrieves the initial conflicts in the environment.

        This function checks for conflicts between flights and unavailability periods,
        considering unavailabilities with probability greater than 0.0.

        Returns:
            set: A set of conflicts currently present in the initial state of the environment.
        """
        initial_conflicts = set()

        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            breakdown_probability = self.unavailabilities_dict[aircraft_id]['Probability']
            if breakdown_probability <= 0.0 or np.isnan(breakdown_probability):
                continue  # Only consider unavailabilities with probability > 0.0

            unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
            unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']

            if not np.isnan(unavail_start) and not np.isnan(unavail_end):
                # Check for conflicts between flights and unavailability periods
                for j in range(4, self.columns_state_space - 2, 3):
                    flight_id = self.state[idx + 1, j]
                    flight_dep = self.state[idx + 1, j + 1]
                    flight_arr = self.state[idx + 1, j + 2]

                    if not np.isnan(flight_dep) and not np.isnan(flight_arr):
                        # Skip cancelled flights
                        if flight_id in self.cancelled_flights:
                            continue

                        # Check for overlaps with unavailability periods
                        if flight_dep < unavail_end and flight_arr > unavail_start:
                            conflict_identifier = (aircraft_id, flight_id)
                            initial_conflicts.add(conflict_identifier)

        return initial_conflicts


    def get_initial_conflicts_with_deptime_before_unavail_start(self):
        """Retrieves the initial conflicts in the environment.

        This function checks for conflicts between flights and unavailability periods,
        considering unavailabilities with probability greater than 0.0.

        Returns:
            set: A set of conflicts currently present in the initial state of the environment.
        """
        initial_conflicts = set()

        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            breakdown_probability = self.unavailabilities_dict[aircraft_id]['Probability']
            if breakdown_probability <= 0.0 or np.isnan(breakdown_probability):
                continue  # Only consider unavailabilities with probability > 0.0

            unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
            unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']

            if not np.isnan(unavail_start) and not np.isnan(unavail_end):
                # Check for conflicts between flights and unavailability periods
                for j in range(4, self.columns_state_space - 2, 3):
                    flight_id = self.state[idx + 1, j]
                    flight_dep = self.state[idx + 1, j + 1]
                    flight_arr = self.state[idx + 1, j + 2]

                    if not np.isnan(flight_dep) and not np.isnan(flight_arr):
                        # Skip cancelled flights
                        if flight_id in self.cancelled_flights:
                            continue

                        # Check for overlaps with unavailability periods
                        if flight_dep < unavail_end and flight_arr > unavail_start:
                            if flight_dep < unavail_start:
                                conflict_identifier = (aircraft_id, flight_id)
                                initial_conflicts.add(conflict_identifier)

        return initial_conflicts


    def get_current_conflicts(self):
        """Retrieves the current conflicts in the environment.

        This function checks for conflicts between flights and unavailability periods,
        considering only unavailabilities with probability > 0.0.
        It excludes cancelled flights which are not considered conflicts.

        Returns:
            set: A set of conflicts currently present in the environment.
        """
        current_conflicts = set()
        # Calculate current_time_minutes once before the loops for performance
        current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60

        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            breakdown_probability = self.unavailabilities_dict[aircraft_id]['Probability']
            if breakdown_probability == 0.0:  # Only consider unavailability with probability > 0.0
                continue  # Skip if probability is 0.0

            unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
            unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']

            if not np.isnan(unavail_start) and not np.isnan(unavail_end):
                # Check for conflicts between flights and unavailability periods
                for j in range(4, self.columns_state_space - 2, 3):
                    flight_id = self.state[idx + 1, j]
                    flight_dep = self.state[idx + 1, j + 1]
                    flight_arr = self.state[idx + 1, j + 2]

                    if not np.isnan(flight_dep) and not np.isnan(flight_arr):
                        # Check if the flight's departure is in the past (relative to current time)
                        if flight_dep < current_time_minutes:
                            continue  # Skip past flights

                        if flight_id in self.cancelled_flights:
                            continue  # Skip cancelled flights

                        # Check for overlaps with unavailability periods with prob > 0.0
                        if flight_dep < unavail_end and flight_arr > unavail_start:
                            conflict_identifier = (aircraft_id, flight_id, flight_dep, flight_arr)
                            current_conflicts.add(conflict_identifier)
                            if DEBUG_MODE_REWARD:
                                print(f"  [CONFLICT DETECTED] {aircraft_id}, Flight {flight_id}: dep={flight_dep:.1f}, arr={flight_arr:.1f} vs unavail {unavail_start:.1f}-{unavail_end:.1f}")
                        elif DEBUG_MODE_REWARD:
                            # Debug: Why wasn't this a conflict?
                            print(f"  [NO CONFLICT] {aircraft_id}, Flight {flight_id}: dep={flight_dep:.1f}, arr={flight_arr:.1f} vs unavail {unavail_start:.1f}-{unavail_end:.1f}")

        return current_conflicts
    

    def get_current_conflicts_with_prob_1(self):
        """Retrieves the current conflicts in the environment.

        This function checks for conflicts between flights and unavailability periods,
        considering only unavailabilities with probability 1.0.
        It excludes cancelled flights which are not considered conflicts.

        Returns:
            set: A set of conflicts currently present in the environment.
        """
        current_conflicts_with_prob_1 = set()
        # Calculate current_time_minutes once before the loops for performance
        current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60

        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            breakdown_probability = self.unavailabilities_dict[aircraft_id]['Probability']
            if breakdown_probability != 1.0:  # Only consider unavailability with probability 1.00
                continue  # Skip if probability is not 1.00

            unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
            unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']

            if not np.isnan(unavail_start) and not np.isnan(unavail_end):
                # Check for conflicts between flights and unavailability periods
                for j in range(4, self.columns_state_space - 2, 3):
                    flight_id = self.state[idx + 1, j]
                    flight_dep = self.state[idx + 1, j + 1]
                    flight_arr = self.state[idx + 1, j + 2]

                    if not np.isnan(flight_dep) and not np.isnan(flight_arr):
                        # Check if the flight's departure is in the past (relative to current time)
                        if flight_dep < current_time_minutes:
                            continue  # Skip past flights

                        if flight_id in self.cancelled_flights:
                            continue  # Skip cancelled flights

                        # Check for overlaps with unavailability periods with prob = 1.00
                        if flight_dep < unavail_end and flight_arr > unavail_start:
                            conflict_identifier = (aircraft_id, flight_id, flight_dep, flight_arr)
                            current_conflicts_with_prob_1.add(conflict_identifier)

        return current_conflicts_with_prob_1

    def check_flight_disruption_overlaps(self):
        """Checks if there are any overlaps between flights and disruptions (certain or uncertain).
        
        Returns:
            bool: True if there are overlaps, False otherwise.
        """
        # Calculate current_time_minutes once before the loops for performance
        current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60
        
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            # Get disruption info
            unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
            unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']
            prob = self.unavailabilities_dict[aircraft_id]['Probability']

            # Skip if no disruption period defined
            if np.isnan(unavail_start) or np.isnan(unavail_end):
                continue

            # Check flights on this aircraft
            for j in range(4, self.columns_state_space - 2, 3):
                flight_id = self.state[idx + 1, j]
                flight_dep = self.state[idx + 1, j + 1]
                flight_arr = self.state[idx + 1, j + 2]

                if not np.isnan(flight_dep) and not np.isnan(flight_arr):
                    # Skip cancelled flights
                    if flight_id in self.cancelled_flights:
                        continue

                    # Skip past flights
                    if flight_arr < current_time_minutes:
                        continue

                    # Check for overlap with disruption period
                    if flight_dep < unavail_end and flight_arr > unavail_start:
                        return True  # Found an overlap

        return False  # No overlaps found

    def termination_criteria_met(self):
        """
        Checks if the stopping criteria are met.

        Stopping criteria:
        - There are no uncertainties in the system anymore, i.e., for all aircraft:
          Probability is either NaN, exactly 0.0, or exactly 1.0.
        - AND there is no overlap of BREAKDOWNS (with Probability == 1.00) and flights.

        Returns:
            bool: True if the stopping criteria are met, False otherwise.
        """
        # Check if all probabilities are either nan, 0.0, or 1.0
        for aircraft_id in self.aircraft_ids:
            prob = self.unavailabilities_dict[aircraft_id]['Probability']
            if not (np.isnan(prob) or prob == 0.0 or prob == 1.0):
                return False

        # Check if there is no overlap of breakdowns (prob == 1.00) and flights
        # If check_flight_disruption_overlaps() returns False, that means no overlap.
        if self.check_flight_disruption_overlaps():
            return False

        return True

    
    def end_of_recovery_period_reached(self):
        """Checks if the scenario has ended because the time exceeds the recovery period.

        Returns:
            tuple: (bool, str) indicating if the scenario is done and the reason.
        """
        if self.current_datetime >= self.end_datetime:
            return True, "Reached the end of the simulation time."
        else:
            return False, "Endtime scenario not yet reached"
        

    def _is_done(self):
        """Checks if the episode is finished.

        The episode is considered done if:
        1. Current time has reached or exceeded the end time, OR
        2. There are no overlaps between flights and disruptions (regardless of uncertainty status)

        Returns:
            tuple: (bool, str) indicating if the episode is done and the reason.
        """
        if self.current_datetime >= self.end_datetime:
            return True, "Reached the end of the simulation time."
        
        # Check for any overlaps between flights and disruptions
        if not self.check_flight_disruption_overlaps():
            return True, "No remaining overlaps between flights and disruptions."
        
        return False, ""

    def get_unresolved_uncertainties(self):
        """Retrieves the uncertainties that have not yet been resolved.

        Returns:
            list: A list of unresolved uncertainties currently present in the environment.
        """
        unresolved_uncertainties = []
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            prob = self.unavailabilities_dict[aircraft_id]['Probability']
            if prob != 0.00 and prob != 1.00 and not np.isnan(prob):
                # Uncertainty not yet resolved
                start_minutes = self.unavailabilities_dict[aircraft_id]['StartTime']
                breakdown_start_time = self.earliest_datetime + timedelta(minutes=start_minutes)
                if self.current_datetime < breakdown_start_time:
                    unresolved_uncertainties.append((aircraft_id, prob))
        return unresolved_uncertainties

    # YES: need to modify for new state-space formulation 
    def get_valid_flight_actions(self):
        """Generates a list of valid flight actions based on flights in state space."""
        # Calculate current time in minutes from earliest_datetime
        current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60

        # Get all valid flight IDs from the actual flights in the scenario
        valid_flight_ids = set()
        for flight_id in self.flights_dict.keys():
            if flight_id not in self.cancelled_flights:
                # Check if flight hasn't departed yet
                flight_info = self.flights_dict[flight_id]
                dep_time = parse_time_with_day_offset(flight_info['DepTime'], self.start_datetime)
                dep_time_minutes = (dep_time - self.earliest_datetime).total_seconds() / 60
                
                if dep_time_minutes >= current_time_minutes:
                    valid_flight_ids.add(flight_id)

        # Convert to sorted list and add 'no action' option
        valid_flight_ids = sorted(list(valid_flight_ids))
        
        # Update flight_id_to_idx mapping using actual flight IDs
        self.flight_id_to_idx = {
            flight_id: flight_id - 1 for flight_id in valid_flight_ids
        }

        if DEBUG_MODE_ACTION:
            print(f"Valid flight indices: {valid_flight_ids}")
        
        # Return [0] + actual flight IDs instead of creating sequential indices
        return [0] + valid_flight_ids


    # NO need to modify for new state-space formulation
    def get_valid_aircraft_actions(self):
        """Generates a list of valid aircraft actions for the agent.

        Returns:
            list: A list of valid aircraft actions that the agent can take.
        """
        return list(range(len(self.aircraft_ids) + 1))  # 0 to len(aircraft_ids)


    # YES: need to modify for new state-space formulation
    # get_action_mask creates a binary vector action_mask that gets filled with 1s for valid flight & ac pairs (assign flight x to aircraft y )
    # it considers all valid (flight has not departed yet) flight & ac pairs regardless whether it resolves a conflict or not
    # whether the chosen pair will create a new conflict we do not know yet - will be checked later in step
    def get_action_mask(self):
        valid_flight_actions = self.get_valid_flight_actions()
        valid_aircraft_actions = self.get_valid_aircraft_actions()

        action_mask = np.zeros(self.action_space.n, dtype=np.uint8)

        for flight_action in valid_flight_actions:
            for aircraft_action in valid_aircraft_actions:
                if flight_action == 0:
                    # Only allow (flight_action=0, aircraft_action=0)
                    if aircraft_action != 0:
                        continue
                index = self.map_action_to_index(flight_action, aircraft_action) 
                if index < self.action_space.n:
                    # Only allow logical actions
                    if self.evaluate_action_impact(flight_action, aircraft_action):
                        action_mask[index] = 1

        # For reactive environment, only allow 0,0 action unless an immediate or imminent conflict with prob==1.00 exists; then use default action mask
        # Check each aircraft for disruptions with probability 1
        # look for ac unavailibility periods of p = 1: if current_time is inside that period rn, we can use default action mask. if it is NOT inside:
        # default action mask can still be used IF the next timestep will reach or pass the start of a disruption or the departure of a conflicted flight
        # Otherwise: only 0,0 action can be chosen aka nothing is done - no flight is allocated to an ac
        if self.env_type == 'reactive':
            reactive_allowed_to_take_action = False
            current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60

            # Find earliest disrupted flight departure time and earliest disruption start time
            earliest_disrupted_dep = float('inf')
            earliest_disruption_start = float('inf')
            earliest_disruption_end = float('inf')

            for idx, aircraft_id in enumerate(self.aircraft_ids):
                breakdown_prob = self.unavailabilities_dict[aircraft_id]['Probability']
                if breakdown_prob == 1.0:
                    unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
                    unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']
                    # print(f"*** unavail_start: {unavail_start}")
                    # print(f"*** unavail_end: {unavail_end}")
                    if not np.isnan(unavail_start):
                        
                        earliest_disruption_start = min(earliest_disruption_start, unavail_start)
                        
                        # Check if current time is inside disruption period
                        if current_time_minutes >= unavail_start and current_time_minutes <= unavail_end:
                            reactive_allowed_to_take_action = True
                            break
                            
                        # Check flights assigned to this aircraft for departures during disruption
                        for j in range(4, self.columns_state_space - 2, 3):
                            flight_id = self.state[idx + 1, j]
                            dep_time = self.state[idx + 1, j + 1]
                            arr_time = self.state[idx + 1, j + 2]
                            if not np.isnan(flight_id) and not np.isnan(dep_time):
                                # print(f"*** flight_id: {flight_id}")
                                # print(f"*** dep_time: {dep_time}")
                                # print(f"*** arr_time: {arr_time}")
                                # there is any overlap (whichever starts first, check the deptime/startime)
                                if dep_time < unavail_end and arr_time > unavail_start:
                                    earliest_disrupted_dep = min(earliest_disrupted_dep, dep_time)
                                    # print(f"*** earliest disrupted dep: {earliest_disrupted_dep} of flight {flight_id}")

            # Allow reactive action if approaching either critical time
            # Use consistent time calculation
            earliest_critical_time = min(earliest_disrupted_dep, earliest_disruption_start)
            
            if current_time_minutes + self.timestep_minutes >= earliest_critical_time:
                reactive_allowed_to_take_action = True
            if not reactive_allowed_to_take_action:
                # Reset mask to all zeros except for 0,0 action
                action_mask[:] = 0
                action_mask[0] = 1  # Only allow 0,0 action

        return action_mask

    # NO need to modify for new state-space formulation
    def map_action_to_index(self, flight_action, aircraft_action):
        """Maps the (flight, aircraft) action pair to a single index in the flattened action space.

        Args:
            flight_action (int): The index of the flight action.
            aircraft_action (int): The index of the aircraft action.

        Returns:
            int: The corresponding index in the flattened action space.
        """
        return flight_action * (len(self.aircraft_ids) + 1) + aircraft_action
    
    # NO need to modify for new state-space formulation
    def map_index_to_action(self, index): 
        """Maps the flattened action space index to the corresponding (flight, aircraft) action pair.

        Args:
            index (int): The index in the flattened action space.

        Returns:
            tuple: A tuple containing the flight and aircraft actions.
        """
        flight_action = index // (len(self.aircraft_ids) + 1)
        aircraft_action = index % (len(self.aircraft_ids) + 1)
        
        if flight_action > 0 and flight_action not in self.flights_dict:
            return -1, -1
            
        return flight_action, aircraft_action

    def evaluate_action_impact(self, flight_action, aircraft_action):
        """
        Evaluates the impact of an action to determine if it's logical.
        
        Returns True if the action is logical (prioritizes delays over cancellations
        when both are possible), False otherwise.
        
        Args:
            flight_action (int): The flight action to evaluate
            aircraft_action (int): The aircraft action to evaluate
            
        Returns:
            bool: True if action is logical, False otherwise
        """
        if flight_action == 0:
            if DEBUG_MODE_ACTION_EVALUATION:
                print(f"Action evaluation: No action (0,0) - always logical")
            return True  # No action is always logical
            
        if aircraft_action == 0:
            # Cancellation action - check if delay is possible instead
            is_necessary = self._is_cancellation_necessary(flight_action)
            if DEBUG_MODE_ACTION_EVALUATION:
                print(f"Action evaluation: Cancellation of flight {flight_action} - necessary: {is_necessary}")
            return is_necessary
        else:
            # Rescheduling action - check if it creates new conflicts
            is_logical = self._is_reschedule_logical(flight_action, aircraft_action)
            if DEBUG_MODE_ACTION_EVALUATION:
                print(f"Action evaluation: Reschedule flight {flight_action} to aircraft {aircraft_action} - logical: {is_logical}")
            return is_logical
    
    def _is_cancellation_necessary(self, flight_action):
        """
        Checks if cancellation is necessary or if delay is possible.
        
        Args:
            flight_action (int): The flight to check
            
        Returns:
            bool: True if cancellation is necessary, False if delay is possible
        """
        if flight_action not in self.rotations_dict:
            if DEBUG_MODE_ACTION_EVALUATION:
                print(f"  Cancellation check: Flight {flight_action} not in rotations_dict - necessary")
            return True  # Flight doesn't exist, cancellation is necessary
            
        current_aircraft_id = self.rotations_dict[flight_action]['Aircraft']
        
        # Check if the flight can be delayed on the same aircraft
        aircraft_idx = self.aircraft_id_to_idx[current_aircraft_id] + 1
        unavail_end = self.state[aircraft_idx, 3]
        
        if not np.isnan(unavail_end):
            # There's an unavailability period - check if flight can be delayed after it
            flight_info = self.flights_dict[flight_action]
            original_dep_time = parse_time_with_day_offset(flight_info['DepTime'], self.start_datetime)
            original_arr_time = parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime)
            
            original_dep_minutes = (original_dep_time - self.earliest_datetime).total_seconds() / 60
            original_arr_minutes = (original_arr_time - self.earliest_datetime).total_seconds() / 60
            
            # If flight can be delayed after unavailability, cancellation is not necessary
            if original_arr_minutes > unavail_end:
                if DEBUG_MODE_ACTION_EVALUATION:
                    print(f"  Cancellation check: Flight {flight_action} can be delayed after unavailability - not necessary")
                return False  # Delay is possible, cancellation not necessary
                
        # Check if flight can be moved to another aircraft
        for other_aircraft_id in self.aircraft_ids:
            if other_aircraft_id != current_aircraft_id:
                if self._can_flight_be_moved_to_aircraft(flight_action, other_aircraft_id):
                    if DEBUG_MODE_ACTION_EVALUATION:
                        print(f"  Cancellation check: Flight {flight_action} can be moved to {other_aircraft_id} - not necessary")
                    return False  # Move is possible, cancellation not necessary
                    
        if DEBUG_MODE_ACTION_EVALUATION:
            print(f"  Cancellation check: Flight {flight_action} has no alternatives - necessary")
        return True  # No alternatives found, cancellation is necessary
    
    def _can_flight_be_moved_to_aircraft(self, flight_id, target_aircraft_id):
        """
        Checks if a flight can be moved to a target aircraft without creating conflicts.
        
        Args:
            flight_id (int): The flight to check
            target_aircraft_id (str): The target aircraft
            
        Returns:
            bool: True if move is possible, False otherwise
        """
        if flight_id not in self.flights_dict:
            return False
            
        flight_info = self.flights_dict[flight_id]
        dep_time = parse_time_with_day_offset(flight_info['DepTime'], self.start_datetime)
        arr_time = parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime)
        
        dep_minutes = (dep_time - self.earliest_datetime).total_seconds() / 60
        arr_minutes = (arr_time - self.earliest_datetime).total_seconds() / 60
        
        # Check for unavailability conflicts on target aircraft
        target_aircraft_idx = self.aircraft_id_to_idx[target_aircraft_id] + 1
        unavail_start = self.state[target_aircraft_idx, 2]
        unavail_end = self.state[target_aircraft_idx, 3]
        unavail_prob = self.state[target_aircraft_idx, 1]
        
        if not np.isnan(unavail_start) and not np.isnan(unavail_end) and unavail_prob > 0.0:
            # Check for overlap with unavailability
            if dep_minutes < unavail_end and arr_minutes > unavail_start:
                if unavail_prob == 1.0:
                    return False  # Certain conflict, move not possible
                # For uncertain conflicts, allow the move (will be resolved later)
        
        # Check for conflicts with existing flights on target aircraft
        for j in range(4, self.columns_state_space - 2, 3):
            existing_flight_id = self.state[target_aircraft_idx, j]
            existing_dep = self.state[target_aircraft_idx, j + 1]
            existing_arr = self.state[target_aircraft_idx, j + 2]
            
            if not np.isnan(existing_flight_id) and not np.isnan(existing_dep) and not np.isnan(existing_arr):
                # Check for overlap
                if dep_minutes < existing_arr and arr_minutes > existing_dep:
                    return False  # Conflict with existing flight
                    
        return True  # No conflicts found, move is possible
    
    def _is_reschedule_logical(self, flight_action, aircraft_action):
        """
        Checks if rescheduling a flight to a different aircraft is logical.
        
        Args:
            flight_action (int): The flight to reschedule
            aircraft_action (int): The target aircraft
            
        Returns:
            bool: True if reschedule is logical, False otherwise
        """
        if flight_action not in self.rotations_dict:
            return False
            
        current_aircraft_id = self.rotations_dict[flight_action]['Aircraft']
        target_aircraft_id = self.aircraft_ids[aircraft_action - 1]
        
        # If same aircraft, it's a delay action which is always logical
        if target_aircraft_id == current_aircraft_id:
            return True
            
        # Check if the move actually resolves a conflict
        if not self._does_move_resolve_conflict(flight_action, current_aircraft_id, target_aircraft_id):
            return False
            
        # Check if the move creates new conflicts
        if not self._can_flight_be_moved_to_aircraft(flight_action, target_aircraft_id):
            return False
            
        return True
    
    def _does_move_resolve_conflict(self, flight_id, current_aircraft_id, target_aircraft_id):
        """
        Checks if moving a flight resolves an existing conflict.
        
        Args:
            flight_id (int): The flight to move
            current_aircraft_id (str): Current aircraft
            target_aircraft_id (str): Target aircraft
            
        Returns:
            bool: True if move resolves a conflict, False otherwise
        """
        # Check if there's a conflict on the current aircraft
        current_aircraft_idx = self.aircraft_id_to_idx[current_aircraft_id] + 1
        unavail_start = self.state[current_aircraft_idx, 2]
        unavail_end = self.state[current_aircraft_idx, 3]
        unavail_prob = self.state[current_aircraft_idx, 1]
        
        if not np.isnan(unavail_start) and not np.isnan(unavail_end) and unavail_prob > 0.0:
            flight_info = self.flights_dict[flight_id]
            dep_time = parse_time_with_day_offset(flight_info['DepTime'], self.start_datetime)
            arr_time = parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime)
            
            dep_minutes = (dep_time - self.earliest_datetime).total_seconds() / 60
            arr_minutes = (arr_time - self.earliest_datetime).total_seconds() / 60
            
            # Check if flight conflicts with unavailability
            if dep_minutes < unavail_end and arr_minutes > unavail_start:
                return True  # Move resolves this conflict
                
        return False  # No conflict to resolve

    def _better_alternatives_exist(self, flight_action):
        """
        Checks if better alternatives exist for a flight than cancellation.
        
        Args:
            flight_action (int): The flight to check
            
        Returns:
            bool: True if better alternatives exist, False otherwise
        """
        if flight_action not in self.rotations_dict:
            return False
            
        current_aircraft_id = self.rotations_dict[flight_action]['Aircraft']
        
        # Check if delay is possible on same aircraft
        aircraft_idx = self.aircraft_id_to_idx[current_aircraft_id] + 1
        unavail_end = self.state[aircraft_idx, 3]
        
        if not np.isnan(unavail_end):
            flight_info = self.flights_dict[flight_action]
            original_arr_time = parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime)
            original_arr_minutes = (original_arr_time - self.earliest_datetime).total_seconds() / 60
            
            if original_arr_minutes > unavail_end:
                return True  # Delay is possible
        
        # Check if move to another aircraft is possible
        for other_aircraft_id in self.aircraft_ids:
            if other_aircraft_id != current_aircraft_id:
                if self._can_flight_be_moved_to_aircraft(flight_action, other_aircraft_id):
                    return True  # Move is possible
                    
        return False  # No better alternatives found

class AircraftDisruptionGreedyReactive(AircraftDisruptionEnv):
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict):
        # Initialize the environment with 'myopic' type (since we want to see all conflicts)
        super().__init__(aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict, env_type='reactive')
        
        # Initialize solution tracking
        self.solution = {
            "objective_value": 0,
            "assignments": {},
            "cancellations": [],
            "delays": {},
            "total_delay_minutes": 0,
            "statistics": {
                "runtime": 0,
                "gap": 0,
                "node_count": 0,
                "status": "In Progress"
            }
        }
        
        # Track start time for runtime calculation
        self.start_time = None
        
        # Track action history
        self.action_history = []

        self.scenario_wide_reward_total = 0
        

    def solve(self):
        """Solve the problem step by step using the environment's mechanics"""
        self.start_time = time.time()
        observation, _ = self.reset()  # Reset environment to initial state
        
        terminated = False
        total_reward = 0
        step_count = 0
        
        while not terminated:
            step_count += 1
            # print(f"\nStep {step_count}:")
            
            # Get current conflicts
            conflicts = self.get_current_conflicts()
            # print(f"Current conflicts: {conflicts}")
            
            if not conflicts:
                # No conflicts - take no action
                action = self.map_action_to_index(0, 0)  # No-op action
                # print("No conflicts - taking no-op action (0, 0)")
            else:
                # Choose the best action for the current state using valid action mask
                action = self.select_best_action()
                flight_action, aircraft_action = self.map_index_to_action(action)
                # print(f"Selected action: index={action} (flight={flight_action}, aircraft={aircraft_action})")
            
            # Take the action in the environment
            observation, reward, terminated, truncated, info = self.step(action)
            total_reward += reward
            # print(f"Action result: reward={reward}, terminated={terminated}")
            
            # Record action history
            flight_action, aircraft_action = self.map_index_to_action(action)
            self.action_history.append({
                'step': step_count,
                'action_index': action,
                'flight': flight_action,
                'aircraft': aircraft_action,
                'reward': reward,
                'conflicts': len(conflicts)
            })
            
            # Update solution based on the action taken
            self.update_solution(info)
        
        # Print action history summary
        # print("\nAction History Summary:")
        # print("----------------------")
        # print(f"{'Step':>4} | {'Flight':>6} | {'Aircraft':>8} | {'Reward':>8} | {'Conflicts':>9}")
        # print("-" * 45)
        # for entry in self.action_history:
        #     print(f"{entry['step']:4d} | {entry['flight']:6} | {entry['aircraft']:8} | {entry['reward']:8.1f} | {entry['conflicts']:9d}")
        # print("-" * 45)
        # print(f"Total Reward: {total_reward:.1f}")

        self.scenario_wide_reward_total = total_reward
        
        # Finalize solution
        self.solution["objective_value"] = -total_reward  # Convert reward to cost
        self.solution["statistics"]["runtime"] = time.time() - self.start_time
        self.solution["statistics"]["status"] = "Complete"
        
        return self.solution

    def select_best_action(self):
        """Select the best action from the valid action space using a greedy heuristic"""
        best_action = None
        best_score = float('-inf')
        # For reactive environment, only allow 0,0 action if no current conflicts with prob==1.00
        reactive_allowed_to_take_action = False
        current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60

        # Find earliest disrupted flight departure time and earliest disruption start time 
        earliest_disrupted_dep = float('inf')
        earliest_disruption_start = float('inf')
        earliest_disruption_end = float('inf')

        # Check each aircraft for disruptions with probability 1
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            breakdown_prob = self.unavailabilities_dict[aircraft_id]['Probability']
            if breakdown_prob == 1.0:
                unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
                unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']
                if not np.isnan(unavail_start):
                    earliest_disruption_start = min(earliest_disruption_start, unavail_start)
                    
                    # Check if current time is inside disruption period
                    if current_time_minutes >= unavail_start and current_time_minutes <= unavail_end:
                        reactive_allowed_to_take_action = True
                        break
                        
                    # Check flights assigned to this aircraft for departures during disruption
                    for j in range(4, self.columns_state_space - 2, 3):
                        flight_id = self.state[idx + 1, j]
                        dep_time = self.state[idx + 1, j + 1]
                        arr_time = self.state[idx + 1, j + 2]
                        if not np.isnan(flight_id) and not np.isnan(dep_time):
                            if dep_time < unavail_end and arr_time > unavail_start:
                                earliest_disrupted_dep = min(earliest_disrupted_dep, dep_time)

        # Allow reactive action if approaching either critical time
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
        earliest_critical_time = min(earliest_disrupted_dep, earliest_disruption_start)
        
        if current_time_minutes + self.timestep_minutes >= earliest_critical_time:
            reactive_allowed_to_take_action = True

        if not reactive_allowed_to_take_action:
            return self.map_action_to_index(0, 0)  # Only allow 0,0 action
        
        # Get valid actions using the environment's action mask
        action_mask = self.get_action_mask()
        valid_actions = np.where(action_mask == 1)[0]
        # print(f"\nEvaluating {len(valid_actions)} valid actions:")
        
        # Try each valid action and evaluate its impact
        for action in valid_actions:
            flight_action, aircraft_action = self.map_index_to_action(action)
            # Create a copy of the environment to simulate the action
            env_copy = copy.deepcopy(self)
            
            # Take the action in the copied environment
            _, reward, _, _, _ = env_copy.step(action)
            
            # print(f"  Action {action} (flight={flight_action}, aircraft={aircraft_action}): reward={reward}")
            
            # Update best action if this one has better reward
            if reward > best_score:
                best_score = reward
                best_action = action
                # print(f"    -> New best action (reward={reward})")
        
        # If no good action found, take no action (should be included in valid_actions)
        if best_action is None:
            best_action = self.map_action_to_index(0, 0)
            # print("No good action found, defaulting to no-op action (0, 0)")
        else:
            flight_action, aircraft_action = self.map_index_to_action(best_action)
            # print(f"\nChosen best action: index={best_action} (flight={flight_action}, aircraft={aircraft_action}) with reward={best_score}")
        
        return best_action

    def update_solution(self, info):
        """Update the solution dictionary based on the action taken"""
        # Update cancellations
        if 'cancelled_flights_count' in info and info['cancelled_flights_count'] > 0:
            self.solution['cancellations'] = list(self.cancelled_flights)
        
        # Update assignments
        if 'flight_action' in info and 'aircraft_action' in info:
            flight_id = info['flight_action']
            aircraft_idx = info['aircraft_action']
            if flight_id != 0 and aircraft_idx != 0:
                aircraft_id = self.aircraft_ids[aircraft_idx - 1]
                if aircraft_id != self.rotations_dict[flight_id]['Aircraft']:
                    self.solution['assignments'][flight_id] = aircraft_id
        
        # Update delays
        if self.environment_delayed_flights:
            self.solution['delays'] = {k: v for k, v in self.environment_delayed_flights.items()}
            self.solution['total_delay_minutes'] = sum(self.environment_delayed_flights.values())




class AircraftDisruptionGreedyProactive(AircraftDisruptionEnv):
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict):
        # Initialize the environment with 'myopic' type (since we want to see all conflicts)
        super().__init__(aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict, env_type='proactive')
        
        # Initialize solution tracking
        self.solution = {
            "objective_value": 0,
            "assignments": {},
            "cancellations": [],
            "delays": {},
            "total_delay_minutes": 0,
            "statistics": {
                "runtime": 0,
                "gap": 0,
                "node_count": 0,
                "status": "In Progress"
            }
        }
        
        # Track start time for runtime calculation
        self.start_time = None
        
        # Track action history
        self.action_history = []

        self.scenario_wide_reward_total = 0
        

    def solve(self):
        """Solve the problem step by step using the environment's mechanics"""
        self.start_time = time.time()
        observation, _ = self.reset()  # Reset environment to initial state
        
        terminated = False
        total_reward = 0
        step_count = 0
        
        while not terminated:
            step_count += 1
            # print(f"\nStep {step_count}:")
            
            # Get current conflicts
            conflicts = self.get_current_conflicts()
            # print(f"Current conflicts: {conflicts}")
            
            if not conflicts:
                # No conflicts - take no action
                action = self.map_action_to_index(0, 0)  # No-op action
                # print("No conflicts - taking no-op action (0, 0)")
            else:
                # Choose the best action for the current state using valid action mask
                action = self.select_best_action()
                flight_action, aircraft_action = self.map_index_to_action(action)
                # print(f"Selected action: index={action} (flight={flight_action}, aircraft={aircraft_action})")
            
            # Take the action in the environment
            observation, reward, terminated, truncated, info = self.step(action)
            total_reward += reward
            # print(f"Action result: reward={reward}, terminated={terminated}")
            
            # Record action history
            flight_action, aircraft_action = self.map_index_to_action(action)
            self.action_history.append({
                'step': step_count,
                'action_index': action,
                'flight': flight_action,
                'aircraft': aircraft_action,
                'reward': reward,
                'conflicts': len(conflicts)
            })
            
            # Update solution based on the action taken
            self.update_solution(info)

        self.scenario_wide_reward_total = total_reward
        
        # Finalize solution
        self.solution["objective_value"] = -total_reward  # Convert reward to cost
        self.solution["statistics"]["runtime"] = time.time() - self.start_time
        self.solution["statistics"]["status"] = "Complete"
        
        return self.solution

    def select_best_action(self):
        """Select the best action from the valid action space using a greedy heuristic"""
        best_action = None
        best_score = float('-inf')
        # For reactive environment, only allow 0,0 action if no current conflicts with prob==1.00
        reactive_allowed_to_take_action = False
        current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60

        # Find earliest disrupted flight departure time and earliest disruption start time 
        earliest_disrupted_dep = float('inf')
        earliest_disruption_start = float('inf')
        earliest_disruption_end = float('inf')

        # Check each aircraft for disruptions with probability 1
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            breakdown_prob = self.unavailabilities_dict[aircraft_id]['Probability']
            if breakdown_prob == 1.0:
                unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
                unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']
                if not np.isnan(unavail_start):
                    earliest_disruption_start = min(earliest_disruption_start, unavail_start)
                    
                    # Check if current time is inside disruption period
                    if current_time_minutes >= unavail_start and current_time_minutes <= unavail_end:
                        reactive_allowed_to_take_action = True
                        break
                        
                    # Check flights assigned to this aircraft for departures during disruption
                    for j in range(4, self.columns_state_space - 2, 3):
                        flight_id = self.state[idx + 1, j]
                        dep_time = self.state[idx + 1, j + 1]
                        arr_time = self.state[idx + 1, j + 2]
                        if not np.isnan(flight_id) and not np.isnan(dep_time):
                            if dep_time < unavail_end and arr_time > unavail_start:
                                earliest_disrupted_dep = min(earliest_disrupted_dep, dep_time)

        # Allow reactive action if approaching either critical time
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
        earliest_critical_time = min(earliest_disrupted_dep, earliest_disruption_start)
        
        if current_time_minutes + self.timestep_minutes >= earliest_critical_time:
            reactive_allowed_to_take_action = True

        # if not reactive_allowed_to_take_action:
        #     return self.map_action_to_index(0, 0)  # Only allow 0,0 action
        
        # Get valid actions using the environment's action mask
        action_mask = self.get_action_mask()
        valid_actions = np.where(action_mask == 1)[0]
        # print(f"\nEvaluating {len(valid_actions)} valid actions:")
        
        # Try each valid action and evaluate its impact
        for action in valid_actions:
            flight_action, aircraft_action = self.map_index_to_action(action)
            # Create a copy of the environment to simulate the action
            env_copy = copy.deepcopy(self)
            
            # Take the action in the copied environment
            _, reward, _, _, _ = env_copy.step(action)
            
            # print(f"  Action {action} (flight={flight_action}, aircraft={aircraft_action}): reward={reward}")
            
            # Update best action if this one has better reward
            if reward > best_score:
                best_score = reward
                best_action = action
                # print(f"    -> New best action (reward={reward})")
        
        # If no good action found, take no action (should be included in valid_actions)
        if best_action is None:
            best_action = self.map_action_to_index(0, 0)
            # print("No good action found, defaulting to no-op action (0, 0)")
        else:
            flight_action, aircraft_action = self.map_index_to_action(best_action)
            # print(f"\nChosen best action: index={best_action} (flight={flight_action}, aircraft={aircraft_action}) with reward={best_score}")
        
        return best_action

    def update_solution(self, info):
        """Update the solution dictionary based on the action taken"""
        # Update cancellations
        if 'cancelled_flights_count' in info and info['cancelled_flights_count'] > 0:
            self.solution['cancellations'] = list(self.cancelled_flights)
        
        # Update assignments
        if 'flight_action' in info and 'aircraft_action' in info:
            flight_id = info['flight_action']
            aircraft_idx = info['aircraft_action']
            if flight_id != 0 and aircraft_idx != 0:
                aircraft_id = self.aircraft_ids[aircraft_idx - 1]
                if aircraft_id != self.rotations_dict[flight_id]['Aircraft']:
                    self.solution['assignments'][flight_id] = aircraft_id
        
        # Update delays
        if self.environment_delayed_flights:
            self.solution['delays'] = {k: v for k, v in self.environment_delayed_flights.items()}
            self.solution['total_delay_minutes'] = sum(self.environment_delayed_flights.values())

class AircraftDisruptionConflicted(AircraftDisruptionEnv):
    """
    A specialized version of AircraftDisruptionEnv that restricts the action mask
    to only allow actions on flights that are currently in conflict.
    
    This is used during exploration to guide the agent toward relevant actions
    (conflicted flights) rather than completely random exploration. During training,
    50% of exploration uses this restricted mask, while the other 50% uses the full mask.
    """
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict, env_type=None):
        # Use 'proactive' as default env_type if not specified, as it allows seeing all information
        # The actual env_type doesn't matter much since we only use this for get_action_mask()
        if env_type is None:
            env_type = 'proactive'
        super().__init__(aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict, env_type=env_type)
    
    def get_action_mask(self):
        """
        Override the parent method to restrict the action mask to only allow actions
        that move conflicted flights away from their conflict aircraft.
        
        Logic:
        1. Get all current conflicts (flight_id, aircraft_id pairs)
        2. For each conflicted flight, allow it to be paired with ALL aircraft actions
           EXCEPT the aircraft it's currently in conflict with (moving it away resolves the conflict)
        3. Also allow no-action (0,0)
        4. Apply reactive environment restrictions if applicable
        
        Returns:
            np.ndarray: Action mask where conflicted flights can be moved to any aircraft except their conflict aircraft
        """
        # Initialize action mask with zeros (like parent does)
        action_mask = np.zeros(self.action_space.n, dtype=np.uint8)
        
        # Get current conflicts
        if DEBUG_MODE_REWARD:
            print(f"current conflicts in get_action_mask:")
        current_conflicts = self.get_current_conflicts()
        
        # Build a mapping: flight_id -> set of aircraft_ids it's in conflict with
        flight_to_conflict_aircraft = {}
        for conflict in current_conflicts:
            # conflict is a tuple: (aircraft_id, flight_id, flight_dep, flight_arr)
            if len(conflict) >= 2:
                aircraft_id = conflict[0]
                flight_id = conflict[1]
                if flight_id not in flight_to_conflict_aircraft:
                    flight_to_conflict_aircraft[flight_id] = set()
                flight_to_conflict_aircraft[flight_id].add(aircraft_id)
        
        # Get valid flight and aircraft actions
        valid_flight_actions = self.get_valid_flight_actions()
        valid_aircraft_actions = self.get_valid_aircraft_actions()
        
        # Always allow no-action (0,0)
        no_action_index = self.map_action_to_index(0, 0)
        no_action_index = int(no_action_index)
        if no_action_index < self.action_space.n:
            action_mask[no_action_index] = 1
        
        # If no conflicts exist, return mask with only no-action
        # Rationale: The conflicted environment is specifically for conflict-guided exploration.
        # When there are no conflicts, the agent should wait (no-action) rather than take random actions.
        # The training code uses this 50% of exploration time, with the other 50% using full random exploration.
        # This ensures we only guide exploration when conflicts actually exist.
        if not flight_to_conflict_aircraft:
            return action_mask
        
        # For each conflicted flight, allow actions with aircraft that are NOT in conflict
        for flight_id in flight_to_conflict_aircraft.keys():
            if flight_id not in valid_flight_actions:
                continue
            if flight_id in self.cancelled_flights:
                continue
            
            conflict_aircraft_ids = flight_to_conflict_aircraft[flight_id]
            
            conflict_aircraft_indices = set()
            for conflict_aircraft_id in conflict_aircraft_ids:
                # Find the index of this aircraft in aircraft_ids
                if conflict_aircraft_id in self.aircraft_id_to_idx:
                    aircraft_idx = self.aircraft_id_to_idx[conflict_aircraft_id]
                    # aircraft_action = aircraft_idx + 1 (since 0 is cancel)
                    conflict_aircraft_indices.add(aircraft_idx + 1)
            
            for aircraft_action in valid_aircraft_actions:
                if aircraft_action in conflict_aircraft_indices:
                    continue
                
                if aircraft_action == 0:
                    index = self.map_action_to_index(flight_id, aircraft_action)
                    index = int(index)
                    if index < self.action_space.n:
                        action_mask[index] = 1
                    continue

                index = self.map_action_to_index(flight_id, aircraft_action)
                index = int(index)
                if index < self.action_space.n:
                    action_mask[index] = 1
        
        # Apply reactive environment restrictions if applicable (from parent logic)
        if self.env_type == 'reactive':
            reactive_allowed_to_take_action = False
            current_time_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60

            earliest_disrupted_dep = float('inf')
            earliest_disruption_start = float('inf')

            for idx, aircraft_id in enumerate(self.aircraft_ids):
                breakdown_prob = self.unavailabilities_dict[aircraft_id]['Probability']
                if breakdown_prob == 1.0:
                    unavail_start = self.unavailabilities_dict[aircraft_id]['StartTime']
                    unavail_end = self.unavailabilities_dict[aircraft_id]['EndTime']
                    if not np.isnan(unavail_start):
                        earliest_disruption_start = min(earliest_disruption_start, unavail_start)
                        
                        if current_time_minutes >= unavail_start and current_time_minutes <= unavail_end:
                            reactive_allowed_to_take_action = True
                            break
                            
                        for j in range(4, self.columns_state_space - 2, 3):
                            flight_id = self.state[idx + 1, j]
                            dep_time = self.state[idx + 1, j + 1]
                            arr_time = self.state[idx + 1, j + 2]
                            if not np.isnan(flight_id) and not np.isnan(dep_time):
                                if dep_time < unavail_end and arr_time > unavail_start:
                                    earliest_disrupted_dep = min(earliest_disrupted_dep, dep_time)

            current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
            earliest_critical_time = min(earliest_disrupted_dep, earliest_disruption_start)
            
            if current_time_minutes + self.timestep_minutes >= earliest_critical_time:
                reactive_allowed_to_take_action = True
            
            if not reactive_allowed_to_take_action:
                action_mask[:] = 0
                action_mask[0] = 1  
        
        return action_mask
