"""
Model 3 (SSF Large Dimensions) Environment
State Space Structure:
- ac_mtx: 3 rows × 96 columns = 288 elements (aircraft unavailability matrix)
- fl_mtx: Variable rows × 97 columns (flight schedule matrix)
  - Default: MAX_AIRCRAFT × MAX_FLIGHTS_PER_AIRCRAFT rows (3 × 17 = 51)
  - Optimized: Can be set via max_flights_total parameter (scans training data to find actual max)
  - 97 = 96 time intervals + 1 aircraft index column
- Total: 288 + (max_flights_total × 97) dimensions

OPTIMIZATION:
To reduce state space size, scan your training data first:
  from scripts.utils_ssf import find_max_flights_in_training_data
  max_flights = find_max_flights_in_training_data("Data/TRAINING/3ac-182-green16/")
  env = AircraftDisruptionEnv(..., max_flights_total=max_flights)

This can significantly reduce the state space if actual max flights < MAX_AIRCRAFT × MAX_FLIGHTS_PER_AIRCRAFT.

This is based on environment_ssf.py but uses the Model 3 state space formulation.
"""

import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
from src.config_ssf import *
from scripts.utils_ssf_large_dimensions import *
import time
import random
import math
from typing import Dict, List, Tuple
import os

class AircraftDisruptionEnv(gym.Env):
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict, env_type, max_flights_total=None):
        """Initializes the AircraftDisruptionEnv class with Model 2 state space.

        Args:
            aircraft_dict (dict): Dictionary containing aircraft information.
            flights_dict (dict): Dictionary containing flight information.
            rotations_dict (dict): Dictionary containing rotation information.
            alt_aircraft_dict (dict): Dictionary containing alternative aircraft information.
            config_dict (dict): Dictionary containing configuration information.
            env_type (str): Type of environment ('myopic' or 'proactive', 'reactive', 'drl-greedy').
            max_flights_total (int, optional): Maximum total flights across all scenarios. 
                If None, uses MAX_AIRCRAFT × MAX_FLIGHTS_PER_AIRCRAFT (default).
                If provided, uses this value to optimize fl_mtx size.
        """
        super(AircraftDisruptionEnv, self).__init__()
        
        # Store the environment type
        self.env_type = env_type  
        
        # Constants for environment configuration
        self.max_aircraft = MAX_AIRCRAFT
        self.max_flights_per_aircraft = MAX_FLIGHTS_PER_AIRCRAFT
        self.max_time_intervals = MAX_TIME_INTERVALS  # 96 intervals (24 hours / 15 minutes)
        
        # Use dynamic max_flights_total if provided, otherwise use default
        # Note: Training script (train_dqn_modular_ssf_large_dimensions.py) already handles the fallback,
        # so this is mainly for defensive programming if environment is created directly
        if max_flights_total is not None:
            self.max_flights_total = max_flights_total
        else:
            # Fallback: use default if not provided (training script should always provide this)
            self.max_flights_total = MAX_AIRCRAFT * MAX_FLIGHTS_PER_AIRCRAFT
        
        self.config_dict = config_dict

        # Define the recovery period
        start_date = config_dict['RecoveryPeriod']['StartDate']
        start_time = config_dict['RecoveryPeriod']['StartTime']
        end_date = config_dict['RecoveryPeriod']['EndDate']
        end_time = config_dict['RecoveryPeriod']['EndTime']
        self.start_datetime = datetime.strptime(f"{start_date} {start_time}", '%d/%m/%y %H:%M')
        self.end_datetime = datetime.strptime(f"{end_date} {end_time}", '%d/%m/%y %H:%M')
        self.timestep = timedelta(hours=TIMESTEP_HOURS)
        self.timestep_minutes = TIMESTEP_HOURS * 60

        # Aircraft information and indexing
        self.aircraft_ids = list(aircraft_dict.keys())
        self.aircraft_id_to_idx = {aircraft_id: idx for idx, aircraft_id in enumerate(self.aircraft_ids)}
        self.conflicted_flights = {}

        # Flight information and indexing
        if flights_dict is None:
            flights_dict = {}
        if flights_dict:
            self.flight_ids = list(flights_dict.keys())
            self.flight_id_to_idx = {flight_id: idx for idx, flight_id in enumerate(self.flight_ids)}
        else:
            self.flight_ids = []
            self.flight_id_to_idx = {}

        # Filter out flights with '+' in DepTime (next day flights)
        this_day_flights = [flight_info for flight_info in flights_dict.values() if '+' not in flight_info['DepTime']]

        # Determine the earliest possible event
        self.earliest_datetime = min(
            min(datetime.strptime(config_dict['RecoveryPeriod']['StartDate'] + ' ' + flight_info['DepTime'], '%d/%m/%y %H:%M') for flight_info in this_day_flights),
            self.start_datetime
        )

        # Define observation and action spaces - MODEL 2 STRUCTURE
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        
        # Model 2 state space dimensions:
        # ac_mtx: MAX_AIRCRAFT × MAX_TIME_INTERVALS = 3 × 96 = 288
        # fl_mtx: max_flights_total × (MAX_TIME_INTERVALS + 1)
        #   +1 in fl_mtx columns for aircraft index (column 0)
        ac_mtx_size = self.max_aircraft * self.max_time_intervals  # 3 × 96 = 288
        fl_mtx_size = self.max_flights_total * (self.max_time_intervals + 1)  # Dynamic based on actual max flights
        total_observation_size = ac_mtx_size + fl_mtx_size
        
        self.observation_space = spaces.Dict({
            'state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(total_observation_size,),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(self.action_space.n,),
                dtype=np.uint8
            )
        })

        # Store the dictionaries
        self.alt_aircraft_dict = alt_aircraft_dict
        self.rotations_dict = rotations_dict
        self.flights_dict = flights_dict
        self.aircraft_dict = aircraft_dict

        # Deep copies for reset
        self.initial_aircraft_dict = copy.deepcopy(aircraft_dict)
        self.initial_flights_dict = copy.deepcopy(flights_dict)
        self.initial_rotations_dict = copy.deepcopy(rotations_dict)
        self.initial_alt_aircraft_dict = copy.deepcopy(alt_aircraft_dict)

        # Track environment state
        self.environment_delayed_flights = {}
        self.penalized_delays = {}
        self.penalized_cancelled_flights = set()
        self.cancelled_flights = set()
        self.penalized_conflicts = set()
        self.resolved_conflicts = set()
        self.swapped_flights = []  # List of (flight_id, new_aircraft_id) tuples
        self.automatically_cancelled_flights = set()
        self.penalized_automatically_cancelled_flights = set()

        # Initialize breakdown tracking
        self.uncertain_breakdowns = {}
        self.current_breakdowns = {}
        self.unavailabilities_dict = {}
        self.info_after_step = {}
        
        # Create mapping from flight_id to fl_mtx row index
        self.flight_id_to_fl_mtx_row = {}
        self.fl_mtx_row_to_flight_id = {}
        self.ac_mtx_idx_to_aircraft_id = {}
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx < self.max_aircraft:
                self.ac_mtx_idx_to_aircraft_id[idx] = aircraft_id

        # Performance optimization
        self.something_happened = False
        self._cached_state = None
        self._total_calc_time = 0.0
        self._step_count = 0

        # Initialize environment state
        self.current_datetime = self.start_datetime
        self.state = self._get_initial_state()
        self.current_ac_mtx, self.current_fl_mtx = self.state

        # Initialize scenario tracking
        self.eligible_flights_for_resolved_bonus = self.get_initial_conflicts()
        self.scenario_wide_delay_minutes = 0
        self.scenario_wide_delay_count = 0
        self.scenario_wide_cancelled_flights = 0
        self.scenario_wide_automatically_cancelled_count = 0
        self.scenario_wide_steps = 0
        self.scenario_wide_resolved_conflicts = 0
        self.scenario_wide_resolved_initial_conflicts = 0
        self.scenario_wide_disruption_resolved_to_zero_count = 0
        self.scenario_wide_tail_swaps = 0
        self.scenario_wide_tail_swaps_resolving = 0
        self.scenario_wide_tail_swaps_inefficient = 0
        self.scenario_wide_inaction_count = 0
        self.scenario_wide_initial_disrupted_flights_list = self.get_current_conflicts()
        self.scenario_wide_actual_disrupted_flights = len(self.get_current_conflicts())

        self.scenario_wide_reward_components = {
            "delay_penalty_total": 0,
            "cancel_penalty": 0,
            "inaction_penalty": 0,
            "proactive_penalty": 0,
            "time_penalty": 0,
            "final_conflict_resolution_reward": 0,
            "automatic_cancellation_penalty": 0,
            "probability_resolution_bonus": 0,
            "low_confidence_action_penalty": 0
        }

        self.tail_swap_happened = False

    def _get_initial_state(self):
        """Initializes the state matrices for Model 2: ac_mtx and fl_mtx.
        
        Returns:
            tuple: (ac_mtx, fl_mtx)
            - ac_mtx: (MAX_AIRCRAFT, MAX_TIME_INTERVALS) - aircraft unavailability matrix
            - fl_mtx: (MAX_AIRCRAFT × MAX_FLIGHTS_PER_AIRCRAFT, MAX_TIME_INTERVALS + 1) - flight schedule matrix
              Column 0: Aircraft index (0-2) or -1 if flight not assigned
              Columns 1-96: Time intervals (1 = flight active, 0 = not active)
        """
        # Simple caching
        if hasattr(self, '_cached_state') and self._cached_state is not None and not self.something_happened:
            return self._cached_state
        
        start_time = time.time()
        
        # Calculate current time and remaining recovery period in minutes
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
        
        # List to keep track of flights to remove from dictionaries
        flights_to_remove = set()
        
        # Set to collect actual flights in state space
        active_flights = set()
        
        # Calculate 15-minute intervals
        interval_minutes = 15
        total_minutes = int((self.end_datetime - self.start_datetime).total_seconds() // 60)
        num_intervals = min(math.ceil(total_minutes / interval_minutes), self.max_time_intervals)
        interval_starts = [self.start_datetime + timedelta(minutes=i*interval_minutes) for i in range(num_intervals)]
        interval_lookup = {dt: idx for idx, dt in enumerate(interval_starts)}

        # 1. Create ac_mtx: (MAX_AIRCRAFT, MAX_TIME_INTERVALS)
        ac_mtx = np.zeros((self.max_aircraft, num_intervals), dtype=np.float32)
        
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            # Check for unavailabilities
            # Priority: Use unavailabilities_dict if it exists (updated by process_uncertainties)
            # Otherwise fall back to alt_aircraft_dict or uncertain_breakdowns
            breakdown_probability = 0.0
            unavail_start_minutes = np.nan
            unavail_end_minutes = np.nan
            
            if aircraft_id in self.unavailabilities_dict and self.unavailabilities_dict[aircraft_id].get('StartTime') is not None:
                # Use updated probability from unavailabilities_dict (source of truth after process_uncertainties)
                unavail_info = self.unavailabilities_dict[aircraft_id]
                breakdown_probability = unavail_info['Probability']
                unavail_start_time = unavail_info['StartTime']
                unavail_end_time = unavail_info['EndTime']
                unavail_start_minutes = (unavail_start_time - self.earliest_datetime).total_seconds() / 60
                unavail_end_minutes = (unavail_end_time - self.earliest_datetime).total_seconds() / 60
            elif aircraft_id in self.alt_aircraft_dict:
                unavails = self.alt_aircraft_dict[aircraft_id]
                if not isinstance(unavails, list):
                    unavails = [unavails]
                breakdown_probability = unavails[0].get('Probability', 1.0)
                
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
            elif aircraft_id in self.uncertain_breakdowns:
                breakdown_info = self.uncertain_breakdowns[aircraft_id][0]
                breakdown_probability = breakdown_info['Probability']
                unavail_start_minutes = (breakdown_info['StartTime'] - self.earliest_datetime).total_seconds() / 60
                unavail_end_minutes = (breakdown_info['EndTime'] - self.earliest_datetime).total_seconds() / 60

            # Fill ac_mtx
            if not np.isnan(breakdown_probability) and breakdown_probability > 0:
                start_idx = max(0, int(unavail_start_minutes // interval_minutes))
                end_idx = min(num_intervals, int(math.ceil(unavail_end_minutes / interval_minutes)))
                ac_mtx[idx, start_idx:end_idx] = breakdown_probability
                
                self.unavailabilities_dict[aircraft_id] = {
                    'Probability': breakdown_probability,
                    'StartColumn': start_idx,
                    'EndColumn': end_idx,
                    'StartTime': self.earliest_datetime + timedelta(minutes=unavail_start_minutes),
                    'EndTime': self.earliest_datetime + timedelta(minutes=unavail_end_minutes)
                }
            else:
                self.unavailabilities_dict[aircraft_id] = {
                    'Probability': breakdown_probability,
                    'StartColumn': -1,
                    'EndColumn': -1,
                    'StartTime': None,
                    'EndTime': None
                }

        # 2. Create fl_mtx: (max_flights_total, MAX_TIME_INTERVALS + 1)
        # Structure: Each row = one flight, Column 0 = aircraft index, Columns 1-96 = time intervals
        # Use self.max_flights_total (which may be optimized based on actual training data)
        fl_mtx = np.zeros((self.max_flights_total, self.max_time_intervals + 1), dtype=np.float32)
        fl_mtx[:, 0] = -1  # Initialize aircraft index column to -1 (unassigned)
        
        # Reset mappings
        self.flight_id_to_fl_mtx_row = {}
        self.fl_mtx_row_to_flight_id = {}
        
        # Track which flight slot we're filling
        flight_slot = 0
        
        # Organize flights by aircraft
        flights_by_aircraft = {ac_idx: [] for ac_idx in range(self.max_aircraft)}
        for flight_id, rotation_info in self.rotations_dict.items():
            if flight_id in self.flights_dict:
                aircraft_id = rotation_info['Aircraft']
                if aircraft_id in self.aircraft_id_to_idx:
                    ac_idx = self.aircraft_id_to_idx[aircraft_id]
                    if ac_idx < self.max_aircraft:
                        flights_by_aircraft[ac_idx].append((flight_id, rotation_info))
        
        # Fill fl_mtx
        for ac_idx in range(self.max_aircraft):
            aircraft_flights = flights_by_aircraft[ac_idx]
            breakdown_probability = self.unavailabilities_dict[self.aircraft_ids[ac_idx]]['Probability']
            unavail_start_col = self.unavailabilities_dict[self.aircraft_ids[ac_idx]]['StartColumn']
            unavail_end_col = self.unavailabilities_dict[self.aircraft_ids[ac_idx]]['EndColumn']
            
            for flight_idx, (flight_id, rotation_info) in enumerate(aircraft_flights):
                if flight_idx >= self.max_flights_per_aircraft:
                    break
                if flight_slot >= self.max_flights_total:
                    break
                
                flight_info = self.flights_dict[flight_id]
                dep_time = parse_time_with_day_offset(flight_info['DepTime'], self.start_datetime)
                arr_time = parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime)
                
                dep_time_minutes = (dep_time - self.earliest_datetime).total_seconds() / 60
                arr_time_minutes = (arr_time - self.earliest_datetime).total_seconds() / 60
                
                # Exclude flights that have already departed and are in conflict (matching environment_ssf.py logic)
                if dep_time_minutes < current_time_minutes:
                    # Flight has already departed
                    if breakdown_probability == 1.00 and unavail_start_col >= 0 and unavail_end_col >= 0:
                        # There is an unavailability with prob == 1.00
                        # Check if the flight overlaps with the unavailability using column indices
                        flight_start_col = max(0, int((dep_time - self.start_datetime).total_seconds() // 60 // interval_minutes))
                        flight_end_col = min(num_intervals, int(math.ceil((arr_time - self.start_datetime).total_seconds() / 60 / interval_minutes)))
                        
                        # Check for overlap using column indices
                        if flight_start_col < unavail_end_col and flight_end_col > unavail_start_col:
                            if DEBUG_MODE_CANCELLED_FLIGHT:
                                print(f"REMOVING FLIGHT {flight_id} DUE TO UNAVAILABILITY AND PAST DEPARTURE")
                            # Flight is in conflict with unavailability
                            flights_to_remove.add(flight_id)
                            continue
                
                # Clamp to recovery period
                s = max(dep_time, self.start_datetime)
                e = min(arr_time, self.end_datetime)
                
                # Convert to interval indices
                dep_interval = max(0, int((s - self.start_datetime).total_seconds() // 60 // interval_minutes))
                arr_interval = min(num_intervals - 1, int(math.ceil((e - self.start_datetime).total_seconds() / 60 / interval_minutes)))
                
                # Check if flight is cancelled
                is_cancelled = flight_id in self.cancelled_flights
                
                # Column 0: Aircraft index
                fl_mtx[flight_slot, 0] = float(ac_idx) if not is_cancelled else -1.0
                
                # Columns 1-96: Time intervals (1 = flight active, 0 = not active)
                if not is_cancelled:
                    # Mark intervals where flight is active
                    for interval_idx in range(dep_interval, arr_interval + 1):
                        if 0 <= interval_idx < self.max_time_intervals:
                            fl_mtx[flight_slot, interval_idx + 1] = 1.0
                    
                    # Store mappings
                    self.flight_id_to_fl_mtx_row[flight_id] = flight_slot
                    self.fl_mtx_row_to_flight_id[flight_slot] = flight_id
                    active_flights.add(flight_id)
                
                flight_slot += 1

        # Update flight_id_to_idx with only the active flights
        self.flight_id_to_idx = {
            flight_id: idx for idx, flight_id in enumerate(sorted(active_flights))
        }
        
        # Remove past flights from dictionaries
        for flight_id in flights_to_remove:
            self.remove_flight(flight_id)
        
        # Pad matrices to fixed size for consistent observation space
        # Use self.max_aircraft and self.max_time_intervals to match observation space definition
        MAX_AIRCRAFT_FIXED = self.max_aircraft  # Use 3, not hardcoded 6
        MAX_TIME_INTERVALS_FIXED = self.max_time_intervals  # 96
        # Use self.max_flights_total for padding (already optimized if max_flights_total was provided)
        MAX_FLIGHTS_TOTAL_FIXED = self.max_flights_total
        
        # Pad ac_mtx to fixed size
        padded_ac_mtx = np.zeros((MAX_AIRCRAFT_FIXED, MAX_TIME_INTERVALS_FIXED), dtype=np.float32)
        padded_ac_mtx[:ac_mtx.shape[0], :min(ac_mtx.shape[1], MAX_TIME_INTERVALS_FIXED)] = ac_mtx[:, :min(ac_mtx.shape[1], MAX_TIME_INTERVALS_FIXED)]
        
        # Pad fl_mtx to fixed size
        padded_fl_mtx = np.zeros((MAX_FLIGHTS_TOTAL_FIXED, MAX_TIME_INTERVALS_FIXED + 1), dtype=np.float32)
        padded_fl_mtx[:, 0] = -1  # Initialize aircraft index column
        padded_fl_mtx[:fl_mtx.shape[0], :min(fl_mtx.shape[1], MAX_TIME_INTERVALS_FIXED + 1)] = fl_mtx[:, :min(fl_mtx.shape[1], MAX_TIME_INTERVALS_FIXED + 1)]

        # Cache the result
        self._cached_state = (padded_ac_mtx, padded_fl_mtx)
        
        # Performance tracking
        self._step_count += 1
        elapsed_time = time.time() - start_time
        self._total_calc_time += elapsed_time
        
        if self._step_count % 1000 == 0:
            avg_time = self._total_calc_time / self._step_count if self._step_count > 0 else 0
            print(f"Model 2 state calculation: {elapsed_time:.4f}s, Avg: {avg_time:.4f}s (step {self._step_count})")
        
        return padded_ac_mtx, padded_fl_mtx
    
    def get_all_flights_for_aircraft(self, ac_idx, fl_mtx):
        """Gets all flights for a specific aircraft from fl_mtx.
        
        Args:
            ac_idx (int): Aircraft index
            fl_mtx: Flight matrix
            
        Returns:
            list: List of tuples (flight_id, dep_interval, arr_interval, status)
        """
        flights = []
        for flight_row in range(fl_mtx.shape[0]):
            if fl_mtx[flight_row, 0] == ac_idx:  # Flight assigned to this aircraft
                flight_id = self.fl_mtx_row_to_flight_id.get(flight_row)
                if flight_id is None:
                    continue
                
                # Get flight active intervals
                flight_active = fl_mtx[flight_row, 1:]  # Columns 1-96
                flight_indices = np.where(flight_active > 0)[0]
                
                if len(flight_indices) > 0:
                    dep_interval = flight_indices[0]
                    arr_interval = flight_indices[-1]
                    status = 1.0 if flight_id not in self.cancelled_flights else 0.0
                    flights.append((flight_id, dep_interval, arr_interval, status))
        
        return flights
    
    def get_flight_data(self, flight_id, fl_mtx):
        """Gets flight data from fl_mtx for a specific flight.
        
        Args:
            flight_id (int): Flight ID
            fl_mtx: Flight matrix
            
        Returns:
            tuple: (ac_idx, flight_row, flight_id, dep_interval, arr_interval, status) or None
        """
        flight_row = self.flight_id_to_fl_mtx_row.get(flight_id)
        if flight_row is None or flight_row >= fl_mtx.shape[0]:
            return None
        
        ac_idx = int(fl_mtx[flight_row, 0])
        if ac_idx < 0:
            return None
        
        # Get flight active intervals
        flight_active = fl_mtx[flight_row, 1:]  # Columns 1-96
        flight_indices = np.where(flight_active > 0)[0]
        
        if len(flight_indices) == 0:
            return None
        
        dep_interval = flight_indices[0]
        arr_interval = flight_indices[-1]
        status = 1.0 if flight_id not in self.cancelled_flights else 0.0
        
        return (ac_idx, flight_row, flight_id, dep_interval, arr_interval, status)

    def get_observation(self):
        """Returns the observation dictionary with state vector and action mask.
        Matches process_observation logic from environment_ssf.py.
        
        Returns:
            tuple: (observation_dict, state_matrices)
            - observation_dict: {'state': flattened_state_vector, 'action_mask': action_mask}
            - state_matrices: (ac_mtx, fl_mtx) for internal use
        """
        ac_mtx, fl_mtx = self.get_current_state()
        return self.process_observation(ac_mtx, fl_mtx)
    
    def process_observation(self, ac_mtx, fl_mtx):
        """Processes the observation by applying env_type-specific masking.
        Does NOT modify internal state or unavailabilities_dict.
        Returns an observation suitable for the agent.
        Matches environment_ssf.py logic but adapted for fl_mtx structure."""
        ac_mtx_to_observe = ac_mtx.copy()
        fl_mtx_to_observe = fl_mtx.copy()

        # Apply env_type-specific masking to ac_mtx (aircraft unavailabilities)
        for ac_idx in range(ac_mtx_to_observe.shape[0]):
            unavail_periods = ac_mtx_to_observe[ac_idx, :]
            unavail_indices = np.where(unavail_periods > 0)[0]
            
            if len(unavail_indices) > 0:
                # Get the real probability values
                real_probabilities = unavail_periods[unavail_indices]
                
                # Apply env_type logic to observed values ONLY
                if self.env_type == 'reactive':
                    # Reactive sees no info about disruptions - mask all unavailabilities
                    ac_mtx_to_observe[ac_idx, :] = 0.0
                elif self.env_type == 'myopic':
                    # Myopic only sees if probability == 1.0
                    for idx, prob in zip(unavail_indices, real_probabilities):
                        if prob != 1.0:
                                ac_mtx_to_observe[ac_idx, idx] = 0.0
                elif self.env_type == 'proactive' or self.env_type == 'drl-greedy':
        # Proactive and drl-greedy see everything (no masking needed)
                    pass

        # For fl_mtx, we don't need masking (values are already in the matrix)
        # Just flatten both matrices
        fl_mtx_flat = fl_mtx_to_observe.flatten()
        ac_mtx_flat = ac_mtx_to_observe.flatten()

        # Combine the matrices
        state_flat = np.concatenate([ac_mtx_flat, fl_mtx_flat])

        # Use get_action_mask to generate the action mask
        action_mask = self.get_action_mask()
        
        # Return the observation dictionary without modifying internal structures
        obs_with_mask = {
            'state': state_flat.astype(np.float32),
            'action_mask': action_mask
        }
        return obs_with_mask, (ac_mtx_to_observe, fl_mtx_to_observe)

    def get_current_state(self):
        """Returns the current state matrices without recreating them."""
        if hasattr(self, '_cached_state') and self._cached_state is not None and not self.something_happened:
            return self._cached_state
            
        if self.current_ac_mtx is None or self.current_fl_mtx is None:
            self.current_ac_mtx, self.current_fl_mtx = self._get_initial_state()
            self.state = (self.current_ac_mtx, self.current_fl_mtx)
        
        return self.current_ac_mtx, self.current_fl_mtx

    def get_valid_aircraft_actions(self):
        """Generates a list of valid aircraft actions for the agent.
        Returns: list: A list of valid aircraft actions that the agent can take.
        """
        return list(range(len(self.aircraft_ids) + 1))  # 0 to len(aircraft_ids)

    def get_valid_flight_actions(self):
        """Generates a list of valid flight actions based on flights in state space using fl_mtx approach."""
        # Calculate current time in minutes from start_datetime
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
        
        # Calculate which 15-minute interval corresponds to current time
        interval_minutes = 15
        current_interval = int(current_time_minutes // interval_minutes)
        
        # Get the current state matrices
        ac_mtx, fl_mtx = self._get_initial_state()
        
        # Get all valid flight IDs from fl_mtx
        valid_flight_ids = set()
        
        # Check each flight row in fl_mtx
        for flight_row in range(fl_mtx.shape[0]):
            flight_id = self.fl_mtx_row_to_flight_id.get(flight_row)
            if flight_id is None:
                continue
            
            # Check if flight is not cancelled
            if flight_id in self.cancelled_flights:
                continue
            
            # Get flight active intervals
            flight_active = fl_mtx[flight_row, 1:]  # Columns 1-96
            flight_indices = np.where(flight_active > 0)[0]
            
            if len(flight_indices) == 0:
                continue
            
            dep_interval = flight_indices[0]
            
            # Check if flight hasn't departed yet (departure interval >= current interval)
            if dep_interval >= current_interval:
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

    def get_action_mask(self):
        """Creates a binary vector action_mask for valid flight & aircraft pairs using fl_mtx approach."""
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

        # For reactive environment, only allow 0,0 action unless an immediate or imminent conflict with prob==1.00 exists
        if self.env_type == 'reactive':
            reactive_allowed_to_take_action = False
            current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
            interval_minutes = 15
            current_interval = int(current_time_minutes // interval_minutes)

            # Get current state matrices
            ac_mtx, fl_mtx = self._get_initial_state()

            # Find earliest disrupted flight departure time and earliest disruption start time
            earliest_disrupted_dep = float('inf')
            earliest_disruption_start = float('inf')

            for ac_idx in range(ac_mtx.shape[0]):
                # Check if this aircraft has unavailability with prob == 1.0
                unavail_periods = ac_mtx[ac_idx, :]
                unavail_indices = np.where(unavail_periods == 1.0)[0]
                
                if len(unavail_indices) > 0:
                    # Find start and end of unavailability periods
                    unavail_starts = []
                    unavail_ends = []
                    
                    # Group consecutive indices into periods
                    start = unavail_indices[0]
                    end = start
                    for i in range(1, len(unavail_indices)):
                        if unavail_indices[i] == end + 1:
                            end = unavail_indices[i]
                        else:
                            unavail_starts.append(start)
                            unavail_ends.append(end)
                            start = unavail_indices[i]
                            end = start
                    unavail_starts.append(start)
                    unavail_ends.append(end)
                    
                    for unavail_start, unavail_end in zip(unavail_starts, unavail_ends):
                        # Convert column indices back to minutes for comparison
                        unavail_start_minutes = unavail_start * interval_minutes
                        unavail_end_minutes = unavail_end * interval_minutes
                        
                        earliest_disruption_start = min(earliest_disruption_start, unavail_start_minutes)
                        
                        # Check if current time is inside disruption period
                        if current_time_minutes >= unavail_start_minutes and current_time_minutes <= unavail_end_minutes:
                            reactive_allowed_to_take_action = True
                            break
                            
                        # Check flights assigned to this aircraft for departures during disruption
                        # Get flights for this aircraft using fl_mtx
                        aircraft_flights = self.get_all_flights_for_aircraft(ac_idx, fl_mtx)
                        for existing_flight_id, existing_dep_interval, existing_arr_interval, existing_status in aircraft_flights:
                            if existing_status <= 0:  # Skip cancelled flights
                                continue
                            
                            # Convert intervals to minutes
                            existing_flight_start_minutes = existing_dep_interval * interval_minutes
                            existing_flight_end_minutes = existing_arr_interval * interval_minutes
                            
                            # Check if flight departs during disruption
                            if existing_flight_start_minutes >= unavail_start_minutes and existing_flight_start_minutes <= unavail_end_minutes:
                                reactive_allowed_to_take_action = True
                                earliest_disrupted_dep = min(earliest_disrupted_dep, existing_flight_start_minutes)
                                break

            # Allow reactive action if approaching either critical time
            earliest_critical_time = min(earliest_disrupted_dep, earliest_disruption_start)
            
            if current_time_minutes + self.timestep_minutes >= earliest_critical_time:
                reactive_allowed_to_take_action = True
                
            if not reactive_allowed_to_take_action:
                # Reset mask to all zeros except for 0,0 action
                action_mask[:] = 0
                action_mask[0] = 1  # Only allow 0,0 action
        
        return action_mask
    
    def map_action_to_index(self, flight_action, aircraft_action):
        """Maps the (flight, aircraft) action pair to a single index in the flattened action space.

        Args:
            flight_action (int): The index of the flight action.
            aircraft_action (int): The index of the aircraft action.

        Returns:
            int: The corresponding index in the flattened action space.
        """
        return flight_action * (len(self.aircraft_ids) + 1) + aircraft_action
    
    def evaluate_action_impact(self, flight_action, aircraft_action):
        """
        Evaluates the impact of an action to determine if it's logical.
        
        SIMPLIFIED VERSION: Allows all actions to prevent over-restriction.
        The agent should learn which actions are better through rewards.
        
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
            
        # SIMPLIFIED: Allow all actions to prevent over-restriction
        # The agent should learn which actions are better through rewards
        if DEBUG_MODE_ACTION_EVALUATION:
            print(f"Action evaluation: Allowing action (flight={flight_action}, aircraft={aircraft_action})")
        return True

    def get_initial_conflicts(self):
        """Retrieves the initial conflicts in the environment.

        This function checks for conflicts between flights and unavailability periods,
        considering unavailabilities with probability greater than 0.0.

        Returns:
            set: A set of conflicts currently present in the initial state of the environment.
        """
        initial_conflicts = set()
        
        # Get the current state matrices
        ac_mtx, fl_mtx = self._get_initial_state()
        
        # For each aircraft (row in ac_mtx)
        for ac_idx in range(ac_mtx.shape[0]):
            # Get unavailability periods for this aircraft (non-zero values)
            unavail_periods = ac_mtx[ac_idx, :]
            unavail_indices = np.where(unavail_periods > 0)[0]
            
            if len(unavail_indices) == 0:
                continue  # No unavailability for this aircraft
                
            # Find start and end of unavailability periods
            unavail_starts = []
            unavail_ends = []
            
            # Group consecutive indices into periods
            if len(unavail_indices) > 0:
                start = unavail_indices[0]
                end = start
                for i in range(1, len(unavail_indices)):
                    if unavail_indices[i] == end + 1:
                        end = unavail_indices[i]
                    else:
                        unavail_starts.append(start)
                        unavail_ends.append(end)
                        start = unavail_indices[i]
                        end = start
                unavail_starts.append(start)
                unavail_ends.append(end)
            
            # Check each flight for this aircraft using fl_mtx
            aircraft_flights = self.get_all_flights_for_aircraft(ac_idx, fl_mtx)
            for flight_id, dep_interval, arr_interval, status in aircraft_flights:
                if status <= 0:  # Skip cancelled flights
                    continue
                
                # Check for overlap with any unavailability period
                conflict_found = False
                for unavail_start, unavail_end in zip(unavail_starts, unavail_ends):
                    if (dep_interval <= unavail_end and arr_interval >= unavail_start):
                        # Conflict detected! Add to set
                        aircraft_id = self.ac_mtx_idx_to_aircraft_id.get(ac_idx)
                        if aircraft_id:
                            initial_conflicts.add((aircraft_id, flight_id))
                        conflict_found = True
                        break  # Found conflict for this flight, no need to check other unavailability periods
        
        return initial_conflicts

    def get_current_conflicts(self):
        """Retrieves the current conflicts in the environment using fl_mtx approach.

        This function checks for conflicts between flights and unavailability periods,
        considering unavailabilities with probability > 0.0.
        It excludes cancelled flights which are not considered conflicts.

        Returns:
            set: A set of conflicts currently present in the environment.
        """
        current_conflicts = set()
        
        # Get the current state matrices
        ac_mtx, fl_mtx = self._get_initial_state()
        
        # Calculate current time in minutes from start_datetime
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
        interval_minutes = 15
        current_interval = int(current_time_minutes // interval_minutes)
        
        # For each aircraft (row in ac_mtx)
        for ac_idx in range(ac_mtx.shape[0]):
            # Get unavailability periods for this aircraft (non-zero values)
            unavail_periods = ac_mtx[ac_idx, :]
            unavail_indices = np.where(unavail_periods > 0)[0]
            
            if len(unavail_indices) == 0:
                continue  # No unavailability for this aircraft
                
            # Find start and end of unavailability periods
            unavail_starts = []
            unavail_ends = []
            
            # Group consecutive indices into periods
            if len(unavail_indices) > 0:
                start = unavail_indices[0]
                end = start
                for i in range(1, len(unavail_indices)):
                    if unavail_indices[i] == end + 1:
                        end = unavail_indices[i]
                    else:
                        unavail_starts.append(start)
                        unavail_ends.append(end)
                        start = unavail_indices[i]
                        end = start
                unavail_starts.append(start)
                unavail_ends.append(end)
            
            # Check each flight for this aircraft using fl_mtx
            aircraft_flights = self.get_all_flights_for_aircraft(ac_idx, fl_mtx)
            for flight_id, dep_interval, arr_interval, status in aircraft_flights:
                # Skip cancelled flights
                if flight_id in self.cancelled_flights or status <= 0:
                    continue
                
                # Convert intervals to minutes for comparison with current time
                flight_start_minutes = dep_interval * interval_minutes
                flight_end_minutes = arr_interval * interval_minutes
                
                # Check if the flight's departure is in the past (relative to current time)
                if dep_interval < current_interval:
                    continue  # Skip past flights
                
                # Check for overlap with any unavailability period
                for unavail_start, unavail_end in zip(unavail_starts, unavail_ends):
                    # Convert unavailability periods to minutes
                    unavail_start_minutes = unavail_start * interval_minutes
                    unavail_end_minutes = unavail_end * interval_minutes
                    
                    if (flight_start_minutes < unavail_end_minutes and flight_end_minutes > unavail_start_minutes):
                        # Conflict detected! Add to set
                        conflict_identifier = (ac_idx, flight_id, flight_start_minutes, flight_end_minutes)
                        current_conflicts.add(conflict_identifier)
                        break  # Found conflict for this flight, no need to check other unavailability periods
        
        return current_conflicts
    
    def get_current_conflicts_with_prob_1(self):
        """Retrieves the current conflicts in the environment using fl_mtx approach.

        This function checks for conflicts between flights and unavailability periods,
        considering only unavailabilities with probability 1.0.
        It excludes cancelled flights which are not considered conflicts.

        Returns:
            set: A set of conflicts currently present in the environment.
        """
        current_conflicts_with_prob_1 = set()
        
        # Get the current state matrices
        ac_mtx, fl_mtx = self._get_initial_state()
        
        # Calculate current time in minutes from start_datetime
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
        interval_minutes = 15
        current_interval = int(current_time_minutes // interval_minutes)
        
        # For each aircraft (row in ac_mtx)
        for ac_idx in range(ac_mtx.shape[0]):
            # Get unavailability periods for this aircraft (only probability == 1.0)
            unavail_periods = ac_mtx[ac_idx, :]
            unavail_indices = np.where(unavail_periods == 1.0)[0]  # Only probability 1.0
            
            if len(unavail_indices) == 0:
                continue  # No unavailability with probability 1.0 for this aircraft
                
            # Find start and end of unavailability periods
            unavail_starts = []
            unavail_ends = []
            
            # Group consecutive indices into periods
            if len(unavail_indices) > 0:
                start = unavail_indices[0]
                end = start
                for i in range(1, len(unavail_indices)):
                    if unavail_indices[i] == end + 1:
                        end = unavail_indices[i]
                    else:
                        unavail_starts.append(start)
                        unavail_ends.append(end)
                        start = unavail_indices[i]
                        end = start
                unavail_starts.append(start)
                unavail_ends.append(end)
            
            # Check each flight for this aircraft using fl_mtx
            aircraft_flights = self.get_all_flights_for_aircraft(ac_idx, fl_mtx)
            for flight_id, dep_interval, arr_interval, status in aircraft_flights:
                # Skip cancelled flights
                if flight_id in self.cancelled_flights or status <= 0:
                    continue
                
                # Convert intervals to minutes for comparison with current time
                flight_start_minutes = dep_interval * interval_minutes
                flight_end_minutes = arr_interval * interval_minutes
                
                # Check if the flight's departure is in the past (relative to current time)
                if dep_interval < current_interval:
                    continue  # Skip past flights
                
                # Check for overlap with any unavailability period (probability 1.0)
                for unavail_start, unavail_end in zip(unavail_starts, unavail_ends):
                    # Convert unavailability periods to minutes
                    unavail_start_minutes = unavail_start * interval_minutes
                    unavail_end_minutes = unavail_end * interval_minutes
                    
                    if (flight_start_minutes < unavail_end_minutes and flight_end_minutes > unavail_start_minutes):
                        # Conflict detected! Add to set
                        conflict_identifier = (ac_idx, flight_id, flight_start_minutes, flight_end_minutes)
                        current_conflicts_with_prob_1.add(conflict_identifier)
                        break  # Found conflict for this flight, no need to check other unavailability periods
        
        return current_conflicts_with_prob_1

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state using fl_mtx approach."""
        # Generate a random seed based on current time if none provided
        if seed is None:
            seed = int(time.time() * 1000000) % (2**32 - 1)
        
        # Set random seeds for all random number generators
        random.seed(seed)
        np.random.seed(seed)
        
        # Rest of the reset method
        self.current_datetime = self.start_datetime
        self.actions_taken = set()

        self.something_happened = False

        # Deep copy the initial dictionaries
        self.aircraft_dict = copy.deepcopy(self.initial_aircraft_dict)
        self.flights_dict = copy.deepcopy(self.initial_flights_dict)
        self.rotations_dict = copy.deepcopy(self.initial_rotations_dict)
        self.alt_aircraft_dict = copy.deepcopy(self.initial_alt_aircraft_dict)
        
        # Clear and regenerate breakdowns
        self.uncertain_breakdowns = {}
        self.current_breakdowns = {}

        # Generate initial state matrices
        ac_mtx, fl_mtx = self._get_initial_state()
        
        # Update the state
        self.state = (ac_mtx, fl_mtx)
        self.current_ac_mtx, self.current_fl_mtx = ac_mtx, fl_mtx

        # Get initial conflicts using fl_mtx approach
        self.initial_conflict_combinations = self.get_initial_conflicts()

        self.swapped_flights = []
        self.environment_delayed_flights = {}
        self.penalized_delays = {}
        self.penalized_conflicts = set()
        self.resolved_conflicts = set()
        self.penalized_cancelled_flights = set()
        self.automatically_cancelled_flights = set()
        self.penalized_automatically_cancelled_flights = set()

        self.cancelled_flights = set()

        # Initialize eligible flights for conflict resolution bonus
        self.eligible_flights_for_resolved_bonus = self.get_initial_conflicts()

        # Process the state into an observation as a NumPy array using fl_mtx approach
        processed_state, _ = self.process_observation(ac_mtx, fl_mtx)

        if DEBUG_MODE:
            print(f"ac_mtx shape: {ac_mtx.shape}, fl_mtx shape: {fl_mtx.shape}")
            print(f"Type of processed state: {type(processed_state)}")
            print(processed_state)
        return processed_state, {}

    def get_unresolved_uncertainties(self):
        """Retrieves the unresolved uncertainties in the environment using fl_mtx approach.

        This function checks for unavailability periods that are neither 0.0 nor 1.0.

        Returns:
            set: A set of unresolved uncertainties.
        """
        unresolved_uncertainties = set()
        
        # Get the current state matrices
        ac_mtx, fl_mtx = self._get_initial_state()
        
        # For each aircraft (row in ac_mtx)
        for ac_idx in range(ac_mtx.shape[0]):
            # Get unavailability periods for this aircraft (non-zero and not 1.0)
            unavail_periods = ac_mtx[ac_idx, :]
            unavail_indices = np.where((unavail_periods > 0) & (unavail_periods != 1.0))[0]
            
            if len(unavail_indices) == 0:
                continue  # No unresolved unavailability for this aircraft
                
            # Find start and end of unavailability periods
            unavail_starts = []
            unavail_ends = []
            
            # Group consecutive indices into periods
            if len(unavail_indices) > 0:
                start = unavail_indices[0]
                end = start
                for i in range(1, len(unavail_indices)):
                    if unavail_indices[i] == end + 1:
                        end = unavail_indices[i]
                    else:
                        unavail_starts.append(start)
                        unavail_ends.append(end)
                        start = unavail_indices[i]
                        end = start
                unavail_starts.append(start)
                unavail_ends.append(end)
            
            # Add to unresolved_uncertainties set
            for unavail_start, unavail_end in zip(unavail_starts, unavail_ends):
                unresolved_uncertainties.add((ac_idx, unavail_start, unavail_end))
        
        return unresolved_uncertainties
    
    def process_uncertainties(self):
        """Processes aircraft unavailability period uncertainties using fl_mtx approach.

        Probabilities evolve stochastically over time but are capped at [0.05, 0.95].
        When the current datetime + timestep reaches the start time of the aircraft unavailability period,
        resolve the uncertainty period fully to 0.00 or 1.00 by rolling the dice.

        The bias term pushes probabilities that are above 0.5 towards 1.0 and probabilities below 0.5 towards 0.0
        """
        if DEBUG_MODE:
            print(f"Current datetime: {self.current_datetime}")

        # Get current state matrices
        ac_mtx, fl_mtx = self._get_initial_state()

        # Iterate over each aircraft's row in ac_mtx to check for unresolved breakdowns
        for ac_idx, aircraft_id in enumerate(self.aircraft_ids):
            if ac_idx >= self.max_aircraft:
                break

            # Get unavailability info from unavailabilities_dict
            unavail_info = self.unavailabilities_dict[aircraft_id]
            prob = unavail_info['Probability']
            start_col = unavail_info['StartColumn']
            end_col = unavail_info['EndColumn']

            # Only process unresolved breakdowns
            if prob != 0.00 and prob != 1.00:
                # Check for valid start and end columns
                if start_col >= 0 and end_col >= 0:
                    # Convert column indices to minutes for time comparison
                    interval_minutes = 15
                    start_minutes = start_col * interval_minutes
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

                # Check if we've reached the breakdown start time
                if self.current_datetime + self.timestep >= breakdown_start_time:
                    # Roll the dice to resolve uncertainty
                    roll = np.random.random()
                    if roll < new_prob:
                        new_prob = 1.00
                    else:
                        new_prob = 0.00

                # Update the probability in unavailabilities_dict
                self.unavailabilities_dict[aircraft_id]['Probability'] = new_prob

                # Update ac_mtx if there's a valid unavailability period
                if start_col >= 0 and end_col >= 0:
                    ac_mtx[ac_idx, start_col:end_col+1] = new_prob

                # Mark that something changed
            self.something_happened = True

        # Update cached state
        if self.something_happened:
            self._cached_state = (ac_mtx, fl_mtx)
    
    def check_termination_criteria(self):
        """
        Checks if the stopping criteria are met using fl_mtx approach:

        Stopping criteria:
        1. There are no uncertainties in the system anymore.
           (All probabilities are either 0.0 or 1.0.)
        2. There is no overlap of breakdowns (Probability == 1.0) and flights.

        Returns:
            bool: True if both criteria are met, False otherwise.
        """
        # Check that all probabilities are either 0.0 or 1.0
        for aircraft_id in self.aircraft_ids:
            prob = self.unavailabilities_dict[aircraft_id]['Probability']
            if not (prob == 0.0 or prob == 1.0):
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"    prob: {prob} is not 0.0 or 1.0, for aircraft {aircraft_id}, so termination = False")
                return False

        # Check that there is no overlap between flights and breakdowns with prob == 1.0
        # get_current_conflicts_with_prob_1() only returns conflicts for probability == 1.0 breakdowns.
        if len(self.get_current_conflicts_with_prob_1()) > 0:
            if DEBUG_MODE_STOPPING_CRITERIA:
                print(f"    get_current_conflicts_with_prob_1() returns {self.get_current_conflicts_with_prob_1()}, so termination = False")
            return False

        return True
    
    def _is_done(self):
        """Checks if the episode is finished using fl_mtx approach.

        The episode is considered done if:
        1. Current time has reached or exceeded the end time, OR
        2. There are no overlaps between flights and disruptions (regardless of uncertainty status)

        Returns:
            tuple: (bool, str) indicating if the episode is done and the reason.
        """
        if self.current_datetime >= self.end_datetime:
            return True, "Reached the end of the simulation time."
        
        # Check for any overlaps between flights and disruptions using fl_mtx approach
        if not self.check_flight_disruption_overlaps():
            return True, "No remaining overlaps between flights and disruptions."
        
        return False, ""
    
    def check_flight_disruption_overlaps(self):
        """Checks if there are any overlaps between flights and disruptions using fl_mtx approach.
        
        Returns:
            bool: True if there are overlaps, False otherwise.
        """
        # Get current state matrices
        ac_mtx, fl_mtx = self._get_initial_state()
        
        # Calculate current time in minutes from start_datetime
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
        
        # For each aircraft (row in ac_mtx)
        for ac_idx in range(ac_mtx.shape[0]):
            # Get unavailability periods for this aircraft (non-zero values)
            unavail_periods = ac_mtx[ac_idx, :]
            unavail_indices = np.where(unavail_periods > 0)[0]
            
            if len(unavail_indices) == 0:
                continue  # No unavailability for this aircraft
                
            # Find start and end of unavailability periods
            unavail_starts = []
            unavail_ends = []
            
            # Group consecutive indices into periods
            if len(unavail_indices) > 0:
                start = unavail_indices[0]
                end = start
                for i in range(1, len(unavail_indices)):
                    if unavail_indices[i] == end + 1:
                        end = unavail_indices[i]
                    else:
                        unavail_starts.append(start)
                        unavail_ends.append(end)
                        start = unavail_indices[i]
                        end = start
                unavail_starts.append(start)
                unavail_ends.append(end)
            
            # Check each flight for this aircraft using fl_mtx
            aircraft_flights = self.get_all_flights_for_aircraft(ac_idx, fl_mtx)
            interval_minutes = 15
            
            for flight_id, dep_interval, arr_interval, status in aircraft_flights:
                # Skip cancelled flights
                if flight_id in self.cancelled_flights or status <= 0:
                    continue
                
                # Convert intervals to minutes
                flight_start_minutes = dep_interval * interval_minutes
                flight_end_minutes = arr_interval * interval_minutes
                
                # Skip past flights
                if arr_interval * interval_minutes < current_time_minutes:
                    continue
                
                # Check for overlap with any unavailability period
                for unavail_start, unavail_end in zip(unavail_starts, unavail_ends):
                    # Convert unavailability periods to minutes
                    unavail_start_minutes = unavail_start * interval_minutes
                    unavail_end_minutes = unavail_end * interval_minutes
                    
                    if (flight_start_minutes < unavail_end_minutes and flight_end_minutes > unavail_start_minutes):
                        return True  # Found an overlap

        return False  # No overlaps found
    
    def handle_no_conflicts(self, flight_action, aircraft_action):
        """Handles the case when there are no conflicts in the current state using fl_mtx approach.

        This function updates the current datetime, checks if the episode is terminated,
        updates the state, and returns the appropriate outputs.
        """
        if DEBUG_MODE:
            print("*** HANDLING NO CONFLICTS")

        # Store the departure time of the flight that is being acted upon (before the action is taken)
        if flight_action != 0:
            original_flight_action_departure_time = self.flights_dict[flight_action]['DepTime']
        else:
            original_flight_action_departure_time = None

        next_datetime = self.current_datetime + self.timestep
        if next_datetime >= self.end_datetime:
            terminated, reason = self._is_done()
            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")
                
                # Get current state matrices for observation processing
                ac_mtx, fl_mtx = self._get_initial_state()
                processed_state, _ = self.process_observation(ac_mtx, fl_mtx)
                truncated = False
                
                reward = self._calculate_reward(set(), set(), flight_action, aircraft_action, original_flight_action_departure_time, terminated)
                return processed_state, reward, terminated, truncated, {}

        self.current_datetime = next_datetime
        # State will be updated when _get_initial_state() is called next
        
        # Get current state matrices for observation processing
        ac_mtx, fl_mtx = self._get_initial_state()
        processed_state, _ = self.process_observation(ac_mtx, fl_mtx)
        
        # Calculate reward for the action
        reward = self._calculate_reward(set(), set(), flight_action, aircraft_action, original_flight_action_departure_time, False)
        
        return processed_state, reward, False, False, {}
    
    def handle_flight_operations(self, flight_action, aircraft_action, pre_action_conflicts):
        """
        Handles flight operation decisions and resolves conflicts using fl_mtx approach.

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
        # Store the departure time of the flight that is being acted upon (before the action is taken)
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

            post_action_conflicts = self.get_current_conflicts()
            resolved_conflicts = pre_action_conflicts - post_action_conflicts

            terminated, reason = self._is_done()
            truncated = False
            done = terminated or truncated
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, done)

            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            ac_mtx, fl_mtx = self.get_current_state()
            processed_state, _ = self.process_observation(ac_mtx, fl_mtx)
            return processed_state, reward, terminated, truncated, {}
        elif aircraft_action == 0:
            # Cancel the flight
            self.cancel_flight(flight_action)
            if DEBUG_MODE_CANCELLED_FLIGHT:
                print(f"Cancelled flight {flight_action}")

            # Proceed to next timestep
            next_datetime = self.current_datetime + self.timestep
            self.current_datetime = next_datetime

            post_action_conflicts = self.get_current_conflicts()
            resolved_conflicts = pre_action_conflicts - post_action_conflicts

            terminated, reason = self._is_done()
            truncated = False
            done = terminated or truncated
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, done)

            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            ac_mtx, fl_mtx = self.get_current_state()
            processed_state, _ = self.process_observation(ac_mtx, fl_mtx)
            return processed_state, reward, terminated, truncated, {}
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
                
                # Handle this case appropriately
                terminated, reason = self._is_done()
                truncated = False

                done = terminated or truncated
                reward = self._calculate_reward(pre_action_conflicts, pre_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, done)

                ac_mtx, fl_mtx = self.get_current_state()
                processed_state, _ = self.process_observation(ac_mtx, fl_mtx)
                return processed_state, reward, terminated, truncated, {}

            current_aircraft_id = self.rotations_dict[selected_flight_id]['Aircraft']

            if selected_aircraft_id == current_aircraft_id:
                # Delay the flight by scheduling it on the same aircraft
                # Get unavailability end time for the aircraft using fl_mtx approach
                current_aircraft_idx = self.aircraft_id_to_idx[current_aircraft_id]
                unavail_info = self.unavailabilities_dict[current_aircraft_id]
                unavail_end_col = unavail_info['EndColumn']
                
                if unavail_end_col >= 0:
                    # Convert column index to minutes
                    interval_minutes = 15
                    unavail_end_minutes = unavail_end_col * interval_minutes
                    unavail_end_datetime = self.earliest_datetime + timedelta(minutes=unavail_end_minutes)
                else:
                    # No unavailability end time, cannot proceed
                    # In this case, set unavail_end to current time
                    unavail_end_minutes = (self.current_datetime - self.earliest_datetime).total_seconds() / 60
                    unavail_end_datetime = self.current_datetime

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

            post_action_conflicts = self.get_current_conflicts()
            resolved_conflicts = pre_action_conflicts - post_action_conflicts

            terminated, reason = self._is_done()
            terminated = self.check_termination_criteria()
            truncated = False
            done = terminated or truncated
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, terminated=done)

            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            ac_mtx, fl_mtx = self._get_initial_state()
            processed_state, _ = self.process_observation(ac_mtx, fl_mtx)
            return processed_state, reward, terminated, truncated, {}
    
    def _calculate_reward(self, resolved_conflicts, remaining_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, terminated):
        """Calculates the reward based on the current state of the environment using fl_mtx approach.

        The reward consists of several components matching environment_ssf.py logic.
        This is a simplified version - full implementation would match environment_ssf.py exactly.

        Args:
            resolved_conflicts (set): The set of conflicts that were resolved during the action
            remaining_conflicts (set): The set of conflicts that remain after the action
            flight_action (int): The flight action taken by the agent
            aircraft_action (int): The aircraft action taken by the agent
            original_flight_action_departure_time (str): The departure time of the flight being acted upon
            terminated (bool): Whether the episode has ended

        Returns:
            float: The calculated reward for the action
        """
        reward = 0

        if DEBUG_MODE_REWARD:
            print("")
            print(f"Calculating reward for action: flight {flight_action}, aircraft {aircraft_action}")

        # 1. Delay Penalty: Penalize additional minutes of delay
        delay_penalty_minutes = sum(
            self.environment_delayed_flights[flight_id] - self.penalized_delays.get(flight_id, 0)
            for flight_id in self.environment_delayed_flights
        )
        self.scenario_wide_delay_minutes += delay_penalty_minutes
        new_delayed_flights = len([fid for fid in self.environment_delayed_flights if self.environment_delayed_flights[fid] > self.penalized_delays.get(fid, 0)])
        self.scenario_wide_delay_count += new_delayed_flights
        
        if PENALTY_1_DELAY_ENABLED:
            delay_minutes_above_threshold = max(0, delay_penalty_minutes - DELAY_PENALTY_THRESHOLD_MINUTES)
            delay_penalty_total = min(delay_minutes_above_threshold * DELAY_MINUTE_PENALTY, MAX_DELAY_PENALTY)
        else:
            delay_penalty_total = 0

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

        # 3. Inaction Penalty: Penalize doing nothing when conflicts exist
        # Always track inaction, regardless of penalty flag
        if flight_action == 0:
            self.scenario_wide_inaction_count += 1  # Track inaction occurrences
        
        # Only apply penalty if enabled
        if PENALTY_3_INACTION_ENABLED:
            if flight_action == 0 and remaining_conflicts:
                inaction_penalty = NO_ACTION_PENALTY  # -10 penalty when conflicts remain
            elif flight_action == 0 and not remaining_conflicts:
                inaction_penalty = NO_ACTION_PENALTY/2  # -5 penalty when no conflicts remain
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
                proactive_penalty = AHEAD_PENALTY

        # 5. Time Penalty: Small penalty per timestep to encourage faster resolution
        if PENALTY_5_TIME_ENABLED:
            time_penalty = TIMESTEP_HOURS * 60 * TIME_MINUTE_PENALTY
        else:
            time_penalty = 0

        # 6. Final Resolution Reward: Bonus for resolving real conflicts at scenario end
        final_conflict_resolution_reward = 0
        scenario_ended_flag = False
        
        # Always calculate metrics when scenario terminates, regardless of penalty #6 being enabled
        # This ensures metrics are saved even when penalty #6 is disabled
        if terminated:
            scenario_ended = self.check_termination_criteria()
            if scenario_ended:
                scenario_ended_flag = True  # Mark that scenario ended (for metrics saving)
                final_resolved_count = 0
                resolved_flights = []
                conflict_combinations = getattr(self, 'initial_conflict_combinations', getattr(self, 'eligible_flights_for_resolved_bonus', []))
                for (aircraft_id, flight_id) in conflict_combinations:
                    if self.unavailabilities_dict[aircraft_id]['Probability'] == 1.00 and flight_id not in self.cancelled_flights and flight_id not in self.automatically_cancelled_flights:
                        final_resolved_count += 1
                        resolved_flights.append(flight_id)

                # Always update metrics when scenario ends
                self.scenario_wide_resolved_conflicts += final_resolved_count
                self.scenario_wide_resolved_initial_conflicts = final_resolved_count
                
                # Count disruptions resolved to zero (probability went from >0 to 0)
                disruption_resolved_to_zero_count = 0
                for aircraft_id in self.aircraft_ids:
                    if (aircraft_id in self.scenario_wide_initial_disrupted_flights_list or 
                        any(conflict[0] == aircraft_id if isinstance(conflict, tuple) else conflict == aircraft_id 
                            for conflict in self.scenario_wide_initial_disrupted_flights_list)):
                        if self.unavailabilities_dict[aircraft_id]['Probability'] == 0.00:
                            disruption_resolved_to_zero_count += 1
                self.scenario_wide_disruption_resolved_to_zero_count = disruption_resolved_to_zero_count
                
                # Only apply reward if penalty #6 is enabled
                if PENALTY_6_FINAL_REWARD_ENABLED:
                    final_conflict_resolution_reward = final_resolved_count * RESOLVED_CONFLICT_REWARD

        # 7. Automatic cancellation of flights that have already departed
        # Always track automatic cancellations, regardless of penalty flag
        new_automatic_cancellations = {
            flight_id for flight_id in self.automatically_cancelled_flights if flight_id not in self.penalized_automatically_cancelled_flights
        }
        automatic_cancellation_penalty_count = len(new_automatic_cancellations)
        
        # Always track the counts
        self.scenario_wide_automatically_cancelled_count += automatic_cancellation_penalty_count
        self.scenario_wide_cancelled_flights += automatic_cancellation_penalty_count
        
        # Only apply penalty if enabled
        if PENALTY_7_AUTO_CANCELLATION_ENABLED:
            automatic_cancellation_penalty = automatic_cancellation_penalty_count * AUTOMATIC_CANCELLATION_PENALTY
            self.penalized_automatically_cancelled_flights.update(new_automatic_cancellations)
        else:
            automatic_cancellation_penalty = 0
            # Still update penalized set to avoid double-counting, even if penalty is disabled
            self.penalized_automatically_cancelled_flights.update(new_automatic_cancellations)
        
        # 8. Probability-aware shaping: reward resolving high-probability conflicts
        probability_resolution_bonus = 0
        resolved_probability_total = 0
        tail_swap_resolved_conflict = False  # Track if this tail swap resolved a conflict
        
        # Always check if tail swap resolved conflicts (for tracking), regardless of penalty flag
        if (resolved_conflicts 
            and flight_action != 0
            and self.something_happened):
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
                # Do not count conflicts that were auto-cancelled by the environment
                if conflict_flight_id in self.automatically_cancelled_flights:
                    continue

                # If we got here, this action resolved a conflict
                if self.tail_swap_happened:
                    tail_swap_resolved_conflict = True
                
                # Calculate probability for bonus (only if penalty is enabled)
                if PENALTY_8_PROBABILITY_RESOLUTION_BONUS_ENABLED and PROBABILITY_RESOLUTION_BONUS_SCALE > 0:
                    pre_prob = np.nan
                    if hasattr(self, "pre_action_probabilities"):
                        pre_prob = self.pre_action_probabilities.get(aircraft_id, np.nan)
                    if np.isnan(pre_prob):
                        pre_prob = self.unavailabilities_dict.get(aircraft_id, {}).get('Probability', np.nan)
                    if np.isnan(pre_prob):
                        pre_prob = 1.0

                    resolved_probability_total += max(0.0, pre_prob)

        # Only apply bonus if penalty is enabled
        if PENALTY_8_PROBABILITY_RESOLUTION_BONUS_ENABLED and resolved_probability_total > 0:
            probability_resolution_bonus = resolved_probability_total * PROBABILITY_RESOLUTION_BONUS_SCALE
        
        # Always track tail swaps that resolved conflicts, regardless of penalty flag
        if tail_swap_resolved_conflict:
            self.scenario_wide_tail_swaps_resolving += 1

        # 9. Low-confidence action penalty
        low_confidence_action_penalty = 0
        if PENALTY_9_LOW_CONFIDENCE_ACTION_ENABLED:
            if (
                flight_action != 0
                and self.something_happened
                and hasattr(self, 'last_action_probability')
                and self.last_action_probability is not None
                and self.last_action_probability < LOW_CONFIDENCE_ACTION_THRESHOLD
                and hasattr(self, "pre_action_conflict_flights")
                and flight_action in self.pre_action_conflict_flights
                and len(resolved_conflicts) == 0
            ):
                low_confidence_action_penalty = LOW_CONFIDENCE_ACTION_PENALTY

        if self.tail_swap_happened and not tail_swap_resolved_conflict:
            self.scenario_wide_tail_swaps_inefficient += 1
        
        # Reset tail swap tracking for next step
        self.tail_swap_happened = False

        # Update penalized delays for next iteration
        for flight_id, delay in self.environment_delayed_flights.items():
            self.penalized_delays[flight_id] = delay

        # Calculate total reward
        reward = (
            - delay_penalty_total
            - cancel_penalty
            - inaction_penalty
            - automatic_cancellation_penalty
            - proactive_penalty
            - time_penalty
            - low_confidence_action_penalty
            + final_conflict_resolution_reward
            + probability_resolution_bonus
        )

        # Update scenario-wide reward components
        self.scenario_wide_reward_components.update({
            "delay_penalty_total": self.scenario_wide_reward_components["delay_penalty_total"] - delay_penalty_total,
            "cancel_penalty": self.scenario_wide_reward_components["cancel_penalty"] - cancel_penalty,
            "inaction_penalty": self.scenario_wide_reward_components["inaction_penalty"] - inaction_penalty,
            "proactive_penalty": self.scenario_wide_reward_components.get("proactive_penalty", 0) - proactive_penalty,
            "time_penalty": self.scenario_wide_reward_components["time_penalty"] - time_penalty,
            "final_conflict_resolution_reward": self.scenario_wide_reward_components["final_conflict_resolution_reward"] + final_conflict_resolution_reward,
            "automatic_cancellation_penalty": self.scenario_wide_reward_components["automatic_cancellation_penalty"] - automatic_cancellation_penalty,
            "probability_resolution_bonus": self.scenario_wide_reward_components["probability_resolution_bonus"] + probability_resolution_bonus,
            "low_confidence_action_penalty": self.scenario_wide_reward_components["low_confidence_action_penalty"] - low_confidence_action_penalty
        })

        # Round final reward
        reward = round(reward, 4)

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
            "delay_penalty_minutes": delay_penalty_minutes,
            "delay_penalty_capped": delay_penalty_total == MAX_DELAY_PENALTY,
            "cancelled_flights_count": cancellation_penalty_count,
            "time_to_departure_minutes": time_to_departure,
            "action_index": None,
            "flight_action": flight_action,
            "aircraft_action": aircraft_action,
            "original_departure_time": original_flight_action_departure_time,
            "scenario_ended": scenario_ended_flag,
            "penalties": {
                "delay_penalty_total": delay_penalty_total,
                "cancel_penalty": cancel_penalty,
                "inaction_penalty": inaction_penalty,
                "automatic_cancellation_penalty": automatic_cancellation_penalty,
                "proactive_penalty": proactive_penalty,
                "time_penalty": time_penalty,
                "final_conflict_resolution_reward": final_conflict_resolution_reward,
                "probability_resolution_bonus": probability_resolution_bonus,
                "low_confidence_action_penalty": low_confidence_action_penalty
            }
        }
        
        # Add scenario-wide metrics ONLY when scenario ends
        # This saves memory by not storing metrics at every step
        # Note: Metrics are saved regardless of whether penalty #6 is enabled
        if scenario_ended_flag:
            self.info_after_step["scenario_metrics"] = {
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
    
    def schedule_flight_on_aircraft(self, aircraft_id, flight_id, dep_time, current_aircraft_id, arr_time=None, delayed_flights=None, secondary=False):
        """Schedules a flight on an aircraft using fl_mtx approach.

        This function schedules a flight on an aircraft, taking into account unavailability periods and conflicts with existing flights.
        It updates the state and flights dictionary accordingly.
        
        Args:
            aircraft_id (str): The ID of the aircraft to schedule the flight on.
            flight_id (str): The ID of the flight to schedule.
            dep_time (float): The departure time of the flight in minutes from earliest_datetime.
            current_aircraft_id (str): The ID of the current aircraft.
            arr_time (float, optional): The arrival time of the flight in minutes from earliest_datetime. Defaults to None.
        """
        if DEBUG_MODE_SCHEDULING:
            print("\n=== Starting schedule_flight_on_aircraft ===")
            print(f"Scheduling flight {flight_id} on aircraft {aircraft_id}")
            print(f"Initial dep_time: {dep_time}, arr_time: {arr_time}")

        if delayed_flights is None:
            delayed_flights = set()
        
        aircraft_idx = self.aircraft_id_to_idx[aircraft_id]

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

        # Ensure dep_time is not earlier than original departure time
        dep_time = max(dep_time, original_dep_minutes)

        if arr_time is None:
            arr_time = dep_time + flight_duration
        else:
            flight_duration = arr_time - dep_time

        # Check for unavailability conflicts using fl_mtx approach
        unavail_info = self.unavailabilities_dict.get(aircraft_id, {})
        unavail_start_col = unavail_info.get('StartColumn', -1)
        unavail_end_col = unavail_info.get('EndColumn', -1)
        unavail_prob = unavail_info.get('Probability', 0.0)

        # Check if flight overlaps with unavailability
        has_unavail_overlap = False
        if (unavail_start_col >= 0 and 
            unavail_end_col >= 0 and 
            unavail_prob > 0.0):
            
            flight_start = float(original_dep_minutes)
            flight_end = float(original_arr_minutes)
            
            interval_minutes = 15
            unavail_start_minutes = unavail_start_col * interval_minutes
            unavail_end_minutes = unavail_end_col * interval_minutes
            
            if flight_end > unavail_start_minutes:
                has_unavail_overlap = True

        current_ac_is_same_as_target_ac = aircraft_id == current_aircraft_id
        if not current_ac_is_same_as_target_ac:
            self.something_happened = True
            self.tail_swap_happened = True

        # If flight is on current aircraft and completes before disruption, keep it there
        if current_ac_is_same_as_target_ac and not has_unavail_overlap:
            self.something_happened = False
            return

        # Handle unavailability overlap
        if has_unavail_overlap:
            if aircraft_id == current_aircraft_id:
                if unavail_prob > 0.00:
                    interval_minutes = 15
                    unavail_end_minutes = unavail_end_col * interval_minutes
                    dep_time = max(dep_time, unavail_end_minutes + MIN_TURN_TIME)
                    dep_time = max(dep_time, original_dep_minutes)
                    arr_time = dep_time + flight_duration
                    delay = dep_time - original_dep_minutes
                    self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + delay
                    self.something_happened = True
                else:
                    self.something_happened = False
            else:
                if unavail_prob == 1.00:
                    interval_minutes = 15
                    unavail_end_minutes = unavail_end_col * interval_minutes
                    dep_time = max(dep_time, unavail_end_minutes + MIN_TURN_TIME)
                    dep_time = max(dep_time, original_dep_minutes)
                    arr_time = dep_time + flight_duration
                    delay = dep_time - original_dep_minutes
                    self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + delay
                    self.something_happened = True
                else:
                    self.something_happened = True

        # Get all flights on this aircraft using fl_mtx
        scheduled_flights = []
        ac_mtx, fl_mtx = self._get_initial_state()
        
        aircraft_flights = self.get_all_flights_for_aircraft(aircraft_idx, fl_mtx)
        interval_minutes = 15
        
        for existing_flight_id, existing_dep_interval, existing_arr_interval, existing_status in aircraft_flights:
            if existing_flight_id != flight_id and existing_status > 0:
                existing_dep_minutes = existing_dep_interval * interval_minutes
                existing_arr_minutes = existing_arr_interval * interval_minutes
                scheduled_flights.append((existing_flight_id, existing_dep_minutes, existing_arr_minutes))
        
        # Sort flights by departure time
        scheduled_flights.sort(key=lambda x: x[1])

        # Update flights_dict
        self.update_flight_times(flight_id, dep_time, arr_time)

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

    def step(self, action_index):
        """Executes a step in the environment based on the provided action using fl_mtx approach.

        This function processes the action taken by the agent, checks for conflicts, updates the environment state,
        and returns the new state, reward, termination status, truncation status, and additional info.

        Args:
            action_index (int): The action index to be taken by the agent.

        Returns:
            tuple: A tuple containing the processed state, reward, terminated flag, truncated flag, and additional info.
        """
        # Reset something_happened at the start of each step
        self.something_happened = False

        # Get current state matrices
        ac_mtx, fl_mtx = self.get_current_state()

        # Print the current state if in debug mode
        if DEBUG_MODE_PRINT_STATE:
            print(f"Current ac_mtx shape: {ac_mtx.shape}, fl_mtx shape: {fl_mtx.shape}")
            print("")

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
        # Gets probabilities ac unavailabilities that have not yet been resolved (they are neither 1 or 0 but 0<p<1)
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

        # Process uncertainties before handling flight operations
        self.process_uncertainties()
        self.scenario_wide_steps += 1

        if len(pre_action_conflicts) == 0 and len(unresolved_uncertainties) == 0:
            # Handle the case when there are no conflicts
            processed_state, reward, terminated, truncated, info = self.handle_no_conflicts(flight_action, aircraft_action)
        else:
            # Resolve the conflict based on the action
            processed_state, reward, terminated, truncated, info = self.handle_flight_operations(flight_action, aircraft_action, pre_action_conflicts)

        # Update the processed state after processing uncertainties
        ac_mtx, fl_mtx = self.get_current_state()
        processed_state, _ = self.process_observation(ac_mtx, fl_mtx)

        terminated = self.check_termination_criteria()
        if DEBUG_MODE_STOPPING_CRITERIA:
            print(f"checked and terminated: {terminated}")

        # Merge info_after_step into info dict to pass penalty details through
        if info is None:
            info = {}
        
        # Always store action_index in info_after_step for accurate action tracking
        # This ensures we track the original action index before flights are removed
        if hasattr(self, 'info_after_step'):
            self.info_after_step["action_index"] = action_index
            info.update(self.info_after_step)

        return processed_state, reward, terminated, truncated, info

    def map_index_to_action(self, index):
        """Maps action index to (flight_action, aircraft_action)."""
        flight_action = index // (len(self.aircraft_ids) + 1)
        aircraft_action = index % (len(self.aircraft_ids) + 1)
        return flight_action, aircraft_action
    
    def remove_flight(self, flight_id):
        """Removes the specified flight from the dictionaries. Adds it to the cancelled_flights set."""
        # Remove from flights_dict
        if flight_id in self.flights_dict:
            del self.flights_dict[flight_id]

        # Remove from rotations_dict
        if flight_id in self.rotations_dict:
            del self.rotations_dict[flight_id]

        # Mark the flight as canceled
        self.cancelled_flights.add(flight_id)
    
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

        # Note: In the fl_mtx approach, we don't need to manually update the state
        # as it will be recalculated in _get_initial_state()
        self.something_happened = True
    
    def update_flight_times(self, flight_id, dep_time, arr_time):
        """Updates the departure and arrival times of a flight in the flights_dict."""
        # Check if the flight still exists in flights_dict before updating
        if flight_id not in self.flights_dict:
            if DEBUG_MODE_CANCELLED_FLIGHT:
                print(f"Warning: Flight {flight_id} not found in flights_dict. Skipping update.")
            return
        
        flight_info = self.flights_dict[flight_id]
        flight_info['DepTime'] = self.earliest_datetime + timedelta(minutes=dep_time)
        flight_info['ArrTime'] = self.earliest_datetime + timedelta(minutes=arr_time)
        
        # Mark that something changed so cache will be invalidated
        self.something_happened = True

