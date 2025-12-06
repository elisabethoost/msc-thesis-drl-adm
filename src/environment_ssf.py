import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
from src.config_ssf import *
from scripts.utils_ssf import *
import time
import random
from typing import Dict, List, Tuple
import os

class AircraftDisruptionEnv(gym.Env):
    def __init__(self, aircraft_dict, flights_dict, rotations_dict, alt_aircraft_dict, config_dict, env_type, max_flights_per_aircraft=None):
        """Initializes the AircraftDisruptionEnv class.

        Args:
            aircraft_dict (dict): Dictionary containing aircraft information.
            flights_dict (dict): Dictionary containing flight information.
            rotations_dict (dict): Dictionary containing rotation information.
            alt_aircraft_dict (dict): Dictionary containing alternative aircraft information.
            config_dict (dict): Dictionary containing configuration information.
            env_type (str): Type of environment ('myopic' or 'proactive', 'reactive', 'drl-greedy').
            max_flights_per_aircraft (int, optional): Maximum flights per aircraft for state space optimization.
                If None, uses MAX_FLIGHTS_PER_AIRCRAFT from config. Training script should provide this.
        """
        super(AircraftDisruptionEnv, self).__init__()
        
        # Store the environment type ('myopic' or 'proactive', 'reactive', 'drl-greedy')
        self.env_type = env_type  
        
        # Constants for environment configuration
        self.max_aircraft = MAX_AIRCRAFT
        self.columns_state_space = COLUMNS_STATE_SPACE + 1  
        self.rows_state_space = ROWS_STATE_SPACE

        if max_flights_per_aircraft is not None:
            self.max_flights_per_aircraft = max_flights_per_aircraft
        else:
            # Fallback: use default if not provided (training script should always provide this)
            self.max_flights_per_aircraft = MAX_FLIGHTS_PER_AIRCRAFT

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

        # Filter out flights with '+' in DepTime (next day flights)
        this_day_flights = [flight_info for flight_info in flights_dict.values() if '+' not in flight_info['DepTime']]

        # Determine the earliest possible event in the environment
        self.earliest_datetime = min(
            min(datetime.strptime(config_dict['RecoveryPeriod']['StartDate'] + ' ' + flight_info['DepTime'], '%d/%m/%y %H:%M') for flight_info in this_day_flights),
            self.start_datetime
        )

        # Define observation and action spaces
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        
        ac_mtx_size = self.max_aircraft * MAX_TIME_INTERVALS
        flight_features_size = self.max_aircraft * self.max_flights_per_aircraft * 4
        actual_observation_size = ac_mtx_size + flight_features_size
        
        self.observation_space = spaces.Dict({
            'state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(actual_observation_size,),
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
        self.environment_delayed_flights = {}   # Tracks delays for flights {flight_id: delay_minutes}
        self.penalized_delays = {}           # Set of penalized delays
        self.penalized_cancelled_flights = set()  # To keep track of penalized cancelled flights
        self.cancelled_flights = set()
        self.automatically_cancelled_flights = set()
        self.penalized_automatically_cancelled_flights = set()
        
        self.penalized_conflicts = set()        # Set of penalized conflicts
        self.resolved_conflicts = set()         # Set of resolved conflicts
        

        # Initialize empty containers for breakdowns
        self.uncertain_breakdowns = {}
        self.current_breakdowns = {}

        # Initialize a dictionary to store unavailabilities
        self.unavailabilities_dict = {}

        self.info_after_step = {}

        # Performance optimization attributes (must be initialized before _get_initial_state)
        self.something_happened = False
        self._cached_state = None
        self._total_calc_time = 0.0
        self._step_count = 0

        # Initialize the environment state without generating probabilities
        self.current_datetime = self.start_datetime
        self.state = self._get_initial_state()
        
        # Store the current state matrices for efficient access
        self.current_ac_mtx, self.current_flight_features = self.state
        
        # Update observation space to match actual matrix dimensions
        self._update_observation_space()

        # Initialize eligible flights for conflict resolution bonus
        self.eligible_flights_for_resolved_bonus = self.get_initial_conflicts()
        #self.eligible_flights_for_not_being_cancelled_when_disruption_happens = self.get_initial_conflicts_with_deptime_before_unavail_start()

        self.scenario_wide_delay_minutes = 0
        self.scenario_wide_delay_count = 0
        self.scenario_wide_cancelled_flights = 0
        self.scenario_wide_automatically_cancelled_count = 0
        self.scenario_wide_steps = 0
        self.scenario_wide_resolved_conflicts = 0
        self.scenario_wide_resolved_initial_conflicts = 0
        self.scenario_wide_disruption_resolved_to_zero_count = 0
        self.scenario_wide_solution_slack = 0
        self.scenario_wide_tail_swaps = 0
        self.scenario_wide_tail_swaps_resolving = 0
        self.scenario_wide_tail_swaps_inefficient = 0
        self.scenario_wide_inaction_count = 0
        self.scenario_wide_initial_disrupted_flights_list = self.get_current_conflicts()
        self.scenario_wide_actual_disrupted_flights = len(self.get_current_conflicts())
        # print(f"*********scenario_wide_actual_disrupted_flights: {self.scenario_wide_actual_disrupted_flights}")
        # print(f"*********scenario_wide_initial_disrupted_flights_list: {self.scenario_wide_initial_disrupted_flights_list}")

        self.scenario_wide_reward_components = {
            "delay_penalty_total": 0,
            "cancel_penalty": 0,
            "inaction_penalty": 0,
            "proactive_penalty": 0,
            "time_penalty": 0,
            "final_conflict_resolution_reward": 0,
            "automatic_cancellation_penalty": 0,
            "probability_resolution_bonus": 0,
            "low_confidence_action_penalty": 0,
        }

        # Initialize tail swap tracking
        self.tail_swap_happened = False

    def _get_initial_state(self):
        """Initializes the state matrices for the environment: ac_mtx and flight_features.
        
        Returns:
            tuple: (ac_mtx, flight_features)
            - ac_mtx: (MAX_AIRCRAFT, MAX_TIME_INTERVALS) - aircraft unavailability matrix
            - flight_features: (MAX_AIRCRAFT, MAX_FLIGHTS_PER_AIRCRAFT * 4) - compact flight representation
              Each row = one aircraft, columns = [fl_id, dep_interval, arr_interval, status] × flights
        """
        # Simple caching: only recalculate if something changed
        if hasattr(self, '_cached_state') and self._cached_state is not None and not self.something_happened:
            return self._cached_state
        
        # Simple timing for performance measurement
        import time
        start_time = time.time()
        
        # Track total calculation time for performance comparison
        import math
        from dateutil import rrule

        # Calculate current time and remaining recovery period in minutes
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60

        # List to keep track of flights to remove from dictionaries
        flights_to_remove = set()

        # Set to collect actual flights in state space
        active_flights = set()

        # 1. Calculate 15-min intervals between start and end
        interval_minutes = 15
        total_minutes = int((self.end_datetime - self.start_datetime).total_seconds() // 60)
        num_intervals = math.ceil(total_minutes / interval_minutes)
        interval_starts = [self.start_datetime + timedelta(minutes=i*interval_minutes) for i in range(num_intervals)]
        # For fast lookup, create a dict: datetime -> col idx
        interval_lookup = {dt: idx for idx, dt in enumerate(interval_starts)}

        # 2. Create ac_mtx
        ac_mtx = np.zeros((self.max_aircraft, num_intervals), dtype=np.float32)
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

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

            # Fill in the matrix for each unavailability and store column indices
            if not np.isnan(breakdown_probability) and breakdown_probability > 0:
                # Convert minutes to interval indices for ac_mtx
                start_idx = max(0, int(unavail_start_minutes // interval_minutes))
                end_idx = min(num_intervals, int(math.ceil(unavail_end_minutes / interval_minutes)))
                ac_mtx[idx, start_idx:end_idx] = breakdown_probability
                
                # Store the unavailability information with column indices instead of minutes
                self.unavailabilities_dict[aircraft_id] = {
                    'Probability': breakdown_probability,
                    'StartColumn': start_idx,
                    'EndColumn': end_idx
                }
            else:
                # Store the unavailability information with no unavailability
                self.unavailabilities_dict[aircraft_id] = {
                    'Probability': breakdown_probability,
                    'StartColumn': -1,  # No unavailability
                    'EndColumn': -1     # No unavailability
                }

        # 3. Create flight_features (organized by aircraft, like Model 1)
        # Structure: 3 rows (aircraft) × (max_flights_per_aircraft * 4) columns
        flight_features = np.full((self.max_aircraft, self.max_flights_per_aircraft * 4), np.nan, dtype=np.float32)
        # Columns per aircraft: [fl_id_1, dep_interval_1, arr_interval_1, status_1, fl_id_2, dep_interval_2, arr_interval_2, status_2, ...]
        
        # Track flight count per aircraft for easy access
        flights_per_aircraft = {ac_idx: 0 for ac_idx in range(self.max_aircraft)}
        
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break

            # Get unavailability info for this aircraft
            breakdown_probability = self.unavailabilities_dict[aircraft_id]['Probability']
            unavail_start_col = self.unavailabilities_dict[aircraft_id]['StartColumn']
            unavail_end_col = self.unavailabilities_dict[aircraft_id]['EndColumn']

            flight_count = 0  # Track flights for this aircraft
            
            # For each flight for this aircraft
            for flight_id, rotation_info in self.rotations_dict.items():
                if flight_id in self.flights_dict and rotation_info['Aircraft'] == aircraft_id:
                    if flight_count >= self.max_flights_per_aircraft:
                        break  # Reached max flights per aircraft
                    
                    flight_info = self.flights_dict[flight_id]
                    dep_time = parse_time_with_day_offset(flight_info['DepTime'], self.start_datetime)
                    arr_time = parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime)

                    dep_time_minutes = (dep_time - self.earliest_datetime).total_seconds() / 60
                    arr_time_minutes = (arr_time - self.earliest_datetime).total_seconds() / 60

                    # Exclude flights that have already departed and are in conflict
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
                    
                    # Store in flight_features[row=aircraft, columns for this flight]
                    # Column structure: [fl_id, dep_interval, arr_interval, status]
                    col_start = flight_count * 4  # Each flight takes 4 columns
                    flight_features[idx, col_start] = float(flight_id)  # Flight ID
                    flight_features[idx, col_start + 1] = float(dep_interval)  # Departure interval
                    flight_features[idx, col_start + 2] = float(arr_interval)  # Arrival interval
                    flight_features[idx, col_start + 3] = 1.0  # Status: 1.0 = active
                    
                    flight_count += 1
                    flights_per_aircraft[idx] = flight_count
                    active_flights.add(flight_id)
                    
                    if flight_count >= self.max_flights_per_aircraft:
                        break
            # Remaining columns for this aircraft stay NaN (flight slot doesn't exist)

        # Update flight_id_to_idx with only the active flights
        self.flight_id_to_idx = {
            flight_id: idx for idx, flight_id in enumerate(sorted(active_flights))
        }

        # Remove past flights from dictionaries
        for flight_id in flights_to_remove:
            self.remove_flight(flight_id)

        # Create mapping from (aircraft_idx, flight_idx) to flight_id for efficient lookup
        # Also create mapping from flight_id to (aircraft_idx, flight_idx)
        self.flight_id_to_position = {}  # flight_id -> (aircraft_idx, flight_idx)
        self.ac_mtx_idx_to_aircraft_id = {}  # aircraft_idx -> aircraft_id
        
        for idx, aircraft_id in enumerate(self.aircraft_ids):
            if idx >= self.max_aircraft:
                break
            # Store mapping from matrix index to aircraft_id
            self.ac_mtx_idx_to_aircraft_id[idx] = aircraft_id
            
            # Map flights in this aircraft row to flight IDs
            num_flights = flights_per_aircraft.get(idx, 0)
            for flight_idx in range(num_flights):
                col_start = flight_idx * 4
                if not np.isnan(flight_features[idx, col_start]):
                    flight_id = int(flight_features[idx, col_start])
                    self.flight_id_to_position[flight_id] = (idx, flight_idx)
        
        # Store flights_per_aircraft for easy access
        self.flights_per_aircraft = flights_per_aircraft
        
        # Pad matrices to fixed size for consistent observation space
        MAX_AIRCRAFT_FIXED = self.max_aircraft  
        MAX_TIME_INTERVALS_FIXED = 96  
        MAX_FLIGHTS_PER_AC_FIXED = self.max_flights_per_aircraft  
        
        # Pad ac_mtx to fixed size
        padded_ac_mtx = np.zeros((MAX_AIRCRAFT_FIXED, MAX_TIME_INTERVALS_FIXED), dtype=np.float32)
        padded_ac_mtx[:ac_mtx.shape[0], :ac_mtx.shape[1]] = ac_mtx
        
        # Pad flight_features to fixed size
        padded_flight_features = np.full((MAX_AIRCRAFT_FIXED, MAX_FLIGHTS_PER_AC_FIXED * 4), np.nan, dtype=np.float32)
        padded_flight_features[:flight_features.shape[0], :flight_features.shape[1]] = flight_features
        
        # Cache the result for future use
        self._cached_state = (padded_ac_mtx, padded_flight_features)
        
        # Simple timing output (only show occasionally to avoid spam)
        self._step_count += 1
            
        if self._step_count % 1000 == 0:  # Show timing every 1000 steps instead of 100
            elapsed_time = time.time() - start_time
            self._total_calc_time += elapsed_time
            avg_time = self._total_calc_time / self._step_count if self._step_count > 0 else 0
            print(f"Matrix calculation: {elapsed_time:.4f}s, Avg: {avg_time:.4f}s, Total: {self._total_calc_time:.4f}s (step {self._step_count})")
        
        return padded_ac_mtx, padded_flight_features
    
    def get_performance_stats(self):
        """Returns simple performance statistics for comparison."""
        avg_time = self._total_calc_time / self._step_count if self._step_count > 0 else 0
        return {
            'total_steps': self._step_count,
            'total_calc_time': self._total_calc_time,
            'avg_calc_time': avg_time,
            'cache_hit_rate': 'N/A'  # Could be enhanced later
        }
    
    def get_current_state(self):
        """Returns the current state matrices without recreating them.
        
        Returns:
            tuple: (ac_mtx, flight_features) - current state matrices
        """
        # Use cached state if available and nothing changed
        if hasattr(self, '_cached_state') and self._cached_state is not None and not self.something_happened:
            return self._cached_state
            
        if self.current_ac_mtx is None or self.current_flight_features is None:
            # If state hasn't been initialized, initialize it
            self.current_ac_mtx, self.current_flight_features = self._get_initial_state()
            self.state = (self.current_ac_mtx, self.current_flight_features)
        
        return self.current_ac_mtx, self.current_flight_features
    
    def update_state(self, new_ac_mtx, new_flight_features):
        """Updates the current state matrices.
        
        Args:
            new_ac_mtx: New aircraft matrix
            new_flight_features: New flight features matrix
        """
        self.current_ac_mtx = new_ac_mtx.copy()
        self.current_flight_features = new_flight_features.copy()
        self.state = (self.current_ac_mtx, self.current_flight_features)
        
        # Update observation space to match new matrix dimensions
        self._update_observation_space()
    
    def _update_observation_space(self):
        """Updates the observation space to use a fixed size for consistency across scenarios."""
        # Use a fixed observation space size that can accommodate all scenarios
        # This prevents observation space mismatch errors when switching between scenarios
        
        # Fixed sizes based on actual dimensions from config
        MAX_AIRCRAFT_FIXED = self.max_aircraft  
        MAX_TIME_INTERVALS_FIXED = 96  
        MAX_FLIGHTS_PER_AC_FIXED = self.max_flights_per_aircraft  
        
        # Calculate fixed observation space size
        ac_mtx_size = MAX_AIRCRAFT_FIXED * MAX_TIME_INTERVALS_FIXED
        flight_features_size = MAX_AIRCRAFT_FIXED * MAX_FLIGHTS_PER_AC_FIXED * 4
        fixed_obs_size = ac_mtx_size + flight_features_size
        
        if DEBUG_MODE:
            print(f"Using fixed observation space size: {fixed_obs_size}")
            print(f"ac_mtx size: {ac_mtx_size}, flight_features size: {flight_features_size}")
        
        # Update the observation space with the fixed dimensions
        self.observation_space = spaces.Dict({
            'state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(fixed_obs_size,),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(self.action_space.n,),
                dtype=np.uint8
            )
        })

    def get_all_flights_for_aircraft(self, ac_idx, flight_features):
        """Gets all flights for a specific aircraft from flight_features.
        
        Args:
            ac_idx (int): Aircraft index
            flight_features: Flight features matrix (MAX_AIRCRAFT, MAX_FLIGHTS_PER_AIRCRAFT * 4)
            
        Returns:
            list: List of tuples (flight_id, dep_interval, arr_interval, status)
        """
        flights = []
        
        if ac_idx >= flight_features.shape[0]:
            return flights
        
        # Get the row for this aircraft
        aircraft_row = flight_features[ac_idx, :]
        
        # Each flight takes 4 columns: [fl_id, dep_interval, arr_interval, status]
        num_flights = self.flights_per_aircraft.get(ac_idx, 0)
        
        for flight_idx in range(num_flights):
            col_start = flight_idx * 4
            
            # Check if we've exceeded the matrix bounds
            if col_start + 3 >= flight_features.shape[1]:
                break
            
            # Extract flight data
            flight_id = aircraft_row[col_start]
            dep_interval = aircraft_row[col_start + 1]
            arr_interval = aircraft_row[col_start + 2]
            status = aircraft_row[col_start + 3]
            
            # Skip NaN values (empty flight slots)
            if np.isnan(flight_id):
                break
            
            flights.append((int(flight_id), int(dep_interval), int(arr_interval), float(status)))
        
        return flights
    
    def get_flight_data(self, flight_id, flight_features):
        """Gets flight data from flight_features for a specific flight.
        
        Args:
            flight_id (int): Flight ID
            flight_features: Flight features matrix
            
        Returns:
            tuple: (ac_idx, flight_idx, flight_id, dep_interval, arr_interval, status) or None
        """
        # Use flight_id_to_position mapping if available
        if hasattr(self, 'flight_id_to_position') and flight_id in self.flight_id_to_position:
            ac_idx, flight_idx = self.flight_id_to_position[flight_id]
            
            if ac_idx >= flight_features.shape[0]:
                return None
            
            col_start = flight_idx * 4
            if col_start + 3 >= flight_features.shape[1]:
                return None
            
            dep_interval = flight_features[ac_idx, col_start + 1]
            arr_interval = flight_features[ac_idx, col_start + 2]
            status = flight_features[ac_idx, col_start + 3]
            
            if np.isnan(dep_interval) or np.isnan(arr_interval):
                return None
            
            return (ac_idx, flight_idx, int(flight_id), int(dep_interval), int(arr_interval), float(status))
        
        # Fallback: search through all aircraft
        for ac_idx in range(flight_features.shape[0]):
            num_flights = self.flights_per_aircraft.get(ac_idx, 0)
            for flight_idx in range(num_flights):
                col_start = flight_idx * 4
                if col_start + 3 >= flight_features.shape[1]:
                    break
                
                if not np.isnan(flight_features[ac_idx, col_start]):
                    if int(flight_features[ac_idx, col_start]) == flight_id:
                        dep_interval = flight_features[ac_idx, col_start + 1]
                        arr_interval = flight_features[ac_idx, col_start + 2]
                        status = flight_features[ac_idx, col_start + 3]
                        
                        if np.isnan(dep_interval) or np.isnan(arr_interval):
                            return None
                        
                        return (ac_idx, flight_idx, int(flight_id), int(dep_interval), int(arr_interval), float(status))
        
        return None

    def get_initial_conflicts(self):
        """Retrieves the initial conflicts in the environment.

        This function checks for conflicts between flights and unavailability periods,
        considering unavailabilities with probability greater than 0.0.

        Returns:
            set: A set of conflicts currently present in the initial state of the environment.
        """
        initial_conflicts = set()
        
        # Get the current state matrices
        ac_mtx, flight_features = self._get_initial_state()
        
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
            
            # Check each flight for this aircraft using flight_features
            aircraft_flights = self.get_all_flights_for_aircraft(ac_idx, flight_features)
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
                
                # Note: We continue to the next flight even if a conflict was found
                # This ensures we check ALL flights for conflicts with ALL unavailability periods
        
        return initial_conflicts

    def get_valid_aircraft_actions(self):
        """Generates a list of valid aircraft actions for the agent.
        Returns: list: A list of valid aircraft actions that the agent can take.
        """
        return list(range(len(self.aircraft_ids) + 1))  # 0 to len(aircraft_ids)

    def get_valid_flight_actions(self):
        """Generates a list of valid flight actions based on flights in state space using matrix-based approach."""
        # Calculate current time in minutes from start_datetime
        current_time_minutes = (self.current_datetime - self.start_datetime).total_seconds() / 60
        
        # Calculate which 15-minute interval corresponds to current time
        interval_minutes = 15
        current_interval = int(current_time_minutes // interval_minutes)
        
        # Get the current state matrices
        ac_mtx, flight_features = self._get_initial_state()
        
        # Get all valid flight IDs from flight_features
        valid_flight_ids = set()
        
        # Check each aircraft row
        for ac_idx in range(flight_features.shape[0]):
            aircraft_flights = self.get_all_flights_for_aircraft(ac_idx, flight_features)
            for flight_id, dep_interval, arr_interval, status in aircraft_flights:
                # Check if flight is not cancelled
                if flight_id in self.cancelled_flights or status <= 0:
                    continue
                
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
        """Creates a binary vector action_mask for valid flight & aircraft pairs using matrix-based approach."""
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
            ac_mtx, flight_features = self._get_initial_state()

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
                        # Get flights for this aircraft using flight_features
                        aircraft_flights = self.get_all_flights_for_aircraft(ac_idx, flight_features)
                        for existing_flight_id, existing_dep_interval, existing_arr_interval, existing_status in aircraft_flights:
                            if existing_status <= 0:  # Skip cancelled flights
                                continue
                            
                            # Convert intervals to minutes
                            interval_minutes = 15
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
    
    def map_index_to_action(self, index): 
        """Maps the flattened action space index to the corresponding (flight, aircraft) action pair.

        Args:
            index (int): The index in the flattened action space.

        Returns:
            tuple: A tuple containing the flight and aircraft actions.
        """
        flight_action = index // (len(self.aircraft_ids) + 1)
        aircraft_action = index % (len(self.aircraft_ids) + 1)
        return flight_action, aircraft_action

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

    def _is_cancellation_necessary(self, flight_action):
        """
        Checks if cancellation is necessary or if delay is possible using matrix-based approach.
        
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
        current_aircraft_idx = self.aircraft_id_to_idx[current_aircraft_id]
        
        # Get current state matrices
        ac_mtx, flight_features = self._get_initial_state()
        
        # Get flight data
        flight_data = self.get_flight_data(flight_action, flight_features)
        if flight_data is None:
            return True  # Flight doesn't exist, cancellation is necessary
        
        _, _, _, dep_interval, arr_interval, status = flight_data
        
        # Check if the flight can be delayed on the same aircraft
        unavail_periods = ac_mtx[current_aircraft_idx, :]
        unavail_indices = np.where(unavail_periods > 0)[0]
        
        if len(unavail_indices) > 0:
            # There's an unavailability period - check if flight can be delayed after it
            # Find the latest unavailability end interval
            latest_unavail_end_interval = max(unavail_indices)
            
            # If flight can be delayed after unavailability, cancellation is not necessary
            if arr_interval > latest_unavail_end_interval:
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
        Checks if a flight can be moved to a target aircraft without creating conflicts using matrix-based approach.
        
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
        
        dep_minutes = (dep_time - self.start_datetime).total_seconds() / 60
        arr_minutes = (arr_time - self.start_datetime).total_seconds() / 60
        
        # Get current state matrices
        ac_mtx, flight_features = self._get_initial_state()
        
        # Check for unavailability conflicts on target aircraft
        target_aircraft_idx = self.aircraft_id_to_idx[target_aircraft_id]
        unavail_periods = ac_mtx[target_aircraft_idx, :]
        unavail_indices = np.where(unavail_periods > 0)[0]
        
        if len(unavail_indices) > 0:
            interval_minutes = 15
            # Convert flight times to interval indices
            flight_start_interval = int(dep_minutes // interval_minutes)
            flight_end_interval = int(arr_minutes // interval_minutes)
            
            # Check for overlap with unavailability periods
            for unavail_idx in unavail_indices:
                if flight_start_interval <= unavail_idx <= flight_end_interval:
                    if unavail_periods[unavail_idx] == 1.0:
                        return False  # Certain conflict, move not possible
                    # For uncertain conflicts, allow the move (will be resolved later)
        
        # Check for conflicts with existing flights on target aircraft
        target_aircraft_flights = self.get_all_flights_for_aircraft(target_aircraft_idx, flight_features)
        interval_minutes = 15
        
        for existing_flight_id, existing_dep_interval, existing_arr_interval, existing_status in target_aircraft_flights:
            if existing_status <= 0:  # Skip cancelled flights
                continue
            
            # Convert to minutes for comparison
            existing_flight_start_minutes = existing_dep_interval * interval_minutes
            existing_flight_end_minutes = existing_arr_interval * interval_minutes
            
            # Check for overlap
            if dep_minutes < existing_flight_end_minutes and arr_minutes > existing_flight_start_minutes:
                return False  # Conflict with existing flight
                    
        return True  # No conflicts found, move is possible
    
    def _is_reschedule_logical(self, flight_action, aircraft_action):
        """
        Checks if rescheduling a flight to a different aircraft is logical using matrix-based approach.
        
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
        Checks if moving a flight resolves an existing conflict using matrix-based approach.
        
        Args:
            flight_id (int): The flight to move
            current_aircraft_id (str): Current aircraft
            target_aircraft_id (str): Target aircraft
            
        Returns:
            bool: True if move resolves a conflict, False otherwise
        """
        # Get current state matrices
        ac_mtx, flight_features = self._get_initial_state()
        
        # Check if there's a conflict on the current aircraft
        current_aircraft_idx = self.aircraft_id_to_idx[current_aircraft_id]
        unavail_periods = ac_mtx[current_aircraft_idx, :]
        unavail_indices = np.where(unavail_periods > 0)[0]
        
        if len(unavail_indices) > 0:
            flight_info = self.flights_dict[flight_id]
            dep_time = parse_time_with_day_offset(flight_info['DepTime'], self.start_datetime)
            arr_time = parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime)
            
            dep_minutes = (dep_time - self.start_datetime).total_seconds() / 60
            arr_minutes = (arr_time - self.start_datetime).total_seconds() / 60
            
            interval_minutes = 15
            flight_start_interval = int(dep_minutes // interval_minutes)
            flight_end_interval = int(arr_minutes // interval_minutes)
            
            # Check if flight conflicts with unavailability
            for unavail_idx in unavail_indices:
                if flight_start_interval <= unavail_idx <= flight_end_interval:
                    return True  # Move resolves this conflict
                    
        return False  # No conflict to resolve

    def _better_alternatives_exist(self, flight_action):
        """
        Checks if better alternatives exist for a flight than cancellation using matrix-based approach.
        
        Args:
            flight_action (int): The flight to check
            
        Returns:
            bool: True if better alternatives exist, False otherwise
        """
        if flight_action not in self.rotations_dict:
            return False
            
        current_aircraft_id = self.rotations_dict[flight_action]['Aircraft']
        
        # Get current state matrices
        ac_mtx, flight_features = self._get_initial_state()
        
        # Check if delay is possible on same aircraft
        current_aircraft_idx = self.aircraft_id_to_idx[current_aircraft_id]
        unavail_periods = ac_mtx[current_aircraft_idx, :]
        unavail_indices = np.where(unavail_periods > 0)[0]
        
        if len(unavail_indices) > 0:
            flight_info = self.flights_dict[flight_action]
            original_arr_time = parse_time_with_day_offset(flight_info['ArrTime'], self.start_datetime)
            original_arr_minutes = (original_arr_time - self.start_datetime).total_seconds() / 60
            
            interval_minutes = 15
            latest_unavail_end = max(unavail_indices) * interval_minutes
            
            if original_arr_minutes > latest_unavail_end:
                return True  # Delay is possible
        
        # Check if move to another aircraft is possible
        for other_aircraft_id in self.aircraft_ids:
            if other_aircraft_id != current_aircraft_id:
                if self._can_flight_be_moved_to_aircraft(flight_action, other_aircraft_id):
                    return True  # Move is possible
                    
        return False  # No better alternatives found
    
    def process_observation(self, ac_mtx, flight_features):
        """Processes the observation by applying env_type-specific masking using matrix-based approach.
        Does NOT modify internal state or unavailabilities_dict.
        Returns an observation suitable for the agent."""
        ac_mtx_to_observe = ac_mtx.copy()
        flight_features_to_observe = flight_features.copy()

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

        # For flight_features, we don't need masking (values are already compact)
        # Just replace NaN with 0.0 for network input
        flight_features_flat = np.nan_to_num(flight_features_to_observe, nan=0.0).flatten()
        ac_mtx_flat = ac_mtx_to_observe.flatten()

        # Combine the matrices
        state_flat = np.concatenate([ac_mtx_flat, flight_features_flat])

        # Use get_action_mask to generate the action mask
        action_mask = self.get_action_mask()

        # Return the observation dictionary without modifying internal structures
        obs_with_mask = {
            'state': state_flat,
            'action_mask': action_mask
        }
        return obs_with_mask, (ac_mtx_to_observe, flight_features_to_observe)
        # obs_with_mask: A dictionary containing both the state vector and the action mask (= a vector indicating which actions are valid (1 = valid, 0 = invalid) in the current state)
        # (ac_mtx_to_observe, flight_features_to_observe): A copy of the state matrices that is used to update the state matrices after each step but with certain information hidden

    def fix_state(self, ac_mtx, flight_features):
        """Fixes state matrices by handling next-day flights (adding 1440 minutes to end times if they're before start times).
        This function modifies the matrices in-place."""
        interval_minutes = 15
        
        # Fix ac_mtx (aircraft unavailabilities)
        for ac_idx in range(ac_mtx.shape[0]):
            unavail_periods = ac_mtx[ac_idx, :]
            unavail_indices = np.where(unavail_periods > 0)[0]
            
            if len(unavail_indices) > 0:
                # Group consecutive indices into periods
                unavail_starts = []
                unavail_ends = []
                
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
                
                # Check and fix each unavailability period
                for unavail_start, unavail_end in zip(unavail_starts, unavail_ends):
                    start_minutes = unavail_start * interval_minutes
                    end_minutes = unavail_end * interval_minutes
                    
                    if end_minutes < start_minutes:
                        # This is a next-day unavailability, add 1440 minutes
                        # Calculate how many intervals 1440 minutes represents
                        additional_intervals = int(1440 // interval_minutes)
                        new_end = unavail_end + additional_intervals
                        
                        # Update the matrix
                        ac_mtx[ac_idx, unavail_start:unavail_end+1] = 0.0  # Clear old period
                        ac_mtx[ac_idx, unavail_start:new_end+1] = unavail_periods[unavail_start]  # Set new period

        # Fix flight_features (flight schedules)
        # Check each aircraft row
        for ac_idx in range(flight_features.shape[0]):
            num_flights = self.flights_per_aircraft.get(ac_idx, 0)
            for flight_idx in range(num_flights):
                col_start = flight_idx * 4
                if col_start + 2 >= flight_features.shape[1]:
                    break
                if np.isnan(flight_features[ac_idx, col_start]):
                    break
                
                dep_interval = int(flight_features[ac_idx, col_start + 1])
                arr_interval = int(flight_features[ac_idx, col_start + 2])
                
                # Convert to minutes
                start_minutes = dep_interval * interval_minutes
                end_minutes = arr_interval * interval_minutes
                
                if end_minutes < start_minutes:
                    # This is a next-day flight, add 1440 minutes (96 intervals)
                    additional_intervals = int(1440 // interval_minutes)
                    new_arr_interval = arr_interval + additional_intervals
                    
                    # Update the matrix
                    flight_features[ac_idx, col_start + 2] = float(new_arr_interval)

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
    
    def get_current_conflicts(self):
        """Retrieves the current conflicts in the environment using matrix-based approach.

        This function checks for conflicts between flights and unavailability periods,
        considering unavailabilities with probability > 0.0.
        It excludes cancelled flights which are not considered conflicts.

        Returns:
            set: A set of conflicts currently present in the environment.
        """
        current_conflicts = set()
        
        # Get the current state matrices
        ac_mtx, flight_features = self._get_initial_state()
        
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
            
            # Check each flight for this aircraft using flight_features
            aircraft_flights = self.get_all_flights_for_aircraft(ac_idx, flight_features)
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
        """Retrieves the current conflicts in the environment using matrix-based approach.

        This function checks for conflicts between flights and unavailability periods,
        considering only unavailabilities with probability 1.0.
        It excludes cancelled flights which are not considered conflicts.

        Returns:
            set: A set of conflicts currently present in the environment.
        """
        current_conflicts_with_prob_1 = set()
        
        # Get the current state matrices
        ac_mtx, flight_features = self._get_initial_state()
        
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
            
            # Check each flight for this aircraft using flight_features
            aircraft_flights = self.get_all_flights_for_aircraft(ac_idx, flight_features)
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
    
    def process_uncertainties(self):
        """Processes aircraft unavailability period uncertainties using matrix-based approach.

        Probabilities evolve stochastically over time but are capped at [0.05, 0.95].
        When the current datetime + timestep reaches the start time of the aircraft unavailability period,
        resolve the uncertainty period fully to 0.00 or 1.00 by rolling the dice.

        The bias term pushes probabilities that are above 0.5 towards 1.0 and probabilities below 0.5 towards 0.0
        """
        if DEBUG_MODE:
            print(f"Current datetime: {self.current_datetime}")

        # Get current state matrices
        ac_mtx, flight_features = self._get_initial_state()

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
                self.unavailabilities_dict[aircraft_id]['Probability'] = new_prob
                
                # Update alt_aircraft_dict
                if aircraft_id in self.alt_aircraft_dict:
                    if isinstance(self.alt_aircraft_dict[aircraft_id], dict):
                        self.alt_aircraft_dict[aircraft_id] = [self.alt_aircraft_dict[aircraft_id]]
                    elif isinstance(self.alt_aircraft_dict[aircraft_id], str):
                        # Handle case where entry is a string
                        end_minutes = end_col * interval_minutes
                        self.alt_aircraft_dict[aircraft_id] = [{
                            'StartDate': breakdown_start_time.strftime('%d/%m/%y'),
                            'StartTime': breakdown_start_time.strftime('%H:%M'),
                            'EndDate': (breakdown_start_time + timedelta(minutes=end_minutes - start_minutes)).strftime('%d/%m/%y'),
                            'EndTime': (breakdown_start_time + timedelta(minutes=end_minutes - start_minutes)).strftime('%H:%M'),
                            'Probability': new_prob
                        }]
                    for breakdown_info in self.alt_aircraft_dict[aircraft_id]:
                        breakdown_info['Probability'] = new_prob

                if DEBUG_MODE:
                    print(f"Aircraft {aircraft_id}: Probability updated from {prob:.2f} to {new_prob:.2f}")
                
                # Invalidate cache to ensure ac_mtx is updated with new probabilities
                self.something_happened = True

                if self.current_datetime + self.timestep >= breakdown_start_time:
                    if DEBUG_MODE_BREAKDOWN:
                        print(f"Rolling the dice for breakdown with updated probability {new_prob} starting at {breakdown_start_time}")

                    # Roll the dice
                    if np.random.rand() < new_prob:
                        if DEBUG_MODE_BREAKDOWN:
                            print(f"Breakdown confirmed for aircraft {aircraft_id} with probability {new_prob:.2f}")
                        
                        # Update unavailabilities_dict
                        self.unavailabilities_dict[aircraft_id]['Probability'] = 1.00
                        
                        # Update alt_aircraft_dict
                        if aircraft_id in self.alt_aircraft_dict:
                            for breakdown_info in self.alt_aircraft_dict[aircraft_id]:
                                breakdown_info['Probability'] = 1.00
                        
                        # Invalidate cache to ensure ac_mtx is updated with new probabilities
                        self.something_happened = True

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

                        # Set probability to 0.00 (no unavailability)
                        self.unavailabilities_dict[aircraft_id]['Probability'] = 0.00
                        self.unavailabilities_dict[aircraft_id]['StartColumn'] = -1
                        self.unavailabilities_dict[aircraft_id]['EndColumn'] = -1
                        
                        # Update alt_aircraft_dict
                        if aircraft_id in self.alt_aircraft_dict:
                            for breakdown_info in self.alt_aircraft_dict[aircraft_id]:
                                breakdown_info['Probability'] = 0.00
                
                # Invalidate cache to ensure ac_mtx is updated with new probabilities
                self.something_happened = True

    def handle_no_conflicts(self, flight_action, aircraft_action):
        """Handles the case when there are no conflicts in the current state using matrix-based approach.

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
                ac_mtx, flight_features = self._get_initial_state()
                processed_state, _ = self.process_observation(ac_mtx, flight_features)
                truncated = False
                
                reward = self._calculate_reward(set(), set(), flight_action, aircraft_action, original_flight_action_departure_time, terminated)
                return processed_state, reward, terminated, truncated, {}

        self.current_datetime = next_datetime
        # Invalidate cache to ensure state reflects new datetime (probabilities may have changed)
        self.something_happened = True
        
        # Get current state matrices for observation processing
        ac_mtx, flight_features = self._get_initial_state()
        processed_state, _ = self.process_observation(ac_mtx, flight_features)
        
        # Calculate reward for the action
        reward = self._calculate_reward(set(), set(), flight_action, aircraft_action, original_flight_action_departure_time, False)
        
        return processed_state, reward, False, False, {}

    def _calculate_reward(self, resolved_conflicts, remaining_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, terminated):
        """Calculates the reward based on the current state of the environment using matrix-based approach.

        The reward consists of several components:
        1. Delay Penalty: Penalty for each minute of delay introduced
        2. Cancellation Penalty: Penalty for each newly cancelled flight
        3. Inaction Penalty: Penalty for taking no action when conflicts exist
        4. Proactive Penalty: Fixed penalty for acting too close to departure
        5. Time Penalty: Small penalty per timestep to encourage faster resolution
        6. Final Resolution Reward: Bonus for resolving real conflicts at scenario end
        7. Automatic Cancellation Penalty: Penalty for flights that have already departed
        8. Probability Resolution Bonus: Reward for resolving high-probability conflicts
        9. Low-confidence Action Penalty: Penalty for acting on low-probability disruptions

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
        # NOTE: Delays solve conflicts, so they should be minimally penalized (or not at all)
        # Only penalize delays that exceed a threshold (e.g., >3 hours) to slightly discourage very long delays
        delay_penalty_minutes = sum(
            self.environment_delayed_flights[flight_id] - self.penalized_delays.get(flight_id, 0)
            for flight_id in self.environment_delayed_flights
        )
        self.scenario_wide_delay_minutes += delay_penalty_minutes
        new_delayed_flights = len([fid for fid in self.environment_delayed_flights if self.environment_delayed_flights[fid] > self.penalized_delays.get(fid, 0)])
        self.scenario_wide_delay_count += new_delayed_flights
        
        if PENALTY_1_DELAY_ENABLED:
            # Only penalize delays exceeding the threshold (e.g., >3 hours)
            # Delays below threshold are free - they solve conflicts, which is good
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
        if PENALTY_2_CANCELLATION_ENABLED:
            new_cancellations = {
                flight_id for flight_id in self.cancelled_flights if flight_id not in self.penalized_cancelled_flights
            }
            cancellation_penalty_count = len(new_cancellations)
            cancel_penalty = cancellation_penalty_count * CANCELLED_FLIGHT_PENALTY
        
            self.scenario_wide_cancelled_flights += cancellation_penalty_count
            self.penalized_cancelled_flights.update(new_cancellations)
        else:
            cancel_penalty = 0
            cancellation_penalty_count = 0
            new_cancellations = set()

        if DEBUG_MODE_REWARD:
            status = "ENABLED" if PENALTY_2_CANCELLATION_ENABLED else "DISABLED"
            print(f"  [Penalty #2: {status}] -{cancel_penalty} penalty for {cancellation_penalty_count} new cancelled flights: {new_cancellations}")

        # 3. Inaction Penalty: Penalize doing nothing when conflicts exist
        if PENALTY_3_INACTION_ENABLED:
            inaction_penalty = NO_ACTION_PENALTY if flight_action == 0 and remaining_conflicts else 0
            if inaction_penalty > 0:
                self.scenario_wide_inaction_count += 1
        else:
            inaction_penalty = 0

        if DEBUG_MODE_REWARD:
            status = "ENABLED" if PENALTY_3_INACTION_ENABLED else "DISABLED"
            print(f"  [Penalty #3: {status}] -{inaction_penalty} penalty for inaction with remaining conflicts")

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
        # Applied per step so that scenarios taking more steps get proportionally more penalty
        # This encourages the agent to resolve conflicts efficiently
        if PENALTY_5_TIME_ENABLED:
            time_penalty = TIMESTEP_HOURS * 60 * TIME_MINUTE_PENALTY  # Penalty per timestep
        else:
            time_penalty = 0

        if DEBUG_MODE_REWARD:
            status = "ENABLED" if PENALTY_5_TIME_ENABLED else "DISABLED"
            print(f"  [Penalty #5: {status}] Time penalty calculation:")
            print(f"    Penalty per timestep: {TIMESTEP_HOURS * 60} minutes * {TIME_MINUTE_PENALTY} = {time_penalty:.4f}")
            print(f"  -{time_penalty} penalty for this timestep")

        # 6. Final Resolution Reward: Bonus for resolving real conflicts at scenario end
        final_conflict_resolution_reward = 0
        scenario_ended_flag = False  # Track if scenario actually ended (for step_info)
        if PENALTY_6_FINAL_REWARD_ENABLED and terminated:
            # Count resolved conflicts for non-cancelled flights (manual or automatic) with probability 1.00
            scenario_ended = self.check_termination_criteria()
            if scenario_ended:
                scenario_ended_flag = True  # Mark that scenario ended and final reward was calculated
                final_resolved_count = 0
                resolved_flights = []
                # Use initial_conflict_combinations if available, otherwise fall back to eligible_flights_for_resolved_bonus
                conflict_combinations = getattr(self, 'initial_conflict_combinations', getattr(self, 'eligible_flights_for_resolved_bonus', []))
                for (aircraft_id, flight_id) in conflict_combinations:
                    if self.unavailabilities_dict[aircraft_id]['Probability'] == 1.00 and flight_id not in self.cancelled_flights and flight_id not in self.automatically_cancelled_flights:
                        final_resolved_count += 1
                        resolved_flights.append(flight_id)

                final_conflict_resolution_reward = final_resolved_count * RESOLVED_CONFLICT_REWARD
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

                # Calculate scenario-wide solution slack
                self._calculate_scenario_wide_solution_slack()
                
                if DEBUG_MODE_REWARD:
                    status = "ENABLED" if PENALTY_6_FINAL_REWARD_ENABLED else "DISABLED"
                    print(f"  [Penalty #6: {status}] +{final_conflict_resolution_reward} final reward for resolving {final_resolved_count} real (non-cancelled, non-auto-cancelled) conflicts at scenario end: {resolved_flights}")
            elif DEBUG_MODE_REWARD:
                print(f"  [Penalty #6] Final reward NOT given because scenario_ended=False (termination criteria not met)")


        # 7. Automatic cancellation of flights that have already departed
        if PENALTY_7_AUTO_CANCELLATION_ENABLED:
            new_automatic_cancellations = {
                flight_id for flight_id in self.automatically_cancelled_flights if flight_id not in self.penalized_automatically_cancelled_flights
            }
            automatic_cancellation_penalty_count = len(new_automatic_cancellations)
            automatic_cancellation_penalty = automatic_cancellation_penalty_count * AUTOMATIC_CANCELLATION_PENALTY
            
            self.scenario_wide_automatically_cancelled_count += automatic_cancellation_penalty_count
            self.scenario_wide_cancelled_flights += automatic_cancellation_penalty_count
            self.penalized_automatically_cancelled_flights.update(new_automatic_cancellations)
        else:
            automatic_cancellation_penalty = 0
            automatic_cancellation_penalty_count = 0
        
        if DEBUG_MODE_REWARD and automatic_cancellation_penalty_count > 0:
            status = "ENABLED" if PENALTY_7_AUTO_CANCELLATION_ENABLED else "DISABLED"
            print(f"  [Penalty #7: {status}] -{automatic_cancellation_penalty} penalty for {automatic_cancellation_penalty_count} automatic cancellations")
            
        # 8. Probability-aware shaping: reward resolving high-probability conflicts
        # Only give bonus if agent took an action that resolved conflicts (not if conflicts disappeared naturally)
        probability_resolution_bonus = 0
        resolved_probability_total = 0
        tail_swap_resolved_conflict = False  # Track if this tail swap resolved a conflict
        if (PENALTY_8_PROBABILITY_RESOLUTION_BONUS_ENABLED 
            and resolved_conflicts 
            and PROBABILITY_RESOLUTION_BONUS_SCALE > 0
            and flight_action != 0  # Agent must have taken an action
            and self.something_happened):  # Action must have changed the state
            for conflict in resolved_conflicts:
                # Handle both 2-tuple (aircraft_id, flight_id) and 4-tuple (aircraft_id, flight_id, flight_dep, flight_arr) formats
                if isinstance(conflict, (tuple, list)) and len(conflict) >= 2:
                    aircraft_id = conflict[0]
                    conflict_flight_id = conflict[1]
                else:
                    aircraft_id = conflict
                    conflict_flight_id = None

                # Only reward conflicts directly resolved by the acted flight
                if conflict_flight_id is None or conflict_flight_id != flight_action:
                    continue
                # Do not reward conflicts that were auto-cancelled by the environment
                if conflict_flight_id in self.automatically_cancelled_flights:
                    continue

                # Use the probability snapshot from BEFORE uncertainties were processed this step
                pre_prob = np.nan
                if hasattr(self, "pre_action_probabilities"):
                    pre_prob = self.pre_action_probabilities.get(aircraft_id, np.nan)
                if np.isnan(pre_prob):
                    pre_prob = self.unavailabilities_dict.get(aircraft_id, {}).get('Probability', np.nan)
                if np.isnan(pre_prob):
                    pre_prob = 1.0  # fallback: treat as certain if unknown

                resolved_probability_total += max(0.0, pre_prob)
                
                # If we got here, this action resolved a conflict
                if self.tail_swap_happened:
                    tail_swap_resolved_conflict = True

            probability_resolution_bonus = resolved_probability_total * PROBABILITY_RESOLUTION_BONUS_SCALE
            
            # Track tail swaps that resolved conflicts (only if bonus was given)
            if tail_swap_resolved_conflict and probability_resolution_bonus > 0:
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
                and hasattr(self, 'last_action_probability')
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

        # Calculate total reward - SIMPLE PENALTY-BASED STRUCTURE
        reward = (
            - delay_penalty_total
            - cancel_penalty
            - inaction_penalty
            - automatic_cancellation_penalty
            - proactive_penalty
            - time_penalty
            - low_confidence_action_penalty
            + final_conflict_resolution_reward  # Only positive reward: final resolution
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

        # Round final reward to 4 decimal places to preserve small penalty values (e.g., time penalty 0.003)
        # This ensures the DQN receives actual penalty values, not rounded zeros
        reward = round(reward, 4)

        if DEBUG_MODE_REWARD:
            print("--------------------------------")
            print(f"Total reward: {reward}")
            print("--------------------------------")

        # Store current unavailability probabilities for tracking evolution
        unavailabilities_probabilities = {}
        interval_minutes = 15  # 15-minute intervals
        for aircraft_id in self.aircraft_ids:
            prob = self.unavailabilities_dict[aircraft_id]['Probability']
            start_col = self.unavailabilities_dict[aircraft_id]['StartColumn']
            end_col = self.unavailabilities_dict[aircraft_id]['EndColumn']
            # Convert column indices to minutes
            start_minutes = float(start_col * interval_minutes) if start_col >= 0 else None
            end_minutes = float(end_col * interval_minutes) if end_col >= 0 else None
            unavailabilities_probabilities[aircraft_id] = {
                'probability': float(prob) if not np.isnan(prob) else None,
                'start_minutes': start_minutes,
                'end_minutes': end_minutes
            }
        
        # Store step information
        self.info_after_step = {
            "total_reward": reward,
            "something_happened": self.something_happened,
            "current_time_minutes": (self.current_datetime - self.earliest_datetime).total_seconds() / 60,
            "current_time_minutes_from_start": (self.current_datetime - self.start_datetime).total_seconds() / 60,  # Add this for debugging
            "start_datetime": str(self.start_datetime),  # Add for debugging
            "earliest_datetime": str(self.earliest_datetime),  # Add for debugging
            "current_datetime": str(self.current_datetime),  # Add for debugging
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
                "final_conflict_resolution_reward": final_conflict_resolution_reward,
                "probability_resolution_bonus": probability_resolution_bonus,
                "low_confidence_action_penalty": low_confidence_action_penalty
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
        
        # Add scenario-wide metrics ONLY when scenario ends (scenario_ended_flag is True)
        # This saves memory by not storing metrics at every step
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

    def _calculate_scenario_wide_solution_slack(self):
        """Calculate the scenario-wide solution slack based on flight schedules using matrix-based approach."""
        # Get current state matrices
        ac_mtx, flight_features = self._get_initial_state()
        
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

        # Organize flights by aircraft using matrix-based approach
        aircraft_flights = {ac: [] for ac in self.aircraft_ids}
        
        # Use flight_features to get flight durations efficiently
        interval_minutes = 15
        for ac_idx in range(flight_features.shape[0]):
            if ac_idx >= len(self.aircraft_ids):
                continue
            
            aircraft_id = self.aircraft_ids[ac_idx]
            aircraft_flights_list = self.get_all_flights_for_aircraft(ac_idx, flight_features)
            
            for flight_id, dep_interval, arr_interval, status in aircraft_flights_list:
                if flight_id not in self.flights_dict or status <= 0:
                    continue
                
                # Calculate duration in minutes using intervals
                flight_duration = (arr_interval - dep_interval + 1) * interval_minutes
                aircraft_flights[aircraft_id].append(flight_duration)

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

    def check_termination_criteria(self):
        """
        Checks if the stopping criteria are met using matrix-based approach:

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

    def step(self, action_index):
        """Executes a step in the environment based on the provided action using matrix-based approach.

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
        ac_mtx, flight_features = self.get_current_state()

        # Fix the state before processing the action (work on copies to avoid modifying internal state)
        ac_mtx_copy = ac_mtx.copy()
        flight_features_copy = flight_features.copy()
        self.fix_state(ac_mtx_copy, flight_features_copy)

        # Print the current state if in debug mode
        if DEBUG_MODE_PRINT_STATE:
            print(f"Current ac_mtx shape: {ac_mtx.shape}, flight_features shape: {flight_features.shape}")
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
        ac_mtx, flight_features = self.get_current_state()
        processed_state, _ = self.process_observation(ac_mtx, flight_features)

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

    def handle_flight_operations(self, flight_action, aircraft_action, pre_action_conflicts):
        """
        Handles flight operation decisions and resolves conflicts using matrix-based approach.

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
            # Invalidate cache to ensure state reflects new datetime
            self.something_happened = True

            post_action_conflicts = self.get_current_conflicts()
            resolved_conflicts = pre_action_conflicts - post_action_conflicts

            terminated, reason = self._is_done()
            truncated = False
            done = terminated or truncated
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, done)

            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            ac_mtx, flight_features = self.get_current_state()
            processed_state, _ = self.process_observation(ac_mtx, flight_features)
            return processed_state, reward, terminated, truncated, {}
        elif aircraft_action == 0:
            # Cancel the flight
            self.cancel_flight(flight_action)
            if DEBUG_MODE_CANCELLED_FLIGHT:
                print(f"Cancelled flight {flight_action}")

            # Proceed to next timestep
            next_datetime = self.current_datetime + self.timestep
            self.current_datetime = next_datetime
            # Invalidate cache to ensure state reflects new datetime
            self.something_happened = True

            post_action_conflicts = self.get_current_conflicts()
            resolved_conflicts = pre_action_conflicts - post_action_conflicts

            terminated, reason = self._is_done()
            truncated = False
            done = terminated or truncated
            reward = self._calculate_reward(resolved_conflicts, post_action_conflicts, flight_action, aircraft_action, original_flight_action_departure_time, done)

            if terminated:
                if DEBUG_MODE_STOPPING_CRITERIA:
                    print(f"Episode ended: {reason}")

            ac_mtx, flight_features = self.get_current_state()
            processed_state, _ = self.process_observation(ac_mtx, flight_features)
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

                ac_mtx, flight_features = self.get_current_state()
                processed_state, _ = self.process_observation(ac_mtx, flight_features)
                return processed_state, reward, terminated, truncated, {}

            current_aircraft_id = self.rotations_dict[selected_flight_id]['Aircraft']

            if selected_aircraft_id == current_aircraft_id:
                # Delay the flight by scheduling it on the same aircraft
                # Get unavailability end time for the aircraft using matrix-based approach
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
            # Invalidate cache to ensure state reflects new datetime
            self.something_happened = True

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

            ac_mtx, flight_features = self._get_initial_state()
            processed_state, _ = self.process_observation(ac_mtx, flight_features)
            return processed_state, reward, terminated, truncated, {}

    def schedule_flight_on_aircraft(self, aircraft_id, flight_id, dep_time, current_aircraft_id, arr_time=None, delayed_flights=None, secondary=False):
        """Schedules a flight on an aircraft using matrix-based approach.

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

        if DEBUG_MODE_SCHEDULING:
            print(f"Original departure minutes: {original_dep_minutes}")
            print(f"Flight duration: {flight_duration}")

        # Ensure dep_time is not earlier than original departure time
        dep_time = max(dep_time, original_dep_minutes)

        if arr_time is None:
            arr_time = dep_time + flight_duration
        else:
            flight_duration = arr_time - dep_time

        # Check for unavailability conflicts using matrix-based approach
        unavail_info = self.unavailabilities_dict.get(aircraft_id, {})
        unavail_start_col = unavail_info.get('StartColumn', -1)
        unavail_end_col = unavail_info.get('EndColumn', -1)
        unavail_prob = unavail_info.get('Probability', 0.0)

        if DEBUG_MODE_SCHEDULING:
            print(f"\nUnavailability check:")
            print(f"Current aircraft: {current_aircraft_id}, Target aircraft: {aircraft_id}")
            print(f"Unavailability - StartCol: {unavail_start_col}, EndCol: {unavail_end_col}, Prob: {unavail_prob}")

        # Check if flight overlaps with unavailability
        has_unavail_overlap = False
        if (unavail_start_col >= 0 and 
            unavail_end_col >= 0 and 
            unavail_prob > 0.0):  # Only check for overlap if there's an actual unavailability
            
            # Convert times to ensure proper comparison
            flight_start = float(original_dep_minutes)
            flight_end = float(original_arr_minutes)
            
            # Convert column indices to minutes
            interval_minutes = 15
            unavail_start_minutes = unavail_start_col * interval_minutes
            unavail_end_minutes = unavail_end_col * interval_minutes
            
            # Check for any overlap between flight and unavailability period
            # A flight overlaps if it doesn't end before the disruption starts
            if flight_end > unavail_start_minutes:
                has_unavail_overlap = True
                
            if DEBUG_MODE_SCHEDULING:
                print(f"\nChecking overlap:")
                print(f"Flight: {flight_start} -> {flight_end}")
                print(f"Unavail: {unavail_start_minutes} -> {unavail_end_minutes}")
                print(f"Overlap detected: {has_unavail_overlap}")

        current_ac_is_same_as_target_ac = aircraft_id == current_aircraft_id
        if not current_ac_is_same_as_target_ac:
            self.something_happened = True
            self.tail_swap_happened = True  # Track that a tail swap occurred

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
                print(f"Unavail times - StartCol: {unavail_start_col}, EndCol: {unavail_end_col}")

            if aircraft_id == current_aircraft_id:
                if unavail_prob > 0.00: #move flight after unavailability
                    if DEBUG_MODE_SCHEDULING:
                        print("Case 1: Current aircraft with prob > 0.00 - Moving flight after unavailability")
                    interval_minutes = 15
                    unavail_end_minutes = unavail_end_col * interval_minutes
                    dep_time = max(dep_time, unavail_end_minutes + MIN_TURN_TIME)
                    dep_time = max(dep_time, original_dep_minutes)
                    arr_time = dep_time + flight_duration
                    delay = dep_time - original_dep_minutes
                    self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + delay
                    self.something_happened = True
                else: #unavail prob = 0.00, keep original schedule
                    if DEBUG_MODE_SCHEDULING:
                        print("Case 2: Current aircraft with prob = 0.00 - Keeping original schedule")
                    self.something_happened = False
            else: #aircraft_id != current_aircraft_id
                if unavail_prob == 1.00: #move flight after unavailability
                    if DEBUG_MODE_SCHEDULING:
                        print("Case 3: Different aircraft with prob = 1.00 - Moving flight after unavailability")
                    interval_minutes = 15
                    unavail_end_minutes = unavail_end_col * interval_minutes
                    dep_time = max(dep_time, unavail_end_minutes + MIN_TURN_TIME)
                    dep_time = max(dep_time, original_dep_minutes)
                    arr_time = dep_time + flight_duration
                    delay = dep_time - original_dep_minutes
                    self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + delay
                    self.something_happened = True
                else: #unavail prob < 1.00, allow overlap
                    if DEBUG_MODE_SCHEDULING:
                        print("Case 4: Different aircraft with prob < 1.00 - Allowing overlap")
                    self.something_happened = True

        # Get all flights on this aircraft using flight_features
        scheduled_flights = []
        ac_mtx, flight_features = self._get_initial_state()
        
        aircraft_flights = self.get_all_flights_for_aircraft(aircraft_idx, flight_features)
        interval_minutes = 15
        
        for existing_flight_id, existing_dep_interval, existing_arr_interval, existing_status in aircraft_flights:
            if existing_flight_id != flight_id and existing_status > 0:  # Skip cancelled flights
                # Convert intervals to minutes
                existing_dep_minutes = existing_dep_interval * interval_minutes
                existing_arr_minutes = existing_arr_interval * interval_minutes
                scheduled_flights.append((existing_flight_id, existing_dep_minutes, existing_arr_minutes))
        
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
                # Check for overlap with previous flight
                if test_dep < current_time + MIN_TURN_TIME:
                    new_dep = current_time + MIN_TURN_TIME
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
        Optimizes the schedule to minimize cascading delays using matrix-based approach.
        
        Args:
            scheduled_flights (list): List of flights to optimize
            aircraft_idx (int): Aircraft index
        """
        current_time = 0
        
        for i, (flight_id, dep_time, arr_time) in enumerate(scheduled_flights):
            # Check for overlap with previous flight
            if dep_time < current_time + MIN_TURN_TIME:
                new_dep_time = current_time + MIN_TURN_TIME
                new_arr_time = new_dep_time + (arr_time - dep_time)
                
                # Update the flight times
                if flight_id == flight_id:  # This is our flight
                    dep_time = new_dep_time
                    arr_time = new_arr_time
                else:
                    # Update the other flight's times
                    self.update_flight_times(flight_id, new_dep_time, new_arr_time)
                
                # Update the scheduled_flights list with new times
                scheduled_flights[i] = (flight_id, new_dep_time, new_arr_time)
                
                # Track the delay (only if flight still exists in flights_dict)
                if flight_id in self.flights_dict:
                    original_dep = parse_time_with_day_offset(
                        self.flights_dict[flight_id]['DepTime'], 
                        self.start_datetime
                    )
                    original_dep_minutes = (original_dep - self.earliest_datetime).total_seconds() / 60
                    delay = new_dep_time - original_dep_minutes
                    self.environment_delayed_flights[flight_id] = self.environment_delayed_flights.get(flight_id, 0) + delay
                
                current_time = new_arr_time
            else:
                current_time = arr_time

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

        # Note: In the matrix-based approach, we don't need to manually update the state
        # as it will be recalculated in _get_initial_state()
        self.something_happened = True

    def get_unresolved_uncertainties(self):
        """Retrieves the unresolved uncertainties in the environment using matrix-based approach.

        This function checks for unavailability periods that are neither 0.0 nor 1.0.

        Returns:
            set: A set of unresolved uncertainties.
        """
        unresolved_uncertainties = set()
        
        # Get the current state matrices
        ac_mtx, flight_features = self._get_initial_state()
        
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

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state using matrix-based approach."""
        # Generate a random seed based on current time if none provided
        if seed is None:
            seed = int(time.time() * 1000000) % (2**32 - 1)
        
        # Set random seeds for all random number generators
        random.seed(seed)
        np.random.seed(seed)
        
        # Rest of the reset method remains unchanged
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

        # Calculate total simulation minutes
        total_simulation_minutes = (self.end_datetime - self.start_datetime).total_seconds() / 60

        # Generate initial state matrices
        ac_mtx, flight_features = self._get_initial_state()
        
        # Update the state and observation space
        self.state = (ac_mtx, flight_features)
        self._update_observation_space()

        # Get initial conflicts using matrix-based approach
        self.initial_conflict_combinations = self.get_initial_conflicts()

        self.swapped_flights = []  # Reset the swapped flights list
        self.environment_delayed_flights = {}  # Reset the delayed flights list
        self.penalized_delays = {}  # Reset the penalized delays
        self.penalized_conflicts = set()
        self.resolved_conflicts = set()
        self.penalized_cancelled_flights = set()  # Reset penalized cancelled flights
        self.automatically_cancelled_flights = set()  # Reset automatically cancelled flights
        self.penalized_automatically_cancelled_flights = set()  # Reset penalized automatically cancelled flights

        self.cancelled_flights = set()

        # Initialize eligible flights for conflict resolution bonus
        self.eligible_flights_for_resolved_bonus = self.get_initial_conflicts()

        # Process the state into an observation as a NumPy array using matrix-based approach
        processed_state, _ = self.process_observation(ac_mtx, flight_features)

        if DEBUG_MODE:
            print(f"ac_mtx shape: {ac_mtx.shape}, flight_features shape: {flight_features.shape}")
            print(f"Type of processed state: {type(processed_state)}")
            print(processed_state)
        return processed_state, {}

    def _is_done(self):
        """Checks if the episode is finished using matrix-based approach.

        The episode is considered done if:
        1. Current time has reached or exceeded the end time, OR
        2. There are no overlaps between flights and disruptions (regardless of uncertainty status)

        Returns:
            tuple: (bool, str) indicating if the episode is done and the reason.
        """
        if self.current_datetime >= self.end_datetime:
            return True, "Reached the end of the simulation time."
        
        # Check for any overlaps between flights and disruptions using matrix-based approach
        if not self.check_flight_disruption_overlaps():
            return True, "No remaining overlaps between flights and disruptions."
        
        return False, ""

    def check_flight_disruption_overlaps(self):
        """Checks if there are any overlaps between flights and disruptions using matrix-based approach.
        
        Returns:
            bool: True if there are overlaps, False otherwise.
        """
        # Get current state matrices
        ac_mtx, flight_features = self._get_initial_state()
        
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
            
            # Check each flight for this aircraft using flight_features
            aircraft_flights = self.get_all_flights_for_aircraft(ac_idx, flight_features)
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

    def get_unresolved_uncertainties(self):
        """Retrieves the unresolved uncertainties in the environment using matrix-based approach.

        This function checks for unavailability periods that are neither 0.0 nor 1.0.

        Returns:
            set: A set of unresolved uncertainties.
        """
        unresolved_uncertainties = set()
        
        # Get the current state matrices
        ac_mtx, flight_features = self._get_initial_state()
        
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