import numpy as np

# General Environment Settings
MAX_AIRCRAFT = 6  # Maximum number of aircraft considered in the environment
MAX_FLIGHTS_PER_AIRCRAFT = 20  # Maximum number of flights per aircraft
ROWS_STATE_SPACE = 1 + MAX_AIRCRAFT  # Number of rows in the state space

# Time-based state space calculation
# Maximum recovery period: 24 hours (1440 minutes) converted to 15-minute intervals
MAX_RECOVERY_HOURS = 24  # Maximum recovery period in hours
MAX_RECOVERY_INTERVALS = (MAX_RECOVERY_HOURS * 60) // 15  # Convert to 15-minute intervals
# ac_mtx: (aircraft, time_intervals) + fl_mtx: (flights, time_intervals + 1 for aircraft allocation)
MAX_TIME_INTERVALS = MAX_RECOVERY_INTERVALS
COLUMNS_STATE_SPACE = MAX_TIME_INTERVALS  # Number of columns in the state space (time intervals)

# Calculate the flattened action space size
ACTION_SPACE_SIZE = (MAX_AIRCRAFT + 1) * (MAX_FLIGHTS_PER_AIRCRAFT + 1)  # Number of possible actions (+1 for the zero for no action and cancellations)


# Data Generation Settings
DEPARTURE_AFTER_END_RECOVERY = 2 # how many hours after the end of the recovery period can a generated flight depart

# Time Settings for intervals
TIMESTEP_HOURS = 1  # Length of each timestep in hours


DUMMY_VALUE = -999  # Dummy value for padding


# Reward and Penalty Values - IMPROVED FOR PROACTIVE LEARNING
RESOLVED_CONFLICT_REWARD = 30000     # Reward for resolving a conflict
DELAY_MINUTE_PENALTY = 50           # Penalty per minute of delay
MAX_DELAY_PENALTY = 25000000            # Maximum penalty for delay
NO_ACTION_PENALTY = 0.2              # Penalty for no action while conflict(s) exist
#  NO_ACTION_PENALTY = 10 
CANCELLED_FLIGHT_PENALTY = 5000    # Penalty for cancelling a flight (increased from 5000)
LAST_MINUTE_THRESHOLD = 120           # Threshold for last-minute changes in minutes
LAST_MINUTE_FLIGHT_PENALTY = 300      # Penalty for last-minute flight changes  
AHEAD_BONUS_PER_MINUTE = 0.05                # Reward for proactive flight changes
TIME_MINUTE_PENALTY = 0.1                 # penalty for every minute passed, each timestep cumulatively
TERMINATION_REWARD = 0                  # Reward for terminating the episode
TAIL_SWAP_COST = 100             # Penalty for swapping a flight to a different aircraft

# NEW PROACTIVE LEARNING REWARDS AND PENALTIES
OVERLAP_PENALTY_PER_MINUTE = 10000   # High penalty for each minute of overlap with unavailability
PROACTIVE_MOVE_REWARD = 1000         # Reward for moving flights before conflicts occur
PROACTIVE_THRESHOLD_HOURS = 2        # Hours before conflict to consider action "proactive"

'''

# FIXED Reward and Penalty Values - SCALED DOWN BY 100x
# This should help with the learning stability
RESOLVED_CONFLICT_REWARD = 300     # Reduced from 30000 (100x smaller)
DELAY_MINUTE_PENALTY = 0.5         # Reduced from 50 (100x smaller)
MAX_DELAY_PENALTY = 250000         # Reduced from 25000000 (100x smaller)
NO_ACTION_PENALTY = 0.2            # Reduced from 10.0 (100x smaller)
CANCELLED_FLIGHT_PENALTY = 500     # Reduced from 50000 (100x smaller)
LAST_MINUTE_THRESHOLD = 120        # Threshold for last-minute changes in minutes
LAST_MINUTE_FLIGHT_PENALTY = 3     # Reduced from 300 (100x smaller)
AHEAD_BONUS_PER_MINUTE = 0.0005    # Reduced from 0.05 (100x smaller)
TIME_MINUTE_PENALTY = 0.001        # Reduced from 0.1 (100x smaller)
TERMINATION_REWARD = 0             # Reward for terminating the episode
TAIL_SWAP_COST = 10                # Reduced from 1000 (100x smaller)


'''


# Environment Settings
MIN_TURN_TIME = 0  # Minimum gap between flights for the same aircraft
MIN_BREAKDOWN_PROBABILITY = 0
# Logging and Debug Settings
DEBUG_MODE = False # Turn on/off debug mode
DEBUG_MODE_TRAINING = False  # Turn on/off debug mode for training
DEBUG_MODE_REWARD = False    # Turn on/off debug mode for reward calculation
DEBUG_MODE_PRINT_STATE = False         # Turn on/off debug mode for printing state
DEBUG_MODE_CANCELLED_FLIGHT = False  # Turn on/off debug mode for cancelled flight
DEBUG_MODE_VISUALIZATION = False
DEBUG_MODE_BREAKDOWN = False  # Turn on/off debug mode for breakdowns (so rolling the dice etc)
DEBUG_MODE_ACTION = False
DEBUG_MODE_STOPPING_CRITERIA = False
DEBUG_MODE_SCHEDULING = False
DEBUG_MODE_REWARD_LAST_MINUTE_PENALTY = False  # Turn on/off debug mode for reward calculation last minute penalty
DEBUG_MODE_REWARD_RESOLVED_CONFLICTS = False
DEBUG_MODE_DELAY_MINUTES = False
DEBUG_MODE_ACTION_EVALUATION = False  # Turn on/off debug mode for action evaluation and filtering
