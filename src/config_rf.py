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

RESOLVED_CONFLICT_REWARD = 5     # Reward for resolving a conflict (increased for steeper reward slope)
DELAY_MINUTE_PENALTY = 0.020          # Penalty per minute of delay (restored from good config)
CANCELLED_FLIGHT_PENALTY = 5    # Penalty for cancelling a flight (restored from good config)
NO_ACTION_PENALTY = 1              # Penalty for no action while conflict(s) exist (CRITICAL: low value for smooth learning)
AHEAD_PENALTY = 1               # Fixed penalty for last-minute actions (less than 3h delay penalty)
TIME_MINUTE_PENALTY = 0.001                 # penalty for every minute passed, each timestep cumulatively
AUTOMATIC_CANCELLATION_PENALTY = 7 # SHOULD BE HIGHER THAN CANCELLED_FLIGHT_PENALTY as this the agent has not acted in time in this case
MAX_DELAY_PENALTY = 25000         # Restored from good config

# Penalty Enable/Disable Flags for Incremental Testing
# Set to True to enable, False to disable
# Start with only PENALTY_1_ENABLED = True, then gradually enable others
PENALTY_1_DELAY_ENABLED = True          # Delay penalty
PENALTY_2_CANCELLATION_ENABLED = True  # Cancellation penalty
PENALTY_3_INACTION_ENABLED = False      # Inaction penalty
PENALTY_4_PROACTIVE_ENABLED = False     # Proactive penalty (last-minute actions)
PENALTY_5_TIME_ENABLED = True          # Time penalty (cumulative)
PENALTY_6_FINAL_REWARD_ENABLED = True  # Final conflict resolution reward (not a penalty, but can be disabled)
PENALTY_7_AUTO_CANCELLATION_ENABLED = True  # Automatic cancellation penalty

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
