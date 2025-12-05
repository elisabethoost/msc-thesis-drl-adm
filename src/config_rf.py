import numpy as np

# General Environment Settings
MAX_AIRCRAFT = 3  # Maximum aircraft considered in the environment
MAX_FLIGHTS_PER_AIRCRAFT = 17  # Maximum number of flights per aircraft
ROWS_STATE_SPACE = 1 + MAX_AIRCRAFT  

# Time-based state space calculation
# Maximum recovery period: 24 hours in 15-minute intervals
MAX_RECOVERY_HOURS = 24  # Maximum recovery period in hours
MAX_RECOVERY_INTERVALS = (MAX_RECOVERY_HOURS * 60) // 15  
MAX_TIME_INTERVALS = MAX_RECOVERY_INTERVALS
COLUMNS_STATE_SPACE = MAX_TIME_INTERVALS  

# Calculate the flattened action space size
ACTION_SPACE_SIZE = (MAX_AIRCRAFT + 1) * (MAX_FLIGHTS_PER_AIRCRAFT + 1)  # Number of possible actions (+1 for the zero for no action and cancellations)


# Data Generation Settings
DEPARTURE_AFTER_END_RECOVERY = 2 # how many hours after the end of the recovery period can a generated flight depart

# Time Settings for intervals
TIMESTEP_HOURS = 1  # Length of each timestep in hours


DUMMY_VALUE = -999  # Dummy value for padding

# ============================================================================
# NEGATIVE-ONLY REWARD STRUCTURE
# Goal: Agent learns swap/delay > wait > manual_cancel > auto_cancel
# Optimal path: Resolve conflicts early via swap/delay, then wait for probabilities to resolve
# ============================================================================

# EPISODE-END PENALTIES (applied at termination)
UNRESOLVED_CONFLICT_PENALTY = 50.0    # -100 per initial conflict NOT properly resolved
                                        # "Properly resolved" = prob=1.00, not cancelled, not auto-cancelled
                                        # This is the BIG penalty that drives learning

# PER-STEP PENALTIES (applied during episode)
TIME_MINUTE_PENALTY = 0.0001           # -0.006 per step (60 min * 0.0001)
                                        # Small but encourages faster resolution
                                        # Example: 100 steps = -0.6 total time penalty
NO_ACTION_PENALTY = 10.0                 # -1.0 for inaction when conflicts exist
                                        # Encourages agent to act on conflicts
CANCELLED_FLIGHT_PENALTY = 20.0         # -10.0 per manual cancellation
                                        # Bad, but not as bad as auto-cancellation
AUTOMATIC_CANCELLATION_PENALTY = 50.0   # -50.0 per automatic cancellation
                                        # Very bad - means agent failed to act in time

# DELAY PENALTIES (currently disabled - delays are good because they solve conflicts)
DELAY_MINUTE_PENALTY = 0.0001           # Penalty per minute of delay (only if enabled)
DELAY_PENALTY_THRESHOLD_MINUTES = 300   # Only penalize delays exceeding 3 hours
MAX_DELAY_PENALTY = 25000               # Cap on delay penalty

# OTHER PENALTIES (currently disabled)
AHEAD_PENALTY = 0.01                    # Fixed penalty for last-minute actions
LOW_CONFIDENCE_ACTION_THRESHOLD = 0.4   # Actions below this probability are low-confidence
LOW_CONFIDENCE_ACTION_PENALTY = 0.05     # Penalty for acting on low-confidence disruptions

# BONUS SCALES (set to 0 for negative-only rewards)
PROBABILITY_RESOLUTION_BONUS_SCALE = 50  
RESOLVED_CONFLICT_REWARD = 10           # Legacy value (not used with negative-only structure)

# Penalty Enable/Disable Flags for Incremental Testing
# Set to True to enable, False to disable
# Start with only PENALTY_1_ENABLED = True, then gradually enable others
PENALTY_1_DELAY_ENABLED = False          # Delay penalty
PENALTY_2_CANCELLATION_ENABLED = False  # Cancellation penalty
PENALTY_3_INACTION_ENABLED = False      # Inaction penalty
PENALTY_4_PROACTIVE_ENABLED = False     # Proactive penalty (last-minute actions)
PENALTY_5_TIME_ENABLED = False          # Time penalty (cumulative)
PENALTY_6_FINAL_REWARD_ENABLED = False  # Final conflict resolution reward (not a penalty, but can be disabled)
PENALTY_7_AUTO_CANCELLATION_ENABLED = False  # Automatic cancellation penalty
PENALTY_8_PROBABILITY_RESOLUTION_BONUS_ENABLED = True  # Probability-weighted resolution bonus
PENALTY_9_LOW_CONFIDENCE_ACTION_ENABLED = False  # Low-confidence action penalty

# Feature engineering / temporal context settings
ENABLE_TEMPORAL_DERIVED_FEATURES = True   # Append engineered temporal features per aircraft
DERIVED_FEATURES_PER_AIRCRAFT = 4         # time_to_start, time_to_end, prob_slope, normalized_time_to_start
OBS_STACK_SIZE = 2                        # Number of consecutive observations stacked for temporal context
STACKING_PADDING_VALUE = 0.0              # Value used to pad the history before enough frames exist



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
