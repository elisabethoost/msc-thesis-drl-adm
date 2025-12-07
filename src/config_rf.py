import numpy as np

# General Environment Settings
MAX_AIRCRAFT = 3  
MAX_FLIGHTS_PER_AIRCRAFT = 17  
ROWS_STATE_SPACE = 1 + MAX_AIRCRAFT 
COLUMNS_STATE_SPACE = 3 + 3 * MAX_FLIGHTS_PER_AIRCRAFT  

# Calculate the flattened action space size
ACTION_SPACE_SIZE = (MAX_AIRCRAFT + 1) * (MAX_FLIGHTS_PER_AIRCRAFT + 1)  # Number of possible actions (+1 for the zero for no action and cancellations)

DEPARTURE_AFTER_END_RECOVERY = 2 
TIMESTEP_HOURS = 1 
DUMMY_VALUE = -999  # Dummy value for padding

# ============================================================================
# NEGATIVE-ONLY REWARD STRUCTURE!!! (Reward #8 is the only positive reward)
# Goal: Agent learns swap/delay > wait > manual_cancel > auto_cancel
# Optimal path: Resolve conflicts early via swap/delay, then wait for probabilities to resolve
# ============================================================================

# EPISODE-END PENALTIES (applied at termination)
UNRESOLVED_CONFLICT_PENALTY = 50.0      # -50 per initial conflict NOT properly resolved
                                        # "Properly resolved" = prob=1.00, not cancelled, not auto-cancelled
                                        # BIG penalty that drives learning
# Only positive Reward in this configuration: given immediately for resolving a conflict (so not at end of scenario)
PROBABILITY_RESOLUTION_BONUS_SCALE = 50  

# PER-STEP PENALTIES
TIME_MINUTE_PENALTY = 0.0001            # -0.006 per step (60 min * 0.0001)
NO_ACTION_PENALTY = 10.0                # -10.0 for inaction when conflicts exist
CANCELLED_FLIGHT_PENALTY = 20.0         # -20.0 per manual cancellation: bad, but not as bad as auto-cancellation
AUTOMATIC_CANCELLATION_PENALTY = 50.0   # -50.0 per automatic cancellation: very bad: means agent failed to act in time

# DELAY PENALTIES: delays are good because they solve conflicts
DELAY_MINUTE_PENALTY = 0.0001           # Penalty per minute of delay (only if enabled)
DELAY_PENALTY_THRESHOLD_MINUTES = 300   # Only penalize delays exceeding 6 hours
MAX_DELAY_PENALTY = 25000               # Cap on delay penalty

# OTHER PENALTIES (currently disabled)
AHEAD_PENALTY = 0.01                    # Fixed penalty for last-minute actions
LOW_CONFIDENCE_ACTION_THRESHOLD = 0.4   # Actions below this probability are low-confidence
LOW_CONFIDENCE_ACTION_PENALTY = 0.05     # Penalty for acting on low-confidence disruptions


PENALTY_1_DELAY_ENABLED = True          
PENALTY_2_CANCELLATION_ENABLED = True  
PENALTY_3_INACTION_ENABLED = True      
PENALTY_4_PROACTIVE_ENABLED = False     
PENALTY_5_TIME_ENABLED = True          
PENALTY_6_FINAL_REWARD_ENABLED = True  
PENALTY_7_AUTO_CANCELLATION_ENABLED = True  
PENALTY_8_PROBABILITY_RESOLUTION_BONUS_ENABLED = True  
PENALTY_9_LOW_CONFIDENCE_ACTION_ENABLED = True  

# Feature engineering / temporal context settings
ENABLE_TEMPORAL_DERIVED_FEATURES = True   # Append engineered temporal features per aircraft
DERIVED_FEATURES_PER_AIRCRAFT = 4         # time_to_start, time_to_end, prob_slope, normalized_time_to_start
OBS_STACK_SIZE = 2                        # Number of consecutive observations stacked for temporal context
STACKING_PADDING_VALUE = 0.0              # Value used to pad the history before enough frames exist

# Environment Settings
MIN_TURN_TIME = 0  # Minimum gap between flights for the same aircraft

# Logging and Debug Settings
DEBUG_MODE = False 
DEBUG_MODE_TRAINING = False  
DEBUG_MODE_REWARD = False    
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
