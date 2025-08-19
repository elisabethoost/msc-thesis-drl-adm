import numpy as np
import matplotlib.pyplot as plt
import os

# Load the reward data
numpy_dir = "Save_Trained_Models/6ac-26-lilac/numpy"

# Load individual seed data
myopic_seed1 = np.load(os.path.join(numpy_dir, "myopic_runs_seed_232323.npy"))
myopic_seed2 = np.load(os.path.join(numpy_dir, "myopic_runs_seed_242424.npy"))

print("=== COMPREHENSIVE REWARD ANALYSIS ===")
print("=" * 50)

print("\n1. REWARD MAGNITUDE ANALYSIS:")
print(f"   Myopic seed 1 - Mean reward: {np.mean(myopic_seed1):.2f}")
print(f"   Myopic seed 1 - Std reward: {np.std(myopic_seed1):.2f}")
print(f"   Myopic seed 1 - Min reward: {np.min(myopic_seed1):.2f}")
print(f"   Myopic seed 1 - Max reward: {np.max(myopic_seed1):.2f}")

print(f"\n   Myopic seed 2 - Mean reward: {np.mean(myopic_seed2):.2f}")
print(f"   Myopic seed 2 - Std reward: {np.std(myopic_seed2):.2f}")
print(f"   Myopic seed 2 - Min reward: {np.min(myopic_seed2):.2f}")
print(f"   Myopic seed 2 - Max reward: {np.max(myopic_seed2):.2f}")

print("\n2. REWARD TREND ANALYSIS:")
# Calculate trend using linear regression
def calculate_trend(rewards):
    x = np.arange(len(rewards))
    slope = np.polyfit(x, rewards, 1)[0]
    return slope

trend1 = calculate_trend(myopic_seed1)
trend2 = calculate_trend(myopic_seed2)

print(f"   Myopic seed 1 trend (slope): {trend1:.2f}")
print(f"   Myopic seed 2 trend (slope): {trend2:.2f}")
print(f"   Average trend: {(trend1 + trend2) / 2:.2f}")

print("\n3. REWARD COMPONENT ANALYSIS:")
print("   Based on config.py, the reward consists of:")
print("   - DELAY_MINUTE_PENALTY = 50 (per minute of delay)")
print("   - CANCELLED_FLIGHT_PENALTY = 50000 (per cancelled flight)")
print("   - NO_ACTION_PENALTY = 10.0 (when conflicts exist)")
print("   - TIME_MINUTE_PENALTY = 0.1 (per minute of simulation time)")
print("   - RESOLVED_CONFLICT_REWARD = 30000 (per resolved conflict)")
print("   - TAIL_SWAP_COST = 1000 (per aircraft swap)")

print("\n4. PROBLEM IDENTIFICATION:")
print("   The extremely negative rewards (-100,000 to -120,000) suggest:")
print("   a) Massive delays accumulating over time")
print("   b) Multiple flight cancellations")
print("   c) Long simulation times (TIME_MINUTE_PENALTY accumulates)")
print("   d) Failed conflict resolution")

print("\n5. ROOT CAUSE ANALYSIS:")
print("   The negative trend suggests the agent is:")
print("   a) Learning to take actions that create more problems")
print("   b) Not effectively resolving conflicts")
print("   c) Accumulating delays instead of reducing them")
print("   d) Possibly over-exploiting learned patterns that don't generalize")

print("\n6. POTENTIAL SOLUTIONS:")
print("   a) Reduce learning rate (currently 0.0001)")
print("   b) Adjust reward scaling (rewards are too large)")
print("   c) Modify exploration strategy")
print("   d) Simplify the environment complexity")
print("   e) Add reward normalization")
print("   f) Implement curriculum learning")

print("\n7. SPECIFIC RECOMMENDATIONS:")
print("   a) Scale down all reward components by 10x or 100x")
print("   b) Reduce learning rate to 0.00001")
print("   c) Increase exploration (higher epsilon_min)")
print("   d) Add reward clipping")
print("   e) Implement experience replay prioritization")
print("   f) Consider using a simpler reward function initially")

# Calculate reward statistics over time windows
window_size = 100
num_windows = len(myopic_seed1) // window_size

print(f"\n8. REWARD EVOLUTION OVER TIME (windows of {window_size} episodes):")
for i in range(min(5, num_windows)):
    start_idx = i * window_size
    end_idx = (i + 1) * window_size
    window_mean = np.mean(myopic_seed1[start_idx:end_idx])
    print(f"   Window {i+1} (episodes {start_idx}-{end_idx-1}): {window_mean:.2f}")

if num_windows > 5:
    print(f"   ...")
    start_idx = (num_windows - 1) * window_size
    end_idx = len(myopic_seed1)
    window_mean = np.mean(myopic_seed1[start_idx:end_idx])
    print(f"   Window {num_windows} (episodes {start_idx}-{end_idx-1}): {window_mean:.2f}")

print("\n9. CONCLUSION:")
print("   The agent is clearly learning suboptimal behavior, likely due to:")
print("   - Reward function design issues (too harsh penalties)")
print("   - Learning rate too high for the reward scale")
print("   - Environment complexity overwhelming the learning process")
print("   - Exploration strategy not allowing proper policy improvement")

print("\n   Immediate action needed: Reduce reward magnitudes and learning rate.")
