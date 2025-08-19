import numpy as np
import matplotlib.pyplot as plt
import os

# Load the reward data
numpy_dir = "Save_Trained_Models/6ac-26-lilac/numpy"

# Load individual seed data
myopic_seed1 = np.load(os.path.join(numpy_dir, "myopic_runs_seed_232323.npy"))
myopic_seed2 = np.load(os.path.join(numpy_dir, "myopic_runs_seed_242424.npy"))

proactive_seed1 = np.load(os.path.join(numpy_dir, "proactive_runs_seed_232323.npy"))
proactive_seed2 = np.load(os.path.join(numpy_dir, "proactive_runs_seed_242424.npy"))

reactive_seed1 = np.load(os.path.join(numpy_dir, "reactive_runs_seed_232323.npy"))
reactive_seed2 = np.load(os.path.join(numpy_dir, "reactive_runs_seed_242424.npy"))

# Load aggregated data
all_myopic = np.load(os.path.join(numpy_dir, "all_myopic_runs.npy"))
all_proactive = np.load(os.path.join(numpy_dir, "all_proactive_runs.npy"))
all_reactive = np.load(os.path.join(numpy_dir, "all_reactive_runs.npy"))

print("=== REWARD ANALYSIS ===")
print(f"Myopic seed 1 shape: {myopic_seed1.shape}")
print(f"Myopic seed 2 shape: {myopic_seed2.shape}")
print(f"All myopic shape: {all_myopic.shape}")

print("\n=== REWARD TRENDS ===")
print(f"Myopic seed 1 - First 5 rewards: {myopic_seed1[:5]}")
print(f"Myopic seed 1 - Last 5 rewards: {myopic_seed1[-5:]}")
print(f"Myopic seed 1 - Mean first 100: {np.mean(myopic_seed1[:100]):.2f}")
print(f"Myopic seed 1 - Mean last 100: {np.mean(myopic_seed1[-100:]):.2f}")

print(f"\nMyopic seed 2 - First 5 rewards: {myopic_seed2[:5]}")
print(f"Myopic seed 2 - Last 5 rewards: {myopic_seed2[-5:]}")
print(f"Myopic seed 2 - Mean first 100: {np.mean(myopic_seed2[:100]):.2f}")
print(f"Myopic seed 2 - Mean last 100: {np.mean(myopic_seed2[-100:]):.2f}")

print(f"\nProactive seed 1 - First 5 rewards: {proactive_seed1[:5]}")
print(f"Proactive seed 1 - Last 5 rewards: {proactive_seed1[-5:]}")
print(f"Proactive seed 1 - Mean first 100: {np.mean(proactive_seed1[:100]):.2f}")
print(f"Proactive seed 1 - Mean last 100: {np.mean(proactive_seed1[-100:]):.2f}")

print(f"\nReactive seed 1 - First 5 rewards: {reactive_seed1[:5]}")
print(f"Reactive seed 1 - Last 5 rewards: {reactive_seed1[-5:]}")
print(f"Reactive seed 1 - Mean first 100: {np.mean(reactive_seed1[:100]):.2f}")
print(f"Reactive seed 1 - Mean last 100: {np.mean(reactive_seed1[-100:]):.2f}")

# Calculate trend (slope of linear regression)
def calculate_trend(rewards):
    x = np.arange(len(rewards))
    slope = np.polyfit(x, rewards, 1)[0]
    return slope

print("\n=== TREND ANALYSIS ===")
print(f"Myopic seed 1 trend (slope): {calculate_trend(myopic_seed1):.2f}")
print(f"Myopic seed 2 trend (slope): {calculate_trend(myopic_seed2):.2f}")
print(f"Proactive seed 1 trend (slope): {calculate_trend(proactive_seed1):.2f}")
print(f"Proactive seed 2 trend (slope): {calculate_trend(proactive_seed2):.2f}")
print(f"Reactive seed 1 trend (slope): {calculate_trend(reactive_seed1):.2f}")
print(f"Reactive seed 2 trend (slope): {calculate_trend(reactive_seed2):.2f}")

# Plot the trends
plt.figure(figsize=(15, 10))

# Plot individual seeds
plt.subplot(2, 3, 1)
plt.plot(myopic_seed1, label='Seed 232323', alpha=0.7)
plt.plot(myopic_seed2, label='Seed 242424', alpha=0.7)
plt.title('Myopic Rewards by Seed')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(proactive_seed1, label='Seed 232323', alpha=0.7)
plt.plot(proactive_seed2, label='Seed 242424', alpha=0.7)
plt.title('Proactive Rewards by Seed')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(reactive_seed1, label='Seed 232323', alpha=0.7)
plt.plot(reactive_seed2, label='Seed 242424', alpha=0.7)
plt.title('Reactive Rewards by Seed')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

# Plot aggregated data with mean and std
plt.subplot(2, 3, 4)
mean_myopic = np.mean(all_myopic, axis=0)
std_myopic = np.std(all_myopic, axis=0)
episodes = np.arange(len(mean_myopic))
plt.plot(episodes, mean_myopic, label='Mean', color='blue')
plt.fill_between(episodes, mean_myopic - std_myopic, mean_myopic + std_myopic, alpha=0.3, color='blue')
plt.title('Myopic Aggregated Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 5)
mean_proactive = np.mean(all_proactive, axis=0)
std_proactive = np.std(all_proactive, axis=0)
episodes = np.arange(len(mean_proactive))
plt.plot(episodes, mean_proactive, label='Mean', color='orange')
plt.fill_between(episodes, mean_proactive - std_proactive, mean_proactive + std_proactive, alpha=0.3, color='orange')
plt.title('Proactive Aggregated Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 6)
mean_reactive = np.mean(all_reactive, axis=0)
std_reactive = np.std(all_reactive, axis=0)
episodes = np.arange(len(mean_reactive))
plt.plot(episodes, mean_reactive, label='Mean', color='green')
plt.fill_between(episodes, mean_reactive - std_reactive, mean_reactive + std_reactive, alpha=0.3, color='green')
plt.title('Reactive Aggregated Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('zz_reward_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== SUMMARY ===")
print("All reward trends are negative, indicating the agent is performing worse over time.")
print("This suggests potential issues with:")
print("1. Learning rate too high")
print("2. Reward function design")
print("3. Environment complexity")
print("4. Exploration strategy")
print("5. Network architecture")
