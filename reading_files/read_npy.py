"""
This file was created so that you can read and plot the npy files of type: 
    {env_type}_steps_runs_seed_{seed}.npy and {env_type}_runs_seed_{seed}.npy

Usage:
python reading_files/read_npy.py <relative_path_to_npy_file>
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # this is the path to the reading_files folder
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR) # this is the path to the workspace folder

def read_and_plot_npy(file_path):
    # Load the data
    data = np.load(file_path)
    file_name = os.path.basename(file_path)
    
    # Print basic information
    print(f"\nFile: {file_path}")
    print(f"Shape: {data.shape}") # the amount of episodes we had during the training 
    print(f"Data type: {data.dtype}")
    print("\nFirst few values:")
    print(data[:10])  # Show first 10 values

    plt.figure(figsize=(10, 5))
    # Per-seed reward file: 1D, rewards
    if file_name.endswith("_runs_seed_232323.npy") or file_name.endswith("_runs_seed_242424.npy") or ("_runs_seed_" in file_name and data.ndim == 1):
        plt.plot(data)
        plt.title(f'Rewards from {file_name}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
    # Per-seed steps file: 1D, steps
    elif file_name.endswith("_steps_runs_seed_232323.npy") or file_name.endswith("_steps_runs_seed_242424.npy") or ("_steps_runs_seed_" in file_name and data.ndim == 1):
        plt.plot(data)
        plt.title(f'Total Timesteps from {file_name}')
        plt.xlabel('Episode')
        plt.ylabel('Total Timesteps')
    # Aggregated rewards: 2D, shape (num_seeds, num_episodes)
    elif file_name.startswith("all_") and file_name.endswith("_runs.npy") and data.ndim == 2:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        episodes = np.arange(mean.shape[0])
        plt.plot(episodes, mean, label='Mean Reward')
        plt.fill_between(episodes, mean-std, mean+std, alpha=0.2, label='Std Dev')
        plt.title(f'Aggregated Rewards from {file_name}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
    # Aggregated steps: 2D, shape (num_seeds, num_episodes)
    elif file_name.startswith("all_") and file_name.endswith("_steps_runs.npy") and data.ndim == 2:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        episodes = np.arange(mean.shape[0])
        plt.plot(episodes, mean, label='Mean Total Timesteps')
        plt.fill_between(episodes, mean-std, mean+std, alpha=0.2, label='Std Dev')
        plt.title(f'Aggregated Total Timesteps from {file_name}')
        plt.xlabel('Episode')
        plt.ylabel('Total Timesteps')
        plt.legend()
    # Fallback: generic 1D
    elif data.ndim == 1:
        plt.plot(data)
        plt.title(f'Data from {file_name}')
        plt.xlabel('Index')
        plt.ylabel('Value')
    else:
        print("Data shape not recognized for plotting.")
        return
    plt.grid(True)
    plt.show()

def main():
    if len(sys.argv) < 2: 
        # Default path if no argument provided
        # change this to whichever results folder you are working on (33-run, 34-run, etc.)
        numpy_dir = os.path.join(WORKSPACE_ROOT, "12-run", "3ac-100-superdiverse", "numpy")
        print(f"No path provided. Looking for numpy files in: {numpy_dir}")
        
        if not os.path.exists(numpy_dir):
            print(f"Directory not found: {numpy_dir}")
            print("Usage: python read_npy.py <path_to_npy_file>")
            return

        # Process all NPY files in directory
        for file_name in os.listdir(numpy_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(numpy_dir, file_name)
                read_and_plot_npy(file_path)
    else:
        # Use provided path
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        read_and_plot_npy(file_path)

if __name__ == "__main__":
    main() 