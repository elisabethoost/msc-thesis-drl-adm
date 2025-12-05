"""
Visualize per-scenario metrics for a specific episode.
This script loads the detailed episode data and creates comprehensive visualizations.

Usage:
    python scripts/visualize_episode_metrics.py
    OR
    cd scripts
    python visualize_episode_metrics.py
    
Examples for different models:
    # Model 1 (RF) - results/model1_rf/training/m1_1/3ac-182-green16
    # Update MODEL_FOLDER in main() to: "results/model1_rf/training/m1_1/3ac-182-green16"
    
    # Model 2 (SSF) - results/model2_ssf/training/m2_1/3ac-182-green16
    # Update MODEL_FOLDER in main() to: "results/model2_ssf/training/m2_1/3ac-182-green16"
    
    # Model 3 (SSF Large Dimensions) - results/model3_ssf_large/training/m3_1/3ac-182-green16
    # Update MODEL_FOLDER in main() to: "results/model3_ssf_large/training/m3_1/3ac-182-green16"
    
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import os
import sys
import glob

# Get the script's directory and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Parent of scripts folder (project root)

sns.set_style("darkgrid")

def load_episode_data(model_folder, env_type, seed):
    """Load detailed episode data from pickle file."""
    # If model_folder is relative, make it relative to PROJECT_ROOT
    if not os.path.isabs(model_folder):
        pkl_path = os.path.join(PROJECT_ROOT, model_folder, "detailed_episodes", f"{env_type}_detailed_episodes_seed_{seed}.pkl")
    else:
        pkl_path = f"{model_folder}/detailed_episodes/{env_type}_detailed_episodes_seed_{seed}.pkl"
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Episode data not found: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def plot_episode_metrics(detailed_episode_data, episode, model_folder, env_type, seed):
    """Create comprehensive visualization of metrics for a single episode."""
    
    # Convert model_folder to absolute path if relative
    if not os.path.isabs(model_folder):
        model_folder = os.path.join(PROJECT_ROOT, model_folder)
    
    if episode not in detailed_episode_data:
        available_episodes = list(detailed_episode_data.keys())
        raise ValueError(f"Episode {episode} not found. Available episodes: {available_episodes}")
    
    # Extract metrics for all scenarios in this episode
    scenario_names = []
    delay_counts = []
    delay_minutes = []
    tail_swaps_resolving = []
    tail_swaps_inefficient = []
    tail_swaps_total = []
    manual_cancellations = []
    auto_cancellations = []
    inaction_counts = []
    resolved_conflicts = []
    total_steps = []
    
    for scenario_folder, scenario_data in detailed_episode_data[episode]["scenarios"].items():
        # Metrics only exist if scenario ended properly (penalty #6 was calculated)
        # If scenario hit step limit or ended early, metrics won't be present
        metrics = scenario_data.get("final_scenario_metrics", None)
        
        if metrics is None:
            # Skip scenarios that didn't end properly (e.g., hit step limit)
            continue
        
        # Extract just the scenario name
        scenario_name = scenario_folder.split('/')[-1] if '/' in scenario_folder else scenario_folder
        scenario_names.append(scenario_name)
        
        delay_counts.append(metrics.get('delay_count', 0))
        delay_minutes.append(metrics.get('delay_minutes', 0))
        tail_swaps_resolving.append(metrics.get('tail_swaps_resolving', 0))
        tail_swaps_inefficient.append(metrics.get('tail_swaps_inefficient', 0))
        tail_swaps_total.append(metrics.get('tail_swaps_total', 0))
        manual_cancellations.append(metrics.get('cancelled_flights', 0))
        auto_cancellations.append(metrics.get('automatically_cancelled_count', 0))
        inaction_counts.append(metrics.get('inaction_count', 0))
        resolved_conflicts.append(metrics.get('resolved_initial_conflicts', 0))
        total_steps.append(metrics.get('steps', 0))
    
    # Print info about scenarios
    total_scenarios = len(detailed_episode_data[episode]["scenarios"])
    scenarios_with_metrics = len(scenario_names)
    if scenarios_with_metrics < total_scenarios:
        print(f"Note: {total_scenarios - scenarios_with_metrics} scenarios skipped (didn't end properly - hit step limit)")
    
    if scenarios_with_metrics == 0:
        print("Warning: No scenarios with metrics found! All scenarios may have hit step limit.")
        print("This usually means MAX_STEPS_PER_SCENARIO is too low.")
        return None
    
    # Calculate derived metrics
    total_cancellations = [m + a for m, a in zip(manual_cancellations, auto_cancellations)]
    
    # Create comprehensive visualization with extra height to prevent overlap
    fig = plt.figure(figsize=(20, 14))
    # Adjust top margin to prevent title overlap with subplot titles
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3, top=0.92, bottom=0.08)
    
    x_pos = np.arange(len(scenario_names))
    
    # 1. Delay Count
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x_pos, delay_counts, color='#ff7f0e', alpha=0.8)
    ax1.set_title(f'Number of Delayed Flights\n(Total: {sum(delay_counts)})', fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_xticks([])
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Delay Minutes
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(x_pos, delay_minutes, color='#d62728', alpha=0.8)
    ax2.set_title(f'Total Delay Minutes\n(Total: {sum(delay_minutes):.0f} min = {sum(delay_minutes)/60:.1f}h)', fontweight='bold')
    ax2.set_ylabel('Minutes')
    ax2.set_xticks([])
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Tail Swaps (Stacked)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(x_pos, tail_swaps_resolving, label='Resolving Conflicts', color='#2ca02c', alpha=0.8)
    ax3.bar(x_pos, tail_swaps_inefficient, bottom=tail_swaps_resolving, 
            label='Inefficient', color='#ff7f0e', alpha=0.8)
    ax3.set_title(f'Tail Swaps\n(Resolving: {sum(tail_swaps_resolving)}, Inefficient: {sum(tail_swaps_inefficient)})', 
                  fontweight='bold')
    ax3.set_ylabel('Count')
    ax3.set_xticks([])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Cancellations (Stacked)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(x_pos, manual_cancellations, label='Manual', color='#9467bd', alpha=0.8)
    ax4.bar(x_pos, auto_cancellations, bottom=manual_cancellations, 
            label='Automatic', color='#d62728', alpha=0.8)
    ax4.set_title(f'Cancellations\n(Manual: {sum(manual_cancellations)}, Auto: {sum(auto_cancellations)})', 
                  fontweight='bold')
    ax4.set_ylabel('Count')
    ax4.set_xticks([])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Inaction Count
    ax5 = fig.add_subplot(gs[1, 1])
    bars = ax5.bar(x_pos, inaction_counts, color='#8c564b', alpha=0.8)
    ax5.set_title(f'Inaction (0,0) Actions\n(Total: {sum(inaction_counts)})', fontweight='bold')
    ax5.set_ylabel('Count')
    ax5.set_xticks([])
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. (Removed - Tail Swap Efficiency)
    # This plot has been removed per user request
    
    # 7. Resolved Conflicts
    ax7 = fig.add_subplot(gs[2, 0])
    bars = ax7.bar(x_pos, resolved_conflicts, color='#2ca02c', alpha=0.8)
    ax7.set_title(f'Resolved Initial Conflicts\n(Total: {sum(resolved_conflicts)})', fontweight='bold')
    ax7.set_ylabel('Count')
    ax7.set_xticks([])
    ax7.grid(axis='y', alpha=0.3)
    
    # 8. Total Steps
    ax8 = fig.add_subplot(gs[2, 1])
    bars = ax8.bar(x_pos, total_steps, color='#1f77b4', alpha=0.8)
    ax8.set_title(f'Steps per Scenario\n(Total: {sum(total_steps)}, Avg: {np.mean(total_steps):.1f})', fontweight='bold')
    ax8.set_ylabel('Steps')
    ax8.set_xticks([])
    ax8.grid(axis='y', alpha=0.3)
    
    # 9. Summary Statistics (moved to middle row where efficiency plot was removed)
    ax9 = fig.add_subplot(gs[1, 2])
    ax9.axis('off')
    
    # Add episode title above the summary box
    ax9.text(0.5, 0.98, f'Episode {episode} - {env_type.upper()}, Seed: {seed}', 
             fontsize=12, family='sans-serif', fontweight='bold',
             horizontalalignment='center', verticalalignment='top', 
             transform=ax9.transAxes)
    
    # Compact summary format (removed reward breakdown to save space)
    summary_text = f"""SUMMARY
{'='*35}
Scenarios: {len(scenario_names)}, Steps: {sum(total_steps)}

Delays: {sum(delay_counts)} flights, {sum(delay_minutes):.0f}min ({sum(delay_minutes)/60:.1f}h)
  Avg/flight: {sum(delay_minutes)/max(sum(delay_counts), 1):.1f}min

Tail Swaps: {sum(tail_swaps_total)} total
  Resolving: {sum(tail_swaps_resolving)} ({sum(tail_swaps_resolving)/max(sum(tail_swaps_total), 1):.0%})
  Inefficient: {sum(tail_swaps_inefficient)} ({sum(tail_swaps_inefficient)/max(sum(tail_swaps_total), 1):.0%})

Cancellations: {sum(total_cancellations)} total
  Manual: {sum(manual_cancellations)}, Automatic: {sum(auto_cancellations)}

Actions:
  Inaction: {sum(inaction_counts)}, Active: {sum(total_steps) - sum(inaction_counts)}

Conflicts Resolved: {sum(resolved_conflicts)}"""
    
    # Position summary text with more vertical centering
    ax9.text(0.02, 0.85, summary_text, fontsize=10, family='monospace',
             verticalalignment='top', transform=ax9.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2, pad=0.6))
    
    # Save figure
    output_path = f"{model_folder}/episode_{episode}_{env_type}_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    
    return fig

def main():
    """Main function to run the visualization."""
    
    # Configuration - MODIFY THESE VALUES
    # Examples:
    #   Model 1: "results/model1_rf/training/m1_1/3ac-182-green16"
    #   Model 2: "results/model2_ssf/training/m2_1/3ac-182-green16"
    #   Model 3: "results/model3_ssf_large/training/m3_1/3ac-182-green16"
    MODEL_FOLDER = "results/model1_rf/training/m1_1/3ac-182-green16"  # Path to your model folder (relative to project root)
    ENV_TYPE = "proactive"  # "myopic", "proactive", or "reactive"
    SEED = 232323  # Your training seed
    EPISODE_X = 0  # First episode to visualize (e.g., early training)
    EPISODE_Y = 9  # Second episode to visualize (e.g., late training)
    
    print(f"Loading episode data from: {MODEL_FOLDER}")
    print(f"Environment type: {ENV_TYPE}, Seed: {SEED}")
    print(f"Visualizing Episode {EPISODE_X} and Episode {EPISODE_Y}")
    
    try:
        # Load data
        detailed_episode_data = load_episode_data(MODEL_FOLDER, ENV_TYPE, SEED)
        print(f"✓ Loaded episode data with {len(detailed_episode_data)} episodes")
        
        # Create visualization for Episode X
        print(f"\nCreating visualization for Episode {EPISODE_X}...")
        fig1 = plot_episode_metrics(detailed_episode_data, EPISODE_X, MODEL_FOLDER, ENV_TYPE, SEED)
        
        # Create visualization for Episode Y
        print(f"Creating visualization for Episode {EPISODE_Y}...")
        fig2 = plot_episode_metrics(detailed_episode_data, EPISODE_Y, MODEL_FOLDER, ENV_TYPE, SEED)
        
        print("\n✓ Both visualizations created successfully!")
        plt.show()
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nAvailable detailed episode files:")
        search_folder = os.path.join(PROJECT_ROOT, MODEL_FOLDER) if not os.path.isabs(MODEL_FOLDER) else MODEL_FOLDER
        pattern = os.path.join(search_folder, "detailed_episodes", "*.pkl")
        files = glob.glob(pattern)
        for f in files:
            print(f"  - {os.path.basename(f)}")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

