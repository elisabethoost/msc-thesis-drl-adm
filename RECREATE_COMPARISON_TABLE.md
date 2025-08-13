# How to Recreate the Comparison Table

This guide explains how to recreate the comparison table that shows mean cancelled flights, delays, tail swaps, and runtime for different models.

## Overview

The workflow consists of three main steps:
1. **Training**: Train models using `main.py` and `train_dqn_modular.py`
2. **Inference**: Run trained models on test scenarios to collect metrics
3. **Analysis**: Use the inference results to create the comparison table

## Step 1: Training Models

### Option A: Using main.py (Recommended)
```bash
# Train all models for all seeds and environment types
python main.py

# Or train specific seed and environment type
python main.py --seed 232323 --env_type myopic --training_folder "Data/TRAINING/6ac-700-diverse"
```

### Option B: Using train_dqn_modular.py directly
```python
from train_dqn_modular import run_train_dqn_both_timesteps

# Run training for a specific configuration
rewards = run_train_dqn_both_timesteps(
    MAX_TOTAL_TIMESTEPS=10000,
    single_seed=232323,
    brute_force_flag=False,
    cross_val_flag=False,
    early_stopping_flag=False,
    CROSS_VAL_INTERVAL=1,
    printing_intermediate_results=False,
    TRAINING_FOLDERS_PATH="Data/TRAINING/6ac-700-diverse",
    stripped_scenario_folder="6ac-700-diverse",
    save_folder="your_save_folder",
    save_results_big_run="your_save_folder/6ac-700-diverse",
    TESTING_FOLDERS_PATH="Data/TESTING/6ac-700-diverse",
    env_type="myopic"
)
```

## Step 2: Running Inference

### Option A: Using the Python Script
```bash
python run_inference.py
```

### Option B: Using the Jupyter Notebook
1. Open `notebooks_utils/run_inference_notebook.ipynb`
2. Update the configuration section with your paths:
   ```python
   # Update these paths to your actual trained models
   model_paths = [
       ("trained_models/dqn/6ac-700-diverse/myopic_232323.zip", "myopic"),
       ("trained_models/dqn/6ac-700-diverse/proactive_232323.zip", "proactive"),
       ("trained_models/dqn/6ac-700-diverse/reactive_232323.zip", "reactive"),
       ("greedy_reactive", "greedy_reactive"),  # Baseline
   ]
   
   # Update data folder to your test scenarios
   data_folder = "Data/TRAINING/6ac-700-diverse"
   ```
3. Run all cells

This will create the file: `logs/scenarios/final_results_df_in_seeds_100.csv`

## Step 3: Creating the Comparison Table

### Option A: Using the existing notebook
1. Open `interpretation/qualitative-analysis/final-results-examples.ipynb`
2. The notebook will automatically load the CSV file created in Step 2
3. Run the cells to generate the comparison table

### Option B: Manual creation
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the inference results
results_df = pd.read_csv("logs/scenarios/final_results_df_in_seeds_100.csv")

# First aggregate by Model and Seed, then calculate mean and std across seeds
comparison_table = (
    results_df
    .groupby(['Model', 'Seed'])
    .agg(
        TotalReward=('TotalReward', 'mean'),
        ScenarioTime=('ScenarioTime', 'mean'), 
        ScenarioSteps=('ScenarioSteps', 'mean'),
        TotalDelays=('TotalDelays', 'mean'),
        TotalCancelledFlights=('TotalCancelledFlights', 'mean'),
        TotalTailSwaps=('TailSwaps', 'mean')
    )
    .groupby('Model')
    .agg(
        Mean_Reward=('TotalReward', 'mean'),
        Std_Reward=('TotalReward', 'std'),
        Mean_Runtime=('ScenarioTime', 'mean'),
        Std_Runtime=('ScenarioTime', 'std'),
        Mean_Steps=('ScenarioSteps', 'mean'),
        Std_Steps=('ScenarioSteps', 'std'),
        Mean_Delays=('TotalDelays', 'mean'),
        Std_Delays=('TotalDelays', 'std'),
        Mean_CancelledFlights=('TotalCancelledFlights', 'mean'),
        Std_CancelledFlights=('TotalCancelledFlights', 'std'),
        Mean_TailSwaps=('TotalTailSwaps', 'mean'),
        Std_TailSwaps=('TotalTailSwaps', 'std')
    )
    .round(2)
)

# Sort the comparison table according to specified order
model_order = ['Greedy Reactive', 'DQN Reactive', 'DQN Proactive-N', 'DQN Proactive-U']
comparison_table = comparison_table.reindex(model_order)

print("Comparison of Models Across All Scenarios:")
print(comparison_table)
```

## Key Files and Their Purposes

### Training Files
- `main.py`: Main training script that orchestrates training for multiple seeds and environment types
- `train_dqn_modular.py`: Core training function that trains a single model
- `notebooks_utils/train_dqn_both_timesteps.ipynb`: Notebook version of training (for development/debugging)

### Inference Files
- `run_inference.py`: Python script for running inference
- `notebooks_utils/run_inference_notebook.ipynb`: Notebook version of inference
- `interpretation/final-results-inference.ipynb`: Original inference notebook (reference)

### Analysis Files
- `interpretation/qualitative-analysis/final-results-examples.ipynb`: Creates the comparison table
- `interpretation/final-results-plots.ipynb`: Creates plots and statistical analysis
- `interpretation/final-results-hypotheses-validation.ipynb`: Hypothesis testing

## Metrics Collected During Inference

The inference process collects the following metrics for each scenario-model-seed combination:

- **TotalReward**: Total reward achieved
- **TotalDelays**: Total delay minutes across all flights
- **TotalCancelledFlights**: Number of cancelled flights
- **ScenarioTime**: Runtime in seconds
- **ScenarioSteps**: Number of environment steps
- **TailSwaps**: Number of aircraft tail swaps
- **ScenarioResolvedConflicts**: Number of conflicts resolved
- **SolutionSlack**: Solution slack metric
- **ActualDisruptedFlights**: Number of actually disrupted flights
- **Reward components**: Breakdown of reward into different components

## Environment Types

The system supports four environment types:
1. **myopic**: DQN Proactive-N (proactive with no uncertainty)
2. **proactive**: DQN Proactive-U (proactive with uncertainty)
3. **reactive**: DQN Reactive (reactive approach)
4. **greedy_reactive**: Greedy Reactive (baseline)

## Troubleshooting

### Common Issues

1. **Model paths not found**: Update the `model_paths` in the inference script to point to your actual trained models
2. **Data folder not found**: Make sure the `data_folder` points to a valid directory with scenario subfolders
3. **Import errors**: Make sure all required packages are installed and the Python path includes the project root

### File Structure Expected

```
project_root/
├── trained_models/
│   └── dqn/
│       └── 6ac-700-diverse/
│           ├── myopic_232323.zip
│           ├── proactive_232323.zip
│           └── reactive_232323.zip
├── Data/
│   └── TRAINING/
│       └── 6ac-700-diverse/
│           ├── scenario_1/
│           ├── scenario_2/
│           └── ...
├── logs/
│   └── scenarios/
│       └── final_results_df_in_seeds_100.csv
└── interpretation/
    └── qualitative-analysis/
        └── final-results-examples.ipynb
```

## Expected Output

The final comparison table should look like:

```
                 Mean_Reward  Std_Reward  Mean_Runtime  Std_Runtime  Mean_Steps  Std_Steps  Mean_Delays  Std_Delays  Mean_CancelledFlights  Std_CancelledFlights  Mean_TailSwaps  Std_TailSwaps
Model                                                                                                                                                                                          
Greedy Reactive     32888.75      403.08          0.17          0.0        6.89       0.02        19.58        2.53                   1.02                  0.02            1.30           0.02   
DQN Reactive        -4062.91      443.99          0.03          0.0        7.95       0.04       226.31        5.54                   1.78                  0.03            2.11           0.04   
DQN Proactive-N      6198.37      474.22          0.03          0.0        7.52       0.03       343.03        4.18                   1.32                  0.03            3.16           0.03   
DQN Proactive-U     10186.83      535.50          0.03          0.0        7.44       0.03       301.04        3.25                   1.25                  0.02            3.87           0.02   
```
