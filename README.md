# Deep Reinforcement Learning for Aircraft Disruption Management

**Note:** This repository uses the `action-logicII` branch as the main branch to consider. All models share the same reward structure, but **Model 1 has a solely negative reward structure** for debug purposes (discussed with Marta).

## Overview
This repository implements three Deep Q-Network (DQN) models for aircraft disruption management, each using a different state space formulation. The models learn to optimize airline operations during disruptions by making decisions about flight delays, cancellations, and aircraft tail swaps.\

## Repository Structure
The Models: Model1, Model2, Model3 consist of 4 main scripts + utils files:

- **Model1**: main.py, train_dqn_modular.py, environment_rf.py, config.py, utils.py
- **Model2**: main_ssf.py, train_dqn_modular_ssf.py, environment_ssf.py, config_ssf.py, utils_ssf.py
- **Model 3**: main_ssf_large_dimensions.py, train_dqn_large_dimensions.py, enviornment_ssf_largedimensions.py, config_ssf, utils_ssf_largedimensions.py

To run each model: use their respective main files!

```
msc-thesis-drl-adm/
├── training/                         # Training scripts organized by model
│   ├── model1_rf/                    # Model 1 (RF - original representation with one Matrix)
│   │   ├── main.py                   
│   │   └── train_dqn_modular.py
│   ├── model2_ssf/                   # Model 2 (SSF - Sparse State Space Formulation)
│   │   ├── main_ssf.py
│   │   └── train_dqn_modular_ssf.py
│   └── model3_ssf_large/             # Model 3 (SSF Large Dimensions)
│       ├── main_ssf_large_dimensions.py
│       └── train_dqn_modular_ssf_large_dimensions.py
├── src/                                    # Core environment and configuration files
│   ├── environment_rf.py                       # Model 1 environment
│   ├── environment_ssf.py                      # Model 2 environment
│   ├── environment_ssf_large_dimensions.py     # Model 3 environment
│   ├── config_rf.py                            # Model 1 configuration
│   └── config_ssf.py                           # Model 2 & 3 configuration
├── scripts/                                # Utility scripts and visualizations
│   ├── utils.py                                # Utilities for Model 1
│   ├── utils_ssf.py                            # Utilities for Model 2
│   ├── utils_ssf_large_dimensions.py           # Utilities for Model 3
│   ├── visualize_episode_detailed.py           # Detailed episode visualization
│   ├── visualize_episode_metrics.py            # Episode metrics visualization
│   └── create_data.py                          # Dataset creation script
├── Data/                       # Training and test datasets
│   ├── TRAINING/                               # Training scenarios
│   │   ├── 3ac-182-green16/                    # Example: 3 aircraft, 182 scenarios/schedules, max flight per ac set to 16 here
│   │   └── 3ac-130-green/                      # Example: 3 aircraft, 130 scenarios/schedules
│   └── Template/               
└── results/                    # Training results (structure preserved, files ignored)
    ├── model1_rf/
    │   ├── training/           
    │   ├── inference/
    │   └── analysis/
    ├── model2_ssf/
    │   ├── training/
    │   ├── inference/
    │   └── analysis/
    └── model3_ssf_large/
        ├── training/
        ├── inference/
        └── analysis/
```

## State Space Models

The three models differ in their state space representation:

| Model | State Space Structure | Dimensions | Description |
|-------|----------------------|------------|-------------|
| **Model 1 (RF)** | Single matrix: `(MAX_AIRCRAFT + 1) × (3 + 3×MAX_FLIGHTS_PER_AIRCRAFT)` | 456 | (3+1) × (3 + 3×17) = 4×54 = 216 base elements. Row 0: time info, Rows 1-3: unavailability (prob, start, end) + flight info (id, dep, arr) per flight. With temporal features (3 aircraft × 4 features = 12) and observation stacking (OBS_STACK_SIZE=2): (216+12)×2 = 456 final dimensions. |
| **Model 2 (SSF)** | Two matrices: `ac_mtx` + `flight_features` | 444 | `ac_mtx`: 3×96 (288 elements - aircraft unavailability over 15-min time intervals, stores probabilities). `flight_features`: 3×52 (156 elements - 3 aircraft × 13 flights × 4 features: flight_id, dep_interval_index, arr_interval_index, status). Note: `ac_mtx` uses time intervals (probability per interval), while `flight_features` stores flight data directly (not a time-interval matrix). Uses actual MAX_AIRCRAFT (3) and optimized max_flights_per_aircraft (13, found dynamically from training data). |
| **Model 3 (SSF Large)** | Two matrices: `ac_mtx` + `fl_mtx` | Variable | `ac_mtx`: 3×96 (288 elements), `fl_mtx`: Variable rows × 97. Currently optimized to 13 rows (max flights per aircraft in current dataset) × 97 = 1,261 elements. Total: 288 + 1,261 = 1,549. More detailed flight schedule representation. |

**Key Differences:**
- **Model 1**: Dense representation with temporal features, largest state space
- **Model 2**: Compact sparse representation, fixed dimensions, fastest computation
- **Model 3**: Detailed sparse representation, optimized dimensions, balances detail and efficiency

## Quick Start

### 1. Installation
See end of this file **Installation Continued** for a guide on how to set up virtual environment, get pytorch & GPU or CPU

### 2. Dataset Preparation

**NOTE:** The data set used right now is 3ac-182-green16. I advise using this and not changing it at first. 

Datasets are located in `Data/TRAINING/`. The naming convention is:
- `{N}ac-{S}-{name}/`: N aircraft, M scnarios/schedules, Data Set name 
- The number S indicates the number of scenarios (e.g., `3ac-182-green16` has 16 scenarios)

To create new datasets:
1. Use `create_datasets.ipynb` (Jupyter notebook)
2. This notebook uses `scripts/create_data.py` to generate training scenarios

### 3. Running Training

Each model has its own main script. For **Model 1** (recommended to start):

```bash
cd training/model1_rf
python main.py
```

For **Model 2**:
```bash
cd training/model2_ssf
python main_ssf.py
```

For **Model 3**:
```bash
cd training/model3_ssf_large
python main_ssf_large_dimensions.py
```

**Training Output:**
- Results are saved to `results/modelX/training/{experiment_name}/{scenario_folder}/`
- Includes:
  - Reward plots over episodes (`plots/averaged_rewards_and_timesteps_*.png`)
  - Detailed episode data (`detailed_episodes/*.pkl`)
  - Trained models (`*.zip`)
  - Configuration files (`config.csv`)

### 4. Analyzing Results

After training, analyze the results using the visualization scripts:

**Step 1: Analyze specific episode and scenario**
```bash
python scripts/visualize_episode_detailed.py \
    results/model1_rf/training/{experiment_name}/{scenario_folder} \
    {env_type} {seed} {episode_num} "{scenario_path}"
```

Example:
```bash
python scripts/visualize_episode_detailed.py \
    results/model1_rf/training/m1_1/3ac-182-green16 \
    proactive 232323 0 "Data/TRAINING/3ac-182-green16/stochastic_Scenario_00061"
```

This creates:
- Step-by-step schedule visualizations
- Reward breakdown per step
- Final statistics (delays, cancellations, tail swaps)

**Step 2: Analyze metrics for all scenarios in an episode**
```bash
# Edit scripts/visualize_episode_metrics.py to set:
# - MODEL_FOLDER = "results/model1_rf/training/m1_1/3ac-182-green16"
# - ENV_TYPE = "proactive"
# - SEED = 232323
# - EPISODE_X = 4  # Early training
# - EPISODE_Y = 100  # Later training

python scripts/visualize_episode_metrics.py
```

This creates comprehensive metrics plots comparing all scenarios in the specified episodes.

## Model Components

Each model consists of:

1. **Main Script** (`main.py` or `main_ssf.py`):
   - Orchestrates training across multiple scenarios
   - Handles result aggregation
   - Saves experiment parameters

2. **Training Script** (`train_dqn_modular*.py`):
   - Implements DQN training loop
   - Handles environment creation and interaction
   - Manages epsilon decay and evaluation
   - Saves detailed episode data

3. **Environment** (`environment*.py`):
   - Implements Gymnasium environment interface
   - Manages state space representation
   - Handles actions (delays, cancellations, tail swaps)
   - Calculates rewards based on operational costs

4. **Configuration** (`config*.py`):
   - Defines hyperparameters
   - State space dimensions
   - Reward weights and penalties
   - Training parameters (learning rate, buffer size, etc.)

## Workflow Summary

1. **Run Training**: Execute `main.py` for the desired model
   - Results saved to `results/modelX/training/{experiment_name}/`

2. **Analyze Episode Details**: Use `visualize_episode_detailed.py`
   - Visualizes step-by-step decisions for a specific scenario
   - Shows reward breakdown and final statistics

3. **Analyze Episode Metrics**: Use `visualize_episode_metrics.py`
   - Compares all scenarios within an episode
   - Shows aggregated metrics (delays, cancellations, tail swaps, etc.)

## Key Files

- **Training Scripts**: `training/modelX/main*.py`, `training/modelX/train_dqn_modular*.py`
- **Environments**: `src/environment*.py`
- **Configurations**: `src/config*.py`
- **Utilities**: `scripts/utils*.py`
- **Visualizations**: `scripts/visualize_episode_*.py`
- **Data Creation**: `create_datasets.ipynb`, `scripts/create_data.py`


## Quick Start Continued

### 1. Setting up Virtual Environment

**Requirements:**
- Python 3.10+ (3.11 recommended)
- Git
- (Optional) CUDA for GPU acceleration

**Setup: I struggled with this so her's a quick guide :D**

**Step 1: Clone and setup repository**
```bash
# Clone the repository
git clone <repository-url>
cd msc-thesis-drl-adm

# Checkout the main branch
git checkout action-logicII
```

**Step 2: Create virtual environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# => you should see (venv) in your prompt now
```
**Step 3: Install PyTorch (with GPU support if available)**

**For GPU (CUDA) support** (recommended if you have an NVIDIA GPU):
```bash
# Check your CUDA version first (if installed)
nvidia-smi

# Install PyTorch with CUDA 11.8 or 12.1 (adjust version as needed)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only** (if no GPU or CUDA issues):
```bash
pip install torch torchvision torchaudio
```

**Step 4: Install remaining dependencies**
```bash
# Upgrade pip
pip install --upgrade pip

# Install all other dependencies
pip install -r requirements.txt
```
**Note:** The code automatically detects and uses GPU if available. If GPU is not detected, it will fall back to CPU automatically
