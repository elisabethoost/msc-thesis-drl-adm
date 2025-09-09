import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''
Steps:
1. change latest_folder = "scenario_folder_scenario_76.json" to the scenario in logs/scenarios/ 
corresponds to your TEST DATA
2. change this too: results_df = pd.read_csv(os.path.join(scenario_folder_path, f"6ac-26-lilac_{len(seeds)}_seeds.csv")) from run_inference.py (testing data)
3. check if results_df is correct (same as output_file in run_inference.py)
4. run the script normally to obtain the comparison table as a csv file in logs/inference_metrics/
5. run this command if you want to see the output as txt file: python interpretation/qualitative-analysis/EO_final-results-examples-fixed.py > output.txt 2>&1
6. change the seeds to 101 if you want (takes longer but better results)

Uses GREEDY REACTIVE BASELINE from environment.py or GREEDY REACTIVE BASELINE from main-optimizer.py
'''
# Set up the environment
sns.set_theme(style="darkgrid")

# Set pandas display options to show all columns and values
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

seeds = []
for i in range(1, 6):
    seeds.append(i)

def time_to_minutes(timestr):
    # Handle '+1' suffix by removing it before parsing
    timestr = timestr.split('+')[0]  # Remove '+1' if present
    hh, mm = timestr.split(':')
    return int(hh) * 60 + int(mm)

def calculate_slack_for_scenario(scenario_data):
    """
    Calculate the slack metric for the given scenario.
    
    Slack is defined as:
        Slack = 1 - (total flight minutes in recovery period / total recovery period aircraft-minutes)
    
    A slack of 1 means no flights during recovery period.
    A slack of 0 means flights occupy the entire recovery period.
    """
    
    # Extract scenario start/end times
    # We assume the same date for start and end for simplicity.
    recovery_start_time_str = scenario_data["recovery_start_time"]  
    recovery_end_time_str = scenario_data["recovery_end_time"]      
    
    recovery_start_minutes = time_to_minutes(recovery_start_time_str)
    recovery_end_minutes = time_to_minutes(recovery_end_time_str)
    total_recovery_period_minutes = recovery_end_minutes - recovery_start_minutes
    
    total_aircraft = scenario_data["total_aircraft"]
    
    # Calculate total flight minutes within the recovery period
    flights = scenario_data["flights"]
    total_flights = len(flights)
    total_flight_minutes_in_recovery = 0
    total_flight_minutes_total = 0
    
    for flight_id, flight_data in flights.items():
        dep_time_str = flight_data["DepTime"]  
        arr_time_str = flight_data["ArrTime"] 
        
        dep_minutes = time_to_minutes(dep_time_str)
        arr_minutes = time_to_minutes(arr_time_str)
        
        total_flight_minutes_total += arr_minutes - dep_minutes
        overlap_start = max(dep_minutes, recovery_start_minutes)
        overlap_end = min(arr_minutes, recovery_end_minutes)
        
        if overlap_end > overlap_start:
            flight_overlap = overlap_end - overlap_start
        else:
            flight_overlap = 0
        
        total_flight_minutes_in_recovery += flight_overlap
    
    # Calculate total aircraft-minutes available during the recovery period
    total_recovery_aircraft_minutes = total_recovery_period_minutes * total_aircraft
    
    # Slack calculation
    if total_recovery_aircraft_minutes == 0:
        slack = 1.0
    else:
        slack = 1 - (total_flight_minutes_in_recovery / total_recovery_aircraft_minutes)
    
    return slack, total_flights, total_flight_minutes_total

def extract_disruption_stats(scenario_data):
    """
    Extract disruption statistics:
    - Count of fully disrupted (prob = 1.0)
    - Count of uncertain disruptions (0 < prob < 1.0)
    - Average probability across all aircraft (where an aircraft's probability is the max disruption probability it faces, 
      with 1.0 for fully disrupted and 0.0 if no disruption)
    - Average uncertainty probability (average of all disruptions where 0<prob<1.0, excluding 0 and 1)
    """
    disruptions_info = scenario_data.get('disruptions', {})
    disruptions_list = disruptions_info.get('disruptions', [])
    total_aircraft = disruptions_info.get('total_aircraft', 0)

    if total_aircraft == 0:
        # No aircraft or no disruptions
        return 0, 0, 0.0, 0.0

    fully_disrupted_count = sum(1 for d in disruptions_list if d.get('probability', 0.0) == 1.0)
    uncertain_disruptions = [d for d in disruptions_list if 0.0 < d.get('probability', 0.0) < 1.0]
    uncertain_count = len(uncertain_disruptions)

    aircraft_ids = scenario_data.get('aircraft_ids', [])
    ac_prob_map = {ac: 0.0 for ac in aircraft_ids}  
    
    for d in disruptions_list:
        ac_id = d.get('aircraft_id')
        p = d.get('probability', 0.0)
        # Keep the max probability for that aircraft
        if ac_id in ac_prob_map:
            ac_prob_map[ac_id] = max(ac_prob_map[ac_id], p)

    avg_ac_prob = sum(ac_prob_map.values()) / total_aircraft if total_aircraft > 0 else 0.0

    # Average uncertainty probability (only consider disruptions where 0<prob<1)
    if len(uncertain_disruptions) > 0:
        avg_uncertainty_prob = np.mean([d['probability'] for d in uncertain_disruptions])
    else:
        avg_uncertainty_prob = 0.0

    return fully_disrupted_count, uncertain_count, avg_ac_prob, avg_uncertainty_prob, total_aircraft

# Path to the scenarios folder
scenario_folder_path = "logs/scenarios/"
latest_folder = "scenario_folder_scenario_85.json" # Training/6ac-26-lilac
# latest_folder = "scenario_folder_scenario_77.json" # Testing/6ac-65-yellow

file_path = os.path.join(scenario_folder_path, latest_folder)

# Extract scenario ID
scenario_id = file_path.split('_')[-1].split('.')[0]
print(f"Scenario ID: {scenario_id}")

# Load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract the scenarios from the JSON data
scenarios = data['outputs']

# Extract the data_folder (not strictly necessary for slack calculation, but we print it for context)
data_folder = data['data_folder']
print(f"Data Folder: {data_folder}")

# Calculate slack and disruption stats for each scenario and store in a list of dicts
results = []
for scenario_name, scenario_data in scenarios.items():
    scenario_slack, total_flights, total_flight_minutes_total = calculate_slack_for_scenario(scenario_data)
    fully_disrupted_count, uncertain_count, avg_ac_prob, avg_uncertain_prob, total_aircraft = extract_disruption_stats(scenario_data)
    results.append({
        "Scenario": scenario_name,
        "ScenarioSlack": scenario_slack,
        "TotalFlights": total_flights,
        "TotalFlightMinutes": total_flight_minutes_total,
        "FullyDisruptedCount": fully_disrupted_count,
        "UncertainCount": uncertain_count,
        "AvgAircraftProbability": avg_ac_prob,
        "AvgUncertaintyProbability": avg_uncertain_prob,
        "TotalAircraft": total_aircraft
    })

# Convert results to DataFrame
scenarios_df = pd.DataFrame(results)
print(scenarios_df)

# Load and process inference results
scenario_folder_path = "logs/inference_metrics/"
# unpack results_df
results_df = pd.read_csv(os.path.join(scenario_folder_path, f"3ac-143-black_{len(seeds)}_4.csv"))

# Merge scenario-level info from scenarios_df into results_df
merged_df = results_df.merge(scenarios_df, on='Scenario', how='left')

# Add scenario category based on prefix and difficulty level
merged_df["ScenarioCategory"] = merged_df["Scenario"].apply(
    lambda x: "Deterministic" if x.startswith("deterministic") else
             "Stochastic High" if x.startswith("stochastic_high") else
             "Stochastic Medium" if x.startswith("stochastic_medium") else
             "Stochastic Low" if x.startswith("stochastic_low") else
             "Mixed High" if x.startswith("mixed_high") else
             "Mixed Medium" if x.startswith("mixed_medium") else
             "Mixed Low" if x.startswith("mixed_low") else
             "Other"
)

# Sort models in desired order - FIXED VERSION
def extract_model_type(model_name):
    # """Extract model type with proper handling for greedy_reactive"""
    if model_name == 'greedy_reactive':
        return 'greedy_reactive'
    # """Extract model type with proper handling for optimal_exact"""
    # if model_name == 'optimal_exact':
    #     return 'optimal_exact'
    elif 'proactive' in model_name:
        return 'proactive'
    elif 'myopic' in model_name:
        return 'myopic'
    elif 'reactive' in model_name:
        return 'reactive'
    else:
        return model_name

merged_df['Model_Type'] = merged_df['Model'].apply(extract_model_type)
# Remove the sorting to keep original order
# merged_df = merged_df.sort_values('Model_Type')
merged_df["Model"] = merged_df["Model_Type"]
merged_df = merged_df.drop('Model_Type', axis=1)
merged_df_backup = merged_df.copy()

# Update model names in merged_df
merged_df['Model'] = merged_df['Model'].apply(lambda x: 
    'DQN Proactive-U' if x.startswith('proactive') else
    'DQN Proactive-N' if x.startswith('myopic') else 
    'DQN Reactive' if x.startswith('reactive') else
    'Greedy Reactive' if x.startswith('greedy_reactive') else
    # 'Optimal Exact' if x.startswith('optimal_exact') else
    x
)

print("Inference Results (After Merging):")
# Reset display options to show truncated version for readability
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 100) 
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 10)  # Show only first and last few rows
print(merged_df)

# Save the merged results to CSV
merged_output_file = os.path.join(scenario_folder_path, f"scenario_inference_metrics_{scenario_id}.csv")
merged_df.to_csv(merged_output_file, index=False)
print(f"Inference results with scenario info saved to {merged_output_file}")

# print all column names
print("==== Columns: ====")
print(merged_df.columns)

print("==== amount of rows: ====")
print(len(merged_df))

print("==== Models: ====")
print(merged_df["Model"].unique())

print('===== len(seeds) =====')
print(len(merged_df['Seed'].unique()))

print('===== len(scenarios) =====')
print(len(merged_df['Scenario'].unique()))

# Comparison of Models Across All Scenarios
# Restore display options to show all columns for comparison table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# Define model colors and order
model_colors = {
    'DQN Proactive-U': ('orange', 'DQN Proactive-U'),
    'DQN Proactive-N': ('blue', 'DQN Proactive-N'),
    'DQN Reactive': ('green', 'DQN Reactive'),
    'Greedy Reactive': ('darkgrey', 'Greedy Reactive')
    # 'Optimal Exact': ('darkgrey', 'Optimal Exact')
}

# First aggregate by Model and Seed, then calculate mean and std across seeds
comparison_table = (
    merged_df
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
# GREEDY REACTIVE BASELINE:
model_order = ['Greedy Reactive', 'DQN Reactive', 'DQN Proactive-N', 'DQN Proactive-U']
# OPTIMAL EXACT BASELINE:
# model_order = ['Optimal Exact', 'DQN Reactive', 'DQN Proactive-N', 'DQN Proactive-U']
comparison_table = comparison_table.reindex(model_order)

print("Comparison of Models Across All Scenarios:")
print(comparison_table)

# Print detailed comparison with better formatting
print("\n" + "="*100)
print("DETAILED COMPARISON OF MODELS ACROSS ALL SCENARIOS")
print("="*100)
print(f"{'Model':<20} {'Mean_Reward':<12} {'Std_Reward':<12} {'Mean_Runtime':<12} {'Std_Runtime':<12} {'Mean_Steps':<12} {'Std_Steps':<12}")
print("-"*100)
for model in model_order:
    if model in comparison_table.index:
        row = comparison_table.loc[model]
        print(f"{model:<20} {row['Mean_Reward']:<12.2f} {row['Std_Reward']:<12.2f} {row['Mean_Runtime']:<12.2f} {row['Std_Runtime']:<12.2f} {row['Mean_Steps']:<12.2f} {row['Std_Steps']:<12.2f}")

print("\n" + "="*100)
print("DELAYS AND CANCELLATIONS COMPARISON")
print("="*100)
print(f"{'Model':<20} {'Mean_Delays':<12} {'Std_Delays':<12} {'Mean_CancelledFlights':<20} {'Std_CancelledFlights':<20}")
print("-"*100)
for model in model_order:
    if model in comparison_table.index:
        row = comparison_table.loc[model]
        print(f"{model:<20} {row['Mean_Delays']:<12.2f} {row['Std_Delays']:<12.2f} {row['Mean_CancelledFlights']:<20.2f} {row['Std_CancelledFlights']:<20.2f}")

print("\n" + "="*100)
print("TAIL SWAPS COMPARISON")
print("="*100)
print(f"{'Model':<20} {'Mean_TailSwaps':<15} {'Std_TailSwaps':<15}")
print("-"*100)
for model in model_order:
    if model in comparison_table.index:
        row = comparison_table.loc[model]
        print(f"{model:<20} {row['Mean_TailSwaps']:<15.2f} {row['Std_TailSwaps']:<15.2f}")

# Save the comparison table to CSV for easy viewing
comparison_csv_path = "logs/inference_metrics/comparison_table_detailed_1.csv"
comparison_table.to_csv(comparison_csv_path)
print(f"\nDetailed comparison table saved to: {comparison_csv_path}")



'''
----------------------------------------------------
----------------------------------------------------
Plot is next 
----------------------------------------------------
----------------------------------------------------
'''
import pandas as pd
import matplotlib.pyplot as plt

# Create bar plot with error bars
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

# Define desired order of models
model_order = ['DQN Proactive-U', 'DQN Proactive-N', 'DQN Reactive', 'Greedy Reactive']

# Define model colors and order
model_colors = {
    'DQN Proactive-U': ('orange', 'DQN Proactive-U'),
    'DQN Proactive-N': ('blue', 'DQN Proactive-N'),
    'DQN Reactive': ('green', 'DQN Reactive'),
    'Greedy Reactive': ('darkgrey', 'Greedy Reactive')
}
# Sort comparison table by desired order
sorted_models = sorted(comparison_table.index, 
                      key=lambda x: model_order.index(model_colors[x][1]))

x = range(len(sorted_models))
width = 0.6

# Plot reward bars using model colors in specified order
for i, model in enumerate(sorted_models):
    print(model)
    color = model_colors[model][0]  # Get color from model_colors dictionary
    label = model_colors[model][1]  # Get label from model_colors dictionary
    ax1.bar(i, comparison_table.loc[model, 'Mean_Reward'], width,
            yerr=comparison_table.loc[model, 'Std_Reward'],
            capsize=5, label=label, color=color)

ax1.set_ylabel('Mean total reward')
ax1.set_title('Inference reward')
ax1.set_xticks([])
ax1.axhline(y=0, color='#404040', linewidth=1)  # Add darker gray horizontal line

# Plot runtime bars
for i, model in enumerate(sorted_models):
    color = model_colors[model][0]  # Get color from model_colors dictionary
    label = model_colors[model][1]  # Get label from model_colors dictionary
    ax2.bar(i, comparison_table.loc[model, 'Mean_Runtime'], width,
            yerr=comparison_table.loc[model, 'Std_Runtime'],
            capsize=5, color=color)  # Removed label here since we only want one legend

ax2.set_ylabel('Mean runtime (minutes)')
ax2.set_title('Runtime')
ax2.set_xticks([])
ax2.axhline(y=0, color='#404040', linewidth=1)  # Add darker gray horizontal line

# Add legend centered below the plots with 2 columns
fig.legend(ncol=4, bbox_to_anchor=(0.5, -0.05), loc='center')

plt.tight_layout()
plt.show()
