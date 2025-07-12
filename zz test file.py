# This is a test file to check if the code is working
import os
'''
def verify_training_folders(path):
    """Verify if the training folders exist and return folder names."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Training folder not found at {path}')

    training_folders = [
        folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))
    ]
    return training_folders

###########################################

all_folders_temp = [
        "data/Example/Example-scenario-1/"
    ]

for i in all_folders_temp:
    print(i)

num_scenarios_training = len(os.listdir(all_folders_temp[0]))
print(f"number of scenarios in training: {num_scenarios_training}")

training_folders = verify_training_folders(all_folders_temp[0])
print(f"training folders: {training_folders}")
print()

scenario_folders = []
folder_path = "3-aaa-130-supertje-diverse-with-old-config"
available_env_types = [
        {'name': 'myopic', 'label': 'DQN Proactive-N', 'color': 'blue'},
        {'name': 'proactive', 'label': 'DQN Proactive-U', 'color': 'orange'},
        {'name': 'reactive', 'label': 'DQN Reactive', 'color': 'green'},
        {'name': 'drl-greedy', 'label': 'DQN Greedy-Guided', 'color': 'red'}
    ]
scenario_folders.append({
                    'path': folder_path,
                    'available_env_types': available_env_types
                })

print(scenario_folders)
'''

l = [1,2,3,4,5,6,7,8,9,10]
m = [6,7,7,9]
b = len(l)
a = list(range(len(l) + 1))
print(f"length of l is {b}")
print(f'a is {a}')

y = []
for i in m:
    y.append(i)

print(f'y is {y}')

all_folders_temp = [
        "data/Training/6ac-100-stochastic-low/",
        "data/Training/6ac-100-stochastic-medium/",
        "data/Training/6ac-100-stochastic-high/",
        "data/Training/6ac-700-diverse/",
        "data/RESULTS/6ac-130-supertje-diverse/"
    ]

print(f"the chosen folder is {all_folders_temp[2]}")








