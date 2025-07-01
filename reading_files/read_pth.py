"""
This file is used to read the .pth files located in the zip files created during each run

Step 0 - Before running this file, unzip the zip file by running the command:
* In your terminal: Expand-Archive -Path "path/to/file.zip" -DestinationPath "destination_folder"   
* For example: Expand-Archive -Path "12-run/3ac-100-superdiverse/myopic_2021.zip" -DestinationPath "temp_extract" -Force

Usage:
python reading_files/read_pth.py <folder_name>

Example:
python reading_files/read_pth.py temp_extract

The script will look for the following files in the specified folder:
- policy.pth
- policy.optimizer.pth
- pytorch_variables.pth
"""

import torch
import json
import os
import sys

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # this is the path to the reading_files folder
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR) # this is the path to the workspace folder

def analyze_pth_file(file_path):
    print(f"\nAnalyzing: {file_path}")
    # Add map_location to handle device issues
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))

    # Convert tensor values to lists for better readability
    readable_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            readable_dict[key] = {
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'min': float(value.min()),
                'max': float(value.max()),
                'mean': float(value.mean()),
                'std': float(value.std())
            }
        else:
            readable_dict[key] = str(value)

    # Print basic model information
    print("\nModel layers/variables:")
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: Shape {list(value.shape)}, Type {value.dtype}")
        else:
            print(f"{key}: {value}")
    
    return readable_dict

def main():
    # Check if folder name is provided
    if len(sys.argv) < 2:
        print("Error: Please provide the folder name as an argument")
        print("Usage: python reading_files/read_pth.py <folder_name>")
        print("Example: python reading_files/read_pth.py temp_extract")
        return

    # Get folder name from command line argument
    folder_name = sys.argv[1]
    folder_path = os.path.join(WORKSPACE_ROOT, folder_name)

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return

    # Define paths to the files
    policy_path = os.path.join(folder_path, "policy.pth")
    optimizer_path = os.path.join(folder_path, "policy.optimizer.pth")
    variables_path = os.path.join(folder_path, "pytorch_variables.pth")

    # Check if files exist
    files_to_analyze = [
        (policy_path, "policy_structure.json"),
        (optimizer_path, "optimizer_structure.json"),
        (variables_path, "variables_structure.json")
    ]

    results = {}
    for file_path, output_name in files_to_analyze:
        if not os.path.exists(file_path):
            print(f"Warning: File '{file_path}' does not exist")
            continue
        
        try:
            results[output_name] = analyze_pth_file(file_path)
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
            continue

    # Save results to JSON files
    print("\nStructures have been saved to:")
    for output_name, data in results.items():
        output_path = os.path.join(SCRIPT_DIR, output_name)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"- {output_path}")

if __name__ == "__main__":
    main() 