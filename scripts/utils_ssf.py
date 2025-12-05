import os
import numpy as np
from datetime import datetime, timedelta
import re
from src.config_ssf import *
import csv
import torch as th
import math
import json


# File reader with comment filtering
def read_csv_with_comments(file_path):
    """Reads a CSV file and skips comment lines (lines starting with '%') and stops at '#'."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data_lines = []
        for line in lines:
            if line.startswith('#'):
                break
            if not line.startswith('%'):
                data_lines.append(line.strip())

        # Return an empty list if no data lines were found
        if not data_lines:
            return []
        
        return data_lines
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    

def count_flights_in_scenario(scenario_folder):
    """Counts the number of flights in a scenario by reading flights.csv.
    
    Args:
        scenario_folder (str): Path to the scenario folder containing flights.csv
        
    Returns:
        int: Number of flights (excluding header row)
    """
    flights_file = os.path.join(scenario_folder, 'flights.csv')
    if not os.path.exists(flights_file):
        return 0
    
    try:
        with open(flights_file, 'r') as file:
            lines = file.readlines()
        
        flight_count = 0
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                break  # Stop at end marker
            if line.startswith('%'):
                continue  # Skip header lines
            if line:  # Count non-empty data lines
                flight_count += 1
        
        return flight_count
    except Exception as e:
        print(f"Error counting flights in {flights_file}: {e}")
        return 0


def find_max_flights_in_training_data(training_folders_path):
    """Scans all training scenarios to find the maximum number of flights.
    
    This function recursively searches through all scenario folders in the training
    data directory and finds the maximum number of flights across all scenarios.
    This is useful for optimizing Model 3 (SSF Large Dimensions) state space size.
    
    Args:
        training_folders_path (str): Path to the training data folder (e.g., 'Data/TRAINING/3ac-182-green16/')
        
    Returns:
        int: Maximum number of flights found across all scenarios, or None if no scenarios found
    """
    if not os.path.exists(training_folders_path):
        print(f"Warning: Training folder path does not exist: {training_folders_path}")
        return None
    
    max_flights = 0
    scenario_count = 0
    
    # Find all scenario folders (directories containing flights.csv)
    for root, dirs, files in os.walk(training_folders_path):
        if 'flights.csv' in files:
            scenario_count += 1
            flight_count = count_flights_in_scenario(root)
            if flight_count > max_flights:
                max_flights = flight_count
                if DEBUG_MODE:
                    print(f"New max found: {max_flights} flights in {root}")
    
    if scenario_count == 0:
        print(f"Warning: No scenarios found in {training_folders_path}")
        return None
    
    print(f"Scanned {scenario_count} scenarios in {training_folders_path}")
    print(f"Maximum flights found: {max_flights}")
    
    return max_flights


def parse_time_with_day_offset(time_str, reference_date):
    """
    Parses time and adds a day offset if '+1' is present, or if the arrival time 
    is earlier than the departure time (indicating a flight crosses midnight).
    
    Args:
        time_str: Either a string in 'HH:MM' format or a datetime object
        reference_date: Reference date for parsing
        
    Returns:
        datetime: Parsed datetime object with day offset if needed
    """
    # If time_str is already a datetime object, return it directly
    if isinstance(time_str, datetime):
        return time_str
    
    # Check if '+1' exists in the time string
    if '+1' in time_str:
        # Remove the '+1' and strip any whitespace
        time_str = time_str.replace('+1', '').strip()
        time_obj = datetime.strptime(time_str, '%H:%M')
        # Add 1 day to the time
        return datetime.combine(reference_date, time_obj.time()) + timedelta(days=1)
    
    # Otherwise, parse normally
    time_obj = datetime.strptime(time_str, '%H:%M')
    return datetime.combine(reference_date, time_obj.time())


# Parsing all the data files. Parsing means converting the data from a string format to a dictionary format.
class FileParsers:
    
    @staticmethod
    def parse_config(data_lines):
        config_dict = {}
        config_dict['RecoveryPeriod'] = {
            'StartDate': data_lines[0].split()[0],
            'StartTime': data_lines[0].split()[1],
            'EndDate': data_lines[0].split()[2],
            'EndTime': data_lines[0].split()[3]
        }

        def parse_costs(line):
            parts = re.split(r'\s+', line)
            costs = [{'Cabin': parts[i], 'Type': parts[i+1], 'Cost': float(parts[i+2])} for i in range(0, len(parts), 3)]
            return costs

        config_dict['DelayCosts'] = parse_costs(data_lines[1])
        config_dict['CancellationCostsOutbound'] = parse_costs(data_lines[2])
        config_dict['CancellationCostsInbound'] = parse_costs(data_lines[3])

        def parse_downgrading_costs(line):
            parts = re.split(r'\s+', line)
            costs = [{'FromCabin': parts[i], 'ToCabin': parts[i+1], 'Type': parts[i+2], 'Cost': float(parts[i+3])} for i in range(0, len(parts), 4)]
            return costs

        config_dict['DowngradingCosts'] = parse_downgrading_costs(data_lines[4])
        config_dict['PenaltyCosts'] = [float(x) for x in re.split(r'\s+', data_lines[5])]
        config_dict['Weights'] = [float(x) for x in re.split(r'\s+', data_lines[6])]
        return config_dict

    @staticmethod
    def parse_airports(data_lines):
        airports_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            airport = parts[0]
            capacities = [{'Dep/h': int(parts[i]), 'Arr/h': int(parts[i+1]), 'StartTime': parts[i+2], 'EndTime': parts[i+3]} for i in range(1, len(parts), 4)]
            airports_dict[airport] = capacities
        return airports_dict

    @staticmethod
    def parse_dist(data_lines):
        dist_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            dist_dict[(parts[0], parts[1])] = {'Dist': int(parts[2]), 'Type': parts[3]}
        return dist_dict

    @staticmethod
    def parse_flights(data_lines):
        flights_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            flights_dict[int(parts[0])] = {'Orig': parts[1], 'Dest': parts[2], 'DepTime': parts[3], 'ArrTime': parts[4], 'PrevFlight': int(parts[5])}
        return flights_dict

    @staticmethod
    def parse_aircraft(data_lines):
        aircraft_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            aircraft_dict[parts[0]] = {'Model': parts[1], 'Family': parts[2], 'Config': parts[3], 'Dist': int(parts[4]), 'Cost/h': float(parts[5]),
                                       'TurnRound': int(parts[6]), 'Transit': int(parts[7]), 'Orig': parts[8], 'Maint': parts[9] if len(parts) > 9 else None}
        return aircraft_dict

    @staticmethod
    def parse_rotations(data_lines):
        rotations_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            rotations_dict[int(parts[0])] = {'DepDate': parts[1], 'Aircraft': parts[2]}
        return rotations_dict

    @staticmethod
    def parse_itineraries(data_lines):
        itineraries_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            itineraries_dict[int(parts[0])] = {'Type': parts[1], 'Price': float(parts[2]), 'Count': int(parts[3]), 'Flights': parts[4:]}
        return itineraries_dict

    @staticmethod
    def parse_position(data_lines):
        positions_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            if parts[0] not in positions_dict:
                positions_dict[parts[0]] = []
            positions_dict[parts[0]].append({'Model': parts[1], 'Config': parts[2], 'Count': int(parts[3])})
        return positions_dict

    @staticmethod
    def parse_alt_flights(data_lines):
        """Parses the alt_flights file into a dictionary."""
        if not data_lines:
            return {}

        alt_flights_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            alt_flights_dict[int(parts[0])] = {'DepDate': parts[1], 'Delay': int(parts[2])}
        return alt_flights_dict

    @staticmethod
    def parse_alt_aircraft(data_lines):
        """Parses the alt_aircraft file into a dictionary."""
        if data_lines is None:
            return {}

        alt_aircraft_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            alt_aircraft_dict[parts[0]] = {
                'StartDate': parts[1],
                'StartTime': parts[2],
                'EndDate': parts[3],
                'EndTime': parts[4],
                'Probability': float(parts[5])
            }
        return alt_aircraft_dict

    @staticmethod
    def parse_alt_airports(data_lines):
        """Parses the alt_airports file into a dictionary."""
        if data_lines is None:
            return {}

        alt_airports_dict = {}
        for line in data_lines:
            parts = re.split(r'\s+', line)
            airport = parts[0]
            if airport not in alt_airports_dict:
                alt_airports_dict[airport] = []
            alt_airports_dict[airport].append({
                'StartDate': parts[1],
                'StartTime': parts[2],
                'EndDate': parts[3],
                'EndTime': parts[4],
                'Dep/h': int(parts[5]),
                'Arr/h': int(parts[6])
            })
        return alt_airports_dict


def load_scenario_data(scenario_folder):
    file_keys = ['aircraft', 'airports', 'alt_aircraft', 'alt_airports', 'alt_flights', 'config', 'dist', 'flights', 'itineraries', 'position', 'rotations']
    file_paths = {key: os.path.join(scenario_folder, f"{key}.csv") for key in file_keys}

    data_dict = {}
    file_parsing_functions = {
        'config': FileParsers.parse_config,
        'airports': FileParsers.parse_airports,
        'dist': FileParsers.parse_dist,
        'flights': FileParsers.parse_flights,
        'aircraft': FileParsers.parse_aircraft,
        'rotations': FileParsers.parse_rotations,
        'itineraries': FileParsers.parse_itineraries,
        'position': FileParsers.parse_position,
        'alt_flights': FileParsers.parse_alt_flights,
        'alt_aircraft': FileParsers.parse_alt_aircraft,
        'alt_airports': FileParsers.parse_alt_airports
    }

    # Iterate over each file and process it using the correct parsing function
    for file_type, file_path in file_paths.items():
        file_lines = read_csv_with_comments(file_path)
        if file_lines:
            parse_function = file_parsing_functions.get(file_type)
            if parse_function:
                parsed_data = parse_function(file_lines)
                data_dict[file_type] = parsed_data
            else:
                print(f"No parser available for file type: {file_type}")
        else:
            data_dict[file_type] = None

    # # Fix the alt_aircraft file: if the arrival time is before the departure time, add 24 hours to the arrival time
    # if data_dict['alt_aircraft']:
    #     print(f"Fixing alt_aircraft file")
    #     print(data_dict['alt_aircraft'])
    #     """
    #     data_dict['alt_aircraft'] is of this form:
    #     {'A320#1': {'StartDate': '14/09/24', 'StartTime': '08:02', 'EndDate': '14/09/24', 'EndTime': '23:29', 'Probability': 0.17}, 'A320#2': {'StartDate': '14/09/24', 'StartTime': '08:15', 'EndDate': '14/09/24', 'EndTime': '13:13', 'Probability': 0.31}, 'A320#3': {'StartDate': '14/09/24', 'StartTime': '08:12', 'EndDate': '14/09/24', 'EndTime': '15:54', 'Probability': 0.0}}
    #     """
    #     for aircraft in data_dict['alt_aircraft']:
    #         for flight in data_dict['alt_aircraft'][aircraft]:
    #             deptime = data_dict['alt_aircraft'][aircraft]['StartTime']
    #             print(f"deptime: {deptime}")
    #             arrtime = data_dict['alt_aircraft'][aircraft]['EndTime']
    #             print(f"arrtime: {arrtime}")
    #             if arrtime < deptime:
    #                 # add one day to the end date
    #                 data_dict['alt_aircraft'][aircraft]['EndDate'] = (datetime.strptime(data_dict['alt_aircraft'][aircraft]['EndDate'], '%d/%m/%y') + timedelta(days=1)).strftime('%d/%m/%y')
    return data_dict


def get_device_info(device):
    """Get detailed information about the device."""
    device_str = str(device).lower()
    print("Device:", device_str)

    if device_str == 'mps':
        print("Using MacBook M1")
        device_info = {"device_type": "MacBook M1"}
    elif device_str == 'cuda':
        print("Using GPU")
        device_info = {"device_type": "GPU"}
    else:
        print("Using CPU")
        device_info = {"device_type": "CPU"}

    return device_info


def check_device_capabilities():
    """Check and print device capabilities."""
    print("CUDA available:", th.cuda.is_available())
    print("Number of GPUs available:", th.cuda.device_count())
    if th.cuda.is_available():
        print("Current GPU name:", th.cuda.get_device_name(0))
    print("cuDNN enabled:", th.backends.cudnn.enabled)


def verify_training_folders(path):
    """Verify if the training folders exist and return folder names."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Training folder not found at {path}')

    training_folders = [
        folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))
    ]
    return training_folders


def calculate_training_days(n_episodes, folders):
    """Calculate total training days."""
    return n_episodes * len(folders)


def format_days(days):
    """Format the number of days for better readability."""
    if days >= 1000000:
        return f"{math.floor(days / 1000000)}M"
    elif days >= 1000:
        return f"{math.floor(days / 1000)}k"
    return str(days)


def create_results_directory(base_dir='../results', append_to_name=''):
    """Create a results directory with the current datetime."""
    base_dir = os.path.join(base_dir, append_to_name)
    now = datetime.now()
    folder_name = now.strftime('%Y%m%d-%H-%M')
    results_dir = os.path.join(base_dir, folder_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    return results_dir


def initialize_device():
    """Initialize and return the computation device."""
    if th.cuda.is_available():
        device = th.device('cuda')
        gpu_name = th.cuda.get_device_name(0)
        print("=" * 60)
        print(f"[OK] NVIDIA GPU DETECTED: {gpu_name}")
        print("=" * 60)
    elif th.backends.mps.is_available():
        device = th.device('mps')
        print(f"[WARNING] Using Apple Silicon (MPS) - GPU acceleration available")
        print(f"Using device: {device}")
    else:
        device = th.device('cpu')
        print("=" * 60)
        print("[WARNING] No GPU detected, using CPU (training will be slower)")
        print("=" * 60)
    return device


def calculate_epsilon_decay_rate(total_timesteps, epsilon_start, epsilon_min, percentage_min=95, EPSILON_TYPE="exponential"):
    """
    Calculates the decay rate for epsilon such that it reaches epsilon_min after the specified percentage of total timesteps.

    Args:
        total_timesteps (int): Total number of timesteps in training.
        epsilon_start (float): Initial epsilon value for exploration.
        epsilon_min (float): Minimum epsilon value for exploration.
        percentage_min (float): Percentage of timesteps at which epsilon should reach epsilon_min (default is 95%).
        EPSILON_TYPE (str): Type of decay ("exponential", "linear", or "mixed").

    Returns:
        float: Calculated epsilon decay rate.
    """
    # Calculate the timesteps at which epsilon should reach epsilon_min
    target_timesteps = total_timesteps * (percentage_min / 100)

    if EPSILON_TYPE == "exponential":
        # Solve for the decay rate using the exponential decay formula
        decay_rate = -np.log(epsilon_min / epsilon_start) / target_timesteps
    elif EPSILON_TYPE == "linear":
        decay_rate = (epsilon_start - epsilon_min) / target_timesteps
    elif EPSILON_TYPE == "mixed":
        decay_rate = -np.log(epsilon_min / epsilon_start) / target_timesteps
        decay_rate_linear = (epsilon_start - epsilon_min) / target_timesteps
        return decay_rate, decay_rate_linear

    print(f"Calculated EPSILON_DECAY_RATE: {decay_rate}")
    return decay_rate


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)
