"""
Utility script to optimize Model 3 (SSF Large Dimensions) state space size.

This script scans all training scenarios to find the maximum number of flights,
then provides instructions on how to use this value to reduce the state space size.

Usage:
    python optimize_model3_state_space.py --training_folder "Data/TRAINING/3ac-182-green16/"
"""

import argparse
import os
import sys
from scripts.utils_ssf import find_max_flights_in_training_data
from src.config_ssf import MAX_AIRCRAFT, MAX_FLIGHTS_PER_AIRCRAFT

def main():
    parser = argparse.ArgumentParser(description='Optimize Model 3 state space size')
    parser.add_argument('--training_folder', type=str, required=True,
                        help='Path to training data folder (e.g., Data/TRAINING/3ac-182-green16/)')
    args = parser.parse_args()
    
    training_folder = args.training_folder
    
    print("=" * 70)
    print("Model 3 (SSF Large Dimensions) State Space Optimization")
    print("=" * 70)
    print(f"\nScanning training folder: {training_folder}\n")
    
    # Find maximum flights
    max_flights = find_max_flights_in_training_data(training_folder)
    
    if max_flights is None:
        print("\nError: Could not determine maximum flights. Exiting.")
        sys.exit(1)
    
    # Calculate current and optimized sizes
    default_max = MAX_AIRCRAFT * MAX_FLIGHTS_PER_AIRCRAFT
    ac_mtx_size = MAX_AIRCRAFT * 96  # 288
    default_fl_mtx_size = default_max * 97
    optimized_fl_mtx_size = max_flights * 97
    
    default_total = ac_mtx_size + default_fl_mtx_size
    optimized_total = ac_mtx_size + optimized_fl_mtx_size
    reduction = default_total - optimized_total
    reduction_percent = (reduction / default_total) * 100
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\nDefault max flights (MAX_AIRCRAFT × MAX_FLIGHTS_PER_AIRCRAFT): {default_max}")
    print(f"Actual max flights found in training data: {max_flights}")
    print(f"\nState Space Size Comparison:")
    print(f"  Default:  {default_total:,} elements")
    print(f"  Optimized: {optimized_total:,} elements")
    print(f"  Reduction: {reduction:,} elements ({reduction_percent:.1f}%)")
    print(f"\nMemory Savings:")
    print(f"  Default:  {default_total * 4 / 1024:.2f} KB per instance")
    print(f"  Optimized: {optimized_total * 4 / 1024:.2f} KB per instance")
    print(f"  Savings:   {reduction * 4 / 1024:.2f} KB per instance")
    
    print("\n" + "=" * 70)
    print("HOW TO USE THIS OPTIMIZATION")
    print("=" * 70)
    print(f"\nWhen creating the environment, pass max_flights_total={max_flights}:")
    print(f"\n  from src.environment_ssf_large_dimensions import AircraftDisruptionEnv")
    print(f"  from scripts.utils_ssf import find_max_flights_in_training_data")
    print(f"\n  # Find max flights (or use the value from this script)")
    print(f"  max_flights = find_max_flights_in_training_data('{training_folder}')")
    print(f"\n  # Create environment with optimized size")
    print(f"  env = AircraftDisruptionEnv(")
    print(f"      aircraft_dict,")
    print(f"      flights_dict,")
    print(f"      rotations_dict,")
    print(f"      alt_aircraft_dict,")
    print(f"      config_dict,")
    print(f"      env_type='proactive',")
    print(f"      max_flights_total={max_flights}  # <-- Add this parameter")
    print(f"  )")
    print("\n" + "=" * 70)
    
    # Save to file for easy reference
    output_file = "model3_optimization_result.txt"
    with open(output_file, 'w') as f:
        f.write(f"Model 3 State Space Optimization Results\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Training Folder: {training_folder}\n")
        f.write(f"Maximum Flights Found: {max_flights}\n")
        f.write(f"Default Maximum: {default_max}\n")
        f.write(f"\nState Space Sizes:\n")
        f.write(f"  Default:  {default_total:,} elements\n")
        f.write(f"  Optimized: {optimized_total:,} elements\n")
        f.write(f"  Reduction: {reduction:,} elements ({reduction_percent:.1f}%)\n")
        f.write(f"\nTo use this optimization, pass max_flights_total={max_flights} to AircraftDisruptionEnv\n")
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()

