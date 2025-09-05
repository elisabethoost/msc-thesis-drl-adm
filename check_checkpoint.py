#!/usr/bin/env python3
"""
Check Checkpoint Information
============================

This script shows you detailed information about your checkpoints,
including at which timestep each one was saved.

Usage:
    python check_checkpoint.py
    python check_checkpoint.py --env_type proactive --seed 232323
"""

import os
import argparse
import torch
from datetime import datetime

def format_timesteps(timesteps):
    """Format timesteps with commas and percentage of 2M"""
    return f"{timesteps:,} ({timesteps/2e6*100:.1f}% of 2M)"

def main():
    parser = argparse.ArgumentParser(description='Check checkpoint information')
    parser.add_argument("--env_type", type=str, 
                       choices=['myopic', 'proactive', 'reactive'], 
                       help="Specific environment type to check")
    parser.add_argument("--seed", type=int, 
                       help="Specific seed to check")
    
    args = parser.parse_args()
    
    checkpoint_dir = "Save_Trained_Models/training_monitor/checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        print("❌ No checkpoints directory found")
        return
    
    print("🔍 Checking checkpoint information...\n")
    
    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not checkpoint_files:
        print("❌ No checkpoint files found")
        return
    
    # Filter by specific env_type and seed if provided
    if args.env_type and args.seed:
        target_file = f"{args.env_type}_{args.seed}_checkpoint.pt"
        if target_file in checkpoint_files:
            checkpoint_files = [target_file]
        else:
            print(f"❌ No checkpoint found for {args.env_type} with seed {args.seed}")
            return
    
    elif args.env_type:
        checkpoint_files = [f for f in checkpoint_files if f.startswith(f"{args.env_type}_")]
    
    elif args.seed:
        checkpoint_files = [f for f in checkpoint_files if f"_{args.seed}_" in f]
    
    # Sort files for consistent output
    checkpoint_files.sort()
    
    print(f"📁 Found {len(checkpoint_files)} checkpoint(s):\n")
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract information
            env_type = checkpoint_file.split('_')[0]
            seed = checkpoint_file.split('_')[1]
            total_timesteps = checkpoint['total_timesteps']
            episode = checkpoint['episode']
            epsilon = checkpoint['epsilon']
            timestamp = checkpoint.get('timestamp', 'Unknown')
            
            # Calculate progress
            progress_percent = (total_timesteps / 2e6) * 100
            remaining_timesteps = 2e6 - total_timesteps
            
            print(f"📊 {checkpoint_file}")
            print(f"   🎯 Environment: {env_type}")
            print(f"   🔢 Seed: {seed}")
            print(f"   ⏱️  Timesteps: {format_timesteps(total_timesteps)}")
            print(f"   📈 Episode: {episode}")
            print(f"   🎲 Epsilon: {epsilon:.4f}")
            print(f"   📅 Timestamp: {timestamp}")
            print(f"   🚀 Progress: {progress_percent:.1f}%")
            print(f"   ⏳ Remaining: {remaining_timesteps:,.0f} timesteps")
            
            # Check if training was completed
            if total_timesteps >= 2e6:
                print(f"   ✅ Training completed! (2M timesteps reached)")
            else:
                print(f"   🔄 Training in progress...")
            
            print()
            
        except Exception as e:
            print(f"❌ Error reading {checkpoint_file}: {str(e)}")
            print()
    
    # Summary
    print("📋 Summary:")
    print(f"   • Total checkpoints: {len(checkpoint_files)}")
    
    # Check completion status
    completed = 0
    in_progress = 0
    
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_file), map_location='cpu')
            if checkpoint['total_timesteps'] >= 2e6:
                completed += 1
            else:
                in_progress += 1
        except:
            pass
    
    print(f"   • Completed training: {completed}")
    print(f"   • In progress: {in_progress}")
    
    if completed == len(checkpoint_files):
        print("\n🎉 All training runs completed successfully!")
    elif completed > 0:
        print(f"\n✅ {completed} out of {len(checkpoint_files)} training runs completed")
    else:
        print("\n🔄 All training runs are still in progress")

if __name__ == "__main__":
    main()
