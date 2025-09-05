#!/usr/bin/env python3
"""
Resume Training from Checkpoint Script
=====================================

This script allows you to resume training from a checkpoint and continue training
beyond the original MAX_TOTAL_TIMESTEPS limit.

Usage:
    python resume_training.py --env_type proactive --seed 232323 --training_folder "Data/TRAINING/6ac-130-green" --additional_timesteps 1000000

This will resume training from the last checkpoint and add 1 million more timesteps.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument("--env_type", type=str, required=True, 
                       choices=['myopic', 'proactive', 'reactive'], 
                       help="Environment type to resume")
    parser.add_argument("--seed", type=int, required=True, 
                       help="Seed to resume training for")
    parser.add_argument("--training_folder", type=str, required=True, 
                       help="Training folder path")
    parser.add_argument("--additional_timesteps", type=int, default=1000000,
                       help="Additional timesteps to train (default: 1M)")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_path = f"Save_Trained_Models/training_monitor/checkpoints/{args.env_type}_{args.seed}_checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        checkpoint_dir = "Save_Trained_Models/training_monitor/checkpoints"
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pt'):
                    print(f"  - {file}")
        return
    
    print(f"✅ Found checkpoint: {checkpoint_path}")
    
    # Check if model file exists
    stripped_folder = os.path.basename(args.training_folder.rstrip('/'))
    model_path = f"Save_Trained_Models/{stripped_folder}/{args.env_type}_{args.seed}.zip"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print(f"✅ Found model file: {model_path}")
    
    # Create a temporary modified main.py for resuming
    print("\n🔄 Creating resume configuration...")
    
    # Read the original main.py
    with open('main.py', 'r') as f:
        main_content = f.read()
    
    # Modify the MAX_TOTAL_TIMESTEPS to add the additional timesteps
    # We'll need to find the current checkpoint timesteps and add to it
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    current_timesteps = checkpoint['total_timesteps']
    new_total_timesteps = current_timesteps + args.additional_timesteps
    
    print(f"📊 Current checkpoint timesteps: {current_timesteps:,}")
    print(f"🎯 New total timesteps: {new_total_timesteps:,}")
    
    # Create a temporary main_resume.py
    resume_content = main_content.replace(
        'MAX_TOTAL_TIMESTEPS = int(2e6)   # 2 million timesteps for testing',
        f'MAX_TOTAL_TIMESTEPS = int({new_total_timesteps})   # Resumed training: {current_timesteps:,} + {args.additional_timesteps:,}'
    )
    
    with open('main_resume.py', 'w') as f:
        f.write(resume_content)
    
    print(f"✅ Created main_resume.py with {new_total_timesteps:,} total timesteps")
    
    # Run the resume training
    print(f"\n🚀 Starting resumed training for {args.env_type} with seed {args.seed}...")
    print(f"   Training folder: {args.training_folder}")
    print(f"   Additional timesteps: {args.additional_timesteps:,}")
    
    # Get the path to the virtual environment's Python interpreter
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Scripts", "python.exe")
    
    cmd = [
        venv_python,
        "main_resume.py",
        "--seed", str(args.seed),
        "--env_type", args.env_type,
        "--training_folder", args.training_folder
    ]
    
    print(f"\n📝 Command: {' '.join(cmd)}")
    print("\n⏳ Starting training... (this will resume from the checkpoint automatically)")
    
    try:
        # Set up environment variables for the subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
        
        # Run the training
        result = subprocess.run(cmd, env=env, check=True)
        print("✅ Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return
    finally:
        # Clean up temporary file
        if os.path.exists('main_resume.py'):
            os.remove('main_resume.py')
            print("🧹 Cleaned up temporary files")

if __name__ == "__main__":
    main()
