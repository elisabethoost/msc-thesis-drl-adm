#!/usr/bin/env python3
"""
Cleanup script to remove old checkpoints that have architecture mismatches
with the new fixed version.
"""

import os
import glob
import shutil

def cleanup_old_checkpoints():
    """Remove old checkpoints that are incompatible with the new architecture"""
    
    # Look for checkpoint directories
    checkpoint_patterns = [
        "Save_Trained_Models/*/checkpoints/*.pt",
        "Save_Trained_Models/training_monitor/checkpoints/*.pt"
    ]
    
    removed_count = 0
    
    for pattern in checkpoint_patterns:
        checkpoints = glob.glob(pattern)
        for checkpoint_path in checkpoints:
            try:
                print(f"Removing old checkpoint: {checkpoint_path}")
                os.remove(checkpoint_path)
                removed_count += 1
            except Exception as e:
                print(f"Failed to remove {checkpoint_path}: {e}")
    
    print(f"Removed {removed_count} old checkpoints")
    
    # Also clean up any backup files
    backup_patterns = [
        "Save_Trained_Models/*/checkpoints/*.pt.backup",
        "Save_Trained_Models/training_monitor/checkpoints/*.pt.backup"
    ]
    
    backup_count = 0
    for pattern in backup_patterns:
        backups = glob.glob(pattern)
        for backup_path in backups:
            try:
                print(f"Removing backup: {backup_path}")
                os.remove(backup_path)
                backup_count += 1
            except Exception as e:
                print(f"Failed to remove {backup_path}: {e}")
    
    print(f"Removed {backup_count} backup files")
    print("Cleanup complete! You can now run main.py without checkpoint conflicts.")

if __name__ == "__main__":
    cleanup_old_checkpoints()
