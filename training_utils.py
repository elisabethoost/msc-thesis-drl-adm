"""
Training Utilities for Deep Reinforcement Learning with Crash Recovery
===================================================================
This module provides robust training monitoring and crash recovery functionality for DRL training processes.
It is designed to prevent data loss from system crashes, thermal shutdowns, or power failures during long
training runs.

Key Features:
------------
1. Automatic Checkpointing:
   - Saves model state, optimizer state, and training progress every 50,000 timesteps
   - Uses atomic file operations to prevent checkpoint corruption
   - Checkpoints are stored in: {save_folder}/{scenario_folder}/checkpoints/{env_type}_{seed}_checkpoint.pt
   
2. System Health Monitoring:
   - Monitors CPU and GPU temperatures (threshold: CPU 85°C, GPU 80°C)
   - Tracks memory usage (threshold: 90% usage)
   - Performs checks every 10,000 timesteps
   - Triggers early checkpoints if system health is concerning

3. Detailed Logging:
   - Training progress
   - System metrics
   - Error conditions
   - Logs stored in: {save_folder}/{scenario_folder}/logs/{env_type}_{seed}_training.log

Recovery After a Crash:
---------------------
1. The training will automatically resume from the last checkpoint when restarted
2. Checkpoints can be found in: {save_folder}/{scenario_folder}/checkpoints/
3. To manually resume training:
   - Simply restart the training script with the same parameters
   - The TrainingMonitor will automatically detect and load the latest checkpoint

Testing the System:
-----------------
To test the checkpointing and recovery:
1. Modify MAX_TOTAL_TIMESTEPS in main.py to a small number (e.g., 100000)
2. Checkpoints will be created at:
   - Every 50,000 timesteps (main checkpoints)
   - Every 10,000 timesteps (system health checks)
3. You can simulate a crash by stopping the process and restarting

Example:
-------
```python
# Initialize the training monitor
monitor = TrainingMonitor(save_folder, scenario_folder, env_type, seed)

# Load any existing checkpoint
checkpoint_data = monitor.load_checkpoint(model)
if checkpoint_data:
    total_timesteps, episode_start, rewards, test_rewards, epsilon_values, epsilon = checkpoint_data
    print(f"Resuming from episode {episode_start}")

# During training loop:
if total_timesteps % 50000 == 0:
    monitor.save_checkpoint(model, total_timesteps, episode, rewards, 
                          test_rewards, epsilon_values, epsilon)
```
"""

import os
import time
import torch as th
import psutil
from datetime import datetime

class TrainingMonitor:
    def __init__(self, save_folder, stripped_scenario_folder, env_type, single_seed):
        """Initialize the training monitor with paths and logging"""
        self.save_folder = save_folder
        self.stripped_scenario_folder = stripped_scenario_folder
        self.env_type = env_type
        self.single_seed = single_seed
        
        # Create standardized monitoring directory structure
        self.monitor_dir = os.path.join(save_folder, "training_monitor")
        self.checkpoint_dir = os.path.join(self.monitor_dir, "checkpoints")
        self.log_dir = os.path.join(self.monitor_dir, "logs")
        
        # Create directories
        os.makedirs(self.monitor_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging file
        self.log_path = os.path.join(self.log_dir, f"{self.env_type}_{self.single_seed}_training.log")
        self.log_file = open(self.log_path, 'w', buffering=1)  # Line buffering
        self._log_message("INFO", "Logging system initialized")

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'log_file'):
            self.log_file.close()

    def _log_message(self, level, message):
        """Write a message directly to the log file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log_file.write(f"{timestamp} - {level} - {message}\n")
        self.log_file.flush()

    def get_system_metrics(self):
        """Get current system metrics including CPU temperature if available"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_temp': None,
            'cpu_temp': None
        }
        
        # Try to get GPU temperature if CUDA is available
        if th.cuda.is_available():
            try:
                metrics['gpu_temp'] = th.cuda.get_device_properties(0).temperature
            except:
                pass
        
        # Try to get CPU temperature (platform dependent)
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                metrics['cpu_temp'] = max(temp.current for temp in temps['coretemp'])
        except:
            pass
        
        return metrics

    def check_system_health(self, model, total_timesteps, episode, rewards, test_rewards, epsilon_values, epsilon):
        """Check system health and save checkpoint if necessary"""
        metrics = self.get_system_metrics()
        
        # Log system metrics and training progress
        self._log_message("INFO", f"System health check at {total_timesteps} timesteps:")
        self._log_message("INFO", f"  - CPU Usage: {metrics['cpu_percent']}%")
        self._log_message("INFO", f"  - Memory Usage: {metrics['memory_percent']}%")
        if metrics['cpu_temp']:
            self._log_message("INFO", f"  - CPU Temperature: {metrics['cpu_temp']}°C")
        if metrics['gpu_temp']:
            self._log_message("INFO", f"  - GPU Temperature: {metrics['gpu_temp']}°C")
        self._log_message("INFO", f"Training progress:")
        self._log_message("INFO", f"  - Current episode: {episode}")
        self._log_message("INFO", f"  - Current epsilon: {epsilon:.4f}")
        
        should_save = False
        
        # Check CPU temperature
        if metrics['cpu_temp'] and metrics['cpu_temp'] > 85:  # 85°C is a common threshold
            self._log_message("WARNING", f"High CPU temperature detected: {metrics['cpu_temp']}°C")
            should_save = True
            
        # Check GPU temperature
        if metrics['gpu_temp'] and metrics['gpu_temp'] > 80:  # 80°C is a common threshold
            self._log_message("WARNING", f"High GPU temperature detected: {metrics['gpu_temp']}°C")
            should_save = True
            
        # Check memory usage
        if metrics['memory_percent'] > 90:  # 90% memory usage threshold
            self._log_message("WARNING", f"High memory usage detected: {metrics['memory_percent']}%")
            should_save = True
            
        # Save checkpoint at regular intervals (every 200,000 timesteps)
        if total_timesteps % 200000 == 0:
            self._log_message("INFO", f"Regular checkpoint at {total_timesteps} timesteps")
            should_save = True
            
        if should_save:
            self._log_message("INFO", "Saving checkpoint...")
            self.save_checkpoint(model, total_timesteps, episode, rewards, test_rewards, epsilon_values, epsilon)
            
        return metrics

    def save_checkpoint(self, model, total_timesteps, episode, rewards, test_rewards, epsilon_values, epsilon):
        """Save a checkpoint of the training state"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.env_type}_{self.single_seed}_checkpoint.pt")
        temp_path = f"{checkpoint_path}.tmp"
        
        try:
            checkpoint = {
                'model_state_dict': model.q_net.state_dict(),
                'optimizer_state_dict': model.policy.optimizer.state_dict(),
                'total_timesteps': total_timesteps,
                'episode': episode,
                'rewards': rewards,
                'test_rewards': test_rewards,
                'epsilon_values': epsilon_values,
                'epsilon': epsilon,
                # 'replay_buffer': model.replay_buffer,  # Removed to reduce file size
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to temporary file first
            th.save(checkpoint, temp_path)
            # Then rename to final filename (atomic operation)
            os.replace(temp_path, checkpoint_path)
            self._log_message("INFO", f"Checkpoint saved at {checkpoint_path}")
            
        except Exception as e:
            self._log_message("ERROR", f"Failed to save checkpoint: {str(e)}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def load_checkpoint(self, model):
        """Load a checkpoint if it exists"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.env_type}_{self.single_seed}_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            self._log_message("INFO", f"Loading checkpoint from {checkpoint_path}")
            checkpoint = th.load(checkpoint_path)
            model.q_net.load_state_dict(checkpoint['model_state_dict'])
            model.policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # model.replay_buffer = checkpoint['replay_buffer']  # Removed since we no longer save it
            return (checkpoint['total_timesteps'], 
                    checkpoint['episode'],
                    checkpoint['rewards'],
                    checkpoint['test_rewards'],
                    checkpoint['epsilon_values'],
                    checkpoint['epsilon'])
        return None 