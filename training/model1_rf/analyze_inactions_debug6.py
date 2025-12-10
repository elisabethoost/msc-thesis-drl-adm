"""Analyze inaction counts from detailed episodes"""
import pickle
from pathlib import Path

base = Path("results/model1_rf/training/m1_DQN_debug6/3ac-182-green16")
episodes_dir = base / "detailed_episodes"

print("="*80)
print("INACTION ANALYSIS: m1_DQN_debug6")
print("="*80)

for env_type in ['proactive', 'reactive', 'myopic']:
    for seed in ['232323', '242424']:
        pkl_file = episodes_dir / f"{env_type}_detailed_episodes_seed_{seed}.pkl"
        if not pkl_file.exists():
            continue
        
        print(f"\n{env_type.upper()} (seed {seed}):")
        print("-"*80)
        
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # Analyze all episodes
        total_inactions = 0
        total_cancellations = 0
        total_normal = 0
        total_exploitation = 0
        total_exploration = 0
        exploitation_inactions = 0
        exploitation_cancellations = 0
        exploitation_normal = 0
        exploitation_resolutions = 0
        
        # Handle different data structures
        episodes_to_process = []
        for ep_key, ep_data in data.items():
            if isinstance(ep_key, str) and ep_key.isdigit():
                episodes_to_process.append((ep_key, ep_data))
            elif isinstance(ep_data, dict) and 'scenarios' in ep_data:
                episodes_to_process.append((str(ep_key), ep_data))
        
        if not episodes_to_process:
            print(f"  !  No episodes found in data structure")
            print(f"  Data keys: {list(data.keys())[:5]}")
            continue
        
        for ep_key, ep_data in episodes_to_process:
            scenarios = ep_data.get('scenarios', {})
            if not scenarios:
                continue
            
            for sc_path, sc_data in scenarios.items():
                steps = sc_data.get('steps', [])
                if not steps:
                    continue
                
                for step in steps:
                    fa = step.get('flight_action', 0)
                    aa = step.get('aircraft_action', 0)
                    action_reason = step.get('action_reason', '').lower()
                    
                    # Classify action
                    if fa == 0 and aa == 0:
                        total_inactions += 1
                        if 'exploitation' in action_reason:
                            exploitation_inactions += 1
                    elif fa != 0 and aa == 0:
                        total_cancellations += 1
                        if 'exploitation' in action_reason:
                            exploitation_cancellations += 1
                    else:
                        total_normal += 1
                        if 'exploitation' in action_reason:
                            exploitation_normal += 1
                    
                    # Count exploration vs exploitation
                    if 'exploitation' in action_reason:
                        total_exploitation += 1
                        if step.get('penalties', {}).get('probability_resolution_bonus', 0) > 0:
                            exploitation_resolutions += 1
                    elif 'exploration' in action_reason or 'explore' in action_reason:
                        total_exploration += 1
        
        total_actions = total_inactions + total_cancellations + total_normal
        
        if total_actions == 0:
            print(f"  !  No actions found in detailed episodes")
            continue
        
        print(f"Total actions: {total_actions}")
        print(f"  Inactions: {total_inactions} ({total_inactions/total_actions*100:.1f}%)")
        print(f"  Cancellations: {total_cancellations} ({total_cancellations/total_actions*100:.1f}%)")
        print(f"  Normal (swap/delay): {total_normal} ({total_normal/total_actions*100:.1f}%)")
        
        print(f"\nExploitation actions: {total_exploitation}")
        if total_exploitation > 0:
            print(f"  Inactions: {exploitation_inactions} ({exploitation_inactions/total_exploitation*100:.1f}%)")
            print(f"  Cancellations: {exploitation_cancellations} ({exploitation_cancellations/total_exploitation*100:.1f}%)")
            print(f"  Normal: {exploitation_normal} ({exploitation_normal/total_exploitation*100:.1f}%)")
            print(f"  Resolutions: {exploitation_resolutions} ({exploitation_resolutions/total_exploitation*100:.1f}%)")
            
            if exploitation_inactions / total_exploitation > 0.5:
                print(f"  ! CRITICAL: >50% inactions in exploitation!")
            if exploitation_resolutions / total_exploitation < 0.05:
                print(f"  ! CRITICAL: <5% resolution rate in exploitation!")
        else:
            print(f"  !  No exploitation actions found")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
