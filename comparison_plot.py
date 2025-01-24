import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import ast
import numpy as np
from typing import Dict, List
import argparse

def plot_augmentation_pair(df: pd.DataFrame, experiment_name: str, output_path: str):
    """Create histograms for a specific experiment's augmentation pair"""
    # Get the data for this experiment
    exp_data = df[df['experiment'] == experiment_name]
    aug1 = exp_data['aug1'].iloc[0]  # Get the first augmentation
    aug2 = exp_data['aug2'].iloc[0]  # Get the second augmentation
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    
    # Add main title with more space
    fig.suptitle(f'Reward Distributions:\n"{aug1}" vs "{aug2}"', 
                 fontsize=16, fontweight='bold', y=0.99)
    
    # Add text boxes for column headers with more space and formatting
    plt.figtext(0.3, 0.945, f'<{aug1}> Poem 1\n<{aug2}> Poem 2', 
                ha='center', weight='bold', style='normal', usetex=False,
                bbox={'boxstyle': 'round', 'facecolor': 'white', 'edgecolor': 'black'})
    plt.figtext(0.7, 0.945, f'<{aug2}> Poem 1\n<{aug1}> Poem 2', 
                ha='center', weight='bold', style='normal', usetex=False,
                bbox={'boxstyle': 'round', 'facecolor': 'white', 'edgecolor': 'black'})
    
    # Get all rewards and set axis limits using percentiles plus padding
    all_rewards = pd.concat([exp_data['reward_first'], exp_data['reward_second']])
    reward_min = np.percentile(all_rewards, 0.5)
    reward_max = np.percentile(all_rewards, 99.5)
    reward_range = reward_max - reward_min
    reward_min = reward_min - reward_range * 0.1
    reward_max = reward_max + reward_range * 0.1
    bins = np.linspace(reward_min, reward_max, 20)
    
    # Forward direction: (poem1, aug1) vs (poem2, aug2)
    forward_data = exp_data[
        (exp_data['aug1'] == aug1) &
        (exp_data['aug2'] == aug2)
    ]
    
    # Reverse direction: (poem2, aug2) vs (poem1, aug1)
    reverse_pairs = set(zip(forward_data['poem2_idx'], forward_data['poem1_idx']))
    reverse_data = exp_data[
        (exp_data['aug1'] == aug2) &  # Swapped aug1 and aug2
        (exp_data['aug2'] == aug1) &  # Swapped aug1 and aug2
        exp_data.apply(lambda x: (x['poem1_idx'], x['poem2_idx']) in reverse_pairs, axis=1)
    ]
    
    print(f"\nForward direction pairs ({aug1} → {aug2}):")
    print(f"Number of pairs: {len(forward_data)}")
    print("Sample of poem pairs and rewards:")
    for _, row in forward_data.head(3).iterrows():
        print(f"Poem pair ({row['poem1_idx']}, {row['poem2_idx']})")
        print(f"  First poem reward: {row['reward_first']:.3f}")
        print(f"  Second poem reward: {row['reward_second']:.3f}")
    
    print(f"\nReverse direction pairs ({aug2} → {aug1}):")
    print(f"Number of pairs: {len(reverse_data)}")
    print("Sample of poem pairs and rewards:")
    for _, row in reverse_data.head(3).iterrows():
        print(f"Poem pair ({row['poem1_idx']}, {row['poem2_idx']})")
        print(f"  First poem reward: {row['reward_first']:.3f}")
        print(f"  Second poem reward: {row['reward_second']:.3f}")
    
    # Calculate differences (aug2 reward - aug1 reward for both directions)
    forward_diffs = forward_data['reward_second'] - forward_data['reward_first']
    if not reverse_data.empty:
        reverse_diffs = reverse_data['reward_first'] - reverse_data['reward_second']
    else:
        print(f"Warning: No reverse pairs found for experiment {experiment_name}")
        reverse_diffs = pd.Series([])
    
    # Get all differences and set axis limits
    all_diffs = pd.concat([forward_diffs, reverse_diffs])
    if not all_diffs.empty:
        diff_min = np.percentile(all_diffs, 0.5)
        diff_max = np.percentile(all_diffs, 99.5)
        diff_range = diff_max - diff_min
        diff_min = diff_min - diff_range * 0.1
        diff_max = diff_max + diff_range * 0.1
        diff_bins = np.linspace(diff_min, diff_max, 20)
    else:
        diff_bins = bins
    
    max_count = 0
    
    # Forward direction plots (first column)
    if len(forward_data) > 0:
        # Rewards for choosing aug1
        n, _, _ = axes[0,0].hist(forward_data['reward_first'], bins=bins, alpha=0.7, 
                                color='blue', label='Reward')
        max_count = max(max_count, max(n))
        mean = forward_data['reward_first'].mean()
        std = forward_data['reward_first'].std()
        axes[0,0].axvline(mean, color='darkblue', linestyle='--', label=f'μ={mean:.2f}')
        axes[0,0].axvspan(mean-std, mean+std, color='blue', alpha=0.2, label=f'σ={std:.2f}')
        axes[0,0].set_title(f'Choosing Poem 1 with <{aug1}>', pad=10, fontweight='bold')
        axes[0,0].legend(fontsize='small', loc='upper right')
        
        # Rewards for choosing aug2
        n, _, _ = axes[1,0].hist(forward_data['reward_second'], bins=bins, alpha=0.7, 
                                color='red', label='Reward')
        max_count = max(max_count, max(n))
        mean = forward_data['reward_second'].mean()
        std = forward_data['reward_second'].std()
        axes[1,0].axvline(mean, color='darkred', linestyle='--', label=f'μ={mean:.2f}')
        axes[1,0].axvspan(mean-std, mean+std, color='red', alpha=0.2, label=f'σ={std:.2f}')
        axes[1,0].set_title(f'Choosing Poem 2 with <{aug2}>', pad=10, fontweight='bold')
        axes[1,0].legend(fontsize='small', loc='upper right')
        
        # Difference plot
        forward_diffs = forward_data['reward_first'] - forward_data['reward_second']
        n, _, _ = axes[2,0].hist(forward_diffs, bins=diff_bins, alpha=0.7,
                                color='purple', label='Reward Difference')
        mean = forward_diffs.mean()
        std = forward_diffs.std()
        axes[2,0].axvline(mean, color='darkviolet', linestyle='--', label=f'μ={mean:.2f}')
        axes[2,0].axvspan(mean-std, mean+std, color='purple', alpha=0.2, label=f'σ={std:.2f}')
        axes[2,0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[2,0].set_title(f'<{aug1}> (Poem 1)\n- <{aug2}> (Poem 2)', pad=10, fontweight='bold')
        axes[2,0].legend(fontsize='small', loc='upper right')
    
    # Reverse direction plots (second column)
    if len(reverse_data) > 0:
        # Rewards for choosing aug2 (now first)
        n, _, _ = axes[0,1].hist(reverse_data['reward_first'], bins=bins, alpha=0.7,
                                color='blue', label='Reward')
        max_count = max(max_count, max(n))
        mean = reverse_data['reward_first'].mean()
        std = reverse_data['reward_first'].std()
        axes[0,1].axvline(mean, color='darkblue', linestyle='--', label=f'μ={mean:.2f}')
        axes[0,1].axvspan(mean-std, mean+std, color='blue', alpha=0.2, label=f'σ={std:.2f}')
        axes[0,1].set_title(f'Choosing Poem 1 with <{aug2}>', pad=10, fontweight='bold')
        axes[0,1].legend(fontsize='small', loc='upper right')
        
        # Rewards for choosing aug1 (now second)
        n, _, _ = axes[1,1].hist(reverse_data['reward_second'], bins=bins, alpha=0.7,
                                color='red', label='Reward')
        max_count = max(max_count, max(n))
        mean = reverse_data['reward_second'].mean()
        std = reverse_data['reward_second'].std()
        axes[1,1].axvline(mean, color='darkred', linestyle='--', label=f'μ={mean:.2f}')
        axes[1,1].axvspan(mean-std, mean+std, color='red', alpha=0.2, label=f'σ={std:.2f}')
        axes[1,1].set_title(f'Choosing Poem 2 with <{aug1}>', pad=10, fontweight='bold')
        axes[1,1].legend(fontsize='small', loc='upper right')
        
        # Difference plot
        reverse_diffs = reverse_data['reward_second'] - reverse_data['reward_first']
        n, _, _ = axes[2,1].hist(reverse_diffs, bins=diff_bins, alpha=0.7,
                                color='purple', label='Reward Difference')
        mean = reverse_diffs.mean()
        std = reverse_diffs.std()
        axes[2,1].axvline(mean, color='darkviolet', linestyle='--', label=f'μ={mean:.2f}')
        axes[2,1].axvspan(mean-std, mean+std, color='purple', alpha=0.2, label=f'σ={std:.2f}')
        axes[2,1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[2,1].set_title(f'<{aug1}> (Poem 2)\n- <{aug2}> (Poem 1)', pad=10, fontweight='bold')
        axes[2,1].legend(fontsize='small', loc='upper right')
    
    # Set consistent axes and labels
    for ax in axes.flatten():
        if ax in axes[2,:]:  # Difference plots
            ax.set_xlim(diff_min, diff_max)
            ax.set_xlabel('Reward Difference')
        else:  # Regular reward plots
            ax.set_xlim(reward_min, reward_max)
            ax.set_xlabel('Reward Score')
        ax.set_ylim(0, max_count * 1.1)
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, left=0.1)  # Increased space at top (changed from 0.92 to 0.90)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create plots from experiment results')
    parser.add_argument('--model', type=str, default=None,
                      help='Specific model name to plot (if None, plots all models)')
    args = parser.parse_args()
    
    # Setup paths
    results_dir = Path("./urop_quick/results")
    plots_base_dir = Path("plots")
    plots_base_dir.mkdir(exist_ok=True)
    
    # Get all result files or specific model results
    if args.model:
        result_files = [results_dir / f"{args.model}.pkl"]
    else:
        result_files = list(results_dir.glob("*.pkl"))
    
    for result_file in result_files:
        # Extract model name from filename
        model_name = result_file.stem
        print(f"\nProcessing results for model: {model_name}")
        
        # Create model-specific plots directory
        plots_dir = plots_base_dir / model_name
        plots_dir.mkdir(exist_ok=True)
        
        # Load results
        with open(result_file, 'rb') as f:
            data = pickle.load(f)
        
        # Convert to DataFrame with experiment name
        analyzed_data = []
        for key_str, value in data['results'].items():
            key = ast.literal_eval(key_str)
            rewards = value['rewards']
            
            analyzed_data.append({
                'experiment': key['experiment'],  # Get experiment from the key
                'aug1': key['aug1'] if key['aug1'] != "" else "no augmentation",
                'aug2': key['aug2'] if key['aug2'] != "" else "no augmentation",
                'reward_first': rewards['I prefer the first poem.'],
                'reward_second': rewards['I prefer the second poem.'],
                'preferred': value['preferred'],
                'poem1_idx': key['poem1_idx'],
                'poem2_idx': key['poem2_idx']
            })
        
        df = pd.DataFrame(analyzed_data)
        
        # Print unique experiments found
        print("\nExperiments found in data:")
        for exp in df['experiment'].unique():
            print(f"- {exp}")
        
        # Create plots for each experiment
        for experiment in df['experiment'].unique():
            print(f"\nCreating plot for experiment: {experiment}")
            exp_data = df[df['experiment'] == experiment]
            print(f"Number of samples in experiment: {len(exp_data)}")
            output_path = plots_dir / f'experiment_{experiment}.png'
            plot_augmentation_pair(df, experiment, output_path)