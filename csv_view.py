import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

def load_and_prepare_data(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create a long-format dataframe for the individual scores
    scores_df = pd.DataFrame()
    for _, row in df.iterrows():
        # Convert string representation of list to actual list
        scores = eval(row['all_scores'])
        # Convert empty string to 'neutral' and handle NaN
        attitude = 'neutral' if pd.isna(row['attitude']) or row['attitude'] == '' else str(row['attitude'])
        
        # Create a shorter feedback label
        if 'disappointing' in row['feedback'].lower():
            feedback_type = 'negative'
        elif 'excellent' in row['feedback'].lower():
            feedback_type = 'positive'
        else:
            feedback_type = 'neutral'
            
        temp_df = pd.DataFrame({
            'attitude': [attitude] * len(scores),
            'feedback': [feedback_type] * len(scores),
            'score': scores
        })
        scores_df = pd.concat([scores_df, temp_df], ignore_index=True)
    
    return df, scores_df

def plot_attitude_pairs(scores_df, output_dir):
    # Get unique attitudes
    attitudes = sorted(scores_df['attitude'].unique())
    feedback_types = ['negative', 'neutral', 'positive']
    colors = {'negative': 'red', 'neutral': 'gray', 'positive': 'green'}
    
    # Create plots for each pair of attitudes
    for att1, att2 in combinations(attitudes, 2):
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Add overall title with padding
        fig.suptitle(f'Score Distribution Comparison\n{att1} vs {att2}', y=1.02, fontsize=14)
        
        # Plot histograms for each feedback type
        for idx, feedback in enumerate(feedback_types):
            ax = axes[idx]
            
            # Data for first attitude
            data1 = scores_df[(scores_df['attitude'] == att1) & 
                            (scores_df['feedback'] == feedback)]['score']
            ax.hist(data1, bins=30, alpha=0.5, 
                   label=f'{att1}',
                   color=colors[feedback])
            
            # Data for second attitude
            data2 = scores_df[(scores_df['attitude'] == att2) & 
                            (scores_df['feedback'] == feedback)]['score']
            ax.hist(data2, bins=30, alpha=0.5, 
                   label=f'{att2}',
                   color='blue')
            
            # Add mean lines
            ax.axvline(data1.mean(), color=colors[feedback], linestyle='-', alpha=0.8)
            ax.axvline(data2.mean(), color='blue', linestyle='-', alpha=0.8)
            
            # Add labels
            ax.set_ylabel('Count')
            ax.set_title(f'{feedback.capitalize()} Feedback', pad=10)
            ax.legend()
            
            # Share x-axis limits across subplots
            ax.set_xlim(scores_df['score'].min() - 0.5, scores_df['score'].max() + 0.5)
        
        # Add x-label to bottom subplot
        axes[-1].set_xlabel('Score')
        
        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.3)
        
        # Save plot with extra padding at top
        safe_name = f'comparison_{att1.replace(" ", "_")}_{att2.replace(" ", "_")}.png'
        plt.savefig(f'{output_dir}/{safe_name}', bbox_inches='tight', pad_inches=0.3)
        plt.close()

def main():
    # Paths
    csv_path = 'results/sycophancy_results.csv'
    output_dir = 'results/attitude_comparisons'
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    df, scores_df = load_and_prepare_data(csv_path)
    
    # Create pairwise comparison plots
    plot_attitude_pairs(scores_df, output_dir)
    
    print(f"Plots have been saved to {output_dir}/")

if __name__ == "__main__":
    main()
