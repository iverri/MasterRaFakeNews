import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
import os

# Define a global color mapping for all recommender types
RECOMMENDER_COLORS = {
    'random': '#1f77b4',           # blue
    'popular': '#e377c2',       # pink
    'content_based': '#2ca02c',    # green
    'user_knn': '#ff7f0e',    # orange
    'item_knn': '#d62728',           # red
}


def load_experiment_data(csv_path):
    """
    Load experiment data from CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing experiment results
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing experiment results
    """
    data = pd.read_csv(csv_path)
    
    # Convert diversity_level to a categorical label for better display in plots
    if 'diversity_level' in data.columns:
        # Create a readable diversity setting label
        data['diversity_setting'] = data['diversity_level'].apply(
            lambda x: "No Diversity" if x == 0 else f"Diversity {x}")
    else:
        # If the column doesn't exist, assume all data is without diversity
        data['diversity_setting'] = "No Diversity"
        data['diversity_level'] = 0
        
    return data

def plot_misinformation_spread(data, output_path=None):
    """
    Plot misinformation infection over time for each recommender type,
    with separate plots for different diversity levels.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Get unique diversity settings
    diversity_settings = sorted(data['diversity_setting'].unique())
    
    # Create a figure with subplots for each diversity setting
    fig, axes = plt.subplots(1, len(diversity_settings), figsize=(6*len(diversity_settings), 8), sharey=True)
    
    # Handle case with only one diversity setting
    if len(diversity_settings) == 1:
        axes = [axes]
    
    # Plot for each diversity setting
    for i, diversity in enumerate(diversity_settings):
        # Filter data for this diversity setting
        filtered_data = data[data["diversity_setting"] == diversity]
        
        # Plot data
        sns.lineplot(data=filtered_data, x="Step", y="Misinformation_Spread_Percentage", 
                     hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS,
                     linewidth=2.5, ax=axes[i])
        
        axes[i].set_title(f"{diversity}", fontsize=16)
        axes[i].set_xlabel("Simulation Step", fontsize=14)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Only add y-label to the first subplot
        if i == 0:
            axes[i].set_ylabel("Infection Rate", fontsize=14)
        else:
            axes[i].set_ylabel("", fontsize=14)
    
    # Add a main title
    plt.suptitle("Misinformation Infection Rate by Recommender Type", fontsize=18, y=1.05)
    
    # Adjust legend for better readability
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='upper center', ncol=len(labels), fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_misinformation_ratio_difference(data, output_path=None):
    """
    Plot misinformation ratio difference over time for each recommender type,
    with separate plots for different diversity levels.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Get unique diversity settings
    diversity_settings = sorted(data['diversity_setting'].unique())
    
    # Create a figure with subplots for each diversity setting
    fig, axes = plt.subplots(1, len(diversity_settings), figsize=(6*len(diversity_settings), 8), sharey=True)
    
    # Handle case with only one diversity setting
    if len(diversity_settings) == 1:
        axes = [axes]
    
    # Plot for each diversity setting
    for i, diversity in enumerate(diversity_settings):
        # Filter data for this diversity setting
        filtered_data = data[data["diversity_setting"] == diversity]
        
        # Plot data
        sns.lineplot(data=filtered_data, x="Step", y="Misinformation_Ratio_Difference", 
                     hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS,
                     linewidth=2.5, ax=axes[i])
        
        axes[i].set_title(f"{diversity}", fontsize=16)
        axes[i].set_xlabel("Simulation Step", fontsize=14)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Only add y-label to the first subplot
        if i == 0:
            axes[i].set_ylabel("MRD (positive = amplifying misinformation)", fontsize=14)
        else:
            axes[i].set_ylabel("", fontsize=14)
    
    # Add a main title
    plt.suptitle("Misinformation Ratio Difference by Recommender Type", fontsize=18, y=1.05)
    
    # Adjust legend for better readability
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='upper center', ncol=len(labels), fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_misinformation_count(data, output_path=None):
    """
    Plot average misinformation count in recommendations for each recommender type,
    with separate plots for different diversity levels.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Get unique diversity settings
    diversity_settings = sorted(data['diversity_setting'].unique())
    
    # Create a figure with subplots for each diversity setting
    fig, axes = plt.subplots(1, len(diversity_settings), figsize=(6*len(diversity_settings), 8), sharey=True)
    
    # Handle case with only one diversity setting
    if len(diversity_settings) == 1:
        axes = [axes]
    
    # Plot for each diversity setting
    for i, diversity in enumerate(diversity_settings):
        # Filter data for this diversity setting
        filtered_data = data[data["diversity_setting"] == diversity]
        
        # Plot data
        sns.lineplot(data=filtered_data, x="Step", y="Misinformation_Count_In_Recommendations", 
                     hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS,
                     linewidth=2.5, ax=axes[i])
        
        axes[i].set_title(f"{diversity}", fontsize=16)
        axes[i].set_xlabel("Simulation Step", fontsize=14)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Only add y-label to the first subplot
        if i == 0:
            axes[i].set_ylabel("Average Number of Misinformation Items", fontsize=14)
        else:
            axes[i].set_ylabel("", fontsize=14)
    
    # Add a main title
    plt.suptitle("Average Misinformation Count in Recommendations by Recommender Type", fontsize=18, y=1.05)
    
    # Adjust legend for better readability
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='upper center', ncol=len(labels), fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_echo_chamber_effect(data, output_path=None):
    """
    Plot echo chamber effect over time for each recommender type,
    with separate plots for different diversity levels.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Get unique diversity settings
    diversity_settings = sorted(data['diversity_setting'].unique())
    
    # Create a figure with subplots for each diversity setting
    fig, axes = plt.subplots(1, len(diversity_settings), figsize=(6*len(diversity_settings), 8), sharey=True)
    
    # Handle case with only one diversity setting
    if len(diversity_settings) == 1:
        axes = [axes]
    
    # Plot for each diversity setting
    for i, diversity in enumerate(diversity_settings):
        # Filter data for this diversity setting
        filtered_data = data[data["diversity_setting"] == diversity]
        
        # Plot data
        sns.lineplot(data=filtered_data, x="Step", y="Echo_Chamber_Effect", 
                     hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS,
                     linewidth=2.5, ax=axes[i])
        
        axes[i].set_title(f"{diversity}", fontsize=16)
        axes[i].set_xlabel("Simulation Step", fontsize=14)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Only add y-label to the first subplot
        if i == 0:
            axes[i].set_ylabel("Echo Chamber Index", fontsize=14)
        else:
            axes[i].set_ylabel("", fontsize=14)
    
    # Add a main title
    plt.suptitle("Echo Chamber Effect by Recommender Type", fontsize=18, y=1.05)
    
    # Adjust legend for better readability
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='upper center', ncol=len(labels), fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_recommender_summary(data, output_path=None):
    """
    Create separate summary plots for each metric, with subplots for each diversity level,
    similar to the timeline plot layout.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plots. If None, the plots are not saved.
    """
    # Define metrics to plot
    metrics = {
        "Misinformation_Spread_Percentage": "Infection Rate", 
        "Misinformation_Ratio_Difference": "Misinformation Ratio Difference", 
        "Misinformation_Count_In_Recommendations": "Misinformation Count",
        "Echo_Chamber_Effect": "Echo Chamber Effect"
    }
    
    # Get unique diversity settings and recommender types
    diversity_settings = sorted(data["diversity_setting"].unique())
    recommender_types = sorted(data["recommender_type"].unique())
    
    # Create a figure for each metric
    for metric, label in metrics.items():
        # Create a figure with subplots for each diversity setting
        fig, axes = plt.subplots(1, len(diversity_settings), 
                                figsize=(16, 6), 
                                sharey=True)
        
        # Handle case with only one diversity setting
        if len(diversity_settings) == 1:
            axes = [axes]
        
        # Calculate average values for each recommender and diversity setting
        avg_data = data.groupby(["recommender_type", "diversity_setting"])[metric].agg(
            ["mean", "std"]).reset_index()
        avg_data.columns = ["recommender_type", "diversity_setting", "mean", "std"]
        
        # Plot for each diversity setting
        for i, diversity in enumerate(diversity_settings):
            # Filter data for this diversity setting
            setting_data = avg_data[avg_data["diversity_setting"] == diversity]
            
            # Sort by mean value for better visualization
            setting_data = setting_data.sort_values("mean")
            
            # Create bar chart
            bars = axes[i].barh(
                setting_data["recommender_type"],
                setting_data["mean"],
                xerr=setting_data["std"],
                capsize=5,
                color=[RECOMMENDER_COLORS[rec] for rec in setting_data["recommender_type"]],
                alpha=0.8
            )
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                axes[i].text(
                    width + (0.01 * avg_data["mean"].max()),
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', 
                    va='center', 
                    fontsize=10, 
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                )
            
            # Add a vertical line at x=0 for metrics where it makes sense
            if metric == "Misinformation_Ratio_Difference":
                axes[i].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            
            axes[i].set_title(f"{diversity}", fontsize=14)
            axes[i].set_xlabel(label, fontsize=12)
            axes[i].grid(True, linestyle='--', alpha=0.7, axis='x')
            
            # Only add y-label to the first subplot
            if i == 0:
                axes[i].set_ylabel("Recommender Type", fontsize=12)
        
        # Add a main title
        plt.suptitle(f"{label} by Recommender Type", fontsize=16, y=1.02)
        
        plt.tight_layout()
        
        # Save the plot if output path is provided
        if output_path:
            metric_name = metric.lower().replace('_', '')
            file_path = output_path.replace(".png", f"_{metric_name}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
    
    # Return the last created figure
    return fig


def create_recommender_ranking_table(data, output_path=None):
    """
    Create a table ranking recommenders by different metrics,
    using average values across all simulation steps,
    comparing different diversity levels.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the table. If None, the table is not saved.
    """
    # Define metrics and whether lower is better
    metrics = {
        "Misinformation_Spread_Percentage": {"label": "Infection Rate", "lower_better": True},
        "Misinformation_Ratio_Difference": {"label": "MRD", "lower_better": True},
        "Misinformation_Count_In_Recommendations": {"label": "Misinformation Count", "lower_better": True},
        "Average_Diversity_Score": {"label": "Diversity Score", "lower_better": False},
        # "Echo_Chamber_Effect": {"label": "Echo Chamber", "lower_better": True}
    }
    
    # Create separate tables for each diversity setting
    for diversity_setting in sorted(data['diversity_setting'].unique()):
        # Filter data for the current diversity setting
        filtered_data = data[data["diversity_setting"] == diversity_setting]
        
        # Calculate average for each metric and recommender across all steps
        summary = {}
        for metric, info in metrics.items():
            if metric in filtered_data.columns:
                # Group by recommender type and calculate mean across all steps
                metric_summary = filtered_data.groupby("recommender_type")[metric].mean().reset_index()
                
                # Sort based on whether lower is better
                metric_summary = metric_summary.sort_values(metric, ascending=info["lower_better"])
                
                # Assign ranks
                metric_summary["rank"] = range(1, len(metric_summary) + 1)
                
                # Store in summary dict
                summary[metric] = metric_summary
        
        # Skip if no metrics were found
        if not summary:
            print(f"No metrics found for {diversity_setting}, skipping table creation.")
            continue
        
        # Create a figure for the table
        fig, ax = plt.subplots(figsize=(10, len(summary[list(summary.keys())[0]]) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        recommender_types = summary[list(summary.keys())[0]]["recommender_type"].tolist()
        
        # Header row
        header = ["Recommender Type"] + [info["label"] for metric, info in metrics.items() if metric in summary]
        table_data.append(header)
        
        # Data rows
        for rec_type in recommender_types:
            row = [rec_type]
            for metric in metrics.keys():
                if metric in summary:
                    rank = summary[metric].loc[summary[metric]["recommender_type"] == rec_type, "rank"].values[0]
                    value = summary[metric].loc[summary[metric]["recommender_type"] == rec_type, metric].values[0]
                    row.append(f"#{rank} ({value:.3f})")
            table_data.append(row)
        
        # Check if we have any data rows before creating the table
        if len(table_data) <= 1:  # Only header row exists
            print(f"No data rows for {diversity_setting}, skipping table creation.")
            plt.close(fig)
            continue
            
        # Create table
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                        loc='center', cellLoc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Color the cells based on rank
        for i in range(len(recommender_types)):
            for j in range(1, len(header)):
                cell = table[i+1, j]
                rank = int(cell.get_text().get_text().split('#')[1].split(' ')[0])
                
                # Color gradient from green (rank 1) to red (last rank)
                color_val = rank / len(recommender_types)
                cell.set_facecolor((color_val, 1 - color_val, 0, 0.3))
        
        plt.title(f"Recommender Algorithm Rankings by Metric ({diversity_setting})", fontsize=16, pad=20)
        plt.tight_layout()
        
        if output_path:
            # Add diversity setting to the filename
            diversity_suffix = diversity_setting.replace(" ", "_").lower()
            file_path = output_path.replace(".png", f"_{diversity_suffix}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
    
    # Return the last created figure or None if no figures were created
    try:
        return fig
    except UnboundLocalError:
        print("No tables were created.")
        return None

def plot_diversity_impact_heatmap(data, output_path=None):
    """
    Create a heatmap showing how different diversity levels affect each recommender type
    for key metrics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Define metrics to analyze
    metrics = {
        "Misinformation_Spread_Percentage": "Infection Rate",
        "Misinformation_Ratio_Difference": "MRD",
        "Misinformation_Count_In_Recommendations": "Misinfo Count",
        "Echo_Chamber_Effect": "Echo Chamber Effect"
    }
    
    # Get unique recommender types and diversity levels
    recommender_types = sorted(data["recommender_type"].unique())
    diversity_levels = sorted(data["diversity_level"].unique())
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # Process each metric
    for i, (metric, label) in enumerate(metrics.items()):
        # Calculate mean for the metric by recommender type and diversity level
        metric_data = data.groupby(["recommender_type", "diversity_level"])[metric].mean().reset_index()
        
        # Pivot the data for heatmap
        pivot_data = metric_data.pivot(index="recommender_type", columns="diversity_level", values=metric)
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="RdYlGn_r" if metric != "Average_Diversity_Score" else "RdYlGn",
                   ax=axes[i], cbar_kws={'label': label})
        
        axes[i].set_title(f"Impact of Diversity Level on {label}", fontsize=14)
        axes[i].set_xlabel("Diversity Level", fontsize=12)
        axes[i].set_ylabel("Recommender Type", fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def generate_all_plots(csv_path, output_dir=None):
    """
    Generate all plots for the given experiment data.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing experiment results
    output_dir : str, optional
        Directory to save the plots. If None, plots are saved in the current directory.
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = ""
    
    # Load data
    data = load_experiment_data(csv_path)
    
    # Generate individual plots
    plot_misinformation_spread(data, os.path.join(output_dir, "misinformation_infection_comparison.png"))
    plot_misinformation_ratio_difference(data, os.path.join(output_dir, "misinformation_ratio_difference_comparison.png"))
    plot_misinformation_count(data, os.path.join(output_dir, "misinformation_count_comparison.png"))
    plot_echo_chamber_effect(data, os.path.join(output_dir, "echo_chamber_effect_comparison.png"))
  
    # Generate summary plots
    plot_recommender_summary(data, os.path.join(output_dir, "recommender_summary.png"))
    
    # Generate ranking table
    create_recommender_ranking_table(data, os.path.join(output_dir, "recommender_ranking_table.png"))
   
    # Generate diversity impact heatmap
    plot_diversity_impact_heatmap(data, os.path.join(output_dir, "diversity_impact_heatmap.png"))
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate plots from experiment data')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing experiment results')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save the plots')
    
    args = parser.parse_args()
    
    generate_all_plots(args.csv_file, args.output_dir)


