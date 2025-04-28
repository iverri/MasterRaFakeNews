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
    
    # Convert increase_diversity to string for better display in plots
    if 'increase_diversity' in data.columns:
        data['diversity_setting'] = data['increase_diversity'].apply(
            lambda x: "With Diversity" if x else "Without Diversity")
    else:
        # If the column doesn't exist, assume all data is without diversity
        data['diversity_setting'] = "Without Diversity"
        data['increase_diversity'] = False
        
    return data

def plot_misinformation_spread(data, output_path=None):
    """
    Plot misinformation infection over time for each recommender type,
    with separate plots for with and without diversity.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    
    # Filter data for each diversity setting
    without_diversity = data[data["diversity_setting"] == "Without Diversity"]
    with_diversity = data[data["diversity_setting"] == "With Diversity"]
    
    # Plot without diversity
    sns.lineplot(data=without_diversity, x="Step", y="Misinformation_Spread_Percentage", 
                 hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS,
                 linewidth=2.5, ax=axes[0])
    
    axes[0].set_title("Without Diversity", fontsize=16)
    axes[0].set_ylabel("Infection Rate", fontsize=14)
    axes[0].set_xlabel("Simulation Step", fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot with diversity
    sns.lineplot(data=with_diversity, x="Step", y="Misinformation_Spread_Percentage", 
                 hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS,
                 linewidth=2.5, ax=axes[1])
    
    axes[1].set_title("With Diversity", fontsize=16)
    axes[1].set_ylabel("", fontsize=14)
    axes[1].set_xlabel("Simulation Step", fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
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
    with separate plots for with and without diversity.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    
    # Filter data for each diversity setting
    without_diversity = data[data["diversity_setting"] == "Without Diversity"]
    with_diversity = data[data["diversity_setting"] == "With Diversity"]
    
    # Plot without diversity
    sns.lineplot(data=without_diversity, x="Step", y="Misinformation_Ratio_Difference", 
                 hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS,
                 linewidth=2.5, ax=axes[0])
    
    axes[0].set_title("Without Diversity", fontsize=16)
    axes[0].set_ylabel("MRD (positive = amplifying misinformation)", fontsize=14)
    axes[0].set_xlabel("Simulation Step", fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Plot with diversity
    sns.lineplot(data=with_diversity, x="Step", y="Misinformation_Ratio_Difference", 
                 hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS,
                 linewidth=2.5, ax=axes[1])
    
    axes[1].set_title("With Diversity", fontsize=16)
    axes[1].set_ylabel("", fontsize=14)
    axes[1].set_xlabel("Simulation Step", fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
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
    with separate plots for with and without diversity.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    
    # Filter data for each diversity setting
    without_diversity = data[data["diversity_setting"] == "Without Diversity"]
    with_diversity = data[data["diversity_setting"] == "With Diversity"]
    
    # Plot without diversity
    sns.lineplot(data=without_diversity, x="Step", y="Misinformation_Count_In_Recommendations", 
                 hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS,
                 linewidth=2.5, ax=axes[0])
    
    axes[0].set_title("Without Diversity", fontsize=16)
    axes[0].set_ylabel("Average Number of Misinformation Items", fontsize=14)
    axes[0].set_xlabel("Simulation Step", fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot with diversity
    sns.lineplot(data=with_diversity, x="Step", y="Misinformation_Count_In_Recommendations", 
                 hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS,
                 linewidth=2.5, ax=axes[1])
    
    axes[1].set_title("With Diversity", fontsize=16)
    axes[1].set_ylabel("", fontsize=14)
    axes[1].set_xlabel("Simulation Step", fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
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
    with separate plots for with and without diversity.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    
    # Filter data for each diversity setting
    without_diversity = data[data["diversity_setting"] == "Without Diversity"]
    with_diversity = data[data["diversity_setting"] == "With Diversity"]
    
    # Plot without diversity
    sns.lineplot(data=without_diversity, x="Step", y="Echo_Chamber_Effect", 
                 hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS,
                 linewidth=2.5, ax=axes[0])
    
    axes[0].set_title("Without Diversity", fontsize=16)
    axes[0].set_ylabel("Echo Chamber Index", fontsize=14)
    axes[0].set_xlabel("Simulation Step", fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot with diversity
    sns.lineplot(data=with_diversity, x="Step", y="Echo_Chamber_Effect", 
                 hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS,
                 linewidth=2.5, ax=axes[1])
    
    axes[1].set_title("With Diversity", fontsize=16)
    axes[1].set_ylabel("", fontsize=14)
    axes[1].set_xlabel("Simulation Step", fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
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
    Create a summary plot comparing recommenders across multiple metrics,
    showing the average values over the entire simulation period,
    comparing with and without diversity.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Define metrics to plot
    metrics = ["Misinformation_Spread_Percentage", 
               "Misinformation_Ratio_Difference", 
               "Misinformation_Count_In_Recommendations",
               "Echo_Chamber_Effect"]
    
    metric_labels = ["Infection Rate", 
                    "Misinformation Ratio Difference", 
                    "Misinformation Count",
                    "Echo Chamber Effect"]
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 5 * len(metrics)))
    
    # Get unique recommender types and diversity settings
    recommender_types = data["recommender_type"].unique()
    diversity_settings = data["diversity_setting"].unique()
    
    # Process each metric
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # Calculate mean and std for the metric by recommender type and diversity setting across all steps
        metric_data = data.groupby(["recommender_type", "diversity_setting"])[metric].agg(
            ["mean", "std"]).reset_index()
        metric_data.columns = ["recommender_type", "diversity_setting", "value", "std"]
        
        # Sort by recommender type first, then by diversity setting
        metric_data = metric_data.sort_values(["recommender_type", "diversity_setting"])
        
        # Set width of bars
        bar_width = 0.35
        
        # Get unique recommender types and sort them
        rec_types = sorted(metric_data["recommender_type"].unique())
        
        # Set positions for bars
        positions = np.arange(len(rec_types))
        
        # Filter data for with and without diversity
        without_diversity = metric_data[metric_data["diversity_setting"] == "Without Diversity"]
        with_diversity = metric_data[metric_data["diversity_setting"] == "With Diversity"]
        
        # Create dictionary to map recommender types to their positions
        rec_to_pos = {rec: i for i, rec in enumerate(rec_types)}
        
        # Get colors for each recommender type
        without_colors = [RECOMMENDER_COLORS.get(rec, '#999999') for rec in without_diversity["recommender_type"]]
        
        # Create darker versions of the same colors for diversity bars
        with_colors = []
        for rec in with_diversity["recommender_type"]:
            base_color = RECOMMENDER_COLORS.get(rec, '#999999')
            # Convert to RGB and darken
            rgb = plt.cm.colors.to_rgb(base_color)
            darker_rgb = [max(0, c * 0.8) for c in rgb]  # Multiply by 0.8 to darken
            with_colors.append(darker_rgb)
        
        # Plot bars for without diversity
        without_bars = axes[i].barh(
            [rec_to_pos[rec] - bar_width/2 for rec in without_diversity["recommender_type"]], 
            without_diversity["value"], 
            bar_width, 
            xerr=without_diversity["std"], 
            capsize=5,
            color=without_colors,
            label="Standard (lighter)",  # Clearer label mentioning color
            alpha=0.9
        )
        
        # Plot bars for with diversity
        with_bars = axes[i].barh(
            [rec_to_pos[rec] + bar_width/2 for rec in with_diversity["recommender_type"]], 
            with_diversity["value"], 
            bar_width, 
            xerr=with_diversity["std"], 
            capsize=5,
            color=with_colors,
            label="Increased Diversity (darker)",  # Clearer label mentioning color
            alpha=0.9
        )
        
        # Add value labels with improved visibility
        for bars in [without_bars, with_bars]:
            for bar in bars:
                width = bar.get_width()
                # Position the label at the end of the bar
                label_x = width + (0.01 * max(metric_data["value"]))
                label_y = bar.get_y() + bar.get_height()/2
                # Add a white background to the text for better visibility
                axes[i].text(label_x, label_y, 
                            f'{width:.3f}', va='center', fontsize=10, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Add a vertical line at x=0 for metrics where it makes sense
        if metric == "Misinformation_Ratio_Difference":
            axes[i].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Set title and labels
        axes[i].set_title(f"Average {label} by Recommender Type and Diversity Setting", fontsize=14)
        axes[i].set_xlabel(label, fontsize=12)
        axes[i].set_ylabel("")
        
        # Set y-ticks to recommender types
        axes[i].set_yticks(positions)
        axes[i].set_yticklabels(rec_types)
        
        # Add grid
        axes[i].grid(True, linestyle='--', alpha=0.7, axis='x')
        
        # Add legend with improved styling
        if i == 0:
            # Create a custom legend with larger font and better positioning
            legend = axes[i].legend(
                loc='upper right', 
                fontsize=12, 
                framealpha=0.9,
                title="Recommendation Strategy", 
                title_fontsize=13,
                bbox_to_anchor=(1.0, 1.0)
            )
            # Add a border to the legend for better visibility
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.5)
        
        # Adjust x-axis limits to make room for the labels
        x_max = max(metric_data["value"] + metric_data["std"]) * 1.2
        axes[i].set_xlim(right=x_max)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def generate_comparison_dashboard(data, output_path=None):
    """
    Generate a dashboard with multiple metrics for comparing recommender algorithms,
    with and without diversity.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Misinformation Infection
    sns.lineplot(data=data, x="Step", y="Misinformation_Spread_Percentage", 
                 hue="recommender_type", style="diversity_setting", 
                 errorbar="sd", ax=axes[0, 0],
                 palette=RECOMMENDER_COLORS)
    axes[0, 0].set_title("Infection Rate by Recommender Type and Diversity Setting", fontsize=14)
    axes[0, 0].set_ylabel("Infection Rate", fontsize=12)
    
    # Plot 2: Misinformation Ratio Difference
    sns.lineplot(data=data, x="Step", y="Misinformation_Ratio_Difference", 
                 hue="recommender_type", style="diversity_setting", 
                 errorbar="sd", ax=axes[0, 1],
                 palette=RECOMMENDER_COLORS)
    axes[0, 1].set_title("Misinformation Ratio Difference", fontsize=14)
    axes[0, 1].set_ylabel("MRD (positive = amplifying misinfo)", fontsize=12)
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Plot 3: Misinformation Count in Recommendations
    sns.lineplot(data=data, x="Step", y="Misinformation_Count_In_Recommendations", 
                 hue="recommender_type", style="diversity_setting", 
                 errorbar="sd", ax=axes[1, 0],
                 palette=RECOMMENDER_COLORS)
    axes[1, 0].set_title("Misinformation Count in Recommendations", fontsize=14)
    axes[1, 0].set_ylabel("Average Number of Misinformation Items", fontsize=12)
    
    # Plot 4: Echo Chamber Effect
    sns.lineplot(data=data, x="Step", y="Echo_Chamber_Effect", 
                 hue="recommender_type", style="diversity_setting", 
                 errorbar="sd", ax=axes[1, 1],
                 palette=RECOMMENDER_COLORS)
    axes[1, 1].set_title("Echo Chamber Effect", fontsize=14)
    axes[1, 1].set_ylabel("Echo Chamber Effect", fontsize=12)
    
    # Adjust layout and add a main title
    plt.suptitle("Recommender Algorithm Comparison Dashboard (With vs. Without Diversity)", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
    
    # Create a single legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flat:
        ax.get_legend().remove()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels)//2, bbox_to_anchor=(0.5, 0), fontsize=12)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_recommender_ranking_table(data, output_path=None):
    """
    Create a table ranking recommenders by different metrics,
    using average values across all simulation steps,
    comparing with and without diversity.
    
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
    
    # Create separate tables for with and without diversity
    for diversity_setting in ["Without Diversity", "With Diversity"]:
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
            diversity_suffix = "with_diversity" if diversity_setting == "With Diversity" else "without_diversity"
            file_path = output_path.replace(".png", f"_{diversity_suffix}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
    
    # Return the last created figure or None if no figures were created
    try:
        return fig
    except UnboundLocalError:
        print("No tables were created.")
        return None

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
    
    # Generate comparison dashboard
    generate_comparison_dashboard(data, os.path.join(output_dir, "recommender_comparison_dashboard.png"))
    
    # Generate final misinformation count bar plot
    
    
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


