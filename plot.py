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
    return data

def plot_misinformation_spread(data, output_path=None):
    """
    Plot misinformation infection over time for each recommender type.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    plt.figure(figsize=(12, 8))
    
    # Use the global color mapping
    sns.lineplot(data=data, x="Step", y="Misinformation_Spread_Percentage", 
                 hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS,
                 linewidth=2.5, marker="o", markersize=6, markevery=10)
    
    plt.title("Misinformation Infection by Recommender Type", fontsize=16)
    plt.ylabel("Percentage of Population Infected/Exposed", fontsize=14)
    plt.xlabel("Simulation Step", fontsize=14)
    
    # Improve legend
    plt.legend(title="Recommender Type", title_fontsize=12, fontsize=10, 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid for easier reading
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_misinformation_ratio_difference(data, output_path=None):
    """
    Plot misinformation ratio difference over time for each recommender type.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, x="Step", y="Misinformation_Ratio_Difference", 
                 hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS)
    plt.title("Misinformation Ratio Difference by Recommender Type")
    plt.ylabel("MRD (positive = amplifying misinformation)")
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)  # Add a reference line at y=0
    
    if output_path:
        plt.savefig(output_path)
    
    return plt.gcf()

def plot_final_mrd_bar(data, output_path=None):
    """
    Create a bar plot of final MRD values for each recommender type.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Get the final step for each run
    final_steps = data.groupby(["RunId", "iteration", "recommender_type"])["Step"].max().reset_index()
    final_data = pd.merge(data, final_steps, 
                         on=["RunId", "iteration", "recommender_type", "Step"])

    # Calculate average final MRD for each recommender type
    mrd_summary = final_data.groupby("recommender_type")["Misinformation_Ratio_Difference"].agg(
        ["mean", "std"]).reset_index()

    # Sort by mean MRD
    mrd_summary = mrd_summary.sort_values("mean")
    
    # Create bar plot
    plt.figure(figsize=(14, 8))
    
    # Create a list of colors matching the order of recommender types in mrd_summary
    bar_colors = []
    for rec_type in mrd_summary["recommender_type"]:
        if rec_type in RECOMMENDER_COLORS:
            bar_colors.append(RECOMMENDER_COLORS[rec_type])
        else:
            # Use a default color if the recommender type is not in the color mapping
            bar_colors.append('#999999')  # Gray as fallback
    
    bars = plt.bar(mrd_summary["recommender_type"], mrd_summary["mean"], 
            yerr=mrd_summary["std"], capsize=10, 
            color=bar_colors)

    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                 height + (0.01 if height >= 0 else -0.03),
                 f'{height:.3f}', 
                 ha='center', va='bottom' if height >= 0 else 'top')

    plt.title("Final Misinformation Ratio Difference by Recommender Type", fontsize=16)
    plt.ylabel("MRD (positive = amplifying misinformation)", fontsize=14)
    plt.xlabel("Recommender Type", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    return plt.gcf(), mrd_summary

def plot_mrd_boxplot(data, output_path=None):
    """
    Create a boxplot showing the distribution of MRD values for each recommender type.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Get the final step for each run
    final_steps = data.groupby(["RunId", "iteration", "recommender_type"])["Step"].max().reset_index()
    final_data = pd.merge(data, final_steps, 
                         on=["RunId", "iteration", "recommender_type", "Step"])

    # Calculate average final MRD for each recommender type to determine order
    mrd_summary = final_data.groupby("recommender_type")["Misinformation_Ratio_Difference"].mean().reset_index()
    mrd_summary = mrd_summary.sort_values("Misinformation_Ratio_Difference")

    # Create a palette that handles missing recommender types
    boxplot_palette = {rec: RECOMMENDER_COLORS.get(rec, '#999999') 
                      for rec in mrd_summary["recommender_type"]}
    
    # Create boxplot
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=final_data, x="recommender_type", y="Misinformation_Ratio_Difference",
               order=mrd_summary["recommender_type"], palette=boxplot_palette)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.title("Distribution of Misinformation Ratio Difference by Recommender Type", fontsize=16)
    plt.ylabel("MRD (positive = amplifying misinformation)", fontsize=14)
    plt.xlabel("Recommender Type", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    return plt.gcf()

def plot_misinformation_count(data, output_path=None):
    """
    Plot average misinformation count in recommendations for each recommender type.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, x="Step", y="Misinformation_Count_In_Recommendations", 
                 hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS)
    plt.title("Average Fake News Items in Recommendations by Recommender Type")
    plt.ylabel("Average Number of Fake News Items")
    
    if output_path:
        plt.savefig(output_path)
    
    return plt.gcf()

def plot_final_misinfo_count_bar(data, output_path=None):
    """
    Create a bar plot of final misinformation count values for each recommender type.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    # Get the final step for each run
    final_steps = data.groupby(["RunId", "iteration", "recommender_type"])["Step"].max().reset_index()
    final_data = pd.merge(data, final_steps, 
                         on=["RunId", "iteration", "recommender_type", "Step"])

    # Calculate average final misinformation count for each recommender type
    misinfo_summary = final_data.groupby("recommender_type")["Misinformation_Count_In_Recommendations"].agg(
        ["mean", "std"]).reset_index()

    # Sort by mean misinformation count
    misinfo_summary = misinfo_summary.sort_values("mean")

    # Create a list of colors for each recommender type
    bar_colors = [RECOMMENDER_COLORS.get(rec_type, '#999999') for rec_type in misinfo_summary["recommender_type"]]

    # Create bar plot
    plt.figure(figsize=(14, 8))
    bars = plt.bar(misinfo_summary["recommender_type"], misinfo_summary["mean"], 
            yerr=misinfo_summary["std"], capsize=10, 
            color=bar_colors)  # Use the list of colors

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                 height + 0.1,
                 f'{height:.2f}', 
                 ha='center', va='bottom')

    plt.title("Average Fake News Items in Recommendations by Recommender Type", fontsize=16)
    plt.ylabel("Number of Fake News Items", fontsize=14)
    plt.xlabel("Recommender Type", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    return plt.gcf(), misinfo_summary

def plot_echo_chamber_effect(data, output_path=None):
    """
    Plot echo chamber effect over time for each recommender type.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, x="Step", y="Echo_Chamber_Effect", 
                 hue="recommender_type", errorbar="sd", palette=RECOMMENDER_COLORS)
    plt.title("Echo Chamber Effect by Recommender Type")
    plt.ylabel("Echo Chamber Index")
    
    if output_path:
        plt.savefig(output_path)
    
    return plt.gcf()

def plot_recommender_summary(data, output_path=None):
    """
    Create a summary plot comparing recommenders across multiple metrics,
    showing the average values over the entire simulation period.
    
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
    
    metric_labels = ["Misinformation Spread (%)", 
                    "Misinformation Ratio Difference", 
                    "Fake News in Recommendations",
                    "Echo Chamber Effect"]
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    
    # Get unique recommender types
    recommender_types = data["recommender_type"].unique()
    
    # Process each metric
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # Calculate mean and std for the metric by recommender type across all steps
        metric_data = data.groupby("recommender_type")[metric].agg(
            ["mean", "std"]).reset_index()
        metric_data.columns = ["recommender_type", "value", "std"]
        
        # Sort by value
        metric_data = metric_data.sort_values("value")
        
        # Create horizontal bar chart
        bar_colors = [RECOMMENDER_COLORS.get(r, '#999999') for r in metric_data["recommender_type"]]
        bars = axes[i].barh(metric_data["recommender_type"], metric_data["value"], 
                xerr=metric_data["std"], capsize=3, color=bar_colors)
        
        # Add value labels with improved visibility
        for bar in bars:
            width = bar.get_width()
            # Position the label at the end of the bar
            label_x = width + (0.01 * max(metric_data["value"]))

            label_y = bar.get_y() - 0.04
            # Add a white background to the text for better visibility
            axes[i].text(label_x, label_y, 
                        f'{width:.3f}', va='center', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Add a vertical line at x=0 for metrics where it makes sense
        if metric == "Misinformation_Ratio_Difference":
            axes[i].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Set title and labels
        axes[i].set_title(f"Average {label} by Recommender Type (Across All Steps)", fontsize=14)
        axes[i].set_xlabel(label, fontsize=12)
        axes[i].set_ylabel("")
        
        # Add grid
        axes[i].grid(True, linestyle='--', alpha=0.7, axis='x')
        
        # Adjust x-axis limits to make room for the labels
        x_max = max(metric_data["value"] + metric_data["std"]) * 1.2
        axes[i].set_xlim(right=x_max)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def generate_comparison_dashboard(data, output_path=None):
    """
    Generate a dashboard with multiple metrics for comparing recommender algorithms.
    
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
                 hue="recommender_type", errorbar="sd", ax=axes[0, 0],
                 palette=RECOMMENDER_COLORS)
    axes[0, 0].set_title("Misinformation Infection by Recommender Type", fontsize=14)
    axes[0, 0].set_ylabel("Percentage of Population Infected", fontsize=12)
    
    # Plot 2: Misinformation Ratio Difference
    sns.lineplot(data=data, x="Step", y="Misinformation_Ratio_Difference", 
                 hue="recommender_type", errorbar="sd", ax=axes[0, 1],
                 palette=RECOMMENDER_COLORS)
    axes[0, 1].set_title("Misinformation Ratio Difference", fontsize=14)
    axes[0, 1].set_ylabel("MRD (positive = amplifying misinfo)", fontsize=12)
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Plot 3: Misinformation Count in Recommendations
    sns.lineplot(data=data, x="Step", y="Misinformation_Count_In_Recommendations", 
                 hue="recommender_type", errorbar="sd", ax=axes[1, 0],
                 palette=RECOMMENDER_COLORS)
    axes[1, 0].set_title("Fake News Items in Recommendations", fontsize=14)
    axes[1, 0].set_ylabel("Average Number of Fake News Items", fontsize=12)
    
    # Plot 4: Echo Chamber Effect
    sns.lineplot(data=data, x="Step", y="Echo_Chamber_Effect", 
                 hue="recommender_type", errorbar="sd", ax=axes[1, 1],
                 palette=RECOMMENDER_COLORS)
    axes[1, 1].set_title("Echo Chamber Effect", fontsize=14)
    axes[1, 1].set_ylabel("Echo Chamber Effect", fontsize=12)
    
    # Adjust layout and add a main title
    plt.suptitle("Recommender Algorithm Comparison Dashboard", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
    
    # Create a single legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flat:
        ax.get_legend().remove()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0), fontsize=12)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_recommender_ranking_table(data, output_path=None):
    """
    Create a table ranking recommenders by different metrics,
    using average values across all simulation steps.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the table. If None, the table is not saved.
    """
    # Define metrics and whether lower is better
    metrics = {
        "Misinformation_Spread_Percentage": {"label": "Misinfo Spread", "lower_better": True},
        "Misinformation_Ratio_Difference": {"label": "MRD", "lower_better": True},
        "Misinformation_Count_In_Recommendations": {"label": "Fake News Count", "lower_better": True},
        "Echo_Chamber_Effect": {"label": "Echo Chamber", "lower_better": True}
    }
    
    # Calculate average for each metric and recommender across all steps
    summary = {}
    for metric, info in metrics.items():
        # Group by recommender type and calculate mean across all steps
        metric_summary = data.groupby("recommender_type")[metric].mean().reset_index()
        
        # Sort based on whether lower is better
        metric_summary = metric_summary.sort_values(metric, ascending=info["lower_better"])
        
        # Assign ranks
        metric_summary["rank"] = range(1, len(metric_summary) + 1)
        
        # Store in summary dict
        summary[metric] = metric_summary
    
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(12, len(summary[list(metrics.keys())[0]]) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    recommender_types = summary[list(metrics.keys())[0]]["recommender_type"].tolist()
    
    # Header row
    header = ["Recommender Type"] + [info["label"] for info in metrics.values()]
    table_data.append(header)
    
    # Data rows
    for rec_type in recommender_types:
        row = [rec_type]
        for metric in metrics.keys():
            rank = summary[metric].loc[summary[metric]["recommender_type"] == rec_type, "rank"].values[0]
            value = summary[metric].loc[summary[metric]["recommender_type"] == rec_type, metric].values[0]
            row.append(f"#{rank} ({value:.3f})")
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                    loc='center', cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Color the cells based on rank
    for i in range(len(recommender_types)):
        for j in range(1, len(metrics) + 1):
            cell = table[i+1, j]
            rank = int(cell.get_text().get_text().split('#')[1].split(' ')[0])
            
            # Color gradient from green (rank 1) to red (last rank)
            color_val = rank / len(recommender_types)
            cell.set_facecolor((color_val, 1 - color_val, 0, 0.3))
    
    plt.title("Recommender Algorithm Rankings by Metric (Across All Steps)", fontsize=16, pad=20)
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
    
    # Generate bar plots
    plot_final_mrd_bar(data, os.path.join(output_dir, "final_mrd_bar.png"))
    plot_final_misinfo_count_bar(data, os.path.join(output_dir, "final_misinfo_count_bar.png"))
    
    # Generate summary plots
    plot_recommender_summary(data, os.path.join(output_dir, "recommender_summary.png"))
    
    # Generate ranking table
    create_recommender_ranking_table(data, os.path.join(output_dir, "recommender_ranking_table.png"))
    
    # Generate comparison dashboard
    generate_comparison_dashboard(data, os.path.join(output_dir, "recommender_comparison_dashboard.png"))
    
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


