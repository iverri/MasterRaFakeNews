import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import networkx as nx
import pickle

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
        axes[i].set_xlabel("Step", fontsize=14)
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
        axes[i].set_xlabel("Step", fontsize=14)
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
        axes[i].set_xlabel("Step", fontsize=14)
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
        axes[i].set_xlabel("Step", fontsize=14)
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
        "Misinformation_Spread_Percentage": "IR", 
        "Misinformation_Ratio_Difference": "MRD", 
        "Misinformation_Count_In_Recommendations": "MC",
        "Echo_Chamber_Effect": "EC"
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
        "Misinformation_Spread_Percentage": {"label": "IR", "lower_better": True},
        "Misinformation_Ratio_Difference": {"label": "MRD", "lower_better": True},
        "Misinformation_Count_In_Recommendations": {"label": "MC", "lower_better": True},
        "Echo_Chamber_Effect": {"label": "EC", "lower_better": True},
        "Average_Diversity_Score": {"label": "DS", "lower_better": False},
    }
    
    # Get a consistent order of recommender types across all diversity settings
    all_recommender_types = sorted(data["recommender_type"].unique())
    
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
        fig, ax = plt.subplots(figsize=(10, len(all_recommender_types) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        
        # Header row
        header = ["Rec Type"] + [info["label"] for metric, info in metrics.items() if metric in summary]
        table_data.append(header)
        
        # Data rows - use the consistent order of recommender types
        for rec_type in all_recommender_types:
            if rec_type not in filtered_data["recommender_type"].unique():
                continue  # Skip if this recommender type isn't in this diversity setting
                
            row = [rec_type]
            for metric in metrics.keys():
                if metric in summary:
                    # Find the rank and value for this recommender type
                    rec_data = summary[metric][summary[metric]["recommender_type"] == rec_type]
                    if len(rec_data) > 0:
                        rank = rec_data["rank"].values[0]
                        value = rec_data[metric].values[0]
                        row.append(f"#{rank} ({value:.3f})")
                    else:
                        row.append("N/A")
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
        for i in range(len(table_data) - 1):  # -1 to exclude header
            for j in range(1, len(header)):
                cell = table[i+1, j]
                cell_text = cell.get_text().get_text()
                
                # Skip cells with N/A
                if cell_text == "N/A":
                    continue
                    
                rank = int(cell_text.split('#')[1].split(' ')[0])
                
                # Color gradient from green (rank 1) to red (last rank)
                color_val = rank / len(filtered_data["recommender_type"].unique())
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
        "Misinformation_Spread_Percentage": "IR",
        "Misinformation_Ratio_Difference": "MRD",
        "Misinformation_Count_In_Recommendations": "MC",
        "Echo_Chamber_Effect": "EC"
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



def plot_community_metrics_table_by_recommender(data, community_data_file, output_path=None):
    """
    Create table visualizations showing metrics for each community, organized by recommender type.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    community_data_file : str
        Path to the pickle file containing community data
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load community data
    with open(community_data_file, 'rb') as f:
        community_data_by_run = pickle.load(f)
    
    # Organize data by recommender type
    community_data_by_recommender = organize_community_data_by_recommender(community_data_file)
    
    if not community_data_by_recommender:
        print("No community data available by recommender type")
        return None
    
    # Get unique recommender types
    recommender_types = sorted(community_data_by_recommender.keys())
    
    # Create a figure with subplots - one per recommender type
    # Reduce the height per recommender and use tighter spacing
    fig, axes = plt.subplots(len(recommender_types), 1, 
                            figsize=(10, 3 * len(recommender_types)),
                            gridspec_kw={'hspace': 0.4})
    
    # If only one recommender type, make axes iterable
    if len(recommender_types) == 1:
        axes = [axes]
    
    # Process each recommender type
    for i, rec_type in enumerate(recommender_types):
        # Get community data for this recommender
        community_data = community_data_by_recommender[rec_type]
        
        if not community_data:
            print(f"No community data available for {rec_type}")
            continue
        
        # Extract data
        communities = community_data['communities']
        fake_ratio = community_data['fake_ratio']
        sizes = community_data['sizes']
        within_sims = community_data['within_sims']
        
        # Use the calculated echo chamber scores if available, otherwise calculate them
        if 'echo_scores' in community_data:
            echo_chamber_scores = community_data['echo_scores']
        else:
            # Fall back to the old calculation method
            echo_chamber_scores = {}
            for comm_id in sorted(set(communities.values())):
                if comm_id in within_sims and comm_id in fake_ratio:
                    # Normalize both metrics to 0-1 range and combine them
                    echo_chamber_scores[comm_id] = (within_sims.get(comm_id, 0) + fake_ratio.get(comm_id, 0)) / 2
        
        # Create a list of unique community IDs
        community_ids = sorted(set(communities.values()))
        
        # Prepare data for the table
        table_data = []
        
        # Header row
        header = ["Community", "Size", "Misinfo Ratio", "Cluster sim", "EC"]
        table_data.append(header)
        
        # Data rows - use shorter format for community ID
        for comm_id in community_ids:
            row = [
                f"{comm_id}",  # Shorter community ID format
                sizes.get(comm_id, 0),
                f"{fake_ratio.get(comm_id, 0):.3f}",
                f"{within_sims.get(comm_id, 0):.3f}",
                f"{echo_chamber_scores.get(comm_id, 0):.3f}"
            ]
            table_data.append(row)
        
        # Set up the subplot
        ax = axes[i]
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                        loc='center', cellLoc='center')
        
        # Style the table - reduce font size and scale for more compact appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.2)
        
        # Set column widths - make first two columns narrower
        table.auto_set_column_width([0, 1, 2, 3, 4])
        
        # Manually adjust column widths - make first two columns narrower
        for j in range(len(table_data[0])):
            if j < 2:  # First two columns (Community ID and Size)
                for k in range(len(table_data)):
                    cell = table[k, j]
                    cell.set_width(0.1)  # Narrower width for first two columns
            else:
                for k in range(len(table_data)):
                    cell = table[k, j]
                    cell.set_width(0.2)  # Normal width for other columns
        
        # Color the cells based on values
        for j in range(len(community_ids)):
            # Color fake news ratio cell (red = high fake news)
            fake_cell = table[j+1, 2]
            fake_val = float(fake_cell.get_text().get_text())
            fake_cell.set_facecolor((fake_val, 1 - fake_val, 0, 0.3))
            
            # Color within similarity cell (blue = high similarity)
            sim_cell = table[j+1, 3]
            sim_val = float(sim_cell.get_text().get_text())
            sim_cell.set_facecolor((0, 0, sim_val, 0.3))
            
            # Color echo chamber score cell (purple = high echo chamber)
            echo_cell = table[j+1, 4]
            echo_val = float(echo_cell.get_text().get_text())
            echo_cell.set_facecolor((echo_val, 0, echo_val, 0.3))
        
        ax.set_title(f"Community Metrics - {rec_type}", fontsize=14, pad=10)
    
    # Add a main title but with less padding
    plt.suptitle("Community Metrics Analysis by Recommender Type", fontsize=16, y=0.98)
    
    # Use tight layout with specific padding
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def organize_community_data_by_recommender(community_data_file):
    """
    Organize community data by recommender type from a single community data file.
    
    Parameters:
    -----------
    community_data_file : str
        Path to the pickle file containing community data
        
    Returns:
    --------
    dict
        Dictionary mapping recommender types to their final step community data
    """
    # Load community data
    with open(community_data_file, 'rb') as f:
        community_data_by_run = pickle.load(f)
    
    # Organize by recommender type
    data_by_recommender = {}
    
    # First, find the last step for each recommender type
    last_steps = {}
    for run_id, data in community_data_by_run.items():
        if 'recommender_type' not in data:
            continue
            
        rec_type = data['recommender_type']
        step = int(run_id.split('_')[-1])
        
        if rec_type not in last_steps or step > last_steps[rec_type]:
            last_steps[rec_type] = step
    
    # Then, get the data for the last step of each recommender type
    for run_id, data in community_data_by_run.items():
        if 'recommender_type' not in data:
            continue
            
        rec_type = data['recommender_type']
        step = int(run_id.split('_')[-1])
        
        if step == last_steps[rec_type]:
            data_by_recommender[rec_type] = data
    
    return data_by_recommender

def plot_community_metrics_by_recommender(data, community_data_file, output_path=None):
    """
    Create plots showing community metrics for each recommender type.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    community_data_file : str
        Path to the pickle file containing community data
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Organize data by recommender type
    community_data_by_recommender = organize_community_data_by_recommender(community_data_file)
    
    if not community_data_by_recommender:
        print("No community data available by recommender type")
        return None
    
    # Get unique recommender types
    recommender_types = sorted(community_data_by_recommender.keys())
    
    # Create a figure with subplots - one row per recommender type, three columns for metrics
    fig, axes = plt.subplots(len(recommender_types), 3, 
                            figsize=(18, 5 * len(recommender_types)),
                            constrained_layout=True)
    
    # If only one recommender type, make axes 2D
    if len(recommender_types) == 1:
        axes = np.array([axes])
    
    # Process each recommender type
    for i, rec_type in enumerate(recommender_types):
        # Get community data for this recommender
        community_data = community_data_by_recommender[rec_type]
        
        if not community_data:
            print(f"No community data available for {rec_type}")
            continue
        
        # Extract data
        communities = community_data['communities']
        fake_ratio = community_data['fake_ratio']
        sizes = community_data['sizes']
        within_sims = community_data['within_sims']
        echo_scores = community_data.get('echo_scores', {})
        
        # Create a list of unique community IDs
        community_ids = sorted(set(communities.values()))
        
        # Plot community sizes
        comm_sizes = [sizes.get(comm_id, 0) for comm_id in community_ids]
        axes[i, 0].bar(
            [f"Comm {comm_id}" for comm_id in community_ids],
            comm_sizes,
            color='skyblue'
        )
        axes[i, 0].set_title(f"Community Sizes - {rec_type}", fontsize=12)
        axes[i, 0].set_ylabel("Number of Agents", fontsize=10)
        axes[i, 0].tick_params(axis='x', rotation=45)
        axes[i, 0].grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Plot fake news ratio
        fake_ratios = [fake_ratio.get(comm_id, 0) for comm_id in community_ids]
        bars = axes[i, 1].bar(
            [f"Comm {comm_id}" for comm_id in community_ids],
            fake_ratios,
            color='salmon'
        )
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[i, 1].text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=9
            )
        axes[i, 1].set_title(f"Fake News Ratio - {rec_type}", fontsize=12)
        axes[i, 1].set_ylabel("Fake News Ratio", fontsize=10)
        axes[i, 1].tick_params(axis='x', rotation=45)
        axes[i, 1].grid(True, linestyle='--', alpha=0.7, axis='y')
        axes[i, 1].set_ylim(0, 1.1)
        
        # Plot echo chamber scores
        if echo_scores:
            echo_values = [echo_scores.get(comm_id, 0) for comm_id in community_ids]
        else:
            # Calculate echo scores from within similarity and fake ratio if not available
            echo_values = [(within_sims.get(comm_id, 0) + fake_ratio.get(comm_id, 0)) / 2 
                          for comm_id in community_ids]
        
        bars = axes[i, 2].bar(
            [f"Comm {comm_id}" for comm_id in community_ids],
            echo_values,
            color='lightgreen'
        )
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[i, 2].text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=9
            )
        axes[i, 2].set_title(f"Echo Chamber Score - {rec_type}", fontsize=12)
        axes[i, 2].set_ylabel("Echo Chamber Score", fontsize=10)
        axes[i, 2].tick_params(axis='x', rotation=45)
        axes[i, 2].grid(True, linestyle='--', alpha=0.7, axis='y')
        axes[i, 2].set_ylim(0, 1.1)
    
    plt.suptitle("Community Metrics by Recommender Type", fontsize=16, y=1.02)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_diversity_impact_table(data, community_data_file, output_path=None):
    """
    Create table visualizations showing how different diversity levels affect metrics
    for each recommender type.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    community_data_file : str
        Path to the pickle file containing community data
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Load community data
    with open(community_data_file, 'rb') as f:
        community_data_by_run = pickle.load(f)
    
    # Extract diversity levels and recommender types from the data
    diversity_levels = sorted(data['diversity_level'].unique())
    recommender_types = sorted(data['recommender_type'].unique())
    
    # Create a figure for the table
    fig = plt.figure(figsize=(12, len(recommender_types) * 5))
    
    # Prepare metrics to display
    metrics = ["Misinfo Ratio", "Cluster sim", "EC"]
    
    # Create a subplot for each recommender type
    for i, rec_type in enumerate(recommender_types):
        ax = fig.add_subplot(len(recommender_types), 1, i+1)
        ax.axis('tight')
        ax.axis('off')
        
        # Get the last step data for this recommender type
        last_step_data = {}
        for run_id, run_data in community_data_by_run.items():
            if ('recommender_type' in run_data and 
                run_data['recommender_type'] == rec_type):
                
                # Extract step and run key
                step = int(run_id.split('_')[-1])
                run_key = '_'.join(run_id.split('_')[:-1])
                
                # Store the highest step for each run key
                if run_key not in last_step_data or step > last_step_data[run_key]['step']:
                    last_step_data[run_key] = {
                        'step': step,
                        'data': run_data,
                        'diversity_level': run_data.get('diversity_level', 0)
                    }
        
        # Group data by diversity level
        data_by_diversity = {}
        for run_info in last_step_data.values():
            div_level = run_info['diversity_level']
            if div_level not in data_by_diversity:
                data_by_diversity[div_level] = []
            data_by_diversity[div_level].append(run_info['data'])
        
        # Get all unique community IDs across all diversity levels
        all_communities = set()
        for div_data_list in data_by_diversity.values():
            for div_data in div_data_list:
                if 'communities' in div_data:
                    all_communities.update(set(div_data['communities'].values()))
        
        # Sort communities
        all_communities = sorted(all_communities)
        
        # Prepare table data
        table_data = []
        
        # Create data rows for each community
        for comm_id in all_communities:
            row = [f"{comm_id}"]
            
            # Get average size across all diversity levels
            sizes = []
            for div_level, div_data_list in data_by_diversity.items():
                for div_data in div_data_list:
                    if 'sizes' in div_data and comm_id in div_data['sizes']:
                        sizes.append(div_data['sizes'][comm_id])
            
            # Add average size to row
            row.append(f"{int(np.mean(sizes)) if sizes else 0}")
            
            # For each metric and diversity level, get the value
            for metric_name in ["fake_ratio", "within_sims", "echo_scores"]:
                for div_level in diversity_levels:
                    if div_level in data_by_diversity:
                        # Get values for this community across all runs with this diversity level
                        values = []
                        for div_data in data_by_diversity[div_level]:
                            # For echo_scores, calculate if not present
                            if metric_name == "echo_scores" and "echo_scores" not in div_data:
                                if ('fake_ratio' in div_data and 
                                    'within_sims' in div_data and 
                                    comm_id in div_data.get('fake_ratio', {}) and 
                                    comm_id in div_data.get('within_sims', {})):
                                    
                                    echo_score = (div_data['fake_ratio'][comm_id] + 
                                                 div_data['within_sims'][comm_id]) / 2
                                    values.append(echo_score)
                            elif metric_name in div_data and comm_id in div_data[metric_name]:
                                values.append(div_data[metric_name][comm_id])
                        
                        # Calculate average value
                        if values:
                            avg_value = np.mean(values)
                            row.append(f"{avg_value:.3f}")
                        else:
                            row.append("N/A")
                    else:
                        row.append("N/A")
            
            table_data.append(row)
        
        # Create column labels
        col_labels = ["ID", "Size"]
        for metric in metrics:
            for level in diversity_levels:
                col_labels.append(f"{level}")
        
        # Create table
        if len(table_data) > 0:  # Only create table if we have data rows
            table = ax.table(
                cellText=table_data,
                colLabels=col_labels,
                loc='center',
                cellLoc='center'
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Fix column widths
            for col in range(len(col_labels)):
                if col < 2:  # Community and Size columns
                    for row in range(len(table_data) + 1):  # +1 for header row
                        cell = table[row, col]
                        cell.set_width(0.06 if col == 0 else 0.04)
            
            # Color the cells based on values
            for j in range(len(table_data)):  # Row index (communities)
                for k in range(2, len(col_labels)):  # Column index (metrics at different diversity levels)
                    cell = table[j+1, k]  # +1 to account for header row
                    cell_text = cell.get_text().get_text()
                    
                    # Skip cells with N/A
                    if cell_text == "N/A":
                        continue
                    
                    try:
                        value = float(cell_text)
                        
                        # Determine which metric this column represents
                        metric_idx = (k - 2) // len(diversity_levels)
                        
                        # Color based on metric type
                        if metric_idx == 0:  # Misinfo Ratio (red = high)
                            cell.set_facecolor((value, 1 - value, 0, 0.3))
                        elif metric_idx == 1:  # Cluster sim (blue = high)
                            cell.set_facecolor((0, 0, value, 0.3))
                        elif metric_idx == 2:  # Echo Chamber (purple = high)
                            cell.set_facecolor((value, 0, value, 0.3))
                    except ValueError:
                        # Skip cells that can't be converted to float
                        continue
            
            # Add "Div" prefix to diversity level headers
            for col in range(2, len(col_labels)):
                cell = table[0, col]
                current_text = cell.get_text().get_text()
                cell.get_text().set_text(f"Div {current_text}")
            
            ax.set_title(f"Community Metrics - {rec_type}", fontsize=14, pad=20)
    
    plt.suptitle("Community Metrics Analysis by Recommender Type", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def generate_all_plots(csv_path, output_dir=None, community_data_file=None):
    """
    Generate all plots for the given experiment results.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing experiment results
    output_dir : str, optional
        Directory to save the plots. If None, plots are saved in the current directory.
    community_data_file : str, optional
        Path to the pickle file containing community data
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
    
    # Generate community-specific plots if community data is available
    if community_data_file:
        # Generate recommender-specific community plots
        plot_community_metrics_table_by_recommender(data, community_data_file, 
                                                  os.path.join(output_dir, "community_metrics_table_by_recommender.png"))
        plot_community_metrics_by_recommender(data, community_data_file, 
                                            os.path.join(output_dir, "community_metrics_by_recommender.png"))
        plot_diversity_impact_table(data, community_data_file, 
                                    os.path.join(output_dir, "diversity_impact_table.png"))
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate plots from experiment data')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing experiment results')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save the plots')
    parser.add_argument('--community-data-file', type=str, default=None, help='Path to the pickle file containing community data')
    
    args = parser.parse_args()
    
    generate_all_plots(args.csv_file, args.output_dir, args.community_data_file)


