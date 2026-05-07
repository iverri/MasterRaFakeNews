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
import matplotlib.ticker as mtick

# Define a global color mapping for all recommender types using label names
RECOMMENDER_COLORS = {
    "Rnd": "#1f77b4",  # blue
    "Pop": "#e377c2",  # pink
    "CBF": "#2ca02c",  # green
    "UB-CF": "#ff7f0e",  # orange
    "IB-CF": "#d62728",  # red
    "HW_D": "#9467bd",  # purple
    "HW_S": "#17becf",  # cyan
    "MF": "#8c564b",  # brown
    "Mix": "#0b2843",  # dark blue
    "Feat": "#F6FF00",  # dark blue
}

# Define a global label mapping for recommender types
RECOMMENDER_LABELS = {
    "random": "Rnd",
    "popular": "Pop",
    "content_based": "CBF",
    "user_knn": "UB-CF",
    "item_knn": "IB-CF",
    "hybrid_weighted_dynamic": "HW_D",
    "hybrid_weighted_static": "HW_S",
    "matrix_factorization": "MF",
    "mixed": "Mix",
    "feature_combination": "Feat",
}


# Helper function to map recommender types to their display labels
def get_recommender_label(recommender_type):
    """
    Map a recommender type to its display label.

    Parameters:
    -----------
    recommender_type : str
        The original recommender type name

    Returns:
    --------
    str
        The display label for the recommender type
    """
    return RECOMMENDER_LABELS.get(recommender_type, recommender_type)


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
    if "diversity_level" in data.columns:
        # Create a readable diversity setting label
        data["diversity_setting"] = data["diversity_level"].apply(
            lambda x: "No Diversity" if x == 0 else f"Diversity {x}"
        )
    else:
        # If the column doesn't exist, assume all data is without diversity
        data["diversity_setting"] = "No Diversity"
        data["diversity_level"] = 0

    return data


def has_columns(data, required_cols):
    return all(col in data.columns for col in required_cols)


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
    diversity_settings = sorted(data["diversity_setting"].unique())

    # Create a figure with subplots for each diversity setting
    fig, axes = plt.subplots(
        1,
        len(diversity_settings),
        figsize=(6 * len(diversity_settings), 8),
        sharey=True,
    )

    # Handle case with only one diversity setting
    if len(diversity_settings) == 1:
        axes = [axes]

    # Create a copy of the data with mapped labels
    plot_data = data.copy()
    plot_data["recommender_label"] = plot_data["recommender_type"].apply(
        get_recommender_label
    )

    # Plot for each diversity setting
    for i, diversity in enumerate(diversity_settings):
        # Filter data for this diversity setting
        filtered_data = plot_data[plot_data["diversity_setting"] == diversity]

        # Plot data using the label column instead of recommender_type
        sns.lineplot(
            data=filtered_data,
            x="Step",
            y="Misinformation_Spread_Percentage",
            hue="recommender_label",
            errorbar="sd",
            palette=RECOMMENDER_COLORS,
            linewidth=2.5,
            ax=axes[i],
        )

        axes[i].set_title(f"{diversity}", fontsize=16)
        axes[i].set_xlabel("Step", fontsize=14)
        axes[i].grid(True, linestyle="--", alpha=0.7)

        # Only add y-label to the first subplot
        if i == 0:
            axes[i].set_ylabel("% Misinformation Spread (I + E)", fontsize=14)
        else:
            axes[i].set_ylabel("", fontsize=14)

    # Add a main title
    plt.suptitle(
        "Misinformation Spread by Recommender Type (I + E)", fontsize=18, y=1.05
    )

    # Adjust legend for better readability
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc="upper center",
        ncol=len(labels),
        fontsize=12,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_infection_rate(data, output_path=None):
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
    diversity_settings = sorted(data["diversity_setting"].unique())

    # Create a figure with subplots for each diversity setting
    fig, axes = plt.subplots(
        1,
        len(diversity_settings),
        figsize=(6 * len(diversity_settings), 8),
        sharey=True,
    )

    # Handle case with only one diversity setting
    if len(diversity_settings) == 1:
        axes = [axes]

    # Create a copy of the data with mapped labels
    plot_data = data.copy()
    plot_data["recommender_label"] = plot_data["recommender_type"].apply(
        get_recommender_label
    )

    # Plot for each diversity setting
    for i, diversity in enumerate(diversity_settings):
        # Filter data for this diversity setting
        filtered_data = plot_data[plot_data["diversity_setting"] == diversity]

        # Plot data using the label column instead of recommender_type
        sns.lineplot(
            data=filtered_data,
            x="Step",
            y="Infection_Rate",
            hue="recommender_label",
            errorbar="sd",
            palette=RECOMMENDER_COLORS,
            linewidth=2.5,
            ax=axes[i],
        )

        axes[i].set_title(f"{diversity}", fontsize=16)
        axes[i].set_xlabel("Step", fontsize=14)
        axes[i].grid(True, linestyle="--", alpha=0.7)

        # Only add y-label to the first subplot
        if i == 0:
            axes[i].set_ylabel("% Infected", fontsize=14)
        else:
            axes[i].set_ylabel("", fontsize=14)

    # Add a main title
    plt.suptitle("Infection Rate by Recommender Type (I)", fontsize=18, y=1.05)

    # Adjust legend for better readability
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc="upper center",
        ncol=len(labels),
        fontsize=12,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

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
    diversity_settings = sorted(data["diversity_setting"].unique())

    # Create a figure with subplots for each diversity setting
    fig, axes = plt.subplots(
        1,
        len(diversity_settings),
        figsize=(6 * len(diversity_settings), 8),
        sharey=True,
    )

    # Handle case with only one diversity setting
    if len(diversity_settings) == 1:
        axes = [axes]

    # Create a copy of the data with mapped labels
    plot_data = data.copy()
    plot_data["recommender_label"] = plot_data["recommender_type"].apply(
        get_recommender_label
    )

    # Plot for each diversity setting
    for i, diversity in enumerate(diversity_settings):
        # Filter data for this diversity setting
        filtered_data = plot_data[plot_data["diversity_setting"] == diversity]

        # Plot data
        sns.lineplot(
            data=filtered_data,
            x="Step",
            y="Misinformation_Ratio_Difference",
            hue="recommender_label",
            errorbar="sd",
            palette=RECOMMENDER_COLORS,
            linewidth=2.5,
            ax=axes[i],
        )

        axes[i].set_title(f"{diversity}", fontsize=16)
        axes[i].set_xlabel("Step", fontsize=14)
        axes[i].grid(True, linestyle="--", alpha=0.7)
        axes[i].axhline(y=0, color="gray", linestyle="--", alpha=0.7)

        # Only add y-label to the first subplot
        if i == 0:
            axes[i].set_ylabel(
                "MRD (positive = amplifying misinformation)", fontsize=14
            )
        else:
            axes[i].set_ylabel("", fontsize=14)

    # Add a main title
    plt.suptitle(
        "Misinformation Ratio Difference by Recommender Type", fontsize=18, y=1.05
    )

    # Adjust legend for better readability
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc="upper center",
        ncol=len(labels),
        fontsize=12,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

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
    diversity_settings = sorted(data["diversity_setting"].unique())

    # Create a figure with subplots for each diversity setting
    fig, axes = plt.subplots(
        1,
        len(diversity_settings),
        figsize=(6 * len(diversity_settings), 8),
        sharey=True,
    )

    # Handle case with only one diversity setting
    if len(diversity_settings) == 1:
        axes = [axes]

    # Create a copy of the data with mapped labels
    plot_data = data.copy()
    plot_data["recommender_label"] = plot_data["recommender_type"].apply(
        get_recommender_label
    )

    # Plot for each diversity setting
    for i, diversity in enumerate(diversity_settings):
        # Filter data for this diversity setting
        filtered_data = plot_data[plot_data["diversity_setting"] == diversity]

        # Plot data
        sns.lineplot(
            data=filtered_data,
            x="Step",
            y="Misinformation_Count_In_Recommendations",
            hue="recommender_label",
            errorbar="sd",
            palette=RECOMMENDER_COLORS,
            linewidth=2.5,
            ax=axes[i],
        )

        axes[i].set_title(f"{diversity}", fontsize=16)
        axes[i].set_xlabel("Step", fontsize=14)
        axes[i].grid(True, linestyle="--", alpha=0.7)

        # Only add y-label to the first subplot
        if i == 0:
            axes[i].set_ylabel("Average Number of Misinformation Items", fontsize=14)
        else:
            axes[i].set_ylabel("", fontsize=14)

    # Add a main title
    plt.suptitle(
        "Average Misinformation Count in Recommendations by Recommender Type",
        fontsize=18,
        y=1.05,
    )

    # Adjust legend for better readability
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc="upper center",
        ncol=len(labels),
        fontsize=12,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_echo_chamber_effect(data, output_path=None, skip_steps=5):
    """
    Plot echo chamber effect over time for each recommender type,
    with separate plots for different diversity levels.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    skip_steps : int, optional
        Number of initial steps to skip in the plot
    """
    # Get unique diversity settings
    diversity_settings = sorted(data["diversity_setting"].unique())

    # Filter out the first few steps
    filtered_data = data[data["Step"] > skip_steps]

    # Create a copy of the data with mapped labels
    plot_data = filtered_data.copy()
    plot_data["recommender_label"] = plot_data["recommender_type"].apply(
        get_recommender_label
    )

    # Create a figure with subplots for each diversity setting
    fig, axes = plt.subplots(
        1,
        len(diversity_settings),
        figsize=(6 * len(diversity_settings), 8),
        sharey=True,
    )

    # Handle case with only one diversity setting
    if len(diversity_settings) == 1:
        axes = [axes]

    # Plot for each diversity setting
    for i, diversity in enumerate(diversity_settings):
        # Filter data for this diversity setting
        setting_filtered_data = plot_data[plot_data["diversity_setting"] == diversity]

        # Plot data
        sns.lineplot(
            data=setting_filtered_data,
            x="Step",
            y="Echo_Chamber_Effect",
            hue="recommender_label",
            errorbar="sd",
            palette=RECOMMENDER_COLORS,
            linewidth=2.5,
            ax=axes[i],
        )

        axes[i].set_title(f"{diversity}", fontsize=16)
        axes[i].set_xlabel("Step", fontsize=14)
        axes[i].grid(True, linestyle="--", alpha=0.7)

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
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc="upper center",
        ncol=len(labels),
        fontsize=12,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_recommender_summary(data, output_path=None, skip_steps=5):
    """
    Create separate summary plots for each metric, with subplots for each diversity level,
    similar to the timeline plot layout.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plots. If None, the plots are not saved.
    skip_steps : int, optional
        Number of initial steps to skip for Echo_Chamber_Effect metric
    """
    # Define metrics to plot
    metrics = {
        "Misinformation_Spread_Percentage": "IR",
        "Misinformation_Ratio_Difference": "MRD",
        "Misinformation_Count_In_Recommendations": "MC",
        "Echo_Chamber_Effect": "EC",
    }

    # Create a copy of the data with mapped labels
    plot_data = data.copy()
    plot_data["recommender_label"] = plot_data["recommender_type"].apply(
        get_recommender_label
    )

    # Get unique diversity settings and recommender types
    diversity_settings = sorted(plot_data["diversity_setting"].unique())

    # Create a figure for each metric
    for metric, label in metrics.items():
        # For Echo_Chamber_Effect, filter out the first few steps
        metric_data = plot_data.copy()
        if metric == "Echo_Chamber_Effect":
            metric_data = metric_data[metric_data["Step"] > skip_steps]

        # Create a figure with subplots for each diversity setting
        fig, axes = plt.subplots(
            1, len(diversity_settings), figsize=(16, 6), sharey=True
        )

        # Handle case with only one diversity setting
        if len(diversity_settings) == 1:
            axes = [axes]

        # Calculate average values for each recommender and diversity setting
        avg_data = (
            metric_data.groupby(
                ["recommender_type", "recommender_label", "diversity_setting"]
            )[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        avg_data.columns = [
            "recommender_type",
            "recommender_label",
            "diversity_setting",
            "mean",
            "std",
        ]

        # Plot for each diversity setting
        for i, diversity in enumerate(diversity_settings):
            # Filter data for this diversity setting
            setting_data = avg_data[avg_data["diversity_setting"] == diversity]

            # Sort by mean value for better visualization
            setting_data = setting_data.sort_values("mean")

            # Create vertical bar chart - use recommender_label for color mapping
            bars = axes[i].bar(
                setting_data["recommender_label"],
                setting_data["mean"],
                yerr=setting_data["std"],
                capsize=5,
                color=[
                    RECOMMENDER_COLORS[label]
                    for label in setting_data["recommender_label"]
                ],
                alpha=0.8,
            )

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    height + (0.01 * avg_data["mean"].max()),
                    f"{height:.3f}",
                    ha="center",
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                )

            # Add a horizontal line at y=0 for metrics where it makes sense
            if metric == "Misinformation_Ratio_Difference":
                axes[i].axhline(y=0, color="gray", linestyle="--", alpha=0.7)

            axes[i].set_title(f"{diversity}", fontsize=14)
            axes[i].set_ylabel(label, fontsize=12)
            axes[i].grid(True, linestyle="--", alpha=0.7, axis="y")

            # Rotate x-axis labels for better readability
            axes[i].tick_params(axis="x", rotation=45)

            # Only add x-label to the first subplot
            if i == 0:
                axes[i].set_xlabel("Recommender Type", fontsize=12)

        # Add a main title
        plt.suptitle(f"{label} by Recommender Type", fontsize=16, y=1.02)

        plt.tight_layout()

        # Save the plot if output path is provided
        if output_path:
            metric_name = metric.lower().replace("_", "")
            file_path = output_path.replace(".png", f"_{metric_name}.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")

    # Return the last created figure
    return fig


def create_recommender_ranking_table(data, output_path=None, skip_steps=5):
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
    skip_steps : int, optional
        Number of initial steps to skip for Echo_Chamber_Effect metric
    """
    # Define metrics and whether lower is better
    metrics = {
        "Misinformation_Spread_Percentage": {"label": "IR", "lower_better": True},
        "Misinformation_Ratio_Difference": {"label": "MRD", "lower_better": True},
        "Misinformation_Count_In_Recommendations": {
            "label": "MC",
            "lower_better": True,
        },
        "Echo_Chamber_Effect": {"label": "EC", "lower_better": True},
        "Average_Diversity_Score": {"label": "DS", "lower_better": False},
    }

    # Create a copy of the data with mapped labels
    plot_data = data.copy()
    plot_data["recommender_label"] = plot_data["recommender_type"].apply(
        get_recommender_label
    )

    # Create separate tables for each diversity setting
    for diversity_setting in sorted(plot_data["diversity_setting"].unique()):
        # Filter data for the current diversity setting
        filtered_data = plot_data[plot_data["diversity_setting"] == diversity_setting]

        # For non-"No Diversity" settings, add the Diversity_Improvement_Percentage metric
        if (
            diversity_setting != "No Diversity"
            and "Diversity_Improvement_Percentage" in filtered_data.columns
        ):
            metrics["Diversity_Improvement_Percentage"] = {
                "label": "DI",
                "lower_better": False,
            }
        elif "Diversity_Improvement_Percentage" in metrics:
            # Remove the metric if we're on "No Diversity" setting
            del metrics["Diversity_Improvement_Percentage"]

        # Calculate average for each metric and recommender across all steps
        summary = {}
        for metric, info in metrics.items():
            if metric in filtered_data.columns:
                # For Echo_Chamber_Effect, filter out the first few steps
                metric_data = filtered_data.copy()
                if metric == "Echo_Chamber_Effect":
                    metric_data = metric_data[metric_data["Step"] > skip_steps]

                # Group by recommender type and calculate mean across all steps
                metric_summary = (
                    metric_data.groupby(["recommender_type", "recommender_label"])[
                        metric
                    ]
                    .mean()
                    .reset_index()
                )

                # Sort based on whether lower is better
                metric_summary = metric_summary.sort_values(
                    metric, ascending=info["lower_better"]
                )

                # Assign ranks
                metric_summary["rank"] = range(1, len(metric_summary) + 1)

                # Store in summary dict
                summary[metric] = metric_summary

        # Skip if no metrics were found
        if not summary:
            print(f"No metrics found for {diversity_setting}, skipping table creation.")
            continue

        # Get a consistent order of recommender types across all diversity settings
        all_recommender_types = sorted(filtered_data["recommender_type"].unique())

        # Create a figure for the table
        fig, ax = plt.subplots(figsize=(10, len(all_recommender_types) * 0.5 + 2))
        ax.axis("tight")
        ax.axis("off")

        # Prepare table data
        table_data = []

        # Header row
        header = ["Rec Type"] + [
            info["label"] for metric, info in metrics.items() if metric in summary
        ]
        table_data.append(header)

        # Data rows - use the consistent order of recommender types
        for rec_type in all_recommender_types:
            rec_label = get_recommender_label(rec_type)
            row = [rec_label]
            for metric in metrics.keys():
                if metric in summary:
                    # Find the rank and value for this recommender type
                    rec_data = summary[metric][
                        summary[metric]["recommender_type"] == rec_type
                    ]
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
        table = ax.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc="center",
            cellLoc="center",
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)

        # Color the cells based on rank
        for i in range(len(table_data) - 1):  # -1 to exclude header
            for j in range(1, len(header)):
                cell = table[i + 1, j]  # +1 to account for header row
                cell_text = cell.get_text().get_text()

                # Skip cells with N/A
                if cell_text == "N/A":
                    continue

                rank = int(cell_text.split("#")[1].split(" ")[0])

                # Color gradient from green (rank 1) to red (last rank)
                color_val = rank / len(filtered_data["recommender_type"].unique())
                cell.set_facecolor((color_val, 1 - color_val, 0, 0.3))

        plt.title(
            f"Recommender Algorithm Rankings by Metric ({diversity_setting})",
            fontsize=16,
            pad=20,
        )
        plt.tight_layout()

        if output_path:
            # Add diversity setting to the filename
            diversity_suffix = diversity_setting.replace(" ", "_").lower()
            file_path = output_path.replace(".png", f"_{diversity_suffix}.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")

    # Return the last created figure or None if no figures were created
    try:
        return fig
    except UnboundLocalError:
        print("No tables were created.")
        return None


def create_success_metrics_ranking_table(data, output_path=None):
    """
    Create a table ranking recommenders by avg Precision, avg Recall,
    and avg F1 using the model data file.

    The averages are computed from the final step of each run.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing model-level experiment results
    output_path : str, optional
        Path to save the table. If None, the table is not saved.
    """
    required_cols = [
        "RunId",
        "iteration",
        "Step",
        "recommender_type",
        "Precision",
        "Recall",
    ]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        print(f"Missing required columns for success metrics table: {missing}")
        return None

    df = data.copy()
    df["recommender_label"] = df["recommender_type"].apply(get_recommender_label)

    # Keep only final step of each run
    df["RunKey"] = df["RunId"].astype(str) + "_" + df["iteration"].astype(str)
    last_steps = df.groupby("RunKey")["Step"].transform("max")
    final_df = df[df["Step"] == last_steps].copy()

    # Compute F1 from Precision and Recall
    final_df["F1"] = np.where(
        (final_df["Precision"] + final_df["Recall"]) > 0,
        2
        * final_df["Precision"]
        * final_df["Recall"]
        / (final_df["Precision"] + final_df["Recall"]),
        0.0,
    )

    metrics = {
        "Precision": {"label": "Precision", "lower_better": False},
        "Recall": {"label": "Recall", "lower_better": False},
        "F1": {"label": "F1", "lower_better": False},
    }

    summary = {}
    for metric, info in metrics.items():
        metric_summary = (
            final_df.groupby(["recommender_type", "recommender_label"])[metric]
            .mean()
            .reset_index()
        )

        metric_summary = metric_summary.sort_values(
            metric, ascending=info["lower_better"]
        )
        metric_summary["rank"] = range(1, len(metric_summary) + 1)
        summary[metric] = metric_summary

    all_recommender_types = sorted(final_df["recommender_type"].unique())

    fig, ax = plt.subplots(figsize=(8, len(all_recommender_types) * 0.55 + 2.5))
    ax.axis("tight")
    ax.axis("off")

    table_data = []

    header = ["Rec Type"] + [info["label"] for metric, info in metrics.items()]
    table_data.append(header)

    for rec_type in all_recommender_types:
        rec_label = get_recommender_label(rec_type)
        row = [rec_label]

        for metric in metrics.keys():
            rec_data = summary[metric][summary[metric]["recommender_type"] == rec_type]
            if len(rec_data) > 0:
                rank = rec_data["rank"].values[0]
                value = rec_data[metric].values[0]
                row.append(f"#{rank} ({value:.3f})")
            else:
                row.append("N/A")

        table_data.append(row)

    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.8)

    n_recommenders = len(all_recommender_types)

    # Color cells by rank: green best -> red worst
    for i in range(len(table_data) - 1):
        for j in range(1, len(header)):
            cell = table[i + 1, j]
            cell_text = cell.get_text().get_text()

            if cell_text == "N/A":
                continue

            rank = int(cell_text.split("#")[1].split(" ")[0])
            color_val = rank / n_recommenders
            cell.set_facecolor((color_val, 1 - color_val, 0, 0.3))

    plt.title(
        "Recommender Algorithm Rankings by Success Metrics",
        fontsize=18,
        pad=20,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_diversity_impact_heatmap(data, output_path=None, skip_steps=5):
    """
    Create a heatmap showing how different diversity levels affect each recommender type
    for key metrics.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    skip_steps : int, optional
        Number of initial steps to skip for Echo_Chamber_Effect metric
    """
    # Define metrics to analyze
    metrics = {
        "Misinformation_Spread_Percentage": "IR",
        "Misinformation_Ratio_Difference": "MRD",
        "Misinformation_Count_In_Recommendations": "MC",
        "Echo_Chamber_Effect": "EC",
    }

    # Get unique recommender types and diversity levels
    recommender_types = sorted(data["recommender_type"].unique())
    diversity_levels = sorted(data["diversity_level"].unique())

    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    # Process each metric
    for i, (metric, label) in enumerate(metrics.items()):
        # For Echo_Chamber_Effect, filter out the first few steps
        metric_data = data.copy()
        if metric == "Echo_Chamber_Effect":
            metric_data = metric_data[metric_data["Step"] > skip_steps]

        # Calculate mean for the metric by recommender type and diversity level
        metric_summary = (
            metric_data.groupby(["recommender_type", "diversity_level"])[metric]
            .mean()
            .reset_index()
        )

        # Pivot the data for heatmap
        pivot_data = metric_summary.pivot(
            index="recommender_type", columns="diversity_level", values=metric
        )

        # Create heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r" if metric != "Average_Diversity_Score" else "RdYlGn",
            ax=axes[i],
            cbar_kws={"label": label},
        )

        axes[i].set_title(f"Impact of Diversity Level on {label}", fontsize=14)
        axes[i].set_xlabel("Diversity Level", fontsize=12)
        axes[i].set_ylabel("Recommender Type", fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_community_metrics_table_by_recommender(
    data, community_data_file, output_path=None
):
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
    with open(community_data_file, "rb") as f:
        community_data_by_run = pickle.load(f)

    # Organize data by recommender type
    community_data_by_recommender = organize_community_data_by_recommender(
        community_data_file
    )

    if not community_data_by_recommender:
        print("No community data available by recommender type")
        return None

    # Get unique recommender types
    recommender_types = sorted(community_data_by_recommender.keys())

    # Create a figure with subplots - one per recommender type
    # Reduce the height per recommender and use tighter spacing
    fig, axes = plt.subplots(
        len(recommender_types),
        1,
        figsize=(10, 4 * len(recommender_types)),
        gridspec_kw={"hspace": 0.4},
    )

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
        communities = community_data["communities"]
        fake_ratio = community_data["fake_ratio"]
        sizes = community_data["sizes"]

        # Use the calculated echo chamber scores if available, otherwise calculate them
        if "echo_scores" in community_data:
            echo_chamber_scores = community_data["echo_scores"]
        else:
            # Fall back to the old calculation method
            echo_chamber_scores = {}

        # Create a list of unique community IDs
        community_ids = sorted(set(communities.values()))

        # Prepare data for the table
        table_data = []

        # Header row
        header = ["Community", "Size", "Misinfo Ratio", "EC"]
        table_data.append(header)

        # Data rows - use shorter format for community ID
        for comm_id in community_ids:
            row = [
                f"{comm_id}",  # Shorter community ID format
                sizes.get(comm_id, 0),
                f"{fake_ratio.get(comm_id, 0):.3f}",
                f"{echo_chamber_scores.get(comm_id, 0):.3f}",
            ]
            table_data.append(row)

        # Set up the subplot
        ax = axes[i]
        ax.axis("tight")
        ax.axis("off")

        # Create table
        table = ax.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc="center",
            cellLoc="center",
        )

        # Style the table - reduce font size and scale for more compact appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.2)

        # Set column widths - make first two columns narrower
        table.auto_set_column_width([0, 1, 2, 3])

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
            fake_cell = table[j + 1, 2]
            fake_val = float(fake_cell.get_text().get_text())
            fake_cell.set_facecolor((fake_val, 1 - fake_val, 0, 0.3))

            # Color echo chamber score cell (purple = high echo chamber)
            echo_cell = table[j + 1, 3]  # Changed from index 4 to 3
            echo_val = float(echo_cell.get_text().get_text())
            echo_cell.set_facecolor((echo_val, 0, echo_val, 0.3))

        ax.set_title(f"Community Metrics - {rec_type}", fontsize=14, pad=10)

    # Add a main title but with less padding
    plt.suptitle("Community Metrics Analysis by Recommender Type", fontsize=16, y=0.98)

    # Use tight layout with specific padding
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

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
    with open(community_data_file, "rb") as f:
        community_data_by_run = pickle.load(f)

    # Organize by recommender type
    data_by_recommender = {}

    # First, find the last step for each recommender type
    last_steps = {}
    for run_id, data in community_data_by_run.items():
        if "recommender_type" not in data:
            continue

        rec_type = data["recommender_type"]
        step = int(run_id.split("_")[-1])

        if rec_type not in last_steps or step > last_steps[rec_type]:
            last_steps[rec_type] = step

    # Then, get the data for the last step of each recommender type
    for run_id, data in community_data_by_run.items():
        if "recommender_type" not in data:
            continue

        rec_type = data["recommender_type"]
        step = int(run_id.split("_")[-1])

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
    community_data_by_recommender = organize_community_data_by_recommender(
        community_data_file
    )

    if not community_data_by_recommender:
        print("No community data available by recommender type")
        return None

    # Get unique recommender types
    recommender_types = sorted(community_data_by_recommender.keys())

    # Create a figure with subplots - one row per recommender type, three columns for metrics
    fig, axes = plt.subplots(
        len(recommender_types),
        3,
        figsize=(18, 5 * len(recommender_types)),
        constrained_layout=True,
    )

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
        communities = community_data["communities"]
        fake_ratio = community_data["fake_ratio"]
        sizes = community_data["sizes"]
        echo_scores = community_data.get("echo_scores", {})

        # Create a list of unique community IDs
        community_ids = sorted(set(communities.values()))

        # Plot community sizes
        comm_sizes = [sizes.get(comm_id, 0) for comm_id in community_ids]
        axes[i, 0].bar(
            [f"Comm {comm_id}" for comm_id in community_ids],
            comm_sizes,
            color="skyblue",
        )
        axes[i, 0].set_title(f"Community Sizes - {rec_type}", fontsize=12)
        axes[i, 0].set_ylabel("Number of Agents", fontsize=10)
        axes[i, 0].tick_params(axis="x", rotation=45)
        axes[i, 0].grid(True, linestyle="--", alpha=0.7, axis="y")

        # Plot fake news ratio
        fake_ratios = [fake_ratio.get(comm_id, 0) for comm_id in community_ids]
        bars = axes[i, 1].bar(
            [f"Comm {comm_id}" for comm_id in community_ids],
            fake_ratios,
            color="salmon",
        )
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[i, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axes[i, 1].set_title(f"Fake News Ratio - {rec_type}", fontsize=12)
        axes[i, 1].set_ylabel("Fake News Ratio", fontsize=10)
        axes[i, 1].tick_params(axis="x", rotation=45)
        axes[i, 1].grid(True, linestyle="--", alpha=0.7, axis="y")
        axes[i, 1].set_ylim(0, 1.1)

        # Plot echo chamber scores
        if echo_scores:
            echo_values = [echo_scores.get(comm_id, 0) for comm_id in community_ids]
        else:
            # Calculate echo scores from within similarity and fake ratio if not available
            echo_values = None

        # Find the actual min and max values in the data
        min_val = min(echo_values) - 0.05  # Add a small padding below
        max_val = max(echo_values) + 0.05  # Add a small padding above

        # Ensure we don't go below 0 or above 1
        min_val = max(0, min_val)
        max_val = min(1.0, max_val)

        # If the range is very small, expand it to show differences better
        if max_val - min_val < 0.2:
            mid_point = (max_val + min_val) / 2
            min_val = max(0, mid_point - 0.1)
            max_val = min(1.0, mid_point + 0.1)

        bars = axes[i, 2].bar(
            [f"Comm {comm_id}" for comm_id in community_ids],
            echo_values,
            color="lightgreen",
        )
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[i, 2].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axes[i, 2].set_title(f"Echo Chamber Score - {rec_type}", fontsize=12)
        axes[i, 2].set_ylabel("Echo Chamber Score", fontsize=10)
        axes[i, 2].tick_params(axis="x", rotation=45)
        axes[i, 2].grid(True, linestyle="--", alpha=0.7, axis="y")
        axes[i, 2].set_ylim(min_val, max_val)

    plt.suptitle("Community Metrics by Recommender Type", fontsize=16, y=1.02)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

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
    with open(community_data_file, "rb") as f:
        community_data_by_run = pickle.load(f)

    # Extract diversity levels and recommender types from the data
    diversity_levels = sorted(data["diversity_level"].unique())
    recommender_types = sorted(data["recommender_type"].unique())

    # Create a figure for the table with wider width to allow for padding
    fig = plt.figure(
        figsize=(14, len(recommender_types) * 8)
    )  # Increased width from 14 to 16

    # Prepare metrics to display
    metrics = ["Misinfo Ratio", "EC"]

    # Create a subplot for each recommender type
    for i, rec_type in enumerate(recommender_types):
        ax = fig.add_subplot(len(recommender_types), 1, i + 1)
        ax.axis("tight")
        ax.axis("off")

        # Get the last step data for this recommender type
        last_step_data = {}
        for run_id, run_data in community_data_by_run.items():
            if (
                "recommender_type" in run_data
                and run_data["recommender_type"] == rec_type
            ):

                # Extract step and run key
                step = int(run_id.split("_")[-1])
                run_key = "_".join(run_id.split("_")[:-1])

                # Store the highest step for each run key
                if (
                    run_key not in last_step_data
                    or step > last_step_data[run_key]["step"]
                ):
                    last_step_data[run_key] = {
                        "step": step,
                        "data": run_data,
                        "diversity_level": run_data.get("diversity_level", 0),
                    }

        # Group data by diversity level
        data_by_diversity = {}
        for run_info in last_step_data.values():
            div_level = run_info["diversity_level"]
            if div_level not in data_by_diversity:
                data_by_diversity[div_level] = []
            data_by_diversity[div_level].append(run_info["data"])

        # Get all unique community IDs across all diversity levels
        all_communities = set()
        for div_data_list in data_by_diversity.values():
            for div_data in div_data_list:
                if "communities" in div_data:
                    all_communities.update(set(div_data["communities"].values()))

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
                    if "sizes" in div_data and comm_id in div_data["sizes"]:
                        sizes.append(div_data["sizes"][comm_id])

            # Add average size to row
            row.append(f"{int(np.mean(sizes)) if sizes else 0}")

            # For each diversity level, get the misinformation ratio value
            for div_level in diversity_levels:
                if div_level in data_by_diversity:
                    # Get values for this community across all runs with this diversity level
                    values = []
                    for div_data in data_by_diversity[div_level]:
                        if (
                            "fake_ratio" in div_data
                            and comm_id in div_data["fake_ratio"]
                        ):
                            values.append(div_data["fake_ratio"][comm_id])

                    # Calculate average value
                    if values:
                        avg_value = np.mean(values)
                        row.append(f"{avg_value:.3f}")
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")

            # For each diversity level, get the echo chamber score
            for div_level in diversity_levels:
                if div_level in data_by_diversity:
                    # Get values for this community across all runs with this diversity level
                    values = []
                    for div_data in data_by_diversity[div_level]:
                        # For echo_scores, calculate if not present
                        if (
                            "echo_scores" in div_data
                            and comm_id in div_data["echo_scores"]
                        ):
                            values.append(div_data["echo_scores"][comm_id])
                        elif (
                            "fake_ratio" in div_data
                            and "within_sims" in div_data
                            and comm_id in div_data.get("fake_ratio", {})
                            and comm_id in div_data.get("within_sims", {})
                        ):

                            echo_score = (
                                div_data["fake_ratio"][comm_id]
                                + div_data["within_sims"][comm_id]
                            ) / 2
                            values.append(echo_score)

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
        # Add Misinfo Ratio columns for each diversity level
        for level in diversity_levels:
            col_labels.append(f"MR {level}")
        # Add Echo Chamber columns for each diversity level
        for level in diversity_levels:
            col_labels.append(f"EC {level}")

        # Create table
        if len(table_data) > 0:  # Only create table if we have data rows
            table = ax.table(
                cellText=table_data,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
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
                for k in range(
                    2, len(col_labels)
                ):  # Column index (metrics at different diversity levels)
                    cell = table[j + 1, k]  # +1 to account for header row
                    cell_text = cell.get_text().get_text()

                    # Skip cells with N/A
                    if cell_text == "N/A":
                        continue

                    try:
                        value = float(cell_text)

                        # Determine which metric this column represents
                        is_misinfo_ratio = 2 <= k < (2 + len(diversity_levels))

                        # Color based on metric type
                        if is_misinfo_ratio:  # Misinfo Ratio (red = high)
                            cell.set_facecolor((value, 1 - value, 0, 0.3))
                        else:  # Echo Chamber (purple = high)
                            cell.set_facecolor((value, 0, value, 0.3))
                    except ValueError:
                        # Skip cells that can't be converted to float
                        continue

            # Add "Div" prefix to diversity level headers
            for col in range(2, len(col_labels)):
                cell = table[0, col]
                current_text = cell.get_text().get_text()
                if current_text.startswith("MR ") or current_text.startswith("EC "):
                    level = current_text.split(" ")[1]
                    prefix = current_text.split(" ")[0]
                    cell.get_text().set_text(f"{prefix} Div {level}")

            ax.set_title(f"Community Metrics - {rec_type}", fontsize=14, pad=20)

    plt.suptitle("Community Metrics Analysis by Recommender Type", fontsize=16, y=0.99)

    # Use tight_layout with more padding on the sides
    plt.tight_layout(rect=[0.05, 0, 0.95, 0.95])  # Added padding on left and right

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_single_diversity_timeline(
    data, metric_name, y_label, title, output_path=None, skip_steps=0
):
    """
    Plot a single timeline for a specific diversity setting (No Diversity).

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    metric_name : str
        Name of the metric column to plot
    y_label : str
        Label for the y-axis
    title : str
        Title for the plot
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    skip_steps : int, optional
        Number of initial steps to skip in the plot
    """

    if metric_name not in data.columns:
        print(f"[plot.py] Skipping '{metric_name}' plot — column not found")
        return None

    # Filter data for No Diversity setting
    filtered_data = data[data["diversity_setting"] == "No Diversity"]

    # Create a copy with mapped labels
    plot_data = filtered_data.copy()
    plot_data["recommender_label"] = plot_data["recommender_type"].apply(
        get_recommender_label
    )

    # Skip initial steps if specified
    if skip_steps > 0:
        plot_data = plot_data[plot_data["Step"] > skip_steps]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot data
    sns.lineplot(
        data=plot_data,
        x="Step",
        y=metric_name,
        hue="recommender_label",
        errorbar="sd",
        palette=RECOMMENDER_COLORS,
        linewidth=2.5,
        ax=ax,
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Step", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add horizontal line at y=0 for MRD
    if metric_name == "Misinformation_Ratio_Difference":
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)

    # Remove the legend from the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()

    # Add the legend at the bottom of the plot
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc="upper center",
        ncol=len(labels),
        fontsize=12,
    )

    plt.tight_layout(
        rect=[0, 0.05, 1, 1]
    )  # Adjust the bottom margin to make room for the legend

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def generate_no_diversity_plots(data, output_dir=None, skip_steps=5):
    """
    Generate standalone plots for the No Diversity setting.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing experiment results
    output_dir : str, optional
        Directory to save the plots. If None, plots are saved in the current directory.
    skip_steps : int, optional
        Number of initial steps to skip for Echo_Chamber_Effect plots
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = ""

    # Generate individual plots for No Diversity
    plot_single_diversity_timeline(
        data,
        "Misinformation_Spread_Percentage",
        "Infection Rate",
        "Misinformation Infection Rate (No Diversity)",
        os.path.join(output_dir, "no_diversity_infection_rate.png"),
    )

    plot_single_diversity_timeline(
        data,
        "Misinformation_Ratio_Difference",
        "MRD (positive = amplifying misinformation)",
        "Misinformation Ratio Difference (No Diversity)",
        os.path.join(output_dir, "no_diversity_mrd.png"),
    )

    plot_single_diversity_timeline(
        data,
        "Misinformation_Count_In_Recommendations",
        "Average Number of Misinformation Items",
        "Average Misinformation Count in Recommendations (No Diversity)",
        os.path.join(output_dir, "no_diversity_misinfo_count.png"),
    )

    plot_single_diversity_timeline(
        data,
        "Echo_Chamber_Effect",
        "Echo Chamber Index",
        "Echo Chamber Effect (No Diversity)",
        os.path.join(output_dir, "no_diversity_echo_chamber.png"),
        skip_steps,
    )
    plot_single_diversity_timeline(
        data,
        "Precision",
        "Precision",
        "Precision (No Diversity)",
        os.path.join(output_dir, "no_diversity_precision.png"),
    )
    plot_single_diversity_timeline(
        data,
        "Recall",
        "Recall",
        "Recall (No Diversity)",
        os.path.join(output_dir, "no_diversity_recall.png"),
    )


def plot_dominant_trait_degree_bar(
    data, degree_type, diversity_setting, recommender_type, output_path=None
):
    trait_letters = ["E", "A", "C", "N", "O"]
    trait_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    if "Count_By_Dominant_Personality_O" not in data.columns:
        print(
            "[plot.py] Skipping dominant trait degree plot — missing personality data"
        )
        return None
    # Map plot arg -> datacollector column prefix
    if degree_type.lower() in ["followers", "in"]:
        mean_prefix = "Mean_Followers_"
        y_label = "Mean Followers"
    elif degree_type.lower() in ["following", "out"]:
        mean_prefix = "Mean_Following_"
        y_label = "Mean Following"
    else:
        raise ValueError(
            "degree_type must be 'Followers' or 'Following' (or 'in'/'out')."
        )

    cols = [f"{mean_prefix}{t}" for t in trait_letters]
    count_cols = [f"Count_By_Dominant_Personality_{t}" for t in trait_letters]

    df = data[
        (data["diversity_setting"] == diversity_setting)
        & (data["recommender_type"] == recommender_type)
    ].copy()

    # final step per run
    df["RunKey"] = df["RunId"].astype(str) + "_" + df["iteration"].astype(str)
    last_steps = df.groupby("RunKey")["Step"].transform("max")
    final_df = df[df["Step"] == last_steps]

    means = final_df[cols].mean()
    stds = final_df[cols].std()
    counts = final_df[count_cols].mean().round().astype(int)

    labels = [
        f"{t}\n(n={counts[f'Count_By_Dominant_Personality_{t}']})"
        for t in trait_letters
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means.values, yerr=stds.values, capsize=5, color=trait_colors)

    ax.set_title(
        f"Final-Step {degree_type} by Dominant Trait\n"
        f"Recommender={get_recommender_label(recommender_type)}, {diversity_setting}"
    )
    ax.set_xlabel("Dominant Trait (regular users only)")
    ax.set_ylabel(y_label)

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_dominant_trait_degree_bar_all_recommenders(
    data, degree_type, diversity_setting, output_path=None
):

    trait_letters = ["E", "A", "C", "N", "O"]
    trait_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    if "Count_By_Dominant_Personality_O" not in data.columns:
        print(
            "[plot.py] Skipping dominant trait degree plot — missing personality data"
        )
        return None

    if degree_type.lower() in ["followers", "in"]:
        mean_prefix = "Mean_Followers_"
        y_label = "Mean Followers"
    elif degree_type.lower() in ["following", "out"]:
        mean_prefix = "Mean_Following_"
        y_label = "Mean Following"
    else:
        raise ValueError(
            "degree_type must be 'Followers' or 'Following' (or 'in'/'out')."
        )

    cols = [f"{mean_prefix}{t}" for t in trait_letters]
    count_cols = [f"Count_By_Dominant_Personality_{t}" for t in trait_letters]

    df = data[data["diversity_setting"] == diversity_setting].copy()

    recommenders = sorted(df["recommender_type"].unique())
    n_rec = len(recommenders)

    # create subplot grid
    fig, axes = plt.subplots(1, n_rec, figsize=(5 * n_rec, 5), sharey=True)

    if n_rec == 1:
        axes = [axes]
    df["RunKey"] = df["RunId"].astype(str) + "_" + df["iteration"].astype(str)
    last_steps = df.groupby("RunKey")["Step"].transform("max")
    final_df = df[df["Step"] == last_steps]

    for ax, recommender_type in zip(axes, recommenders):

        sub = final_df[final_df["recommender_type"] == recommender_type]

        if sub.empty:
            ax.set_visible(False)
            continue

        means = sub[cols].mean()
        stds = sub[cols].std()
        counts = sub[count_cols].mean().round().astype(int)

        labels = [
            f"{t}\n(n={counts[f'Count_By_Dominant_Personality_{t}']})"
            for t in trait_letters
        ]

        ax.bar(labels, means.values, yerr=stds.values, capsize=5, color=trait_colors)

        ax.set_title(get_recommender_label(recommender_type))
        ax.set_xlabel("Dominant Trait")

        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")

    axes[0].set_ylabel(y_label)

    fig.suptitle(
        f"Final-Step {degree_type} by Dominant Trait\n{diversity_setting}", fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_recommendation_like_rate(data, output_path=None):
    """
    Plot recommendation like rate for each recommender type,
    with separate bar plots for each diversity setting.

    Like rate is computed as:
        total Recommendation_Likes / total Recommendation_Impressions

    Parameters:
    -----------
    data : pandas.DataFrame
        Experiment results
    output_path : str, optional
        Path to save the plot
    """

    required_cols = [
        "recommender_type",
        "Recommendation_Impressions",
        "Recommendation_Likes",
        "diversity_setting",
    ]

    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return None

    # Create a copy of the data
    df = data.copy()
    diversity_settings = sorted(df["diversity_setting"].unique())

    if len(diversity_settings) == 0:
        print("No diversity settings found.")
        return None

    # Create a figure with subplots for each diversity setting
    fig, axes = plt.subplots(
        1,
        len(diversity_settings),
        figsize=(7 * len(diversity_settings), 7),
        sharey=True,
    )

    if len(diversity_settings) == 1:
        axes = [axes]

    all_summaries = []

    # Plot for each diversity setting
    for i, diversity in enumerate(diversity_settings):
        ax = axes[i]
        subset = df[df["diversity_setting"] == diversity].copy()

        if subset.empty:
            ax.set_visible(False)
            continue

        summary = subset.groupby("recommender_type", as_index=False).agg(
            total_recommendations=("Recommendation_Impressions", "sum"),
            total_likes=("Recommendation_Likes", "sum"),
        )

        summary["like_rate"] = np.where(
            summary["total_recommendations"] > 0,
            summary["total_likes"] / summary["total_recommendations"],
            0,
        )

        summary["label"] = summary["recommender_type"].apply(get_recommender_label)
        summary = summary.sort_values("like_rate", ascending=False)

        all_summaries.append(summary)

        bars = ax.bar(
            summary["label"],
            summary["like_rate"],
            color=[
                RECOMMENDER_COLORS.get(label, "#888888") for label in summary["label"]
            ],
            alpha=0.85,
        )

        # add value labels
        for bar, value in zip(bars, summary["like_rate"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title(f"{diversity}", fontsize=16)
        ax.set_xlabel("Recommender Type", fontsize=14)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)

        if i == 0:
            ax.set_ylabel("Total Likes / Total Recommended", fontsize=14)
        else:
            ax.set_ylabel("")

    # Use common y-limit across all plots
    max_rate = 0
    for summary in all_summaries:
        if not summary.empty:
            max_rate = max(max_rate, summary["like_rate"].max())

    ymax = max_rate * 1.15 if max_rate > 0 else 0.05
    for ax in axes:
        ax.set_ylim(0, ymax)

    plt.suptitle("Recommendation Like Rate by Recommender Type", fontsize=18, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_like_rate_vs_misinformation_count(data, output_path=None):
    """
    Plot the tradeoff between recommendation like rate and average
    misinformation count in recommendations, with one subplot per
    diversity setting.

    x-axis: average misinformation count in recommendations
    y-axis: total recommendation likes / total recommendation impressions

    Parameters:
    -----------
    data : pandas.DataFrame
        Experiment results
    output_path : str, optional
        Path to save the plot
    """
    required_cols = [
        "recommender_type",
        "Precision",
        "Misinformation_Count_In_Recommendations",
        "diversity_setting",
    ]

    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return None

    df = data.copy()

    # Optional: enforce a readable/fixed order
    preferred_order = ["Diversity 0.75", "Diversity 1.0", "No Diversity"]
    diversity_settings = [
        d for d in preferred_order if d in df["diversity_setting"].unique()
    ]
    if not diversity_settings:
        diversity_settings = sorted(df["diversity_setting"].unique())

    fig, axes = plt.subplots(
        1,
        len(diversity_settings),
        figsize=(7 * len(diversity_settings), 7),
        sharex=True,
        sharey=True,
    )

    if len(diversity_settings) == 1:
        axes = [axes]

    panel_summaries = []

    for i, diversity in enumerate(diversity_settings):
        ax = axes[i]
        subset = df[df["diversity_setting"] == diversity].copy()

        if subset.empty:
            ax.set_visible(False)
            continue

        # Aggregate per recommender
        summary = subset.groupby("recommender_type", as_index=False).agg(
            precision=("Precision", "mean"),
            avg_mc=("Misinformation_Count_In_Recommendations", "mean"),
        )

        summary["label"] = summary["recommender_type"].apply(get_recommender_label)
        panel_summaries.append(summary)

        # Scatter points
        for _, row in summary.iterrows():
            ax.scatter(
                row["avg_mc"],
                row["precision"],
                s=140,
                color=RECOMMENDER_COLORS.get(row["label"], "#888888"),
                alpha=0.9,
                edgecolor="black",
                linewidth=0.8,
            )

            ax.annotate(
                row["label"],
                (row["avg_mc"], row["precision"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=11,
                fontweight="bold",
            )

        ax.set_title(f"{diversity}", fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Helpful visual guide
        ax.annotate(
            "Better",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=11,
            color="darkgreen",
            fontweight="bold",
        )

        if i == 0:
            ax.set_ylabel("Precision", fontsize=14)
        else:
            ax.set_ylabel("")

        ax.set_xlabel("Average Misinformation Count", fontsize=14)

    # Shared limits across panels
    max_x = max(
        (summary["avg_mc"].max() for summary in panel_summaries if not summary.empty),
        default=1,
    )
    max_y = max(
        (
            summary["precision"].max()
            for summary in panel_summaries
            if not summary.empty
        ),
        default=0.05,
    )

    x_max = max_x * 1.1 if max_x > 0 else 1
    y_max = max_y * 1.15 if max_y > 0 else 0.05

    for ax in axes:
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)

    plt.suptitle(
        "Precision vs. Misinformation Count in Recommendations",
        fontsize=18,
        y=1.02,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def generate_all_plots(
    csv_path, output_dir=None, community_data_file=None, skip_steps=5
):
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
    skip_steps : int, optional
        Number of initial steps to skip for Echo_Chamber_Effect plots
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = ""

    # Load data
    data = load_experiment_data(csv_path)

    # Generate individual plots
    plot_misinformation_spread(
        data, os.path.join(output_dir, "misinformation_spread_comparison.png")
    )
    plot_infection_rate(data, os.path.join(output_dir, "infection_rate_comparison.png"))
    plot_misinformation_ratio_difference(
        data, os.path.join(output_dir, "misinformation_ratio_difference_comparison.png")
    )
    plot_misinformation_count(
        data, os.path.join(output_dir, "misinformation_count_comparison.png")
    )
    plot_echo_chamber_effect(
        data, os.path.join(output_dir, "echo_chamber_effect_comparison.png"), skip_steps
    )
    plot_diversity_impact_heatmap(
        data, os.path.join(output_dir, "diversity_impact_heatmap.png"), skip_steps
    )

    # Generate No Diversity standalone plots
    generate_no_diversity_plots(data, output_dir, skip_steps)

    # Generate summary plots
    plot_recommender_summary(
        data, os.path.join(output_dir, "recommender_summary.png"), skip_steps
    )

    # Generate ranking table
    create_recommender_ranking_table(
        data, os.path.join(output_dir, "recommender_ranking_table.png"), skip_steps
    )

    create_success_metrics_ranking_table(
        data, os.path.join(output_dir, "success_metrics_ranking_table.png")
    )

    # Generate community-specific plots if community data is available
    if community_data_file:
        # Generate recommender-specific community plots
        plot_community_metrics_table_by_recommender(
            data,
            community_data_file,
            os.path.join(output_dir, "community_metrics_table_by_recommender.png"),
        )
        plot_community_metrics_by_recommender(
            data,
            community_data_file,
            os.path.join(output_dir, "community_metrics_by_recommender.png"),
        )
        plot_diversity_impact_table(
            data,
            community_data_file,
            os.path.join(output_dir, "diversity_impact_table.png"),
        )

    # plot_dominant_trait_degree_bar(
    #    data,
    #    degree_type="Followers",
    #    diversity_setting="No Diversity",
    #    recommender_type="random",
    #    output_path=os.path.join(output_dir, "dominant_trait_followers_final.png"),
    # )
    # plot_dominant_trait_degree_bar(
    #    data,
    #    degree_type="Following",
    #    diversity_setting="No Diversity",
    #    recommender_type="random",
    #    output_path=os.path.join(output_dir, "dominant_trait_following_final.png"),
    # )
    # plot_dominant_trait_degree_bar_all_recommenders(
    #    data,
    #    degree_type="Followers",
    #    diversity_setting="No Diversity",
    #    output_path=os.path.join(
    #        output_dir, "dominant_trait_followers_final_all_recommenders.png"
    #    ),
    # )
    # plot_dominant_trait_degree_bar_all_recommenders(
    #    data,
    #    degree_type="Following",
    #    diversity_setting="No Diversity",
    #    output_path=os.path.join(
    #        output_dir, "dominant_trait_following_final_all_recommenders.png"
    #    ),
    # )
    # plot_recommendation_like_rate(data, os.path.join(output_dir, "recommendation_like_rate.png"))

    plot_like_rate_vs_misinformation_count(
        data, os.path.join(output_dir, "precision_vs_misinformation_count.png")
    )
    # Show all plots
    plt.show()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Generate plots from experiment data")
    parser.add_argument(
        "csv_file", type=str, help="Path to the CSV file containing experiment results"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save the plots"
    )
    parser.add_argument(
        "--community-data-file",
        type=str,
        default=None,
        help="Path to the pickle file containing community data",
    )
    parser.add_argument(
        "--skip-steps",
        type=int,
        default=5,
        help="Number of initial steps to skip for Echo Chamber Effect plots",
    )

    args = parser.parse_args()

    generate_all_plots(
        args.csv_file, args.output_dir, args.community_data_file, args.skip_steps
    )
