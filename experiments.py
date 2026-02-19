import mesa
import pandas as pd
import numpy as np
import os
from datetime import datetime
from model import FakeNewsModel
from recommender.types import RecommenderType
from utils.network_storage import NetworkStorage


def run_recommender_comparison_experiment(
    iterations,
    max_steps,
    n_agents,
    m_links,
    news_amount,
    fake_news_percentage,
    bot_percentage,
    influencer_percentage,
    num_recommendations,
):
    """
    Run a batch experiment comparing different recommender algorithms.

    Parameters:
    -----------
    iterations : int
        Number of iterations to run for each parameter combination
    max_steps : int
        Maximum number of steps to run each model
    n_agents : int
        Number of agents in the model
    m_links : int
        Number of links per new node in the network
    news_amount : int
        Initial amount of news content
    fake_news_percentage : int
        Percentage of fake news in the content pool
    bot_percentage : int
        Percentage of bot agents
    influencer_percentage : int
        Percentage of influencer agents
    num_recommendations : int
        Number of recommendations per agent
    """
    # Create output directory if it doesn't exist
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)

    # Get current timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clear any existing stored network to ensure we create a fresh one with current parameters
    from utils.network_storage import NetworkStorage

    NetworkStorage.clear()

    # Create an initial model to generate and store the network with current parameters
    # Set use_stored_network=False to force creation of a new network
    initial_model = FakeNewsModel(
        N=n_agents,
        m_links=m_links,
        news_amount=news_amount,
        fake_news_percentage=fake_news_percentage,
        bot_percentage=bot_percentage,
        influencer_percentage=influencer_percentage,
        diversity_level=0,
        num_recommendations=num_recommendations,
        use_stored_network=False,
        stored_network=None,  # Force creation of new network
        recommender_type=RecommenderType.RANDOM.value,  # Use any recommender for initial setup
        max_steps=max_steps,
    )

    # Store the network to a file that can be accessed by all processes
    network_file = f"{output_dir}/network_{timestamp}.pkl"
    NetworkStorage.store_network_to_file(
        initial_model.social_media_platform.social_network.network,
        initial_model.preference_vectors,
        network_file,
    )
    print(f"Created and stored initial network to {network_file}")

    # Define parameters for batch run
    # We'll vary the recommender type while keeping other parameters fixed
    parameters = {
        "N": n_agents,
        "m_links": m_links,
        "news_amount": news_amount,
        "fake_news_percentage": fake_news_percentage,
        "bot_percentage": bot_percentage,
        "influencer_percentage": influencer_percentage,
        "diversity_level": [0, 0.75, 1.0],
        "num_recommendations": num_recommendations,
        "use_stored_network": True,  # Now use the stored network for all runs
        "network_file": network_file,  # Pass the network file path instead of the network object
        "stored_network": None,  # No longer needed
        "recommender_type": [type.value for type in RecommenderType],
        "collect_personality_data_metrics": True,  # Enable personality data collection
    }

    print(f"Starting batch run with {iterations} iterations per recommender type...")
    print(f"Recommender types: {[type.value for type in RecommenderType]}")

    # Run the batch experiment
    results = mesa.batch_run(
        FakeNewsModel,
        parameters=parameters,
        iterations=iterations,
        max_steps=max_steps,
        number_processes=8,  # Set to higher number for parallel processing
        data_collection_period=1,  # Collect data at each step
        display_progress=True,
    )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Store community data separately since it's complex and not CSV-friendly
    community_data_by_run = {}

    # Extract community data before dropping it from the main dataframe
    for idx, row in results_df.iterrows():
        if "Community_Data" in results_df.columns and pd.notna(row["Community_Data"]):
            run_id = f"{row['RunId']}_{row['iteration']}_{row['Step']}"
            # Add recommender type to the community data
            community_data = (
                row["Community_Data"].copy()
                if isinstance(row["Community_Data"], dict)
                else {}
            )
            community_data["recommender_type"] = row["recommender_type"]
            community_data["diversity_level"] = row["diversity_level"]
            community_data_by_run[run_id] = community_data

    # Save community data to a separate pickle file
    community_data_file = f"{output_dir}/community_data_{timestamp}.pkl"
    with open(community_data_file, "wb") as f:
        import pickle

        pickle.dump(community_data_by_run, f)
    print(f"Community data saved to {community_data_file}")

    # Remove complex objects that can't be easily stored in CSV
    if "Community_Data" in results_df.columns:
        results_df = results_df.drop(columns=["Community_Data"])

    # Create separate files for model-level and agent-level data
    model_vars = [
        "RunId",
        "iteration",
        "Step",
        "recommender_type",
        "num_recommendations",
        "fake_news_percentage",
        "diversity_level",
        "Number_of_Infected",
        "Number_of_Susceptible",
        "Number_of_Exposed",
        "Average_Diversity_Score",
        "Misinformation_Count_In_Recommendations",
        "Misinformation_Ratio_Difference",
        "Misinformation_Spread_Percentage",
        "Echo_Chamber_Effect",
        "Diversity_Improvement_Percentage",
        "Number_Of_Communities",
    ]
   

    include_personality_metrics = results_df["collect_personality_data_metrics"].iloc[0]
    if include_personality_metrics:
        trait_letters = ["E", "A", "C", "N", "O"]

        for t in trait_letters:
            model_vars += [
                f"Dom_{t}_Count",
                f"Dom_{t}_Followers_Mean",
                f"Dom_{t}_Following_Mean",]

        model_vars += [
            "Mean_Personality_Similarity_On_Edges_RegularOnly"
        ]

        for i in range(5):
            for j in range(5):
                model_vars += [f"DomFollowShare_{i}_{j}", f"DomFollowCount_{i}_{j}"]



    # Filter for model-level variables
    model_data = results_df[[col for col in model_vars if col in results_df.columns]]

    # Save model-level data
    model_file = f"{output_dir}/recommender_comparison_model_data_{timestamp}.csv"
    model_data.to_csv(model_file, index=False)
    print(f"Model-level data saved to {model_file}")

    # Create summary statistics for each recommender type at the final step
    final_step_data = []

    for recommender in [type.value for type in RecommenderType]:
        # Get data for the final step of each run with this recommender
        recommender_data = model_data[
            (model_data["recommender_type"] == recommender)
            & (
                model_data.groupby(["RunId", "iteration"])["Step"].transform("max")
                == model_data["Step"]
            )
        ]

        # Calculate summary statistics
        summary = {
            "recommender_type": recommender,
            "runs": len(recommender_data),
            "avg_infected_pct": recommender_data["Number_of_Infected"].mean()
            / n_agents
            * 100,
            "avg_misinformation_spread": recommender_data[
                "Misinformation_Spread_Percentage"
            ].mean()
            * 100,
            "avg_echo_chamber_effect": recommender_data["Echo_Chamber_Effect"].mean(),
            "avg_misinfo_in_recs": recommender_data[
                "Misinformation_Count_In_Recommendations"
            ].mean(),
            "avg_misinfo_ratio_diff": recommender_data[
                "Misinformation_Ratio_Difference"
            ].mean()
            * 100,
        }

        final_step_data.append(summary)

    # Create summary DataFrame and save
    summary_df = pd.DataFrame(final_step_data)
    summary_file = f"{output_dir}/recommender_comparison_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to {summary_file}")

    return results_df, model_data, summary_df, community_data_file


def analyze_results(model_data, summary_df):
    """
    Perform basic analysis on the experiment results.

    Parameters:
    -----------
    model_data : DataFrame
        Model-level data from the experiment
    summary_df : DataFrame
        Summary statistics for each recommender type
    """
    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"Compared {len(summary_df)} recommender types")

    # Print summary table
    print("\nFinal state comparison:")
    print(
        summary_df[
            [
                "recommender_type",
                "avg_infected_pct",
                "avg_misinformation_spread",
                "avg_echo_chamber_effect",
            ]
        ].to_string(index=False)
    )

    # Find the recommender with lowest misinformation spread
    best_for_misinfo = summary_df.loc[summary_df["avg_misinformation_spread"].idxmin()]
    print(
        f"\nLowest misinformation spread: {best_for_misinfo['recommender_type']} "
        f"({best_for_misinfo['avg_misinformation_spread']:.2f}%)"
    )

    # Find the recommender with lowest echo chamber effect
    best_for_echo = summary_df.loc[summary_df["avg_echo_chamber_effect"].idxmin()]
    print(
        f"Lowest echo chamber effect: {best_for_echo['recommender_type']} "
        f"({best_for_echo['avg_echo_chamber_effect']:.2f})"
    )

    print(
        "\nNote: For detailed analysis and visualization, load the saved CSV files into your analysis tools."
    )


if __name__ == "__main__":

    # Run the experiment
    results_df, model_data, summary_df, community_data_file = (
        run_recommender_comparison_experiment(
            iterations=5,  # Number of runs per recommender type
            max_steps=700,  # Steps per run
            n_agents=200,  # Number of agents
            m_links=8,  # Links per new node
            news_amount=400,  # Initial news items
            fake_news_percentage=10,  # Percentage of fake news
            bot_percentage=7,  # Percentage of bots
            influencer_percentage=3,  # Percentage of influencers
            num_recommendations=10,  # Number of recommendations
        )
    )

    # Analyze the results
    analyze_results(model_data, summary_df)
