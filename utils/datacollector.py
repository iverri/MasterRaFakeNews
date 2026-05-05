from mesa import DataCollector
from utils.metrics import (
    calculate_echo_chamber_effect,
    calculate_misinformation_count,
    calculate_misinformation_ratio_difference,
    calculate_misinformation_spread,
    TRAIT_NAMES,
    calculate_mean_degree_by_dominant_personality,
    calculate_diversity_improvement,
    calculate_count_by_dominant_personality,
    infection_by_personality,
)

def setup_datacollector(model):
    """Initialize the datacollector with metrics."""

    model_reporters = {
        "Number_of_Infected": lambda m: sum(1 for a in m.agents if getattr(a, "state", None) == "I"),
        "Number_of_Susceptible": lambda m: sum(1 for a in m.agents if getattr(a, "state", None) == "S"),
        "Number_of_Exposed": lambda m: sum(1 for a in m.agents if getattr(a, "state", None) == "E"),
        "Current_Hour": lambda m: m.current_hour,
        "Average_Feed_Size": lambda m: (
            sum(len(getattr(a, "recommended_content", [])) for a in m.agents) / len(m.agents)
            if len(m.agents) > 0 else 0
        ),
        "Average_Diversity_Score": lambda m: (
            sum(getattr(a, "diversity_score", 0) for a in m.agents) / len(m.agents)
            if len(m.agents) > 0 else 0
        ),
        "Diversity_Improvement_Percentage": lambda m: calculate_diversity_improvement(m),
        "Misinformation_Count_In_Recommendations": lambda m: calculate_misinformation_count(m),
        "Misinformation_Ratio_Difference": lambda m: calculate_misinformation_ratio_difference(m),
        "Misinformation_Spread_Percentage": lambda m: calculate_misinformation_spread(m),
        "Echo_Chamber_Effect": lambda m: calculate_echo_chamber_effect(m),
    }

    if getattr(model, "collect_community_data", False):
        model_reporters["Community_Data"] = lambda m: getattr(m, "community_data", None)
        model_reporters["Number_Of_Communities"] = lambda m: (
            len(set(m.community_data["communities"].values()))
            if hasattr(m, "community_data") and m.community_data is not None else 0
        )
    if getattr(model, "collect_personality_degree_stats", False):
        for idx, name in enumerate(TRAIT_NAMES):
            model_reporters[f"Mean_Followers_{name}"] = lambda m, idx=idx: calculate_mean_degree_by_dominant_personality(m, idx, mode="in")
            model_reporters[f"Mean_Following_{name}"] = lambda m, idx=idx: calculate_mean_degree_by_dominant_personality(m, idx, mode="out")
            model_reporters[f"Count_By_Dominant_Personality_{name}"] = lambda m, idx=idx: calculate_count_by_dominant_personality(m, idx)


    for idx, name in enumerate(TRAIT_NAMES):
        model_reporters[f"Infected_Rate_{name}"] = lambda m, idx=idx: infection_by_personality(m, idx)
   
    agent_reporters = (
    {
        "State": lambda a: getattr(a, "state", None),
        "Followers": lambda a: a.social_media_platform.social_network.network.in_degree(a.pos),
        "Following": lambda a: a.social_media_platform.social_network.network.out_degree(a.pos),
        "Misinformation_In_Recommendations": lambda a: sum(
            1 for c in getattr(a, "recommended_content", [])
            if getattr(c, "isFake", False)
        ),
    }
    if getattr(model, "collect_agent_stats", False)
    else {}
    )

    return DataCollector(model_reporters=model_reporters, agent_reporters=agent_reporters)