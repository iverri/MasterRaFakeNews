from mesa import DataCollector
from utils.metrics import (
    calculate_echo_chamber_effect,
    calculate_misinformation_count,
    calculate_misinformation_ratio_difference,
    calculate_misinformation_spread,
    calculate_diversity_improvement,
    TRAIT_NAMES,
    make_cached_reporter,
)

import numpy as np

def setup_datacollector(model):
    """Initialize the datacollector with metrics."""

    model_reporters = {
        "Number_of_Infected": lambda m: sum(1 for a in m.agents if getattr(a, "state", None) == "I"),
        "Number_of_Susceptible": lambda m: sum(1 for a in m.agents if getattr(a, "state", None) == "S"),
        "Number_of_Exposed": lambda m: sum(1 for a in m.agents if getattr(a, "state", None) == "E"),
        "Current_Hour": lambda m: m.current_hour,
        "Average_Feed_Size": lambda m: (
            sum(len(getattr(a, "recommended_content", [])) for a in m.agents) / len(m.agents)
            if m.agents else 0
        ),
        "Average_Diversity_Score": lambda m: (
            sum(getattr(a, "diversity_score", 0) for a in m.agents if getattr(a, "diversity_score", 0) != 0) / len(m.agents)
            if m.agents else 0
        ),
        "Diversity_Improvement_Percentage": lambda m: calculate_diversity_improvement(m),
        "Misinformation_Count_In_Recommendations": lambda m: calculate_misinformation_count(m),
        "Misinformation_Ratio_Difference": lambda m: calculate_misinformation_ratio_difference(m),
        "Misinformation_Spread_Percentage": lambda m: calculate_misinformation_spread(m),
        "Echo_Chamber_Effect": lambda m: calculate_echo_chamber_effect(m),
        "Community_Data": lambda m: getattr(m, "community_data", None),
        "Number_Of_Communities": lambda m: (
            len(set(getattr(m, "community_data", {}).get("communities", {}).values()))
        ),
    }

    if getattr(model, "collect_personality_data_metrics", False):
        for name in TRAIT_NAMES:
            for suffix in ("Count", "Followers_Mean", "Following_Mean"):
                key = f"Dom_{name}_{suffix}"
                model_reporters[key] = make_cached_reporter(key, model)


        ##see if this really slows down the model too much before adding it back in
        ##T = len(TRAIT_NAMES)
        ##for i in range(T):
        ##    for j in range(T):
        ##        for prefix in ("DomFollowShare", "DomFollowCount"):
        ##            key = f"{prefix}_{i}_{j}"
        ##            model_reporters[key] = make_cached_reporter(key, model)
    
    agent_reporters = {
        "State": lambda a: getattr(a, "state", None),
        "Followers": lambda a: a.social_media_platform.social_network.network.in_degree(a.pos),
        "Following": lambda a: a.social_media_platform.social_network.network.out_degree(a.pos),
        "Misinformation_In_Recommendations": lambda a: sum(
            1 for c in getattr(a, "recommended_content", []) if c.isFake
        ),
    }

    return DataCollector(
        model_reporters=model_reporters,
        agent_reporters=agent_reporters,
    )
