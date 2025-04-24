
from mesa import DataCollector
import networkx as nx
from utils.metrics import calculate_agent_echo_chamber, calculate_content_propagation_clustering, calculate_echo_chamber_effect, calculate_misinformation_count, calculate_misinformation_ratio_difference, calculate_misinformation_spread

def setup_datacollector(model):
    """Initialize the datacollector with metrics."""
    return DataCollector(
        model_reporters={
            "Number_of_Infected": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "I"),
            "Number_of_Susceptible": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "S"),
            "Number_of_Exposed": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "E"),
            "Active_Users": lambda m: sum(1 for a in m.agents if hasattr(a, "is_active") and a.is_active),
            "Active_Percentage": lambda m: sum(1 for a in m.agents if hasattr(a, "is_active") and a.is_active) / len(m.agents) if len(m.agents) > 0 else 0,
            "Active_Infected": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "I" and hasattr(a, "is_active") and a.is_active),
            "Current_Hour": lambda m: m.current_hour,
            "Average_Feed_Size": lambda m: sum(len(a.recommended_content) for a in m.agents if hasattr(a, "recommended_content")) / len(m.agents) if len(m.agents) > 0 else 0,
            "Average_Diversity_Score": lambda m: sum(a.diversity_score for a in m.agents if hasattr(a, "diversity_score") and a.diversity_score != 0) / len(m.agents) if len(m.agents) > 0 else 0,
            "Misinformation_Count_In_Recommendations": lambda m: calculate_misinformation_count(m),
            "Misinformation_Ratio_Difference": lambda m: calculate_misinformation_ratio_difference(m),
            "Misinformation_Spread_Percentage": lambda m: calculate_misinformation_spread(m),
            "Echo_Chamber_Effect": lambda m: calculate_echo_chamber_effect(m),
            "Preference_Similarity": lambda m: sum([calculate_agent_echo_chamber(a) 
                                    for a in m.agents if hasattr(a, "recommended_content") and len(a.recommended_content) > 0]) / 
                                    sum(1 for a in m.agents if hasattr(a, "recommended_content") and len(a.recommended_content) > 0) 
                                    if sum(1 for a in m.agents if hasattr(a, "recommended_content") and len(a.recommended_content) > 0) > 0 else 0,
            "Content_Propagation_Clustering": lambda m: calculate_content_propagation_clustering(m),
        },
        agent_reporters={
            "State": lambda a: getattr(a, "state", None),
            "Followers": lambda a: a.social_media_platform.social_network.network.in_degree(a.pos),
            "Following": lambda a: a.social_media_platform.social_network.network.out_degree(a.pos),
            "Is_Active": lambda a: getattr(a, "is_active", False),
            "Activity_Probability": lambda a: getattr(a, "activity_probability", 0),
            "Feed_Size": lambda a: len(a.feed),
            "Diversity_Score": lambda a: getattr(a, "diversity_score", 0),
            "Misinformation_In_Recommendations": lambda a: sum(1 for c in a.recommended_content if c.isFake) if hasattr(a, "recommended_content") else 0,
            "Echo_Chamber_Score": lambda a: calculate_agent_echo_chamber(a) if hasattr(a, "recommended_content") else 0,
        }
    )