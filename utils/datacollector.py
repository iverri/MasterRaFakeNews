from mesa import DataCollector
import networkx as nx
from utils.metrics import calculate_echo_chamber_effect, calculate_misinformation_count, calculate_misinformation_ratio_difference, calculate_misinformation_spread, calculate_cluster_content_similarity, calculate_diversity_improvement

def setup_datacollector(model):
    """Initialize the datacollector with metrics."""
    return DataCollector(
        model_reporters={
            "Number_of_Infected": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "I"),
            "Number_of_Susceptible": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "S"),
            "Number_of_Exposed": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "E"),
            "Current_Hour": lambda m: m.current_hour,
            "Average_Feed_Size": lambda m: sum(len(a.recommended_content) for a in m.agents if hasattr(a, "recommended_content")) / len(m.agents) if len(m.agents) > 0 else 0,
            "Average_Diversity_Score": lambda m: sum(a.diversity_score for a in m.agents if hasattr(a, "diversity_score") and a.diversity_score != 0) / len(m.agents) if len(m.agents) > 0 else 0,
            "Diversity_Improvement_Percentage": lambda m: calculate_diversity_improvement(m),
            "Misinformation_Count_In_Recommendations": lambda m: calculate_misinformation_count(m),
            "Misinformation_Ratio_Difference": lambda m: calculate_misinformation_ratio_difference(m),
            "Misinformation_Spread_Percentage": lambda m: calculate_misinformation_spread(m),
            "Echo_Chamber_Effect": lambda m: calculate_echo_chamber_effect(m),
            "Community_Data": lambda m: getattr(m, 'community_data', None),
            "Number_Of_Communities": lambda m: len(set(m.community_data['communities'].values())) if hasattr(m, 'community_data') else 0,
        },
        if getattr(model, "collect_personality_data_metrics", False):
            for idx, name in enumerate(TRAIT_NAMES):
                model_reporters[f"Dom_{name}_Count"] = lambda m, i=idx: count_by_dominant_trait(m, i)
                model_reporters[f"Dom_{name}_Followers_Mean"] = lambda m, i=idx: mean_degree_by_dominant_trait(m, i, "in")
                model_reporters[f"Dom_{name}_Following_Mean"] = lambda m, i=idx: mean_degree_by_dominant_trait(m, i, "out")

            model_reporters["Mean_Personality_Similarity_On_Edges_RegularOnly"] = lambda m: mean_personality_similarity_on_edges_regular_only(m)
    
            for i in range(5):
                for j in range(5):
                    model_reporters[f"DomFollowShare_{i}_{j}"] = (
                        lambda ii=i, jj=j: (
                            lambda m: dominant_trait_follow_matrix(m).get(f"DomFollowShare_{ii}_{jj}", 0.0)
                        )
                    )()
                    model_reporters[f"DomFollowCount_{i}_{j}"] = (
                        lambda ii=i, jj=j: (
                            lambda m: dominant_trait_follow_matrix(m).get(f"DomFollowCount_{ii}_{jj}", 0)
                        )
                    )()

        agent_reporters={
            "State": lambda a: getattr(a, "state", None),
            "Followers": lambda a: a.social_media_platform.social_network.network.in_degree(a.pos),
            "Following": lambda a: a.social_media_platform.social_network.network.out_degree(a.pos),
            "Misinformation_In_Recommendations": lambda a: sum(1 for c in a.recommended_content if c.isFake) if hasattr(a, "recommended_content") else 0,
        }
    )