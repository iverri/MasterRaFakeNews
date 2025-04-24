import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_misinformation_count(model):
    """Calculate the average number of fake news items in agents' recommendation lists."""
    total_fake = sum(sum(1 for c in a.recommended_content if c.isFake) 
                    for a in model.agents if hasattr(a, "recommended_content"))
    active_agents = sum(1 for a in model.agents if hasattr(a, "recommended_content") and len(a.recommended_content) > 0)
    return total_fake / active_agents if active_agents > 0 else 0

def calculate_misinformation_ratio_difference(model):
    """
    Calculate the average difference between the ratio of fake news in each user's recommendations 
    vs. the overall ratio in the content pool.
    Positive values indicate recommendations have more fake news than the overall pool.
    """
    # Overall fake news ratio in content pool
    if not hasattr(model, 'news_content') or not model.news_content:
        return 0
    
    overall_fake_ratio = sum(1 for c in model.news_content if c.isFake) / len(model.news_content)
    
    # Calculate per-user differences
    user_differences = []
    
    for agent in model.agents:
        if hasattr(agent, "recommended_content") and agent.recommended_content:
            # Calculate fake news ratio in this user's recommendations
            user_recs_count = len(agent.recommended_content)
            user_fake_count = sum(1 for c in agent.recommended_content if c.isFake)
            user_fake_ratio = user_fake_count / user_recs_count
            
            # Calculate difference for this user
            user_difference = user_fake_ratio - overall_fake_ratio
            user_differences.append(user_difference)
    
    # Return the average difference across all users
    if user_differences:
        return sum(user_differences) / len(user_differences)
    else:
        return 0

def calculate_misinformation_spread(model):
    """Calculate the percentage of agents who have been exposed to fake news."""
    exposed_agents = sum(1 for a in model.agents if hasattr(a, "state") and ( a.state == "I"))
    return exposed_agents / len(model.agents) if len(model.agents) > 0 else 0

def calculate_echo_chamber_effect(model):
    """
    Calculate the average echo chamber effect across all agents.
    Higher values indicate stronger echo chambers.
    """
    # Get both preference-based and propagation-based scores
    preference_scores = [calculate_agent_echo_chamber(a) 
                        for a in model.agents if hasattr(a, "recommended_content") and len(a.recommended_content) > 0]
    
    propagation_scores = calculate_content_propagation_clustering(model)
    
    # Combine both metrics (equal weighting)
    if preference_scores and propagation_scores:
        return (sum(preference_scores) / len(preference_scores) + propagation_scores) / 2
    elif preference_scores:
        return sum(preference_scores) / len(preference_scores)
    elif propagation_scores:
        return propagation_scores
    else:
        return 0

def calculate_agent_echo_chamber(agent):
    """
    Calculate the echo chamber score for an individual agent.
    This measures how similar the content in their recommendations is to their preferences.
    """
    if not hasattr(agent, "recommended_content") or not agent.recommended_content:
        return 0
    
    # Calculate similarity between agent preferences and each content item
    agent_preference = np.array(agent.preference_vector).reshape(1, -1)
    content_topics = np.array([content.topic_vector for content in agent.recommended_content])
    similarities = cosine_similarity(agent_preference, content_topics)[0]
    
    # Higher average similarity indicates stronger echo chamber
    return sum(similarities) / len(similarities) if similarities.size > 0 else 0

def calculate_content_propagation_clustering(model):
    """
    Calculate echo chamber effect based on content propagation patterns in the network.
    This measures how much content stays within community clusters rather than spreading broadly.
    
    Higher values indicate stronger echo chambers (content stays within clusters).
    """
    import networkx as nx
    import community as community_louvain
    
    # Get the social network
    network = model.social_media_platform.social_network.network
    
    # If network is too small, return 0
    if network.number_of_nodes() < 10:
        return 0
    
    # Convert to undirected for community detection
    undirected_network = network.to_undirected()
    
    # Detect communities using Louvain method
    communities = community_louvain.best_partition(undirected_network)
    
    # Create a mapping of node to community
    node_to_community = communities
    
    # Track content sharing within and across communities
    within_community_shares = 0
    across_community_shares = 0
    
    # For each agent, check where they shared content
    for agent in model.agents:
        if not hasattr(agent, "shared_content") or not len(agent.shared_content) > 0:
            continue
            
        # Get agent's community
        agent_community = node_to_community.get(agent.pos, -1)
        
        # Get agent's followers
        followers = [a for a in model.agents if hasattr(a, "pos") and 
                    network.has_edge(agent.pos, a.pos)]
        
        for follower in followers:
            follower_community = node_to_community.get(follower.pos, -2)
            
            # Count shares within same community vs across communities
            if agent_community == follower_community:
                within_community_shares += 1
            else:
                across_community_shares += 1
    
    # Calculate ratio of within-community sharing
    total_shares = within_community_shares + across_community_shares
    if total_shares == 0:
        return 0
        
    # Echo chamber score: proportion of content shared within same community
    # Higher values indicate stronger echo chambers
    return within_community_shares / total_shares
