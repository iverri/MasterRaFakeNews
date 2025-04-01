from mesa.datacollection import DataCollector
import random
from utils.metrics import get_community_modularity
import networkx as nx
import numpy as np
from utils.network_storage import NetworkStorage

#------------------------------------------------------------------------------
# PREFERENCE AND CONTENT GENERATION FUNCTIONS
#------------------------------------------------------------------------------

def random_preferences(model=None):
    """Generate random normalized preference vector."""
    # Generate random preferences with 15 dimensions
    preferences = [random.random() for i in range(15)]
    
    # Normalize the vector (convert to unit vector)
    magnitude = sum(x**2 for x in preferences)**0.5
    if magnitude > 0:  # Avoid division by zero
        preferences = [x/magnitude for x in preferences]
    
    return preferences

def generate_new_content(model):
    """Generate multiple new content pieces with some probability."""
    # Determine how many content pieces to generate (between 30-80)
    content_count = model.random.randint(50, 100)
    
    # Filter out content with low engagement instead of clearing everything
    model.news_content = [content for content in model.news_content 
                          if content.engagement >= 0.3]
    
    # Generate new content
    for _ in range(content_count):
        if model.random.random() < 0.9:  # 80% chance of new content
            # Import inside function to avoid circular import
            from objects.news_content import NewsContent
            
            # Use model's fake_news_percentage parameter
            is_fake = model.random.random() < model.fake_news_percentage
            
            new_content = NewsContent(
                len(model.news_content), 
                is_fake,  # Now using the model parameter
                random_preferences(model)
            )
            # Set creation step to current model step
            new_content.creation_step = model.steps
            model.news_content.append(new_content)

    # Update engagement for all news content
    for content in model.news_content:
        content.update_engagement(model.steps)

def distribute_news(model):
    """Distribute news content to agents.
    
    Distributes content from the model's news_content pool to agents.
    Each agent receives a random sample of the content with varying sizes.
    """
    # If no content to distribute, return early
    if not model.news_content:
        return
    
    # Distribute content to all agents
    for agent in model.agents:
        # Determine a random number of content pieces for this agent
        content_sample_size = model.random.randint(10, min(15, len(model.news_content)))
        
        # Select random content from the content pool
        content_sample = model.random.sample(
            model.news_content, 
            content_sample_size
        )
        
        # Add the content to the agent's feed
        agent.feed.extend(content_sample)

#------------------------------------------------------------------------------
# AGENT AND NETWORK ANALYSIS FUNCTIONS
#------------------------------------------------------------------------------

def get_agent_types(model):
    """Get a dictionary mapping node IDs to agent types"""
    # Import inside function to avoid circular import
    from agents.user_agent import InfluencerAgent, BotAgent
    
    agent_types = {}
    for agent in model.agents:
        node_id = agent.pos
        if isinstance(agent, InfluencerAgent):
            agent_types[node_id] = 'influencer'
        elif isinstance(agent, BotAgent):
            agent_types[node_id] = 'bot'
        else:
            agent_types[node_id] = 'user'
    return agent_types

#------------------------------------------------------------------------------
# DATA COLLECTION FUNCTIONS
#------------------------------------------------------------------------------

def setup_datacollector(model):
    """Initialize the datacollector with metrics."""
    return DataCollector(
        model_reporters={
            "Number_of_Infected": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "I"),
            "Number_of_Susceptible": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "S"),
            "Number_of_Exposed": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "E"),
            "Network_Density": lambda m: nx.density(m.social_media_platform.social_network.network),
            "Average_Clustering": lambda m: nx.average_clustering(m.social_media_platform.social_network.network.to_undirected()),
            "In_Degree_Centrality": lambda m: nx.in_degree_centrality(m.social_media_platform.social_network.network),
            "Out_Degree_Centrality": lambda m: nx.out_degree_centrality(m.social_media_platform.social_network.network),
            "Community_Modularity": lambda m: get_community_modularity(m.social_media_platform.social_network.network.to_undirected()),
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
        },
        agent_reporters={
            "State": lambda a: getattr(a, "state", None),
            "Influence": lambda a: getattr(a, "influence_level", 0),
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

def clear_stored_network():
    """Clear any stored network"""
    NetworkStorage().clear()
    print("Cleared stored network")

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
    exposed_agents = sum(1 for a in model.agents if hasattr(a, "state") and (a.state == "E" or a.state == "I"))
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
    
    from utils.common import cosine_similarity
    
    # Calculate similarity between agent preferences and each content item
    similarities = [cosine_similarity(agent.preference_vector, content.topic_vector) 
                   for content in agent.recommended_content]
    
    # Higher average similarity indicates stronger echo chamber
    return sum(similarities) / len(similarities) if similarities else 0

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
        if not hasattr(agent, "shared_content") or not agent.shared_content:
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
