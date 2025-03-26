from mesa.datacollection import DataCollector
import random
from utils.metrics import get_community_modularity
import networkx as nx
import numpy as np

#------------------------------------------------------------------------------
# PREFERENCE AND CONTENT GENERATION FUNCTIONS
#------------------------------------------------------------------------------

def random_preferences(model=None):
    """Generate random normalized preference vector."""
    preferences = [random.random() for i in range(10)]
    magnitude = sum(x*x for x in preferences) ** 0.5
    return [x/magnitude for x in preferences]

def generate_new_content(model):
    """Generate multiple new content pieces with some probability."""
    # Determine how many content pieces to generate (between 30-80)
    content_count = model.random.randint(30, 60)
    
    # Filter out content with low engagement instead of clearing everything
    model.news_content = [content for content in model.news_content 
                          if content.engagement >= 0.3]
    
    # Generate new content
    for _ in range(content_count):
        if model.random.random() < 0.8:  # 80% chance of new content
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

    # Distribute the newly generated content
    # distribute_news(model)

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
        content_sample_size = model.random.randint(1, min(5, len(model.news_content)))
        
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
            "Average_Feed_Size": lambda m: sum(len(a.feed) for a in m.agents if hasattr(a, "feed")) / len(m.agents) if len(m.agents) > 0 else 0,
            # New metrics for content-based recommendations
        },
        agent_reporters={
            "State": lambda a: getattr(a, "state", None),
            "Influence": lambda a: getattr(a, "influence_level", 0),
            "Followers": lambda a: a.social_media_platform.social_network.network.in_degree(a.pos),
            "Following": lambda a: a.social_media_platform.social_network.network.out_degree(a.pos),
            "Is_Active": lambda a: getattr(a, "is_active", False),
            "Activity_Probability": lambda a: getattr(a, "activity_probability", 0),
            "Feed_Size": lambda a: len(a.feed),
            # New agent metrics
        }
    )