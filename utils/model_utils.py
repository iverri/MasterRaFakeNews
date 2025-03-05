from mesa.datacollection import DataCollector
import random
from utils.metrics import get_community_modularity
import networkx as nx

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
    # Determine how many content pieces to generate (between 10-20)
    content_count = model.random.randint(10, 20)
    # Clear the existing content pool
    model.news_content = []
    
    for _ in range(content_count):
        if model.random.random() < 0.8:  # 80% chance of new content
            # Import inside function to avoid circular import
            from objects.news_content import NewsContent
            
            new_content = NewsContent(
                len(model.news_content), 
                model.random.random() < 0.2,  # 20% chance of fake news
                random_preferences(model)
            )
            model.news_content.append(new_content)
    
    # Distribute the newly generated content
    distribute_news(model)

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
            "Number_of_Believers": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "B"),
            "Number_of_Susceptible": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "S"),
            "Number_of_Exposed": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "E"),
            "Network_Density": lambda m: nx.density(m.social_media_platform.social_network.network),
            "Average_Clustering": lambda m: nx.average_clustering(m.social_media_platform.social_network.network.to_undirected()),
            "In_Degree_Centrality": lambda m: nx.in_degree_centrality(m.social_media_platform.social_network.network),
            "Out_Degree_Centrality": lambda m: nx.out_degree_centrality(m.social_media_platform.social_network.network),
            "Community_Modularity": lambda m: get_community_modularity(m.social_media_platform.social_network.network.to_undirected())
        },
        agent_reporters={
            "State": lambda a: getattr(a, "state", None),
            "Influence": lambda a: getattr(a, "influence_level", 0),
            "Followers": lambda a: a.social_media_platform.social_network.network.in_degree(a.pos),
            "Following": lambda a: a.social_media_platform.social_network.network.out_degree(a.pos)
        }
    )