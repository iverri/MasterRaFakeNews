from mesa.datacollection import DataCollector
import random
from objects.news_content import NewsContent
from utils.metrics import get_community_modularity
import networkx as nx

def random_preferences(model=None):
    """Generate random normalized preference vector."""
    preferences = [random.random() for i in range(3)]
    magnitude = sum(x*x for x in preferences) ** 0.5
    return [x/magnitude for x in preferences]

def initialize_news_content(model):
    """Create a mix of real and fake news content."""
    news_items = []
    for i in range(200):  
        topic_vector = random_preferences(model)
        is_fake = i % 5 == 0
        news_items.append(NewsContent(i, is_fake, topic_vector))
    return news_items
    
def distribute_initial_news(model):
    """Distribute news content to seed agents."""
    seed_agents = model.random.sample(list(model.agents), min(5, len(model.agents)))
    
    for content in model.news_content:
        seed_agent = model.random.choice(seed_agents)
        seed_agent.feed.append(content)

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