from mesa import Model, Agent
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from numpy import dot
from numpy.linalg import norm
import networkx as nx
from agents.user_agent import BotAgent, InfluencerAgent, UserAgent
from objects.news_content import NewsContent
from recommender.recommender import Recommender
import random
from objects.social_network import Social_Network
from utils.metrics import (get_network_metrics, get_community_modularity )


class SocialMediaPlatform:
    def __init__(self, num_agents, m_links):
        self.recommender = Recommender() # TODO: Implement recommender
        self.social_network = Social_Network(num_agents, m_links)


# Create a model class
class FakeNewsModel(Model):
    '''This model simulates the spread of fake news in a social network.  
    At each timestep, users receive a content feed, engage with news,  
    and may transition from Susceptible (S) → Exposed (E) → Believer (B).  
    Bots and influencers accelerate spread, while moderation reduces visibility.  
    The process repeats over multiple timesteps, influencing network dynamics. 
    '''

    #Initialize number of agents
    #Initialize agents
    def __init__(self, N: int = 5, m_links: int = 1, seed: int = None):
        """Initialize the Fake News Model."""
        super().__init__(seed=seed)  # Required in Mesa 3.0
        
        if N <= 0:
            raise ValueError("Number of agents must be positive")
        if m_links >= N:
            raise ValueError("Number of edges must be less than number of nodes")
        
        self.num_agents = N
        self.m_links = m_links
        self.social_media_platform = SocialMediaPlatform(self.num_agents, self.m_links)
        self.grid = NetworkGrid(self.social_media_platform.social_network.network)
        
        # Create agents and add them to the grid
        for i in range(self.num_agents):
            preference_vector = self.random_preferences()
            
            if i % 5 == 0:  # every 5th agent is a bot
                user = BotAgent(self, preference_vector)
            elif i % 6 == 0:  # every 6th agent is an influencer
                user = InfluencerAgent(self, preference_vector)
            else:
                credibility_level = min(max(random.gauss(0.5, 0.15), 0), 1)
                influence_level = min(max(random.gauss(0.3, 0.1), 0), 1)
                user = UserAgent(self, preference_vector, credibility_level, influence_level)
            
            # Place agent in grid using integer node ID
            self.grid.place_agent(user, i)

        # Initialize news content
        self.news_content = self.initialize_news_content()

        # Distribute news to random agents
        for content in self.news_content:
            # Select random agent to receive the content
            random_agent = self.random.choice(list(self.agents))
            random_agent.feed.append(content)

        # Initialize datacollector
        self.datacollector = DataCollector(
            model_reporters={
                "Number_of_Believers": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "B"),
                "Number_of_Susceptible": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "S"),
                "Number_of_Exposed": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "E"),
                "Network_Density": lambda m: nx.density(m.social_media_platform.social_network.network),
                "Average_Clustering": lambda m: nx.average_clustering(m.social_media_platform.social_network.network),
                "Average_Path_Length": lambda m: nx.average_shortest_path_length(m.social_media_platform.social_network.network),
                "Degree_Centrality": lambda m: nx.degree_centrality(m.social_media_platform.social_network.network),
                "Community_Modularity": lambda m: get_community_modularity(m.social_media_platform.social_network.network)
            },
            agent_reporters={
                "State": lambda a: getattr(a, "state", None),
                "Influence": lambda a: getattr(a, "influence_level", 0)
            }
        )

    def step(self):
        """Advance the model by one step."""
        self.agents.shuffle_do("step")  # Replace scheduler with agents.shuffle_do
        self.datacollector.collect(self)

    def random_preferences(self):
        preferences = [random.random() for i in range(3)]
        # Normalize the vector
        magnitude = sum(x*x for x in preferences) ** 0.5
        return [x/magnitude for x in preferences]
    
    def get_metrics(self):
        """Get network metrics for the current state"""
        return get_network_metrics(self.social_media_platform.social_network.network)

    def initialize_news_content(self):
        """Create a mix of real and fake news content"""
        news_items = []
        for i in range(10):  # Create 10 news items
            # Create topic vector similar to preference vectors
            topic_vector = self.random_preferences()
            # Alternate between real and fake news
            is_fake = i % 5 == 0
            news_items.append(NewsContent(i, is_fake, topic_vector))
        return news_items



if __name__ == "__main__":
    from utils.visualization import create_visualization
    # Create the visualization and assign it to 'page'
    page = create_visualization(FakeNewsModel) 