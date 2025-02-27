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
        super().__init__(seed=seed)
        
        if N <= 0:
            raise ValueError("Number of agents must be positive")
        if m_links >= N:
            raise ValueError("Number of edges must be less than number of nodes")
        
        self.num_agents = N
        self.m_links = m_links
        
        # Generate preference vectors first
        self.preference_vectors = [self.random_preferences() for _ in range(self.num_agents)]
        
        # Create social network with preference vectors
        self.social_media_platform = SocialMediaPlatform(self.num_agents, self.m_links)
        self.social_media_platform.social_network = Social_Network(
            self.num_agents, 
            self.m_links,
            self.preference_vectors
        )
        self.grid = NetworkGrid(self.social_media_platform.social_network.network)
        
        # Create agents and add them to the grid
        for i in range(self.num_agents):
            if i < int(0.05 * self.num_agents):  # First 5% are influencers
                user = InfluencerAgent(self, self.preference_vectors[i])
            elif i < int(0.10 * self.num_agents):  # Next 5% are bots
                user = BotAgent(self, self.preference_vectors[i])
            else:  # Rest are regular users
                credibility_level = min(max(random.gauss(0.5, 0.15), 0), 1)
                influence_level = min(max(random.gauss(0.3, 0.1), 0), 1)
                user = UserAgent(self, self.preference_vectors[i], credibility_level, influence_level)
            
            # Place agent in grid using integer node ID
            self.grid.place_agent(user, i)

        # Initialize news content
        self.news_content = self.initialize_news_content()

        # Distribute news to random agents
        for content in self.news_content:
            # Select random agent to receive the content
            random_agent = self.random.choice(list(self.agents))
            random_agent.feed.append(content)

        # Initialize datacollector with updated metrics for directed graph
        self.datacollector = DataCollector(
            model_reporters={
                "Number_of_Believers": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "B"),
                "Number_of_Susceptible": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "S"),
                "Number_of_Exposed": lambda m: sum(1 for a in m.agents if hasattr(a, "state") and a.state == "E"),
                "Network_Density": lambda m: nx.density(m.social_media_platform.social_network.network),
                "Average_Clustering": lambda m: nx.average_clustering(m.social_media_platform.social_network.network.to_undirected()),
                # Remove average_path_length as it requires strongly connected graph
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

    def step(self):
        """Advance the model by one step."""

        # Recommend news to agents
        self.social_media_platform.recommender.recommend_news(self.agents)

        # All agents evaluate their feed and engage with news
        self.agents.shuffle_do("step")  

        # Collect data
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

    def visualize_network(self):
        """Visualize the current state of the network"""
        # Create agent_types dictionary
        agent_types = {}
        for agent in self.agents:
            node_id = agent.pos
            if isinstance(agent, InfluencerAgent):
                agent_types[node_id] = 'influencer'
            elif isinstance(agent, BotAgent):
                agent_types[node_id] = 'bot'
            else:
                agent_types[node_id] = 'user'
        
        # Visualize the network
        self.social_media_platform.social_network.visualize_network(agent_types)

    def get_network_metrics(self):
        """Get detailed network metrics"""
        return self.social_media_platform.social_network.get_clustering_metrics()


if __name__ == "__main__":
    from utils.visualization import create_visualization
    # Create the visualization and assign it to 'page'
    page = create_visualization(FakeNewsModel)

    # Create and visualize model
    model = FakeNewsModel(N=100, m_links=10)
    model.social_media_platform.social_network.network.to_undirected()
    model.visualize_network()

    # Get detailed metrics
    metrics = model.get_network_metrics()
    print("\nDetailed Network Metrics:")
    print(f"Average Clustering: {metrics['average_clustering']:.3f}")
    print(f"Network Modularity: {metrics['modularity']:.3f}")
    print(f"Network Transitivity: {metrics['transitivity']:.3f}") 