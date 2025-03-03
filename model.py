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
from utils.visualization import project_info

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
        self.info = project_info
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
        
        # Create a simple UNDIRECTED graph for Mesa's internal mechanisms
        # This is just a placeholder grid - we'll use our own network for actual connections
        G = nx.Graph()  # Undirected graph just for Mesa's grid
        G.add_nodes_from(range(self.num_agents))
        
        # Add edges from our social network (convert to undirected for visualization)
        undirected_edges = list(self.social_media_platform.social_network.network.to_undirected().edges())
        G.add_edges_from(undirected_edges)
        
        self.grid = NetworkGrid(G)
        
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
            
            # Store a reference to the social network in each agent
            user.social_network = self.social_media_platform.social_network

        # Initialize news content
        self.news_content = self.initialize_news_content()

        # Distribute news to agents based on social network
        self.distribute_initial_news()

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
        # Generate some new content each step (increased probability)
        if self.random.random() < 0.5:  # 50% chance of new content each step
            new_content = NewsContent(
                len(self.news_content), 
                self.random.random() < 0.3,  # 30% chance of fake news
                self.random_preferences()
            )
            self.news_content.append(new_content)
            
            # Equal probability for all agents to receive new content
            for agent in self.agents:
                # Each agent has the same base probability to receive new content
                seed_probability = 0.8  # 20% chance for any agent
                
                # Determine if this agent receives the content
                if self.random.random() < seed_probability:
                    agent.feed.append(new_content)

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
        for i in range(200):  
            # Create topic vector similar to preference vectors
            topic_vector = self.random_preferences()
            # Alternate between real and fake news
            is_fake = i % 5 == 0
            news_items.append(NewsContent(i, is_fake, topic_vector))
        return news_items
        
    def distribute_initial_news(self):
        """Distribute news content to agents based on the social network structure"""
        # Select a few seed agents to receive initial news
        seed_agents = self.random.sample(list(self.agents), min(5, len(self.agents)))
        
        # Distribute news randomly among seed agents
        for content in self.news_content:
            seed_agent = self.random.choice(seed_agents)
            seed_agent.feed.append(content)

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
