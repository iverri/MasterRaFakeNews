from mesa import Model
import networkx as nx
from mesa.space import NetworkGrid
from agents.user_agent import BotAgent, InfluencerAgent, UserAgent
from recommender.recommender import Recommender
from objects.news_content import NewsContent
import random
from objects.news_content import NewsContent, initialize_news_content
from objects.social_network import Social_Network
from utils.visualization import project_info
from utils.model_utils import (
    distribute_news,
    random_preferences,
    setup_datacollector,
    generate_new_content,
    get_agent_types
)
from objects.social_media_platform import SocialMediaPlatform

# Create a model class
class FakeNewsModel(Model):
    '''This model simulates the spread of fake news in a social network.  
    At each timestep, users receive a content feed, engage with news,  
    and may transition from Susceptible (S) → Exposed (E) → Believer (B).  
    Bots and influencers accelerate spread, while moderation reduces visibility.  
    The process repeats over multiple timesteps, influencing network dynamics. 
    '''

    #Initialize agents
    def __init__(self, N: int = 10, m_links: int = 2, seed: int = None, news_amount: int = 200):
        """Initialize the Fake News Model."""
        super().__init__(seed=seed)
        
        self._validate_parameters(N, m_links)
        self.info = project_info
        self.num_agents = N
        self.m_links = m_links
        
        # Generate preference vectors first
        self.preference_vectors = [random_preferences() for _ in range(self.num_agents)]
        
        # Setup social network
        self._setup_social_network()
        
        # Setup grid for Mesa
        self._setup_grid()
        
        # Create and place agents
        self._create_agents()

        # Initialize news content
        self.news_content = initialize_news_content(self, news_amount)

        # Initialize recommender system (starts with random recommendations)
        self.recommender = Recommender(type="collaborative_filtering")

        # Distribute news to agents based on social network
        distribute_news(self)

        # Initialize datacollector
        self.datacollector = setup_datacollector(self)

        # Add time-related properties
        self.hours_per_step = 3  # Each step represents 3 hours
        self.current_hour = 0  # Track the current hour (0-23)

    def step(self):
        """Advance the model by one step."""
        # Update the current hour
        self.current_hour = (self.current_hour + self.hours_per_step) % 24
        
        # Generate new content
        generate_new_content(self)
        
        # Update recommendations for all agents
        self.recommender.update_recommendations(self.agents)
        
        # Let agents process their feed and recommendations
        self.agents.shuffle_do("step")
        
        # Collect data
        self.datacollector.collect(self)

    def _validate_parameters(self, N, m_links):
        """Validate model parameters."""
        if N <= 0:
            raise ValueError("Number of agents must be positive")
        if m_links >= N:
            raise ValueError("Number of edges must be less than number of nodes")
            
    def _setup_social_network(self):
        """Setup the social network."""
        self.social_media_platform = SocialMediaPlatform(self.num_agents, self.m_links)
        self.social_media_platform.social_network = Social_Network(
            self.num_agents, 
            self.m_links,
            self.preference_vectors
        )
        
    def _setup_grid(self):
        """Setup the grid for Mesa."""
        G = nx.Graph()
        G.add_nodes_from(range(self.num_agents))
        undirected_edges = list(self.social_media_platform.social_network.network.to_undirected().edges())
        G.add_edges_from(undirected_edges)
        self.grid = NetworkGrid(G)
        
    def _create_agents(self):
        """Create and place agents in the grid."""
        for i in range(self.num_agents):
            user = self._create_agent_by_type(i)
            self.grid.place_agent(user, i)
            user.social_network = self.social_media_platform.social_network
            
    def _create_agent_by_type(self, index):
        """Create an agent based on its index/type."""
        if index < int(0.05 * self.num_agents):  # Influencers
            return InfluencerAgent(self, self.preference_vectors[index])
        elif index < int(0.10 * self.num_agents):  # Bots
            return BotAgent(self, self.preference_vectors[index])
        else:  # Regular users
            credibility_level = min(max(random.gauss(0.5, 0.15), 0), 1)
            influence_level = min(max(random.gauss(0.3, 0.1), 0), 1)
            return UserAgent(self, self.preference_vectors[index], credibility_level, influence_level)

    def visualize_network(self):
        """Visualize the current state of the network"""
        # Create agent_types dictionary
        agent_types = get_agent_types(self)
        
        # Import the visualization function from utils.visualization
        from utils.visualization import visualize_network
        
        # Visualize the network
        visualize_network(self.social_media_platform.social_network.network, agent_types)

    def get_network_metrics(self):
        """Get detailed network metrics"""
        return self.social_media_platform.social_network.get_clustering_metrics()

if __name__ == "__main__":
    from utils.visualization import create_visualization
    # Create the visualization and assign it to 'page'
    page = create_visualization(FakeNewsModel)