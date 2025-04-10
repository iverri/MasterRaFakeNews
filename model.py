from mesa import Model
import networkx as nx
from mesa.space import NetworkGrid
from agents.user_agent import BotAgent, InfluencerAgent, UserAgent
from recommender.recommender import Recommender
import random
from objects.news_content import NewsContent, generate_news_content
from objects.social_network import Social_Network
from utils.visualization import project_info
from utils.model_utils import (
    distribute_news,
    random_preferences,
    setup_datacollector,
    get_agent_types
)
from objects.social_media_platform import SocialMediaPlatform

# Create a model class
class FakeNewsModel(Model):
    '''This model simulates the spread of fake news in a social network.  
    At each timestep, users receive a content feed, engage with news,  
    and may transition from Susceptible (S) → Exposed (E) → Infected (I).  
    Bots and influencers accelerate spread, while moderation reduces visibility.  
    The process repeats over multiple timesteps, influencing network dynamics. 
    '''
    #Initialize agents
    def __init__(self, N=100, m_links=10, news_amount=500, fake_news_percentage=10, 
                 recommender_type="random", bot_percentage=5, influencer_percentage=5, 
                 diversity_lambda=0.1, increase_diversity=False, num_recommendations=10, use_stored_network=False, 
                 seed: int = None):
        """Initialize the Fake News Model."""
        super().__init__(seed=seed)
        
        # Validate parameters
        self._validate_parameters(N, m_links)
        
        # Set model parameters
        self.info = project_info
        self.num_agents = N
        self.m_links = m_links
        self.fake_news_percentage = fake_news_percentage / 100
        self.recommender_type = recommender_type
        self.bot_percentage = bot_percentage / 100
        self.influencer_percentage = influencer_percentage / 100
        self.diversity_lambda = diversity_lambda
        self.news_amount = news_amount
        self.increase_diversity = increase_diversity
        self.num_recommendations = num_recommendations
        self.use_stored_network = use_stored_network
        
        # Generate preference vectors first (these might be replaced if using stored network)
        self.preference_vectors = [random_preferences() for _ in range(self.num_agents)]

        self.social_media_platform = SocialMediaPlatform(
            self, self.num_agents, self.m_links, self.preference_vectors, 
            self.recommender_type, self.increase_diversity, 
            self.num_recommendations, self.use_stored_network
        )
        
        # Setup grid for Mesa
        self._setup_grid()
        
        # Create and place agents
        self._create_agents()

        # Initialize news content
        self.news_content = generate_news_content(self.fake_news_percentage, self.news_amount, self.steps)

        # Distribute news to agents based on social network to get initial engagement
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

        # Update network
        # self.social_media_platform.social_network.update_network()
        
        # Generate new content
        # TODO: remove when added functionality for users to post content
        self.news_content.extend(generate_news_content(self.fake_news_percentage, 50, self.steps))

        # Update engagement for all news content
        for content in self.news_content:
            content.update_engagement(self.steps)

        self.news_content = [content for content in self.news_content if content.engagement > 0.2]
        
        fake_news_items = [item for item in self.news_content if item.isFake]
        print(f"Fake news items: {len(fake_news_items)} of {len(self.news_content)}, percentage: {len(fake_news_items) / len(self.news_content)}")
        # Update recommendations for all agents
        self.social_media_platform.recommender.update_recommendations(self.agents)
        
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
        if index < int(self.influencer_percentage * self.num_agents):  # Influencers
            return InfluencerAgent(self, self.preference_vectors[index])
        elif index < int(self.influencer_percentage * self.num_agents + self.bot_percentage * self.num_agents):  # Bots
            return BotAgent(self, self.preference_vectors[index])
        else:  # Regular users
            naivety_level = min(max(random.gauss(0.5, 0.15), 0), 1)
            return UserAgent(self, self.preference_vectors[index], naivety_level)

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

    def agents_post_content(self):
        """Allow agents to generate and post new content."""
        for agent in self.agents:
            agent.post_content()

if __name__ == "__main__":
    from utils.visualization import create_visualization
    # Create the visualization and assign it to 'page'
    page = create_visualization(FakeNewsModel)