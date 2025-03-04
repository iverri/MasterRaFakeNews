from mesa import Model
import networkx as nx
from mesa.space import NetworkGrid
from agents.user_agent import BotAgent, InfluencerAgent, UserAgent
from objects.news_content import NewsContent
import random
from objects.social_network import Social_Network
from utils.visualization import project_info
from utils.model_utils import (
    initialize_news_content,
    distribute_initial_news,
    random_preferences,
    setup_datacollector
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

    #Initialize number of agents
    #Initialize agents
    def __init__(self, N: int = 5, m_links: int = 1, seed: int = None):
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
        self.news_content = initialize_news_content(self)

        # Distribute news to agents based on social network
        distribute_initial_news(self)

        # Initialize datacollector
        self.datacollector = setup_datacollector(self)

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

    def step(self):
        """Advance the model by one step."""
        self._generate_new_content()
        self.agents.shuffle_do("step")  
        self.datacollector.collect(self)
        
    def _generate_new_content(self):
        """Generate new content with some probability."""
        if self.random.random() < 0.5:  # 50% chance of new content
            new_content = NewsContent(
                len(self.news_content), 
                self.random.random() < 0.3,  # 30% chance of fake news
                random_preferences(self)
            )
            self.news_content.append(new_content)
            self._distribute_new_content(new_content)
            
    def _distribute_new_content(self, content):
        """Distribute new content to agents."""
        for agent in self.agents:
            if self.random.random() < 0.8:  # 80% chance to receive
                agent.feed.append(content)

    def visualize_network(self):
        """Visualize the current state of the network"""
        # Create agent_types dictionary
        agent_types = self.get_agent_types()
        
        # Import the visualization function from utils.visualization
        from utils.visualization import visualize_network
        
        # Visualize the network
        visualize_network(self.social_media_platform.social_network.network, agent_types)

    def get_agent_types(self):
        """Get a dictionary mapping node IDs to agent types"""
        agent_types = {}
        for agent in self.agents:
            node_id = agent.pos
            if isinstance(agent, InfluencerAgent):
                agent_types[node_id] = 'influencer'
            elif isinstance(agent, BotAgent):
                agent_types[node_id] = 'bot'
            else:
                agent_types[node_id] = 'user'
        return agent_types

    def get_network_metrics(self):
        """Get detailed network metrics"""
        return self.social_media_platform.social_network.get_clustering_metrics()


if __name__ == "__main__":
    from utils.visualization import create_visualization
    # Create the visualization and assign it to 'page'
    page = create_visualization(FakeNewsModel)
