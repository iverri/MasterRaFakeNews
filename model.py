from mesa import Model
import networkx as nx
from mesa.space import NetworkGrid
from agents.user_agent import BotAgent, InfluencerAgent, UserAgent
import random
from objects.news_content import generate_news_content
from utils.network_storage import NetworkStorage
from utils.visualization import project_info
from utils.model_utils import (
    distribute_news,
    random_preferences,
    generate_personalities,
)
from utils.datacollector import setup_datacollector
from objects.social_media_platform import SocialMediaPlatform
from typing import Optional, List


# Create a model class
class FakeNewsModel(Model):
    """
    This model simulates the spread of fake news in a social network.
    At each timestep, users receive a content feed, engage with news,
    and may transition from Susceptible (S) → Exposed (E) → Infected (I).
    Bots and influencers accelerate spread, while moderation reduces visibility.
    The process repeats over multiple timesteps, influencing network dynamics.

    Parameters:
    -----------
    N : int, default=200
        Number of agents in the simulation
    m_links : int, default=10
        Average number of connections per agent in the network
    news_amount : int, default=500
        Initial amount of news content to generate
    fake_news_percentage : int, default=10
        Percentage of news content that is fake (0-100)
    recommender_type : str, default="random"
        Type of recommendation algorithm to use
    bot_percentage : int, default=5
        Percentage of agents that are bots (0-100)
    influencer_percentage : int, default=5
        Percentage of agents that are influencers (0-100)
    diversity_level : float, default=0
        Level of diversity in recommendations (0-1)
    num_recommendations : int, default=10
        Number of recommendations to show each agent per step
    use_stored_network : bool, default=True
        Whether to use a pre-generated network structure
    stored_network : NetworkStorage, optional
        Pre-existing network storage object
    network_file : str, optional
        Path to stored network file for parallel processing
    seed : int, optional
        Random seed for reproducibility
    """

    # Initialize agents
    def __init__(
        self,
        N: int = 200,
        m_links: int = 10,
        news_amount: int = 500,
        fake_news_percentage: int = 10,
        recommender_type: str = "random",
        bot_percentage: int = 5,
        influencer_percentage: int = 5,
        diversity_level: float = 0,
        num_recommendations: int = 10,
        use_stored_network: bool = True,
        stored_network: Optional[NetworkStorage] = None,
        network_file: Optional[str] = None,
        seed: Optional[int] = None,
        max_steps: int = 100,
    ) -> None:
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
        self.news_amount = news_amount
        self.diversity_level = diversity_level
        self.num_recommendations = num_recommendations
        self.use_stored_network = use_stored_network
        self.network_file = network_file
        self.network_storage = (
            stored_network
            if use_stored_network and stored_network
            else NetworkStorage()
        )
        self.max_steps = max_steps

        # Generate preference vectors first (these might be replaced if using stored network)
        self.preference_vectors = [random_preferences() for _ in range(self.num_agents)]
        # self.personality_vectors = [
        #     generate_personalities() for _ in range(self.num_agents)
        # ]

        self.social_media_platform = SocialMediaPlatform(
            self,
            self.num_agents,
            self.m_links,
            self.preference_vectors,
#            self.personality_vectors,
            self.recommender_type,
            self.diversity_level,
            self.num_recommendations,
            self.use_stored_network,
            self.network_file,
        )

        # Setup grid for Mesa
        self._setup_grid()

        # Create and place agents
        self._create_agents()

        # Initialize news content
        self.news_content = generate_news_content(
            self.fake_news_percentage, self.news_amount, self.steps
        )

        # Distribute news to agents based on social network to get initial engagement
        distribute_news(self)

        # Initialize datacollector
        self.datacollector = setup_datacollector(self)

        # Add time-related properties
        self.hours_per_step = 3  # Each step represents 3 hours
        self.current_hour = 0  # Track the current hour (0-23)

    def step(self):
        """Advance the model by one step."""

        print(f"Step: {self.steps}")
        # Update the current hour
        self.current_hour = (self.current_hour + self.hours_per_step) % 24

        # Generate new content to ensure enough content for generating recommendations
        self.news_content.extend(
            generate_news_content(self.fake_news_percentage, 50, self.steps)
        )

        # Update engagement for all news content
        for content in self.news_content:
            content.update_engagement(self.steps)

        self.news_content = [
            content for content in self.news_content if content.engagement > 0.2
        ]

        # Update recommendations for all agents
        self.social_media_platform.recommender.update_recommendations(self.agents)

        # Let agents process their feed and recommendations
        self.agents.shuffle_do("step")

        # Collect data
        self.datacollector.collect(self)

    def _validate_parameters(self, N, m_links):
        """
        Validate model parameters to ensure they are within acceptable ranges.

        Parameters:
        -----------
        N : int
            Number of agents (must be positive)
        m_links : int
            Number of edges (must be less than number of nodes)

        Raises:
        -------
        ValueError
            If parameters are invalid
        """
        if N <= 0:
            raise ValueError("Number of agents must be positive")
        if m_links >= N:
            raise ValueError("Number of edges must be less than number of nodes")

    def _setup_grid(self):
        """
        Setup the Mesa grid using the social network structure.
        Creates a NetworkGrid from the directed social network graph.
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.num_agents))
        undirected_edges = list(
            self.social_media_platform.social_network.network.to_undirected().edges()
        )
        G.add_edges_from(undirected_edges)
        self.grid = NetworkGrid(G)

    def _create_agents(self):
        """
        Create and place agents in the grid according to their types.
        Agents are created based on their index position, with influencers
        having the lowest indices, followed by bots, then regular users.
        """

        for i in range(self.num_agents):
            user = self._create_agent_by_type(i)
            self.grid.place_agent(user, i)
            user.social_network = self.social_media_platform.social_network

    def _create_agent_by_type(self, index):
        """Create an agent based on its index/type."""
        if index < int(self.influencer_percentage * self.num_agents):  # Influencers
            return InfluencerAgent(
                self, self.preference_vectors[index], self.personality_vectors[index]
            )
        elif index < int(
            self.influencer_percentage * self.num_agents
            + self.bot_percentage * self.num_agents
        ):  # Bots
            return BotAgent(self, self.preference_vectors[index], [0, 0, 0, 0, 0])
        else:  # Regular users
            return UserAgent(
                self, self.preference_vectors[index], self.personality_vectors[index]
            )


if __name__ == "__main__":
    from utils.visualization import create_visualization

    # Create the visualization and assign it to 'page'
    page = create_visualization(FakeNewsModel)
