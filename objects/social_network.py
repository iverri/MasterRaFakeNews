import numpy as np
from utils.objects_utils import (
    create_preference_based_network,
    create_basic_network,
)
from utils.network_storage import NetworkStorage
import os


class Social_Network:

    def __init__(
        self,
        model,
        num_agents,
        m_links,
        preference_vectors=None,
        personality_vectors=None,
        use_stored_network=True,
        network_file=None,
    ):
        self.model = model
        self.num_agents = num_agents
        self.m_links = m_links

        # Calculate number of each agent type
        self.num_influencers = int(model.influencer_percentage * num_agents)
        self.num_bots = int(model.bot_percentage * num_agents)
        self.num_regular = num_agents - (self.num_influencers + self.num_bots)

        # Check if we should use a stored network
        storage = model.network_storage

        if use_stored_network:
            if network_file and os.path.exists(network_file):
                # Load network from file for parallel processing
                self.network, stored_preferences = (
                    NetworkStorage.load_network_from_file(network_file)
                )
                print(f"Using network from file: {network_file}")

                # If we're using a stored network, we might want to use stored preference vectors too
                if (
                    stored_preferences is not None
                    and len(stored_preferences) == num_agents
                ):
                    # Replace the model's preference vectors with stored ones
                    model.preference_vectors = stored_preferences

            elif storage.has_stored_network():
                # Use the in-memory stored network (for non-parallel usage)
                self.network = storage.get_network()
                print("Using stored network from memory")

                # If we're using a stored network, we might want to use stored preference vectors too
                stored_preferences = storage.get_preference_vectors()
                if (
                    stored_preferences is not None
                    and len(stored_preferences) == num_agents
                ):
                    # Replace the model's preference vectors with stored ones
                    model.preference_vectors = stored_preferences
            else:
                # Create a new network if no stored network is available
                self._create_new_network(model, num_agents, m_links, preference_vectors)
        else:
            # Create a new network
            self._create_new_network(model, num_agents, m_links, preference_vectors, personality_vectors)

    def _create_new_network(self, model, num_agents, m_links, preference_vectors, personality_vectors):
        """Create a new network and store it"""
        if preference_vectors is not None:
            self.network = create_preference_based_network(
                model, num_agents, m_links, preference_vectors, personality_vectors
            )
        else:
            # Fallback to simpler network if no preferences provided
            self.network = create_basic_network(num_agents, m_links)

        # Always store the network for future use
        model.network_storage.store_network(self.network, model.preference_vectors)

    # Not in use
    def update_network(self, probability=0.1):
        """
        Update network structure to simulate real social network dynamics.

        Args:
            probability (float): Probability of network change
        """

        if np.random.random() < probability:  # Default 10% chance of network change
            if np.random.random() < 0.5:
                # Add edge between nodes
                nodes = list(self.network.nodes())
                source = np.random.choice(nodes)
                # Prefer connecting similar nodes
                target = np.random.choice(nodes)
                if source != target and not self.network.has_edge(source, target):
                    self.network.add_edge(source, target)
            else:
                # Remove edge, preferentially between dissimilar nodes
                edges = list(self.network.edges())
                if edges:
                    # Choose a random index instead of using np.random.choice on the edges list
                    random_index = np.random.randint(0, len(edges))
                    edge = edges[random_index]
                    self.network.remove_edge(*edge)
