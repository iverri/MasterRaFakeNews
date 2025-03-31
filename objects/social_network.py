import numpy as np
from utils.objects_utils import (
    create_preference_based_network,
    create_basic_network,
    adjust_node_connectivity,
)
from utils.metrics import get_clustering_metrics

class Social_Network:

    def __init__(self, model, num_agents, m_links, preference_vectors=None):
        self.model = model
        self.num_agents = num_agents
        self.m_links = m_links
        
        # Calculate number of each agent type
        self.num_influencers = int(model.influencer_percentage * num_agents)
        self.num_bots = int(model.bot_percentage * num_agents)
        self.num_regular = num_agents - (self.num_influencers + self.num_bots)
        
        # Create network with community structure based on preferences
        if preference_vectors is not None:
            self.network = create_preference_based_network(model, num_agents, m_links, preference_vectors)
        else:
            # Fallback to simpler network if no preferences provided
            self.network = create_basic_network(num_agents, m_links)
            
        # Ensure minimum connectivity for bots and high connectivity for influencers
        adjust_node_connectivity(self.network, self.num_agents, self.num_influencers, self.num_bots)


    def update_network(self, probability=0.1):
        """
        Update network structure to simulate real social network dynamics.
        
        Args:
            G (nx.Graph): The network graph
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

    def get_clustering_metrics(self):
        """Return detailed clustering metrics"""
        return get_clustering_metrics(self.network)
