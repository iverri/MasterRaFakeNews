import numpy as np
from utils.objects_utils import (
    create_preference_based_network,
    create_basic_network,
    adjust_node_connectivity,
    get_clustering_metrics,
)

class Social_Network:

    def __init__(self, num_agents, m_links, preference_vectors=None):
        self.num_agents = num_agents
        self.m_links = m_links
        
        # Calculate number of each agent type
        self.num_influencers = int(0.05 * num_agents)
        self.num_bots = int(0.05 * num_agents)
        self.num_regular = num_agents - (self.num_influencers + self.num_bots)
        
        # Create network with community structure based on preferences
        if preference_vectors is not None:
            self.network = create_preference_based_network(num_agents, m_links, preference_vectors)
        else:
            # Fallback to simpler network if no preferences provided
            self.network = create_basic_network(num_agents, m_links)
            
        # Ensure minimum connectivity for bots and high connectivity for influencers
        adjust_node_connectivity(self.network, self.num_agents, self.num_influencers, self.num_bots)


    def update_network(G, probability=0.1):
        """
        Update network structure to simulate real social network dynamics.
        
        Args:
            G (nx.Graph): The network graph
            probability (float): Probability of network change
        """
        if np.random.random() < probability:  # Default 10% chance of network change
            if np.random.random() < 0.5:
                # Add edge between nodes
                nodes = list(G.nodes())
                source = np.random.choice(nodes)
                # Prefer connecting similar nodes
                target = np.random.choice(nodes)
                if source != target and not G.has_edge(source, target):
                    G.add_edge(source, target)
            else:
                # Remove edge, preferentially between dissimilar nodes
                if G.edges():
                    edge = np.random.choice(list(G.edges()))
                    G.remove_edge(*edge)

    def get_clustering_metrics(self):
        """Return detailed clustering metrics"""
        return get_clustering_metrics(self.network)
