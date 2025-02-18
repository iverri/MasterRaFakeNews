import networkx as nx
import random

class Social_Network:

    def __init__(self, num_agents, m_links):
        self.num_agents=num_agents
        self.m_links=m_links

        # Barbasi Albert is an algorithm for creating random scale-free networks
        self.network = nx.barabasi_albert_graph(n=self.num_agents, m=self.m_links)
    
    def updateNetwork(self):
        """Update network structure to simulate real social network dynamics"""
        # Randomly add/remove edges with small probability
        if random.random() < 0.1:  # 10% chance of network change
            if random.random() < 0.5:
                # Add edge
                nodes = list(self.social_media_platform.social_network.network.nodes())
                source = random.choice(nodes)
                target = random.choice(nodes)
                if source != target and not self.social_media_platform.social_network.network.has_edge(source, target):
                    self.social_media_platform.social_network.network.add_edge(source, target)
            else:
                # Remove edge
                if self.social_media_platform.social_network.network.edges():
                    edge = random.choice(list(self.social_media_platform.social_network.network.edges()))
                    self.social_media_platform.social_network.network.remove_edge(*edge)
    
    def create_community_network(num_users, num_communities=3, intra_prob=0.3, inter_prob=0.05):
        """
        Generates a network with community structure.
        
        - num_users: Total agents
        - num_communities: Number of distinct communities
        - intra_prob: Probability of connecting within the same community
        - inter_prob: Probability of connecting between communities
        """
        sizes = [num_users // num_communities] * num_communities
        probs = [[intra_prob if i == j else inter_prob for j in range(num_communities)] for i in range(num_communities)]

        G = nx.stochastic_block_model(sizes, probs, seed=42)
        return G
