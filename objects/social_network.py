import networkx as nx

class Social_Network:

    def __init__(self, num_agents, m_links):
        self.num_agents=num_agents
        self.m_links=m_links

        # Barbasi Albert is an algorithm for creating random scale-free networks
        self.network = nx.barabasi_albert_graph(n=self.num_agents, m=self.m_links)

    def updateNetwork(self):
        # TODO create random changes to network for realism
        return
    
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
