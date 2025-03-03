import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import community  # python-louvain package

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
            self.network = self._create_preference_based_network(preference_vectors)
        else:
            # Fallback to simpler network if no preferences provided
            self.network = self._create_basic_network()
            
        # Ensure minimum connectivity for bots and high connectivity for influencers
        self._adjust_node_connectivity()

    def _create_preference_based_network(self, preference_vectors):
        """Create directed network with communities based on preference similarity"""
        # Initialize empty directed graph
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_agents))
        
        # Calculate similarity matrix between all pairs of nodes
        similarity_matrix = np.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                sim = np.dot(preference_vectors[i], preference_vectors[j])
                similarity_matrix[i,j] = similarity_matrix[j,i] = sim
        
        # First, handle regular users and bots following others
        for i in range(self.num_agents):
            if i < self.num_influencers:
                continue  # Handle influencers' connections separately
            
            # Number of outgoing connections for this node
            if i >= self.num_agents - self.num_bots:
                k = min(int(self.m_links * 3), self.num_agents-1)  # More outgoing for bots
            else:
                k = self.m_links  # Regular users
            
            # Get potential nodes to follow based on preference similarity
            potential_edges = []
            
            # Higher probability to follow influencers
            for j in range(self.num_influencers):
                sim = similarity_matrix[i,j] * 1.5  # Boost similarity for influencers
                potential_edges.append((j, sim))
            
            # Regular preference-based edges for other users
            for j in range(self.num_influencers, self.num_agents):
                if j != i:
                    potential_edges.append((j, similarity_matrix[i,j]))
            
            potential_edges.sort(key=lambda x: x[1], reverse=True)
            
            # Add directed edges (i follows j)
            edges_added = 0
            for j, sim in potential_edges:
                if edges_added >= k:
                    break
                
                if not G.has_edge(i, j):
                    # Higher probability for bots to follow others
                    if i >= self.num_agents - self.num_bots:
                        prob = min(sim * 1.5, 1.0)
                    else:
                        prob = sim
                    
                    if np.random.random() < prob:
                        G.add_edge(i, j)
                        edges_added += 1
        
        # Now handle influencers' outgoing connections (they follow fewer accounts)
        for i in range(self.num_influencers):
            k = max(int(self.m_links * 0.5), 1)  # Fewer outgoing for influencers
            
            # Influencers are more likely to follow other influencers
            for j in range(self.num_influencers):
                if i != j:
                    if np.random.random() < similarity_matrix[i,j]:
                        G.add_edge(i, j)
        
        # Ensure the graph is weakly connected
        if not nx.is_weakly_connected(G):
            components = list(nx.weakly_connected_components(G))
            for i in range(len(components)-1):
                node1 = list(components[i])[0]
                node2 = list(components[i+1])[0]
                G.add_edge(node1, node2)
                
        return G

    def _create_basic_network(self):
        """Fallback method using Barabási-Albert model"""
        return nx.barabasi_albert_graph(n=self.num_agents, m=self.m_links)

    def _adjust_node_connectivity(self):
        """Adjust network to ensure proper connectivity for different agent types"""
        # Get node degrees
        degrees = dict(self.network.degree())
        nodes_by_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        
        # Ensure influencers have high connectivity
        for i in range(self.num_influencers):
            old_node = nodes_by_degree[i][0]
            if i >= self.num_influencers:  # If high-degree node isn't an influencer
                # Swap with a lower degree node that should be an influencer
                new_node = i
                self._swap_node_connections(old_node, new_node)
        
        # Ensure bots have low connectivity
        bot_start_idx = self.num_agents - self.num_bots
        for i in range(bot_start_idx, self.num_agents):
            old_node = nodes_by_degree[-i][0]
            if i < bot_start_idx:  # If low-degree node isn't a bot
                # Swap with a higher degree node that should be a bot
                new_node = i
                self._swap_node_connections(old_node, new_node)

    def _swap_node_connections(self, node1, node2):
        """Swap all connections between two nodes"""
        if node1 == node2:
            return
            
        # Get neighbors of both nodes
        node1_neighbors = set(self.network.neighbors(node1))
        node2_neighbors = set(self.network.neighbors(node2))
        
        # Remove all edges of both nodes
        self.network.remove_edges_from([(node1, n) for n in node1_neighbors])
        self.network.remove_edges_from([(node2, n) for n in node2_neighbors])
        
        # Add swapped edges
        self.network.add_edges_from([(node2, n) for n in node1_neighbors])
        self.network.add_edges_from([(node1, n) for n in node2_neighbors])

    def updateNetwork(self):
        """Update network structure to simulate real social network dynamics"""
        if np.random.random() < 0.1:  # 10% chance of network change
            if np.random.random() < 0.5:
                # Add edge between similar nodes
                nodes = list(self.network.nodes())
                source = np.random.choice(nodes)
                # Prefer connecting similar nodes
                target = np.random.choice(nodes)
                if source != target and not self.network.has_edge(source, target):
                    self.network.add_edge(source, target)
            else:
                # Remove edge, preferentially between dissimilar nodes
                if self.network.edges():
                    edge = np.random.choice(list(self.network.edges()))
                    self.network.remove_edge(*edge)

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

    def visualize_network(self, agent_types=None):
        """
        Visualize the network with communities and agent types
        
        Parameters:
        - agent_types: dictionary mapping node ids to agent types ('influencer', 'bot', 'user')
        """
        plt.figure(figsize=(12, 8))
        
        # Convert to undirected graph for community detection
        undirected_network = self.network.to_undirected()
        
        # Detect communities using Louvain method on undirected graph
        communities = community.best_partition(undirected_network)
        
        # Get position layout that groups communities together
        pos = nx.spring_layout(self.network)
        
        # Define colors for agent types - using lighter colors
        type_colors = {
            'influencer': '#d057d9',  
            'bot': '#53b028',        
            'user': '#4e6ac7'    
        }
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        
        for node in self.network.nodes():
            # Set node size based on agent type and in-degree (number of followers)
            if agent_types:
                base_size = 100
                in_degree = self.network.in_degree(node)
                in_degree_factor = min(in_degree / (self.num_agents * 0.1), 2.0)  # Cap the scaling factor
                
                if agent_types[node] == 'influencer':
                    node_colors.append(type_colors['influencer'])
                    node_sizes.append(base_size * 1.5 * (1 + in_degree_factor * 0.5))  # Reduced size multiplier
                elif agent_types[node] == 'bot':
                    node_colors.append(type_colors['bot'])
                    node_sizes.append(base_size * 0.7 * (1 + in_degree_factor * 0.3))
                else:
                    node_colors.append(type_colors['user'])
                    node_sizes.append(base_size * (1 + in_degree_factor * 0.3))
            else:
                # If no agent types provided, color by community
                node_colors.append(communities[node])
                node_sizes.append(100)
        
        # Draw the network
        nx.draw_networkx_nodes(self.network, pos, 
                             node_color=node_colors, 
                             node_size=node_sizes)
        
        # Draw edges with arrows
        nx.draw_networkx_edges(self.network, pos, 
                              alpha=0.2,
                              arrows=True,  # Show direction of edges
                              arrowsize=10)  # Size of arrow head
        
        # Add labels for agent types if provided
        if agent_types:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=type_colors[type_name], markersize=15, 
                          label=f'{type_name} (avg followers: {self._get_avg_followers(type_name, agent_types):.1f})')
                for type_name in ['influencer', 'bot', 'user']
            ]
            plt.legend(handles=legend_elements, loc='upper left')
        
        # Add title with metrics
        clustering_coef = nx.average_clustering(undirected_network)
        modularity = community.modularity(communities, undirected_network)
        plt.title(f'Network Communities\nClustering Coefficient: {clustering_coef:.3f}\n'
                 f'Modularity: {modularity:.3f}\n'
                 f'Total Connections: {self.network.number_of_edges()}')
        
        plt.axis('off')
        plt.show()

    def _get_avg_followers(self, agent_type, agent_types):
        """Helper method to calculate average followers for each agent type"""
        followers = [self.network.in_degree(node) 
                    for node, type_ in agent_types.items() 
                    if type_ == agent_type]
        return sum(followers) / len(followers) if followers else 0

    def get_clustering_metrics(self):
        """Return detailed clustering metrics"""
        # Convert to undirected for metrics that require it
        undirected_network = self.network.to_undirected()
        
        metrics = {
            'average_clustering': nx.average_clustering(undirected_network),
            'clustering_by_node': nx.clustering(undirected_network),
            'communities': community.best_partition(undirected_network),
            'modularity': community.modularity(
                community.best_partition(undirected_network), 
                undirected_network
            ),
            'transitivity': nx.transitivity(undirected_network)
        }
        
        # Add directed-specific metrics
        metrics.update({
            'in_degree_centrality': nx.in_degree_centrality(self.network),
            'out_degree_centrality': nx.out_degree_centrality(self.network),
            'average_in_degree': sum(dict(self.network.in_degree()).values()) / self.network.number_of_nodes(),
            'average_out_degree': sum(dict(self.network.out_degree()).values()) / self.network.number_of_nodes(),
        })
        
        return metrics
