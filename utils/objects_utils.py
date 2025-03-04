"""
Utility functions for the objects module.
Contains helper functions for network creation, manipulation, and analysis.
"""

import networkx as nx
import numpy as np
# import community  # python-louvain package

def create_preference_based_network(num_agents, m_links, preference_vectors):
    """
    Create directed network with communities based on preference similarity.
    
    Args:
        num_agents (int): Number of agents in the network
        m_links (int): Base number of links per node
        preference_vectors (list): List of preference vectors for each agent
        
    Returns:
        nx.DiGraph: Directed graph with preference-based connections
    """
    # Initialize empty directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_agents))
    
    # Calculate similarity matrix between all pairs of nodes
    similarity_matrix = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            sim = np.dot(preference_vectors[i], preference_vectors[j])
            similarity_matrix[i,j] = similarity_matrix[j,i] = sim
    
    # Calculate number of each agent type
    num_influencers = int(0.05 * num_agents)
    num_bots = int(0.05 * num_agents)
    
    # First, handle regular users and bots following others
    for i in range(num_agents):
        if i < num_influencers:
            continue  # Handle influencers' connections separately
        
        # Number of outgoing connections for this node
        if i >= num_agents - num_bots:
            k = min(int(m_links * 3), num_agents-1)  # More outgoing for bots
        else:
            k = m_links  # Regular users
        
        # Get potential nodes to follow based on preference similarity
        potential_edges = []
        
        # Higher probability to follow influencers
        for j in range(num_influencers):
            sim = similarity_matrix[i,j] * 1.5  # Boost similarity for influencers
            potential_edges.append((j, sim))
        
        # Regular preference-based edges for other users
        for j in range(num_influencers, num_agents):
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
                if i >= num_agents - num_bots:
                    prob = min(sim * 1.5, 1.0)
                else:
                    prob = sim
                
                if np.random.random() < prob:
                    G.add_edge(i, j)
                    edges_added += 1
    
    # Now handle influencers' outgoing connections (they follow fewer accounts)
    for i in range(num_influencers):
        k = max(int(m_links * 0.5), 1)  # Fewer outgoing for influencers
        
        # Influencers are more likely to follow other influencers
        for j in range(num_influencers):
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

def create_basic_network(num_agents, m_links):
    """
    Fallback method using Barabási-Albert model.
    
    Args:
        num_agents (int): Number of agents in the network
        m_links (int): Number of links per node
        
    Returns:
        nx.Graph: Undirected graph with scale-free properties
    """
    return nx.barabasi_albert_graph(n=num_agents, m=m_links)

def swap_node_connections(G, node1, node2):
    """
    Swap all connections between two nodes.
    
    Args:
        G (nx.Graph): The network graph
        node1 (int): First node ID
        node2 (int): Second node ID
    """
    if node1 == node2:
        return
        
    # Get neighbors of both nodes
    node1_neighbors = set(G.neighbors(node1))
    node2_neighbors = set(G.neighbors(node2))
    
    # Remove all edges of both nodes
    G.remove_edges_from([(node1, n) for n in node1_neighbors])
    G.remove_edges_from([(node2, n) for n in node2_neighbors])
    
    # Add swapped edges
    G.add_edges_from([(node2, n) for n in node1_neighbors])
    G.add_edges_from([(node1, n) for n in node2_neighbors])

def adjust_node_connectivity(G, num_agents, num_influencers, num_bots):
    """
    Adjust network to ensure proper connectivity for different agent types.
    
    Args:
        G (nx.Graph): The network graph
        num_agents (int): Total number of agents
        num_influencers (int): Number of influencer agents
        num_bots (int): Number of bot agents
    """
    # Get node degrees
    degrees = dict(G.degree())
    nodes_by_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    
    # Ensure influencers have high connectivity
    for i in range(num_influencers):
        old_node = nodes_by_degree[i][0]
        if i >= num_influencers:  # If high-degree node isn't an influencer
            # Swap with a lower degree node that should be an influencer
            new_node = i
            swap_node_connections(G, old_node, new_node)
    
    # Ensure bots have low connectivity
    bot_start_idx = num_agents - num_bots
    for i in range(bot_start_idx, num_agents):
        old_node = nodes_by_degree[-i][0]
        if i < bot_start_idx:  # If low-degree node isn't a bot
            # Swap with a higher degree node that should be a bot
            new_node = i
            swap_node_connections(G, old_node, new_node)

def get_clustering_metrics(G):
    """
    Return detailed clustering metrics for a network.
    
    Args:
        G (nx.Graph): The network graph
        
    Returns:
        dict: Dictionary of network metrics
    """
    # Convert to undirected for metrics that require it
    if isinstance(G, nx.DiGraph):
        undirected_network = G.to_undirected()
    else:
        undirected_network = G
    
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
    
    # Add directed-specific metrics if applicable
    if isinstance(G, nx.DiGraph):
        metrics.update({
            'in_degree_centrality': nx.in_degree_centrality(G),
            'out_degree_centrality': nx.out_degree_centrality(G),
            'average_in_degree': sum(dict(G.in_degree()).values()) / G.number_of_nodes(),
            'average_out_degree': sum(dict(G.out_degree()).values()) / G.number_of_nodes(),
        })
    
    return metrics


