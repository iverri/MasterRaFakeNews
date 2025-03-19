"""
Utility functions for the objects module.
Contains helper functions for network creation, manipulation, and analysis.
"""

import networkx as nx
import numpy as np
# import community  # python-louvain package

#------------------------------------------------------------------------------
# NETWORK CREATION FUNCTIONS
#------------------------------------------------------------------------------

# TODO: look at bot handling

def create_preference_based_network(model, num_agents, m_links, preference_vectors):
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
    num_influencers = int(model.influencer_percentage * num_agents)
    num_bots = int(model.bot_percentage * num_agents)
    
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
        
        # Higher probability to follow influencers, but not overwhelmingly so
        for j in range(num_influencers):
            sim = similarity_matrix[i,j] * 1.2  # Reduced boost for influencers
            potential_edges.append((j, sim))
        
        # Regular preference-based edges for other users and reduced probability for bots
        for j in range(num_influencers, num_agents):
            if j != i:
                sim = similarity_matrix[i,j]
                # Boost for regular users to increase their followers
                if j < num_agents - num_bots:
                    sim *= 1.1  # Slight boost for regular users
                # Reduction for bots
                else:
                    sim *= 0.3  # Less extreme reduction for bots
                potential_edges.append((j, sim))
        
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
    
    # Now handle influencers' outgoing connections
    for i in range(num_influencers):
        k = max(int(m_links * 0.7), 2)  # Slightly more outgoing for influencers
        
        # Influencers follow other influencers and some regular users
        potential_edges = []
        
        # Follow other influencers
        for j in range(num_influencers):
            if i != j:
                sim = similarity_matrix[i,j] * 1.2
                potential_edges.append((j, sim))
        
        # Follow some regular users too
        for j in range(num_influencers, num_agents - num_bots):
            sim = similarity_matrix[i,j] * 0.8  # Reduced but still significant
            potential_edges.append((j, sim))
            
        potential_edges.sort(key=lambda x: x[1], reverse=True)
        
        # Add edges
        edges_added = 0
        for j, sim in potential_edges:
            if edges_added >= k:
                break
                
            if not G.has_edge(i, j) and np.random.random() < sim:
                G.add_edge(i, j)
                edges_added += 1
    
    # Ensure the graph is weakly connected
    if not nx.is_weakly_connected(G):
        components = list(nx.weakly_connected_components(G))
        for i in range(len(components)-1):
            node1 = list(components[i])[0]
            node2 = list(components[i+1])[0]
            G.add_edge(node1, node2)
            
    # PHASE 2: Balance follower distribution
    bot_indices = list(range(num_agents - num_bots, num_agents))
    user_indices = list(range(num_influencers, num_agents - num_bots))
    influencer_indices = list(range(num_influencers))
    
    # Calculate current average followers
    user_followers = [G.in_degree(i) for i in user_indices]
    bot_followers = [G.in_degree(i) for i in bot_indices]
    influencer_followers = [G.in_degree(i) for i in influencer_indices]
    
    avg_user_followers = sum(user_followers) / len(user_followers) if user_followers else 0
    avg_bot_followers = sum(bot_followers) / len(bot_followers) if bot_followers else 0
    avg_influencer_followers = sum(influencer_followers) / len(influencer_followers) if influencer_followers else 0
    
    # Target ratios: influencers should have ~8x more followers than users, bots ~0.7x of users
    target_user_followers = max(avg_user_followers, m_links * 1.5)  # Ensure users have decent followers
    target_bot_followers = target_user_followers * 1.5
    target_influencer_followers = target_user_followers * 8
    
    # Adjust bot followers (reduce if needed)
    if avg_bot_followers > target_bot_followers:
        for bot in bot_indices:
            followers = list(G.predecessors(bot))
            np.random.shuffle(followers)
            
            current_followers = len(followers)
            target_followers = int(target_bot_followers)
            
            if current_followers > target_followers:
                edges_to_remove = followers[:current_followers - target_followers]
                for follower in edges_to_remove:
                    if G.has_edge(follower, bot):
                        G.remove_edge(follower, bot)
    
    # Boost influencer followers if needed
    if avg_influencer_followers < target_influencer_followers:
        # Add more followers to influencers from regular users
        for i in user_indices:
            for j in influencer_indices:
                # Add edge with probability based on how far we are from target
                if not G.has_edge(i, j) and np.random.random() < 0.3:
                    G.add_edge(i, j)
    
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

#------------------------------------------------------------------------------
# NETWORK MANIPULATION FUNCTIONS
#------------------------------------------------------------------------------

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





