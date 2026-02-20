"""
Utility functions for the objects module.
Contains helper functions for network creation, manipulation, and analysis.
"""

import networkx as nx
import numpy as np
from utils.personality_utils import personality_similarity, likely_to_follow, likely_to_be_followed, clip
# import community  # python-louvain package

#------------------------------------------------------------------------------
# NETWORK CREATION FUNCTIONS
#------------------------------------------------------------------------------


def create_preference_based_network(model, num_agents, m_links, preference_vectors, personality_vectors):
    """
    Create directed network with communities based on preference similarity.
    
    Args:
        model: The simulation model
        num_agents (int): Number of agents in the network
        m_links (int): Base number of links per node (average)
        preference_vectors (list): List of preference vectors for each agent
        
    Returns:
        nx.DiGraph: Directed graph with preference-based connections
    """
    # Initialize empty directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_agents))
    
    # Calculate number of each agent type
    num_influencers = int(model.influencer_percentage * num_agents)
    num_bots = int(model.bot_percentage * num_agents)
    
    # Group nodes by type
    influencer_indices = list(range(num_influencers))
    bot_indices = list(range(num_agents - num_bots, num_agents))
    user_indices = list(range(num_influencers, num_agents - num_bots))
    
    # Calculate similarity matrix (only needed once)
    similarity_matrix = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            sim = np.dot(preference_vectors[i], preference_vectors[j])
            similarity_matrix[i,j] = similarity_matrix[j,i] = sim
    
    # Create initial connections based on agent types and preferences
    _create_initial_connections(G, num_agents, num_influencers, num_bots, 
                               similarity_matrix, m_links, personality_vectors)
    
    # Ensure the graph is weakly connected
    if not nx.is_weakly_connected(G):
        components = list(nx.weakly_connected_components(G))
        for i in range(len(components)-1):
            node1 = list(components[i])[0]
            node2 = list(components[i+1])[0]
            G.add_edge(node1, node2)
    
    # Adjust follower distributions to match expected patterns
    _adjust_follower_distributions(G, m_links, influencer_indices, user_indices, bot_indices, personality_vectors)
    
    # Print network statistics
    _print_network_stats(G, influencer_indices, user_indices, bot_indices)
    
    return G

def _create_initial_connections(G, num_agents, num_influencers, num_bots, similarity_matrix, m_links, personality_vectors):
    """Create initial connections based on agent types and preferences."""
    for i in range(num_agents):
        # Determine agent type and target outgoing connections
        if i < num_influencers:
            agent_type = 'influencer'
            k_out = max(int(m_links * 0.5 * np.random.uniform(0.7, 1.3)), 2)
        elif i >= num_agents - num_bots:
            agent_type = 'bot'
            k_out = min(int(m_links * 3 * np.random.uniform(0.9, 1.3)), num_agents-1)
        else:
            agent_type = 'regular'
            base = np.random.randint(max(1, m_links - 4), m_links + 5)  # make a bigger interval of how many out links a regular user can have
            fp = likely_to_follow(personality_vectors[i])
            k_out = clip(int(round(base * (1 + fp))), 1, num_agents - 1)

        
        # Calculate connection probabilities
        potential_edges = []
        for j in range(num_agents):
            if i == j:
                continue  # Skip self-connections
                
            # Determine target agent type
            if j < num_influencers:
                target_type = 'influencer'
                multiplier = 4.0
            elif j >= num_agents - num_bots:
                target_type = 'bot'
                multiplier = 0.001
            else:
                target_type = 'regular'
                multiplier = 1.0
                
            # Calculate base similarity
            base_sim = 0.3 * similarity_matrix[i,j] + 0.2
            
            # Apply special case rules and calculate final probability
            if agent_type == 'influencer' and target_type == 'bot':
                prob = 0.001  # Influencers almost never follow bots
            elif agent_type == 'regular' and target_type == 'bot':
                prob = 0.01   # Regular users rarely follow bots
            elif target_type == 'bot':
                prob = min(base_sim ** 100, 0.001)  # Hard cap on bot follow probability
            elif target_type == 'influencer':
                prob = min(base_sim ** (1/multiplier), 0.60)  # Influencers are followed more
            else:
                prob = base_sim  # Regular case
            
            pers_sim = personality_similarity(personality_vectors[i], personality_vectors[j])
            att_j = likely_to_be_followed(personality_vectors[j])
            att = clip(att_j, -0.2, 0.8)  # Ensure attractiveness is within bounds
            att = (att -0.3)/0.5  # Normalize to roughly -1 to +1 range

            #homophily_strength : how much similarity matters (0 = no effect) 
            homophily_strength = 0.3
            #attractiveness_strength : how much personality attracts followers
            attractiveness_strength = 0.4

            prob *= (1.0 + attractiveness_strength * att)
            prob *= (1.0 + homophily_strength * pers_sim)

            prob = clip(prob, 0.00, 0.95)  # Ensure probabilities are within bounds


            potential_edges.append((j, prob))
        
        # Sort by probability and create connections
        potential_edges.sort(key=lambda x: x[1], reverse=True)
        edges_added = 0
        for j, prob in potential_edges:
            if edges_added >= k_out:
                break
                
            if not G.has_edge(i, j) and np.random.random() < prob:
                G.add_edge(i, j)
                edges_added += 1

def _adjust_follower_distributions(G, m_links, influencer_indices, user_indices, bot_indices, personality_vectors):
    """Adjust follower distributions to match expected patterns."""
    # Calculate current average user followers
    user_followers = [G.in_degree(i) for i in user_indices]
    avg_user_followers = sum(user_followers) / len(user_followers) if user_followers else 0
    target_user_followers = max(avg_user_followers, m_links)
    
    # 1. Reset bot followers to very low counts
    for bot in bot_indices:
        # Remove all followers and add back just a few
        for follower in list(G.predecessors(bot)):
            G.remove_edge(follower, bot)
        
        # Add 0-3 random followers
        random_followers = np.random.randint(0, 4)
        potential_followers = list(range(len(G)))
        np.random.shuffle(potential_followers)
        
        for follower in potential_followers[:random_followers]:
            if follower != bot:
                G.add_edge(follower, bot)
    
    # 2. Boost influencer followers with power-law distribution
    for i, influencer in enumerate(influencer_indices):
        # Calculate target based on rank (higher ranked = more followers)
        rank_factor = 1.0 - (i / max(1, len(influencer_indices) - 1)) * 0.7
        random_factor = np.random.uniform(0.3, 1.7)
        target = int(target_user_followers * 6 * random_factor * (0.3 + 0.7 * rank_factor))
        att = likely_to_be_followed(personality_vectors[influencer])
        target = int(target * (1.0 + 0.25*att))

        # Add followers if needed
        current = G.in_degree(influencer)
        if current < target:
            needed = target - current
            potential_followers = [u for u in user_indices if not G.has_edge(u, influencer)]
            np.random.shuffle(potential_followers)
            
            for follower in potential_followers[:needed]:
                G.add_edge(follower, influencer)

def _print_network_stats(G, influencer_indices, user_indices, bot_indices):
    """Print network statistics."""
    # Calculate follower statistics for each group
    groups = {
        "Influencers": influencer_indices,
        "Regular Users": user_indices,
        "Bots": bot_indices
    }
    
    print("NETWORK CREATION - FINAL FOLLOWER COUNTS:")
    for name, indices in groups.items():
        if not indices:
            continue
            
        followers = [G.in_degree(i) for i in indices]
        avg = sum(followers) / len(followers)
        std = np.std(followers)
        print(f"{name}: {avg:.1f} ± {std:.1f}")

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


