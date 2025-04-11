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
    num_regular_users = num_agents - num_influencers - num_bots
    
    # Calculate similarity matrix between all pairs of nodes
    similarity_matrix = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            sim = np.dot(preference_vectors[i], preference_vectors[j])
            similarity_matrix[i,j] = similarity_matrix[j,i] = sim
    
    # Define BASE target outgoing connections (following) for each agent type
    outgoing_targets = {
        'influencer': max(int(m_links * 0.5), 2),      # Influencers follow fewer
        'regular': m_links,                            # Regular users follow average
        'bot': min(int(m_links * 3), num_agents-1)     # Bots follow many more
    }
    
    # Define follower attractiveness multipliers (how likely to be followed)
    follower_multipliers = {
        'influencer': 6.0,    # Influencers are much more likely to be followed
        'regular': 1.0,        # Regular users have normal follow probability
        'bot': 0.001          # Bots are extremely unlikely to be followed
    }
    
    # Create edges based on agent type and preference similarity
    for i in range(num_agents):
        # Determine agent type and outgoing connection target
        if i < num_influencers:
            agent_type = 'influencer'
        elif i >= num_agents - num_bots:
            agent_type = 'bot'
        else:
            agent_type = 'regular'
            
        # Number of outgoing connections for this node - ADD RANDOMNESS HERE
        base_k_out = outgoing_targets[agent_type]
        
        # Add randomness to connection count (±30% variation)
        variation_factor = np.random.uniform(0.7, 1.3)
        k_out = max(1, int(base_k_out * variation_factor))
        
        # For bots, ensure they still follow many accounts
        if agent_type == 'bot':
            k_out = max(k_out, base_k_out)
        
        # Calculate weighted probabilities for all potential connections
        potential_edges = []
        
        for j in range(num_agents):
            if i == j:
                continue  # Skip self-connections
                
            # Determine target agent type
            if j < num_influencers:
                target_type = 'influencer'
            elif j >= num_agents - num_bots:
                target_type = 'bot'
            else:
                target_type = 'regular'
                
            # Base similarity with reduced impact (mix with constant)
            base_sim = 0.3 * similarity_matrix[i,j] + 0.2
            
            # Apply follower multiplier based on target type using nonlinear transformation
            if follower_multipliers[target_type] >= 1.0:
                # For high multipliers (influencers), use power function to amplify
                adjusted_sim = base_sim ** (1 / follower_multipliers[target_type])
            else:
                # For low multipliers (bots), use power function to reduce
                adjusted_sim = base_sim ** (1 / (follower_multipliers[target_type] * 0.01))
            
            # Special case: Influencers almost never follow bots
            if agent_type == 'influencer' and target_type == 'bot':
                adjusted_sim = 0.001  # Extremely low probability
                
            # Special case: Regular users rarely follow bots
            if agent_type == 'regular' and target_type == 'bot':
                adjusted_sim = 0.01  # Very low probability
            
            # Hard caps based on agent types
            if target_type == 'bot':
                adjusted_sim = min(adjusted_sim, 0.001)  # Hard cap on bot follow probability
            elif target_type == 'influencer':
                adjusted_sim = min(adjusted_sim, 0.90)  # Cap for influencers too
                
            potential_edges.append((j, adjusted_sim))
        
        # Sort by adjusted similarity
        potential_edges.sort(key=lambda x: x[1], reverse=True)
        
        # Add directed edges (i follows j)
        edges_added = 0
        for j, sim in potential_edges:
            if edges_added >= k_out:
                break
                
            if not G.has_edge(i, j) and np.random.random() < sim:
                G.add_edge(i, j)
                edges_added += 1
    
    # Ensure the graph is weakly connected
    if not nx.is_weakly_connected(G):
        components = list(nx.weakly_connected_components(G))
        for i in range(len(components)-1):
            # Try to connect components through non-bot nodes if possible
            component1 = [n for n in components[i] if n < num_agents - num_bots]
            component2 = [n for n in components[i+1] if n < num_agents - num_bots]
            
            # If no non-bot nodes available, use any nodes
            node1 = list(components[i])[0] if not component1 else component1[0]
            node2 = list(components[i+1])[0] if not component2 else component2[0]
            
            G.add_edge(node1, node2)
    
    # VERIFICATION AND ADJUSTMENT PHASE - More aggressive to ensure proper ratios
    # Group nodes by type
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
    
    # Define target follower ratios with randomness
    target_user_followers = max(avg_user_followers, m_links)
    target_bot_followers = max(int(target_user_followers * 0.3), 1)  # Bots have 10% of user followers
    target_influencer_followers = max(int(target_user_followers * 6), m_links * 6)  # Influencers have 8x user followers
    
    # 1. RESET bot followers but with some randomness
    for bot in bot_indices:
        # Remove ALL followers
        followers = list(G.predecessors(bot))
        for follower in followers:
            G.remove_edge(follower, bot)
        
        # Add back a very small number of followers with randomness
        potential_followers = list(range(num_agents))
        np.random.shuffle(potential_followers)
        
        # Random number of followers between 0 and max_bot_followers
        max_bot_followers = min(4, target_bot_followers)
        random_followers = np.random.randint(0, max_bot_followers + 1)
        
        for follower in potential_followers[:random_followers]:
            if follower != bot:
                G.add_edge(follower, bot)
    
    # 2. Boost influencer followers with randomness
    for influencer in influencer_indices:
        current_followers = G.in_degree(influencer)
        
        # Much greater variation for influencers (±70%)
        # This creates a more realistic power-law distribution
        random_factor = np.random.uniform(0.3, 1.7)
        
        # Apply a power-law-like distribution where some influencers get much more followers
        # Use rank within influencers to create a natural hierarchy
        rank_factor = 1.0 - (influencer / max(1, len(influencer_indices) - 1)) * 0.7
        combined_factor = random_factor * (0.3 + 0.7 * rank_factor)
        
        random_target = int(target_influencer_followers * combined_factor)
        
        if current_followers < random_target:
            needed_followers = random_target - current_followers
            
            # Try to add followers from regular users first
            potential_followers = [u for u in user_indices if not G.has_edge(u, influencer)]
            np.random.shuffle(potential_followers)
            
            for follower in potential_followers[:needed_followers]:
                G.add_edge(follower, influencer)
    
    # 3. Boost regular user followers with randomness
    if avg_user_followers < target_user_followers:
        for user in user_indices:
            current_followers = G.in_degree(user)
            
            # Random target (±40% variation)
            random_factor = np.random.uniform(0.4, 1.6)
            random_target = target_user_followers * random_factor
            
            if current_followers < random_target * 0.8:
                needed_followers = int(random_target * 0.8) - current_followers
                
                # Try to add followers from other regular users
                potential_followers = [u for u in user_indices if u != user and not G.has_edge(u, user)]
                np.random.shuffle(potential_followers)
                
                for follower in potential_followers[:needed_followers]:
                    G.add_edge(follower, user)
    
    # 4. Final verification - ensure bots have MUCH fewer followers than regular users
    bot_followers = [G.in_degree(i) for i in bot_indices]
    user_followers = [G.in_degree(i) for i in user_indices]
    
    avg_bot_followers = sum(bot_followers) / len(bot_followers) if bot_followers else 0
    avg_user_followers = sum(user_followers) / len(user_followers) if user_followers else 0
    
    # If bots still have too many followers, RESET them again
    if avg_bot_followers > avg_user_followers * 0.1:
        for bot in bot_indices:
            # Remove ALL followers again
            followers = list(G.predecessors(bot))
            for follower in followers:
                G.remove_edge(follower, bot)
            
            # Add back at most 0-2 followers (random)
            potential_followers = list(range(num_agents))
            np.random.shuffle(potential_followers)
            
            random_followers = np.random.randint(0, 4)  # 0, 1, or 2 followers
            for follower in potential_followers[:random_followers]:
                if follower != bot:
                    G.add_edge(follower, bot)
    
    # Final verification - print actual follower counts
    bot_followers = [G.in_degree(i) for i in bot_indices]
    user_followers = [G.in_degree(i) for i in user_indices]
    influencer_followers = [G.in_degree(i) for i in influencer_indices]
    
    avg_bot_followers = sum(bot_followers) / len(bot_followers) if bot_followers else 0
    avg_user_followers = sum(user_followers) / len(user_followers) if user_followers else 0
    avg_influencer_followers = sum(influencer_followers) / len(influencer_followers) if influencer_followers else 0
    
    # Calculate standard deviations to show the spread
    std_bot_followers = np.std(bot_followers) if bot_followers else 0
    std_user_followers = np.std(user_followers) if user_followers else 0
    std_influencer_followers = np.std(influencer_followers) if influencer_followers else 0
    
    print(f"NETWORK CREATION - FINAL FOLLOWER COUNTS:")
    print(f"Influencers: {avg_influencer_followers:.1f} ± {std_influencer_followers:.1f}")
    print(f"Regular Users: {avg_user_followers:.1f} ± {std_user_followers:.1f}")
    print(f"Bots: {avg_bot_followers:.1f} ± {std_bot_followers:.1f}")
    
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





