import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_misinformation_count(model):
    """Calculate the average number of fake news items in agents' recommendation lists."""
    total_fake = sum(sum(1 for c in a.recommended_content if c.isFake) 
                    for a in model.agents if hasattr(a, "recommended_content"))
    active_agents = sum(1 for a in model.agents if hasattr(a, "recommended_content") and len(a.recommended_content) > 0)
    return total_fake / active_agents if active_agents > 0 else 0

def calculate_misinformation_ratio_difference(model):
    """
    Calculate the average difference between the ratio of fake news in each user's recommendations 
    vs. the overall ratio in the content pool.
    Positive values indicate recommendations have more fake news than the overall pool.
    """
    # Overall fake news ratio in content pool
    if not hasattr(model, 'news_content') or not model.news_content:
        return 0
    
    overall_fake_ratio = sum(1 for c in model.news_content if c.isFake) / len(model.news_content)
    
    # Calculate per-user differences
    user_differences = []
    
    for agent in model.agents:
        if hasattr(agent, "recommended_content") and agent.recommended_content:
            # Calculate fake news ratio in this user's recommendations
            user_recs_count = len(agent.recommended_content)
            user_fake_count = sum(1 for c in agent.recommended_content if c.isFake)
            user_fake_ratio = user_fake_count / user_recs_count
            
            # Calculate difference for this user
            user_difference = user_fake_ratio - overall_fake_ratio
            user_differences.append(user_difference)
    
    # Return the average difference across all users
    if user_differences:
        return sum(user_differences) / len(user_differences)
    else:
        return 0

def calculate_misinformation_spread(model):
    """Calculate the percentage of agents who have been exposed to fake news."""
    exposed_agents = sum(1 for a in model.agents if hasattr(a, "state") and ( a.state == "I"))
    return exposed_agents / len(model.agents) if len(model.agents) > 0 else 0

def calculate_echo_chamber_effect(model):
    """
    Calculate the echo chamber effect based on the ratio of content similarity
    within communities versus between communities.
    
    Higher values indicate stronger echo chambers - communities consuming
    similar content internally but different content from other communities.
    """
    # Get community content similarity data
    within_similarity, between_similarity = calculate_cluster_content_similarity(model)
    
    # If we don't have community data, fall back to simpler metrics
    if within_similarity is None or between_similarity is None:
        # Fall back to the agent-based calculation
        preference_scores = [calculate_agent_echo_chamber(a) 
                            for a in model.agents if hasattr(a, "recommended_content") and len(a.recommended_content) > 0]
        return sum(preference_scores) / len(preference_scores) if preference_scores else 0
    
    # Calculate the ratio of within-community similarity to between-community similarity
    # Higher ratio means stronger echo chambers
    if between_similarity > 0:
        echo_chamber_ratio = within_similarity / between_similarity
        
        # Normalize the ratio to a 0-1 scale for easier interpretation
        # A ratio of 1.0 means no echo chamber (within = between)
        # Higher values indicate stronger echo chambers
        normalized_ratio = min(echo_chamber_ratio / 3.0, 1.0)  # Cap at 1.0, assuming ratios above 3.0 are strong echo chambers
        
        # Adjust so that 0 means no echo chamber and 1 means strong echo chamber
        echo_chamber_score = (normalized_ratio - 0.33) * 1.5
        echo_chamber_score = max(0, min(echo_chamber_score, 1.0))  # Ensure it stays in 0-1 range
        
        return echo_chamber_score
    else:
        # If between_similarity is 0, this is an extreme echo chamber
        return 1.0

def calculate_agent_echo_chamber(agent):
    """
    Calculate the echo chamber score for an individual agent.
    This measures how similar the content in their recommendations is to their preferences.
    """
    if not hasattr(agent, "recommended_content") or not agent.recommended_content:
        return 0
    
    # Calculate similarity between agent preferences and each content item
    agent_preference = np.array(agent.preference_vector).reshape(1, -1)
    content_topics = np.array([content.topic_vector for content in agent.recommended_content])
    similarities = cosine_similarity(agent_preference, content_topics)[0]
    
    # Higher average similarity indicates stronger echo chamber
    return sum(similarities) / len(similarities) if similarities.size > 0 else 0


def calculate_cluster_content_similarity(model):
    """
    Use Louvain to detect communities, then compute:
    - Average pairwise similarity of shared content within each community
    - Average pairwise similarity of shared content between communities
    - Echo chamber score per community
    Returns: (within_similarity, between_similarity)
    """
    import community as community_louvain
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Get the social network
    network = model.social_media_platform.social_network.network
    if network.number_of_nodes() < 10:
        return None, None

    # Detect communities
    undirected_network = network.to_undirected()
    communities = community_louvain.best_partition(undirected_network)

    # Map: community_id -> list of content topic vectors shared by members
    community_content = {}
    # Track fake news per community
    community_fake_news = {}
    # Track community sizes
    community_sizes = {}
    
    for agent in model.agents:
        if not hasattr(agent, "shared_content") or not agent.shared_content:
            continue
        # Get the community id of the agent
        comm_id = communities.get(agent.pos, -1)
        if comm_id not in community_content:
            community_content[comm_id] = []
            community_fake_news[comm_id] = 0
            community_sizes[comm_id] = 0
        
        community_sizes[comm_id] += 1
        
        # Add all topic vectors of content this agent has shared
        for item in agent.shared_content:
            community_content[comm_id].append(item['content'].topic_vector)
            # Count fake news
            if item['content'].isFake:
                community_fake_news[comm_id] += 1

    # Remove empty communities
    community_content = {k: v for k, v in community_content.items() if len(v) > 1}
    
    # Calculate fake news ratio per community
    community_fake_ratio = {}
    for comm_id in community_fake_news:
        total_content = len(community_content.get(comm_id, []))
        if total_content > 0:
            community_fake_ratio[comm_id] = community_fake_news[comm_id] / total_content
        else:
            community_fake_ratio[comm_id] = 0

    # Compute within-cluster similarity
    within_sims = []
    community_within_sims = {}  # Store per-community similarity
    
    for comm_id, vectors in community_content.items():
        arr = np.array(vectors)
        if len(arr) < 2:
            continue
        sim_matrix = cosine_similarity(arr)
        # Take upper triangle, excluding diagonal
        triu_indices = np.triu_indices_from(sim_matrix, k=1)
        sims = sim_matrix[triu_indices]
        if len(sims) > 0:
            avg_sim = np.mean(sims)
            within_sims.append(avg_sim)
            community_within_sims[comm_id] = avg_sim
    
    within_similarity = np.mean(within_sims) if within_sims else None

    # Compute between-cluster similarity
    between_sims = []
    community_between_sims = {}  # Store per-community pair similarity
    
    comm_ids = list(community_content.keys())
    for i in range(len(comm_ids)):
        for j in range(i+1, len(comm_ids)):
            comm_i = comm_ids[i]
            comm_j = comm_ids[j]
            arr1 = np.array(community_content[comm_i])
            arr2 = np.array(community_content[comm_j])
            sims = cosine_similarity(arr1, arr2).flatten()
            if len(sims) > 0:
                avg_sim = np.mean(sims)
                between_sims.append(avg_sim)
                pair_key = (comm_i, comm_j)
                community_between_sims[pair_key] = avg_sim
    
    between_similarity = np.mean(between_sims) if between_sims else None

    # Calculate echo chamber score per community
    community_echo_scores = {}
    
    # For each community, calculate its average similarity with all other communities
    for comm_id in community_within_sims:
        # Get this community's within similarity
        within_sim = community_within_sims[comm_id]
        
        # Calculate average between similarity for this community with all others
        comm_between_sims = []
        for pair_key, sim in community_between_sims.items():
            if comm_id in pair_key:
                comm_between_sims.append(sim)
        
        avg_between_sim = np.mean(comm_between_sims) if comm_between_sims else 0
        
        # Calculate echo chamber score for this community
        if avg_between_sim > 0:
            ratio = within_sim / avg_between_sim
            # Normalize to 0-1 scale
            # A ratio of 1.0 means no echo chamber (within = between)
            # Higher values indicate stronger echo chambers
            normalized_ratio = min(ratio / 3.0, 1.0)  # Cap at 1.0, assuming ratios above 3.0 are strong echo chambers
            
            # Adjust so that 0 means no echo chamber and 1 means strong echo chamber
            echo_score = (normalized_ratio - 0.33) * 1.5
            echo_score = max(0, min(echo_score, 1.0))  # Ensure it stays in 0-1 range
        else:
            # If between_similarity is 0, this is an extreme echo chamber
            echo_score = 1.0
            
        community_echo_scores[comm_id] = echo_score

    # Store community data in model for access by datacollector
    model.community_data = {
        'communities': communities,
        'sizes': community_sizes,
        'within_sims': community_within_sims,
        'between_sims': community_between_sims,
        'fake_ratio': community_fake_ratio,
        'echo_scores': community_echo_scores
    }

    return within_similarity, between_similarity
