import networkx as nx
import numpy as np
from utils.network_storage import NetworkStorage

TRAIT_NAMES = ["E", "A", "C", "N", "O"]  # Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness

def calculate_misinformation_count(model):
    """Calculate the average number of fake news items in agents' recommendation lists."""
    total_fake = sum(
        sum(1 for c in a.recommended_content if c.isFake)
        for a in model.agents
        if hasattr(a, "recommended_content")
    )
    active_agents = sum(
        1
        for a in model.agents
        if hasattr(a, "recommended_content") and len(a.recommended_content) > 0
    )
    return total_fake / active_agents if active_agents > 0 else 0


def calculate_misinformation_ratio_difference(model):
    """
    Calculate the average difference between the ratio of fake news in each user's recommendations
    vs. the overall ratio in the content pool.
    Positive values indicate recommendations have more fake news than the overall pool.
    """
    # Overall fake news ratio in content pool
    if not hasattr(model, "news_content") or not model.news_content:
        return 0

    overall_fake_ratio = sum(1 for c in model.news_content if c.isFake) / len(
        model.news_content
    )

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
    exposed_agents = sum(
        1 for a in model.agents if hasattr(a, "state") and (a.state == "I")
    )
    return exposed_agents / len(model.agents) if len(model.agents) > 0 else 0


def calculate_echo_chamber_effect(model):
    """
    Calculate the echo chamber effect based on the within-community content similarity.

    Higher values indicate stronger echo chambers - communities consuming
    similar content internally.

    This calculation is only performed every 5 steps to improve performance.
    """
    # Only calculate every 5 steps
    if model.steps % 5 != 0:
        # Return the last calculated value if available
        return getattr(model, "last_echo_chamber_score", 0)

    # Get community content similarity data
    within_similarity = calculate_cluster_content_similarity(model)

    # If we don't have community data, return 0 or last value
    if within_similarity is None:
        return getattr(model, "last_echo_chamber_score", 0)

    # Normalize the within-community similarity to a 0-1 scale for easier interpretation
    # Higher values indicate stronger echo chambers
    echo_chamber_score = min(within_similarity, 1.0)

    # Store the result for use in non-calculation steps
    model.last_echo_chamber_score = echo_chamber_score
    return echo_chamber_score


def calculate_cluster_content_similarity(model):
    """
    Use community detection to identify communities, then compute:
    - Average pairwise similarity of shared content within each community
    Returns: within_similarity

    Communities are detected once and cached for the entire simulation.
    Content similarity is recalculated every 5 steps.
    """
    # Only calculate content similarity every 5 steps
    if model.steps % 5 != 0:
        # If we have stored previous results, return those
        if hasattr(model, "last_similarity_results"):
            return model.last_similarity_results
        # Otherwise return None values
        return None

    import numpy as np

    # Get the social network
    network = model.social_media_platform.social_network.network
    if network.number_of_nodes() < 10:
        return None

    # Use a class-level cache for communities across all model instances
    # This ensures all models use exactly the same community structure
    if not model.network_storage.global_communities:
        print("FIRST TIME DETECTING COMMUNITIES - THIS SHOULD HAPPEN ONLY ONCE")
        # First time - detect communities
        try:
            import infomap

            # Create an Infomap instance with fixed seed for reproducibility
            im = infomap.Infomap("--directed --silent --seed 42")

            # Add links to Infomap - use sorted edges for determinism
            sorted_edges = sorted(network.edges())
            for source, target in sorted_edges:
                im.add_link(source, target)

            # Run the Infomap algorithm
            im.run()

            # Extract communities
            communities = {}
            for node, module in im.modules:
                communities[node] = module

            # Debug: Print community sizes to verify consistency
            community_counts = {}
            for comm_id in set(communities.values()):
                community_counts[comm_id] = sum(
                    1 for v in communities.values() if v == comm_id
                )
            print(
                f"COMMUNITY DETECTION - Network size: {network.number_of_nodes()}, Communities: {len(community_counts)}"
            )
            print(
                f"Community sizes: {sorted(community_counts.values(), reverse=True)[:5]}..."
            )

        except ImportError:
            # Fallback to Louvain on directed graph using weight adjustments
            import community as community_louvain

            # Set random seed for reproducibility
            np.random.seed(42)

            # Create a weighted undirected graph that preserves directional information
            weighted_undirected = nx.Graph()
            # Use sorted edges for determinism
            sorted_edges = sorted(network.edges())
            for u, v in sorted_edges:
                # Check if reciprocal edge exists
                if network.has_edge(v, u):
                    # Reciprocal connection (both follow each other) gets higher weight
                    weighted_undirected.add_edge(u, v, weight=2.0)
                else:
                    # One-way connection gets lower weight
                    weighted_undirected.add_edge(u, v, weight=1.0)

            # Run Louvain on the weighted undirected graph
            communities = community_louvain.best_partition(weighted_undirected)

            # Debug: Print community sizes to verify consistency
            community_counts = {}
            for comm_id in set(communities.values()):
                community_counts[comm_id] = sum(
                    1 for v in communities.values() if v == comm_id
                )
            print(
                f"COMMUNITY DETECTION (Louvain) - Network size: {network.number_of_nodes()}, Communities: {len(community_counts)}"
            )
            print(
                f"Community sizes: {sorted(community_counts.values(), reverse=True)[:5]}..."
            )

            # Reset random seed to avoid affecting other parts of the simulation
            np.random.seed(None)

        # Store communities in the NetworkStorage singleton for global access
        model.network_storage.global_communities = communities

        # Also cache in this model instance
        model.cached_communities = communities
    else:
        # Reuse globally cached communities
        communities = model.network_storage.global_communities
        model.cached_communities = communities

    # Map: community_id -> list of content topic vectors shared by members
    community_content = {}
    # Track fake news per community
    community_fake_news = {}
    # Track community sizes
    community_sizes = {}

    for agent in model.agents:
        if not hasattr(agent, "recent_content") or not agent.recent_content:
            continue
        # Get the community id of the agent
        comm_id = communities.get(agent.pos, -1)
        if comm_id not in community_content:
            community_content[comm_id] = []
            community_fake_news[comm_id] = 0
            community_sizes[comm_id] = 0

        community_sizes[comm_id] += 1

        # Add all topic vectors of content this agent has shared
        for item in agent.recent_content:
            community_content[comm_id].append(item["content"].topic_vector)
            # Count fake news
            if item["content"].isFake:
                community_fake_news[comm_id] += 1

    # Remove empty communities
    community_content = {k: v for k, v in community_content.items() if len(v) > 1}

    # Calculate fake news ratio per community
    community_fake_ratio = {}
    for comm_id in community_fake_news:
        # Get the actual count of content items, not just the length of topic vectors list
        total_content = len(community_content.get(comm_id, []))
        fake_content = community_fake_news[comm_id]

        if total_content > 0:
            community_fake_ratio[comm_id] = fake_content / total_content
        else:
            community_fake_ratio[comm_id] = 0

    # Compute within-cluster similarity
    within_sims = []
    community_within_sims = {}  # Store per-community similarity

    for comm_id, vectors in community_content.items():
        arr = np.array(vectors)
        if len(arr) < 2:
            continue
        sim_matrix = matrix_cosine_similarity(arr)
        # Take upper triangle, excluding diagonal
        triu_indices = np.triu_indices_from(sim_matrix, k=1)
        sims = sim_matrix[triu_indices]
        if len(sims) > 0:
            avg_sim = np.mean(sims)
            within_sims.append(avg_sim)
            community_within_sims[comm_id] = avg_sim

    if within_sims and community_sizes:
        # Create weights based on community sizes
        weights = []
        for comm_id in community_within_sims:
            weights.append(community_sizes[comm_id])

        # Calculate weighted average
        within_similarity = (
            np.average(list(community_within_sims.values()), weights=weights)
            if weights
            else None
        )
    else:
        within_similarity = None

    # Store community data in model for access by datacollector
    model.community_data = {
        "communities": communities,
        "sizes": community_sizes,
        "echo_scores": community_within_sims,
        "fake_ratio": community_fake_ratio,
    }

    # Store the results for use in non-calculation steps
    model.last_similarity_results = within_similarity
    return within_similarity


def calculate_diversity_improvement(model):
    """Calculate the percentage improvement in diversity from reranking."""
    total_improvement = 0
    count = 0

    for agent in model.agents:
        if hasattr(agent, "original_diversity_score") and hasattr(
            agent, "diversity_score"
        ):
            if agent.original_diversity_score > 0:
                improvement = (
                    agent.diversity_score - agent.original_diversity_score
                ) / agent.original_diversity_score
                total_improvement += improvement
                count += 1

    return (total_improvement / count * 100) if count > 0 else 0


# ------------------------------------------------------------------------------
# CONTENT EVALUATION METRICS
# ------------------------------------------------------------------------------


def cosine_similarity(preference_vector, topic_vector):
    x = np.asarray(preference_vector).ravel()
    y = np.asarray(topic_vector).ravel()

    denominator = np.linalg.norm(x) * np.linalg.norm(y)

    if denominator == 0:
        return 0.0
    else:
        return (x @ y) / denominator


def vec_mat_cosine_similarity(vector, matrix):
    x = np.asarray(vector).ravel()
    y = np.asarray(matrix)

    denominator = np.linalg.norm(x) * np.linalg.norm(y, axis=1)
    denominator[denominator == 0] = 1e-12

    return (y @ x) / denominator


def matrix_cosine_similarity(matrix):
    x = np.asarray(matrix)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    x_normalized = x / norms
    return x_normalized @ x_normalized.T
def get_graph_and_personality(model):
    G = model.social_media_platform.social_network.network
    P = np.asarray(model.personality_vectors, dtype=float)
    return G, P


def _regular_user_mask(model):
    """
    Boolean mask selecting only regular users (exclude influencers + bots)
    based on index ranges defined in the model.
    """
    N = model.num_agents
    n_inf = int(model.influencer_percentage * N)
    n_bot = int(model.bot_percentage * N)

    mask = np.ones(N, dtype=bool)
    mask[:n_inf] = False                # exclude influencers
    if n_bot > 0:
        mask[N - n_bot:] = False        # exclude bots
    return mask


def dominant_trait_indices(model):
    _, P = get_graph_and_personality(model)
    if P.ndim != 2:
        return np.array([])
    return np.argmax(P, axis=1)


def count_by_dominant_trait(model, trait_index):
    _, P = get_graph_and_personality(model)
    if P.ndim != 2 or trait_index >= P.shape[1]:
        return 0

    dom = np.argmax(P, axis=1)
    reg_mask = _regular_user_mask(model)
    return int(np.sum((dom == trait_index) & reg_mask))


def mean_degree_by_dominant_trait(model, trait_index, mode="in"):
    G, P = get_graph_and_personality(model)
    if P.ndim != 2 or trait_index >= P.shape[1]:
        return 0.0

    dom = np.argmax(P, axis=1)
    reg_mask = _regular_user_mask(model)

    deg = np.array([
        G.in_degree(i) if mode == "in" else G.out_degree(i)
        for i in range(model.num_agents)
    ], dtype=float)

    mask = (dom == trait_index) & reg_mask
    return float(np.mean(deg[mask])) if np.any(mask) else 0.0


def mean_personality_similarity_on_edges_regular_only(model):
    """
    Homophily metric using only edges where BOTH endpoints are regular users.
    """
    G, P = get_graph_and_personality(model)
    if P.ndim != 2:
        return 0.0

    reg_mask = _regular_user_mask(model)

    sims = []
    for i, j in G.edges():
        if not (reg_mask[i] and reg_mask[j]):
            continue

        a = P[i]
        b = P[j]
        da = np.linalg.norm(a)
        db = np.linalg.norm(b)

        if da == 0 or db == 0:
            sims.append(0.0)
        else:
            sims.append(float(np.dot(a, b) / (da * db)))

    return float(np.mean(sims)) if sims else 0.0

def dominant_trait_follow_matrix(model):
    """
    Compute 5x5 matrix M where:
      M[i,j] = share of outgoing follows from dominant-trait i users
               that go to dominant-trait j users.

    Only counts REGULAR users (excludes influencers + bots).
    Returns flattened dict for DataCollector.
    """
    G = model.social_media_platform.social_network.network
    P = np.asarray(model.personality_vectors, dtype=float)
    N = model.num_agents

    n_inf = int(model.influencer_percentage * N)
    n_bot = int(model.bot_percentage * N)

    reg = np.ones(N, dtype=bool)
    reg[:n_inf] = False
    if n_bot > 0:
        reg[N - n_bot:] = False

    if P.ndim != 2 or P.shape[0] != N:
        return {}

    dom = np.argmax(P, axis=1)
    K = P.shape[1]

    counts = np.zeros((K, K), dtype=int)

    for i, j in G.edges():
        if not (reg[i] and reg[j]):
            continue
        counts[dom[i], dom[j]] += 1

    # row-normalize → shares
    row_sums = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums > 0)

    result = {}
    for i in range(K):
        for j in range(K):
            result[f"DomFollowShare_{i}_{j}"] = float(shares[i, j])
            result[f"DomFollowCount_{i}_{j}"] = int(counts[i, j])

    return result

#used for datacollection
def ensure_personality_metrics_cache(m):
    step = getattr(m, "schedule", None).steps if hasattr(m, "schedule") else getattr(m, "steps", None)

    # If already computed for this step, do nothing
    if getattr(m, "_pm_cache_step", None) == step:
        return

    # Compute ONCE
    cache = {}
    for idx, name in enumerate(TRAIT_NAMES):
        cache[f"Dom_{name}_Count"] = count_by_dominant_trait(m, idx)
        cache[f"Dom_{name}_Followers_Mean"] = mean_degree_by_dominant_trait(m, idx, "in")
        cache[f"Dom_{name}_Following_Mean"] = mean_degree_by_dominant_trait(m, idx, "out")


    # Compute matrix ONCE
    #mat = dominant_trait_follow_matrix(m)
    #for i in range(5):
    #    for j in range(5):
    #        cache[f"DomFollowShare_{i}_{j}"] = mat.get(f"DomFollowShare_{i}_{j}", 0.0)
    #        cache[f"DomFollowCount_{i}_{j}"] = mat.get(f"DomFollowCount_{i}_{j}", 0)

    m._pm_cache = cache
    m._pm_cache_step = step

def final_only(m):
    return m.steps >= (m.max_steps - 1) 

def make_cached_reporter(key, m):
    def f(m):
        if not final_only(m):
            return np.nan
        ensure_personality_metrics_cache(m)   # ensure cache exists
        return m._pm_cache.get(key, np.nan)  # always return numeric-ish
    return f
