import numpy as np

# ------------------------------------------------------------------------------
# NETWORK TRAVERSAL FUNCTIONS
# ------------------------------------------------------------------------------


def get_network_neighbors(model, social_network, pos, direction="predecessors"):
    """Get neighbors from the network in specified direction."""
    if direction == "predecessors":
        neighbor_ids = [n for n in social_network.network.predecessors(pos)]
    elif direction == "successors":
        neighbor_ids = [n for n in social_network.network.successors(pos)]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return [agent for agent in model.agents if agent.pos in neighbor_ids]


def cosine_similarity(preference_vector, topic_vector):
    x = np.asarray(preference_vector).ravel()
    y = np.asarray(topic_vector).ravel()

    denominator = np.linalg.norm(x) * np.linalg.norm(y)

    if denominator == 0:
        return 0.0
    else:
        return (x @ y) / denominator
