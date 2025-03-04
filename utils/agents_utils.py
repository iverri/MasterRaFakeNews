import numpy as np

def evaluate_content_interest(belief_probability, threshold=0.8):
    """Determine if content is interesting enough to share."""
    return belief_probability > threshold

def update_agent_state(agent, content, belief_probability):
    """Update agent state based on content evaluation."""
    if content.isFake:
        # Only become exposed if not already a believer
        if agent.state != "B":
            agent.state = "E"
        
        # Chance to become a believer
        if agent.state == "E" and np.random.random() < belief_probability * agent.credibility_level:
            agent.state = "B"

def get_network_neighbors(model, social_network, pos, direction="predecessors"):
    """Get neighbors from the network in specified direction."""
    if direction == "predecessors":
        neighbor_ids = [n for n in social_network.network.predecessors(pos)]
    elif direction == "successors":
        neighbor_ids = [n for n in social_network.network.successors(pos)]
    else:
        raise ValueError(f"Unknown direction: {direction}")
        
    return [agent for agent in model.agents if agent.pos in neighbor_ids]