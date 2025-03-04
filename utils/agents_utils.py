import numpy as np

#------------------------------------------------------------------------------
# CONTENT EVALUATION FUNCTIONS
#------------------------------------------------------------------------------

def evaluate_content_interest(belief_probability, threshold=0.6):
    """Determine if content is interesting enough to share."""
    return belief_probability > threshold

#------------------------------------------------------------------------------
# AGENT STATE MANAGEMENT FUNCTIONS
#------------------------------------------------------------------------------

def update_agent_state(agent, content, belief_probability):
    """Update agent state based on content evaluation."""
    # TODO: Implement this

       

#------------------------------------------------------------------------------
# NETWORK TRAVERSAL FUNCTIONS
#------------------------------------------------------------------------------

def get_network_neighbors(model, social_network, pos, direction="predecessors"):
    """Get neighbors from the network in specified direction."""
    if direction == "predecessors":
        neighbor_ids = [n for n in social_network.network.predecessors(pos)]
    elif direction == "successors":
        neighbor_ids = [n for n in social_network.network.successors(pos)]
    else:
        raise ValueError(f"Unknown direction: {direction}")
        
    return [agent for agent in model.agents if agent.pos in neighbor_ids]