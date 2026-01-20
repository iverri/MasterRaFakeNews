from mesa.datacollection import DataCollector
import random
import networkx as nx
import numpy as np
from utils.network_storage import NetworkStorage

# ------------------------------------------------------------------------------
# PREFERENCE AND CONTENT GENERATION FUNCTIONS
# ------------------------------------------------------------------------------


def random_preferences(model=None):
    """Generate random normalized preference vector."""
    # Generate random preferences with 15 dimensions
    preferences = [random.random() for i in range(8)]

    # Normalize the vector (convert to unit vector)
    magnitude = sum(x**2 for x in preferences) ** 0.5
    if magnitude > 0:  # Avoid division by zero
        preferences = [x / magnitude for x in preferences]

    return preferences


def generate_personalities(model=None):
    return [
        np.random.normal(loc=0.5, scale=0.1) for _ in range(5)
    ]  # Values for traits: [Extraversion, Agreeableness, Conscienciousness, Neuroticism, Openness]


def distribute_news(model):
    """Distribute news content to agents.

    Distributes content from the model's news_content pool to agents.
    Each agent receives a random sample of the content with varying sizes.
    """
    # If no content to distribute, return early
    if not model.news_content:
        return

    # Distribute content to all agents
    for agent in model.agents:
        # Determine a random number of content pieces for this agent
        content_sample_size = model.random.randint(10, min(15, len(model.news_content)))

        # Select random content from the content pool
        content_sample = model.random.sample(model.news_content, content_sample_size)

        # Add the content to the agent's feed
        agent.feed.extend(content_sample)


# ------------------------------------------------------------------------------
# AGENT AND NETWORK ANALYSIS FUNCTIONS
# ------------------------------------------------------------------------------


def get_agent_types(model):
    """Get a dictionary mapping node IDs to agent types"""
    # Import inside function to avoid circular import
    from agents.user_agent import InfluencerAgent, BotAgent

    agent_types = {}
    for agent in model.agents:
        node_id = agent.pos
        if isinstance(agent, InfluencerAgent):
            agent_types[node_id] = "influencer"
        elif isinstance(agent, BotAgent):
            agent_types[node_id] = "bot"
        else:
            agent_types[node_id] = "user"
    return agent_types
