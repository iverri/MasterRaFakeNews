import random
import numpy as np
from recommender.diversity import calculate_diversity


def random_recommendation(agent, num_recommendations, add_to_feed=True):
    """Recommend random news content to an agent"""

    # Get content pool from model and ensure it exists
    if not hasattr(agent.model, "news_content"):
        return

    if not agent.model.news_content:
        return

    # Cache feed set for faster lookups
    feed_set = set(agent.feed)

    # Get content that isn't in the agent's current feed - use set difference for efficiency
    available_content = [c for c in agent.model.news_content if c not in feed_set]

    if available_content:

        recommendations = random.sample(available_content, num_recommendations)

        # Apply diversity reranking if enabled (commented out in original)
        if add_to_feed:
            agent.recommended_content.extend(recommendations)
        else:
            return recommendations

        # After reranking
        topic_vectors = np.array([rec.topic_vector for rec in recommendations])
        diversity_score = calculate_diversity(topic_vectors)
        agent.diversity_score = diversity_score
