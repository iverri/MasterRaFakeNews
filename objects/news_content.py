# News content class

import numpy as np


class NewsContent:
    def __init__(self, content_id, isFake, topic_vector, creation_step):
        self.content = content_id
        self.isFake = isFake
        self.topic_vector = topic_vector
        self.creation_step = creation_step  # Will be set when added to the model
        # Fake news starts with higher engagement
        self.engagement = 1.2  # All content starts on equal ground
        self._last_engagement_update_step = creation_step  # Cache when last updated

    def update_engagement(self, current_step, decay_rate=0.1):
        """Update content engagement based on age using exponential decay.
        Optimized to skip recalculation if engagement hasn't changed significantly."""

        # Only update if we haven't updated recently (batch updates for efficiency)
        # Update every 5 steps to reduce np.exp() calls
        if current_step - self._last_engagement_update_step < 5:
            return

        age = current_step - self.creation_step
        # Exponential decay function: engagement = initial_engagement * e^(-decay_rate * age)
        self.engagement = (1.5 if self.isFake else 1.0) * np.exp(-decay_rate * age)
        # Set a minimum engagement level
        self.engagement = max(0.05, self.engagement)
        self._last_engagement_update_step = current_step


def topic_likely_to_be_fake(topic_vector):
    """Determine if a topic vector is likely to be fake news."""
    sum_of_first_two_elements = sum(topic_vector[:2])
    return sum_of_first_two_elements > 0.3


def generate_news_content(fake_news_percentage, news_amount, creation_step):
    """Create a mix of real and fake news content based on the model's fake_news_percentage."""
    from utils.model_utils import random_preferences

    topic_vectors = [random_preferences() for _ in range(news_amount)]
    news_items = []

    for i in range(news_amount):
        if topic_likely_to_be_fake(topic_vectors[i]):
            is_fake = np.random.random() < fake_news_percentage * 2
        else:
            is_fake = np.random.random() < fake_news_percentage

        news_items.append(NewsContent(i, is_fake, topic_vectors[i], creation_step))

    return news_items
