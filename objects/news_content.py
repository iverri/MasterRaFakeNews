# News content class

import numpy as np

class NewsContent:
    def __init__(self, content_id, isFake, topic_vector):
        self.content = content_id
        self.isFake = isFake
        self.topic_vector = topic_vector
        self.creation_step = 0  # Will be set when added to the model
        # Fake news starts with higher engagement
        self.engagement = 1.5 if isFake else 1.0

    def update_engagement(self, current_step, decay_rate=0.1):
        """Update content engagement based on age using exponential decay."""
        age = current_step - self.creation_step
        # Exponential decay function: engagement = initial_engagement * e^(-decay_rate * age)
        self.engagement = (1.5 if self.isFake else 1.0) * np.exp(-decay_rate * age)
        # Set a minimum engagement level
        self.engagement = max(0.05, self.engagement)
        return self.engagement

def initialize_news_content(model, news_amount):
    """Create a mix of real and fake news content based on the model's fake_news_percentage."""
    from utils.model_utils import random_preferences
    
    news_items = []
    fake_news_count = int(news_amount * model.fake_news_percentage)
    
    # Create fake news items
    for i in range(fake_news_count):
        topic_vector = random_preferences(model)
        news_items.append(NewsContent(i, True, topic_vector))
    
    # Create real news items
    for i in range(fake_news_count, news_amount):
        topic_vector = random_preferences(model)
        news_items.append(NewsContent(i, False, topic_vector))

    return news_items


    