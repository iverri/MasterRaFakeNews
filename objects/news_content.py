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
    """Create a mix of real and fake news content."""
    from utils.model_utils import random_preferences
    
    news_items = []
    for i in range(news_amount):  
        topic_vector = random_preferences(model)
        is_fake = i % 5 == 0 # Should be between 20-40 percent fake news
        news_items.append(NewsContent(i, is_fake, topic_vector))

    print(f"Initialized {len(news_items)} news items")
    return news_items


    