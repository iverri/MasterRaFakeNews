from mesa import Agent
import numpy as np
from numpy import dot
from numpy.linalg import norm
from utils.similarity import cosine_similarity

class UserAgent(Agent):
    # Initialize the agent
    def __init__(self, model, preference_vector, credibility_level, influence_level):
        super().__init__(model)
        self.preference_vector = preference_vector
        self.credibility_level = credibility_level
        self.influence_level = influence_level
        self.state = "S"
        self.feed = [] # feed with NewsContent
        self.recommended_content = []

    def step(self):
        """Execute one step for the agent"""
        # Process each content in feed
        processed_feed = self.feed.copy()  # Create copy to avoid modifying list during iteration
        self.feed = []  # Clear feed
        
        for content in processed_feed:
            if self.evaluate_content(content):
                self.share_content(content)

    # TODO Come back to this later, and improve
    def evaluate_content(self, content):
        # Evaluate content based on interest vector
        # Return True if content is interesting
        if content.isFake:
            self.state = "E"

        belief_probability = cosine_similarity(self.preference_vector, content.topic_vector)
        return belief_probability > 0.5
    
    def share_content(self, content):
        """Share content with neighboring agents in the network"""
        # Get all neighbors from the network
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        
        # Share content with each neighbor
        for neighbor in neighbors:
            if content not in neighbor.feed:
                neighbor.feed.append(content)
        
        # If content is fake, change state to Believer
        if content.isFake:
            self.state = "B"

class BotAgent(UserAgent):
    def __init__(self, model, preference_vector):
        # TODO update values based on more thorough assessment
        super().__init__(model, preference_vector, influence_level=0.3, credibility_level=0.9)

class InfluencerAgent(UserAgent):
    def __init__(self, model, preference_vector):
        # TODO update values based on more thorough assessment
        super().__init__(model, preference_vector, influence_level=0.9, credibility_level=0.7)
   