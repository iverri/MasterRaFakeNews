from mesa import Agent
import numpy as np
from numpy import dot
from numpy.linalg import norm
from model import cosine_similarity


class UserAgent(Agent):
    # Initialize the agent
    def __init__(self, unique_id, model, interestVector):
        #In mesa we usually have unique_id for each agent and model
        #Inherit the unique_id and model
        super().__init__(unique_id, model)
        self.interestvector = interestVector
        self.feed = [] # feed with NewsContent

    def step(self):
        for content in self.feed:
            if self.evaluate_content(content):
                self.share_content(content) #If belief is high enough, share
    
    def evaluate_content(self, content):
        # Evaluate content based on interest vector
        # Return True if content is interesting
        belief_probability = cosine_similarity(self.interestvector, content.topic_vector)
        return belief_probability > 0.5
    
    def share_content(self, content):
        pass

class BotAgent(UserAgent):
    def __init__(self, model, interestVector, unique_id):
        super().__init__(model, interestVector, unique_id)
        self.influence_level = 0.3
        self.credibility_level = 0.9

class InfluencerAgent(UserAgent):
    def __init__(self, model, interestVector, unique_id):
        super().__init__(model, interestVector, unique_id)
        self.influence_level = 0.9
        self.credibility_level = 0.7