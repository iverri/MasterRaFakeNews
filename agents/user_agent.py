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
        self.feed = [] # feed with NewsContent
        self.state = "S"
        self.social_network = model.social_media_platform.social_network
        self.social_media_platform = model.social_media_platform

    def step(self):
        """Execute one step for the agent"""
        # Process each content in feed
        processed_feed = self.feed.copy()  # Create copy to avoid modifying list during iteration
        self.feed = []  # Clear feed
        
        for content in processed_feed:
            if self.evaluate_content(content):
                self.share_content(content)

    def evaluate_content(self, content):
        # Evaluate content based on interest vector
        # Return True if content is interesting
        belief_probability = cosine_similarity(self.preference_vector, content.topic_vector)
        
        # Update state based on content evaluation
        if content.isFake:
            # Only become exposed if not already a believer
            if self.state != "B":
                self.state = "E"
            
            # Chance to become a believer based on belief probability and credibility
            if self.state == "E" and np.random.random() < belief_probability * self.credibility_level:
                self.state = "B"
        
        return belief_probability > 0.8
    
    def share_content(self, content):
        """Share content with neighboring agents in the network"""
        # Get all neighbors from the network
        neighbors = self.get_followers()
        
        # Share content with each neighbor
        for neighbor in neighbors:
            if content not in neighbor.feed:
                neighbor.feed.append(content)
        
        # If sharing fake content and not already a believer, become one
        if content.isFake and self.state != "B":
            # Higher influence level increases chance of becoming a believer when sharing
            if np.random.random() < self.influence_level:
                self.state = "B"

    def get_followers(self):
        """Get list of agents that follow this agent"""
        follower_ids = [n for n in self.social_network.network.predecessors(self.pos)]
        return [agent for agent in self.model.agents if agent.pos in follower_ids]

    def get_following(self):
        """Get list of agents this agent follows"""
        following_ids = [n for n in self.social_network.network.successors(self.pos)]
        return [agent for agent in self.model.agents if agent.pos in following_ids]

class BotAgent(UserAgent):
    def __init__(self, model, preference_vector):
        # TODO update values based on more thorough assessment
        super().__init__(model, preference_vector, influence_level=0.3, credibility_level=0.9)

class InfluencerAgent(UserAgent):
    def __init__(self, model, preference_vector):
        # TODO update values based on more thorough assessment
        super().__init__(model, preference_vector, influence_level=0.9, credibility_level=0.7)
   

