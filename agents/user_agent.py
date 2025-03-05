from mesa import Agent
import numpy as np
from utils.common import cosine_similarity
from utils.agents_utils import (
    evaluate_content_interest,
    get_network_neighbors
)

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
        self.social_network = model.social_media_platform.social_network
        self.social_media_platform = model.social_media_platform

    def step(self):
        """Execute one step for the agent"""
        processed_feed = self.feed.copy()
        self.feed = []
        
        for content in processed_feed:
            if self.evaluate_content(content):
                self.share_content(content)

    def evaluate_content(self, content):
        """Evaluate if content is interesting enough to share."""
        belief_probability = cosine_similarity(self.preference_vector, content.topic_vector) * self.credibility_level
        
        believe_content = evaluate_content_interest(belief_probability)

        # Update agent state based on content
        if content.isFake:
            if self.state == "S":
                self.state = "E"
        
        return believe_content
    
    def share_content(self, content):
        """Share content with followers."""
         # Update state if sharing fake content
        if content.isFake and self.state == "E":
            self.state = "B"

        followers = self.get_followers()
        
        # Share with each follower
        for follower in followers:
            if content not in follower.feed:
                follower.feed.append(content)
        
       

    def get_followers(self):
        """Get list of agents that follow this agent."""
        return get_network_neighbors(self.model, self.social_network, self.pos, "predecessors")

    def get_following(self):
        """Get list of agents this agent follows."""
        return get_network_neighbors(self.model, self.social_network, self.pos, "successors")

class BotAgent(UserAgent):
    def __init__(self, model, preference_vector):
        # TODO update values based on more thorough assessment
        super().__init__(model, preference_vector, influence_level=0.3, credibility_level=0.9)

class InfluencerAgent(UserAgent):
    def __init__(self, model, preference_vector):
        # TODO update values based on more thorough assessment
        super().__init__(model, preference_vector, influence_level=0.9, credibility_level=0.7)
   

