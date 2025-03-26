from mesa import Agent
import numpy as np
from utils.common import cosine_similarity
from utils.agents_utils import (
    evaluate_content_interest,
    get_network_neighbors
)
import random

class UserAgent(Agent):
    # Initialize the agent
    def __init__(self, model, preference_vector, credibility_level, influence_level):
        super().__init__(model)
        self.preference_vector = preference_vector
        self.credibility_level = credibility_level
        self.influence_level = influence_level
        self.state = "S"  # S: Susceptible, E: Exposed, I: Infected
        self.infection_start_step = 0  # Track when infection started
        self.feed = [] # feed with NewsContent
        self.recommended_content = []
        self.social_network = model.social_media_platform.social_network
        self.social_media_platform = model.social_media_platform
        
        # Activity-related properties
        self.is_active = False
        self.activity_probability = min(max(random.gauss(0.3, 0.1), 0.1), 0.7)  # Base probability of being active
        self.activity_pattern = self._generate_activity_pattern()  # Time-of-day preferences
        self.last_active_step = 0  # Track when the agent was last active

    def _generate_activity_pattern(self):
        """Generate a time-of-day activity pattern for this agent."""
        # Create a 24-hour activity pattern (simplified to 8 time slots for the simulation)
        # Higher values mean higher probability of being active during that time slot
        pattern = [random.random() * 0.5 for _ in range(8)]  # Base low activity
        
        # Most users have 1-3 peak activity times
        num_peaks = random.randint(1, 3)
        for _ in range(num_peaks):
            peak_time = random.randint(0, 7)
            pattern[peak_time] = random.uniform(0.7, 1.0)  # Peak activity time
            
        return pattern

    def step(self):
        """Execute one step for the agent"""
        # Update infection state if infected
        if self.state == "I":
            if self.model.steps - self.infection_start_step >= 40:  # After 40 timesteps
                # Check if agent still has fake news in feed
                has_fake_news = any(content.isFake for content in self.feed)
                self.state = "E" if has_fake_news else "S"
        
        # Determine if the agent becomes active in this step
        self.update_activity_state()
        
        # Manage feed size (even when inactive)
        self._manage_feed()
        
        # Only process feed if the agent is active
        if self.is_active:
            # Process regular feed
            processed_feed = self.feed.copy()
            self.feed = []
            
            # Process both feed and recommendations
            all_content = processed_feed + self.recommended_content
            self.recommended_content = []  # Clear recommendations after processing
            
            for content in all_content:
                if self.evaluate_content(content):
                    self.share_content(content)
            
            self.last_active_step = self.model.steps
        else:
            # Inactive users still accumulate content in their feed
            pass  # Feed remains unchanged until they become active

    def update_activity_state(self):
        """Update whether the agent is active in the current step."""
        # Get the current time slot (0-7) based on the model's step count
        current_time_slot = (self.model.steps % 24) // 3  # 8 time slots per day
        
        # Base probability adjusted by time-of-day pattern
        time_factor = self.activity_pattern[current_time_slot]
        
        # Increase probability if the agent hasn't been active for a while
        steps_since_active = self.model.steps - self.last_active_step
        recency_factor = min(1.0, steps_since_active / 10)  # Caps at 1.0 after 10 steps
        
        # Calculate final probability
        final_probability = self.activity_probability * time_factor * (1 + recency_factor)
        
        # Determine if agent is active
        self.is_active = random.random() < final_probability

    def evaluate_content(self, content):
        """Evaluate if content is interesting enough to share."""
        # Base interest based on topic similarity and credibility
        belief_probability = cosine_similarity(self.preference_vector, content.topic_vector) * self.credibility_level
        
        # Adjust probability based on content engagement
        engagement_factor = min(1.5, content.engagement)  # Cap the boost at 1.5x
        adjusted_probability = belief_probability * engagement_factor
        
        believe_content = evaluate_content_interest(adjusted_probability)

        # Update agent state based on content
        if content.isFake:
            if self.state == "S":
                self.state = "E"
        
        return believe_content
    
    def share_content(self, content):
        """Share content with followers."""
        # Update state if sharing fake content
        if content.isFake:
            if self.state == "E":
                self.state = "I"
                self.infection_start_step = self.model.steps  # Record when infection started

        followers = self.get_followers()

        # Record interaction even if no followers (for CF)
        # Calculate base rating from content similarity
        base_rating = cosine_similarity(self.preference_vector, content.topic_vector)
        
        # Adjust rating based on content engagement (normalized to 0-1 range)
        engagement_factor = min(1.0, content.engagement / 1.5)  # Normalize by max possible engagement
        
        # Combine ratings (70% similarity, 30% engagement)
        final_rating = 0.7 * base_rating + 0.3 * engagement_factor
        
        # Record interaction for collaborative filtering
        self.model.social_media_platform.recommender.add_interaction(
            self.pos, 
            content.content, 
            final_rating
        )
        
        # Only share with followers if we have any
        if followers:
            for follower in followers:
                if content not in follower.feed:
                    follower.feed.append(content)

    def get_followers(self):
        """Get list of agents that follow this agent."""
        return get_network_neighbors(self.model, self.social_network, self.pos, "predecessors")

    def get_following(self):
        """Get list of agents this agent follows."""
        return get_network_neighbors(self.model, self.social_network, self.pos, "successors")

    def _manage_feed(self):
        """Manage feed size and content relevance."""
        # Update engagement of all content in feed
        for content in self.feed:
            content.update_engagement(self.model.steps)
        
        

class BotAgent(UserAgent):
    def __init__(self, model, preference_vector):
        super().__init__(model, preference_vector, influence_level=0.3, credibility_level=0.9)
        # Bots are more consistently active
        self.activity_probability = min(max(random.gauss(0.7, 0.15), 0.4), 0.9)
        self.activity_pattern = [random.uniform(0.7, 1.0) for _ in range(8)]  # More consistent

class InfluencerAgent(UserAgent):
    def __init__(self, model, preference_vector):
        super().__init__(model, preference_vector, influence_level=0.8, credibility_level=0.7)
        # Influencers are more active than regular users
        self.activity_probability = min(max(random.gauss(0.5, 0.15), 0.3), 0.8)
        # Influencers might have more strategic posting times
        self.activity_pattern = self._generate_influencer_activity_pattern()
        
    def _generate_influencer_activity_pattern(self):
        """Generate activity pattern optimized for audience engagement."""
        pattern = [random.random() * 0.3 for _ in range(8)]  # Base low activity
        # Influencers focus on 2-3 optimal posting times
        num_peaks = random.randint(2, 3)
        for _ in range(num_peaks):
            # Peak times often align with high general user activity
            peak_time = random.choices(range(8), weights=[0.1, 0.15, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1])[0]
            pattern[peak_time] = random.uniform(0.8, 1.0)  # Very high activity during peak times
        return pattern
   

