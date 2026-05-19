from mesa import Agent
import numpy as np
from utils.agents_utils import get_network_neighbors
from utils.metrics import cosine_similarity
import random


class UserAgent(Agent):
    # Define class constants
    INFECTION_DURATION = 40  # Steps before infected agents can recover
    FEED_CLEANUP_INTERVAL = 5  # Steps between feed cleanup operations
    THOROUGH_CLEANUP_INTERVAL = 10  # Steps between thorough cleanup
    MAX_RECENT_CONTENT = 30  # Maximum items in recent_content
    MAX_SHARED_CONTENT = 50  # Maximum items in shared_content
    ENGAGEMENT_THRESHOLD = 0.2  # Minimum engagement to keep content
    LIKE_THRESHOLD = 0.6  # Threshold for liking content
    SHARE_THRESHOLD = 0.8  # Threshold for sharing content
    COEFFICIENTS = {
        "p0": 0.10,
        "alpha_E": 0.30,
        "alpha_A": 0.25,
        "alpha_C": 0.20,
        "alpha_N": 0.10,
    }

    # Initialize the agent
    def __init__(self, model, preference_vector, personality_vector=None):
        super().__init__(model)
        self.preference_vector = preference_vector
        self.personality_vector = personality_vector
        # self.p_share = self._calculate_p_share(
        #     self.personality_vector, self.COEFFICIENTS
        # )
        self.naivety_level = min(max(random.gauss(0.5, 0.15), 0), 1)
        # whatto do with this
        self.state = "S"  # S: Susceptible, E: Exposed, I: Infected
        self.infection_start_step = 0  # Track when infection started
        self.feed = []  # feed with NewsContent
        self.recommended_content = []
        self.shared_content = []  # Track content this agent has shared
        self.recent_content = []
        self.social_network = model.social_media_platform.social_network
        self.social_media_platform = model.social_media_platform
        self.diversity_score = 0
        self.original_diversity_score = 0
        # Activity-related properties
        self.is_active = False
        self.activity_probability = min(
            max(random.gauss(0.3, 0.1), 0.1), 0.7
        )  # Base probability of being active
        self.activity_pattern = (
            self._generate_activity_pattern()
        )  # Time-of-day preferences
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
            if self.model.steps - self.infection_start_step >= self.INFECTION_DURATION:
                # Check if agent still has fake news in feed
                has_fake_news = any(content.isFake for content in self.feed)
                self.state = "E" if has_fake_news else "S"

        # Determine if the agent becomes active in this step
        self.update_activity_state()

        # Manage feed size (even when inactive)
        self._manage_feed()

        # Only process feed if the agent is active
        if self.is_active:
            # Process both feed and recommendations
            all_content = self.feed + self.recommended_content

            # Evaluate all content in feed and recommendations
            for content in all_content:
                self.evaluate_content(content)

            self.feed = []
            self.recommended_content = []
            self.last_active_step = self.model.steps

            # Attempt to post new content
            self.post_content()

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
        final_probability = (
            self.activity_probability * time_factor * (1 + recency_factor)
        )

        # Determine if agent is active
        self.is_active = random.random() < final_probability

    def evaluate_content(self, content):
        """Evaluate if content is interesting enough to share."""
        # If content is fake, update agent state
        if content.isFake:
            if self.state == "S":
                self.state = "E"

        # Base interest based on topic similarity and credibility
        user_preference = np.array(self.preference_vector).reshape(1, -1)
        content_topic = np.array(content.topic_vector).reshape(1, -1)
        user_evaluation = (
            cosine_similarity(user_preference, content_topic) * self.naivety_level
        )

        # Adjust probability based on content engagement
        engagement_factor = min(1.5, content.engagement)  # Cap the boost at 1.5x
        adjusted_evaluation = user_evaluation * engagement_factor

        # Like content but not share
        if adjusted_evaluation > self.LIKE_THRESHOLD:
            self.model.social_media_platform.recommender.add_interaction(
                self.pos, content.content, user_evaluation
            )
            if content.isFake:
                if self.state == "E":
                    self.state = "I"
                    self.infection_start_step = (
                        self.model.steps
                    )  # Record when infection started
        # Share content
        if (
            adjusted_evaluation
            > self.SHARE_THRESHOLD
            #            or np.random.random() < self.p_share
        ):
            self.share_content(content, user_evaluation)

    def share_content(self, content, user_evaluation):
        """Share content with followers."""
        followers = self.get_followers()

        # Track this content as shared by this agent
        self.shared_content.append(
            {
                "content": content,
                "step": self.model.steps,
                "followers_received": len(followers),
                "evaluation": user_evaluation,
            }
        )

        # Only share with followers if we have any
        if followers:
            for follower in followers:
                if content not in follower.feed:
                    follower.feed.append(content)

    def get_followers(self):
        """Get list of agents that follow this agent."""
        return get_network_neighbors(
            self.model, self.social_network, self.pos, "predecessors"
        )

    def get_following(self):
        """Get list of agents this agent follows."""
        return get_network_neighbors(
            self.model, self.social_network, self.pos, "successors"
        )

    def _manage_feed(self):
        """Manage feed size and content relevance."""
        # Update engagement of all content in feed AND recommendations
        # This is critical because recommendations can accumulate when agents are inactive
        for content in self.feed + self.recommended_content:
            content.update_engagement(self.model.steps)

        # Clean up old shared content (keep only last 50 items or last 20 steps)
        if len(self.shared_content) > self.MAX_SHARED_CONTENT:
            current_step = self.model.steps
            self.shared_content = [
                item
                for item in self.shared_content
                if (current_step - item["step"] <= 20)
                or (len(self.shared_content) <= self.MAX_SHARED_CONTENT)
            ]

        # Optimize recent_content management
        current_step = self.model.steps

        # Only process recent_content every few steps to reduce overhead
        if current_step % self.FEED_CLEANUP_INTERVAL == 0:  # Only update every 3 steps
            # Use a set to track content IDs for faster duplicate checking
            existing_content_ids = {
                item["content"].content for item in self.recent_content
            }

            # Add new content to recent_content (avoiding duplicates)
            for content in self.feed + self.recommended_content:
                if content.content not in existing_content_ids:
                    self.recent_content.append(
                        {"content": content, "step": current_step}
                    )
                    existing_content_ids.add(content.content)

            # Only sort and trim if we have more than the target number
            if len(self.recent_content) > self.MAX_RECENT_CONTENT:
                # Keep only the 20 most recent items
                self.recent_content.sort(key=lambda x: x["step"], reverse=True)
                self.recent_content = self.recent_content[: self.MAX_RECENT_CONTENT]

        # Every 10 steps, do a more thorough cleanup to remove very old content
        if current_step % self.THOROUGH_CLEANUP_INTERVAL == 0 and self.recent_content:
            self.recent_content = [
                item
                for item in self.recent_content
                if (current_step - item["step"] <= 30)
            ]

    def post_content(self):
        """Generate and post new content."""
        # Determine if the agent decides to post content
        if random.random() < self.activity_probability:
            # Create a topic vector similar to the preference vector
            topic_vector = self._generate_similar_topic_vector()
            # Determine the fake news percentage based on the agent type
            if isinstance(self, BotAgent):
                fake_news_percentage = self.model.fake_news_percentage * 1.5
            else:
                fake_news_percentage = (
                    self.model.fake_news_percentage * self.naivety_level
                )

            # Create new content
            from objects.news_content import NewsContent

            is_fake = random.random() < fake_news_percentage
            new_content = NewsContent(
                len(self.model.news_content),
                is_fake,
                topic_vector,
                self.model.steps,  # Add creation_step here
            )
            # Add the new content to the model's news content
            self.model.news_content.append(new_content)
            # Add the new content to followers' feed
            for follower in self.get_followers():
                follower.feed.append(new_content)

    def _generate_similar_topic_vector(self):
        """Generate a topic vector similar to the agent's preference vector."""
        # Add small random noise to the preference vector
        noise = np.random.normal(0, 0.05, len(self.preference_vector))
        topic_vector = np.array(self.preference_vector) + noise

        # Normalize the topic vector to maintain it as a unit vector
        magnitude = np.linalg.norm(topic_vector)
        if magnitude > 0:
            topic_vector = topic_vector / magnitude

        return topic_vector

    def _calculate_p_share(self, personality_vector, coefficients):
        p_share = (
            coefficients["p0"]
            + coefficients["alpha_E"] * personality_vector[0]
            - coefficients["alpha_A"] * personality_vector[1]
            - coefficients["alpha_C"] * personality_vector[2]
            + coefficients["alpha_N"] * personality_vector[3]
        )

        return max(0, min(p_share, 1))  # restrict value between 0 and 1


class BotAgent(UserAgent):
    def __init__(self, model, preference_vector, personality_vector=None):
        super().__init__(model, preference_vector, personality_vector)

        self.naivety_level = min(max(random.gauss(0.9, 0.05), 0), 1)
        # Bots are more consistently active
        self.activity_probability = min(max(random.gauss(0.7, 0.15), 0.4), 0.9)
        self.activity_pattern = [
            random.uniform(0.7, 1.0) for _ in range(8)
        ]  # More consistent


class InfluencerAgent(UserAgent):
    def __init__(self, model, preference_vector, personality_vector=None):
        super().__init__(model, preference_vector, personality_vector)
        self.naivety_level = min(max(random.gauss(0.5, 0.15), 0), 1)
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
            peak_time = random.choices(
                range(8), weights=[0.1, 0.15, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1]
            )[0]
            pattern[peak_time] = random.uniform(
                0.8, 1.0
            )  # Very high activity during peak times
        return pattern
