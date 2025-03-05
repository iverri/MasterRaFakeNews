import random

class Recommender():
    def __init__(self, type="random"):
        self.news_content = []
        self.type = type
        self.user_preferences = []
        self.user_interactions = []

    def update_recommendations(self, agents):
        """Update recommendations for all agents"""
        for agent in agents:
            if self.type == "random":
                self.random_recommendation(agent)
            elif self.type == "collaborative_filtering":
                self.collaborative_filtering(agent)
            elif self.type == "content_based":
                self.content_based(agent)
            elif self.type == "hybrid":
                self.hybrid(agent)

    def recommend(self, agent, content_pool):
        """Recommend news content to an agent"""
        pass

    # Recommender algorithms
    def collaborative_filtering(self, agent):
        pass

    def content_based(self):
        pass
    def hybrid(self):
        pass
    def random_recommendation(self, agent):
        """Recommend random news content to an agent"""
        # Clear previous recommendations
        agent.recommended_content = []
        
        # Get content pool from model and ensure it exists
        if not hasattr(agent.model, 'news_content'):
            return
            
        if not agent.model.news_content:
            return
            
        content_pool = agent.model.news_content
        
        # Get content that isn't in the agent's current feed
        available_content = [c for c in content_pool if c not in agent.feed]
        
        if available_content:
            # Always recommend 3 items if possible
            num_recommendations = min(3, len(available_content))
            recommendations = random.sample(available_content, num_recommendations)
            agent.recommended_content.extend(recommendations)
        else:
            # TODO: remove this? Only want to recommend news that is available?
            # If no new content available, recommend from the entire pool
            if content_pool:
                num_recommendations = min(2, len(content_pool))
                recommendations = random.sample(content_pool, num_recommendations)
                agent.recommended_content.extend(recommendations)


