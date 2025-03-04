from agents.user_agent import UserAgent
from objects.news_content import NewsContent
# from utils.similarity import generate_random_topic_vector
import random
from utils.common import generate_random_topic_vector

# generate random topic vector
generate_random_topic_vector = lambda: [random.random() for i in range(10)]
content_pool = [NewsContent(i, generate_random_topic_vector(), False) for i in range(100)] # Create 100 news content items

class Recommender():
    def __init__(self, type="random"):
        self.news_content = []
        self.user_preferences = []
        self.recommendations = []
        self.type = type

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
        
        # Get a random sample of 5 content items from the pool
        num_recommendations = 5
        recommendations = random.sample(content_pool, min(num_recommendations, len(content_pool)))
        
        # Add the recommendations to the agent's recommended content
        agent.recommended_content.extend(recommendations)



