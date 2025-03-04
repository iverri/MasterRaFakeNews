
from objects.news_content import NewsContent
# from utils.similarity import generate_random_topic_vector
import random
# from agents.user_agent import UserAgent
from lenskit.datasets import ML100K
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn
from utils.common import generate_random_topic_vector



# generate random topic vector
generate_random_topic_vector = lambda: [random.random() for i in range(10)]

# user_agents = [UserAgent(i, generate_random_topic_vector(), False) for i in range(10)] # Create 10 user agents
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
        user_user = knn.UserUser(15)
        user_user.fit(self.news_content)
        recs = user_user.recommend(agent, 10)
        # agent.recommended_content = recs
        item_item = knn.ItemItem(15)
        item_item.fit(self.news_content)
        recs = item_item.recommend(agent, 10)
        # agent.recommended_content = recs
        recommendations = user_user.recommend(agent, 10)
        print(recommendations)

    def content_based(self):
        pass
    def hybrid(self):
        pass
    def random_recommendation(self, agent):
        """Recommend random news content to an agent"""
        # use the content_pool to recommend a random news content as candidate
        for content in content_pool:
            # Select if content should be recommended or not
            if random.random() < 0.5: # TODO: hvordan velge hvilken agent som skal få hvilket anbefalt innhold?
                agent.recommended_content.append(content)
            else:
                continue

    

