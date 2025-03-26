
from recommender.recommender import Recommender
from objects.social_network import Social_Network

class SocialMediaPlatform:
    def __init__(self, model, num_agents, m_links, preference_vectors, recommender_type, diversity_lambda, increase_diversity, num_recommendations):
        self.social_network = Social_Network(model, num_agents, m_links, preference_vectors)
        self.recommender = Recommender(recommender_type, diversity_lambda, increase_diversity, num_recommendations)