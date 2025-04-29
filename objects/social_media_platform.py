from recommender.recommender import Recommender
from objects.social_network import Social_Network

class SocialMediaPlatform:
    def __init__(self, model, num_agents, m_links, preference_vectors, recommender_type, diversity_level, num_recommendations, use_stored_network=False):
        # Create social network
        self.social_network = Social_Network(model, num_agents, m_links, preference_vectors, use_stored_network)
        
        # Create recommender system
        self.recommender = Recommender(recommender_type, diversity_level, num_recommendations)