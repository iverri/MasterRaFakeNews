
from recommender.recommender import Recommender
from objects.social_network import Social_Network

class SocialMediaPlatform:
    def __init__(self, num_agents, m_links):
        self.recommender = Recommender() # TODO: Implement recommender
        self.social_network = Social_Network(num_agents, m_links)