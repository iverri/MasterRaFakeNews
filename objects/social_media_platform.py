from recommender.factory import create_recommender
from objects.social_network import Social_Network


class SocialMediaPlatform:
    def __init__(
        self,
        model,
        num_agents,
        m_links,
        preference_vectors,
        personality_vectors,
        library,
        recommender_type,
        diversity_level,
        num_recommendations,
        use_stored_network=False,
        network_file=None,
    ):
        # Create social network
        self.social_network = Social_Network(
            model,
            num_agents,
            m_links,
            preference_vectors,
            personality_vectors,
            use_stored_network,
            network_file,
        )

        # Create recommender system
        self.recommender = create_recommender(
            library, recommender_type, diversity_level, num_recommendations
        )
