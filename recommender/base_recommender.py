from abc import ABC, abstractmethod


# Defines methods all recommenders have to include
class BaseRecommender(ABC):

    @abstractmethod
    def update_recommendations(self, agents):
        pass

    @abstractmethod
    def add_interaction(self, agent_id, content_id, rating):
        pass
