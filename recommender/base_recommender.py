from abc import ABC, abstractmethod


class BaseRecommender(ABC):

    @abstractmethod
    def update_recommendations(self, agents):
        pass

    @abstractmethod
    def add_interaction(self, agent_id, content_id, rating):
        pass
