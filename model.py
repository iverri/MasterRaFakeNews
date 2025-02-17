from mesa import Model, Agent
from mesa.time import RandomActivation
from numpy import dot
from numpy.linalg import norm
from agents.user_agent import BotAgent, InfluencerAgent, UserAgent
from recommender.recommender import Recommender
import random

class SocialMediaPlatform:
    def __init__(self):
        self.social_network = [] # TODO: Implement network
        self.recommender = Recommender() # TODO: Implement recommender

# Create a model class
class FakeNewsModel(Model):
    '''This model simulates the spread of fake news in a social network.  
    At each timestep, users receive a content feed, engage with news,  
    and may transition from Susceptible (S) → Exposed (E) → Believer (B).  
    Bots and influencers accelerate spread, while moderation reduces visibility.  
    The process repeats over multiple timesteps, influencing network dynamics. 
    '''

    #Initialize number of agents
    #Initialize agents
    def __init__(self, N):
        self.num_agents = N # number of agents
        self.schedule = RandomActivation(self) # schedule for agents
        self.social_media_platform = SocialMediaPlatform() # social media platform

        # Create agents
        for i in range(self.num_agents):
            preference_vector = self.random_preferences()
            if i % 5 == 0: # every 5th agent is a bot: Kan endre!
                user = BotAgent(self, preference_vector, i)
            elif i % 6 == 0: # every 6th agent is an influencer: Kan endre!
                user = InfluencerAgent(self, preference_vector, i)
            else:
                user = UserAgent(self, preference_vector, i)
            self.schedule.add(user)

    def step(self):
        for agent in self.schedule.agents:
            agent.step()

    def random_preferences(self):
        return [random.random() for i in range(3)]