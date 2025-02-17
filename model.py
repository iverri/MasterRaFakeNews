from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from numpy import dot
from numpy.linalg import norm
import networkx as nx
from agents.user_agent import BotAgent, InfluencerAgent, UserAgent
from recommender.recommender import Recommender
import random
from objects.social_network import Social_Network

class SocialMediaPlatform:
    def __init__(self, num_agents, m_links):
        self.recommender = Recommender() # TODO: Implement recommender
        self.social_network = Social_Network(num_agents, m_links)


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
    def __init__(self, N, m_links):
        self.num_agents = N # number of agents
        self.m_links = m_links
        self.schedule = RandomActivation(self) # schedule for agents
        self.social_media_platform = SocialMediaPlatform(self.num_agents, self.m_links) # social media platform

        self.grid = NetworkGrid(self.social_media_platform.social_network)
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
            self.grid.place_agent(user, i)

        

    def step(self):
        for agent in self.schedule.agents:
            agent.step()

    def random_preferences(self):
        return [random.random() for i in range(3)]