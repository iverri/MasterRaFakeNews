import numpy as np
from utils.objects_utils import (
    create_preference_based_network,
    create_basic_network,
    adjust_node_connectivity,
)
from utils.network_storage import NetworkStorage

class Social_Network:

    def __init__(self, model, num_agents, m_links, preference_vectors=None, use_stored_network=False):
        self.model = model
        self.num_agents = num_agents
        self.m_links = m_links
        
        # Calculate number of each agent type
        self.num_influencers = int(model.influencer_percentage * num_agents)
        self.num_bots = int(model.bot_percentage * num_agents)
        self.num_regular = num_agents - (self.num_influencers + self.num_bots)
        
        # Check if we should use a stored network
        storage = NetworkStorage()
        
        if use_stored_network and storage.has_stored_network():
            # Use the stored network
            self.network = storage.get_network()
            print("Using stored network")
            # If we're using a stored network, we might want to use stored preference vectors too
            stored_preferences = storage.get_preference_vectors()
            if stored_preferences is not None and len(stored_preferences) == num_agents:
                # Replace the model's preference vectors with stored ones
                model.preference_vectors = stored_preferences

        else:
            # Create a new network
            if preference_vectors is not None:
                self.network = create_preference_based_network(model, num_agents, m_links, preference_vectors)
            else:
                # Fallback to simpler network if no preferences provided
                self.network = create_basic_network(num_agents, m_links)
            
            # Always store the network for future use
            storage.store_network(self.network, model.preference_vectors)

    def _debug_network(self, stage):
        """Debug helper to print network stats at different stages"""
        # Calculate number of each agent type
        num_influencers = self.num_influencers
        num_bots = self.num_bots
        num_agents = self.num_agents
        
        # Group nodes by type
        bot_indices = list(range(num_agents - num_bots, num_agents))
        user_indices = list(range(num_influencers, num_agents - num_bots))
        influencer_indices = list(range(num_influencers))
        
        # Calculate follower counts
        bot_followers = [self.network.in_degree(i) for i in bot_indices]
        user_followers = [self.network.in_degree(i) for i in user_indices]
        influencer_followers = [self.network.in_degree(i) for i in influencer_indices]
        
        avg_bot_followers = sum(bot_followers) / len(bot_followers) if bot_followers else 0
        avg_user_followers = sum(user_followers) / len(user_followers) if user_followers else 0
        avg_influencer_followers = sum(influencer_followers) / len(influencer_followers) if influencer_followers else 0
        
        print(f"\n{stage} - FOLLOWER COUNTS:")
        print(f"Influencers: {avg_influencer_followers:.1f}")
        print(f"Regular Users: {avg_user_followers:.1f}")
        print(f"Bots: {avg_bot_followers:.1f}")
        
        # Print individual bot followers
        print(f"Individual bot followers: {bot_followers}")
        print(f"Individual user followers: {user_followers}")
        print(f"Individual influencer followers: {influencer_followers}")
    
    # Not in use
    def update_network(self, probability=0.1):
        """
        Update network structure to simulate real social network dynamics.
        
        Args:
            probability (float): Probability of network change
        """
            
        if np.random.random() < probability:  # Default 10% chance of network change
            if np.random.random() < 0.5:
                # Add edge between nodes
                nodes = list(self.network.nodes())
                source = np.random.choice(nodes)
                # Prefer connecting similar nodes
                target = np.random.choice(nodes)
                if source != target and not self.network.has_edge(source, target):
                    self.network.add_edge(source, target)
            else:
                # Remove edge, preferentially between dissimilar nodes
                edges = list(self.network.edges())
                if edges:
                    # Choose a random index instead of using np.random.choice on the edges list
                    random_index = np.random.randint(0, len(edges))
                    edge = edges[random_index]
                    self.network.remove_edge(*edge)

    
