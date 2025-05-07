import networkx as nx
import pickle
import os

class NetworkStorage:
    """Class to store network between model runs"""
    _instance = None
    _network = None
    _preference_vectors = None
    global_communities = None  # Added to store communities across all model instances
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NetworkStorage, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def store_network(cls, network, preference_vectors=None):
        """Store a network for future use"""
        cls._network = network.copy() if network else None
        cls._preference_vectors = preference_vectors.copy() if preference_vectors else None
    
    @classmethod
    def store_network_to_file(cls, network, preference_vectors=None, filename=None):
        """Store a network to a file for future use across processes"""
        if filename is None:
            filename = "network_storage.pkl"
        
        data = {
            'network': network.copy() if network else None,
            'preference_vectors': preference_vectors.copy() if preference_vectors else None
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        # Also store in memory for non-parallel usage
        cls.store_network(network, preference_vectors)
        
        return filename
    
    @classmethod
    def load_network_from_file(cls, filename):
        """Load a network from a file"""
        if not os.path.exists(filename):
            return None, None
            
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        network = data.get('network')
        preference_vectors = data.get('preference_vectors')
        
        # Also store in memory for non-parallel usage
        cls.store_network(network, preference_vectors)
        
        return network, preference_vectors
    
    @classmethod
    def get_network(cls):
        """Retrieve the stored network"""
        return cls._network.copy() if cls._network else None
    
    @classmethod
    def get_preference_vectors(cls):
        """Retrieve the stored preference vectors"""
        return cls._preference_vectors.copy() if cls._preference_vectors else None
    
    @classmethod
    def has_stored_network(cls):
        """Check if a network is stored"""
        return cls._network is not None
    
    @classmethod
    def clear(cls):
        """Clear the stored network"""
        cls._network = None
        cls._preference_vectors = None
        cls.global_communities = None  # Also clear the communities when clearing the network