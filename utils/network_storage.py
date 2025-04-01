import networkx as nx

class NetworkStorage:
    """Singleton class to store network between model runs"""
    _instance = None
    _network = None
    _preference_vectors = None
    
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