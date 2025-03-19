from enum import Enum

class RecommenderType(Enum):
    RANDOM = "random"
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
