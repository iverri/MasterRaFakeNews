from enum import Enum

class RecommenderType(Enum):
    RANDOM = "random"
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    ITEM_KNN = "item_knn"
    USER_KNN = "user_knn"
    CONTENT_BASED = "content_based"
    POPULAR = "popular"
