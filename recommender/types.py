from enum import Enum

class RecommenderType(Enum):
    RANDOM = "random"
    ITEM_KNN = "item_knn"
    USER_KNN = "user_knn"
    CONTENT_BASED = "content_based"
    POPULAR = "popular"
    HYBRID_WEIGHTED_DYNAMIC = "hybrid_weighted_dynamic"
    HYBRID_WEIGHTED_STATIC = "hybrid_weighted_static"
    MATRIX_FACTORIZATION = "matrix_factorization"
    
