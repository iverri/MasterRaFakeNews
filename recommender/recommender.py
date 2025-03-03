
from objects.news_content import NewsContent
from utils.common import generate_random_topic_vector



class Recommender:
    def __init__(self):
        self.news_content = []
        self.user_preferences = []
        self.recommendations = []

    

