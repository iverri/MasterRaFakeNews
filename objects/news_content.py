# News content class
class NewsContent:
    def __init__(self, content_id, isFake, topic_vector):
        self.content = content_id
        self.isFake = isFake
        self.topic_vector = topic_vector