import random
import pandas as pd
import numpy as np
from lenskit.knn import ItemKNNScorer, ItemKNNConfig
from lenskit.data import Dataset, ItemList
from lenskit.pipeline import Pipeline, topn_pipeline
# from lenskit.data import from_interactions_df
# from lenskit.data import sparse_ratings
# from scipy.sparse import csr_matrix
import traceback

from recommender.types import RecommenderType


class Recommender():
    def __init__(self, recommender_type):
        self.type = recommender_type
        self.user_interactions = []  # List to store user-content interactions
        self._last_training_count = 0  # Add this line to track when retraining is needed
        
        # Configure ItemKNN with new API
        config = ItemKNNConfig(
            max_nbrs=20,  # Maximum number of neighbors
            min_nbrs=1,   # Minimum number of neighbors
            min_sim=0.1,  # Minimum similarity threshold
            feedback='explicit',  # Using explicit ratings
        )
        
        # Create the scorer
        self.item_knn = ItemKNNScorer(config)
        
        # Create a recommendation pipeline
        self.pipeline = None
        
    def update_recommendations(self, agents):
        """Update recommendations for all agents"""
        print(f"Recommender type: {self.type}")
        for agent in agents:
            if self.type == RecommenderType.RANDOM.value:
                self.random_recommendation(agent)
            elif self.type == RecommenderType.COLLABORATIVE_FILTERING.value:
                self.collaborative_filtering(agent)
            elif self.type == RecommenderType.ITEM_KNN.value:
                self.collaborative_filtering(agent)
            elif self.type == RecommenderType.USER_KNN.value:
                self.random_recommendation(agent)
            elif self.type == RecommenderType.CONTENT_BASED.value:
                self.content_based(agent)
            elif self.type == RecommenderType.HYBRID.value:
                self.hybrid(agent)

    def add_interaction(self, agent_id, content_id, rating):
        """Add an interaction between an agent and content item"""
        # Validate inputs
        if not isinstance(agent_id, (int, np.integer)):
            agent_id = int(agent_id)
        if not isinstance(content_id, (int, np.integer)):
            content_id = int(content_id)
        
        # Ensure rating is between 0 and 1
        rating = max(0.0, min(1.0, float(rating)))
        
        # Create new interaction
        new_interaction = {
            'user_id': agent_id,
            'item_id': content_id,
            'rating': rating,
        }
        
        # Remove any existing interaction for this user-item pair
        self.user_interactions = [
            inter for inter in self.user_interactions 
            if not (inter['user_id'] == agent_id and inter['item_id'] == content_id)
        ]
        
        # Add the new interaction
        self.user_interactions.append(new_interaction)

    def _create_dataset(self):
        """Create a LensKit Dataset from interactions"""
        if not self.user_interactions:
            return None
        
        # Convert interactions to DataFrame
        df = pd.DataFrame(self.user_interactions)
        
        # Ensure proper data types
        df['user_id'] = df['user_id'].astype('int32')
        df['item_id'] = df['item_id'].astype('int32')
        df['rating'] = df['rating'].astype('float64')
        
        # Remove duplicates keeping most recent
        df = df.drop_duplicates(['user_id', 'item_id'], keep='last')
        
        # Create dataset using from_interactions_df
        try:
            from lenskit.data import from_interactions_df
            dataset = from_interactions_df(df)
            return dataset
        except Exception as e:
            print(f"Error creating dataset: {e}")
            return None

    def collaborative_filtering(self, agent):
        """Recommend content using item-based collaborative filtering"""
        # Clear previous recommendations
        agent.recommended_content = []
        
        # Get content pool from model
        if not hasattr(agent.model, 'news_content') or not agent.model.news_content:
            return

        # Check if this specific user has enough interactions (at least 2)
        user_interactions = [inter for inter in self.user_interactions if inter['user_id'] == agent.pos]
        if len(user_interactions) < 3:
            # Fall back to random recommendations if user doesn't have enough interactions
            self.random_recommendation(agent)
            return
        
        # Check if the system as a whole has enough interactions
        min_interactions = 30  # Minimum total interactions needed
        if len(self.user_interactions) < min_interactions:
            # Fall back to random recommendations if not enough data overall
            self.random_recommendation(agent)
            return
        
        try:
            # Create dataset
            dataset = self._create_dataset()
            if dataset is None:
                self.random_recommendation(agent)
                return
                
            # Create and train pipeline if not already created or if it needs retraining
            if self.pipeline is None or len(self.user_interactions) > self._last_training_count:
                print(f"Training new pipeline with {len(self.user_interactions)} interactions")
                if self.pipeline is None:
                    self.pipeline = topn_pipeline(self.item_knn)
                self.pipeline.train(dataset)
                self._last_training_count = len(self.user_interactions)
            
            # Get all available content IDs
            all_content_ids = {c.content for c in agent.model.news_content}
            
            # Get items already in user's feed
            feed_ids = {c.content for c in agent.feed}
            
            # Get candidate items (not in feed)
            candidate_ids = all_content_ids - feed_ids
            if not candidate_ids:
                return
                
            # Convert to list for recommendation
            from lenskit.data import ItemList
            candidates = ItemList(list(candidate_ids))
            
            # Get recommendations using the recommend function
            from lenskit import recommend
            try:
                
                recs = recommend(self.pipeline, agent.pos, n=5, items=candidates)
                
                # Convert to NewsContent objects
                content_dict = {c.content: c for c in agent.model.news_content}
                recommendations = []
                
                rec_ids = recs.ids()
                # Process the recommendations
                for item_id in rec_ids:
                    if item_id in content_dict:
                        recommendations.append(content_dict[item_id])
                
                if recommendations:
                    agent.recommended_content.extend(recommendations)
                    print(f"Added {len(recommendations)} recommendations to agent {agent.pos}")
                else:
                    # Fall back to random if no recommendations were generated
                    self.random_recommendation(agent)
            except Exception as e:
                print(f"Error getting recommendations: {e}")
                traceback.print_exc()  # Add traceback for better debugging
                self.random_recommendation(agent)
                
        except Exception as e:
            print(f"Error in collaborative filtering: {e}")
            traceback.print_exc()  # Add traceback for better debugging
            self.random_recommendation(agent)
            
    def content_based(self, agent):
        pass
        
    def hybrid(self, agent):
        pass
        
    def random_recommendation(self, agent):
        """Recommend random news content to an agent"""
        # Clear previous recommendations
        agent.recommended_content = []
        
        # Get content pool from model and ensure it exists
        if not hasattr(agent.model, 'news_content'):
            return
            
        if not agent.model.news_content:
            return
            
        content_pool = agent.model.news_content
        
        # Get content that isn't in the agent's current feed
        available_content = [c for c in content_pool if c not in agent.feed]
        
        if available_content:
            # Always recommend 3 items if possible
            num_recommendations = min(5, len(available_content))
            recommendations = random.sample(available_content, num_recommendations)
            agent.recommended_content.extend(recommendations)


