import random
import pandas as pd
import numpy as np
from lenskit.knn import ItemKNNScorer, ItemKNNConfig
from lenskit.data import from_interactions_df
# from lenskit.data import sparse_ratings
# from scipy.sparse import csr_matrix


class Recommender():
    def __init__(self, type):
        self.type = type
        self.user_interactions = []  # List to store user-content interactions
        self.ratings_df = None
        self.from_interactions_df = None
        
        # Configure ItemKNN with new API
        config = ItemKNNConfig(
            max_nbrs=20,  # Maximum number of neighbors
            min_nbrs=1,   # Minimum number of neighbors
            min_sim=0.1,  # Minimum similarity threshold
            feedback='explicit',  # Using explicit ratings
            block_size=250,  # Block size for parallel computation
        )
        self.item_knn = ItemKNNScorer(config)
        
    def update_recommendations(self, agents):
        """Update recommendations for all agents"""
        for agent in agents:
            if self.type == "random":
                self.random_recommendation(agent)
            elif self.type == "collaborative_filtering":
                self.collaborative_filtering(agent)
            elif self.type == "content_based":
                self.content_based(agent)
            elif self.type == "hybrid":
                self.hybrid(agent)

    def add_interaction(self, agent_id, content_id, rating):
        """Add an interaction between an agent and content item
        
        Args:
            agent_id: The ID of the agent (grid position)
            content_id: The ID of the content
            rating: The rating value (should be between 0 and 1)
            current_step: The current model step (optional)
        """
        # Validate inputs
        if not isinstance(agent_id, (int, np.integer)):
            agent_id = int(agent_id)
        if not isinstance(content_id, (int, np.integer)):
            content_id = int(content_id)
        
        # Ensure rating is between 0 and 1
        rating = max(0.0, min(1.0, float(rating)))
        
        # Create new interaction
        new_interaction = {
            'user': agent_id,
            'item': content_id,
            'rating': rating,
            # 'timestamp': current_step if current_step is not None else len(self.user_interactions)
        }
        
        # Remove any existing interaction for this user-item pair
        self.user_interactions = [
            inter for inter in self.user_interactions 
            if not (inter['user'] == agent_id and inter['item'] == content_id)
        ]
        
        # Add the new interaction
        self.user_interactions.append(new_interaction)
        #print the user interactions with agent.pos

    def recommend(self, agent, content_pool):
        """Recommend news content to an agent"""
        pass

    def _update_rating_matrix(self):
        """Update the ratings DataFrame from interactions"""
        if not self.user_interactions:
            return
            
        # Convert interactions to DataFrame
        self.ratings_df = pd.DataFrame(self.user_interactions)
        
        # Ensure proper data types
        self.ratings_df['user'] = self.ratings_df['user'].astype('int32')
        self.ratings_df['item'] = self.ratings_df['item'].astype('int32')
        self.ratings_df['rating'] = self.ratings_df['rating'].astype('float64')
        
        # Remove duplicates keeping most recent
        self.ratings_df = self.ratings_df.drop_duplicates(['user', 'item'], keep='last')
        
        # Create LensKit Dataset
        self.from_interactions_df = from_interactions_df(self.ratings_df)
        print(self.from_interactions_df)
        print(self.user_interactions)

    def collaborative_filtering(self, agent):
        """Recommend content using item-based collaborative filtering"""
        # Clear previous recommendations
        agent.recommended_content = []
        
        # Get content pool from model
        if not hasattr(agent.model, 'news_content') or not agent.model.news_content:
            # print(f"No content pool available for agent {agent.pos}")
            return

        if len(self.user_interactions) < 20:  # Need some minimum interactions
            # print(f"Not enough interactions ({len(self.user_interactions)}) for CF, using random")
            # self.random_recommendation(agent)
            return
        
        try:
            # Update the rating matrix
            self._update_rating_matrix()
            
            if self.ratings_df is None or len(self.ratings_df) < 5:
                self.random_recommendation(agent)
                return
            
            # Get user's rated items
            user_ratings = self.ratings_df[self.ratings_df['user'] == agent.pos]
            rated_items = user_ratings['item'].values
            print(f"rated_items: {rated_items}")
            
            # User needs at least one interaction to get recommendations
            if len(rated_items) == 0:
                self.random_recommendation(agent)
                return
            
            # Get all available items that have been rated by any user
            all_items = self.ratings_df['item'].unique()
            print(f"all_items: {all_items}")
            # Get candidate items not rated by user
            candidate_items = np.setdiff1d(all_items, rated_items)
            print(f"candidate_items: {candidate_items}")
            if len(candidate_items) == 0:
                print(f"No candidate items found")
                # self.random_recommendation(agent)
                return
            
            # Fit model with the rating matrix
            try:
                # Train the model with the rating matrix
                self.item_knn.train(self.from_interactions_df)
                
                # Create query DataFrame for scoring
                query = pd.DataFrame({
                    'user': [agent.pos] * len(candidate_items),
                    'item': candidate_items
                })
                
                # Score items
                scores = self.item_knn(query)
                
                if scores is None or len(scores) == 0:
                    return
                
                # Convert scores to Series with item indices
                scores = pd.Series(scores, index=candidate_items)
                
                # Remove any NaN scores
                scores = scores.dropna()
                
                if len(scores) == 0:
                    return
                
                # Sort items by score and get top 3
                top_items = scores.nlargest(3).index
                
                # Convert content IDs back to NewsContent objects
                content_dict = {int(c.content): c for c in agent.model.news_content}
                recommendations = [content_dict[int(item)] 
                                for item in top_items 
                                if int(item) in content_dict]
                
                if recommendations:
                    agent.recommended_content.extend(recommendations)
                    
            except Exception as e:
                print(f"Error in collaborative filtering scoring: {e}")
                # self.random_recommendation(agent)
                
        except Exception as e:
            # self.random_recommendation(agent)
            print(f"Error in collaborative filtering: {e}")
            
    def content_based(self):
        pass
    def hybrid(self):
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
            num_recommendations = min(3, len(available_content))
            recommendations = random.sample(available_content, num_recommendations)
            agent.recommended_content.extend(recommendations)
        else:
            # TODO: remove this? Only want to recommend news that is available?
            # If no new content available, recommend from the entire pool
            if content_pool:
                num_recommendations = min(2, len(content_pool))
                recommendations = random.sample(content_pool, num_recommendations)
                agent.recommended_content.extend(recommendations)


