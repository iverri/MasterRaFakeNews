import random
import pandas as pd
import numpy as np
from lenskit.algorithms import item_knn as knn
from lenskit.data import sparse_ratings
from scipy.sparse import csr_matrix


class Recommender():
    def __init__(self, type):
        self.type = type
        self.user_interactions = []  # List to store user-content interactions
        self.rating_matrix = None  # Store the user-item matrix
        self.user_index = None  # Store user index mapping
        self.item_index = None  # Store item index mapping
        self.ratings_df = None
        
        # Configure ItemItem for explicit feedback with proper parameters
        self.item_knn = knn.ItemItem(
            20,         # n_neighbors
            min_nbrs=0,
            min_sim=0.0,  # Non-negative similarities for our rating scale
            feedback='explicit',  # Using explicit ratings
            center=False,  # Don't center ratings to preserve scale
            aggregate='weighted-average'  # Use weighted average for predictions
        )
        
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
        """Update the user-item rating matrix from interactions"""
        # print(f"User interactions count: {len(self.user_interactions)}")
        # print(f"User interactions: {self.user_interactions}")

        if not self.user_interactions:
            return
            
        # Convert interactions to DataFrame
        self.ratings_df = pd.DataFrame(self.user_interactions)
        
        # Ensure proper data types
        self.ratings_df['user'] = self.ratings_df['user'].astype('int32')
        self.ratings_df['item'] = self.ratings_df['item'].astype('int32')
        self.ratings_df['rating'] = self.ratings_df['rating'].astype('float64')
        
        # print("Users in ratings_df before sparse conversion:", ratings_df['user'].unique())

        # Remove duplicates keeping most recent
        self.ratings_df = self.ratings_df.drop_duplicates(['user', 'item'], keep='last')
        # print(f"ratings_df: {self.ratings_df}")
        
        # Create sparse rating matrix
        self.rating_matrix, raw_user_index, self.item_index = sparse_ratings(self.ratings_df)

        # Convert to Python int to avoid type mismatch
        self.user_index = pd.Index([int(uid) for uid in raw_user_index])

        # print(f"User index after sparse_ratings(): {self.user_index}")
        # print(f"Item index: {self.item_index}")

    def collaborative_filtering(self, agent):
        """Recommend content using item-based collaborative filtering"""
        # Clear previous recommendations
        agent.recommended_content = []
        
        # Get content pool from model
        if not hasattr(agent.model, 'news_content') or not agent.model.news_content:
            # print(f"No content pool available for agent {agent.pos}")
            return

        if len(self.user_interactions) < 5:  # Need some minimum interactions
            # print(f"Not enough interactions ({len(self.user_interactions)}) for CF, using random")
            self.random_recommendation(agent)
            return
        
        try:
            # Update the rating matrix
            self._update_rating_matrix()
            # print(f"User index mapping: {self.user_index}")
            # print(f"Agent position: {agent.pos} (type: {type(agent.pos)})") 
            # Need minimum interactions for meaningful recommendations
            if self.rating_matrix is None or self.rating_matrix.nnz < 5:
                # print(f"Not enough interactions ({self.rating_matrix.nnz}) for CF, using random")
                self.random_recommendation(agent)
                return
            
            # Get user's rated items using matrix indices
            try:
                # print(f"User index: {list(self.user_index)}")
                # print(f"Agent pos: {agent.pos}")
                # If user is not in the index, they haven't had any interactions yet
                if agent.pos not in list(self.user_index):
                    # print(f"User {agent.pos} not in index. Available users: {list(self.user_index)}")
                    # For new users, use random recommendations until they have interactions
                    self.random_recommendation(agent)
                    return
                    
                user_idx = self.user_index.get_loc(agent.pos)
                print(f"it worked: {user_idx}")
                # print(f"Agent pos: {agent.pos}")
                # print(f"User index2: {user_idx}")
            except KeyError:
                print(f"User {agent.pos} not in index. Available users: {list(self.user_index)}")
                # print(f"User {agent.pos} not in index: {self.user_index.keys()}")
                print(f"User not in index, using random recommendations")
                self.random_recommendation(agent)
                return
                
            try:
                user_ratings = self.rating_matrix.row(user_idx)
                # print(f"user_ratings: {user_ratings}")
            except Exception as e:
                print(f"Error getting user ratings: {e}")
                self.random_recommendation(agent)
                return
            
            rated_items = np.where(user_ratings > 0)[0]
            print(f"rated_items: {rated_items}")
            if len(rated_items) == 0:
                self.random_recommendation(agent)
                return
                
            # get item ids
            rated_items = [self.item_index[i] for i in rated_items]
            print(f"rated_itemsID: {rated_items}")
            # Get candidate items using matrix operations
            try:
                # all_items = np.arange(self.rating_matrix.ncols)
                all_items = self.item_index.values
                # print(f"all_items: {all_items}")
            except Exception as e:
                print(f"Error getting candidate items: {e}")
                # self.random_recommendation(agent)
                return

            try:
                # get candidate items not rated by user
                candidate_items = np.setdiff1d(all_items, rated_items)
                # print(f"rated_items: {rated_items}")
                print(f"candidate_items: {candidate_items}")
                print(f"item_index: {self.item_index}")
            except Exception as e:
                print(f"Error getting candidate items: {e}")
                # self.random_recommendation(agent)
                return

            if len(candidate_items) == 0:
                print(f"No candidate items found")
                # self.random_recommendation(agent)
                return
            
            # Fit model with the rating matrix
            try:
                self.item_knn.fit(self.ratings_df)
                print(f"Model fitted successfully")
            except Exception as e:
                print(f"Error fitting model: {e}")
                # self.random_recommendation(agent)
                return
            
            try:
                print(f"Number of rated items: {len(rated_items)}")
                print(f"Number of candidate items: {len(candidate_items)}")
                
                if self.item_knn is None:
                    # print("Error: Model has not been trained yet.")
                    return

                predictions = self.item_knn.predict_for_user(user_idx, candidate_items)

                # Remove NaN values and handle empty predictions
                predictions = predictions.dropna()
                print(f"predictions: {predictions}")
                
                if predictions is None or len(predictions) == 0:
                    print(f"No predictions found")
                    # self.random_recommendation(agent)
                    return
            
                # Get top 3 items with highest predicted ratings
                top_items = predictions.nlargest(3)
                print(f"top_items: {top_items}")
                
                # Convert content IDs back to NewsContent objects
                content_dict = {int(c.content): c for c in agent.model.news_content}
                recommendations = [content_dict[int(self.item_index[i])] 
                                for i in top_items.index 
                                if int(self.item_index[i]) in content_dict]
                print(f"recommendations: {recommendations}")
                
                if recommendations:
                    agent.recommended_content.extend(recommendations)
                else:
                    self.random_recommendation(agent)# Get top 3 items with highest predicted ratings

            except Exception as e:
                print(f"Error generating predictions: {e}")
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


