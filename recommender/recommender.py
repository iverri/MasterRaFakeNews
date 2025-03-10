import random
import pandas as pd
import numpy as np
from lenskit.algorithms import item_knn as knn


class Recommender():
    def __init__(self, type):
        self.type = type
        self.user_interactions = []  # List to store user-content interactions
        # Configure ItemItem with better parameters for our use case
        self.item_knn = knn.ItemItem(
            20,         # n_neighbors
            min_nbrs=1,
            min_sim=-1, # Allow negative similarities
            center=False, # Don't center the ratings
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

    def add_interaction(self, agent_id, content_id, rating, current_step=None):
        """Add an interaction between an agent and content item
        
        Args:
            agent_id: The ID of the agent
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
            'timestamp': current_step if current_step is not None else len(self.user_interactions)
        }
        
        # Remove any existing interaction for this user-item pair
        self.user_interactions = [
            inter for inter in self.user_interactions 
            if not (inter['user'] == agent_id and inter['item'] == content_id)
        ]
        
        # Add the new interaction
        self.user_interactions.append(new_interaction)

    def recommend(self, agent, content_pool):
        """Recommend news content to an agent"""
        pass

    # Recommender algorithms
    def collaborative_filtering(self, agent):
        """Recommend content using item-based collaborative filtering"""
        # Clear previous recommendations
        agent.recommended_content = []
        
        # Get content pool from model
        if not hasattr(agent.model, 'news_content') or not agent.model.news_content:
            # print(f"No content pool available for agent {agent.pos}")
            return
            
        # Convert interactions to DataFrame
        if len(self.user_interactions) < 5:  # Need some minimum interactions
            # print(f"Not enough interactions ({len(self.user_interactions)}) for CF, using random")
            self.random_recommendation(agent)
            return
        
        # Convert to DataFrame and ensure proper data types
        ratings_df = pd.DataFrame(self.user_interactions)
        
        # Remove duplicates keeping most recent
        ratings_df = ratings_df.sort_values('timestamp').drop_duplicates(['user', 'item'], keep='last')

        
        try:
            # Check rating distribution
            rating_mean = ratings_df['rating'].mean()
            rating_std = ratings_df['rating'].std()
            # print(f"Rating stats for agent {agent.pos}: mean={rating_mean:.3f}, std={rating_std:.3f}")
            
            if rating_std < 0.01:  # If ratings are too similar
                # print(f"Ratings not varied enough, using random")
                self.random_recommendation(agent)
                return
            
            # Get items the user hasn't interacted with
            user_items = set(ratings_df[ratings_df['user'] == agent.pos]['item'])
            all_items = set(ratings_df['item'].unique())
            candidate_items = list(all_items - user_items)
            
            if not candidate_items:
                # print(f"No new items for agent {agent.pos}, using random")
                self.random_recommendation(agent)
                return
            
            # Fit the model with current interactions
            try:
                self.item_knn.fit(ratings_df)
            except ValueError as e:
                # print(f"Error fitting model: {e}, using random recommendations")
                self.random_recommendation(agent)
                return
                
            # Get recommendations
            try:
                recs = self.item_knn.predict_for_user(int(agent.pos), candidate_items)
                if recs is None or len(recs) == 0:
                    # print(f"No recommendations generated for agent {agent.pos}, using random")
                    self.random_recommendation(agent)
                    return
                    
                # Sort by predicted rating and get top 3
                top_items = recs.sort_values(ascending=False).head(3)
                
                # Convert content IDs back to NewsContent objects
                content_dict = {int(c.content): c for c in agent.model.news_content}
                recommendations = [content_dict[int(item_id)] for item_id in top_items.index 
                                 if int(item_id) in content_dict]
                
                if recommendations:
                    # print(f"CF recommending {len(recommendations)} items to agent {agent.pos}")
                    agent.recommended_content.extend(recommendations)
                else:
                    # print(f"No valid recommendations for agent {agent.pos}, using random")
                    self.random_recommendation(agent)
                    
            except (ValueError, TypeError) as e:
                # print(f"Error generating predictions: {e}, using random recommendations")
                self.random_recommendation(agent)
                
        except Exception as e:
            # print(f"Collaborative filtering failed for agent {agent.pos}: {e}")
            self.random_recommendation(agent)

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


