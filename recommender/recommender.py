# Standard library imports
import random
import traceback

# Third-party library imports
import pandas as pd
import numpy as np
from lenskit.knn import ItemKNNScorer, ItemKNNConfig, UserKNNScorer, UserKNNConfig
from lenskit.data import ItemList, from_interactions_df
from lenskit.pipeline import topn_pipeline
from lenskit import recommend

# Local imports
from utils.common import cosine_similarity
from recommender.diversity import calculate_diversity, diversity_reranking
from recommender.types import RecommenderType


class Recommender():
    def __init__(self, recommender_type, diversity_lambda, increase_diversity, num_recommendations):
        self.type = recommender_type
        self.diversity_lambda = diversity_lambda
        self.increase_diversity = increase_diversity
        self.num_recommendations = num_recommendations
        self.user_interactions = []  # List to store user-content interactions
        self._last_training_count = 0  # Add this line to track when retraining is needed
        
        # Configure ItemKNN with new API
        ItemKNNconfig = ItemKNNConfig(
            max_nbrs=20,  # Maximum number of neighbors
            min_nbrs=1,   # Minimum number of neighbors
            min_sim=0.1,  # Minimum similarity threshold
            feedback='explicit',  # Using explicit ratings
        )
        UserKNNconfig = UserKNNConfig(
            max_nbrs=20,  # Maximum number of neighbors
            min_nbrs=1,   # Minimum number of neighbors
            min_sim=0.1,  # Minimum similarity threshold
            feedback='explicit',  # Using explicit ratings
        )
        self.user_knn = UserKNNScorer(UserKNNconfig)
        # Create the scorer
        self.item_knn = ItemKNNScorer(ItemKNNconfig)
        
        # Create a recommendation pipeline
        self.pipeline = None
        
    def update_recommendations(self, agents):
        """Update recommendations for all agents"""
        print(f"Recommender type: {self.type}")
        for agent in agents:
            if self.type == RecommenderType.RANDOM.value:
                self.random_recommendation(agent)
            elif self.type == RecommenderType.ITEM_KNN.value:
                self.collaborative_filtering(agent, "item")
            elif self.type == RecommenderType.USER_KNN.value:
                self.collaborative_filtering(agent, "user")
            elif self.type == RecommenderType.CONTENT_BASED.value:
                self.content_based(agent)
            elif self.type == RecommenderType.POPULAR.value:
                self.popular_recommendation(agent)

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
            dataset = from_interactions_df(df)
            return dataset
        except Exception as e:
            print(f"Error creating dataset: {e}")
            return None
        
    def collaborative_filtering(self, agent, type):
        """Recommend content using item-based collaborative filtering"""
        
        # Get content pool from model
        if not hasattr(agent.model, 'news_content') or not agent.model.news_content:
            return

        # Check if this specific user has enough interactions (at least 2)
        user_interactions = [inter for inter in self.user_interactions if inter['user_id'] == agent.pos]
        
        # Check if the system as a whole has enough interactions
        min_interactions = max(150, agent.model.num_agents // 2)  # Minimum total interactions needed
        if len(self.user_interactions) < min_interactions or len(user_interactions) < 3:
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
                if self.pipeline is None:
                    self.pipeline = topn_pipeline(self.item_knn if type == "item" else self.user_knn)
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
            candidates = ItemList(list(candidate_ids))
            
            # Get recommendations using the recommend function
            try:
                num_recommendations = self.num_recommendations * 2 if self.increase_diversity else self.num_recommendations
                
                recs = recommend(self.pipeline, agent.pos, n=num_recommendations, items=candidates)
                # Convert to NewsContent objects
                content_dict = {c.content: c for c in agent.model.news_content}
                recommendations = []
                
                rec_ids = recs.ids()
                # Process the recommendations
                for item_id in rec_ids:
                    if item_id in content_dict:
                        recommendations.append(content_dict[item_id])
                
                if recommendations:
                    # get the first half of the recommendations
                    if len(recommendations) < num_recommendations:
                        self.random_recommendation(agent, num_recommendations - len(recommendations))

                    if self.increase_diversity:
                        rec_vectors = [recommendations[i].topic_vector for i in range(len(recommendations)//2)]
                        # print(f"Diversity score before reranking: {calculate_diversity(rec_vectors)}")
                        recommendations = diversity_reranking(agent.preference_vector, recommendations, self.diversity_lambda, k=(len(recommendations)//2))
                    
                    agent.recommended_content.extend(recommendations)

                    rec_vectors = [agent.recommended_content[i].topic_vector for i in range(len(agent.recommended_content))]
                    #print(f"Diversity score after reranking: {calculate_diversity(rec_vectors)}")
                    # print(f"Added {len(recommendations)} recommendations to agent {agent.pos}")
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
        """Recommend content based on topic vector similarity."""
        
        # Clear previous recommendations
        agent.recommended_content = []
        
        # Get content not already in agent's feed
        available_content = [c for c in agent.model.news_content if c not in agent.feed]
        
        if not available_content:
            print("No new content available for recommendation")
            return
        
        # Calculate similarity scores for each content item
        scores = []
        for content in available_content:
            # Calculate cosine similarity between agent's preference vector and content's topic vector
            similarity = cosine_similarity(agent.preference_vector, content.topic_vector)
            
            scores.append((content, similarity))
        
        # Sort by score and get top recommendations
        scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [content for content, _ in scores[:3]]
        
        # Add recommendations to agent's recommended_content
        agent.recommended_content.extend(recommendations)
        
    def hybrid(self, agent):
        pass
        
    def random_recommendation(self, agent, num_recommendations=10):
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
            num_recommendations = min(10, len(available_content))
            recommendations = random.sample(available_content, num_recommendations)
            agent.recommended_content.extend(recommendations)

    def popular_recommendation(self, agent):
        """Recommend popular news content to an agent with personalization and exploration"""
        # Clear previous recommendations
        agent.recommended_content = []
        
        # Get content pool from model and ensure it exists
        if not hasattr(agent.model, 'news_content') or not agent.model.news_content:
            return
        
        # Create a dataset from interactions if we have enough
        if len(self.user_interactions) < 10:  # Need some minimum interactions
            self.random_recommendation(agent)
            return
        
        try:
            # Get content that isn't in the agent's current feed
            content_pool = agent.model.news_content
            available_content = [c for c in content_pool if c not in agent.feed]
            
            if not available_content:
                return
            
            # Calculate base popularity scores
            content_counts = {}
            for interaction in self.user_interactions:
                item_id = interaction['item_id']
                if item_id in content_counts:
                    content_counts[item_id] += 1
                else:
                    content_counts[item_id] = 1
            
            # Calculate recency-adjusted popularity
            current_step = agent.model.schedule.steps
            recency_adjusted_scores = {}
            
            for content in available_content:
                # Base popularity (number of interactions)
                base_popularity = content_counts.get(content.content, 0)
                
                # Recency factor - newer content gets a boost
                content_age = current_step - content.creation_step
                recency_boost = np.exp(-0.05 * content_age)  # Exponential decay with age
                
                # Engagement factor - more engaging content (like fake news) gets a boost
                engagement_factor = content.engagement
                
                # Combine factors - this balances popularity with recency and engagement
                score = (base_popularity + 1) * recency_boost * engagement_factor
                
                recency_adjusted_scores[content.content] = score
            
            # Add personalization and exploration components
            final_scores = {}
            for content in available_content:
                # 1. Base score from popularity and recency
                base_score = recency_adjusted_scores[content.content]
                
                # 2. Personalization component - similarity to user preferences
                preference_similarity = cosine_similarity(agent.preference_vector, content.topic_vector)
                
                # 3. Exploration component - random factor to discover new content
                exploration_factor = np.random.random() * 0.2  # 20% randomness
                
                # 4. Novelty boost - give extra weight to content with fewer interactions
                interaction_count = content_counts.get(content.content, 0)
                novelty_boost = 1.0 + (0.5 * np.exp(-0.1 * interaction_count))
                
                # Combine all factors - weighted sum
                # Adjust these weights to control the balance between popularity, personalization and exploration
                popularity_weight = 0.5
                personalization_weight = 0.3
                exploration_weight = 0.2
                
                final_score = (
                    (popularity_weight * base_score) + 
                    (personalization_weight * preference_similarity * novelty_boost) + 
                    (exploration_weight * exploration_factor)
                )
                
                final_scores[content.content] = final_score
            
            # Sort content by final score
            scored_content = [(c, final_scores[c.content]) for c in available_content]
            scored_content.sort(key=lambda x: x[1], reverse=True)
            
            # Take top recommendations
            num_to_recommend = min(self.num_recommendations, len(scored_content))
            recommendations = [content for content, _ in scored_content[:num_to_recommend]]
            
            # Apply diversity reranking if enabled
            if self.increase_diversity and len(recommendations) > 1:
                recommendations = diversity_reranking(
                    agent.preference_vector, 
                    recommendations, 
                    self.diversity_lambda
                )
            
            # Add recommendations to agent
            agent.recommended_content.extend(recommendations)
            
        except Exception as e:
            print(f"Error in popular recommendation: {e}")
            traceback.print_exc()
            # Fall back to random recommendations
            self.random_recommendation(agent)
        