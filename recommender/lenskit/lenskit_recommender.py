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
from recommender.diversity import calculate_diversity, diversity_reranking
from recommender.types import RecommenderType
from recommender.general_recommender import random_recommendation
from utils.metrics import vec_mat_cosine_similarity


class LenskitRecommender:
    def __init__(self, recommender_type, diversity_level, num_recommendations):
        self.type = recommender_type
        self.diversity_level = diversity_level
        self.num_recommendations = num_recommendations
        self.user_interactions = {}  # Dict to store user interactions
        self.user_interactions_count = (
            {}
        )  # Dict to store number of interactions with specific user for collaborative filtering

        self._user_interactions_cache = []
        self._user_interactions_cache_dirty = True

        self._last_training_count = (
            0  # Add this line to track when retraining is needed
        )
        self._training_threshold = 200

        self.content_dict_cache = {}  # Add cache for content dictionaries
        self.last_content_update = -1  # Track when content was last updated
        print(
            f"Recommender type: {self.type}, diversity_level: {self.diversity_level}, num_recommendations: {self.num_recommendations}"
        )
        # Configure ItemKNN with new API
        ItemKNNconfig = ItemKNNConfig(
            max_nbrs=20,  # Maximum number of neighbors
            min_nbrs=1,  # Minimum number of neighbors
            min_sim=0.1,  # Minimum similarity threshold
            feedback="explicit",  # Using explicit ratings
        )
        UserKNNconfig = UserKNNConfig(
            max_nbrs=20,  # Maximum number of neighbors
            min_nbrs=1,  # Minimum number of neighbors
            min_sim=0.1,  # Minimum similarity threshold
            feedback="explicit",  # Using explicit ratings
        )
        self.user_knn = UserKNNScorer(UserKNNconfig)
        # Create the scorer
        self.item_knn = ItemKNNScorer(ItemKNNconfig)

        # Create a recommendation pipeline
        self.pipeline = None

    def update_recommendations(self, agents):
        """Update recommendations for all agents"""

        for agent in agents:
            if self.type == RecommenderType.RANDOM.value:
                random_recommendation(agent, self.num_recommendations)
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

        if (agent_id, content_id) not in self.user_interactions:
            self.user_interactions_count[agent_id] = (
                self.user_interactions_count.get(agent_id, 0) + 1
            )

        # Create new interaction, overwrites preexisting interaction
        self.user_interactions[(agent_id, content_id)] = {
            "user_id": agent_id,
            "item_id": content_id,
            "rating": rating,
        }

        self._user_interactions_cache_dirty = True

    def _create_dataset(self):
        """Create a LensKit Dataset from interactions"""
        if not self.user_interactions:
            return None

        interaction_list = self._get_interaction_list()
        n_interactions = len(interaction_list)

        # Convert interactions to DataFrame - only if needed
        if (
            hasattr(self, "_cached_dataset")
            and n_interactions == self._last_dataset_size
        ):
            return self._cached_dataset

        # Convert interactions to DataFrame
        df = pd.DataFrame(interaction_list)

        # Ensure proper data types
        df["user_id"] = df["user_id"].astype("int32")
        df["item_id"] = df["item_id"].astype("int32")
        df["rating"] = df["rating"].astype("float64")

        # Remove duplicates keeping most recent
        df = df.drop_duplicates(["user_id", "item_id"], keep="last")

        # Create dataset using from_interactions_df
        try:
            dataset = from_interactions_df(df)
            # Cache the dataset
            self._cached_dataset = dataset
            self._last_dataset_size = n_interactions
            return dataset
        except Exception as e:
            print(f"Error creating dataset: {e}")
            return None

    def collaborative_filtering(self, agent, type):
        """
        Recommend content using collaborative filtering.

        Parameters:
        -----------
        agent : UserAgent
            The agent to generate recommendations for
        type : str
            Type of collaborative filtering ("item" or "user")
        """
        try:
            # Get content pool from model
            if not hasattr(agent.model, "news_content") or not agent.model.news_content:
                random_recommendation(agent, self.num_recommendations)
                return

            interaction_list = self._get_interaction_list()
            n_interactions = len(interaction_list)

            # Check if this specific user has enough interactions (at least 2)
            user_interactions_count = self.user_interactions_count.get(agent.pos, 0)

            # Check if the system as a whole has enough interactions
            min_interactions = max(
                150, agent.model.num_agents // 2
            )  # Minimum total interactions needed
            if n_interactions < min_interactions or user_interactions_count < 3:
                # Fall back to random recommendations if not enough data overall
                recommendations = random_recommendation(
                    agent, self.num_recommendations, add_to_feed=False
                )
                agent.recommended_content.extend(recommendations)
                agent.diversity_score = calculate_diversity(
                    np.array([rec.topic_vector for rec in recommendations])
                )
                return

            # Create dataset only if needed
            dataset = self._create_dataset()
            if dataset is None:
                random_recommendation(agent, self.num_recommendations)
                return

            # Create and train pipeline if not already created or if it needs retraining
            if self.pipeline is None or n_interactions > self._last_training_count:
                if self.pipeline is None:
                    self.pipeline = topn_pipeline(
                        self.item_knn if type == "item" else self.user_knn
                    )
                self.pipeline.train(dataset)
                self._last_training_count = n_interactions

            # Cache content IDs if model content has changed
            current_step = agent.model.steps
            if (
                current_step != self.last_content_update
                or agent.pos not in self.content_dict_cache
            ):
                # Get all available content IDs - do this once and cache
                all_content_ids = {c.content for c in agent.model.news_content}
                self.content_dict_cache[agent.pos] = {
                    "all_content_ids": all_content_ids,
                    "content_dict": {c.content: c for c in agent.model.news_content},
                }
                self.last_content_update = current_step

            # Use cached values
            all_content_ids = self.content_dict_cache[agent.pos]["all_content_ids"]
            content_dict = self.content_dict_cache[agent.pos]["content_dict"]

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
                num_recommendations = (
                    self.num_recommendations * 3
                    if self.diversity_level > 0
                    else self.num_recommendations
                )

                recs = recommend(
                    self.pipeline, agent.pos, n=num_recommendations, items=candidates
                )
                recommendations = []

                rec_ids = recs.ids()
                # Process the recommendations
                for item_id in rec_ids:
                    if item_id in content_dict:
                        recommendations.append(content_dict[item_id])

                if recommendations:
                    # get the first half of the recommendations
                    if len(recommendations) < num_recommendations:
                        recommendations.extend(
                            random_recommendation(
                                agent,
                                num_recommendations - len(recommendations),
                                add_to_feed=False,
                            )
                        )
                    if self.diversity_level > 0:
                        # Calculate diversity before reranking on the same number of items that will be in final set
                        num_final_recs = min(
                            len(recommendations) // 3, self.num_recommendations
                        )

                        # Before reranking
                        original_topic_vectors = np.array(
                            [
                                rec.topic_vector
                                for rec in recommendations[: self.num_recommendations]
                            ]
                        )
                        original_diversity = calculate_diversity(original_topic_vectors)
                        agent.original_diversity_score = original_diversity

                        # Apply diversity reranking with optimized implementation
                        recommendations, new_diversity = (
                            self._optimized_diversity_reranking(
                                agent.preference_vector,
                                recommendations,
                                k=num_final_recs,
                            )
                        )
                        diversity_score = new_diversity

                    else:
                        topic_vectors = np.array(
                            [rec.topic_vector for rec in recommendations]
                        )
                        diversity_score = calculate_diversity(topic_vectors)

                    agent.recommended_content.extend(recommendations)

                    # After reranking
                    agent.diversity_score = diversity_score
                else:
                    # Fall back to random if no recommendations were generated
                    random_recommendation(agent, self.num_recommendations)
            except Exception as e:
                print(f"Error getting recommendations: {e}")
                traceback.print_exc()  # Add traceback for better debugging
                random_recommendation(agent, self.num_recommendations)

        except Exception as e:
            print(f"Error in collaborative filtering for agent {agent.pos}: {e}")
            # Fallback to random recommendations
            random_recommendation(agent, self.num_recommendations)

    def content_based(self, agent):
        """Recommend content based on topic vector similarity."""

        # Cache content if needed
        current_step = agent.model.steps
        if (
            current_step != self.last_content_update
            or agent.pos not in self.content_dict_cache
        ):
            self.content_dict_cache[agent.pos] = {
                "all_content": agent.model.news_content,
                "content_dict": {c.content: c for c in agent.model.news_content},
            }
            self.last_content_update = current_step

        # Get content not already in agent's feed - use set operations for efficiency
        feed_set = set(agent.feed)
        available_content = [
            c
            for c in self.content_dict_cache[agent.pos]["all_content"]
            if c not in feed_set
        ]

        if not available_content:
            return

        # Calculate similarity scores for each content item
        scores = []
        topic_vectors = []

        # Prepare arrays for vectorized cosine similarity
        user_preference = np.array(agent.preference_vector).reshape(1, -1)
        content_topics = np.array(
            [content.topic_vector for content in available_content]
        )

        # Calculate all similarities at once using cosine_similarity
        similarities = vec_mat_cosine_similarity(user_preference, content_topics)

        # Create scores list with content and similarity pairs
        scores = list(zip(available_content, similarities))
        topic_vectors = content_topics.tolist()
        relevance_scores = similarities.tolist()

        # Sort by score and get top recommendations
        scores.sort(key=lambda x: x[1], reverse=True)
        num_to_select = (
            self.num_recommendations * 3
            if self.diversity_level > 0
            else self.num_recommendations
        )
        recommendations = [content for content, _ in scores[:num_to_select]]

        # Apply diversity reranking with pre-calculated data
        if self.diversity_level > 0 and len(recommendations) > 1:

            # Before reranking
            original_topic_vectors = np.array(
                [
                    rec.topic_vector
                    for rec in recommendations[: self.num_recommendations]
                ]
            )
            original_diversity = calculate_diversity(original_topic_vectors)
            agent.original_diversity_score = original_diversity

            pre_calculated = {
                "topic_vectors": np.array(topic_vectors),
                "relevance_scores": np.array(relevance_scores),
            }
            recommendations, new_diversity = self._optimized_diversity_reranking(
                agent.preference_vector,
                recommendations,
                k=self.num_recommendations,
                pre_calculated=pre_calculated,
            )
            diversity_score = new_diversity
        else:
            topic_vectors = np.array([rec.topic_vector for rec in recommendations])
            diversity_score = calculate_diversity(topic_vectors)
        # Add recommendations to agent's recommended_content
        agent.recommended_content.extend(recommendations)
        agent.diversity_score = diversity_score

    def popular_recommendation(self, agent):
        """Recommend popular news content to an agent with personalization and exploration"""

        # Get content pool from model and ensure it exists
        if not hasattr(agent.model, "news_content") or not agent.model.news_content:
            return

        interaction_list = self._get_interaction_list()
        n_interactions = len(interaction_list)

        # Create a dataset from interactions if we have enough
        if n_interactions < 10:  # Need some minimum interactions
            recommendations = random_recommendation(
                agent, self.num_recommendations, add_to_feed=False
            )
            agent.recommended_content.extend(recommendations)
            agent.diversity_score = calculate_diversity(
                np.array([rec.topic_vector for rec in recommendations])
            )
            return

        try:
            # Cache content if needed
            current_step = agent.model.steps
            if (
                current_step != self.last_content_update
                or "popularity_scores" not in self.content_dict_cache
            ):
                # Calculate content counts once per step
                content_counts = {}
                for interaction in interaction_list:
                    item_id = interaction["item_id"]
                    content_counts[item_id] = content_counts.get(item_id, 0) + 1

                # Store in cache
                self.content_dict_cache["popularity_scores"] = {
                    "content_counts": content_counts,
                    "step": current_step,
                }
                self.last_content_update = current_step

            # Get cached values
            content_counts = self.content_dict_cache["popularity_scores"][
                "content_counts"
            ]

            # Get content that isn't in the agent's current feed
            feed_set = set(agent.feed)
            available_content = [
                c for c in agent.model.news_content if c not in feed_set
            ]

            if not available_content:
                return

            # Prepare arrays for vectorized operations
            content_ids = [c.content for c in available_content]
            topic_vectors = np.array([c.topic_vector for c in available_content])
            creation_steps = np.array([c.creation_step for c in available_content])
            engagement_factors = np.array([c.engagement for c in available_content])

            # Calculate base popularity for all content at once
            base_popularity = np.array(
                [content_counts.get(cid, 0) for cid in content_ids]
            )

            # Calculate recency factors
            content_age = current_step - creation_steps
            recency_boost = np.exp(-0.05 * content_age)

            # Calculate base scores
            base_scores = (base_popularity + 1) * recency_boost * engagement_factors

            # Calculate personalization component using cosine_similarity
            user_preference = np.array(agent.preference_vector).reshape(1, -1)
            preference_similarities = vec_mat_cosine_similarity(
                user_preference, topic_vectors
            )

            # Calculate novelty boost
            interaction_counts = np.array(
                [content_counts.get(cid, 0) for cid in content_ids]
            )
            novelty_boost = 1.0 + (0.5 * np.exp(-0.1 * interaction_counts))

            # Generate exploration factors
            exploration_factors = np.random.random(len(available_content)) * 0.2

            # Combine all factors with weights
            popularity_weight = 0.5
            personalization_weight = 0.3
            exploration_weight = 0.2

            final_scores = (
                (popularity_weight * base_scores)
                + (personalization_weight * preference_similarities * novelty_boost)
                + (exploration_weight * exploration_factors)
            )

            # Get indices of top scores
            num_to_recommend = min(
                (
                    self.num_recommendations * 3
                    if self.diversity_level > 0
                    else self.num_recommendations
                ),
                len(available_content),
            )
            top_indices = np.argsort(-final_scores)[:num_to_recommend]
            recommendations = [available_content[i] for i in top_indices]

            # Apply diversity reranking if enabled
            if self.diversity_level > 0 and len(recommendations) > 1:
                # Before reranking
                original_topic_vectors = np.array(
                    [
                        rec.topic_vector
                        for rec in recommendations[: self.num_recommendations]
                    ]
                )
                original_diversity = calculate_diversity(original_topic_vectors)
                agent.original_diversity_score = original_diversity

                # Pass pre-calculated data to diversity reranking
                pre_calculated = {
                    "topic_vectors": topic_vectors[top_indices],
                    "relevance_scores": preference_similarities[top_indices],
                }

                recommendations, new_diversity = self._optimized_diversity_reranking(
                    agent.preference_vector,
                    recommendations,
                    k=self.num_recommendations,
                    pre_calculated=pre_calculated,
                )
                diversity_score = new_diversity
            else:
                topic_vectors = np.array([rec.topic_vector for rec in recommendations])
                diversity_score = calculate_diversity(topic_vectors)
            # Add recommendations to agent
            agent.recommended_content.extend(recommendations)
            agent.diversity_score = diversity_score

        except Exception as e:
            print(f"Error in popular recommendation: {e}")
            traceback.print_exc()
            # Fall back to random recommendations
            random_recommendation(agent, self.num_recommendations)

    def _optimized_diversity_reranking(
        self, preference_vector, recommendations, k=10, pre_calculated=None
    ):
        """Optimized version of diversity reranking that avoids redundant calculations"""
        if pre_calculated is None:
            # Pre-calculate all topic vectors and relevance scores at once
            topic_vectors = np.array([rec.topic_vector for rec in recommendations])

            # Calculate relevance scores using cosine_similarity
            user_preference = np.array(preference_vector).reshape(1, -1)
            relevance_scores = vec_mat_cosine_similarity(user_preference, topic_vectors)

            pre_calculated = {
                "topic_vectors": topic_vectors,
                "relevance_scores": relevance_scores,
            }

        # Use the enhanced diversity reranking function with pre-calculated data
        from recommender.diversity import diversity_reranking

        # Apply reranking
        reranked, new_diversity = diversity_reranking(
            preference_vector,
            recommendations,
            k=k,
            pre_calculated=pre_calculated,
            diversity_level=self.diversity_level,
        )

        return reranked, new_diversity

    def _get_interaction_list(self):
        if self._user_interactions_cache_dirty:
            self._user_interactions_cache = list(self.user_interactions.values())
            self._user_interactions_cache_dirty = False

        return self._user_interactions_cache
