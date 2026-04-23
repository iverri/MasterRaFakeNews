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
from lenskit.als import BiasedMFScorer

# Local imports
from recommender.diversity import calculate_diversity, diversity_reranking
from recommender.types import RecommenderType
from utils.metrics import vec_mat_cosine_similarity


class Recommender:
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

        # Biased Matrix Factorization model

        # Paramteres for BiasedMFScorer was decided through parameter tuning
        self.mf_model = BiasedMFScorer(
            embedding_size=32,  # Number of latent factors
            epochs=40,  # Number of iterations for training
            regularization=0.1,  # Regularization parameter
            damping=15,  # Damping factor for bias terms
        )

        self.mf_pipeline = None  # Pipeline for matrix factorization model
        self._last_mf_training_count = 0  # Track interactions for MF retraining

    def update_recommendations(self, agents):
        """Update recommendations for all agents"""

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
            elif self.type == RecommenderType.HYBRID_WEIGHTED_DYNAMIC.value:
                self.hybrid_weighted(agent, "dynamic")
            elif self.type == RecommenderType.HYBRID_WEIGHTED_STATIC.value:
                self.hybrid_weighted(agent, "static")
            elif self.type == RecommenderType.MATRIX_FACTORIZATION.value:
                self.matrix_factorization(agent)
            elif self.type == RecommenderType.MIXED.value:
                self.mixed_hybrid_recommender(agent)
            elif self.type == RecommenderType.FEATURE_COMBINATION.value:
                self.hybrid_feature_combination(agent)

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

    # =============================
    def add_implicit_interaction(self, agent_id, content_id, rating=0.3):
        """Add an implicit interaction (view/impression) between an agent and content item.
        Used for recommendations that agents see but don't explicitly like."""
        # Only track if not already explicitly interacted
        if (agent_id, content_id) not in self.user_interactions:
            if not isinstance(agent_id, (int, np.integer)):
                agent_id = int(agent_id)
            if not isinstance(content_id, (int, np.integer)):
                content_id = int(content_id)

            # Use lower rating for implicit interactions
            rating = max(0.0, min(1.0, float(rating)))

            self.user_interactions_count[agent_id] = (
                self.user_interactions_count.get(agent_id, 0) + 1
            )

            self.user_interactions[(agent_id, content_id)] = {
                "user_id": agent_id,
                "item_id": content_id,
                "rating": rating,
            }

            self._user_interactions_cache_dirty = True
        # =============================

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

    def collaborative_filtering(
        self, agent, type, num_recommendations=None, add_to_feed=True
    ):
        """
        Recommend content using collaborative filtering.

        Parameters:
        -----------
        agent : UserAgent
            The agent to generate recommendations for
        type : str
            Type of collaborative filtering ("item" or "user")
        """

        if num_recommendations is None:
            num_recommendations = self.num_recommendations

        try:
            # Get content pool from model
            if not hasattr(agent.model, "news_content") or not agent.model.news_content:
                self.random_recommendation(agent)
                return

            interaction_list = self._get_interaction_list()
            n_interactions = len(interaction_list)

            # Check if this specific user has enough interactions (at least 2)
            user_interactions_count = self.user_interactions_count.get(agent.pos, 0)

            # Check if the system as a whole has enough interactions
            min_interactions = max(
                20, agent.model.num_agents // 2
            )  # Minimum total interactions needed
            if n_interactions < min_interactions or user_interactions_count < 1:
                # Fall back to random recommendations if not enough data overall
                recommendations = self.random_recommendation(
                    agent, num_recommendations=num_recommendations, add_to_feed=False
                )
                agent.recommended_content.extend(recommendations)
                agent.diversity_score = (
                    calculate_diversity(
                        np.array([rec.topic_vector for rec in recommendations])
                    )
                    or []
                )
                if add_to_feed:
                    agent.recommended_content.extend(recommendations)
                    if recommendations:
                        agent.diversity_score = calculate_diversity(
                            np.array([rec.topic_vector for rec in recommendations])
                        )
                else:
                    return recommendations
                return

            # Create dataset only if needed
            dataset = self._create_dataset()
            if dataset is None:
                self.random_recommendation(agent)
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
                    "all_content": agent.model.news_content,
                    "all_content_ids": all_content_ids,
                    "content_dict": {c.content: c for c in agent.model.news_content},
                }
                self.last_content_update = current_step

            # Use cached values
            all_content_ids = self.content_dict_cache[agent.pos]["all_content_ids"]
            content_dict = self.content_dict_cache[agent.pos]["content_dict"]

            # Get items already in user's feed (use cached method)
            feed_set = agent.get_seen_content_ids()

            # Convert to list for recommendation
            candidates = ItemList(list(all_content_ids - feed_set))

            # Get recommendations using the recommend function
            try:
                n = (
                    self.num_recommendations * 3
                    if self.diversity_level > 0
                    else self.num_recommendations
                )

                recs = recommend(self.pipeline, agent.pos, n=n, items=candidates)
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
                            self.random_recommendation(
                                agent,
                                num_recommendations - len(recommendations),
                                add_to_feed=False,
                            )
                        )
                    recommendations, diversity_score = (
                        self._calculate_and_apply_diversity(
                            agent,
                            recommendations,
                            k=num_recommendations,
                            add_to_feed=add_to_feed,
                        )
                    )

                    if add_to_feed:
                        agent.recommended_content.extend(recommendations)
                        # After reranking
                        agent.diversity_score = diversity_score
                    else:
                        return recommendations[:num_recommendations]
                else:
                    # Fall back to random if no recommendations were generated

                    if add_to_feed:
                        self.random_recommendation(agent)
                    else:
                        return self.random_recommendation(
                            agent, num_recommendations, add_to_feed=False
                        )
            except Exception as e:
                print(f"Error getting recommendations: {e}")
                traceback.print_exc()  # Add traceback for better debugging
                self.random_recommendation(agent)

        except Exception as e:
            print(f"Error in collaborative filtering for agent {agent.pos}: {e}")
            # Fallback to random recommendations
            self.random_recommendation(agent)

    def content_based(self, agent, num_recommendations=None, add_to_feed=True):
        """Recommend content based on topic vector similarity."""

        if num_recommendations is None:
            num_recommendations = self.num_recommendations

        # Cache content if needed
        current_step = agent.model.steps
        if (
            current_step != self.last_content_update
            or agent.pos not in self.content_dict_cache
        ):
            all_content_ids = {c.content for c in agent.model.news_content}
            self.content_dict_cache[agent.pos] = {
                "all_content": agent.model.news_content,
                "all_content_ids": all_content_ids,
                "content_dict": {c.content: c for c in agent.model.news_content},
            }
            self.last_content_update = current_step

        # Get content not already in agent's feed - use cached method
        feed_set = agent.get_seen_content_objects()

        # Get candidate items (not in feed)
        available_content = [c for c in agent.model.news_content if c not in feed_set]
        if not available_content:
            return

        if num_recommendations is None:
            num_recommendations = self.num_recommendations

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

        pre_calculated = {
            "topic_vectors": np.array(topic_vectors),
            "relevance_scores": np.array(relevance_scores),
        }

        # Sort by score and get top recommendations
        scores.sort(key=lambda x: x[1], reverse=True)
        num_to_select = (
            num_recommendations * 3
            if add_to_feed and self.diversity_level > 0
            else self.num_recommendations
        )
        recommendations = [content for content, _ in scores[:num_to_select]]

        # Apply diversity reranking with pre-calculated data
        recommendations, diversity_score = self._calculate_and_apply_diversity(
            agent,
            recommendations,
            k=num_recommendations,
            add_to_feed=add_to_feed,
            pre_calculated=pre_calculated,
        )

        if add_to_feed:
            # Add recommendations to agent's recommended_content
            agent.recommended_content.extend(recommendations)
            agent.diversity_score = diversity_score
        else:
            return recommendations[:num_recommendations]

    def random_recommendation(self, agent, num_recommendations=None, add_to_feed=True):
        """Recommend random news content to an agent"""

        # Use provided num_recommendations or default to self.num_recommendations
        if num_recommendations is None:
            num_recommendations = self.num_recommendations

        # Get content pool from model and ensure it exists
        if not hasattr(agent.model, "news_content"):
            return

        if not agent.model.news_content:
            return

        # Cache feed set for faster lookups - use cached method
        feed_set = agent.get_seen_content_objects()

        # Get content that isn't in the agent's current feed - use set difference for efficiency
        available_content = [c for c in agent.model.news_content if c not in feed_set]

        if available_content:

            recommendations = random.sample(available_content, num_recommendations)

            # Apply diversity reranking if enabled (commented out in original)
            if add_to_feed:
                agent.recommended_content.extend(recommendations)
            else:
                return recommendations

            # After reranking
            topic_vectors = np.array([rec.topic_vector for rec in recommendations])
            diversity_score = calculate_diversity(topic_vectors)
            agent.diversity_score = diversity_score

    def popular_recommendation(self, agent, num_recommendations=None, add_to_feed=True):
        """Recommend popular news content to an agent with personalization and exploration"""

        if num_recommendations is None:
            num_recommendations = self.num_recommendations

        # Get content pool from model and ensure it exists
        if not hasattr(agent.model, "news_content") or not agent.model.news_content:
            return

        interaction_list = self._get_interaction_list()
        n_interactions = len(interaction_list)

        # Create a dataset from interactions if we have enough
        if n_interactions < 10:  # Need some minimum interactions
            recommendations = self.random_recommendation(agent, add_to_feed=False)
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

            # Get content that isn't in the agent's current feed - use cached method
            feed_set = agent.get_seen_content_objects()
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
                    num_recommendations * 3
                    if self.diversity_level > 0
                    else num_recommendations
                ),
                len(available_content),
            )
            top_indices = np.argsort(-final_scores)[:num_to_recommend]
            recommendations = [available_content[i] for i in top_indices]

            pre_calculated = {
                "topic_vectors": topic_vectors[top_indices],
                "relevance_scores": preference_similarities[top_indices],
            }

            # Apply diversity reranking if enabled

            recommendations, diversity_score = self._calculate_and_apply_diversity(
                agent,
                recommendations,
                k=num_recommendations,
                add_to_feed=add_to_feed,
                pre_calculated=pre_calculated,
            )

            if add_to_feed:
                # Add recommendations to agent
                agent.recommended_content.extend(recommendations)
            else:
                return recommendations

            agent.diversity_score = diversity_score

        except Exception as e:
            print(f"Error in popular recommendation: {e}")
            traceback.print_exc()
            # Fall back to random recommendations
            self.random_recommendation(agent)

    def mixed_hybrid_recommender(self, agent):
        n_recs_dict = {
            "popular": int(self.num_recommendations * 0.1),
            "item_knn": int(self.num_recommendations * 0.4),
            "user_knn": int(self.num_recommendations * 0.2),
            "content_based": int(self.num_recommendations * 0.3),
        }

        recommendations = []

        for key, value in n_recs_dict.items():
            if value > 0:
                try:
                    match key:
                        case RecommenderType.ITEM_KNN.value:
                            recommendations.extend(
                                self.collaborative_filtering(
                                    agent,
                                    "item",
                                    num_recommendations=value,
                                    add_to_feed=False,
                                )
                            )
                        case RecommenderType.USER_KNN.value:
                            recommendations.extend(
                                self.collaborative_filtering(
                                    agent,
                                    "user",
                                    num_recommendations=value,
                                    add_to_feed=False,
                                )
                            )
                        case RecommenderType.CONTENT_BASED.value:
                            recommendations.extend(
                                self.content_based(
                                    agent, num_recommendations=value, add_to_feed=False
                                )
                            )
                        case RecommenderType.POPULAR.value:
                            recommendations.extend(
                                self.popular_recommendation(
                                    agent, num_recommendations=value, add_to_feed=False
                                )
                            )
                except Exception:
                    recommendations.extend(
                        self.random_recommendation(
                            agent, num_recommendations=value, add_to_feed=False
                        )
                    )

        diff = self.num_recommendations - len(recommendations)

        if diff > 0:
            recommendations.extend(
                self.random_recommendation(
                    agent, num_recommendations=diff, add_to_feed=False
                )
            )
        if diff < 0:
            remove_indices = set(
                [random.randint(0, self.num_recommendations) for _ in range(diff)]
            )
            recommendations = [
                value
                for index, value in enumerate(recommendations)
                if index not in remove_indices
            ]

            # Calculate diversity before reranking on the same number of items that will be in final set
        num_final_recs = min(len(recommendations) // 3, self.num_recommendations)

        # Before reranking
        recommendations, diversity_score = self._calculate_and_apply_diversity(
            agent, recommendations, k=self.num_recommendations, add_to_feed=True
        )

        agent.recommended_content.extend(recommendations)
        agent.diversity_score = diversity_score

    def _calculate_and_apply_diversity(
        self, agent, recommendations, k=None, add_to_feed=True, pre_calculated=None
    ):

        if k is None:
            k = self.num_recommendations

        if add_to_feed and self.diversity_level > 0 and len(recommendations) > 1:
            original_topic_vectors = np.array(
                [rec.topic_vector for rec in recommendations[:k]]
            )
            original_diversity = calculate_diversity(original_topic_vectors)
            agent.original_diversity_score = original_diversity

            reranked, new_diversity = self._optimized_diversity_reranking(
                agent.preference_vector,
                recommendations,
                k=k,
                pre_calculated=pre_calculated,
            )

            return reranked, new_diversity

        else:
            if recommendations:
                topic_vectors = np.array([rec.topic_vector for rec in recommendations])
                diversity_score = calculate_diversity(topic_vectors)
            else:
                diversity_score = 0.0

            return recommendations, diversity_score

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

    def hybrid_weighted(self, agent, alpha):
        """Combine collaborative filtering and content-based recommendations with weighted scoring

        Parameters:
        -----------
        agent : UserAgent
            The agent to generate recommendations for
        alpha : str
            Type of alpha weighting ("dynamic" or "static")
        """

        try:
            num_candidates = (
                self.num_recommendations * 3
                if self.diversity_level > 0
                else self.num_recommendations
            )
            # get recommendations from both methods
            cb_recommendations = (
                self.content_based(
                    agent, num_recommendations=num_candidates, add_to_feed=False
                )
                or []
            )

            cf_recommendations = (
                self.collaborative_filtering(
                    agent, "item", num_recommendations=num_candidates, add_to_feed=False
                )
                or []
            )

            if not cb_recommendations and not cf_recommendations:
                self.random_recommendation(agent)
                return

            if alpha == "dynamic":
                alpha = self._get_hybrid_weight(agent)
            else:
                alpha = 0.5  # Static weight

            combined_scores = {}
            content_lookup = {}

            for rank, content in enumerate(cb_recommendations):
                content_id = content.content
                content_lookup[content_id] = content
                # Higher rank gets higher score
                # rank 1 gets score of 1, rank 2 gets 0.5, rank 3 gets 0.33...
                rank_score = 1 / (rank + 1)
                combined_scores[content_id] = (
                    combined_scores.get(content_id, 0) + alpha * rank_score
                )

            for rank, content in enumerate(cf_recommendations):
                content_id = content.content
                content_lookup[content_id] = content
                rank_score = 1 / (rank + 1)
                combined_scores[content_id] = (
                    combined_scores.get(content_id, 0) + (1 - alpha) * rank_score
                )

            ranked_content_ids = sorted(
                combined_scores, key=combined_scores.get, reverse=True
            )
            recommendations = [
                content_lookup[item_id]
                for item_id in ranked_content_ids[:num_candidates]
            ]

            recommendations, diversity_score = self._calculate_and_apply_diversity(
                agent, recommendations
            )

            agent.recommended_content.extend(recommendations)
            agent.diversity_score = diversity_score

        except Exception as e:
            print(f"Error in hybrid recommendation: {e}")
            traceback.print_exc()
            self.random_recommendation(agent)

    def _get_hybrid_weight(self, agent, min_alpha=0.2, max_alpha=0.8, c=10):
        n_user = self.user_interactions_count.get(agent.pos, 0)
        alpha = c / (c + n_user)
        return max(min_alpha, min(max_alpha, alpha))

    def matrix_factorization(self, agent):
        """Collaborative filtering using matrix factorization with ALS"""
        try:
            if not hasattr(agent.model, "news_content") or not agent.model.news_content:
                self.random_recommendation(agent)

            diversity_score = 0.0

            interaction_list = self._get_interaction_list()
            n_interactions = len(interaction_list)

            # Check if this specific user has enough interactions (at least 2)
            user_interactions_count = self.user_interactions_count.get(agent.pos, 0)

            # Check if the system as a whole has enough interactions
            min_interactions = max(20, agent.model.num_agents // 2)
            # same fallback logic as KNN
            if n_interactions < min_interactions or user_interactions_count < 1:
                recommendations = self.random_recommendation(agent, add_to_feed=False)
                agent.recommended_content.extend(recommendations)
                agent.diversity_score = calculate_diversity(
                    np.array([rec.topic_vector for rec in recommendations])
                )
                return

            dataset = self._create_dataset()
            if dataset is None:
                self.random_recommendation(agent)
                return

            # Train MF model if needed
            if (
                self.mf_pipeline is None
                or n_interactions > self._last_mf_training_count
            ):
                if self.mf_pipeline is None:
                    self.mf_pipeline = topn_pipeline(self.mf_model)
                self.mf_pipeline.train(dataset)
                self._last_mf_training_count = n_interactions

            # Cache content
            current_step = agent.model.steps
            if (
                current_step != self.last_content_update
                or agent.pos not in self.content_dict_cache
            ):
                all_content_ids = {c.content for c in agent.model.news_content}
                self.content_dict_cache[agent.pos] = {
                    "all_content": agent.model.news_content,
                    "all_content_ids": {c.content for c in agent.model.news_content},
                    "content_dict": {c.content: c for c in agent.model.news_content},
                }
                self.last_content_update = current_step

            # Use cached values
            all_content_ids = self.content_dict_cache[agent.pos]["all_content_ids"]
            content_dict = self.content_dict_cache[agent.pos]["content_dict"]

            # Remove already seen items - use cached method
            feed_ids = agent.get_seen_content_ids()
            candidate_ids = all_content_ids - feed_ids
            if not candidate_ids:
                return

            candidates = ItemList(list(candidate_ids))

            # Get recommendation
            num_to_select = (
                self.num_recommendations * 3
                if self.diversity_level > 0
                else self.num_recommendations
            )
            recs = recommend(
                self.mf_pipeline, agent.pos, n=num_to_select, items=candidates
            )

            recommendations = []

            rec_ids = recs.ids()
            for item_id in rec_ids:
                if item_id in content_dict:
                    recommendations.append(content_dict[item_id])

            # Fill with random if needed
            if len(recommendations) < num_to_select:
                recommendations.extend(
                    self.random_recommendation(
                        agent,
                        num_recommendations=num_to_select - len(recommendations),
                        add_to_feed=False,
                    )
                )

            # Apply diversity reranking
            recommendations, diversity_score = self._calculate_and_apply_diversity(
                agent, recommendations
            )

            # Add to agent
            agent.recommended_content.extend(recommendations)
            agent.diversity_score = diversity_score

        except Exception as e:
            print(f"Error in matrix factorization recommendation: {e}")
            traceback.print_exc()
            self.random_recommendation(agent)

    def hybrid_feature_combination(
        self, agent, num_recommendations=None, add_to_feed=True
    ):
        if num_recommendations is None:
            num_recommendations = self.num_recommendations

        interaction_list = self._get_interaction_list()
        n_interactions = len(interaction_list)
        min_interactions = max(150, agent.model.num_agents // 2)

        if n_interactions < min_interactions:
            self.random_recommendation(agent)
            return

        dataset = self._create_dataset()
        if dataset is None:
            self.random_recommendation(agent)
            return

        # Train MF if needed
        if self.mf_pipeline is None or n_interactions > self._last_mf_training_count:
            if self.mf_pipeline is None:
                self.mf_pipeline = topn_pipeline(self.mf_model)
            self.mf_pipeline.train(dataset)
            self._last_mf_training_count = n_interactions

        current_step = agent.model.steps
        if (
            current_step != self.last_content_update
            or agent.pos not in self.content_dict_cache
        ):
            self.content_dict_cache[agent.pos] = {
                "all_content": agent.model.news_content,
                "all_content_ids": {c.content for c in agent.model.news_content},
                "content_dict": {c.content: c for c in agent.model.news_content},
            }
            self.last_content_update = current_step

        feed_set = agent.get_seen_content_objects()
        candidates = [
            c
            for c in self.content_dict_cache[agent.pos]["all_content"]
            if c not in feed_set
        ]

        if not candidates:
            return

        # Check if user embedding exists in trained model
        try:
            user_embedding = self.mf_model.user_embeddings[agent.pos]
        except (IndexError, KeyError):
            # User not in trained model, fall back to content-based
            self.content_based(agent, num_recommendations, add_to_feed)
            return
        user_preference = np.array(agent.preference_vector).reshape(1, -1)
        combined_user = np.concatenate([user_preference.flatten(), user_embedding])

        combined_item_features = []
        valid_candidates = []

        for content in candidates:
            topic_vector = np.array(content.topic_vector)

            # Check if item embedding exists (item may be new, created after training)
            try:
                item_embedding = self.mf_model.item_embeddings[content.content]
            except (IndexError, KeyError):
                # Item doesn't have an embedding - skip it or use zero vector
                # Skip it for now to maintain feature consistency
                continue

            combined_vector = np.concatenate([topic_vector, item_embedding])
            combined_item_features.append(combined_vector)
            valid_candidates.append(content)

        # If no items have embeddings, fall back to content-based
        if not combined_item_features:
            self.content_based(agent, num_recommendations, add_to_feed)
            return

        combined_item_features = np.array(combined_item_features)

        similarities = vec_mat_cosine_similarity(
            combined_user.reshape(1, -1), combined_item_features
        )

        scores = list(zip(candidates, similarities))

        # Sort by score and get top recommendations
        scores.sort(key=lambda x: x[1], reverse=True)
        num_to_select = (
            num_recommendations * 3
            if add_to_feed and self.diversity_level > 0
            else self.num_recommendations
        )
        recommendations = [content for content, _ in scores[:num_to_select]]

        # Apply diversity reranking with pre-calculated data
        recommendations, diversity_score = self._calculate_and_apply_diversity(
            agent,
            recommendations,
            k=num_recommendations,
            add_to_feed=add_to_feed,
        )

        if add_to_feed:
            # Add recommendations to agent's recommended_content
            agent.recommended_content.extend(recommendations)
            agent.diversity_score = diversity_score
        else:
            return recommendations[:num_recommendations]
