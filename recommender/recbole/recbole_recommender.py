# NB! This is not completed yet. DO NOT attempt to run 'python experiments.py recbole'

# Make sure that a virtual environment with requirements/recbole.txt is installed before running.
# To avoid red lines set the VSCode interpreter path to your specified environment. Example: .envs/recbole/bin/python

import os

import numpy as np
import pandas as pd
import torch

from recommender.base_recommender import BaseRecommender
from recommender.general_recommender import random_recommendation
from recbole.config import Config
from recbole.model.abstract_recommender import AbstractRecommender
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import get_model
from recbole.trainer import Trainer


class RecboleRecommender(BaseRecommender):

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
        self._last_training_count = 0
        self._retrain = False

        self.recbole_model: AbstractRecommender
        self.config = Config(
            model=self.type,
            dataset=f"simulation_{os.getpid()}",
            config_dict={
                "USER_ID_FIELD": "user_id",
                "ITEM_ID_FIELD": "item_id",
                "RATING_FIELD": "rating",
                "load_col": {"inter": ["user_id", "item_id", "rating"]},
                # implicit feedback
                "neg_sampling": {"uniform": 1},
                # keep training fast for simulation
                "epochs": 10,
                "train_batch_size": 2048,
                "eval_batch_size": 2048,
                # avoid GPU complications
                "device": "cpu",
                "checkpoint_dir": f"saved/saved_{os.getpid()}",  # Ensure unique checkpoint for each Trainer
                "save_dataset": False,
                # disable logging to avoid thread exhaustion (DOES NOT WORK, wth)
                "log_tensorboard": False,
                "enable_wandb": False,
            },
        )

        self.content_dict_cache = {}  # Add cache for content dictionaries
        self.last_content_update = -1  # Track when content was last updated
        print(
            f"Recommender type: {self.type}, diversity_level: {self.diversity_level}, num_recommendations: {self.num_recommendations}"
        )

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

    def update_recommendations(self, agents):
        match self.type:
            case "BPR":
                self.train_recommender(agents)
            case "random":
                for agent in agents:
                    random_recommendation(agent, self.num_recommendations)

    # TODO: Not properly implemented, needs rewriting if recbole is to be included in experiments
    def train_recommender(self, agents):
        """Trains a recommender based on what type of model is specified.

        Args:
            agents (_type_): _description_
        """
        interaction_list = self._get_interaction_list()
        n_interactions = len(interaction_list)

        # If there are too few interactions overall, return
        if n_interactions < self._training_threshold:
            return

        if n_interactions - self._last_training_count < self._training_threshold:
            return

        dataset = self._create_dataset()

        if dataset is None:
            for agent in agents:
                random_recommendation(agent, self.num_recommendations)
            return

        else:
            model_class = get_model(self.type)
            self.recbole_model = model_class(self.config, dataset)

            trainer = Trainer(self.config, self.recbole_model)
            train_data, val_data, test_data = data_preparation(self.config, dataset)

            try:
                trainer.fit(train_data)
            except RuntimeError as e:
                # Suppress TensorBoard thread errors during parallel execution
                if "can't start new thread" in str(e):
                    pass
                else:
                    raise

        # TODO: MOVE

        for agent in agents:
            if agent.pos not in dataset.field2id_token[dataset.uid_field]:
                random_recommendation(agent, self.num_recommendations)

            else:
                topk_recs = self.recommend_topk(dataset, agent.pos)
                recs_to_content = [agent.model.news_content[item] for item in topk_recs]

                agent.recommended_content.extend(recs_to_content)

    def recommend_topk(self, dataset, user_id):
        """Recommends the top k items scored by a recommender

        Args:
            dataset: collection of every user-item interaction
            user_id (int): unique identifier of the agent the recommendation is for

        Returns:
            Tensor: the ids of items to be recommended
        """
        uid = dataset.token2id(dataset.uid_field, str(user_id))
        interaction = {dataset.uid_field: torch.tensor([uid])}

        scores = self.recbole_model.full_sort_predict(interaction)

        already_interacted = dataset.history_item_matrix()[0][uid]
        scores[already_interacted] = -float("inf")

        topk_scores, topk_items = torch.topk(scores, self.num_recommendations)

        item_ids = [
            int(dataset.id2token(dataset.iid_field, iid)) for iid in topk_items.tolist()
        ]

        return item_ids

    def _create_dataset(self):
        """Create a recbole Dataset from interactions"""
        if not self.user_interactions:
            return None

        interaction_list = self._get_interaction_list()
        n_interactions = len(interaction_list)

        os.makedirs(
            f"dataset/simulation_{os.getpid()}", exist_ok=True
        )  # Ensure separation of datasets between threads

        df = pd.DataFrame(interaction_list)

        df = df[["user_id", "item_id", "rating"]]

        df["user_id"] = df["user_id"].astype("int32")
        df["item_id"] = df["item_id"].astype("int32")
        df["rating"] = df["rating"].astype("float64")

        df = df.drop_duplicates(["user_id", "item_id"], keep="last")
        df_to_save = df.rename(
            columns={
                "user_id": "user_id:token",
                "item_id": "item_id:token",
                "rating": "rating:float",
            }
        )
        df_to_save.to_csv(
            f"dataset/simulation_{os.getpid()}/simulation_{os.getpid()}.inter",
            sep="\t",
            index=False,
        )

        try:
            config = Config(
                model=self.type,
                dataset=f"simulation_{os.getpid()}",
                config_dict={
                    "USER_ID_FIELD": "user_id",
                    "ITEM_ID_FIELD": "item_id",
                    "RATING_FIELD": "rating",
                    "load_col": {
                        "inter": {
                            "user_id": "user_id:token",
                            "item_id": "item_id:token",
                            "rating": "rating:float",
                        }
                    },
                },
            )

            dataset = create_dataset(config)

            return dataset

        except Exception as e:
            print(f"Error creating dataset: {e}")
            return None

    def _get_interaction_list(self):
        if self._user_interactions_cache_dirty:
            self._user_interactions_cache = list(self.user_interactions.values())
            self._user_interactions_cache_dirty = False

        return self._user_interactions_cache
