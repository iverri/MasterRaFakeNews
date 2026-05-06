from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from model import FakeNewsModel
from recommender.recommender import Recommender
from recommender.types import RecommenderType


@dataclass
class TuneConfig:
    k: int = 10
    diversity_level: float = 0.0
    min_train_interactions_per_user: int = 1
    output_csv: str = "biased_mf_tuning_results.csv"
    random_seed: int = 42


DEFAULT_PARAM_GRID = {
    "embedding_size": [4, 8, 16],
    "epochs": [40, 60, 80],
    "regularization": [ 0.001, 0.01, 0.05],
    "damping": [0.5, 1.0, 2.5],
}


def recall_at_k(recommended_ids: List[int], relevant_ids: Iterable[int], k: int = 10) -> float:
    relevant_ids = set(relevant_ids)
    if not relevant_ids:
        return 0.0

    hits = len(set(recommended_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)

def precision_at_k(recommended_ids: List[int], relevant_ids: Iterable[int], k: int = 10) -> float:
    recommended_ids = recommended_ids[:k]
    if not recommended_ids:
        return 0.0

    relevant_ids = set(relevant_ids)
    hits = len(set(recommended_ids) & relevant_ids)

    return hits / len(recommended_ids)

def random_split(
    interactions_df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = interactions_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def interactions_from_model(model: FakeNewsModel) -> pd.DataFrame:
    interaction_list = list(model.social_media_platform.recommender.user_interactions.values())

    if not interaction_list:
        return pd.DataFrame(columns=["user_id", "item_id", "rating"])

    df = pd.DataFrame(interaction_list)
    df["user_id"] = df["user_id"].astype("int32")
    df["item_id"] = df["item_id"].astype("int32")
    df["rating"] = df["rating"].astype("float64")
    return df


def load_interactions_into_recommender(rec: Recommender, interactions_df: pd.DataFrame) -> None:
    required = {"user_id", "item_id", "rating"}
    missing = required - set(interactions_df.columns)
    if missing:
        raise ValueError(f"interactions_df is missing required columns: {missing}")

    for row in interactions_df.itertuples(index=False):
        rec.add_interaction(row.user_id, row.item_id, row.rating)


def clone_agent(agent: Any) -> Any:
    return SimpleNamespace(
        pos=agent.pos,
        model=agent.model,
        feed=list(agent.feed) if hasattr(agent, "feed") and agent.feed is not None else [],
        recommended_content=[],
        preference_vector=np.array(agent.preference_vector, copy=True),
        diversity_score=0.0,
        original_diversity_score=0.0,
    )


def build_agents_by_id(agents: Iterable[Any]) -> Dict[int, Any]:
    return {int(agent.pos): agent for agent in agents}


def train_mf_if_supported(rec: Recommender) -> None:
    if hasattr(rec, "train_mf") and callable(rec.train_mf):
        rec.train_mf(force_retrain=True)


def get_mf_recommendation_ids(rec: Recommender, agent: Any, k: int) -> List[int]:
    try:
        recommendations = rec.matrix_factorization(
            agent,
            num_recommendations=k,
            add_to_feed=False,
        )
        if recommendations is not None:
            return [int(content.content) for content in recommendations[:k]]
    except TypeError:
        pass

    agent.recommended_content = []
    old_num = rec.num_recommendations
    rec.num_recommendations = k
    try:
        rec.matrix_factorization(agent)
        return [int(content.content) for content in agent.recommended_content[:k]]
    finally:
        rec.num_recommendations = old_num


def evaluate_mf_params(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    agents_by_id: Dict[int, Any],
    mf_params: Dict[str, Any],
    config: TuneConfig,
) -> Dict[str, float]:
    rec = Recommender(
        recommender_type=RecommenderType.MATRIX_FACTORIZATION.value,
        diversity_level=config.diversity_level,
        num_recommendations=config.k,
        mf_params=mf_params,
    )

    load_interactions_into_recommender(rec, train_df)
    train_mf_if_supported(rec)

    val_user_items = val_df.groupby("user_id")["item_id"].apply(list)
    train_user_counts = train_df.groupby("user_id").size().to_dict()

    recall_scores: List[float] = []
    precision_scores: List[float] = []
    n_users_evaluated = 0

    for user_id, relevant_items in val_user_items.items():
        user_id = int(user_id)

        if user_id not in agents_by_id:
            continue

        if train_user_counts.get(user_id, 0) < config.min_train_interactions_per_user:
            continue

        agent = clone_agent(agents_by_id[user_id])
        rec_ids = get_mf_recommendation_ids(rec, agent, config.k)

        recall_scores.append(recall_at_k(rec_ids, relevant_items, config.k))
        precision_scores.append(precision_at_k(rec_ids, relevant_items, config.k))
        n_users_evaluated += 1

    return {
        "recall@k": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "precision@k": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "users_evaluated": int(n_users_evaluated),
    }


def evaluate_best_on_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    agents: Iterable[Any],
    best_params: Dict[str, Any],
    config: TuneConfig,
) -> Dict[str, float]:
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    agents_by_id = build_agents_by_id(agents)
    return evaluate_mf_params(train_val_df, test_df, agents_by_id, best_params, config)


def tune_biased_mf(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    agents: Iterable[Any],
    param_grid: Optional[Dict[str, List[Any]]] = None,
    config: Optional[TuneConfig] = None,
) -> pd.DataFrame:
    if config is None:
        config = TuneConfig()
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    agents_by_id = build_agents_by_id(agents)
    rows = []

    for params in ParameterGrid(param_grid):
        metrics = evaluate_mf_params(train_df, val_df, agents_by_id, params, config)
        rows.append({**params, **metrics})
        print(
            "Finished params:",
            params,
            "-> precision@k=",
            round(metrics["precision@k"], 5),
            "recall@k=",
            round(metrics["recall@k"], 5),
            "users=",
            metrics["users_evaluated"],
        )

    results_df = pd.DataFrame(rows).sort_values(["precision@k", "recall@k"], ascending=False)
    return results_df.reset_index(drop=True)


def main() -> None:
    model = FakeNewsModel(
        N=200,
        m_links=8,
        news_amount=400,
        fake_news_percentage=10,
        bot_percentage=7,
        influencer_percentage=3,
        diversity_level=0,
        num_recommendations=10,
        recommender_type=RecommenderType.MATRIX_FACTORIZATION.value,
        max_steps=700,
    )

    print("Running simulation to collect interaction data...")
    for _ in range(model.max_steps):
        model.step()

    interactions_df = interactions_from_model(model)
    if interactions_df.empty:
        raise ValueError("No interactions were collected from the simulation.")

    print(f"Collected {len(interactions_df)} interactions.")

    train_df, val_df, test_df = random_split(
        interactions_df,
        train_frac=0.7,
        val_frac=0.15,
        seed=42,
    )

    agents = model.agents

    config = TuneConfig(
        k=10,
        diversity_level=0.0,
        min_train_interactions_per_user=1,
        output_csv="biased_mf_tuning_results.csv",
        random_seed=42,
    )

    results_df = tune_biased_mf(
        train_df=train_df,
        val_df=val_df,
        agents=agents,
        param_grid=DEFAULT_PARAM_GRID,
        config=config,
    )

    results_df.to_csv(config.output_csv, index=False)

    print("\nTop 10 parameter settings:")
    print(results_df.head(10).to_string(index=False))

    if not results_df.empty:
        best = results_df.iloc[0].to_dict()
        best_params = {
            "embedding_size": int(best["embedding_size"]),
            "epochs": int(best["epochs"]),
            "regularization": float(best["regularization"]),
            "damping": float(best["damping"]),
        }

        print("\nBest validation config:")
        print(best_params)

        test_metrics = evaluate_best_on_test(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            agents=agents,
            best_params=best_params,
            config=config,
        )

        print("\nHeld-out test metrics:")
        print(test_metrics)
        print(f"\nSaved full tuning results to: {config.output_csv}")


if __name__ == "__main__":
    main()
