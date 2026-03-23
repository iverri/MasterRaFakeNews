from recommender.base_recommender import BaseRecommender


def create_recommender(
    library: str,
    recommender_type: str,
    diversity_level: float,
    num_recommendations: int,
) -> BaseRecommender:
    """Creates a recommender based on specifications given by the simulation

    Args:
        library (str): 'lenskit' or 'recbole' depending on experiment specifications
        recommender_type (str): type of recommender to create
        diversity_level (float): level of diversity
        num_recommendations (int): number of items to recommend (top k)

    Raises:
        ValueError: Raises if the Library is not specified correctly

    Returns:
        BaseRecommender: The recommender type matching the specifications
    """
    if library == "lenskit":
        from recommender.lenskit.lenskit_recommender import LenskitRecommender

        return LenskitRecommender(
            recommender_type, diversity_level, num_recommendations
        )

    elif library == "recbole":
        from recommender.recbole.recbole_recommender import RecboleRecommender

        return RecboleRecommender(
            recommender_type, diversity_level, num_recommendations
        )

    else:
        raise ValueError(
            "Library must be either 'lenskit' or 'recbole'. Make sure the recommender matches your current environment."
        )
