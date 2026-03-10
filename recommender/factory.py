def create_recommender(library, recommender_type, diversity_level, num_recommendations):
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
            "Library must be either 'lenskit' or 'recbole'. Make sure the recommender matches your current environment. OWAAAAAA"
        )
