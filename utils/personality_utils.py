import numpy as np

def cosine_similarity(vec_a, vec_b):
    """Calculate the cosine similarity between two vectors."""
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def personality_similarity(personality_a, personality_b):
    """Calculate similarity between two personality vectors."""
    return cosine_similarity(personality_a, personality_b)

def likely_to_follow(personality_vector):
    """Calculate follow propensity based on personality traits.
    Higher -> agent follows more accounts, Lower -> agent follows fewer accounts.
    """

    E, A, C, N, O = personality_vector
    #these numbers may need to be adjusted, just to try out as of now
    #extraversion, openness and agreeableness are positively correlated with following more accounts, while neuroticism and conscientiousness are negatively correlated

    return (0.4 * E) + (0.2 * O) + (0.1 * A) - (0.2 * N) - (0.2 * C)

def likely_to_be_followed(personality_vector):
    """Calculate attractiveness based on personality traits.
    Higher -> agent is more likely to be followed, Lower -> agent is less likely to be followed.
    """

    E, A, C, N, O = personality_vector
    #extraversion, openness and agreeableness are positively correlated with being followed, while neuroticism is negatively correlated. No findings on conscientiousness

    return (0.4 * E) + (0.15 * O) + (0.1 * A) - (0.1 * N)
