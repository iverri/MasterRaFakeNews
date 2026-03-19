import numpy as np

def likely_to_follow(personality_vector):
    """Calculate follow propensity based on personality traits.
    Higher -> agent follows more accounts, Lower -> agent follows fewer accounts.
    """

    E, A, C, N, O = personality_vector
    #these numbers may need to be adjusted, just to try out as of now
    #extraversion, openness and agreeableness are positively correlated with following more accounts, while neuroticism and conscientiousness are negatively correlated

    return (0.4 * E) + (0.2 * O) + (0.1 * A) + (0.1 * N) - (0.2 * C)

def likely_to_be_followed(personality_vector):
    """Calculate attractiveness based on personality traits.
    Higher -> agent is more likely to be followed, Lower -> agent is less likely to be followed.
    """

    E, A, C, N, O = personality_vector
    #extraversion, openness and agreeableness are positively correlated with being followed, while conscientiousness is negatively correlated. No findings on neuroticism

    return (0.4 * E) + (0.2 * O) + (0.2 * A) - (0.1 * C)

def personality_homophily(personality_vector_i, personality_vector_j):
    """Calculate homophily based on personality similarity.
    Higher -> more similar personalities, more likely to connect.
    Lower -> less similar personalities, less likely to connect.
    """

    E_i, A_i, C_i, N_i, O_i = personality_vector_i
    E_j, A_j, C_j, N_j, O_j = personality_vector_j

    #No findings for conscitiousness and neuroticism, so we will not include them in the homophily calculation for now. We can always adjust this later if we find relevant research.
    sim_E = 1 - abs(E_i - E_j)
    sim_A = 1 - abs(A_i - A_j)
    sim_O = 1 - abs(O_i - O_j)

    return (0.4 * sim_E) + (0.3 * sim_O) + (0.3 * sim_A)

    
