# TODO: check the diversity score of the recommendations before and after reranking

'''
    Use Maximal Marginal Relevance (MMR) to rerank the recommendations
    MMR is a reranking algorithm that tries to maximize the relevance of the recommendations while minimizing the similarity between them.
'''

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_diversity(topic_vectors):
    """Calculate the diversity score of the recommendations."""
    if not topic_vectors or len(topic_vectors) < 2:
        return 0.0  # No diversity with 0 or 1 item
        
    # Ensure topic vectors are properly formatted for cosine_similarity
    vectors = np.array(topic_vectors)
    
    # Calculate the cosine similarity between the recommendations
    similarity_matrix = cosine_similarity(vectors)
    
    # Calculate the diversity score
    diversity_score = 1 - similarity_matrix.mean()
    
    return diversity_score

def diversity_reranking__MMR(user_preferences, recs, lambda_param, k=10):
    """
    Rerank the recommendations based on diversity using Maximal Marginal Relevance (MMR).
    
    Parameters:
    - user_preferences: vector representing user preferences
    - recs: list of NewsContent objects to be reranked
    - lambda_param: balance between relevance and diversity (0-1)
                    higher values favor relevance, lower values favor diversity
    - k: number of recommendations to select
    Returns:
    - reranked list of NewsContent objects
    """
    # Extract topic vectors from recommendations
    rec_topic_vectors = [rec.topic_vector for rec in recs]
    
    # Calculate relevance scores (similarity between user preferences and recommendations)
    relevance_scores = cosine_similarity([user_preferences], rec_topic_vectors)[0]
    
    selected_indices = []
    unselected_indices = list(range(len(recs)))
    
    # Select items one by one using MMR
    while unselected_indices and len(selected_indices) < k:
        mmr_scores = []
        
        for i in unselected_indices:
            if not selected_indices:
                # For the first item, MMR is just the relevance score
                mmr_score = relevance_scores[i]
            else:
                # Calculate similarity to already selected items
                selected_vectors = [rec_topic_vectors[j] for j in selected_indices]
                similarity_to_selected = cosine_similarity([rec_topic_vectors[i]], selected_vectors)[0]
                max_similarity = max(similarity_to_selected) ** 0.5
                
                # MMR formula: λ * relevance - (1-λ) * max_similarity
                mmr_score = lambda_param * relevance_scores[i] - (1 - lambda_param) * max_similarity
            
            mmr_scores.append((i, mmr_score))
        
        # Select the item with the highest MMR score
        best_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)
    
    # Create the reranked list
    reranked_recs = [recs[i] for i in selected_indices]
    
    return reranked_recs

def diversity_reranking(user_preferences, recs, k=10, pre_calculated=None):
    """
    Rerank recommendations using Determinantal Point Process (DPP).
    
    Parameters:
    - user_preferences: vector representing user preferences
    - recs: list of NewsContent objects to be reranked
    - k: number of recommendations to select
    - pre_calculated: optional dict with pre-calculated data (topic_vectors, relevance_scores)
    Returns:
    - reranked list of NewsContent objects
    """
    from dppy.finite_dpps import FiniteDPP
    
    # Use pre-calculated data if provided, otherwise calculate
    if pre_calculated and 'topic_vectors' in pre_calculated and 'relevance_scores' in pre_calculated:
        # Make sure pre-calculated data matches the recommendations list
        if len(pre_calculated['topic_vectors']) == len(recs):
            rec_topic_vectors = pre_calculated['topic_vectors']
            relevance_scores = pre_calculated['relevance_scores']
        else:
            # If lengths don't match, we need to recalculate
            rec_topic_vectors = [rec.topic_vector for rec in recs]
            relevance_scores = cosine_similarity([user_preferences], rec_topic_vectors)[0]
    else:
        # Extract topic vectors from recommendations
        rec_topic_vectors = [rec.topic_vector for rec in recs]
        
        # Calculate relevance scores (similarity between user preferences and recommendations)
        relevance_scores = cosine_similarity([user_preferences], rec_topic_vectors)[0]
    
    # Handle edge cases
    if len(recs) <= 1 or k <= 1:
        # Sort by relevance and return top-k
        indices = np.argsort(-np.array(relevance_scores))
        return [recs[i] for i in indices[:k]]
    
    # Create quality vector (relevance scores)
    quality = np.array(relevance_scores)
    
    # Create similarity kernel
    similarity = cosine_similarity(rec_topic_vectors)
    
    # Create L-ensemble kernel: L = diag(quality) * similarity * diag(quality)
    L = np.diag(quality) @ similarity @ np.diag(quality)
    
    # Initialize DPP with L-ensemble
    dpp = FiniteDPP('likelihood', **{'L': L})
    
    # Sample from DPP
    dpp.sample_exact_k_dpp(size=min(k, len(recs)))
    selected_indices = list(dpp.list_of_samples[0])
    
    # Create the reranked list
    reranked_recs = [recs[i] for i in selected_indices]
    
    return reranked_recs
