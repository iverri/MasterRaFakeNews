# TODO: check the diversity score of the recommendations before and after reranking

'''
    Use Maximal Marginal Relevance (MMR) to rerank the recommendations
    MMR is a reranking algorithm that tries to maximize the relevance of the recommendations while minimizing the similarity between them.
'''

from sklearn.metrics.pairwise import cosine_similarity

def calculate_diversity(content):
    """Calculate the diversity score of the recommendations."""
    # Calculate the cosine similarity between the recommendations
    similarity_matrix = cosine_similarity(content)
    
    # Calculate the diversity score
    diversity_score = 1 - similarity_matrix.mean()
    
    return diversity_score

def diversity_reranking(user_preferences, recs, lambda_param=0.5, k=10):
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
                max_similarity = max(similarity_to_selected)
                
                # MMR formula: λ * relevance - (1-λ) * max_similarity
                mmr_score = lambda_param * relevance_scores[i] - (1 - lambda_param) * max_similarity
            
            mmr_scores.append((i, mmr_score))
        
        # Select the item with the highest MMR score
        best_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)
    
    # Create the reranked list
    reranked_recs = [recs[i] for i in selected_indices]
    
    # Calculate diversity score of the reranked recommendations
    diversity_score = calculate_diversity([rec.topic_vector for rec in reranked_recs])
    
    return reranked_recs, diversity_score

    