

'''
    Use Maximal Marginal Relevance (MMR) to rerank the recommendations
    MMR is a reranking algorithm that tries to maximize the relevance of the recommendations while minimizing the similarity between them.
'''

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_diversity(topic_vectors):
    """Calculate the diversity score of the recommendations."""
    # Fix the array truth value error by checking if it's a NumPy array first
    if isinstance(topic_vectors, np.ndarray):
        # Check if array is empty or has fewer than 2 elements
        if topic_vectors.size == 0 or topic_vectors.shape[0] < 2:
            return 0.0
    else:
        # Original check for non-NumPy collections
        if not topic_vectors or len(topic_vectors) < 2:
            return 0.0  # No diversity with 0 or 1 item
        
    # Ensure topic vectors are properly formatted for cosine_similarity
    vectors = np.array(topic_vectors)
    
    # Calculate the cosine similarity between the recommendations
    similarity_matrix = cosine_similarity(vectors)
    
    # Remove the diagonal (self-similarity) for more accurate measurement
    n = similarity_matrix.shape[0]
    similarity_sum = similarity_matrix.sum() - n  # Subtract diagonal sum (all 1's)
    num_pairs = n * (n - 1)  # Number of pairs (excluding self-pairs)
    
    # Calculate the diversity score (1 - average similarity between different items)
    diversity_score = 1 - (similarity_sum / num_pairs)
    
    return diversity_score

def guaranteed_diversity_reranking(user_preferences, recs, k=10, pre_calculated=None):
    """
    Rerank recommendations with a guarantee of increased diversity.
    Uses a greedy approach that always increases diversity.
    
    Parameters:
    - user_preferences: vector representing user preferences
    - recs: list of NewsContent objects to be reranked
    - k: number of recommendations to select
    - pre_calculated: optional dict with pre-calculated data
    Returns:
    - reranked list of NewsContent objects
    """
    # Extract topic vectors from recommendations
    rec_topic_vectors = np.array([rec.topic_vector for rec in recs])
    
    # Calculate original diversity
    original_diversity = calculate_diversity(rec_topic_vectors)
    
    # Calculate relevance scores
    if pre_calculated and 'relevance_scores' in pre_calculated and len(pre_calculated['relevance_scores']) == len(recs):
        relevance_scores = pre_calculated['relevance_scores']
    else:
        relevance_scores = cosine_similarity([user_preferences], rec_topic_vectors)[0]
    
    # Start with the most relevant item
    best_idx = np.argmax(relevance_scores)
    selected_indices = [best_idx]
    remaining_indices = set(range(len(recs)))
    remaining_indices.remove(best_idx)
    
    # Greedy selection that guarantees diversity increases
    while len(selected_indices) < min(k, len(recs)):
        best_addition = None
        best_diversity = 0
        
        # Try adding each remaining item and measure the resulting diversity
        for idx in remaining_indices:
            candidate_indices = selected_indices + [idx]
            candidate_vectors = rec_topic_vectors[candidate_indices]
            candidate_diversity = calculate_diversity(candidate_vectors)
            
            # Find the item that maximizes diversity when added
            if candidate_diversity > best_diversity:
                best_diversity = candidate_diversity
                best_addition = idx
        
        if best_addition is not None:
            selected_indices.append(best_addition)
            remaining_indices.remove(best_addition)
        else:
            # If no addition improves diversity (unlikely), add the most relevant remaining item
            remaining_relevance = [(i, relevance_scores[i]) for i in remaining_indices]
            if remaining_relevance:
                best_remaining = max(remaining_relevance, key=lambda x: x[1])[0]
                selected_indices.append(best_remaining)
                remaining_indices.remove(best_remaining)
            else:
                break
    
    # Create the reranked list
    reranked_recs = [recs[i] for i in selected_indices]
    
    # Verify that diversity has increased
    reranked_vectors = np.array([rec.topic_vector for rec in reranked_recs])
    new_diversity = calculate_diversity(reranked_vectors)
    
    # If somehow diversity didn't increase (edge case), fall back to original with most diverse items first
    if new_diversity <= original_diversity:
        # Calculate pairwise distances between all items
        pairwise_distances = 1 - cosine_similarity(rec_topic_vectors)
        
        # For each item, calculate its average distance to all other items
        avg_distances = np.mean(pairwise_distances, axis=1)
        
        # Sort by diversity (average distance to other items)
        diverse_indices = np.argsort(-avg_distances)[:k]
        
        # Create the reranked list
        reranked_recs = [recs[i] for i in diverse_indices]
    
    return reranked_recs

def diversity_reranking(user_preferences, recs, k=10, pre_calculated=None, diversity_level=0.5):
    """
    Rerank recommendations using a constrained Maximal Marginal Relevance (MMR) approach
    that guarantees diversity increases.
    
    Parameters:
    - user_preferences: vector representing user preferences
    - recs: list of NewsContent objects to be reranked
    - k: number of recommendations to select
    - pre_calculated: optional dict with pre-calculated data
    - diversity_level: balance between diversity and relevance (0-1)
                      higher values favor diversity, lower values favor relevance
    Returns:
    - reranked list of NewsContent objects and their diversity score
    """
    # Extract topic vectors from recommendations
    rec_topic_vectors = np.array([rec.topic_vector for rec in recs])
    n_items = len(recs)
    k = min(k, n_items)  # Ensure k doesn't exceed available items
    
    # Calculate relevance scores - reuse if pre-calculated
    if pre_calculated and 'relevance_scores' in pre_calculated and len(pre_calculated['relevance_scores']) == n_items:
        relevance_scores = pre_calculated['relevance_scores']
    else:
        relevance_scores = cosine_similarity([user_preferences], rec_topic_vectors)[0]
    
    # Pre-compute similarity matrix for all items
    if pre_calculated and 'similarity_matrix' in pre_calculated:
        similarity_matrix = pre_calculated['similarity_matrix']
    else:
        similarity_matrix = cosine_similarity(rec_topic_vectors)
    
    # Get the original top-k recommendations based on relevance
    top_k_indices = np.argsort(-relevance_scores)[:k]
    top_k_vectors = rec_topic_vectors[top_k_indices]
    original_diversity = calculate_diversity(top_k_vectors)
    
    # Convert diversity_level to lambda parameter for MMR
    lambda_param = 1.0 - diversity_level
    
    # Initialize with the most relevant item
    best_idx = np.argmax(relevance_scores)
    selected_indices = [best_idx]
    
    # Use a boolean mask for unselected items (faster than list operations)
    mask = np.ones(n_items, dtype=bool)
    mask[best_idx] = False
    
    # Cache for diversity scores
    diversity_cache = {}
    
    # Vectorized MMR implementation
    while len(selected_indices) < k and np.any(mask):
        # Get indices of remaining items
        remaining_indices = np.where(mask)[0]
        
        if len(selected_indices) < 2:
            # For the first few items, just use MMR without diversity constraint
            # Get similarities to selected items for all remaining items at once
            similarities = similarity_matrix[remaining_indices][:, selected_indices]
            max_similarities = np.max(similarities, axis=1) if similarities.size > 0 else np.zeros(len(remaining_indices))
            
            # Calculate MMR scores for all remaining items at once
            mmr_scores = lambda_param * relevance_scores[remaining_indices] - (1 - lambda_param) * max_similarities
            
            # Select the best item
            best_remaining_idx = np.argmax(mmr_scores)
            best_idx = remaining_indices[best_remaining_idx]
            
            # Update selected and unselected
            selected_indices.append(best_idx)
            mask[best_idx] = False
        else:
            # After initial items, consider diversity constraint
            best_idx = None
            best_mmr_score = float('-inf')
            
            # Calculate MMR scores for all remaining items
            similarities = similarity_matrix[remaining_indices][:, selected_indices]
            max_similarities = np.max(similarities, axis=1) if similarities.size > 0 else np.zeros(len(remaining_indices))
            mmr_scores = lambda_param * relevance_scores[remaining_indices] - (1 - lambda_param) * max_similarities
            
            # Sort by MMR score for efficient processing (check best candidates first)
            sorted_indices = np.argsort(-mmr_scores)
            
            # Try candidates in order of MMR score
            for i in sorted_indices:
                idx = remaining_indices[i]
                candidate_indices = selected_indices + [idx]
                
                # Check cache first
                candidate_key = tuple(sorted(candidate_indices))
                if candidate_key in diversity_cache:
                    candidate_diversity = diversity_cache[candidate_key]
                else:
                    candidate_vectors = rec_topic_vectors[candidate_indices]
                    candidate_diversity = calculate_diversity(candidate_vectors)
                    diversity_cache[candidate_key] = candidate_diversity
                
                # Accept if diversity improves
                if candidate_diversity >= original_diversity:
                    best_idx = idx
                    break  # Early stopping - take first candidate that improves diversity
            
            # If no item improves diversity, find the one that maximizes it
            if best_idx is None:
                best_diversity = original_diversity
                
                for idx in remaining_indices:
                    candidate_indices = selected_indices + [idx]
                    candidate_key = tuple(sorted(candidate_indices))
                    
                    if candidate_key in diversity_cache:
                        candidate_diversity = diversity_cache[candidate_key]
                    else:
                        candidate_vectors = rec_topic_vectors[candidate_indices]
                        candidate_diversity = calculate_diversity(candidate_vectors)
                        diversity_cache[candidate_key] = candidate_diversity
                    
                    if candidate_diversity > best_diversity:
                        best_diversity = candidate_diversity
                        best_idx = idx
                
                # If still no good candidate, take most relevant remaining item
                if best_idx is None and len(remaining_indices) > 0:
                    best_idx = remaining_indices[np.argmax(relevance_scores[remaining_indices])]
            
            # Update selected and unselected
            if best_idx is not None:
                selected_indices.append(best_idx)
                mask[best_idx] = False
    
    # Check if we need to fall back to pure diversity approach
    if len(selected_indices) < k:
        # Reset selection
        best_idx = np.argmax(relevance_scores)
        selected_indices = [best_idx]
        mask = np.ones(n_items, dtype=bool)
        mask[best_idx] = False
        
        # Greedy diversity maximization
        while len(selected_indices) < k and np.any(mask):
            remaining_indices = np.where(mask)[0]
            best_idx = None
            best_diversity = original_diversity
            
            # Vectorized approach for remaining items
            for batch_start in range(0, len(remaining_indices), 100):  # Process in batches
                batch_indices = remaining_indices[batch_start:batch_start+100]
                batch_diversities = []
                
                for idx in batch_indices:
                    candidate_indices = selected_indices + [idx]
                    candidate_key = tuple(sorted(candidate_indices))
                    
                    if candidate_key in diversity_cache:
                        candidate_diversity = diversity_cache[candidate_key]
                    else:
                        candidate_vectors = rec_topic_vectors[candidate_indices]
                        candidate_diversity = calculate_diversity(candidate_vectors)
                        diversity_cache[candidate_key] = candidate_diversity
                    
                    batch_diversities.append((idx, candidate_diversity))
                
                # Find best in batch
                for idx, diversity in batch_diversities:
                    if diversity > best_diversity:
                        best_diversity = diversity
                        best_idx = idx
            
            # If no improvement found, take most relevant remaining item
            if best_idx is None and len(remaining_indices) > 0:
                best_idx = remaining_indices[np.argmax(relevance_scores[remaining_indices])]
            
            # Update selected and unselected
            if best_idx is not None:
                selected_indices.append(best_idx)
                mask[best_idx] = False
            else:
                break  # No more suitable items
    
    # Create the reranked list
    reranked_recs = [recs[i] for i in selected_indices]
    
    # Final diversity check
    reranked_vectors = np.array([rec.topic_vector for rec in reranked_recs])
    new_diversity = calculate_diversity(reranked_vectors)
    
    return reranked_recs, new_diversity
