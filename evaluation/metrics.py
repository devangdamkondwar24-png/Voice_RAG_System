"""
evaluation/metrics.py
─────────────────────
Standard RAG evaluation metrics for retrieval and generation.
"""

from __future__ import annotations

import math
from typing import List


def calculate_mrr(ranks: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        ranks: List of ranks of the first relevant document for each query.
               (1-indexed). Use 0 if no relevant doc found.
               
    Returns:
        MRR score [0.0, 1.0]
    """
    if not ranks:
        return 0.0
        
    reciprocal_ranks = [1.0 / r if r > 0 else 0.0 for r in ranks]
    return sum(reciprocal_ranks) / len(ranks)


def calculate_recall_at_k(hits_at_k: List[bool]) -> float:
    """
    Calculate Recall@K (percentage of queries where at least one 
    relevant document was found in the top K).
    
    Args:
        hits_at_k: List of booleans (True if relevant doc in top K)
        
    Returns:
        Recall@K score [0.0, 1.0]
    """
    if not hits_at_k:
        return 0.0
        
    return sum(1 for hit in hits_at_k if hit) / len(hits_at_k)


def calculate_ndcg_at_k(
    relevance_scores_list: List[List[float]],
    k: int = 10,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG@K).
    
    Args:
        relevance_scores_list: List of lists of relevance scores (e.g. 1.0 for relevant, 0.0 for not)
                               for the top retrieved documents per query.
        k: The cutoff rank.
        
    Returns:
        Mean nDCG@K [0.0, 1.0]
    """
    if not relevance_scores_list:
        return 0.0

    ndcg_scores = []
    
    for scores in relevance_scores_list:
        scores = scores[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, score in enumerate(scores):
            # i is 0-indexed, rank is i+1
            # Standard DCG formula: rel_i / log2(i + 2)
            dcg += score / math.log2(i + 2)
            
        # Calculate IDCG (Ideal DCG)
        ideal_scores = sorted(scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / math.log2(i + 2)
            
        if idcg == 0.0:
            ndcg_scores.append(0.0)
        else:
            ndcg_scores.append(dcg / idcg)
            
    return sum(ndcg_scores) / len(ndcg_scores)


def calculate_abstention_rate(abstained_list: List[bool]) -> float:
    """Calculate percentage of queries where the system abstained."""
    if not abstained_list:
        return 0.0
    return sum(1 for a in abstained_list if a) / len(abstained_list)


def calculate_mean_grounding(scores: List[float]) -> float:
    """Calculate average grounding score across all non-abstained generations."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
