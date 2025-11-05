import numpy as np
import pandas as pd
from .registry import register_metric

def get_ranks(scores: np.ndarray) -> np.ndarray:
    """Convert scores to ranks (1-based)"""
    sorted_indices = scores.argsort()[::-1]
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(scores) + 1)
    return ranks

def get_dcg(relevance_labels: np.ndarray, ranks: np.ndarray, max_rank: int = None) -> float:
    """Calculate DCG with proper rank handling using 2**rel formula"""
    dcg = 0.0
    max_rank = max_rank or len(ranks)
    for i in range(len(relevance_labels)):
        if ranks[i] <= max_rank:
            dcg += (2**relevance_labels[i] - 1) / np.log2(ranks[i] + 1.0)
    return dcg

def _calculate_ndcg_for_column(group_df, relevance_column, k=10):
    """Generic NDCG calculation for any relevance column"""
    if len(group_df) == 0 or relevance_column not in group_df.columns:
        return None  # Will be neglected in aggregated metrics
    
    # Check if there are ANY relevant items in the ENTIRE dataset for this query
    all_relevance_labels = group_df[relevance_column].values
    if np.all(all_relevance_labels == 0):
        return None  # No relevant items exist - NDCG is undefined
    
    # Sort by rank (predicted ranking) and take top k for DCG calculation
    df_sorted = group_df.sort_values('rank').head(k)
    top_k_relevance = df_sorted[relevance_column].values
    
    # Calculate DCG using actual ranking (top k only)
    dcg = 0.0
    for i, rel in enumerate(top_k_relevance):
        dcg += (2**rel - 1) / np.log2(i + 2.0)  # i+2 because ranks are 1-based
    
    # Calculate IDCG using optimal ranking from ALL items (not just top k)
    # This is the key fix: IDCG should use the best k items from the entire dataset
    ideal_relevance = np.sort(all_relevance_labels)[::-1][:k]  # Best k items from entire set
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        idcg += (2**rel - 1) / np.log2(i + 2.0)
    
    # Return NDCG (can be 0.0 if DCG=0 but IDCG>0, indicating poor ranking)
    return dcg / idcg if idcg > 0 else None

@register_metric("ndcg_engagement")
def ndcg_engagement(group_df, k=10):
    """Calculate NDCG@k using engagement labels (0,1,2,3)"""
    return _calculate_ndcg_for_column(group_df, 'engagement', k)

@register_metric("ndcg_purchase") 
def ndcg_purchase(group_df, k=10):
    """Calculate NDCG@k using purchase labels (0,1)"""
    return _calculate_ndcg_for_column(group_df, 'purchase', k)

@register_metric("ndcg_autoship")
def ndcg_autoship(group_df, k=10):
    """Calculate NDCG@k using autoship labels (0,1)"""
    return _calculate_ndcg_for_column(group_df, 'autoship', k)
