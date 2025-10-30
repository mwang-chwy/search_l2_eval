import numpy as np
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

@register_metric("ndcg")
def ndcg(group_df, k=10):
    """Calculate NDCG@k using proper ranking logic with 2**rel formula"""
    # Sort by rank (predicted ranking)
    df_sorted = group_df.sort_values('rank').head(k)
    
    if len(df_sorted) == 0:
        return None  # Will be neglected in aggregated metrics
    
    relevance_labels = df_sorted['relevance'].values
    # Use negative rank because higher rank = better position
    scores = -df_sorted['rank'].values  
    
    # Calculate DCG using actual ranking
    dcg = 0.0
    for i, rel in enumerate(relevance_labels):
        dcg += (2**rel - 1) / np.log2(i + 2.0)  # i+2 because ranks are 1-based
    
    # Calculate IDCG using optimal ranking
    ideal_relevance = np.sort(relevance_labels)[::-1]
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        idcg += (2**rel - 1) / np.log2(i + 2.0)
    
    return dcg / idcg if idcg > 0 else 0.0
