import numpy as np
from .registry import register_metric

@register_metric("revenue_per_search")
def revenue_per_search(group_df, k=None):
    """Calculate Revenue Per Search at rank k"""
    df = group_df.sort_values("rank")
    if k is not None:
        df = df.head(k)
    
    if len(df) == 0:
        return None  # Will be neglected in aggregated metrics
    
    # Sum actual revenue for top k items
    total_revenue = df["revenue"].sum()
    return float(total_revenue)

@register_metric("revenue_recall")
def revenue_recall(group_df, k=None):
    """Calculate what fraction of total revenue is captured in top k"""
    if len(group_df) == 0:
        return None  # Will be neglected in aggregated metrics
        
    total_revenue = group_df["revenue"].sum()
    if total_revenue == 0:
        return None  # Will be neglected in aggregated metrics
    
    df_sorted = group_df.sort_values("rank")
    if k is not None:
        df_sorted = df_sorted.head(k)
    
    if len(df_sorted) == 0:
        return None  # Will be neglected in aggregated metrics
    
    top_k_revenue = df_sorted["revenue"].sum()
    return float(top_k_revenue / total_revenue)
