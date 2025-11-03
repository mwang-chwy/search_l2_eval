import numpy as np
import pandas as pd
from .registry import register_metric

@register_metric("ctr")
def ctr(group_df, k=10, ctr_label=1):
    """Calculate CTR@k (Click-Through Rate at k)
    
    CTR@k measures the fraction of queries where at least one item 
    in the top k results has engagement >= ctr_label.
    
    Args:
        group_df: DataFrame for a single query group
        k: Number of top results to consider
        ctr_label: Minimum engagement level to count as a "click" (default: 1)
    
    Returns:
        1.0 if any item in top k has engagement >= ctr_label, 0.0 otherwise
    """
    # Sort by rank (predicted ranking) and take top k
    df_sorted = group_df.sort_values('rank').head(k)
    
    if len(df_sorted) == 0 or 'engagement' not in df_sorted.columns:
        return None
    
    # Check if any item in top k has engagement >= ctr_label
    has_click = (df_sorted['engagement'] >= ctr_label).any()
    
    return 1.0 if has_click else 0.0

@register_metric("cvr")
def cvr(group_df, k=10, cvr_label=1):
    """Calculate CVR@k using purchase column (Conversion Rate at k)
    
    CVR@k measures the fraction of queries where at least one item
    in the top k results has purchase >= cvr_label (purchase conversion).
    
    Args:
        group_df: DataFrame for a single query group
        k: Number of top results to consider  
        cvr_label: Minimum purchase value to count as a "conversion" (default: 1)
    
    Returns:
        1.0 if any item in top k has purchase >= cvr_label, 0.0 otherwise
    """
    # Sort by rank (predicted ranking) and take top k
    df_sorted = group_df.sort_values('rank').head(k)
    
    if len(df_sorted) == 0 or 'purchase' not in df_sorted.columns:
        return None
    
    # Check if any item in top k has purchase >= cvr_label
    has_conversion = (df_sorted['purchase'] >= cvr_label).any()
    
    return 1.0 if has_conversion else 0.0