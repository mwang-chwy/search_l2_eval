import pandas as pd
import numpy as np
from .registry import register_metric

@register_metric("revenue_per_purchased_search")
def revenue_per_purchased_search(group_df, k=10):
    """
    Calculate revenue per search, but only for searches that have at least one purchase.
    Returns None for searches with no purchases (excluded from aggregation).
    
    This metric helps evaluate revenue performance specifically for purchase-intent searches,
    excluding searches where users didn't purchase anything.
    
    Args:
        group_df: DataFrame for a single search query with columns 'rank', 'purchase', 'revenue'
        k: Number of top results to consider
        
    Returns:
        float: Total revenue from top k results if any purchase exists, None otherwise
    """
    if len(group_df) == 0:
        return None
        
    # Sort by rank and take top k results
    top_k = group_df.nsmallest(k, 'rank')
    
    # Check if there are any purchases in the ENTIRE search (not just top k)
    # This determines if this is a "purchase search" or not
    has_any_purchase = False
    if 'purchase' in group_df.columns:
        has_any_purchase = (group_df['purchase'] > 0).any()
    
    # If no purchases in entire search, return None (excluded from aggregation)
    if not has_any_purchase:
        return None
        
    # Calculate revenue from top k results
    # Even if top k has no purchases, we still return revenue (could be 0)
    # because this search had purchase intent (purchases exist elsewhere)
    if 'revenue' in top_k.columns:
        total_revenue = top_k['revenue'].fillna(0).sum()
        return float(total_revenue)
    
    # Fallback: calculate revenue from purchase and price if revenue column doesn't exist
    if 'purchase' in top_k.columns and 'price' in top_k.columns:
        # Revenue = purchase * price for each item
        revenue_calculated = (top_k['purchase'].fillna(0) * top_k['price'].fillna(0)).sum()
        return float(revenue_calculated)
    
    # If we can't calculate revenue, return 0 for purchase searches
    return 0.0