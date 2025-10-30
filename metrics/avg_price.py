import numpy as np
from .registry import register_metric

@register_metric("avg_price")
def avg_price(group_df, k=None):
    """Calculate average price at rank k"""
    df = group_df.sort_values("rank")
    if k is not None:
        df = df.head(k)
    
    if len(df) == 0:
        return None  # Will be neglected in aggregated metrics
    
    return float(df["price"].mean())
