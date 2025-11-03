import pandas as pd
import numpy as np
from typing import Dict, Any

def load_pdm_data(pdm_path: str, pdm_config: Dict[str, Any]) -> pd.DataFrame:
    """Load and clean PDM data"""
    print(f"Loading PDM data from: {pdm_path}")
    
    # Load with string dtype for part numbers to preserve leading zeros
    pdm_df = pd.read_csv(
        pdm_path, 
        dtype={pdm_config["part_number_col"]: "string"}, 
        low_memory=False
    )
    
    # Clean part numbers in place
    pdm_df[pdm_config["part_number_col"]] = (
        pdm_df[pdm_config["part_number_col"]]
        .astype('string')
        .str.strip()
    )
    
    return pdm_df

def enrich_with_pdm(df: pd.DataFrame, pdm_df: pd.DataFrame, pdm_config: Dict[str, Any]) -> pd.DataFrame:
    """Enrich evaluation data (eval_data.csv) with PDM information
    
    This function adds mc1, mc2, and price to each product in eval_data.csv
    based on matching part numbers from the PDM table.
    
    Args:
        df: Evaluation data with part_number column
        pdm_df: PDM table with part numbers, MC categories, and prices
        pdm_config: Configuration specifying PDM column names
    """
    
    # Copy dataframe to avoid modifying original
    df = df.copy()
    
    # Ensure consistent data types for merge columns
    part_number_col = pdm_config["part_number_col"]
    if 'part_number' in df.columns:
        df['part_number'] = df['part_number'].astype('string')
    if part_number_col in pdm_df.columns:
        pdm_df = pdm_df.copy()
        pdm_df[part_number_col] = pdm_df[part_number_col].astype('string')
    
    # Select relevant PDM columns (only part number, MC categories, and price)
    pdm_cols = [
        pdm_config["part_number_col"],  # Part number for matching
        pdm_config["mc1_col"],         # MC1 category  
        pdm_config["mc2_col"],         # MC2 category
        pdm_config["price_col"]        # Product price
    ]
    
    pdm_subset = pdm_df[pdm_cols].drop_duplicates(
        subset=[pdm_config["part_number_col"]], 
        keep='first'
    )
    
    # Merge evaluation data with PDM data on part numbers
    enriched_df = df.merge(
        pdm_subset,
        left_on='part_number',                   # Part number in eval data
        right_on=pdm_config["part_number_col"], # Part number in PDM data
        how='left'
    )
    
    # Rename columns to standard names
    enriched_df = enriched_df.rename(columns={
        pdm_config["mc1_col"]: "mc1",
        pdm_config["mc2_col"]: "mc2",
        pdm_config["price_col"]: "price"
    })
    
    return enriched_df

def assign_search_mc(df: pd.DataFrame) -> pd.DataFrame:
    """Assign dominant MC categories to each search query based on products in that query
    
    For each query_id (search_id), calculates the mode (most frequent) mc1 and mc2
    from all products associated with that query in the evaluation data.
    This gives us the dominant product categories for each search query.
    """
    
    def get_dominant_mc(series):
        """Get the most common (mode) MC category for a search query"""
        mc_series = series.astype(str).fillna('MISSING_MC')
        mode_result = mc_series.mode()
        
        if not mode_result.empty and mode_result.iloc[0] != 'MISSING_MC':
            return mode_result.iloc[0]
        return 'UNKNOWN'
    
    # Group by search_id (query_id) and find dominant MC categories from products in each query
    search_mc = df.groupby('search_id').agg({
        'mc1': get_dominant_mc,  # Mode of mc1 across products in this query
        'mc2': get_dominant_mc   # Mode of mc2 across products in this query
    }).reset_index()
    
    search_mc = search_mc.rename(columns={
        'mc1': 'search_mc1',
        'mc2': 'search_mc2'
    })
    
    # Merge back with original dataframe
    df_with_search_mc = df.merge(search_mc, on='search_id', how='left')
    
    return df_with_search_mc

def calculate_revenue(df: pd.DataFrame, purchase_label: int) -> pd.DataFrame:
    """Calculate revenue based on purchase label
    
    Revenue is calculated dynamically from PDM price data:
    If purchase == purchase_label:
        - Use price from PDM product table (preferred approach)
        - Fallback to existing revenue column if present and not null
    If purchase != purchase_label:
        - Revenue = 0 (no purchase)
    """
    df = df.copy()
    
    # For purchased items (purchase == purchase_label):
    # 1. Use existing revenue if available and not null
    # 2. Otherwise use price from PDM table
    # For non-purchased items: revenue = 0
    
    def get_revenue_value(row):
        # Check if purchase column exists, fallback to relevance for backward compatibility
        purchase_col = 'purchase' if 'purchase' in df.columns else 'relevance'
        
        if row[purchase_col] != purchase_label:
            return 0
        
        # Check if revenue column exists and has a value
        if 'revenue' in df.columns and pd.notna(row['revenue']):
            return row['revenue']
        
        # Fall back to price from PDM table
        return row['price'] if pd.notna(row['price']) else 0
    
    df['revenue'] = df.apply(get_revenue_value, axis=1)
    
    return df