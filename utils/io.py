import pandas as pd
import yaml
import os
from pathlib import Path

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(path):
    """Load data from CSV or Parquet file based on file extension"""
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if file_path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    elif file_path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .csv, .parquet")

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
