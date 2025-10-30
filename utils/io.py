import pandas as pd
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_csv(path):
    return pd.read_csv(path)

def save_csv(df, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
