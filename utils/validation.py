def rename_columns(df, mapping):
    rename_map = {v: k for k, v in mapping.items() if v in df.columns}
    return df.rename(columns=rename_map)

def validate_schema(df, required_fields):
    missing = [f for f in required_fields if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
