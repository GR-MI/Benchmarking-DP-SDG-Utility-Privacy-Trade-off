import pandas as pd
import json


file_path = ''
df = pd.read_csv(file_path)

def infer_sdtype(dtype):
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    elif pd.api.types.is_numeric_dtype(dtype):
        return "numerical"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    elif pd.api.types.is_object_dtype(dtype):
        return "categorical"
    else:
        return "other"

primary_key = "id"
for col in df.columns:
    if df[col].is_unique:
        primary_key = col
        break

metadata = {
    "primary_key": primary_key,
    "columns": {
        col: {"sdtype": infer_sdtype(df[col].dtype)} for col in df.columns
    }
}

metadata_file = ''
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Metadata saved to {metadata_file}")
