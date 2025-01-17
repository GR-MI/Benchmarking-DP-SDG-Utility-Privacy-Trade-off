import pandas as pd
import numpy as np
import json
import os

# Read the dataset
df = pd.read_csv('')

# Specify the columns for binning
columns_to_bin = ['col1', 'col2'] 

bin_mappings = {}

# Apply Binning based on 5% of the max value for each specified column
for col in columns_to_bin:
    max_value = df[col].max()
    bin_edges = [i * 0.05 * max_value for i in range(0, 21)]
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges) - 1)]
    df[col + '_binned'] = pd.cut(df[col], bins=bin_edges, labels=bin_labels, right=False, include_lowest=True)

    bin_mappings[col] = {
        'bin_edges': bin_edges,
        'bin_labels': bin_labels
    }

processed_data_path = '' 
df.to_csv(processed_data_path, index=False)

bin_mappings_path = '' 
os.makedirs(os.path.dirname(bin_mappings_path), exist_ok=True) 
with open(bin_mappings_path, 'w') as json_file:
    json.dump(bin_mappings, json_file, indent=4)

print("Processed DataFrame with Binned Columns:")
print(df.head())

print(f"\nProcessed dataset saved to: {processed_data_path}")
print(f"Bin mappings saved to: {bin_mappings_path}")
