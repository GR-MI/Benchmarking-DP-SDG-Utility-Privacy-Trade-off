import pandas as pd
import json
import os

input_files = [
]
label_mappings_path = ''
output_dir = ''
os.makedirs(output_dir, exist_ok=True)

with open(label_mappings_path, 'r') as json_file:
    label_mappings = json.load(json_file)

for input_file in input_files:
    df_encoded = pd.read_csv(input_file)
    
    for col, mapping in label_mappings.items():
        reverse_mapping = {v: k for k, v in mapping.items()}
        df_encoded[col] = df_encoded[col].map(reverse_mapping)
    
    file_name = os.path.basename(input_file)
    output_file = os.path.join(output_dir, file_name)
    df_encoded.to_csv(output_file, index=False)
    
    print(f"Decoded DataFrame for {file_name}:")
    print(df_encoded.head())
    print(f"\nDecoded dataset saved to: {output_file}")

