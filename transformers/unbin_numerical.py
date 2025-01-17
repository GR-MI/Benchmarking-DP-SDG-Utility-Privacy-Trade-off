import pandas as pd
import numpy as np
import json
import os
import random

df = pd.read_csv('')

bin_mappings_path = '' 
with open(bin_mappings_path, 'r') as json_file:
    bin_mappings = json.load(json_file)


columns_to_unbin = ['col1', 'col2'] 
for col in columns_to_unbin:
    bin_edges = bin_mappings[col]['bin_edges']
    bin_labels = bin_mappings[col]['bin_labels']
    
    def unbin_value(binned_label):
        bin_index = bin_labels.index(binned_label)
        lower_bound = bin_edges[bin_index]
        upper_bound = bin_edges[bin_index + 1]
        return random.uniform(lower_bound, upper_bound)
    
    df[col + '_unbinned'] = df[col].apply(lambda x: unbin_value(x))

processed_data_path = ''  
df.to_csv(processed_data_path, index=False)
print("Processed DataFrame with Unbinned Columns:")
print(df.head())
print(f"\nProcessed dataset saved to: {processed_data_path}")
