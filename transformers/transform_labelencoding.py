import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
import os

df = pd.read_csv('')

categorical_columns = df.select_dtypes(include=['object']).columns
label_mappings = {}

# Apply Label Encoding to categorical columns
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    
    # Save mapping: Create reverse mappings for the column
    mapping = {original: encoded for encoded, original in enumerate(le.classes_)}
    label_mappings[col] = mapping

processed_data_path = ''
df.to_csv(processed_data_path, index=False)

label_mappings_path = ''
os.makedirs(os.path.dirname(label_mappings_path), exist_ok=True)
with open(label_mappings_path, 'w') as json_file:
    json.dump(label_mappings, json_file, indent=4)

print("Processed DataFrame:")
print(df.head())

print(f"\nProcessed dataset saved to: {processed_data_path}")
print(f"Label mappings saved to: {label_mappings_path}")
