import json
import pandas as pd

# Load the dataset
df = pd.read_csv('')

# Define a function to create the domain metadata
def create_domain_file(df, file_path):
    domain_metadata = {
        col: df[col].nunique() for col in df.columns
    }

    # Save the domain file as JSON
    with open(file_path, 'w') as f:
        json.dump(domain_metadata, f, indent=4)

# Generate the domain file
domain_file_path = ''
create_domain_file(df, domain_file_path)
