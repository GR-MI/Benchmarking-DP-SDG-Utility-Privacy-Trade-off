import argparse
import pandas as pd
from synthesis.synthesizers.privbayes import PrivBayes

# Argument parser to accept command-line inputs
parser = argparse.ArgumentParser(description="Generate synthetic data using PrivBayes.")
parser.add_argument("--dataset", type=str, required=True, help="Path to the input CSV file.")
parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file.")
parser.add_argument("--epsilon", type=float, required=True, help="Epsilon value for differential privacy.")

args = parser.parse_args()

# Load data
input_file = args.dataset
output_file = args.output
epsilon = args.epsilon

# Load dataset
df = pd.read_csv(input_file)

# Instantiate and fit synthesizer
pb = PrivBayes(epsilon=epsilon)
pb.fit(df)

# Synthesize data
df_synth = pb.sample()

# Save synthetic data
df_synth.to_csv(output_file, index=False)
print(f"Synthetic data saved to {output_file}")
