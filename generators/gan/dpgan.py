from snsynth import Synthesizer
import pandas as pd
import argparse

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

# Load the data into a DataFrame
input_data = pd.read_csv(input_file)

# Create a synthesizer and generate synthetic data
synth = Synthesizer.create(
    'dpgan',
    epsilon=epsilon,
    binary=False, 
    latent_dim=64, 
    batch_size=64, 
    epochs=1000, 
    delta=None
)

# Fit and sample synthetic data
synthetic_data = synth.fit_sample(input_data, preprocessor_eps=0.5)

# Save the synthetic data to a CSV file
synthetic_data.to_csv(output_file, index=False)
