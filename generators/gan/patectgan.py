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
    'patectgan', 
    epsilon=epsilon,
    verbose=True,
    embedding_dim=128, 
    generator_dim=(256, 256), 
    discriminator_dim=(256, 256), 
    generator_lr=0.0002, 
    generator_decay=1e-06, 
    discriminator_lr=0.0002, 
    discriminator_decay=1e-06, 
    batch_size=500, 
    discriminator_steps=1, 
    epochs=300, 
    pac=1, 
    cuda=True, 
    binary=False, 
    regularization=None, 
    loss='cross_entropy', 
    teacher_iters=5, 
    student_iters=5, 
    sample_per_teacher=1000, 
    delta=None, 
    noise_multiplier=0.001, 
    moments_order=100
)

# Fit and sample synthetic data
synthetic_data = synth.fit_sample(input_data, preprocessor_eps=0.5)

# Save the synthetic data to a CSV file
synthetic_data.to_csv(output_file, index=False)
