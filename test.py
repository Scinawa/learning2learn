import torch
from plots import plot_sine_approximation
from model import SinApproximator
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import set_seed
from data import NoisySinDataModule
from plots import plot_testing

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Test the model with a specified checkpoint."
)
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
)
parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
parser.add_argument(
    "--dataset_size",
    type=int,
    default=100,
    help="Number of samples in the dataset",
)
parser.add_argument("--noise_of_noisy_feature", default=0.1, type=float)
parser.add_argument("--how", default="linspace", type=str)

args = parser.parse_args()

# Load the checkpoint
checkpoint_path = args.checkpoint
checkpoint = torch.load(checkpoint_path)

# Assume the model class is defined in model.py and the model state dict is saved in the checkpoint
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SinApproximator.load_from_checkpoint(checkpoint_path)
model = model.to(device)
model.eval()


set_seed(args.seed)

datamodule = NoisySinDataModule(
    interval=2,
    dataset_size=args.dataset_size,
    validation_ratio=0,
    batch_size=1,
    noise_of_noisy_feature=args.noise_of_noisy_feature,
    how=args.how,
)
datamodule.prepare_data()

datamodule.setup()


# ...existing code...

# Test the model on the random dataset
with torch.no_grad():
    # Get the dataset from the dataloader
    train_dataloader = datamodule.train_dataloader()

    # Get all data at once since we're testing
    all_data = []
    all_targets = []
    for batch in train_dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)  # Move inputs to correct device
        targets = targets.to(device)  # Move targets to correct device
        all_data.append(inputs)
        all_targets.append(targets)

    # Concatenate all batches
    inputs = torch.cat(all_data)
    targets = torch.cat(all_targets)

    # Generate predictions
    predictions = model(inputs).cpu()  # Move predictions back to CPU for plotting

    # Create evaluation points for smooth curve
    x_plot = torch.linspace(-2 * np.pi, 2 * np.pi, 2000).T.unsqueeze(1).to(device)
    x_plot = torch.hstack([x_plot, torch.zeros_like(x_plot)])
    smooth_predictions = model(x_plot).cpu()

    # Plot results
    plot_testing(
        inputs=inputs.cpu(),
        targets=targets.cpu(),
        predictions=predictions,
        x_plot=x_plot.cpu(),
        smooth_predictions=smooth_predictions,
    )
