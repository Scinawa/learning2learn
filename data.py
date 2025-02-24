import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from lightning.pytorch import LightningDataModule


class NoisySinDataModule(LightningDataModule):
    def __init__(
        self,
        interval=2,
        dataset_size=1000,
        validation_ratio=0.2,
        batch_size=32,
        noise_of_noisy_feature=0.0,
        how="linspace",
    ):
        super().__init__()
        self.dataset_size = dataset_size
        self.validation_ratio = validation_ratio
        self.batch_size = batch_size
        self.how = how
        self.interval = interval
        self.noise_of_noisy_feature = noise_of_noisy_feature

    def prepare_data(self):
        # Generate x according to the chosen method
        if self.how == "linspace":
            x = np.linspace(
                -self.interval * np.pi, self.interval * np.pi, self.dataset_size
            )
        elif self.how == "random":
            x = np.random.uniform(
                -self.interval * np.pi, self.interval * np.pi, self.dataset_size
            )
        else:
            raise ValueError("Unknown method: choose 'linspace' or 'random'")

        # Generate noisy feature and targets
        noise_x = 7 * np.random.rand(self.dataset_size)
        y = np.sin(x) + np.random.normal(
            loc=0.0, scale=self.noise_of_noisy_feature, size=x.shape
        )

        # Stack features: x and noise_x
        X = np.vstack([x, noise_x]).T

        # Convert to torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def setup(self, stage=None):
        # Create TensorDataset
        self.dataset = TensorDataset(self.X, self.y)

        # Calculate split sizes
        validation_set_size = int(self.dataset_size * self.validation_ratio)
        training_set_size = self.dataset_size - validation_set_size

        # Split dataset into training and validation sets
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [training_set_size, validation_set_size]
        )

        # Debugging
        # train_indices = self.train_dataset.indices
        # val_indices = self.val_dataset.indices
        # print(f"First 5 training indices: {train_indices[:5]}")
        # print(f"First 5 validation indices: {val_indices[:5]}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
