import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from PIL import Image
from lightning.pytorch import LightningModule
from nn import SimpleNN  # Import the neural network from nn.py

from plots import plot_sine_approximation

class SinApproximator(LightningModule):
    def __init__(self, nodes=10, learning_rate=0.01, lambda_reg=0.5, order=0):
        # Fix the super() initialization
        super().__init__()
        
        # Move save_hyperparameters after super init
        self.save_hyperparameters()
        
        # Initialize model and criterion
        self.model = SimpleNN(nodes=nodes)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # Ensure forward returns the model's output
        return self.model(x)

    def configure_optimizers(self):
        # Use self.parameters() instead of self.model.parameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def _compute_loss(self, batch):
        X, y = batch
        outputs = self(X)
        plain_loss = self.criterion(outputs, y)

        if self.hparams.order == 0:
            reg_loss = 0
        elif self.hparams.order == 1:
            reg_loss = self.hparams.lambda_reg * torch.sum(torch.abs(self.model.fc.weight[:, 1]))
        elif self.hparams.order == 2:
            reg_loss = self.hparams.lambda_reg * torch.sum(self.model.fc.weight[:, 1] ** 2)

        total_loss = plain_loss + reg_loss
        return total_loss, {"plain_train_loss": plain_loss, "reg_loss": reg_loss}

    def training_step(self, batch, batch_idx):
        total_loss, aux = self._compute_loss(batch)
        self.log("train_loss", total_loss, prog_bar=True, logger=True)
        # Optional: log the aux dict
        self.log_dict(aux, prog_bar=False, logger=True)
        return total_loss


    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = self.criterion(outputs, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        # Access the train/val datasets from self.trainer.datamodule
        if self.current_epoch % 10 == 0 and self.trainer is not None:
            train_data = self.trainer.datamodule.train_dataset
            val_data = self.trainer.datamodule.val_dataset
            self._plot_model_performance(train_data, val_data)

        return loss

    def _plot_model_performance(self, train_data, val_data):
        # Generate X for plotting
        x_plot = torch.linspace(-2 * np.pi, 2 * np.pi, 1000).T.unsqueeze(1)
        x_plot = torch.hstack([x_plot, torch.zeros_like(x_plot)])
        y_plot = self(x_plot.to(self.device)).detach().cpu().numpy()

        # Retrieve training tensors and move them to the current device
        X_train_cpu = train_data.dataset.tensors[0]
        y_train_cpu = train_data.dataset.tensors[1]
        X_train = X_train_cpu.to(self.device)
        # Optionally, y_train = y_train_cpu.to(self.device) only if you need forward pass with labels

        # Retrieve validation tensors and move them to the current device
        X_val_cpu = val_data.dataset.tensors[0]
        y_val_cpu = val_data.dataset.tensors[1]
        X_val = X_val_cpu.to(self.device)

        # Generate predictions on training/validation data
        y_train_pred = self(X_train).detach().cpu().numpy()
        y_val_pred   = self(X_val).detach().cpu().numpy()

        fig, ax = plt.subplots()
        ax = plot_sine_approximation(
            ax,
            x_plot.cpu().numpy(),        # X-values for the sin curve
            X_train_cpu,                 # CPU copy for plotting axis
            y_train_cpu,
            y_plot,                      
            y_train_pred,                # CPU copy of the model output
            X_val_cpu,
            y_val_cpu,
            self.hparams.nodes
        )
        # ...existing code for saving and logging plot...
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            plt.savefig(tmpfile.name)
            plt.close(fig)
            try:
                img = Image.open(tmpfile.name)
                self.logger.experiment.log_image(
                    image=img,
                    artifact_file=f"model_output_epoch_{self.current_epoch:04d}.png",
                    run_id=self.logger.run_id
                )
            except AttributeError:
                print("Could not log the image file as an artifact.")
        return None
