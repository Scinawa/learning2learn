import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from PIL import Image
from lightning.pytorch import LightningModule
from nn import CustomArch  # Import the neural network from nn.py

from plots import plot_sine_approximation
import ast
import pdb


class SinApproximator(LightningModule):
    def __init__(
        self,
        nn_architecture,
        non_linearity,
        learning_rate=0.01,
        lambda_reg=0.5,
        order=0,
    ):
        # Fix the super() initialization
        super().__init__()

        # Move save_hyperparameters after super init
        self.save_hyperparameters()

        # Initialize model and criterion
        if nn_architecture == "SimpleNN":
            self.model = CustomArch(
                non_linearity=non_linearity, nn_architecture=[2, 3, 5, 2, 1]
            )

        elif type(ast.literal_eval(nn_architecture)) == list:
            self.model = CustomArch(
                nn_architecture=ast.literal_eval(nn_architecture),
                non_linearity=non_linearity,
            )
        else:
            raise ValueError(
                "Invalid architecture specified. Use 'SimpleNN' or '[list]'"
            )

        self.criterion = nn.MSELoss()

        # self.avg_activations = {}

        # Register hooks for each layer (including submodules)
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):  # Only register on non-linear layers
                module.register_forward_hook(self._activation_hook(name))

    def _activation_hook(self, layer_name):
        """Creates a hook function that stores activations."""

        def hook(module, input, output):

            # avg_activations = {}
            # pdb.set_trace()
            if self.training:
                #     # Store activations as a list of tensors
                #     # if layer_name not in self.activations:
                #     # avg_activations[layer_name] = []
                #     avg_activations[layer_name].append(output.mean(dim=0).detach().clone())
                #     self.log_dict(
                #         avg_activations, prog_bar=False, logger=True, on_epoch=True
                #     )
                # # avg_activations.clear()
                self.log_dict(
                    {
                        f"{layer_name}_{i}": value
                        for i, value in enumerate(output.mean(dim=0).detach().clone())
                    },
                    prog_bar=False,
                    # on_epoch=True,
                )

        return hook

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
            reg_loss = self.hparams.lambda_reg * torch.sum(
                torch.abs(self.model.layers[0].weight[:, 1])
            )
        elif self.hparams.order == 2:
            reg_loss = self.hparams.lambda_reg * torch.sum(
                self.model.layers[0].weight[:, 1] ** 2
            )

        total_loss = plain_loss + reg_loss
        return total_loss, {"plain_train_loss": plain_loss, "reg_loss": reg_loss}

    def training_step(self, batch, batch_idx):
        total_loss, aux = self._compute_loss(batch)

        # # Compute average activation across steps
        # avg_activations = {}
        # for module, values in self.activations.items():
        #     # import pdb

        #     # pdb.set_trace()
        #     avg_activations[f"activation_{module}"] = torch.stack(values).mean().item()

        # self.log(
        #     avg_activations,
        #     prog_bar=False,
        #     logger=True,
        #     on_epoch=True,
        #     on_step=True,
        #     # reduce_fx=torch.mean,
        # )

        # # Clear activations for the next batch
        # self.activations.clear()

        self.log(
            "train_loss",
            total_loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        # Optional: log the aux dict
        self.log_dict(
            aux,
            prog_bar=False,
            logger=True,
            on_epoch=True,
        )

        # self.log_dict(
        #     self.activations,
        #     prog_bar=False,
        #     logger=True,
        #     on_epoch=True,
        # )
        return total_loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = self.criterion(outputs, y)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

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
        fitted_model = self(x_plot.to(self.device)).detach().cpu().numpy()

        # Retrieve training tensors and move them to the current device
        X_train_cpu = train_data.dataset.tensors[0]
        y_train_cpu = train_data.dataset.tensors[1]
        X_train = X_train_cpu.to(self.device)
        # Optionally, y_train = y_train_cpu.to(self.device) only if you need forward pass with labels

        # Retrieve validation tensors and move them to the current device
        X_val_cpu = val_data.dataset.tensors[0]
        X_val = X_val_cpu.to(self.device)
        y_val = val_data.dataset.tensors[1]

        # Generate predictions on training/validation data
        y_train_pred = self(X_train).detach().cpu().numpy()

        fig, ax = plt.subplots()
        ax = plot_sine_approximation(
            ax,
            x_plot.cpu().numpy(),  # X-values evaluating the model
            fitted_model,  # Fitted model evaluated on the linspace
            X_train_cpu,  # CPU copy for plotting axis
            y_train_cpu,
            y_train_pred,  # CPU copy of the model output
            X_val_cpu,
            y_val,
            self.hparams.nn_architecture,
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
                    run_id=self.logger.run_id,
                )
            except AttributeError:
                print("Could not log the image file as an artifact.")
        return None
