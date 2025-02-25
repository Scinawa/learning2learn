import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

import numpy as np
import ast
import matplotlib.pyplot as plt
import tempfile
from PIL import Image

from nn import CustomArch
from plots import plot_sine_approximation, plot_all_weights, plot_firing_value
import utils

import pdb


class SinApproximator(LightningModule):
    def count_nonlinear_layers(self):
        return len(
            [layer for layer in self.model.layers if not isinstance(layer, nn.Linear)]
        )

    def _activation_hook(self, layer_name):
        """
        Creates a hook function that stores the mean (over the batch) of the activations.
        This is performed only the first batch of the training.
        """

        def hook(module, input, output):
            if self._is_validation_step == 1:
                if self._first_validation_done_in_epoch == 0:
                    if not isinstance(module, nn.Linear):
                        if self.storing_of_activation <= self.count_nonlinear_layers():
                            try:
                                with torch.no_grad():
                                    # print(
                                    #     f"Hook called for layer: {layer_name}"
                                    # )  # Debug call
                                    # print(f"Module type: {type(module)}")
                                    # print(f"Output shape: {output.shape}")
                                    # # pdb.set_trace()

                                    averages_at_layer = (
                                        output.mean(dim=0)
                                        .detach()
                                        .cpu()
                                        .clone()
                                        .numpy()
                                    )

                                    self.activations_at_layer[layer_name].append(
                                        averages_at_layer
                                    )
                                    self.storing_of_activation += 1
                            except Exception as e:
                                # print(e)
                                pass

        return hook

    def __init__(
        self,
        **args,
    ):

        # Fix the super() initialization
        super().__init__()

        # Move save_hyperparameters after super init
        self.save_hyperparameters()

        # Initialize model
        if type(ast.literal_eval(self.hparams.nn_architecture)) == list:
            self.model = CustomArch(
                nn_architecture=ast.literal_eval(self.hparams.nn_architecture),
                non_linearity=self.hparams.non_linearity,
            )
        else:
            raise ValueError(
                "Invalid architecture specified. Specify this as '[int, int, ...]'"
            )

        # MSELoss vs HuberLoss vs L1loss
        # JI note
        self.criterion = getattr(nn, self.hparams.loss_name)()

        # Store average firing value
        self.activations_at_layer = {}
        self.storing_of_activation = 0
        self._is_validation_step = 0
        self._first_validation_done_in_epoch = 0
        for i, layer in enumerate(self.model.layers):
            if not isinstance(layer, nn.Linear):
                self.activations_at_layer[f"layers.{i}"] = []
        # Register the hook for the activations
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):  # Only register on non-linear layers
                module.register_forward_hook(self._activation_hook(name))

        # Store weights at layer for each layer at each iteration
        self.weights_at_layer = {}
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, nn.Linear):
                self.weights_at_layer[f"layers.{i}"] = []

        # Define a linspace for plotting
        x_plot = torch.linspace(-2 * np.pi, 2 * np.pi, 2000).T.unsqueeze(1)
        self.x_plot = torch.hstack([x_plot, torch.zeros_like(x_plot)])

    def forward(self, x):
        return self.model(x)  # self(X)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams["learning_rate"]
        )

        # If scheduler=FIXED, keep the LR specified in the config file
        if self.hparams.scheduler == "FIXED":
            return {"optimizer": optimizer}

        if self.hparams.scheduler == "ReduceLROnPlateau":
            # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10, verbose=True
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        if self.hparams.scheduler == "StepLR":
            # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=500, gamma=0.96
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        if self.hparams.scheduler == "CosineAnnealingLR":
            # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        raise ValueError("Invalid scheduler specified")

    def _compute_loss(self, batch):
        """
        Compute the loss for a batch of data.
        """
        X, y = batch
        outputs = self(X)
        plain_loss = self.criterion(outputs, y)

        if self.hparams.order == 0:
            reg_loss = 0
        # DEPRECATED
        # elif self.hparams.order == 1:
        #     reg_loss = self.hparams.lambda_reg * torch.sum(
        #         torch.abs(self.model.layers[0].weight[:, 1])
        #     )
        # elif self.hparams.order == 2:
        #     reg_loss = self.hparams.lambda_reg * torch.sum(
        #         self.model.layers[0].weight[:, 1] ** 2
        #     )

        total_loss = plain_loss + reg_loss
        return total_loss, {"plain_train_loss": plain_loss, "reg_loss": reg_loss}

    def training_step(self, batch, batch_idx):
        total_loss, aux = self._compute_loss(batch)

        # log the training loss
        self.log(
            "train_loss",
            total_loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )

        # Log the aux dict
        self.log_dict(aux, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):

        self._is_validation_step = 1
        X, y = batch
        outputs = self(X)
        loss = self.criterion(outputs, y)
        self._is_validation_step = 0
        self._first_validation_done_in_epoch = 1

        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)

        # Plot model performances
        if (
            self.current_epoch % self.hparams.how_often_to_plot == 0
            and self.trainer is not None
        ):
            self._dump_plot_of_performances()

        return loss

    def on_validation_end(self):
        self._first_validation_done_in_epoch = 0

    def on_train_epoch_end(self):
        # For every layer in the model,  store the INPUT weights

        # Memo layer.weight[:, 1] is the right way to access the input weight
        # of the second feature if we are talking about the first layer
        # however we take all the weights for our plot
        with torch.no_grad():
            for i, layer in enumerate(self.model.layers):
                try:
                    self.weights_at_layer[f"layers.{i}"].append(
                        layer.weight.detach().cpu().clone()
                    )
                except Exception as e:
                    pass

        # Now signal that we want to store the value of the firing of the neurons
        # at the next iteration
        self.storing_of_activation = 0

    def _dump_plot_of_performances(self):
        with torch.no_grad():
            # Access the train/val datasets correctly using indices
            train_data = self.trainer.datamodule.train_dataset
            val_data = self.trainer.datamodule.val_dataset

            # Get the original dataset tensors
            full_dataset = self.trainer.datamodule.dataset

            # Use indices to get the correct split data
            X_train = full_dataset.tensors[0][train_data.indices]
            y_train = full_dataset.tensors[1][train_data.indices]
            X_val = full_dataset.tensors[0][val_data.indices]
            y_val = full_dataset.tensors[1][val_data.indices]

            fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
            ax = plot_sine_approximation(
                axes=ax,
                x_values=self.x_plot.cpu().numpy(),
                # fitted model on linspace of values
                fitted_model_on_x_values=self(self.x_plot.to(self.device))
                .detach()
                .cpu()
                .numpy(),
                # Use the correctly split data
                X_train=X_train,
                y_train=y_train,
                # model on training data
                model_on_training_data=self.model(X_train.to(self.device))
                .detach()
                .cpu()
                .numpy(),
                # validation data
                X_val=X_val,
                y_val=y_val,
                # model on validation data
                model_on_validation_data=self.model(X_val.to(self.device))
                .detach()
                .cpu()
                .numpy(),
                # title is the arch structure
                plot_title=self.hparams.nn_architecture,
            )
            # pdb.set_trace()

            utils.save_and_log_plot(
                fig, self.logger.run_id, self.current_epoch, self.logger
            )

    def on_train_end(self):
        plot_all_weights(self, self.weights_at_layer)
        plot_firing_value(self, self.activations_at_layer)
