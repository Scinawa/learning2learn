import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import yaml
from datetime import datetime

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from data import NoisySinDataModule
from model import SinApproximator
from utils import return_parser
import argparse

import pdb


def set_seed(seed):
    random.seed(seed)  # 1
    np.random.seed(seed)  # 2
    torch.manual_seed(seed)  # 3
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 4


# OK Moved to a proper python project instead of a notebook
# OK Found a way to fix the randomness
# OK Yaml file for configuration of paramters, with argparse for command line arguments
# OK Added logging and plotting with MLFlow
# OK Schedulers for learning rate
# OK Saving models in /checkpoints for lookback


def main():

    parser = return_parser()

    with open("config.yaml", "r") as file:
        final_args = yaml.safe_load(file)
        cmd_args = parser.parse_args()

        update_dict = {k: v for k, v in vars(cmd_args).items() if v is not None}
        run_instance = "_".join(f"{k}-{v}" for k, v in update_dict.items())
        final_args.update(update_dict)

    # pdb.set_trace()

    set_seed(final_args["seed"])

    datamodule = NoisySinDataModule(
        interval=final_args["interval"],
        dataset_size=final_args["dataset_size"],
        validation_ratio=final_args["validation_ratio"],
        batch_size=final_args["batch_size"],
        noise_of_noisy_feature=final_args["noise_of_noisy_feature"],
        how=final_args["how"],
    )

    model = SinApproximator(
        learning_rate=final_args["learning_rate"],
        lambda_reg=final_args["lambda_reg"],
        non_linearity=final_args["non_linearity"],
        order=final_args["regularization_order"],
        nn_architecture=final_args["nn_architecture"],
        scheduler=final_args["scheduler"],
        loss_name=final_args["loss_name"],
        how_often_to_plot=final_args["how_often_to_plot"],
    )
    print("Model created")

    mlflow_logger = MLFlowLogger(
        experiment_name=final_args["experiment_name"], run_name=run_instance
    )
    run_id = mlflow_logger.run_id

    print(f"---Run ID: {run_id}")

    filename = f"run_id-{run_id}"

    # LOGGING WITH MLFLOW
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=filename + "-{epoch:03d}",  # Add epoch number to filename
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=5,  # Save every 5 epochs
        save_last=True,  # Save the last checkpoint
    )

    hparams = final_args
    mlflow_logger.log_hyperparams(hparams)

    trainer = Trainer(
        accelerator="auto",  # let Lightning decide (CPU, GPU, or MPS)
        max_epochs=final_args["max_epochs"],
        logger=mlflow_logger,
        log_every_n_steps=0,
        limit_val_batches=3,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
