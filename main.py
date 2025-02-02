import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from nn import SimpleNN
import yaml
from datetime import datetime

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from data import NoisySinDataModule  
from model import SinApproximator


def set_seed(seed):
    random.seed(seed)           # 1 
    np.random.seed(seed)        # 2
    torch.manual_seed(seed)     # 3
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # 4


set_seed(44)






def main():


    with open("config.yaml", "r") as file:
        args = yaml.safe_load(file)

    # Remove any manual device code
    # device = torch.device("cpu")

    datamodule = NoisySinDataModule(
        interval=args["interval"],
        dataset_size=args["dataset_size"],
        validation_ratio=args["validation_ratio"],
        batch_size=args["batch_size"],
        noise_of_noisy_feature=args["noise_of_noisy_feature"],
        how=args["how"],
    )

    model = SinApproximator(
        nodes=args["nodes"],
        nn_architecture=args["nn_architecture"],
    )

    print("Model created")

    # optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    # Define the checkpoint callback

    current_date_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    filename = f"best-checkpoint-{current_date_time}-nodes{args['nodes']}-lr{args['learning_rate']}-lambda{args['lambda_reg']}-order{args['regularization_order']}-arch{args['nn_architecture']}"

    # LOGGING WITH MLFLOW
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=filename,
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=5,  # Save every 5 epochs
    )

    mlflow_logger = MLFlowLogger(experiment_name=args["experiment_name"])
    run_id = mlflow_logger.run_id
    print(f"---Run ID: {run_id}")

    hparams = args
    mlflow_logger.log_hyperparams(hparams)

    trainer = Trainer(
        accelerator="auto",   # let Lightning decide (CPU, GPU, or MPS)
        max_epochs=args["max_epochs"],
        logger=mlflow_logger,
        log_every_n_steps=1,
        limit_val_batches=3,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model=model, datamodule=datamodule)
    

if __name__ == "__main__":
    main()