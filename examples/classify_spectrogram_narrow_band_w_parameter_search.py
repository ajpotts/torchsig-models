#!/usr/bin/env python
# coding: utf-8

# # TorchSig → EfficientNet: End‑to‑End Spectrogram Classification Pipeline

# Created: 2026-05-07
# 
# This notebook demonstrates a full TorchSig → EfficientNet training pipeline, with optional TensorBoard logging and MLflow tracking.

# ### Imports

# In[ ]:


# --------------------------------------------------------------
# Standard‑library imports
# --------------------------------------------------------------
import copy
import json
import os
import pathlib
from pathlib import Path

# --------------------------------------------------------------
# Third‑party scientific / plotting / ML libraries
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import random
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from tqdm import tqdm
from dotenv import load_dotenv

# --------------------------------------------------------------
# PyTorch utilities
# --------------------------------------------------------------
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # optional, for TensorBoard

# --------------------------------------------------------------
# TorchSig imports
# --------------------------------------------------------------
from torchsig.datasets.datamodules import TorchSigDataModule
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.models.model_utils.training_tools import (
    collate_one_channel,
    compute_class_weights_tensor,
    compute_num_params,
    micro_metrics,
    plot_history,
    run_test_and_confusion,
    set_deterministic,
    simple_train_validate,
)
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer
from torchsig.utils.yaml import load_config_from_yaml, write_dict_to_yaml

import optuna
import mlflow
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader


# In[ ]:


# Load .env if it exists (no‑op if you already set the vars in your shell)
load_dotenv(Path(".env"))

# Verify that MLflow sees the right values
print("TRACKING_URI :", os.getenv("MLFLOW_TRACKING_URI"))
print("CA bundle    :", os.getenv("REQUESTS_CA_BUNDLE"))


# ### Load configuration + parameters

# In[ ]:


CONFIG_PATH = Path("./configs") / "spectrogram_single_label_config.yml"

config = yaml.safe_load(CONFIG_PATH.read_text())
params = config["params"]

config["num_classes"] = len(config["classes"])

config


# In[ ]:


params["max_epochs"]


# In[ ]:


#torchsig_config = load_config_from_yaml(CONFIG_PATH)
#torchsig_config


# ### Set seed for reproducibility

# In[ ]:


SEED = config["seed"] or 123
set_deterministic(SEED)

print(f"Global seed set to {SEED}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")


# ### Setup directories for checkpointing and data generation

# In[ ]:


# -----------------------------------------------------------------
#  Build the *root* directory where the HDF5 dataset will live
# -----------------------------------------------------------------
ROOT = Path("../datasets") / (
    f"{config['dataset_id']}_"
    f"{params['batch_size']}_"
    f"{config['impairment_level']}_"
    f"{config['dataset_length']}"
)
ROOT.mkdir(parents=True, exist_ok=True)   # make sure the folder exists
config["root"] = str(ROOT)                # store as a plain string for TorchSig

ROOT


# In[ ]:


conf_dir = Path("../datasets") / (f"{params["experiment_name"]}")
conf_dir.mkdir(parents=True, exist_ok=True)
write_dict_to_yaml(conf_dir/"conf.yml", config)

conf_dir


# In[ ]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#   For TensorBoard checkpointing
CHECKPOINT_DIR  = conf_dir/"checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ### Prepare the data

# In[ ]:


data_mod = TorchSigDataModule(
    root= config["root"],
    metadata = config["dataset_metadata"],
    dataset_size = config["dataset_length"],
    dataset_splits = config["dataset_splits"],
    # dataloader params
    batch_size = params["batch_size"],
    num_workers = config["num_workers"],
    collate_fn = collate_one_channel,
    # dataset creator params
    create_batch_size = params["batch_size"],
    create_num_workers = 4,
    file_writer = HDF5Writer,
    file_reader = HDF5Reader,
    overwrite = config["overwrite"],
    # transforms
    impairment_level = config["impairment_level"],
    transforms=[Spectrogram(fft_size=config["dataset_metadata"]["fft_size"])],
    target_labels=["class_index"],
    signal_generators = config["classes"],
    seed = 123,
)


# In[ ]:


if not Path(config["root"]).exists() or config["overwrite"]:
    data_mod.prepare_data()
else:
    print("Dataset already exists – skipping generation.")


# In[ ]:


data_mod.setup(stage="fit")

print("Train/Val/Test split sizes:", len(data_mod.train), len(data_mod.val), len(data_mod.test))


# In[ ]:


# -----------------------------------------------------------------
# Load the static datasets
# -----------------------------------------------------------------
train_dataset = data_mod.train
val_dataset = data_mod.val

print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(val_dataset))


# In[ ]:


train_loader = data_mod.train_dataloader()
val_loader = data_mod.val_dataloader()
test_loader = data_mod.test_dataloader()


# ### Setup the Model

# In[ ]:


# --------------------------------------------------------------
# Optional TensorBoard writer
# --------------------------------------------------------------
USE_TENSORBOARD = True
if USE_TENSORBOARD:
    tb_writer = SummaryWriter(log_dir="./runs/torchsig_experiment")
else:
    tb_writer = None


# In[ ]:


class_weights_tensor = compute_class_weights_tensor(train_loader, config["num_classes"]).to(DEVICE)
class_weights_tensor


# In[ ]:


# -------------------------------------------------
# Model & loss
# -------------------------------------------------

model = timm.create_model(params["model_name"], in_chans=params["num_channels"],
                          pretrained=params["pretrained"], num_classes=config["num_classes"], drop_path_rate=params["drop_path"],
        drop_rate=params["drop_rate"]).to(DEVICE)

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=params["dropout"]),
    torch.nn.Linear(model.classifier.in_features, config["num_classes"]),
).to(DEVICE)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=params["label_smoothing"])
optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"],
                              weight_decay=params["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["max_epochs"])

# Warm‑up + cosine schedule
scheduler_warm = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=5)
scheduler_cos  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=params["max_epochs"]-5)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[scheduler_warm, scheduler_cos],
    milestones=[5],
)


# In[ ]:


num_params = compute_num_params(model)
params["num_params"] = num_params
print(f"Trainable parameters: {num_params:,}")


# ### Model Training

# In[ ]:


num_channels = params["num_channels"]

def train_and_validate(params, train_loader, val_loader, num_classes, class_weights_tensor, device):

    model = timm.create_model(params["model_name"], in_chans=num_channels,
                          pretrained=params["pretrained"], num_classes=config["num_classes"], drop_path_rate=params["drop_path"],
        drop_rate=params["drop_rate"]).to(DEVICE)

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=params["dropout"]),
        torch.nn.Linear(model.classifier.in_features, config["num_classes"]),
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=params["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"],
                                weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["max_epochs"])

    # Warm‑up + cosine schedule
    scheduler_warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=5)
    scheduler_cos  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params["max_epochs"]-5)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warm, scheduler_cos],
        milestones=[5],
    )

    best_val_acc = 0.0

    for epoch in range(params['max_epochs']):
        # --- TRAIN ---
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # --- VALIDATE ---
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_preds.append(logits.argmax(dim=1).cpu())
                val_targets.append(y.cpu())

        all_preds = torch.cat(val_preds)
        all_targets = torch.cat(val_targets)
        accuracy = (all_preds == all_targets).float().mean().item()

        if accuracy > best_val_acc:
            best_val_acc = accuracy

    return best_val_acc


# In[ ]:


def objective(trial):
    params = {
        "model_name": trial.suggest_categorical("model_name", ["efficientnet_b0", "mobilenetv2_100"]),
        "pretrained": True,
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "dropout": trial.suggest_float("dropout", 0.2, 0.5),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
        "max_epochs": 15, # Adjusted for speed during search
    }

    with mlflow.start_run(nested=True):
        merged = {**config, **params} # combine to record both
        mlflow.log_params(merged)
        mlflow.log_param("classes", config["classes"])

        # IMPORTANT: Pass all necessary objects explicitly!
        try:
            val_acc = train_and_validate(
                params, 
                train_loader, 
                val_loader, 
                num_classes, 
                class_weights_tensor, 
                DEVICE
            )
            mlflow.log_metric("val_acc", val_acc)
            return val_acc
        except Exception as e:
            print(f"Trial failed due to: {e}")
            return 0.0 # Return 0 if the trial crashes (e.g. Out of Memory)


# In[ ]:


# Ensure MLflow is configured
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("torchsig-optimization-study")

# Create a Study
study = optuna.create_study(direction="maximize")

with mlflow.start_run(run_name="Optimization_Parent_Run"):
    # 1. Run the optimization
    study.optimize(objective, n_trials=30) 

    # 2. Extract the best results
    best_value = study.best_trial.value
    best_params = study.best_trial.params
    best_trial_number = study.best_trial.number

    # 3. Log everything to the Parent Run
    # Log the best metric found
    mlflow.log_metric("best_val_acc", best_value)

    # Log the best trial index/name
    mlflow.log_param("best_trial_number", best_trial_number)

    # Log all the best parameters as individual MLflow params
    # This allows you to use the MLflow UI to compare the "Winner" 
    # against other successful runs.
    mlflow.log_params(best_params)

    print("Optimization Complete!")
    print(f"Best Value: {best_value}")
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

