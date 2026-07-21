"""Script for training and evaluating the anomalib DRAEM model
with the torchsig-models dataset adapter infrastructure.

DRAEM is a discriminative resconstruction type model that uses a discriminative
sub-network to compare the input image with its reconstruction.

Usage:
    python3 train_draem.py 
        --run_name <name> --output_dir <dir> --train_root <root> --eval_root <root>

"""
import argparse
import json
import os
from pathlib import Path
from torch import set_float32_matmul_precision
from lightning.pytorch.callbacks import EarlyStopping

from anomalib.metrics import AUROC, AUPR, F1Score, F1Max, Evaluator
from anomalib.callbacks.checkpoint import ModelCheckpoint
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.engine import Engine
from anomalib.models import Draem

from torchsig_models.adapters.anomalib_utils import TorchSigAnomalibDataModule


set_float32_matmul_precision("high")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True, type=str, help="Name to label model training run.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save outputs.")
    parser.add_argument("--train_root", required=True, type=str, help="Root for TorchSig train dataset.")
    parser.add_argument("--eval_root", required=True, type=str, help="Root for TorchSig evaluation dataset.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loading. Default is 64.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for data loading. Default is 16.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train. Default is 10.")
    args = parser.parse_args()

    run_name = args.run_name
    output_dir = args.output_dir
    os.makedirs(output_dir + "/results", exist_ok=True)
    os.makedirs(output_dir + "/checkpoints", exist_ok=True)

    # config parameters
    cfg = {
        "model": {
            "name": "DRAEM", 
            "enable_sspcab": False, 
            "sspcab_lambda": 0.1, 
            "anomaly_source_path": None, 
            "beta": (0.1, 1.0)
        },
        "data": {"dataset": "torchsig_anomalies"},
        "engine": {"num_epochs": args.num_epochs},
    }

    # Setup
    # data loading 
    train_dm = TorchSigAnomalibDataModule(
        root               = args.train_root,
        train_batch_size   = args.batch_size,
        eval_batch_size    = args.batch_size,
        num_workers        = args.num_workers,
        seed               = 42,
        splits             = (0.70, 0.20, 0.10),
        size               = None,
        use_anomaly_labels = False,    # use/ignore anomaly labels
        augmentations      = None,
        transform          = None,
        torchsig_transforms= [],
        target_labels      = ["class_name", "start", "stop", "lower_freq", "upper_freq", "anomaly"],
    )
    train_dm.setup()

    # TensorBoard logger
    logger = AnomalibTensorBoardLogger(
        save_dir=output_dir + "/logs",
        name=run_name,
        version=1.0,
    )  

    # metrics
    metrics = [
        AUROC(fields=["pred_score", "gt_label"]),
        AUPR(fields=["pred_score", "gt_label"]),
        F1Score(fields=["pred_label", "gt_label"]),
        F1Max(fields=["pred_score", "gt_label"])
    ]
    evaluator = Evaluator(test_metrics = metrics)

    # model: DRAEM (see documentations for parameters)
    model = Draem(
        enable_sspcab = cfg["model"]["enable_sspcab"], 
        sspcab_lambda = cfg["model"]["sspcab_lambda"],
        #anomaly_source_path = cfg["model"]["anomaly_source_path"],
        beta = cfg["model"]["beta"],
        pre_processor = True, # default
        post_processor = True, # default
        evaluator = evaluator,
        visualizer = False
    )

    # Anomalib engine
    model_checkpoint = ModelCheckpoint(
        dirpath=args.output_dir + "/checkpoints",
        filename="draem-best",
        save_top_k=1,
        save_last=False,
        monitor="train_loss",
        mode='min'
    )
    early_stopping = EarlyStopping(monitor="train_loss", patience=5)
    callbacks = [
        model_checkpoint,
        early_stopping,
    ]
    kwargs = {"log_every_n_steps": 1}
    
    engine = Engine(
        default_root_dir = output_dir + "/results",
        max_epochs = cfg["engine"]["num_epochs"],
        callbacks = callbacks,
        logger = logger,
        **kwargs
    )

    # Training
    engine.train(model=model, datamodule=train_dm)

    # Evaluation

    # Setup dataset
    eval_dm = TorchSigAnomalibDataModule(
        root               = args.eval_root,
        train_batch_size   = args.batch_size,
        eval_batch_size    = args.batch_size,
        num_workers        = args.num_workers,
        seed               = 42,
        splits             = (0.70, 0.20, 0.10),
        size               = None,
        use_anomaly_labels = True,    # use/ignore anomaly labels
        augmentations      = None,
        transform          = None,
        torchsig_transforms= [],
        target_labels      = ["class_name", "start", "stop", "lower_freq", "upper_freq", "anomaly"],
    )
    eval_dm.setup()


    # Test the best trained model on the test set
    test_results = engine.test(
        model = model, 
        datamodule = eval_dm,
        ckpt_path = engine.trainer.checkpoint_callback.best_model_path,
    )
    with open(Path(output_dir) / "results" / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=4)
    print(test_results)


if __name__ == "__main__":
    main()
