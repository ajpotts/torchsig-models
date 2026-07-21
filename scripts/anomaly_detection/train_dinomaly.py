"""Script for training and evaluating the anomalib Dinomaly model
with the torchsig-models dataset adapter infrastructure.

Pipeline: Dinomaly model with modified post-processor.

Dinomaly is a ViT Reconstruction type model that employs a DINOv2 ViT encoder and decoder, 
using random dropout in the bottleneck to mimic denoising, achieving high-performance 
reconstruction.

Usage:
    python3 train_dinomaly.py 
        --run_name <name> --output_dir <dir> --train_root <root> --eval_root <root>

"""
import argparse
import json
import os
from pathlib import Path
from torch import set_float32_matmul_precision
from torchvision.transforms.v2 import Compose, Resize, Normalize
from lightning.pytorch.callbacks import EarlyStopping

from anomalib.metrics import AUROC, AUPR, F1Score, F1Max, Evaluator
from anomalib.callbacks.checkpoint import ModelCheckpoint
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.pre_processing import PreProcessor
from anomalib.engine import Engine
from anomalib.models import Dinomaly

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
    parser.add_argument("--max_epochs", type=int, default=20, help="Number of steps to train. Default is 20.")
    args = parser.parse_args()

    run_name = args.run_name
    output_dir = args.output_dir
    os.makedirs(output_dir + "/results", exist_ok=True)
    os.makedirs(output_dir + "/checkpoints", exist_ok=True)

    # config parameters
    cfg = {
        "model": {
                "name": "Dinomaly", 
                "encoder_name": "dinov2reg_vit_base_14", 
                "bottleneck_dropout": 0.2, 
                "decoder_depth": 8, 
                "target_layers": [2, 3, 4, 5, 6, 7, 8, 9], 
                "fuse_layer_encoder": [[0, 1, 2, 3], [4, 5, 6, 7]],
                "fuse_layer_decoder": [[0, 1, 2, 3], [4, 5, 6, 7]],
                "remove_class_token": False,
        },
        "data": {"dataset": "torchsig_anomalies"},
        "engine": {
            "max_epochs": args.max_epochs,
        },
    }

    # Setup
    # data loading 
    train_dm = TorchSigAnomalibDataModule(
        root               = args.train_root,
        train_batch_size   = args.batch_size,
        eval_batch_size    = args.batch_size,
        num_workers        = args.num_workers,
        seed               = 42,
        splits             = (0.8, 0.15, 0.05),
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
        #F1Score(fields=["pred_label", "gt_label"]), # only use with appropriate post-processor
        F1Max(fields=["pred_score", "gt_label"]),
    ]
    evaluator = Evaluator(test_metrics = metrics)

    # preserve spectrogram square aspect ratio and conform to DINO multiple-of-14 dimensions
    pre_processor = PreProcessor(
        transform=Compose([
            # 392 x 392  = square baseline, no crop
            Resize((392, 392), antialias=True),
            # ImageNet standard channel-wise normalization
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )

    # model: Dinomaly (see documentation for parameters)
    model = Dinomaly(
        encoder_name = cfg["model"]["encoder_name"],
        bottleneck_dropout = cfg["model"]["bottleneck_dropout"],
        decoder_depth = cfg["model"]["decoder_depth"],
        target_layers = cfg["model"]["target_layers"],
        fuse_layer_encoder = cfg["model"]["fuse_layer_encoder"],
        fuse_layer_decoder = cfg["model"]["fuse_layer_decoder"],
        remove_class_token = cfg["model"]["remove_class_token"],       
        pre_processor = pre_processor,
        post_processor = False, # raw score outputs for metrics
        evaluator = evaluator,
        visualizer = False
    )

    # Anomalib engine
    model_checkpoint = ModelCheckpoint(
        dirpath=args.output_dir + "/checkpoints",
        filename="dinomaly-best",
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
        max_epochs = cfg["engine"]["max_epochs"],
        num_sanity_val_steps=0,         # skip initial validation check
        check_val_every_n_epoch=None,   # skip validation (normal-only training)
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
        splits             = (0.1, 0.1, 0.8),
        size               = None,
        use_anomaly_labels = True,    # use/ignore anomaly labels
        augmentations      = None,
        transform          = None,
        torchsig_transforms= [],
        target_labels      = ["class_name", "start", "stop", "lower_freq", "upper_freq", "anomaly"],
    )
    eval_dm.setup()

    # Test best trained model on the evaluation test set
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
