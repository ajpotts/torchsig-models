"""Script for training and evaluating the anomalib SuperSimpleNet model
with the torchsig-models dataset adapter infrastructure.

Pipeline: SuperSimpleNet model with default pre-, post-processors.

SuperSimpleNet is an adapted discriminator type model, where a simple, fast discriminative model
that combines a pretrained feature extractor with an adapted segmentation head.

Usage:
    python3 train_supersimplenet.py 
        --run_name <name> --output_dir <dir> --train_root <root> --eval_root <root>

"""
import argparse
import json
import os
from pathlib import Path
from torch import set_float32_matmul_precision
from lightning.pytorch.callbacks import EarlyStopping
from torchvision.transforms.v2 import Normalize

from anomalib.metrics import AUROC, AUPR, F1Score, F1Max, Evaluator
from anomalib.callbacks.checkpoint import ModelCheckpoint
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.pre_processing import PreProcessor
from anomalib.engine import Engine
from anomalib.models import Supersimplenet

from torchsig_models.adapters.anomalib_utils import TorchSigAnomalibDataModule


set_float32_matmul_precision("high")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True, type=str, help="Name to label model training run.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save outputs.")
    parser.add_argument("--train_root", required=True, type=str, help="Root for TorchSig train dataset.")
    parser.add_argument("--eval_root", required=True, type=str, help="Root for TorchSig evaluation dataset.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loading. Default is 64.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for data loading. Default is 32.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train. Default is 10.")
    parser.add_argument("--unsupervised", action='store_true', help="Train model in unsupervised mode.")
    args = parser.parse_args()

    run_name = args.run_name
    output_dir = args.output_dir
    os.makedirs(output_dir + "/results", exist_ok=True)
    os.makedirs(output_dir + "/checkpoints", exist_ok=True)

    supervised_mode = not bool(args.unsupervised)

    # config parameters
    cfg = {
        "model": {
            "name": "Supersimplenet", 
            "backbone": "wide_resnet50_2.tv_in1k", 
            "layers": ['layer2', 'layer3'],
            "supervised": supervised_mode
        },
        "data": {"dataset": "torchsig_anomalies"},
        "engine": {"max_epochs": args.num_epochs},
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
        use_anomaly_labels = supervised_mode,    # use/ignore anomaly labels
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

    # Configure PreProcessor 
    # assume input image matched to (256x256), no resizing or cropping
    # ImageNet standard channel-wise normalization
    pre_processor = PreProcessor(
        transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )

    # model: SuperSimpleNet (see documentation for parameters)
    model = Supersimplenet(
        backbone=cfg["model"]["backbone"],
        layers=cfg["model"]["layers"],
        supervised=cfg["model"]["supervised"],
        pre_processor=pre_processor,  
        post_processor=True,
        evaluator=evaluator,
        visualizer=False,
    )

    # Anomalib engine
    model_checkpoint = ModelCheckpoint(
        dirpath=args.output_dir + "/checkpoints",
        filename="ssn-wr50-best",
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
