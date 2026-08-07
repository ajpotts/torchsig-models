"""Training XCiT 1D Classifier Model."""

from torchsig_models.models import XCiTClassifier
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.datasets.datamodules import TorchSigDataModule
from torchsig.transforms.transforms import ComplexTo2D

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchsig_models.utils import ClassifierMetricsTrackerCallback
from torchsig.utils.yaml import load_config_from_yaml

import os
import torch
import pytorch_lightning as pl

torch.set_float32_matmul_precision("high")


def xcit1d_trainer(root, config_file, pt_dir, metrics_dir, num_epochs) -> None:
    """Train an XCiT classifier on the dataset at root, saving checkpoints to pt_dir and
    metrics to metrics_dir. Training parameters are set in config_file. num_epochs specifies
    how many epochs to train for.

    Args:
        root: Path to dataset.
        config_file: Path to yaml config file specifying dataset metadata and training parameters.
        pt_dir: Path to place checkpoints.
        metrics_dir: Path to place metric graphs.
        num_epochs: Number of epochs to train.

    """

    # directories
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(os.path.join(root, "conf_mats"), exist_ok=True)

    # logger
    logger = TensorBoardLogger(
        os.path.join(metrics_dir, "tb_logs"), name="1dxcit_nb_classifier"
    )

    # metadata
    cfg = load_config_from_yaml(config_file)
    base = TorchSigDefaults().default_dataset_metadata
    dataset_metadata = dict(base)
    dataset_metadata.update(cfg.dataset_metadata)

    # assume all signal classes
    class_list = TorchSigSignalLists.all_signals
    num_classes = len(class_list)

    # DataModule
    dm = TorchSigDataModule(
        root=root,
        metadata=dataset_metadata,
        dataset_size=cfg.dataset_length,
        dataset_splits=[0.7, 0.2, 0.1],
        batch_size=64,
        num_workers=32,
        collate_fn=None,
        overwrite=False,
        impairment_level=cfg.impairment_level,
        transforms=[ComplexTo2D()],
        target_labels=["class_index"],
        seed=cfg.seed,
    )
    dm.prepare_data()
    dm.setup()

    # model
    model = XCiTClassifier(input_channels=2, num_classes=num_classes)

    # callbacks
    callbacks = []
    metrics = ClassifierMetricsTrackerCallback(num_classes, str(metrics_dir))
    callbacks.append(metrics)

    checkpoint_callback = ModelCheckpoint(
        dirpath=pt_dir,
        filename="xcit-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)

    # trainer
    trainer = pl.Trainer(
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        default_root_dir=pt_dir,
        logger=logger,
    )
    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(str(pt_dir) + "/classification_model_final.ckpt")

    metrics.plot()
    metrics.save_to_csv()
