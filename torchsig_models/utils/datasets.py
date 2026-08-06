"""Dataset and dataloader utilities for TorchSig model training."""

from __future__ import annotations


from pathlib import Path
from typing import Any

import torch

from torchsig.datasets.datasets import (
    StaticTorchSigDataset,
    TorchSigDatasetConfig,
    TorchSigIterableDataset,
)
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import (
    ComplexTo2D,
    Transform,
    Spectrogram,
)
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.writer import DatasetCreator


__all__ = ["prepare_torchsig_datasets"]


def _dataset_metadata(cfg: TorchSigDatasetConfig) -> dict[str, Any]:
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(cfg.dataset_metadata)
    return metadata


def _transforms(cfg: TorchSigDatasetConfig) -> list[Transform]:
    if cfg.output_representation.lower() == "iq":
        return [ComplexTo2D()]

    if cfg.output_representation.lower() == "spectrogram":
        fft_size = cfg.fft_size
        return [Spectrogram(fft_size=fft_size), YOLOLabel()]

    fft_size = getattr(cfg, "fft_size", 256)
    return [Spectrogram(fft_size=fft_size)]


def _create_static_dataset(
    cfg: TorchSigDatasetConfig,
    split: str,
    root: Path,
    transforms: list[Transform],
    batch_size: int,
    overwrite: bool,
    signal_generators: str | list[str] = "all",
) -> StaticTorchSigDataset:
    split_root = root / split

    creator = DatasetCreator(
        dataloader=WorkerSeedingDataLoader(
            TorchSigIterableDataset(
                metadata=_dataset_metadata(cfg),
                transforms=transforms,
                signal_generators=signal_generators,
            ),
            batch_size=batch_size,
            collate_fn=lambda batch: batch,
        ),
        root=str(split_root),
        overwrite=overwrite,
        dataset_length=int(cfg.dataset_length),
    )
    creator.create()

    return StaticTorchSigDataset(
        root=str(split_root),
        target_labels=getattr(cfg, "target_labels", ["class_index"]),
    )


def prepare_torchsig_datasets(
    train_cfg: TorchSigDatasetConfig,
    val_cfg: TorchSigDatasetConfig,
    test_cfg: TorchSigDatasetConfig,
    *,
    signal_generators: str | list[str] = "all",
    dataset_root: str | Path = "datasets",
    batch_size: int = 64,
    overwrite: bool = False,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    dict[str, Any],
]:
    """Generate static TorchSig datasets and return train/val/test loaders."""
    root = Path(dataset_root) / train_cfg.dataset_id
    root.mkdir(parents=True, exist_ok=True)

    transforms = _transforms(train_cfg)

    train_ds = _create_static_dataset(
        train_cfg,
        "train",
        root,
        transforms,
        batch_size,
        overwrite,
        signal_generators=signal_generators,
    )
    val_ds = _create_static_dataset(
        val_cfg,
        "val",
        root,
        transforms,
        batch_size,
        overwrite,
        signal_generators=signal_generators,
    )
    test_ds = _create_static_dataset(
        test_cfg,
        "test",
        root,
        transforms,
        batch_size,
        overwrite,
        signal_generators=signal_generators,
    )

    return (
        WorkerSeedingDataLoader(train_ds, batch_size=batch_size),
        WorkerSeedingDataLoader(val_ds, batch_size=batch_size),
        WorkerSeedingDataLoader(test_ds, batch_size=batch_size),
        {"root": str(root)},
    )
