"""Dataset and dataloader utilities for TorchSig models."""

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
    Spectrogram,
    Transform,
)
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.writer import DatasetCreator


__all__ = [
    "prepare_torchsig_datasets",
    "prepare_torchsig_inference_dataset",
]


def _dataset_metadata(
    cfg: TorchSigDatasetConfig,
) -> dict[str, Any]:
    """Combine TorchSig defaults with dataset-specific metadata."""
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(cfg.dataset_metadata)
    return metadata


def _transforms(
    cfg: TorchSigDatasetConfig,
) -> list[Transform]:
    """Build transforms for the configured output representation."""
    if cfg.output_representation.lower() == "iq":
        return [ComplexTo2D()]

    if cfg.output_representation.lower() == "spectrogram":
        fft_size = cfg.dataset_metadata.get(
            "fft_size",
            getattr(cfg, "fft_size", 256),
        )
        return [
            Spectrogram(fft_size=fft_size),
            YOLOLabel(),
        ]

    fft_size = getattr(cfg, "fft_size", 256)
    return [Spectrogram(fft_size=fft_size)]


def _create_static_dataset(
    cfg: TorchSigDatasetConfig,
    split: str,
    root: Path,
    transforms: list[Transform],
    batch_size: int,
    overwrite: bool,
    *,
    signal_generators: str | list[str] = "all",
) -> StaticTorchSigDataset:
    """Generate and load one static TorchSig dataset split."""
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
        target_labels=getattr(
            cfg,
            "target_labels",
            ["class_index"],
        ),
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
    transforms: list[Transform] | None = None,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    dict[str, Any],
]:
    """Generate static TorchSig datasets and return split dataloaders.

    Args:
        train_cfg: Configuration for the training split.
        val_cfg: Configuration for the validation split.
        test_cfg: Configuration for the test split.
        signal_generators: Signal generators used for dataset creation.
        dataset_root: Parent directory for the generated dataset.
        batch_size: Batch size used for creation and returned loaders.
        overwrite: Whether existing static datasets may be overwritten.
        transforms: Optional transforms applied while generating every split.
            When omitted, transforms are inferred from ``train_cfg``.

    Returns:
        Training, validation, and test loaders followed by dataset metadata.
    """
    root = Path(dataset_root) / train_cfg.dataset_id
    root.mkdir(parents=True, exist_ok=True)

    if transforms is None:
        transforms = _transforms(train_cfg)

    train_dataset = _create_static_dataset(
        train_cfg,
        "train",
        root,
        transforms,
        batch_size,
        overwrite,
        signal_generators=signal_generators,
    )
    val_dataset = _create_static_dataset(
        val_cfg,
        "val",
        root,
        transforms,
        batch_size,
        overwrite,
        signal_generators=signal_generators,
    )
    test_dataset = _create_static_dataset(
        test_cfg,
        "test",
        root,
        transforms,
        batch_size,
        overwrite,
        signal_generators=signal_generators,
    )

    return (
        WorkerSeedingDataLoader(
            train_dataset,
            batch_size=batch_size,
        ),
        WorkerSeedingDataLoader(
            val_dataset,
            batch_size=batch_size,
        ),
        WorkerSeedingDataLoader(
            test_dataset,
            batch_size=batch_size,
        ),
        {"root": str(root)},
    )


def prepare_torchsig_inference_dataset(
    root: str | Path,
    *,
    batch_size: int = 4,
    num_workers: int = 8,
    target_labels: list[str] | None = None,
    pin_memory: bool | None = None,
) -> torch.utils.data.DataLoader:
    """Load a static TorchSig dataset for inference.

    Args:
        root: Root directory of the static TorchSig dataset.
        batch_size: Number of examples per inference batch.
        num_workers: Number of dataloader worker processes.
        target_labels: Labels returned by the static dataset. Defaults to
            ``["class_index"]``.
        pin_memory: Whether the dataloader should pin memory. If omitted,
            pinning is enabled when CUDA is available.

    Returns:
        Dataloader for the static inference dataset.

    Raises:
        FileNotFoundError: If the dataset root does not exist.
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    if target_labels is None:
        target_labels = ["class_index"]

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataset = StaticTorchSigDataset(
        root=str(root),
        target_labels=target_labels,
    )

    return WorkerSeedingDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )
