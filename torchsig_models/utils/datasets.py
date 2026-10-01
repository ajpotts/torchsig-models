"""Dataset and dataloader utilities for TorchSig models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
    target_labels: list[str] | None = None,
) -> tuple[StaticTorchSigDataset, list[str]]:
    """Generate and load one static TorchSig dataset split."""
    split_root = root / split

    iterable_dataset = TorchSigIterableDataset(
        metadata=_dataset_metadata(cfg),
        transforms=transforms,
        signal_generators=signal_generators,
    )

    creation_loader = WorkerSeedingDataLoader(
        iterable_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: batch,
        seed=cfg.seed,
    )
    # Worker initialization does not run when num_workers=0, so explicitly
    # seed both the loader and its iterable dataset before generation.
    creation_loader.seed(cfg.seed)

    creator = DatasetCreator(
        dataloader=creation_loader,
        root=str(split_root),
        overwrite=overwrite,
        dataset_length=int(cfg.dataset_length),
    )
    creator.create()

    static_dataset = StaticTorchSigDataset(
        root=str(split_root),
        target_labels=(
            target_labels
            if target_labels is not None
            else getattr(cfg, "target_labels", ["class_index"])
        ),
    )

    return static_dataset, list(iterable_dataset.class_names)


def _loader_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@dataclass(frozen=True)
class _TorchSigWorkerSeeder:
    """Seed a TorchSig dataset from PyTorch's deterministic per-worker seed."""

    worker_init_fn: Callable[[int], None] | None = None

    def __call__(self, worker_id: int) -> None:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = int(torch.initial_seed() % (2**32))
            dataset_seed = getattr(worker_info.dataset, "seed", None)
            if callable(dataset_seed):
                dataset_seed(seed)
        if self.worker_init_fn is not None:
            self.worker_init_fn(worker_id)


def _lightning_compatible_dataloader(
    dataset: torch.utils.data.Dataset,
    *,
    seed: int,
    **kwargs: Any,
) -> torch.utils.data.DataLoader:
    """Build a Lightning-reconstructable loader with TorchSig-style seeding.

    Temporary TorchSig 2.2 compatibility workaround: its
    ``WorkerSeedingDataLoader(dataset, seed=None, **kwargs)`` constructor is
    reconstructed incorrectly by PyTorch Lightning, which can forward a
    literal ``kwargs`` argument to PyTorch's ``DataLoader``. This standard
    loader preserves deterministic dataset and per-worker seeding and can be
    removed once TorchSig includes the upstream Lightning compatibility fix.
    """
    dataset_seed = getattr(dataset, "seed", None)
    if callable(dataset_seed):
        dataset_seed(seed)

    loader_kwargs = dict(kwargs)
    loader_kwargs.setdefault("generator", _loader_generator(seed))
    loader_kwargs["worker_init_fn"] = _TorchSigWorkerSeeder(
        loader_kwargs.get("worker_init_fn")
    )
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


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
    target_labels: list[str] | None = None,
    collate_fn: Callable[[list[Any]], Any] | None = None,
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
        target_labels: Optional metadata labels returned by each static
            dataset. This is useful for detection datasets whose targets are
            object lists rather than one class index.
        collate_fn: Optional function used to collate samples in all returned
            dataloaders.

    Returns:
        Training, validation, and test loaders followed by dataset metadata.
    """
    root = Path(dataset_root) / train_cfg.dataset_id
    root.mkdir(parents=True, exist_ok=True)

    if transforms is None:
        transforms = _transforms(train_cfg)

    train_dataset, class_names = _create_static_dataset(
        train_cfg,
        "train",
        root,
        transforms,
        batch_size,
        overwrite,
        signal_generators=signal_generators,
        target_labels=target_labels,
    )

    val_dataset, _ = _create_static_dataset(
        val_cfg,
        "val",
        root,
        transforms,
        batch_size,
        overwrite,
        signal_generators=signal_generators,
        target_labels=target_labels,
    )

    test_dataset, _ = _create_static_dataset(
        test_cfg,
        "test",
        root,
        transforms,
        batch_size,
        overwrite,
        signal_generators=signal_generators,
        target_labels=target_labels,
    )

    return (
        _lightning_compatible_dataloader(
            train_dataset,
            seed=train_cfg.seed,
            batch_size=batch_size,
            shuffle=True,
            generator=_loader_generator(train_cfg.seed),
            collate_fn=collate_fn,
        ),
        _lightning_compatible_dataloader(
            val_dataset,
            seed=val_cfg.seed,
            batch_size=batch_size,
            shuffle=False,
            generator=_loader_generator(val_cfg.seed),
            collate_fn=collate_fn,
        ),
        _lightning_compatible_dataloader(
            test_dataset,
            seed=test_cfg.seed,
            batch_size=batch_size,
            shuffle=False,
            generator=_loader_generator(test_cfg.seed),
            collate_fn=collate_fn,
        ),
        {"root": str(root), "class_names": class_names},
    )


def prepare_torchsig_inference_dataset(
    root: str | Path,
    *,
    batch_size: int = 4,
    num_workers: int = 8,
    target_labels: list[str] | None = None,
    pin_memory: bool | None = None,
    collate_fn: Callable[[list[Any]], Any] | None = None,
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
        collate_fn: Optional function used to collate inference samples.

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

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": False,
        "pin_memory": pin_memory,
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn

    return _lightning_compatible_dataloader(dataset, seed=0, **loader_kwargs)
