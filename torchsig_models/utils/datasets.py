"""Dataset and dataloader utilities for TorchSig models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import yaml
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
from torchsig.utils.file_handlers.base_handler import FileReader, FileWriter
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer
from torchsig.utils.file_handlers.homogeneous_hdf5 import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)
from torchsig.utils.file_handlers.packed_hdf5 import PackedHDF5Reader, PackedHDF5Writer
from torchsig.utils.writer import DatasetCreator

__all__ = [
    "DatasetMode",
    "StorageBackend",
    "prepare_torchsig_datasets",
    "prepare_torchsig_inference_dataset",
]


DatasetMode = Literal["auto", "create", "existing"]
StorageBackend = Literal["legacy", "packed", "homogeneous"]


_STORAGE_BACKENDS: dict[
    StorageBackend,
    tuple[type[FileWriter], type[FileReader]],
] = {
    "legacy": (HDF5Writer, HDF5Reader),
    "packed": (PackedHDF5Writer, PackedHDF5Reader),
    "homogeneous": (HomogeneousHDF5Writer, HomogeneousHDF5Reader),
}


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
    file_handler: type[FileWriter] | None = None,
    file_reader: type[FileReader] | None = None,
    file_handler_options: dict[str, Any] | None = None,
    storage_backend: StorageBackend | None = None,
) -> tuple[StaticTorchSigDataset, list[str]]:
    """Generate and load one static TorchSig dataset split."""
    split_root = root / split
    file_handler, file_reader = _configured_file_handlers(
        cfg, file_handler, file_reader, storage_backend
    )
    if file_handler_options is None:
        file_handler_options = dict(getattr(cfg, "file_writer_kwargs", {}))

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
        file_handler=file_handler,
        file_reader=file_reader,
        **file_handler_options,
    )
    creator.create()

    static_dataset = StaticTorchSigDataset(
        root=str(split_root),
        file_handler_class=file_reader,
        target_labels=getattr(
            cfg,
            "target_labels",
            ["class_index"],
        ),
    )

    return static_dataset, list(iterable_dataset.class_names)


def _configured_file_handlers(
    cfg: TorchSigDatasetConfig,
    file_handler: type[FileWriter] | None,
    file_reader: type[FileReader] | None,
    storage_backend: StorageBackend | None = None,
) -> tuple[type[FileWriter], type[FileReader]]:
    """Resolve the configured storage writer and reader classes."""
    configured_backend = storage_backend or getattr(cfg, "file_writer_name", "legacy")
    if configured_backend not in _STORAGE_BACKENDS:
        raise ValueError(f"Unsupported dataset storage backend: {configured_backend!r}")
    configured_writer, configured_reader = _STORAGE_BACKENDS[configured_backend]
    return (
        configured_writer if file_handler is None else file_handler,
        configured_reader if file_reader is None else file_reader,
    )


def _load_existing_dataset(
    cfg: TorchSigDatasetConfig,
    split: str,
    root: Path,
    file_reader: type[FileReader] | None,
    storage_backend: StorageBackend | None = None,
) -> tuple[StaticTorchSigDataset, list[str]]:
    """Load and validate an existing static dataset without modifying it."""
    split_description = "validation" if split == "val" else split
    if not root.is_dir():
        raise FileNotFoundError(
            f"Existing TorchSig {split_description} dataset directory not found: {root}"
        )

    _, configured_reader = _configured_file_handlers(
        cfg, None, file_reader, storage_backend
    )
    try:
        dataset = StaticTorchSigDataset(
            root=str(root),
            file_handler_class=configured_reader,
            target_labels=getattr(cfg, "target_labels", ["class_index"]),
        )
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(
            f"Invalid TorchSig {split_description} dataset at {root}: {error}"
        ) from error

    class_names = list(cfg.dataset_metadata.get("class_names", []))
    dataset_info = root / "dataset_info.yaml"
    if dataset_info.is_file():
        try:
            with dataset_info.open(encoding="utf-8") as file:
                metadata = (yaml.safe_load(file) or {}).get("dataset_metadata", {})
            class_names = list(metadata.get("class_names", class_names))
        except (OSError, TypeError, yaml.YAMLError) as error:
            raise ValueError(
                f"Invalid TorchSig {split_description} dataset metadata at "
                f"{dataset_info}: {error}"
            ) from error

    return dataset, class_names


def _existing_split_roots(
    dataset_root: Path,
    configs: tuple[
        TorchSigDatasetConfig,
        TorchSigDatasetConfig,
        TorchSigDatasetConfig,
    ],
) -> tuple[Path, Path, Path] | None:
    """Find an existing nested or dataset-ID-based split layout."""
    split_names = ("train", "val", "test")
    nested_root = dataset_root / configs[0].dataset_id
    nested_roots = tuple(nested_root / split for split in split_names)
    if any(path.exists() for path in nested_roots):
        return nested_roots

    dataset_ids = tuple(cfg.dataset_id for cfg in configs)
    separate_roots = tuple(dataset_root / dataset_id for dataset_id in dataset_ids)
    if len(set(dataset_ids)) > 1 and any(path.exists() for path in separate_roots):
        return separate_roots

    return None


def _configured_split_roots(
    dataset_root: Path,
    configs: tuple[
        TorchSigDatasetConfig,
        TorchSigDatasetConfig,
        TorchSigDatasetConfig,
    ],
) -> tuple[Path, Path, Path]:
    """Return the split roots implied by the dataset configurations."""
    dataset_ids = tuple(cfg.dataset_id for cfg in configs)
    if len(set(dataset_ids)) > 1:
        return tuple(dataset_root / dataset_id for dataset_id in dataset_ids)

    root = dataset_root / dataset_ids[0]
    return tuple(root / split for split in ("train", "val", "test"))


def _loader_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class _SeedableSubset(torch.utils.data.Subset):
    """Subset compatible with TorchSig's seed-propagating data loader."""

    def seed(self, seed: int) -> None:
        """Forward a loader seed to the underlying dataset when supported."""
        seed_dataset = getattr(self.dataset, "seed", None)
        if seed_dataset is not None:
            seed_dataset(seed)


def _subsample_existing_dataset(
    dataset: torch.utils.data.Dataset,
    requested_length: int,
    seed: int,
) -> torch.utils.data.Dataset:
    """Return a deterministic, non-mutating subset of an existing dataset."""
    if requested_length < 0:
        raise ValueError("dataset_length must be greater than or equal to zero.")
    subset_length = min(requested_length, len(dataset))
    indices = torch.randperm(len(dataset), generator=_loader_generator(seed))[
        :subset_length
    ].tolist()
    return _SeedableSubset(dataset, indices)


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
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    file_handler: type[FileWriter] | None = None,
    file_reader: type[FileReader] | None = None,
    file_handler_options: dict[str, Any] | None = None,
    dataset_mode: DatasetMode = "auto",
    storage_backend: StorageBackend | None = None,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    dict[str, Any],
]:
    """Load or generate static TorchSig datasets and return split dataloaders.

    Existing datasets may either use ``<dataset_id>/<split>`` directories or
    one directory per split configuration's dataset ID. ``dataset_mode`` can
    require loading existing data without entering the generation workflow.

    Args:
        train_cfg: Configuration for the training split.
        val_cfg: Configuration for the validation split.
        test_cfg: Configuration for the test split.
        signal_generators: Signal generators used for dataset creation.
        dataset_root: Parent directory for existing or generated datasets.
        batch_size: Batch size used for creation and returned loaders.
        overwrite: Whether existing static datasets may be overwritten.
        transforms: Optional transforms applied while generating every split.
            When omitted, transforms are inferred from ``train_cfg``.
        num_workers: Number of worker processes used by the returned loaders.
        pin_memory: Whether the returned loaders pin host memory. If omitted,
            pinning is enabled when CUDA is available.
        persistent_workers: Whether worker processes remain alive between
            epochs. If omitted, persistence is enabled when ``num_workers`` is
            greater than zero. It is always disabled when ``num_workers`` is
            zero.
        file_handler: Optional writer override. When omitted, the backend is
            selected from ``cfg.file_writer_name`` (the dataset YAML's
            ``storage.writer`` field).
        file_reader: Optional reader override paired with ``file_handler``.
            When omitted, the reader matching the configured backend is used.
        file_handler_options: Optional keyword arguments forwarded to the
            file handler. When omitted, ``cfg.file_writer_kwargs`` (the
            dataset YAML's ``storage.options`` field) is used. Homogeneous HDF5
            requires every top-level sample to have the same shape and dtype.
        dataset_mode: Dataset handling policy. ``"auto"`` preserves the
            legacy behavior of loading a detected layout when ``overwrite`` is
            false and generating otherwise. ``"create"`` always enters the
            generation workflow. ``"existing"`` only loads configured split
            directories and never constructs generation datasets or loaders.
        storage_backend: Optional ``"legacy"``, ``"packed"``, or
            ``"homogeneous"`` storage override. When omitted, each dataset
            configuration selects its backend. For existing datasets, each
            split is deterministically limited to its configured
            ``dataset_length`` using that split's seed; requests larger than a
            split retain all available samples.

    Returns:
        Training, validation, and test loaders followed by dataset metadata.

    Raises:
        FileNotFoundError: If an existing split layout is incomplete.
        ValueError: If an existing split cannot be loaded by its configured
            storage reader, or if ``dataset_mode`` is unsupported.
    """
    dataset_root = Path(dataset_root)
    root = dataset_root / train_cfg.dataset_id

    if num_workers < 0:
        raise ValueError("num_workers must be greater than or equal to zero.")
    if dataset_mode not in ("auto", "create", "existing"):
        raise ValueError("dataset_mode must be one of 'auto', 'create', or 'existing'.")
    if storage_backend is not None and storage_backend not in _STORAGE_BACKENDS:
        raise ValueError(f"Unsupported dataset storage backend: {storage_backend!r}")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = num_workers > 0
    elif persistent_workers and num_workers == 0:
        persistent_workers = False

    if transforms is None:
        transforms = _transforms(train_cfg)

    configs = (train_cfg, val_cfg, test_cfg)
    split_names = ("train", "val", "test")
    if dataset_mode == "existing":
        existing_roots = _configured_split_roots(dataset_root, configs)
    elif dataset_mode == "create" or overwrite:
        existing_roots = None
    else:
        existing_roots = _existing_split_roots(dataset_root, configs)
    if existing_roots is not None:
        loaded = tuple(
            _load_existing_dataset(cfg, split, split_root, file_reader, storage_backend)
            for cfg, split, split_root in zip(
                configs, split_names, existing_roots, strict=True
            )
        )
        train_dataset, class_names = loaded[0]
        val_dataset = loaded[1][0]
        test_dataset = loaded[2][0]
        train_dataset, val_dataset, test_dataset = tuple(
            _subsample_existing_dataset(dataset, int(cfg.dataset_length), int(cfg.seed))
            for dataset, cfg in zip(
                (train_dataset, val_dataset, test_dataset), configs, strict=True
            )
        )
    else:
        root.mkdir(parents=True, exist_ok=True)
        created = tuple(
            _create_static_dataset(
                cfg,
                split,
                root,
                transforms,
                batch_size,
                overwrite,
                signal_generators=signal_generators,
                file_handler=file_handler,
                file_reader=file_reader,
                file_handler_options=file_handler_options,
                storage_backend=storage_backend,
            )
            for cfg, split in zip(configs, split_names, strict=True)
        )
        train_dataset, class_names = created[0]
        val_dataset = created[1][0]
        test_dataset = created[2][0]

    return (
        WorkerSeedingDataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            seed=train_cfg.seed,
            generator=_loader_generator(train_cfg.seed),
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
        WorkerSeedingDataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            seed=val_cfg.seed,
            generator=_loader_generator(val_cfg.seed),
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
        WorkerSeedingDataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            seed=test_cfg.seed,
            generator=_loader_generator(test_cfg.seed),
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
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
