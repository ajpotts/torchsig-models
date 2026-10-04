"""Train Ultralytics RT-DETR on TorchSig wideband spectrogram datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
import yaml
from torchsig.datasets.datasets import TorchSigDatasetConfig
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.yaml import load_config_from_yaml

from torchsig_models.models.spectrogram_models.detr.detr import rtdetr_l
from torchsig_models.utils.datasets import prepare_torchsig_datasets
from torchsig_models.utils.normalization import (
    compute_dataset_channel_stats,
    resolve_normalization_mode,
)

DETRModelName = Literal["rtdetr_l"]
NormalizationMode = Literal["sample", "dataset", "none"]
MODEL_FACTORY = {"rtdetr_l": rtdetr_l}

__all__ = [
    "MODEL_FACTORY",
    "DETRModelName",
    "export_ultralytics_dataset",
    "load_training_params",
    "spectrogram_to_uint8",
    "train_detr",
]


def load_training_params(
    model_name: DETRModelName,
    params_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load RT-DETR training parameters from YAML."""
    path = (
        Path(params_path)
        if params_path is not None
        else Path(__file__).parent / "training_params" / f"{model_name}.yaml"
    )
    if not path.exists():
        raise FileNotFoundError(f"Training parameter file not found: {path}")
    params = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        raise TypeError(f"Training parameter file must contain a mapping: {path}")
    return params


def _spectrogram_transforms(cfg: TorchSigDatasetConfig) -> list[Any]:
    """Create the spectrogram and multi-object label transforms."""
    fft_size = getattr(cfg, "output_spectrogram_fft", None)
    if fft_size is None:
        fft_size = getattr(cfg, "dataset_metadata", {}).get("fft_size", 256)
    return [Spectrogram(fft_size=int(fft_size)), YOLOLabel()]


def _detection_collate(batch: Sequence[Any]) -> tuple[torch.Tensor, list[Any]]:
    """Collate spectrograms while preserving variable-length object lists."""
    images, labels = zip(*batch)
    return torch.as_tensor(np.stack(images), dtype=torch.float32), list(labels)


def spectrogram_to_uint8(
    image: np.ndarray | torch.Tensor,
    *,
    normalization: NormalizationMode,
    mean: float | None = None,
    std: float | None = None,
    eps: float = 1e-6,
    clip_sigma: float = 4.0,
    image_min: float = -80.0,
    image_max: float = 40.0,
) -> np.ndarray:
    """Encode a floating-point TorchSig spectrogram as an 8-bit image.

    Sample and dataset normalization use symmetric standard-deviation clipping.
    ``none`` performs only the fixed-range mapping required to represent a
    floating-point spectrogram in Ultralytics' image input format.
    """
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 3:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D spectrogram, got shape {values.shape}.")

    if normalization == "none":
        if image_max <= image_min:
            raise ValueError("image_max must be greater than image_min.")
        scaled = (values - image_min) / (image_max - image_min)
        return np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)

    if clip_sigma <= 0:
        raise ValueError("clip_sigma must be greater than zero.")
    if normalization == "sample":
        center, scale = float(values.mean()), float(values.std())
    elif normalization == "dataset":
        if mean is None or std is None:
            raise ValueError("Dataset normalization requires mean and std.")
        center, scale = float(mean), float(std)
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization!r}.")

    normalized = np.clip(
        (values - center) / max(scale, eps), -clip_sigma, clip_sigma
    )
    encoded = (normalized + clip_sigma) * (255.0 / (2.0 * clip_sigma))
    return np.rint(encoded).astype(np.uint8)


def _write_labels(path: Path, objects: Iterable[Sequence[float]]) -> None:
    lines: list[str] = []
    for obj in objects:
        if len(obj) != 5:
            raise ValueError(f"Expected a five-value YOLO label, got {obj!r}.")
        class_id, center_x, center_y, width, height = obj
        lines.append(
            f"{int(class_id)} {float(center_x):.9g} {float(center_y):.9g} "
            f"{float(width):.9g} {float(height):.9g}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def export_ultralytics_dataset(
    datasets: dict[str, Any],
    output_root: str | Path,
    class_names: Sequence[str],
    *,
    normalization: NormalizationMode,
    mean: float | None = None,
    std: float | None = None,
    eps: float = 1e-6,
    clip_sigma: float = 4.0,
    image_min: float = -80.0,
    image_max: float = 40.0,
    overwrite: bool = False,
) -> Path:
    """Export TorchSig train/validation/test splits in Ultralytics format."""
    output_root = Path(output_root).resolve()
    if output_root.exists() and not overwrite:
        config_path = output_root / "dataset.yaml"
        if config_path.exists():
            return config_path
        raise FileExistsError(
            f"Incomplete Ultralytics dataset exists: {output_root}. "
            "Pass --overwrite to regenerate it."
        )
    if output_root.exists():
        for split in ("train", "val", "test"):
            for pattern in (f"images/{split}/*.png", f"labels/{split}/*.txt"):
                for generated_file in output_root.glob(pattern):
                    generated_file.unlink()
        (output_root / "dataset.yaml").unlink(missing_ok=True)

    for split in ("train", "val", "test"):
        if split not in datasets:
            raise ValueError(f"Missing required dataset split: {split}.")
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(len(datasets[split])):
            image, labels = datasets[split][index]
            encoded = spectrogram_to_uint8(
                image,
                normalization=normalization,
                mean=mean,
                std=std,
                eps=eps,
                clip_sigma=clip_sigma,
                image_min=image_min,
                image_max=image_max,
            )
            stem = f"{index:010d}"
            image_path = image_dir / f"{stem}.png"
            if not cv2.imwrite(str(image_path), encoded):
                raise OSError(f"Failed to write image {image_path}.")
            _write_labels(label_dir / f"{stem}.txt", labels)

    config = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {index: name for index, name in enumerate(class_names)},
    }
    config_path = output_root / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _metric_summary(metrics: Any) -> dict[str, float]:
    boxes = metrics.box
    return {
        "map": float(boxes.map),
        "map_50": float(boxes.map50),
        "precision": float(boxes.mp),
        "recall": float(boxes.mr),
    }


def train_detr(
    train_cfg: TorchSigDatasetConfig,
    val_cfg: TorchSigDatasetConfig,
    test_cfg: TorchSigDatasetConfig,
    params: dict[str, Any],
    output_dir: str | Path,
    *,
    dataset_root: str | Path = "datasets",
    overwrite: bool = False,
    model_name: DETRModelName = "rtdetr_l",
    signal_generators: str | list[str] = "all",
    device: str | int | None = None,
    workers: int = 1,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    """Train and optionally test RT-DETR on multi-signal TorchSig data."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, data_info = prepare_torchsig_datasets(
        train_cfg,
        val_cfg,
        test_cfg,
        dataset_root=dataset_root,
        batch_size=int(params["batch_size"]),
        overwrite=overwrite,
        signal_generators=signal_generators,
        transforms=_spectrogram_transforms(train_cfg),
        target_labels=["yolo_label"],
        collate_fn=_detection_collate,
    )

    normalization = resolve_normalization_mode(
        params.get("normalization"), params.get("normalize")
    )
    if normalization not in ("sample", "dataset", "none"):
        raise ValueError(f"Unsupported normalization mode: {normalization!r}.")
    eps = float(params.get("normalization_eps", 1e-6))
    mean = std = None
    if normalization == "dataset":
        means, stds = compute_dataset_channel_stats(train_loader, add_channel_dim=True)
        mean, std = float(means[0]), float(stds[0])

    normalization_metadata = {
        "mode": normalization,
        "eps": eps,
        "mean": mean,
        "std": std,
        "clip_sigma": float(params.get("clip_sigma", 4.0)),
        "image_min": float(params.get("image_min", -80.0)),
        "image_max": float(params.get("image_max", 40.0)),
    }
    dataset_yaml = export_ultralytics_dataset(
        {
            "train": train_loader.dataset,
            "val": val_loader.dataset,
            "test": test_loader.dataset,
        },
        output_dir / "dataset",
        data_info["class_names"],
        normalization=normalization,
        mean=mean,
        std=std,
        eps=eps,
        clip_sigma=normalization_metadata["clip_sigma"],
        image_min=normalization_metadata["image_min"],
        image_max=normalization_metadata["image_max"],
        overwrite=overwrite,
    )

    model = MODEL_FACTORY[model_name](
        pretrained=bool(params.get("pretrained", True)),
        path=params.get("model_path"),
    )
    image_size = int(params.get("image_size", 512))
    train_metrics = model.train(
        data=str(dataset_yaml),
        epochs=int(params["max_epochs"]),
        batch=int(params["batch_size"]),
        imgsz=image_size,
        project=str(output_dir),
        name="train",
        exist_ok=overwrite,
        device=device,
        workers=workers,
        seed=int(train_cfg.seed),
        deterministic=False,
        optimizer=str(params.get("optimizer", "AdamW")),
        lr0=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        cos_lr=bool(params.get("cos_lr", True)),
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    best_checkpoint = Path(model.trainer.best).resolve()
    val_summary = _metric_summary(model.metrics)
    test_metrics = None
    test_summary = None
    if evaluate_test:
        test_metrics = model.val(
            data=str(dataset_yaml),
            split="test",
            imgsz=image_size,
            batch=int(params["batch_size"]),
            device=device,
            workers=workers,
            project=str(output_dir),
            name="test",
            exist_ok=overwrite,
        )
        test_summary = _metric_summary(test_metrics)

    metadata = {
        "model_name": model_name,
        "class_names": data_info["class_names"],
        "normalization": normalization_metadata,
        "dataset_yaml": str(dataset_yaml),
        "best_checkpoint": str(best_checkpoint),
        "validation": val_summary,
        "test": test_summary,
    }
    (output_dir / "training_metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps({"validation": val_summary, "test": test_summary}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "best_checkpoint": str(best_checkpoint),
        "dataset_yaml": str(dataset_yaml),
        "data_info": data_info,
        "normalization": normalization_metadata,
    }


def _parse_device(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    """Parse RT-DETR training command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--val-config", type=Path)
    parser.add_argument("--test-config", type=Path)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--model", choices=list(MODEL_FACTORY), default="rtdetr_l")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--dataset-length", type=int)
    parser.add_argument("--dataset-id")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--signal-generators", nargs="+", default="all")
    parser.add_argument("--device", type=_parse_device)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dataset_config is None and args.train_config is None:
        parser.error("one of --dataset-config or --train-config is required")
    return args


def _load_configs(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    train_path = args.train_config or args.dataset_config
    assert train_path is not None
    val_path = args.val_config or args.dataset_config or train_path
    test_path = args.test_config or args.dataset_config or train_path
    train_cfg = load_config_from_yaml(train_path)
    val_cfg = load_config_from_yaml(val_path)
    test_cfg = load_config_from_yaml(test_path)
    if args.val_config is None:
        val_cfg = replace(val_cfg, seed=train_cfg.seed + 1)
    if args.test_config is None:
        test_cfg = replace(test_cfg, seed=train_cfg.seed + 2)
    updates = {
        key: value
        for key, value in {
            "dataset_length": args.dataset_length,
            "dataset_id": args.dataset_id,
        }.items()
        if value is not None
    }
    if updates:
        train_cfg, val_cfg, test_cfg = (
            replace(cfg, **updates) for cfg in (train_cfg, val_cfg, test_cfg)
        )
    return train_cfg, val_cfg, test_cfg


def main() -> None:
    """Run RT-DETR training from the command line."""
    args = parse_args()
    train_cfg, val_cfg, test_cfg = _load_configs(args)
    params = load_training_params(args.model, args.params)
    if args.epochs is not None:
        params["max_epochs"] = args.epochs
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    run_dir = args.output_dir / train_cfg.dataset_id / args.model
    result = train_detr(
        train_cfg,
        val_cfg,
        test_cfg,
        params,
        run_dir,
        dataset_root=args.dataset_root,
        overwrite=args.overwrite,
        model_name=args.model,
        signal_generators=args.signal_generators,
        device=args.device,
        workers=args.workers,
    )
    print(json.dumps({"validation": result["val_summary"], "test": result["test_summary"]}, indent=2))
    print(f"Best checkpoint: {result['best_checkpoint']}")


if __name__ == "__main__":
    main()
