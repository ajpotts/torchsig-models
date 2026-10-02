"""Train Ultralytics RT-DETR on the same TorchSig data as the local DETR.

TorchSig static datasets store floating-point spectrograms, whereas Ultralytics
expects image files and YOLO text labels. This module provides the small export
adapter needed for a direct comparison while retaining the same generated
samples, split seeds, class order, bounding boxes, and normalization strategy.
"""

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
from torchsig.utils.yaml import load_config_from_yaml
from ultralytics import RTDETR

from torchsig_models.models.spectrogram_models.detr.detr_train import (
    _spectrogram_transforms,
    detr_collate,
    load_training_params,
)
from torchsig_models.utils.datasets import prepare_torchsig_datasets
from torchsig_models.utils.normalization import (
    compute_dataset_channel_stats,
    resolve_normalization_mode,
)

NormalizationMode = Literal["sample", "dataset"]

__all__ = [
    "export_ultralytics_dataset",
    "spectrogram_to_uint8",
    "train_ultralytics_rtdetr",
]


def spectrogram_to_uint8(
    image: np.ndarray | torch.Tensor,
    *,
    normalization: NormalizationMode,
    mean: float | None = None,
    std: float | None = None,
    eps: float = 1e-6,
    clip_sigma: float = 4.0,
) -> np.ndarray:
    """Normalize a spectrogram and encode it as an 8-bit grayscale image.

    The local DETR normalizes floating-point tensors inside the model. RT-DETR
    consumes ordinary image files, so normalized values are clipped symmetrically
    and mapped to the full uint8 range. Clipping is the only lossy conversion.
    """
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 3:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D spectrogram, got shape {values.shape}.")
    if clip_sigma <= 0:
        raise ValueError("clip_sigma must be greater than zero.")

    if normalization == "sample":
        center = float(values.mean())
        scale = float(values.std())
    elif normalization == "dataset":
        if mean is None or std is None:
            raise ValueError("Dataset normalization requires mean and std.")
        center, scale = float(mean), float(std)
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization!r}.")

    normalized = (values - center) / max(scale, eps)
    normalized = np.clip(normalized, -clip_sigma, clip_sigma)
    encoded = (normalized + clip_sigma) * (255.0 / (2.0 * clip_sigma))
    return np.rint(encoded).astype(np.uint8)


def _write_labels(path: Path, objects: Iterable[Sequence[float]]) -> None:
    lines = []
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
    overwrite: bool = False,
) -> Path:
    """Export TorchSig splits and return an Ultralytics dataset YAML path."""
    output_root = Path(output_root).resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Ultralytics dataset already exists: {output_root}. "
                "Pass --overwrite to regenerate it."
            )
        # Remove only files owned by this exporter. Keep any unrelated files a
        # caller may have placed beside the generated dataset.
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
            )
            stem = f"{index:010d}"
            if not cv2.imwrite(str(image_dir / f"{stem}.png"), encoded):
                raise OSError(f"Failed to write image {image_dir / f'{stem}.png'}.")
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


def train_ultralytics_rtdetr(
    train_cfg: TorchSigDatasetConfig,
    val_cfg: TorchSigDatasetConfig,
    test_cfg: TorchSigDatasetConfig,
    params: dict[str, Any],
    output_dir: str | Path,
    *,
    dataset_root: str | Path = "datasets",
    overwrite: bool = False,
    signal_generators: str | list[str] = "all",
    model: str | Path = "rtdetr-l.pt",
    device: str | int | None = None,
    workers: int = 1,
    clip_sigma: float = 4.0,
) -> dict[str, Any]:
    """Train and test Ultralytics RT-DETR on matching TorchSig splits."""
    # Ultralytics resolves relative ``project`` paths beneath its configured
    # runs directory. Pass an absolute path so artifacts land exactly where
    # this wrapper reports them.
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
        collate_fn=detr_collate,
    )

    normalization = resolve_normalization_mode(
        params.get("normalization"), params.get("normalize")
    )
    if normalization not in ("sample", "dataset"):
        raise ValueError(
            "RT-DETR image export supports sample or dataset normalization; "
            f"got {normalization!r}."
        )
    eps = float(params.get("normalization_eps", 1e-6))
    mean = std = None
    if normalization == "dataset":
        means, stds = compute_dataset_channel_stats(train_loader)
        mean, std = float(means[0]), float(stds[0])

    export_root = output_dir / "dataset"
    dataset_yaml = export_ultralytics_dataset(
        {
            "train": train_loader.dataset,
            "val": val_loader.dataset,
            "test": test_loader.dataset,
        },
        export_root,
        data_info["class_names"],
        normalization=normalization,
        mean=mean,
        std=std,
        eps=eps,
        clip_sigma=clip_sigma,
        overwrite=overwrite,
    )

    detector = RTDETR(str(model))
    image_size = int(getattr(train_cfg, "output_spectrogram_fft", None) or 512)
    train_result = detector.train(
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
        optimizer="AdamW",
        lr0=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        cos_lr=True,
        # Keep the inputs aligned with the local DETR comparison. Generic
        # photographic augmentation is not representative of RF spectrograms.
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
    # Model.train() has already loaded this checkpoint back into ``detector``.
    # Read the actual trainer path instead of reconstructing it from project
    # arguments, since Ultralytics may rewrite relative run directories.
    best_checkpoint = Path(detector.trainer.best).resolve()
    test_metrics = detector.val(
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
    summary = _metric_summary(test_metrics)
    summary.update(
        {
            "model": str(model),
            "best_checkpoint": str(best_checkpoint),
            "dataset_yaml": str(dataset_yaml),
            "normalization": normalization,
        }
    )
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "model": detector,
        "train_result": train_result,
        "test_metrics": test_metrics,
        "summary": summary,
        "data_info": data_info,
    }


def _parse_device(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    """Parse RT-DETR comparison arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--val-config", type=Path)
    parser.add_argument("--test-config", type=Path)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--dataset-length", type=int)
    parser.add_argument("--dataset-id")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--signal-generators", nargs="+", default="all")
    parser.add_argument("--model", default="rtdetr-l.pt")
    parser.add_argument("--device", type=_parse_device)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--clip-sigma", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dataset_config is None and args.train_config is None:
        parser.error("one of --dataset-config or --train-config is required")
    return args


def _load_configs(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    train_path = args.train_config or args.dataset_config
    assert train_path is not None
    train_cfg = load_config_from_yaml(train_path)
    val_cfg = load_config_from_yaml(args.val_config or args.dataset_config or train_path)
    test_cfg = load_config_from_yaml(args.test_config or args.dataset_config or train_path)
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
    """Run the Ultralytics RT-DETR comparison from the command line."""
    args = parse_args()
    train_cfg, val_cfg, test_cfg = _load_configs(args)
    params = load_training_params("detr_b0_nano", args.params)
    if args.epochs is not None:
        params["max_epochs"] = args.epochs
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    run_dir = args.output_dir / train_cfg.dataset_id / "ultralytics_rtdetr"
    result = train_ultralytics_rtdetr(
        train_cfg,
        val_cfg,
        test_cfg,
        params,
        run_dir,
        dataset_root=args.dataset_root,
        overwrite=args.overwrite,
        signal_generators=args.signal_generators,
        model=args.model,
        device=args.device,
        workers=args.workers,
        clip_sigma=args.clip_sigma,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
