"""Inference and evaluation entry point for trained RT-DETR detectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from torchsig_models.models.spectrogram_models.detr.detr import rtdetr_l

__all__ = ["detr_inference", "evaluate_detr"]


def _validate_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def _format_result(result: Any) -> dict[str, Any]:
    boxes = result.boxes
    return {
        "path": str(result.path),
        "boxes": boxes.xywhn.detach().cpu(),
        "scores": boxes.conf.detach().cpu(),
        "labels": boxes.cls.to(dtype=torch.int64).detach().cpu(),
        "class_names": dict(result.names),
    }


def detr_inference(
    source: str | Path,
    checkpoint_path: str | Path,
    *,
    confidence_threshold: float = 0.25,
    image_size: int = 512,
    batch_size: int = 4,
    device: str | int | None = None,
    workers: int = 1,
    save: bool = False,
    output_dir: str | Path = "runs/predict",
) -> list[dict[str, Any]]:
    """Run RT-DETR inference and return normalized multi-signal detections.

    Args:
        source: Image, directory, glob, or other source accepted by Ultralytics.
        checkpoint_path: Trained RT-DETR ``.pt`` checkpoint.
        confidence_threshold: Minimum retained detection confidence.
        image_size: Square inference image size.
        batch_size: Inference batch size.
        device: Ultralytics device selection, such as ``0`` or ``"cpu"``.
        workers: Number of inference data-loading workers.
        save: Save annotated prediction images.
        output_dir: Parent directory for saved prediction artifacts.

    Returns:
        One dictionary per image containing normalized ``xywh`` boxes, scores,
        integer labels, source path, and checkpoint class names.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    checkpoint = _validate_checkpoint(checkpoint_path)
    model = rtdetr_l(path=checkpoint)
    results = model.predict(
        source=str(source),
        conf=confidence_threshold,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=workers,
        save=save,
        project=str(Path(output_dir).resolve()),
        name="predict",
        exist_ok=True,
        verbose=False,
    )
    return [_format_result(result) for result in results]


def evaluate_detr(
    dataset_yaml: str | Path,
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    image_size: int = 512,
    batch_size: int = 4,
    device: str | int | None = None,
    workers: int = 1,
    output_dir: str | Path = "runs/val",
) -> dict[str, float]:
    """Evaluate a trained RT-DETR checkpoint on an exported dataset split."""
    checkpoint = _validate_checkpoint(checkpoint_path)
    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    model = rtdetr_l(path=checkpoint)
    metrics = model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=workers,
        project=str(Path(output_dir).resolve()),
        name=split,
        exist_ok=True,
    )
    return {
        "map": float(metrics.box.map),
        "map_50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def _parse_device(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    """Parse RT-DETR inference and evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", help="Image source for prediction.")
    mode.add_argument("--dataset-yaml", type=Path, help="Dataset YAML to evaluate.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=_parse_device)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run inference or dataset evaluation from the command line."""
    args = parse_args()
    if args.dataset_yaml is not None:
        summary = evaluate_detr(
            args.dataset_yaml,
            args.checkpoint,
            split=args.split,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=args.device,
            workers=args.workers,
            output_dir=args.output_dir,
        )
        print(json.dumps(summary, indent=2))
        return
    detections = detr_inference(
        args.source,
        args.checkpoint,
        confidence_threshold=args.confidence_threshold,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
        save=args.save,
        output_dir=args.output_dir,
    )
    print(f"Processed {len(detections)} images")


if __name__ == "__main__":
    main()
