"""Inference entry point for DETR wideband spectrogram detectors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from torchsig_models.models.spectrogram_models.detr.detr_train import (
    DETRModelName,
    MODEL_FACTORY,
    detr_collate,
)
from torchsig_models.utils.datasets import prepare_torchsig_inference_dataset
from torchsig_models.utils.normalization import (
    NormalizationMode,
    normalization_from_state_dict,
    resolve_checkpoint_normalization_mode,
)

__all__ = ["detr_inference", "postprocess_detections"]


def _strip_lightning_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove the Lightning wrapper's ``model.`` prefix."""
    if not any(key.startswith("model.") for key in state_dict):
        return state_dict
    return {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }


def _checkpoint_metadata(checkpoint: dict[str, Any]) -> tuple[str | None, int | None, dict[str, Any]]:
    hyperparameters = checkpoint.get("hyper_parameters", {})
    model_name = hyperparameters.get("model_name")
    num_classes = hyperparameters.get("num_classes")
    model_params = hyperparameters.get("model_params") or {}
    return model_name, num_classes, model_params


def _infer_num_classes(state_dict: dict[str, torch.Tensor]) -> int:
    weights = [value for key, value in state_dict.items() if key.endswith("linear_class.weight")]
    if len(weights) != 1:
        raise ValueError("Could not infer num_classes from checkpoint linear_class.weight.")
    return int(weights[0].shape[0]) - 1


def postprocess_detections(
    outputs: dict[str, torch.Tensor], confidence_threshold: float = 0.5
) -> list[dict[str, torch.Tensor]]:
    """Convert raw DETR outputs to normalized multi-signal detections."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    probabilities = outputs["pred_logits"].softmax(dim=-1)
    scores, labels = probabilities[..., :-1].max(dim=-1)
    results = []
    for boxes, image_scores, image_labels in zip(outputs["pred_boxes"], scores, labels):
        keep = image_scores >= confidence_threshold
        results.append(
            {
                "boxes": boxes[keep].detach().cpu(),
                "scores": image_scores[keep].detach().cpu(),
                "labels": image_labels[keep].detach().cpu(),
            }
        )
    return results


def detr_inference(
    root: str | Path,
    checkpoint_path: str | Path,
    *,
    batch_size: int = 4,
    num_workers: int = 8,
    num_classes: int | None = None,
    model_name: DETRModelName | None = None,
    confidence_threshold: float = 0.5,
    output_path: str | Path | None = None,
    normalization: NormalizationMode | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Run DETR over a static TorchSig dataset and return every detection.

    The checkpoint's normalization metadata is used by default. Set
    ``normalization`` to explicitly override it; legacy DETR checkpoints fall
    back to the per-sample normalization used by the former collate function.
    """
    root, checkpoint_path = Path(root), Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _strip_lightning_prefix(checkpoint.get("state_dict", checkpoint))
    resolved_normalization = resolve_checkpoint_normalization_mode(
        checkpoint,
        normalization,
    )
    hyperparameters = checkpoint.get("hyper_parameters", {})
    normalization_metadata = hyperparameters.get("normalization", {})
    normalization_eps = (
        float(normalization_metadata.get("eps", 1e-6))
        if isinstance(normalization_metadata, dict)
        else 1e-6
    )
    normalization_kwargs = normalization_from_state_dict(
        state_dict,
        resolved_normalization,
        legacy_mode="sample",
        eps=normalization_eps,
    )
    stored_model_name, stored_num_classes, model_params = _checkpoint_metadata(checkpoint)
    if "learned_object_queries" not in model_params:
        model_params = {
            **model_params,
            "learned_object_queries": any(
                key.endswith("transformer.query_embed.weight") for key in state_dict
            ),
        }
    resolved_model_name = model_name or stored_model_name or "detr_b0_nano"
    if resolved_model_name not in MODEL_FACTORY:
        raise ValueError(f"Unsupported DETR model in checkpoint: {resolved_model_name}")
    inferred_num_classes = _infer_num_classes(state_dict)
    resolved_num_classes = num_classes or stored_num_classes or inferred_num_classes
    if int(resolved_num_classes) != inferred_num_classes:
        raise ValueError(
            f"num_classes={resolved_num_classes} does not match checkpoint output size "
            f"({inferred_num_classes})."
        )
    model = MODEL_FACTORY[resolved_model_name](
        num_classes=int(resolved_num_classes),
        **model_params,
        **normalization_kwargs,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    loader = prepare_torchsig_inference_dataset(
        root,
        batch_size=batch_size,
        num_workers=num_workers,
        target_labels=["yolo_label"],
        collate_fn=detr_collate,
    )
    predictions: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for images, _ in loader:
            predictions.extend(
                postprocess_detections(model(images.to(device)), confidence_threshold)
            )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(predictions, output_path)
    return predictions


def parse_args() -> argparse.Namespace:
    """Parse DETR inference command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODEL_FACTORY))
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--normalization", choices=["dataset", "sample", "none"])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detections = detr_inference(
        args.root,
        args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        model_name=args.model,
        confidence_threshold=args.confidence_threshold,
        output_path=args.output,
        normalization=args.normalization,
    )
    print(f"Processed {len(detections)} examples with {sum(len(x['labels']) for x in detections)} detections.")
