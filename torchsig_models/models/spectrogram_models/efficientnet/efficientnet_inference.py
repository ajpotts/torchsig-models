"""Inference entry point for EfficientNet-2D TorchSig classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from torchsig_models.models.spectrogram_models.efficientnet.efficientnet_train import (
    EfficientNet2DModelName,
    MODEL_FACTORY,
    load_training_params,
)
from torchsig_models.utils.datasets import (
    prepare_torchsig_inference_dataset,
)
from torchsig_models.utils.training import evaluate_classifier


__all__ = [
    "efficientnet_inference",
    "parse_args",
]


def _strip_lightning_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove Lightning's wrapped-model prefix from checkpoint keys.

    Args:
        state_dict: Model state dictionary loaded from a checkpoint.

    Returns:
        State dictionary with a leading ``"model."`` removed from keys when
        present.
    """
    if not any(key.startswith("model.") for key in state_dict):
        return state_dict

    return {key.removeprefix("model."): value for key, value in state_dict.items()}


def efficientnet_inference(
    root: str | Path,
    checkpoint_path: str | Path,
    *,
    params_path: str | Path | None = None,
    batch_size: int = 4,
    num_workers: int = 8,
    num_classes: int = 57,
    model_name: EfficientNet2DModelName = "efficientnet_b4",
) -> float:
    """Evaluate an EfficientNet-2D model on a static TorchSig dataset.

    Args:
        root: Root directory of the static TorchSig test dataset.
        checkpoint_path: Path to a Lightning or PyTorch checkpoint.
        params_path: Optional path to the model training-parameter YAML.
        batch_size: Evaluation batch size.
        num_workers: Number of dataloader worker processes.
        num_classes: Number of classifier output classes.
        model_name: EfficientNet architecture represented by the checkpoint.

    Returns:
        Final test-set classification accuracy.

    Raises:
        FileNotFoundError: If the dataset root or checkpoint does not exist.
    """
    root = Path(root)
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    params = load_training_params(
        model_name,
        params_path=params_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODEL_FACTORY[model_name](
        num_classes=num_classes,
        drop_path_rate=params.get("drop_path", 0.2),
        drop_rate=params.get("drop_rate", 0.3),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get(
        "state_dict",
        checkpoint,
    )
    state_dict = _strip_lightning_prefix(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing_keys:
        print(f"Missing checkpoint keys: {missing_keys}")

    if unexpected_keys:
        print(f"Unexpected checkpoint keys: {unexpected_keys}")

    model.to(device)
    model.eval()

    test_loader = prepare_torchsig_inference_dataset(
        root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    tracker = evaluate_classifier(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
    )

    accuracy = float(tracker.history["accuracy"][-1])
    print(f"\nTest accuracy: {accuracy:.4%}")

    return accuracy


def parse_args() -> argparse.Namespace:
    """Parse EfficientNet-2D inference command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained EfficientNet-2D checkpoint on a static "
            "TorchSig spectrogram dataset."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory of the static TorchSig test dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a Lightning .ckpt or PyTorch .pth checkpoint.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_FACTORY),
        default="efficientnet_b4",
        help="EfficientNet architecture to instantiate.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        help=("Training params YAML. Defaults to training_params/<model>.yaml."),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=57,
        help="Number of output classes.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    efficientnet_inference(
        root=args.root,
        checkpoint_path=args.checkpoint,
        params_path=args.params,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        model_name=args.model,
    )
