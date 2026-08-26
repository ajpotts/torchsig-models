"""Inference entry point for XCiT spectrogram classifiers."""

import argparse
from pathlib import Path

import torch

from torchsig_models.models.spectrogram_models.xcit import xcit_nano
from torchsig_models.models.spectrogram_models.xcit.xcit_train import (
    load_training_params,
)
from torchsig_models.utils.datasets import prepare_torchsig_inference_dataset
from torchsig_models.utils.training import evaluate_classifier

__all__ = ["xcit_inference"]


def xcit_inference(
    root: str | Path,
    checkpoint_path: str | Path,
    *,
    params_path: str | Path | None = None,
    batch_size: int = 4,
    num_workers: int = 8,
    num_classes: int = 72,
) -> float:
    """Evaluate XCiT-Nano on a static TorchSig spectrogram dataset.

    Args:
        root: Static TorchSig test-dataset directory.
        checkpoint_path: Lightning or PyTorch model checkpoint.
        params_path: Optional training-parameter YAML.
        batch_size: Evaluation batch size.
        num_workers: Data-loader worker count.
        num_classes: Number of classifier output classes.

    Returns:
        Final classification accuracy.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    params = load_training_params(params_path)
    model = xcit_nano(
        num_classes=num_classes,
        input_channels=params.get("input_channels", 1),
        drop_path_rate=params.get("drop_path", 0.2),
        drop_rate=params.get("drop_rate", 0.3),
        checkpoint_path=checkpoint_path,
        normalize=params.get("normalize", False),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    test_loader = prepare_torchsig_inference_dataset(
        root, batch_size=batch_size, num_workers=num_workers
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
    """Parse XCiT inference command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate XCiT-Nano on a static TorchSig dataset."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=72)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    xcit_inference(
        root=args.root,
        checkpoint_path=args.checkpoint,
        params_path=args.params,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
    )
