"""Inference utilities for EfficientNet-1D models trained on TorchSig datasets.

This module loads a trained EfficientNet-1D checkpoint and evaluates it on a
static TorchSig dataset. It reconstructs the model using the original training
configuration, restores the checkpoint weights, and reports classification
accuracy on the evaluation dataset.

The module can be imported as a library or executed as a command-line script.
"""

from pathlib import Path

import argparse
import torch
from lightning_fabric.utilities.seed import seed_everything

from torchsig.datasets.datasets import StaticTorchSigDataset
from torchsig.utils.data_loading import WorkerSeedingDataLoader

from torchsig_models.models.iq_models.efficientnet.efficientnet1d_train import (
    EfficientNetModelName,
    MODEL_FACTORY,
    load_training_params,
)
from torchsig_models.utils.training import evaluate_classifier, configure_determinism


def _strip_lightning_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove Lightning's wrapped-model prefix from checkpoint keys."""
    if not any(key.startswith("model.") for key in state_dict):
        return state_dict

    return {key.removeprefix("model."): value for key, value in state_dict.items()}


def _resolve_num_classes(
    state_dict: dict[str, torch.Tensor],
    num_classes: int | None,
) -> int:
    """Resolve and validate the class count against checkpoint weights."""
    classifier_weights = [
        value for key, value in state_dict.items() if key.endswith("classifier.weight")
    ]
    if len(classifier_weights) != 1:
        if num_classes is None:
            raise ValueError(
                "Could not infer num_classes from a unique classifier.weight "
                "checkpoint tensor; pass num_classes explicitly."
            )
        return num_classes

    checkpoint_num_classes = int(classifier_weights[0].shape[0])
    if num_classes is not None and num_classes != checkpoint_num_classes:
        raise ValueError(
            f"num_classes={num_classes} does not match the checkpoint classifier "
            f"size ({checkpoint_num_classes})."
        )
    return checkpoint_num_classes


def efficientnet1d_inference(
    root: str | Path,
    checkpoint_path: str | Path,
    *,
    params_path: str | Path | None = None,
    batch_size: int = 4,
    num_workers: int = 8,
    num_classes: int | None = None,
    model_name: EfficientNetModelName = "efficientnet_b4",
) -> float:
    """Evaluate a trained EfficientNet-1D model on a static TorchSig dataset.

    The model architecture is reconstructed from the specified training
    parameters, the checkpoint weights are restored, and inference is performed
    on a ``StaticTorchSigDataset``. Reproducibility settings are configured
    before evaluation.

    Args:
        root: Root directory containing the static TorchSig dataset.
        checkpoint_path: Path to a Lightning ``.ckpt`` or PyTorch ``.pth``
            checkpoint.
        params_path: Optional path to a training parameter YAML file. If
            omitted, the default parameter file for the selected model is used.
        batch_size: Number of samples per evaluation batch.
        num_workers: Number of data loader worker processes.
        num_classes: Optional output class count. By default, this is inferred
            from the checkpoint classifier weights.
        model_name: EfficientNet architecture to instantiate.

    Returns:
        Classification accuracy on the evaluation dataset.
    """
    root = Path(root)
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    params = load_training_params(
        model_name,
        params_path=params_path,
    )
    seed = params.get("seed", 42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _strip_lightning_prefix(state_dict)
    num_classes = _resolve_num_classes(state_dict, num_classes)

    model = MODEL_FACTORY[model_name](
        num_classes=num_classes,
        drop_path_rate=params.get("drop_path", 0.2),
        drop_rate=params.get("drop_rate", 0.3),
    )

    seed_everything(seed, workers=True)
    configure_determinism()

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if missing_keys:
        print(f"Missing checkpoint keys: {missing_keys}")

    if unexpected_keys:
        print(f"Unexpected checkpoint keys: {unexpected_keys}")

    model.to(device)
    model.eval()

    test_dataset = StaticTorchSigDataset(
        root=str(root),
        target_labels=["class_index"],
    )

    test_loader = WorkerSeedingDataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
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
    """Parse command-line arguments for EfficientNet-1D inference.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained EfficientNet-1D checkpoint on a static TorchSig dataset."
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
        choices=["efficientnet_b0", "efficientnet_b2", "efficientnet_b4"],
        default="efficientnet_b4",
        help="EfficientNet architecture to instantiate.",
    )

    parser.add_argument(
        "--params",
        type=Path,
        help="Training params YAML. Defaults to training_params/<model>.yaml.",
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
        help="Output classes. Defaults to the checkpoint classifier size.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    efficientnet1d_inference(
        root=args.root,
        checkpoint_path=args.checkpoint,
        params_path=args.params,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        model_name=args.model,
    )
