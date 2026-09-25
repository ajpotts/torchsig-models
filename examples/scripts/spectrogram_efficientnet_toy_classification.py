"""Train and reload any 2D EfficientNet on toy spectrogram data."""

from __future__ import annotations

import argparse
from pathlib import Path

from torchsig_models.models.spectrogram_models.efficientnet import (
    efficientnet_b0,
    efficientnet_b2,
    efficientnet_b4,
)

from toy_classification_utils import (
    CLASS_NAMES,
    demonstrate_prediction,
    make_spectrogram_dataset,
    reload_checkpoint,
    save_checkpoint,
    train_steps,
)

MODEL_FACTORIES = {
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b4": efficientnet_b4,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_FACTORIES, default="efficientnet_b0")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/toy_models"))
    return parser.parse_args()


def main() -> None:
    """Generate data, train, save, reload, and label one prediction."""
    args = parse_args()
    dataset = make_spectrogram_dataset(args.samples_per_class, args.size, seed=7)
    factory = MODEL_FACTORIES[args.model]
    model_kwargs = {
        "num_classes": len(CLASS_NAMES),
        "drop_path_rate": 0.0,
        "drop_rate": 0.0,
        "normalize": True,
    }
    model = factory(**model_kwargs, class_names=CLASS_NAMES)
    train_steps(
        model,
        dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=3e-3,
    )
    checkpoint_path = args.output_dir / f"spectrogram_{args.model}_toy.ckpt"
    save_checkpoint(
        checkpoint_path,
        model,
        model_name=args.model,
        model_kwargs=model_kwargs,
    )
    loaded_model = reload_checkpoint(checkpoint_path, factory)
    demonstrate_prediction(loaded_model, dataset)
    print(f"checkpoint: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
