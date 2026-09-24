"""Train and reload XCiT-1D on a small synthetic IQ dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch

from torchsig_models.models import XCiTClassifier

from toy_classification_utils import (
    CLASS_NAMES,
    demonstrate_prediction,
    make_iq_dataset,
    train_steps,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/toy_models"))
    return parser.parse_args()


def main() -> None:
    """Generate data, train, save, reload, and label one prediction."""
    args = parse_args()
    dataset = make_iq_dataset(args.samples_per_class, args.length, seed=7)
    model_kwargs = {
        "input_channels": 2,
        "num_classes": len(CLASS_NAMES),
        "xcit_version": "nano_12_p16_224",
        "ds_rate": 8,
    }
    model = XCiTClassifier(**model_kwargs, class_names=CLASS_NAMES)
    for parameter in model.model.backbone.parameters():
        parameter.requires_grad = False
    train_steps(
        model,
        dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=3e-3,
    )
    checkpoint_path = args.output_dir / "xcit_toy.ckpt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        checkpoint_path,
    )
    loaded_model = XCiTClassifier.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    demonstrate_prediction(loaded_model, dataset)
    print(f"checkpoint: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
