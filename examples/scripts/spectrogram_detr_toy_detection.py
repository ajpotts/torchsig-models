"""Train DETR on a small, generated TorchSig wideband detection problem.

TorchSig generates mixtures of tones and LFM radar signals, converts the IQ
samples to spectrograms, and supplies normalized YOLO boxes. This example then
adapts those boxes to DETR's target format and trains TorchSig Models'
DETR-B0-Nano for a few steps.

Example:
    python examples/scripts/spectrogram_detr_toy_detection.py --steps 10
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults

from torchsig_models.models.spectrogram_models.detr import detr_b0_nano
from torchsig_models.models.spectrogram_models.detr.modules import SetCriterion
from torchsig_models.models.spectrogram_models.detr.utils import format_preds
from torchsig_models.models.spectrogram_models.efficientnet.efficientnet import (
    SpectrogramNormalization,
)


SIGNAL_GENERATORS = ["tone", "lfm-radar"]


def parse_args() -> argparse.Namespace:
    """Parse example command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fft-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/spectrogram_detr_toy/detr_b0_nano.pt"),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


def make_dataset(fft_size: int) -> TorchSigIterableDataset:
    """Create an endless TorchSig wideband detection dataset."""
    metadata = {
        **TorchSigDefaults().default_dataset_metadata,
        "sample_rate": 10_000_000,
        "num_iq_samples_dataset": fft_size**2,
        "fft_size": fft_size,
        "fft_stride": fft_size,
        "num_signals_min": 1,
        "num_signals_max": 2,
        "cochannel_overlap_probability": 0.0,
        "snr_db_min": 20.0,
        "snr_db_max": 35.0,
        "noise_power_db": 0.0,
        "signal_duration_in_samples_min": fft_size**2 // 4,
        "signal_duration_in_samples_max": fft_size**2,
        "bandwidth_min": 500_000,
        "bandwidth_max": 2_000_000,
        "signal_center_freq_min": -3_500_000,
        "signal_center_freq_max": 3_500_000,
        "frequency_min": -5_000_000,
        "frequency_max": 4_999_999,
    }
    return TorchSigIterableDataset(
        metadata=metadata,
        transforms=[Spectrogram(fft_size=fft_size), YOLOLabel()],
        target_labels=["yolo_label"],
        signal_generators=SIGNAL_GENERATORS,
    )


def detr_collate(
    batch: Sequence[tuple[np.ndarray, Iterable[Sequence[float]]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Convert TorchSig spectrograms and YOLO labels into a DETR batch."""
    spectrograms, yolo_targets = zip(*batch)
    images = torch.from_numpy(np.stack(spectrograms)).float().unsqueeze(1)

    # TorchSig's Spectrogram is a single-channel magnitude image. The current
    # TorchSig Models DETR backbone accepts two channels, so expose that image
    # in both channels before applying the shared spectrogram normalization.
    images = images.repeat(1, 2, 1, 1)
    images = SpectrogramNormalization()(images)

    targets: list[dict[str, torch.Tensor]] = []
    for objects in yolo_targets:
        target = torch.as_tensor(list(objects), dtype=torch.float32).reshape(-1, 5)
        targets.append(
            {
                "labels": target[:, 0].to(dtype=torch.int64),
                "boxes": target[:, 1:],
            }
        )
    return images, targets


def make_loader(fft_size: int, batch_size: int, seed: int) -> DataLoader[Any]:
    """Build a deterministic TorchSig loader for generated examples."""
    loader = WorkerSeedingDataLoader(
        make_dataset(fft_size),
        batch_size=batch_size,
        collate_fn=detr_collate,
        seed=seed,
        num_workers=0,
    )
    loader.seed(seed)
    return loader


def weighted_loss(
    losses: dict[str, torch.Tensor], criterion: SetCriterion
) -> torch.Tensor:
    """Combine the DETR losses using the criterion's configured weights."""
    return sum(
        losses[name] * weight
        for name, weight in criterion.weight_dict.items()
        if name in losses
    )


def select_device(requested: str) -> torch.device:
    """Resolve the requested training device."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def main() -> None:
    """Generate TorchSig data, train DETR, and run one inference batch."""
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.fft_size < 32:
        raise ValueError("--fft-size must be at least 32")

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    loader = make_loader(args.fft_size, args.batch_size, args.seed)
    model = detr_b0_nano(
        num_classes=len(SIGNAL_GENERATORS),
        drop_rate_backbone=0.0,
        drop_path_rate_backbone=0.0,
        drop_path_rate_transformer=0.0,
    ).to(device)
    criterion = SetCriterion(num_classes=len(SIGNAL_GENERATORS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    batches = iter(loader)
    for step in range(1, args.steps + 1):
        images, targets = next(batches)
        images = images.to(device)
        targets = [
            {name: value.to(device) for name, value in target.items()}
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        losses = criterion(outputs, targets)
        loss = weighted_loss(losses, criterion)
        loss.backward()
        optimizer.step()

        components = ", ".join(
            f"{name}={value.detach().item():.4f}"
            for name, value in losses.items()
            if name in criterion.weight_dict
        )
        print(f"step {step:03d}: total={loss.detach().item():.4f}, {components}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)

    model.eval()
    images, targets = next(batches)
    with torch.no_grad():
        outputs = model(images.to(device))
    predictions = format_preds(outputs)

    print("\nDETR example complete")
    print(f"Device: {device}")
    print(f"Signals: {', '.join(SIGNAL_GENERATORS)}")
    print(f"Input shape: {tuple(images.shape)}")
    print(f"Ground-truth objects: {[len(target['labels']) for target in targets]}")
    print(f"Detections at confidence >= 0.5: {[len(pred['labels']) for pred in predictions]}")
    print(f"Checkpoint: {args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
