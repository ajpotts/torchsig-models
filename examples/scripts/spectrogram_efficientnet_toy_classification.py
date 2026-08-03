"""Train EfficientNet-B0 on a small two-class TorchSig spectrogram dataset.

This example uses the visually distinct ``tone`` and ``lfm-radar`` signal
generators so a short CPU run can demonstrate learning above chance. Generated
datasets, metrics, and checkpoints are written beneath ``--output-dir``.

Example:
    python examples/scripts/spectrogram_efficientnet_toy_classification.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from torchsig.datasets.datasets import TorchSigDatasetConfig
from torchsig.utils.defaults import TorchSigDefaults

from torchsig_models.models.spectrogram_models.efficientnet.efficientnet_train import (
    train_efficientnet_2d,
)


SIGNAL_GENERATORS = ["tone", "lfm-radar"]


def parse_args() -> argparse.Namespace:
    """Parse example command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a toy TorchSig spectrogram classification experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/spectrogram_effnet_toy"),
        help="Directory for generated datasets, metrics, and checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate static datasets if they already exist.",
    )
    return parser.parse_args()


def make_config(
    *,
    dataset_length: int,
    seed: int,
) -> TorchSigDatasetConfig:
    """Create a high-SNR, single-signal 64x64 spectrogram configuration."""
    metadata = {
        **TorchSigDefaults().default_dataset_metadata,
        "sample_rate": 10_000_000,
        "num_iq_samples_dataset": 4096,
        "fft_size": 64,
        "fft_stride": 64,
        "num_signals_min": 1,
        "num_signals_max": 1,
        "cochannel_overlap_probability": 0.0,
        "snr_db_min": 30.0,
        "snr_db_max": 50.0,
        "noise_power_db": 0.0,
        "signal_duration_in_samples_min": 3600,
        "signal_duration_in_samples_max": 4096,
        "bandwidth_min": 1_500_000,
        "bandwidth_max": 2_000_000,
        "signal_center_freq_min": -2_500_000,
        "signal_center_freq_max": 2_499_999,
        "frequency_min": -2_500_000,
        "frequency_max": 2_499_999,
    }
    return TorchSigDatasetConfig(
        dataset_id="spectrogram_effnet_toy",
        dataset_length=dataset_length,
        seed=seed,
        impairment_level=0,
        output_representation="spectrogram",
        output_spectrogram_fft=64,
        signal_sampling_mode="per_signal",
        dataset_metadata=metadata,
        target_labels=["class_index"],
    )


def main() -> None:
    """Generate data, train EfficientNet-B0, and report baseline metrics."""
    args = parse_args()
    params = {
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": 0.003,
        "weight_decay": 1e-5,
        "drop_path": 0.0,
        "drop_rate": 0.0,
        "label_smoothing": 0.0,
    }

    result = train_efficientnet_2d(
        train_cfg=make_config(dataset_length=args.train_samples, seed=5101),
        val_cfg=make_config(dataset_length=args.val_samples, seed=5102),
        test_cfg=make_config(dataset_length=args.test_samples, seed=5103),
        params=params,
        checkpoint_dir=args.output_dir / "checkpoints",
        metrics_dir=args.output_dir / "metrics",
        dataset_root=args.output_dir / "datasets",
        overwrite=args.overwrite,
        model_name="efficientnet_b0",
        signal_generators=SIGNAL_GENERATORS,
        logger=False,
        accelerator="auto",
        devices="auto",
    )

    test_dataset = result["test_loader"].dataset
    label_counts = Counter(int(test_dataset[index][1]) for index in range(len(test_dataset)))
    majority_baseline = max(label_counts.values()) / len(test_dataset)

    print("\nToy classification results")
    print(f"Signals: {', '.join(SIGNAL_GENERATORS)}")
    print(f"Test label counts: {dict(label_counts)}")
    print(f"Two-class uniform baseline: {1 / len(SIGNAL_GENERATORS):.2%}")
    print(f"Test majority-class baseline: {majority_baseline:.2%}")
    print(f"Final validation accuracy: {result['metrics'].val_accuracies[-1]:.2%}")
    print(f"Final test accuracy: {result['test_metrics'].history['accuracy'][-1]:.2%}")
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
