"""Train XCiT-Nano on an easy two-class TorchSig spectrogram dataset.

TorchSig generates stationary tones and LFM radar signals. The example then
uses the public ``train_xcit_2d`` interface for dataset preparation, training,
evaluation, metrics, and checkpoints.

Example:
    python examples/spectrogram_xcit/easy_classification.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from torchsig.datasets.datasets import TorchSigDatasetConfig
from torchsig.utils.defaults import TorchSigDefaults

from torchsig_models.models.spectrogram_models.xcit.xcit_train import (
    train_xcit_2d,
)


SIGNAL_GENERATORS = ["tone", "lfm-radar"]


def parse_args() -> argparse.Namespace:
    """Parse example command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/spectrogram_xcit_easy"),
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
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "gpu", "mps"),
        default="auto",
    )
    parser.add_argument("--devices", default="auto")
    return parser.parse_args()


def make_config(dataset_length: int, seed: int) -> TorchSigDatasetConfig:
    """Create a high-SNR, single-signal spectrogram configuration."""
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
        dataset_id="spectrogram_xcit_easy",
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
    """Generate TorchSig data, train XCiT-Nano, and report metrics."""
    args = parse_args()
    devices = int(args.devices) if args.devices.isdigit() else args.devices
    params = {
        "input_channels": 1,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": 5e-4,
        "weight_decay": 5e-5,
        "drop_path": 0.0,
        "drop_rate": 0.0,
        "label_smoothing": 0.0,
    }
    result = train_xcit_2d(
        train_cfg=make_config(args.train_samples, 7101),
        val_cfg=make_config(args.val_samples, 7102),
        test_cfg=make_config(args.test_samples, 7103),
        params=params,
        checkpoint_dir=args.output_dir / "checkpoints",
        metrics_dir=args.output_dir / "metrics",
        dataset_root=args.output_dir / "datasets",
        overwrite=args.overwrite,
        signal_generators=SIGNAL_GENERATORS,
        logger=False,
        accelerator=args.accelerator,
        devices=devices,
    )

    test_dataset = result["test_loader"].dataset
    label_counts = Counter(int(test_dataset[index][1]) for index in range(len(test_dataset)))
    majority_baseline = max(label_counts.values()) / len(test_dataset)
    print("\nXCiT easy classification results")
    print(f"Signals: {', '.join(SIGNAL_GENERATORS)}")
    print(f"Test label counts: {dict(label_counts)}")
    print(f"Uniform baseline: {1 / len(SIGNAL_GENERATORS):.2%}")
    print(f"Majority-class baseline: {majority_baseline:.2%}")
    print(f"Final validation accuracy: {result['metrics'].val_accuracies[-1]:.2%}")
    print(f"Final test accuracy: {result['test_metrics'].history['accuracy'][-1]:.2%}")
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
