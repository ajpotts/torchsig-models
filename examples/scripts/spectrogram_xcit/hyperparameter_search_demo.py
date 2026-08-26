"""Run the production XCiT search on a small two-class TorchSig dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

from torchsig_models.models.spectrogram_models.xcit.xcit_hyperparameter_search import (
    main as run_hyperparameter_search,
)


SIGNAL_GENERATORS = ["tone", "lfm-radar"]


def parse_args() -> argparse.Namespace:
    """Parse demo options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--dataset-length", type=int, default=64)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/xcit_search_demo")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_demo_configs(output_dir: Path, trials: int) -> tuple[Path, Path]:
    """Write small TorchSig dataset and Optuna search configurations."""
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = {
        "schema_version": "2.1.1",
        "dataset_id": "xcit_search_demo",
        "dataset_length": 64,
        "seed": 8101,
        "impairment_level": 0,
        "output": {"representation": "spectrogram", "spectrogram_fft": 64},
        "signal_sampling": {"mode": "per_signal"},
        "dataset_metadata": {
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
        },
        "target_labels": ["class_index"],
    }
    search_config = {
        "experiment_name": "xcit_search_demo",
        "run_name": "tone_vs_lfm_radar",
        "direction": "maximize",
        "metric_name": "val_f1",
        "n_trials": trials,
        "search_space": {
            "learning_rate": {
                "type": "float",
                "low": 1e-4,
                "high": 2e-3,
                "log": True,
            },
            "weight_decay": {
                "type": "float",
                "low": 1e-6,
                "high": 1e-3,
                "log": True,
            },
            "drop_rate": {
                "type": "categorical",
                "choices": [0.0, 0.1, 0.2],
            },
            "batch_size": {"type": "categorical", "choices": [8, 16]},
        },
    }
    dataset_path = config_dir / "dataset.yaml"
    search_path = config_dir / "search.yaml"
    dataset_path.write_text(yaml.safe_dump(dataset_config), encoding="utf-8")
    search_path.write_text(yaml.safe_dump(search_config), encoding="utf-8")
    return dataset_path, search_path


def main() -> None:
    """Create demo configs and invoke the production search entry point."""
    args = parse_args()
    dataset_config, search_config = write_demo_configs(args.output_dir, args.trials)
    search_args = [
        "xcit_hyperparameter_search.py",
        "--dataset-config",
        str(dataset_config),
        "--search-config",
        str(search_config),
        "--dataset-root",
        str(args.output_dir / "datasets"),
        "--output-dir",
        str(args.output_dir / "search"),
        "--dataset-length",
        str(args.dataset_length),
        "--n-trials",
        str(args.trials),
        "--max-epochs",
        str(args.epochs),
        "--signal-generators",
        *SIGNAL_GENERATORS,
    ]
    if args.overwrite:
        search_args.append("--overwrite")
    sys.argv = search_args
    run_hyperparameter_search()


if __name__ == "__main__":
    main()
