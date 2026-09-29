"""Compare EfficientNet spectrogram normalization strategies on TorchSig data.

The default experiment is sized for a laptop GPU: it trains EfficientNet-B0
three times on the same 2,400-example impaired narrowband dataset. Validation
uses the training noise-power distribution, while the test split applies a
-10 dB absolute-power shift at the same SNR range. This measures both ordinary
classification performance and sensitivity to an operational power shift.

Run from the repository root::

    python examples/compare_efficientnet_normalization.py

For a quicker smoke test::

    python examples/compare_efficientnet_normalization.py \
        --train-size 600 --eval-size 180 --epochs 2
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
from pathlib import Path
import time
from typing import Any

import torchsig
import yaml
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.yaml import load_config_from_yaml

from torchsig_models.models.spectrogram_models.efficientnet.efficientnet_train import (
    train_efficientnet_2d,
)
from torchsig_models.utils.datasets import prepare_torchsig_datasets


NORMALIZATION_MODES = ("none", "sample", "dataset")
DEFAULT_SIGNALS = (
    "bpsk",
    "qpsk",
    "8psk",
    "16qam",
    "64qam",
    "2fsk",
    "4gfsk",
    "ofdm-64",
    "am-dsb",
    "fm",
    "lfm-radar",
    "80211a",
)


def _default_dataset_config() -> Path:
    """Return TorchSig's installed impaired narrowband configuration."""
    return (
        Path(torchsig.__file__).parent
        / "datasets"
        / "default_configs"
        / "narrowband_impaired_train_all.yaml"
    )


def _build_split_configs(
    base_cfg: Any,
    *,
    dataset_id: str,
    train_size: int,
    eval_size: int,
    seed: int,
    test_power_shift_db: float,
) -> tuple[Any, Any, Any]:
    """Create fixed train, validation, and shifted-test configurations."""
    train_cfg = replace(
        base_cfg,
        dataset_id=dataset_id,
        dataset_length=train_size,
        seed=seed,
    )
    val_cfg = replace(
        base_cfg,
        dataset_id=dataset_id,
        dataset_length=eval_size,
        seed=seed + 1,
    )
    test_metadata = dict(base_cfg.dataset_metadata)
    test_metadata["noise_power_db"] = (
        float(test_metadata.get("noise_power_db", 0.0)) + test_power_shift_db
    )
    test_cfg = replace(
        base_cfg,
        dataset_id=dataset_id,
        dataset_length=eval_size,
        seed=seed + 2,
        dataset_metadata=test_metadata,
    )
    return train_cfg, val_cfg, test_cfg


def _metric(history: dict[str, list[float]], name: str) -> float:
    """Return the final value from a classifier metric history."""
    values = history[name]
    if not values:
        raise ValueError(f"Metric history {name!r} is empty.")
    return float(values[-1])


def _result_row(
    mode: str,
    result: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Convert one training result into a serializable comparison row."""
    validation = result["metrics"].val_metrics.history
    shifted_test = result["test_metrics"].history
    normalization = result["normalization"]
    return {
        "normalization": mode,
        "val_loss": _metric(validation, "loss"),
        "val_accuracy": _metric(validation, "accuracy"),
        "val_macro_f1": _metric(validation, "f1 score"),
        "shifted_test_loss": _metric(shifted_test, "loss"),
        "shifted_test_accuracy": _metric(shifted_test, "accuracy"),
        "shifted_test_macro_f1": _metric(shifted_test, "f1 score"),
        "elapsed_seconds": elapsed_seconds,
        "normalization_mean": normalization.get("mean"),
        "normalization_std": normalization.get("std"),
    }


def _write_results(
    rows: list[dict[str, Any]],
    output_dir: Path,
    experiment: dict[str, Any],
) -> None:
    """Write compact CSV results and a complete YAML record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "normalization",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "shifted_test_loss",
        "shifted_test_accuracy",
        "shifted_test_macro_f1",
        "elapsed_seconds",
    ]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "experiment.yaml").open("w", encoding="utf-8") as output:
        yaml.safe_dump({"experiment": experiment, "results": rows}, output, sort_keys=False)


def parse_args() -> argparse.Namespace:
    """Parse experiment command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=_default_dataset_config())
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/normalization_comparison"),
    )
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--eval-size", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2501)
    parser.add_argument("--test-power-shift-db", type=float, default=-10.0)
    parser.add_argument("--signal-generators", nargs="+", default=list(DEFAULT_SIGNALS))
    parser.add_argument("--accelerator", default="auto", choices=["auto", "cpu", "gpu", "mps"])
    parser.add_argument("--devices", default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate the shared dataset before running the comparison.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate shared data and run the controlled normalization comparison."""
    args = parse_args()
    if min(args.train_size, args.eval_size, args.epochs, args.batch_size) < 1:
        raise ValueError("Dataset sizes, epochs, and batch size must be positive.")

    signal_signature = hashlib.sha1(
        ",".join(args.signal_generators).encode("utf-8")
    ).hexdigest()[:8]
    shift_label = f"{args.test_power_shift_db:+g}".replace("+", "p").replace("-", "m")
    dataset_id = (
        f"efficientnet_normalization_seed_{args.seed}_"
        f"n{args.train_size}_{args.eval_size}_{signal_signature}_shift{shift_label}db"
    )
    base_cfg = load_config_from_yaml(args.dataset_config)
    train_cfg, val_cfg, test_cfg = _build_split_configs(
        base_cfg,
        dataset_id=dataset_id,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
        test_power_shift_db=args.test_power_shift_db,
    )
    fft_size = int(train_cfg.dataset_metadata.get("fft_size", 64))

    # Materialize all splits once. Every mode then reads exactly the same
    # examples, eliminating dataset generation as an experimental variable.
    prepare_torchsig_datasets(
        train_cfg,
        val_cfg,
        test_cfg,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        signal_generators=args.signal_generators,
        transforms=[Spectrogram(fft_size=fft_size)],
    )

    base_params = {
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": 1e-3,
        "weight_decay": 3e-6,
        "drop_path": 0.2,
        "drop_rate": 0.2,
        "label_smoothing": 0.03,
        "normalization_eps": 1e-6,
    }
    devices: int | str = int(args.devices) if args.devices.isdigit() else args.devices
    rows: list[dict[str, Any]] = []
    for mode in NORMALIZATION_MODES:
        print(f"\n{'=' * 72}\nTraining normalization={mode!r}\n{'=' * 72}")
        params = {**base_params, "normalization": mode}
        run_dir = args.output_dir / mode
        started = time.perf_counter()
        result = train_efficientnet_2d(
            train_cfg,
            val_cfg,
            test_cfg,
            params,
            run_dir / "checkpoints",
            metrics_dir=run_dir / "metrics",
            dataset_root=args.dataset_root,
            overwrite=False,
            model_name="efficientnet_b0",
            signal_generators=args.signal_generators,
            accelerator=args.accelerator,
            devices=devices,
        )
        row = _result_row(mode, result, time.perf_counter() - started)
        rows.append(row)
        print(
            f"{mode}: val F1={row['val_macro_f1']:.4f}, "
            f"shifted-test F1={row['shifted_test_macro_f1']:.4f}"
        )

    experiment = {
        "model": "efficientnet_b0",
        "dataset_config": str(args.dataset_config),
        "dataset_id": dataset_id,
        "train_size": args.train_size,
        "validation_size": args.eval_size,
        "shifted_test_size": args.eval_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "fft_size": fft_size,
        "test_power_shift_db": args.test_power_shift_db,
        "signal_generators": args.signal_generators,
    }
    _write_results(rows, args.output_dir, experiment)
    print(f"\nComparison written to {(args.output_dir / 'comparison.csv').resolve()}")


if __name__ == "__main__":
    main()
