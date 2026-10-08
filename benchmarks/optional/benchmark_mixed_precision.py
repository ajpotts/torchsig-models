#!/usr/bin/env python3
"""Benchmark EfficientNet-1D training across Lightning precision modes.

The benchmark uses the real EfficientNet-1D model and shared ``train_validate``
path with a deterministic, learnable synthetic IQ dataset. Run it on the target
deployment GPU to compare fit throughput, peak allocated CUDA memory, and final
validation accuracy without requiring a generated TorchSig dataset.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/torchsig-models-matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchsig_models.models.iq_models.efficientnet.efficientnet1d import (
    efficientnet_b0,
    efficientnet_b2,
    efficientnet_b4,
)
from torchsig_models.utils.training import set_deterministic, train_validate


PRECISIONS = ("32-true", "16-mixed", "bf16-mixed")
MODEL_FACTORIES = {
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b4": efficientnet_b4,
}


@dataclass(frozen=True)
class TrialResult:
    """Measurements from one complete training trial."""

    elapsed_seconds: float
    samples_per_second: float
    peak_memory_bytes: int
    final_validation_accuracy: float


class SyntheticIQDataset(TensorDataset):
    """Deterministic IQ signals whose frequency identifies the class."""

    def __init__(
        self,
        samples: int,
        sample_length: int,
        num_classes: int,
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(samples, dtype=torch.long) % num_classes
        phase = torch.linspace(0, 2 * torch.pi, sample_length, dtype=torch.float32)
        frequency = (labels + 1).to(torch.float32).unsqueeze(1)
        signal_phase = frequency * phase.unsqueeze(0)
        iq = torch.stack((torch.cos(signal_phase), torch.sin(signal_phase)), dim=1)
        noise = torch.randn(iq.shape, generator=generator, dtype=iq.dtype) * 0.05
        super().__init__(iq + noise, labels)


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=PRECISIONS,
        default=list(PRECISIONS),
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_FACTORIES),
        default="efficientnet_b0",
    )
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--sample-length", type=int, default=4096)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for machine-readable benchmark results.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "train samples": args.train_samples,
        "validation samples": args.val_samples,
        "sample length": args.sample_length,
        "number of classes": args.num_classes,
        "batch size": args.batch_size,
        "epochs": args.epochs,
        "runs": args.runs,
    }
    for name, value in positive_values.items():
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}.")
    if args.warmup_runs < 0 or args.num_workers < 0:
        raise ValueError("warmup runs and worker count must be non-negative.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError(
            "learning rate must be positive and weight decay non-negative."
        )


def _make_loader(
    dataset: TensorDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def _run_trial(
    args: argparse.Namespace,
    precision: str,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
) -> TrialResult:
    set_deterministic(args.seed)
    model = MODEL_FACTORIES[args.model](
        num_classes=args.num_classes,
        normalization="none",
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = _make_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
    )
    val_loader = _make_loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    _pl_model, metrics = train_validate(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        max_epochs=args.epochs,
        num_classes=args.num_classes,
        accelerator="gpu",
        devices=1,
        precision=precision,
        enable_progress_bar=False,
        logger=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated()
    samples_processed = len(train_dataset) * args.epochs
    final_accuracy = metrics.val_accuracies[-1]

    return TrialResult(
        elapsed_seconds=elapsed,
        samples_per_second=samples_processed / elapsed,
        peak_memory_bytes=peak_memory,
        final_validation_accuracy=final_accuracy,
    )


def _summarize(trials: list[TrialResult]) -> dict[str, Any]:
    return {
        "trials": [asdict(trial) for trial in trials],
        "median_elapsed_seconds": statistics.median(
            trial.elapsed_seconds for trial in trials
        ),
        "median_samples_per_second": statistics.median(
            trial.samples_per_second for trial in trials
        ),
        "median_peak_memory_bytes": statistics.median(
            trial.peak_memory_bytes for trial in trials
        ),
        "median_final_validation_accuracy": statistics.median(
            trial.final_validation_accuracy for trial in trials
        ),
    }


def _add_relative_results(results: dict[str, dict[str, Any]]) -> None:
    baseline = results.get("32-true")
    if baseline is None or "skipped" in baseline:
        return

    for precision, result in results.items():
        if "skipped" in result:
            continue
        result["relative_to_32_true"] = {
            "throughput_speedup": (
                result["median_samples_per_second"]
                / baseline["median_samples_per_second"]
            ),
            "peak_memory_reduction_fraction": 1.0
            - (
                result["median_peak_memory_bytes"]
                / baseline["median_peak_memory_bytes"]
            ),
            "validation_accuracy_delta": (
                result["median_final_validation_accuracy"]
                - baseline["median_final_validation_accuracy"]
            ),
        }


def main() -> None:
    """Run all requested precision modes and print a JSON summary."""
    args = parse_args()
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-capable GPU.")

    train_dataset = SyntheticIQDataset(
        args.train_samples,
        args.sample_length,
        args.num_classes,
        args.seed,
    )
    val_dataset = SyntheticIQDataset(
        args.val_samples,
        args.sample_length,
        args.num_classes,
        args.seed + 1,
    )

    results: dict[str, dict[str, Any]] = {}
    for precision in args.precisions:
        if precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
            results[precision] = {"skipped": "CUDA device does not support bf16."}
            continue

        for _ in range(args.warmup_runs):
            warmup_result = _run_trial(args, precision, train_dataset, val_dataset)
            del warmup_result
            gc.collect()
            torch.cuda.empty_cache()

        trials = []
        for _ in range(args.runs):
            trials.append(_run_trial(args, precision, train_dataset, val_dataset))
            gc.collect()
            torch.cuda.empty_cache()
        results[precision] = _summarize(trials)

    _add_relative_results(results)
    device = torch.cuda.current_device()
    output = {
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                device
            ).total_memory,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "pytorch_lightning": pl.__version__,
        },
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "json_output"
        },
        "throughput_scope": (
            "training samples divided by total Lightning fit time, including "
            "validation and metric callbacks"
        ),
        "results": results,
    }

    serialized = json.dumps(output, indent=2) + "\n"
    print(serialized, end="")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
