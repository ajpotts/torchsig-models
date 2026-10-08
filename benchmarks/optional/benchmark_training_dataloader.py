#!/usr/bin/env python3
"""Benchmark TorchSig training-loader throughput across repository branches.

The benchmark replaces static dataset creation with a deterministic synthetic
dataset, then consumes the training loader returned by
``prepare_torchsig_datasets``. On revisions that expose loader performance
options it requests the configured worker count; older revisions automatically
exercise their legacy loader defaults.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

# The benchmark mocks TorchSig dataset generation and does not execute its DSP
# kernels. Disabling JIT avoids editable-install cache failures during import.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/torchsig-models-matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

import torch
from torch.utils.data import Dataset

from torchsig_models.utils.datasets import prepare_torchsig_datasets


class DelayedTensorDataset(Dataset):
    """Synthetic static dataset with configurable per-sample read latency."""

    def __init__(self, length: int, sample_length: int, delay_ms: float) -> None:
        self.length = length
        self.delay_seconds = delay_ms / 1_000.0
        self.sample = torch.arange(
            2 * sample_length,
            dtype=torch.float32,
        ).reshape(2, sample_length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return self.sample, index % 8

    def seed(self, seed: int) -> None:
        """Support deterministic seeding expected by WorkerSeedingDataLoader."""
        del seed


@dataclass
class BenchmarkConfig:
    """Minimal dataset configuration consumed by the loader utility."""

    dataset_id: str
    seed: int


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--sample-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--delay-ms", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device receiving each batch; auto selects CUDA when available.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for machine-readable benchmark results.",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _consume_epoch(loader: Any, device: torch.device) -> tuple[int, float]:
    sample_count = 0
    checksum = torch.zeros((), device=device)
    _synchronize(device)
    start = time.perf_counter()

    inference_context = torch.inference_mode() if device.type == "cuda" else nullcontext()
    with inference_context:
        for samples, _labels in loader:
            samples = samples.to(
                device,
                non_blocking=bool(getattr(loader, "pin_memory", False)),
            )
            checksum += samples[:, :, 0].sum()
            sample_count += samples.shape[0]

    _synchronize(device)
    elapsed = time.perf_counter() - start
    return sample_count, elapsed


def main() -> None:
    """Run the loader benchmark and print a cross-branch comparable summary."""
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be greater than or equal to zero.")
    if min(args.samples, args.sample_length, args.batch_size, args.epochs) < 1:
        raise ValueError("samples, sample length, batch size, and epochs must be positive.")
    if args.delay_ms < 0 or args.warmup_epochs < 0:
        raise ValueError("delay and warmup epochs must be non-negative.")

    device = _resolve_device(args.device)
    dataset = DelayedTensorDataset(args.samples, args.sample_length, args.delay_ms)
    config = BenchmarkConfig(dataset_id="loader-benchmark", seed=args.seed)

    loader_parameters = inspect.signature(prepare_torchsig_datasets).parameters
    supports_loader_options = "num_workers" in loader_parameters
    loader_kwargs: dict[str, Any] = {}
    if supports_loader_options:
        loader_kwargs.update(
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

    with patch(
        "torchsig_models.utils.datasets._create_static_dataset",
        return_value=(dataset, [f"class_{index}" for index in range(8)]),
    ):
        train_loader, _, _, _ = prepare_torchsig_datasets(
            config,
            config,
            config,
            dataset_root=Path("/tmp/torchsig-models-loader-benchmark"),
            batch_size=args.batch_size,
            transforms=[],
            **loader_kwargs,
        )

    for _ in range(args.warmup_epochs):
        _consume_epoch(train_loader, device)

    epoch_seconds: list[float] = []
    sample_count = 0
    for _ in range(args.epochs):
        sample_count, elapsed = _consume_epoch(train_loader, device)
        epoch_seconds.append(elapsed)

    median_seconds = statistics.median(epoch_seconds)
    result = {
        "loader_options_supported": supports_loader_options,
        "requested_num_workers": args.num_workers,
        "effective_num_workers": train_loader.num_workers,
        "pin_memory": train_loader.pin_memory,
        "persistent_workers": train_loader.persistent_workers,
        "device": str(device),
        "samples": sample_count,
        "sample_length": args.sample_length,
        "batch_size": args.batch_size,
        "delay_ms": args.delay_ms,
        "epoch_seconds": epoch_seconds,
        "median_epoch_seconds": median_seconds,
        "median_samples_per_second": sample_count / median_seconds,
    }

    print(json.dumps(result, indent=2))
    if not supports_loader_options:
        print(
            "\nThis revision does not expose loader performance options; "
            "the legacy zero-worker path was benchmarked."
        )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
