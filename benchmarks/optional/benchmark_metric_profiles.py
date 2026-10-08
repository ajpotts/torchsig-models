#!/usr/bin/env python3
"""Compare full and lightweight metric overhead in short training trials."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from torchsig_models.utils.training import train_validate


PROFILES = ("full", "lightweight")


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=PROFILES)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--classes", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def make_loader(
    samples: int,
    features: int,
    classes: int,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """Build a deterministic synthetic classification loader."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, features, generator=generator)
    labels = torch.arange(samples) % classes
    return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size)


def run_trial(args: argparse.Namespace, profile: str, accelerator: str) -> float:
    """Run one training trial and return elapsed wall-clock seconds."""
    torch.manual_seed(args.seed)
    train_loader = make_loader(
        args.train_samples,
        args.features,
        args.classes,
        args.batch_size,
        args.seed,
    )
    val_loader = make_loader(
        args.val_samples,
        args.features,
        args.classes,
        args.batch_size,
        args.seed + 1,
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(args.features, args.features),
        torch.nn.ReLU(),
        torch.nn.Linear(args.features, args.classes),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    if accelerator == "gpu":
        torch.cuda.synchronize()
    start = time.perf_counter()
    train_validate(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=None,
        max_epochs=args.epochs,
        num_classes=args.classes,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        logger=False,
        metric_profile=profile,
    )
    if accelerator == "gpu":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    """Run and report the requested profile comparisons."""
    args = parse_args()
    if (
        min(
            args.train_samples,
            args.val_samples,
            args.features,
            args.classes,
            args.batch_size,
            args.epochs,
            args.runs,
        )
        < 1
        or args.warmup_runs < 0
    ):
        raise ValueError("sizes, epochs, and runs must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    accelerator = (
        "gpu"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    results: dict[str, object] = {"accelerator": accelerator, "profiles": {}}
    profile_results = results["profiles"]
    assert isinstance(profile_results, dict)
    for profile in args.profiles:
        for _ in range(args.warmup_runs):
            run_trial(args, profile, accelerator)
        elapsed = [run_trial(args, profile, accelerator) for _ in range(args.runs)]
        median = statistics.median(elapsed)
        profile_results[profile] = {
            "elapsed_seconds": elapsed,
            "median_elapsed_seconds": median,
            "median_train_samples_per_second": (
                args.train_samples * args.epochs / median
            ),
        }

    if "full" in profile_results and "lightweight" in profile_results:
        full = profile_results["full"]["median_elapsed_seconds"]
        lightweight = profile_results["lightweight"]["median_elapsed_seconds"]
        results["lightweight_speedup"] = full / lightweight

    output = json.dumps(results, indent=2)
    print(output)
    if args.json_output is not None:
        args.json_output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
