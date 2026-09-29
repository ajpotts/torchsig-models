"""Tests for the runnable EfficientNet normalization experiment."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "examples" / "scripts" / "compare_efficientnet_normalization.py"
SPEC = importlib.util.spec_from_file_location("normalization_experiment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


@dataclass
class Config:
    dataset_id: str
    dataset_length: int
    seed: int
    dataset_metadata: dict[str, float]


def test_build_split_configs_uses_fixed_splits_and_shifted_test() -> None:
    base = Config("base", 100, 1, {"noise_power_db": 0.0, "fft_size": 64})

    train, val, test = experiment._build_split_configs(
        base,
        dataset_id="comparison",
        train_size=2400,
        eval_size=600,
        seed=25,
        test_power_shift_db=-10.0,
    )

    assert (train.dataset_length, val.dataset_length, test.dataset_length) == (2400, 600, 600)
    assert (train.seed, val.seed, test.seed) == (25, 26, 27)
    assert train.dataset_id == val.dataset_id == test.dataset_id == "comparison"
    assert train.dataset_metadata["noise_power_db"] == 0.0
    assert val.dataset_metadata["noise_power_db"] == 0.0
    assert test.dataset_metadata["noise_power_db"] == -10.0


def test_result_row_reports_validation_and_shifted_test_metrics() -> None:
    validation = {
        "loss": [1.0],
        "accuracy": [0.75],
        "f1 score": [0.7],
    }
    shifted_test = {
        "loss": [1.2],
        "accuracy": [0.65],
        "f1 score": [0.6],
    }
    result = {
        "metrics": SimpleNamespace(val_metrics=SimpleNamespace(history=validation)),
        "test_metrics": SimpleNamespace(history=shifted_test),
        "normalization": {"mode": "dataset", "mean": [-42.0], "std": [9.0]},
    }

    row = experiment._result_row("dataset", result, 12.5)

    assert row["val_macro_f1"] == pytest.approx(0.7)
    assert row["shifted_test_macro_f1"] == pytest.approx(0.6)
    assert row["normalization_mean"] == [-42.0]
    assert row["elapsed_seconds"] == pytest.approx(12.5)
