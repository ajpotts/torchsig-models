"""Tests for hyperparameter optimization utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import optuna
import pytest
import yaml

import torchsig_models.utils.hyperparameter_search as hyperparameter_search
from torchsig_models.utils.hyperparameter_search import (
    _MLflowLogger,
    _extract_metric,
    _mlflow_http_settings,
    load_search_config,
    run_hyperparameter_optimization,
    suggest_params,
)


class RecordingTrial:
    """Minimal Optuna trial replacement that records suggestion calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = []

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float:
        self.calls.append(
            (
                "float",
                name,
                (low, high),
                {"log": log},
            )
        )
        return 0.01

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        self.calls.append(
            (
                "int",
                name,
                (low, high),
                {
                    "step": step,
                    "log": log,
                },
            )
        )
        return 32

    def suggest_categorical(
        self,
        name: str,
        choices: list[Any],
    ) -> Any:
        self.calls.append(
            (
                "categorical",
                name,
                (choices,),
                {},
            )
        )
        return choices[0]


def test_load_search_config(tmp_path: Path) -> None:
    """Load a valid YAML search configuration."""
    config_path = tmp_path / "search.yaml"
    expected = {
        "metric_name": "val_f1",
        "direction": "maximize",
        "n_trials": 3,
        "search_space": {
            "learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-2,
                "log": True,
            }
        },
    }

    config_path.write_text(
        yaml.safe_dump(expected),
        encoding="utf-8",
    )

    assert load_search_config(config_path) == expected


def test_load_search_config_accepts_string_path(tmp_path: Path) -> None:
    """Accept both string and Path configuration paths."""
    config_path = tmp_path / "search.yaml"
    config_path.write_text(
        "search_space: {}\n",
        encoding="utf-8",
    )

    assert load_search_config(str(config_path)) == {
        "search_space": {},
    }


def test_load_search_config_raises_for_missing_file(
    tmp_path: Path,
) -> None:
    """Raise a useful error when the configuration does not exist."""
    config_path = tmp_path / "missing.yaml"

    with pytest.raises(
        FileNotFoundError,
        match="Search config not found",
    ):
        load_search_config(config_path)


@pytest.mark.parametrize(
    "contents",
    [
        "- one\n- two\n",
        "null\n",
        "search-space\n",
    ],
)
def test_load_search_config_requires_mapping(
    tmp_path: Path,
    contents: str,
) -> None:
    """Reject YAML documents whose root is not a mapping."""
    config_path = tmp_path / "search.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Search config must contain a YAML mapping",
    ):
        load_search_config(config_path)


def test_suggest_params_applies_supported_parameter_types() -> None:
    """Apply float, integer, and categorical suggestions."""
    trial = RecordingTrial()
    base_params = {
        "learning_rate": 0.1,
        "batch_size": 16,
        "optimizer": "sgd",
        "max_epochs": 10,
    }
    search_space = {
        "learning_rate": {
            "type": "float",
            "low": 1e-5,
            "high": 1e-2,
            "log": True,
        },
        "batch_size": {
            "type": "int",
            "low": 16,
            "high": 64,
            "step": 16,
        },
        "optimizer": {
            "type": "categorical",
            "choices": ["adam", "sgd"],
        },
    }

    params = suggest_params(
        trial,  # type: ignore[arg-type]
        base_params,
        search_space,
    )

    assert params == {
        "learning_rate": 0.01,
        "batch_size": 32,
        "optimizer": "adam",
        "max_epochs": 10,
    }
    assert trial.calls == [
        (
            "float",
            "learning_rate",
            (1e-5, 1e-2),
            {"log": True},
        ),
        (
            "int",
            "batch_size",
            (16, 64),
            {
                "step": 16,
                "log": False,
            },
        ),
        (
            "categorical",
            "optimizer",
            (["adam", "sgd"],),
            {},
        ),
    ]


def test_suggest_params_does_not_modify_base_params() -> None:
    """Return a new parameter dictionary."""
    trial = RecordingTrial()
    base_params = {"learning_rate": 0.1}

    params = suggest_params(
        trial,  # type: ignore[arg-type]
        base_params,
        {
            "learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-2,
            }
        },
    )

    assert params is not base_params
    assert base_params == {"learning_rate": 0.1}


def test_suggest_params_rejects_unsupported_type() -> None:
    """Reject unknown search parameter types."""
    trial = RecordingTrial()

    with pytest.raises(
        ValueError,
        match="Unsupported search parameter type: boolean",
    ):
        suggest_params(
            trial,  # type: ignore[arg-type]
            {},
            {
                "enabled": {
                    "type": "boolean",
                }
            },
        )


def test_extract_metric_from_top_level_result() -> None:
    """Prefer a metric stored directly in the result."""
    result = {
        "val_f1": 0.85,
        "metrics": SimpleNamespace(val_f1s=[0.50]),
    }

    assert _extract_metric(result, "val_f1") == pytest.approx(0.85)


def test_extract_metric_from_plural_metric_history() -> None:
    """Extract the final value from a plural metric history."""
    result = {
        "metrics": SimpleNamespace(
            val_f1s=[0.60, 0.75, 0.82],
        )
    }

    assert _extract_metric(result, "val_f1") == pytest.approx(0.82)


def test_extract_metric_from_singular_list() -> None:
    """Extract the final value from a singular list attribute."""
    result = {
        "metrics": SimpleNamespace(
            loss=[1.0, 0.5, 0.25],
        )
    }

    assert _extract_metric(result, "loss") == pytest.approx(0.25)


def test_extract_metric_from_scalar_attribute() -> None:
    """Extract a scalar metric attribute."""
    result = {
        "metrics": SimpleNamespace(
            accuracy=0.91,
        )
    }

    assert _extract_metric(result, "accuracy") == pytest.approx(0.91)


@pytest.mark.parametrize(
    ("metrics", "metric_name", "message"),
    [
        (
            SimpleNamespace(val_f1s=[]),
            "val_f1",
            "Metric history 'val_f1s' is empty",
        ),
        (
            SimpleNamespace(loss=[]),
            "loss",
            "Metric history 'loss' is empty",
        ),
    ],
)
def test_extract_metric_rejects_empty_history(
    metrics: SimpleNamespace,
    metric_name: str,
    message: str,
) -> None:
    """Reject empty metric histories."""
    with pytest.raises(ValueError, match=message):
        _extract_metric(
            {"metrics": metrics},
            metric_name,
        )


def test_extract_metric_raises_when_metric_is_missing() -> None:
    """Raise when the requested metric cannot be found."""
    with pytest.raises(
        KeyError,
        match="Could not find metric 'val_f1'",
    ):
        _extract_metric(
            {"metrics": SimpleNamespace(train_loss=[1.0])},
            "val_f1",
        )


def test_run_hyperparameter_optimization_creates_trial_directories(
    tmp_path: Path,
) -> None:
    """Create one output directory for each Optuna trial."""
    observed_calls: list[
        tuple[dict[str, Any], Path, int]
    ] = []

    def train_fn(
        params: dict[str, Any],
        trial_dir: Path,
        trial: optuna.Trial,
    ) -> dict[str, Any]:
        observed_calls.append(
            (
                params,
                trial_dir,
                trial.number,
            )
        )

        assert trial_dir.exists()
        assert trial_dir.is_dir()

        return {
            "val_f1": float(params["score"]),
            "num_params": 100,
        }

    study = run_hyperparameter_optimization(
        base_params={"max_epochs": 1},
        search_space={
            "score": {
                "type": "categorical",
                "choices": [0.5, 0.8],
            }
        },
        train_fn=train_fn,
        metric_name="val_f1",
        direction="maximize",
        n_trials=2,
        output_dir=tmp_path,
        show_progress_bar=False,
        mlflow_enabled=False,
    )

    assert len(study.trials) == 2
    assert len(observed_calls) == 2

    assert observed_calls[0][1] == tmp_path / "trial_0000"
    assert observed_calls[1][1] == tmp_path / "trial_0001"

    assert (tmp_path / "trial_0000").is_dir()
    assert (tmp_path / "trial_0001").is_dir()

    assert study.best_value in {0.5, 0.8}


def test_run_hyperparameter_optimization_extracts_nested_metric(
    tmp_path: Path,
) -> None:
    """Allow the training callback to return a metrics object."""
    def train_fn(
        params: dict[str, Any],
        trial_dir: Path,
        trial: optuna.Trial,
    ) -> dict[str, Any]:
        del params, trial_dir, trial

        return {
            "metrics": SimpleNamespace(
                val_f1s=[0.4, 0.7],
            )
        }

    study = run_hyperparameter_optimization(
        base_params={},
        search_space={},
        train_fn=train_fn,
        metric_name="val_f1",
        n_trials=1,
        output_dir=tmp_path,
        show_progress_bar=False,
        mlflow_enabled=False,
    )

    assert study.best_value == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"n_trials": 0},
            "n_trials must be at least 1",
        ),
        (
            {"mlflow_timeout_seconds": 0},
            "mlflow_timeout_seconds must be at least 1",
        ),
        (
            {"mlflow_max_retries": -1},
            "mlflow_max_retries cannot be negative",
        ),
    ],
)
def test_run_hyperparameter_optimization_validates_arguments(
    tmp_path: Path,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    """Validate optimization and MLflow settings."""
    arguments: dict[str, Any] = {
        "base_params": {},
        "search_space": {},
        "train_fn": lambda params, trial_dir, trial: {
            "val_f1": 1.0
        },
        "output_dir": tmp_path,
        "show_progress_bar": False,
        "mlflow_enabled": False,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        run_hyperparameter_optimization(**arguments)


def test_training_errors_propagate(
    tmp_path: Path,
) -> None:
    """Do not suppress errors raised by the training callback."""
    def train_fn(
        params: dict[str, Any],
        trial_dir: Path,
        trial: optuna.Trial,
    ) -> dict[str, Any]:
        del params, trial_dir, trial
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        run_hyperparameter_optimization(
            base_params={},
            search_space={},
            train_fn=train_fn,
            n_trials=1,
            output_dir=tmp_path,
            show_progress_bar=False,
            mlflow_enabled=False,
        )


def test_mlflow_http_settings_sets_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore MLflow environment variables after optimization."""
    monkeypatch.setenv(
        "MLFLOW_HTTP_REQUEST_TIMEOUT",
        "30",
    )
    monkeypatch.delenv(
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES",
        raising=False,
    )

    with _mlflow_http_settings(
        timeout_seconds=5,
        max_retries=0,
    ):
        # Existing values are preserved because the implementation uses
        # setdefault().
        assert (
            hyperparameter_search.os.environ[
                "MLFLOW_HTTP_REQUEST_TIMEOUT"
            ]
            == "30"
        )
        assert (
            hyperparameter_search.os.environ[
                "MLFLOW_HTTP_REQUEST_MAX_RETRIES"
            ]
            == "0"
        )

    assert (
        hyperparameter_search.os.environ[
            "MLFLOW_HTTP_REQUEST_TIMEOUT"
        ]
        == "30"
    )
    assert (
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES"
        not in hyperparameter_search.os.environ
    )


def test_mlflow_logger_disables_tracking_when_package_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Disable tracking and warn once when MLflow is not installed."""
    monkeypatch.setattr(
        hyperparameter_search,
        "mlflow",
        None,
    )

    with caplog.at_level(logging.WARNING):
        tracking = _MLflowLogger(enabled=True)

    assert tracking.enabled is False
    assert "the mlflow package is not installed" in caplog.text


def test_mlflow_logger_disables_after_first_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn once and skip later MLflow calls after a failure."""
    calls = 0

    def failing_log_metric(
        name: str,
        value: float,
    ) -> None:
        nonlocal calls
        del name, value
        calls += 1
        raise ConnectionError("MLflow unavailable")

    fake_mlflow = SimpleNamespace(
        log_metric=failing_log_metric,
    )
    monkeypatch.setattr(
        hyperparameter_search,
        "mlflow",
        fake_mlflow,
    )

    tracking = _MLflowLogger(enabled=True)

    with caplog.at_level(logging.WARNING):
        tracking.log_metric("val_f1", 0.5)
        tracking.log_metric("val_f1", 0.6)

    assert tracking.enabled is False
    assert calls == 1
    assert caplog.text.count("MLflow is unavailable") == 1
