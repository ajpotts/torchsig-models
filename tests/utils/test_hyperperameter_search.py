"""Tests for hyperparameter optimization utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import optuna
import pytest
import yaml

import torchsig_models.utils.hyperparameter_search as hyperparameter_search
from torchsig_models.utils.hyperparameter_search import (
    _MLflowLogger,
    _TrialMLFlowLogger,
    _extract_metric,
    _final_result_metrics,
    _mlflow_http_settings,
    active_mlflow_run_id,
    create_trial_loggers,
    load_search_config,
    run_hyperparameter_optimization,
    suggest_params,
    training_run_metadata,
)


PACKAGED_SEARCH_CONFIGS = [
    Path("torchsig_models/models")
    / representation
    / "efficientnet"
    / "search_configs"
    / filename
    for representation in ("iq_models", "spectrogram_models")
    for filename in (
        "efficientnet_b0_search_config.yaml",
        "efficientnet_b2_search_config.yaml",
        "efficientnet_b4_search_config.yaml",
    )
]


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


@pytest.mark.parametrize("config_path", PACKAGED_SEARCH_CONFIGS)
def test_packaged_search_configs_include_normalization(config_path: Path) -> None:
    """Search every packaged model over all supported normalization modes."""
    config = load_search_config(config_path)

    assert config["n_trials"] == 30
    assert config["search_space"]["normalization"] == {
        "type": "categorical",
        "choices": ["dataset", "sample", "none"],
    }


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
    observed_calls: list[tuple[dict[str, Any], Path, int]] = []

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
        "train_fn": lambda params, trial_dir, trial: {"val_f1": 1.0},
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
        assert hyperparameter_search.os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "30"
        assert (
            hyperparameter_search.os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "0"
        )

    assert hyperparameter_search.os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "30"
    assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in hyperparameter_search.os.environ


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


def test_create_trial_loggers_reuses_active_mlflow_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Create reusable CSV and existing-run MLflow loggers for any search."""
    csv_logger = Mock()
    mlflow_logger = Mock()
    csv_factory = Mock(return_value=csv_logger)
    mlflow_factory = Mock(return_value=mlflow_logger)
    monkeypatch.setattr(hyperparameter_search, "CSVLogger", csv_factory)
    monkeypatch.setattr(hyperparameter_search, "_TrialMLFlowLogger", mlflow_factory)
    monkeypatch.setattr(
        hyperparameter_search,
        "active_mlflow_run_id",
        lambda: "run-123",
    )

    loggers = create_trial_loggers(
        trial_dir=tmp_path,
        mlflow_enabled=True,
        experiment_name="model-search",
        hyperparameters={"batch_size": 32},
    )

    assert loggers == [csv_logger, mlflow_logger]
    csv_factory.assert_called_once_with(
        save_dir=tmp_path,
        name="lightning_logs",
        version="",
    )
    mlflow_factory.assert_called_once_with(
        experiment_name="model-search",
        run_id="run-123",
        log_model=False,
    )
    csv_logger.log_hyperparams.assert_called_once_with({"batch_size": 32})
    mlflow_logger.log_hyperparams.assert_called_once_with({"batch_size": 32})


def test_create_trial_loggers_falls_back_to_csv_on_mlflow_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep the shared CSV logger when MLflow cannot initialize."""
    csv_logger = Mock()
    monkeypatch.setattr(
        hyperparameter_search,
        "CSVLogger",
        Mock(return_value=csv_logger),
    )
    monkeypatch.setattr(
        hyperparameter_search,
        "active_mlflow_run_id",
        lambda: "run-123",
    )
    monkeypatch.setattr(
        hyperparameter_search,
        "_TrialMLFlowLogger",
        Mock(side_effect=ConnectionError("tracking server unavailable")),
    )

    with caplog.at_level(logging.WARNING):
        training_logger = create_trial_loggers(
            trial_dir=tmp_path,
            mlflow_enabled=True,
            experiment_name="model-search",
        )

    assert training_logger is csv_logger
    assert "CSV logging only" in caplog.text


def test_trial_mlflow_logger_disables_after_metric_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Suppress runtime MLflow failures so Lightning training can continue."""
    mlflow_logger = object.__new__(_TrialMLFlowLogger)
    mlflow_logger._available = True
    mlflow_logger._warning_emitted = False

    def fail(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ConnectionError("tracking server unavailable")

    monkeypatch.setattr(hyperparameter_search.MLFlowLogger, "log_metrics", fail)

    with caplog.at_level(logging.WARNING):
        mlflow_logger.log_metrics({"val_f1": 0.75}, step=1)
        mlflow_logger.log_metrics({"val_f1": 0.80}, step=2)

    assert mlflow_logger._available is False
    assert caplog.text.count("CSV logging will continue") == 1


def test_final_result_metrics_includes_test_history() -> None:
    """Extract generic scalar and final evaluation metrics for MLflow."""
    result = {
        "val_f1": 0.8,
        "num_params": 123,
        "ignored": object(),
        "test_metrics": SimpleNamespace(
            history={
                "loss": [0.25],
                "accuracy": [0.9],
                "f1 score": [0.85],
                "precision": [0.8],
                "recall": [0.88],
            }
        ),
    }

    assert _final_result_metrics(result) == {
        "val_f1": 0.8,
        "num_params": 123.0,
        "test_loss": 0.25,
        "test_acc": 0.9,
        "test_f1": 0.85,
        "test_precision": 0.8,
        "test_recall": 0.88,
    }


def test_training_run_metadata_builds_common_search_metadata() -> None:
    """Build the complete metadata contract shared by model searches."""
    metadata = training_run_metadata(
        params={"batch_size": 32, "learning_rate": 0.001},
        trial_number=4,
        model_name="efficientnet_b0",
        train_cfg=SimpleNamespace(dataset_id="train", dataset_length=100, seed=1),
        val_cfg=SimpleNamespace(dataset_id="val", dataset_length=20, seed=2),
        test_cfg=SimpleNamespace(dataset_id="test", dataset_length=30, seed=3),
        storage_backend="homogeneous",
        optimizer="AdamW",
        scheduler="LinearLR+CosineAnnealingLR",
    )

    assert metadata == {
        "batch_size": 32,
        "learning_rate": 0.001,
        "trial_number": 4,
        "model_name": "efficientnet_b0",
        "train_dataset_id": "train",
        "val_dataset_id": "val",
        "test_dataset_id": "test",
        "train_dataset_length": 100,
        "val_dataset_length": 20,
        "test_dataset_length": 30,
        "train_seed": 1,
        "val_seed": 2,
        "test_seed": 3,
        "storage_backend": "homogeneous",
        "optimizer": "AdamW",
        "scheduler": "LinearLR+CosineAnnealingLR",
    }


def test_optimization_exposes_existing_trial_run_to_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Expose the one HPO-created trial run without creating a second run."""
    run_ids = iter(["parent-run", "trial-run"])
    started_runs: list[str] = []

    def start_run(**kwargs: Any) -> SimpleNamespace:
        del kwargs
        run_id = next(run_ids)
        started_runs.append(run_id)
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    fake_mlflow = SimpleNamespace(
        start_run=start_run,
        end_run=lambda: None,
        set_experiment=lambda name: None,
        log_params=lambda params: None,
        log_param=lambda name, value: None,
        log_metric=lambda name, value: None,
    )
    monkeypatch.setattr(hyperparameter_search, "mlflow", fake_mlflow)

    observed_run_ids: list[str | None] = []

    def train_fn(
        params: dict[str, Any],
        trial_dir: Path,
        trial: optuna.Trial,
    ) -> dict[str, float]:
        del params, trial_dir, trial
        observed_run_ids.append(active_mlflow_run_id())
        return {"val_f1": 0.75}

    run_hyperparameter_optimization(
        base_params={},
        search_space={},
        train_fn=train_fn,
        n_trials=1,
        output_dir=tmp_path,
        show_progress_bar=False,
        mlflow_enabled=True,
    )

    assert started_runs == ["parent-run", "trial-run"]
    assert observed_run_ids == ["trial-run"]
    assert active_mlflow_run_id() is None
