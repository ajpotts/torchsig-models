"""Tests for the EfficientNet-1D hyperparameter search CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import optuna
import pytest
import yaml

import torchsig_models.models.iq_models.efficientnet.efficientnet1d_hyperparameter_search as search_module
from torchsig_models.models.iq_models.efficientnet.efficientnet1d_hyperparameter_search import (
    _apply_dataset_overrides,
    _final_metrics,
    _write_best_trial_summary,
    _load_split_configs,
    main,
    parse_args,
)


@dataclass
class FakeDatasetConfig:
    """Minimal dataset config supporting dataclasses.replace."""

    dataset_id: str
    dataset_length: int
    seed: int


class FakeCSVLogger:
    """Record CSVLogger initialization and hyperparameter logging."""

    instances: list[FakeCSVLogger] = []

    def __init__(
        self,
        *,
        save_dir: Path,
        name: str,
        version: str,
    ) -> None:
        self.save_dir = save_dir
        self.name = name
        self.version = version
        self.hyperparameters: dict[str, Any] | None = None

        self.instances.append(self)

    def log_hyperparams(
        self,
        params: dict[str, Any],
    ) -> None:
        self.hyperparameters = params


@pytest.fixture(autouse=True)
def clear_fake_loggers() -> None:
    """Clear recorded fake logger instances between tests."""
    FakeCSVLogger.instances.clear()


def test_parse_args_accepts_shared_dataset_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accept one dataset config for all dataset splits."""
    dataset_config = tmp_path / "dataset.yaml"

    monkeypatch.setattr(
        search_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            dataset_config=dataset_config,
            train_config=None,
            val_config=None,
            test_config=None,
            search_config=tmp_path / "search.yaml",
            params=None,
            model="efficientnet_b0",
            dataset_root=tmp_path / "datasets",
            output_dir=tmp_path / "runs",
            dataset_length=None,
            dataset_id=None,
            overwrite=False,
            n_trials=None,
            env_file=tmp_path / ".env",
            max_epochs=None,
            enable_mlflow=False,
            mlflow_timeout=5,
            mlflow_max_retries=0,
        ),
    )

    args = parse_args()

    assert args.dataset_config == dataset_config
    assert args.train_config is None


def test_parse_args_accepts_train_config_without_shared_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accept a split-specific training config without a shared config."""
    train_config = tmp_path / "train.yaml"

    monkeypatch.setattr(
        search_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            dataset_config=None,
            train_config=train_config,
            val_config=None,
            test_config=None,
            search_config=tmp_path / "search.yaml",
            params=None,
            model="efficientnet_b0",
            dataset_root=tmp_path / "datasets",
            output_dir=tmp_path / "runs",
            dataset_length=None,
            dataset_id=None,
            overwrite=False,
            n_trials=None,
            env_file=tmp_path / ".env",
            max_epochs=None,
            enable_mlflow=False,
            mlflow_timeout=5,
            mlflow_max_retries=0,
        ),
    )

    args = parse_args()

    assert args.dataset_config is None
    assert args.train_config == train_config


def test_parse_args_requires_dataset_or_train_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject calls without a shared or training dataset config."""
    monkeypatch.setattr(
        "sys.argv",
        ["efficientnet1d_hyperparameter_search.py"],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_load_split_configs_uses_shared_config_for_all_splits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use a shared config and offset validation and test seeds."""
    dataset_path = tmp_path / "dataset.yaml"

    def fake_load_config(path: Path) -> FakeDatasetConfig:
        assert path == dataset_path

        return FakeDatasetConfig(
            dataset_id="shared",
            dataset_length=100,
            seed=10,
        )

    monkeypatch.setattr(
        search_module,
        "load_config_from_yaml",
        fake_load_config,
    )

    train_cfg, val_cfg, test_cfg = _load_split_configs(
        dataset_config=dataset_path,
        train_config=None,
        val_config=None,
        test_config=None,
    )

    assert train_cfg.seed == 10
    assert val_cfg.seed == 11
    assert test_cfg.seed == 12

    assert train_cfg.dataset_id == "shared"
    assert val_cfg.dataset_id == "shared"
    assert test_cfg.dataset_id == "shared"


def test_load_split_configs_prefers_explicit_split_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preserve explicitly configured split seeds."""
    train_path = tmp_path / "train.yaml"
    val_path = tmp_path / "val.yaml"
    test_path = tmp_path / "test.yaml"

    configs = {
        train_path: FakeDatasetConfig(
            dataset_id="train",
            dataset_length=100,
            seed=10,
        ),
        val_path: FakeDatasetConfig(
            dataset_id="val",
            dataset_length=20,
            seed=40,
        ),
        test_path: FakeDatasetConfig(
            dataset_id="test",
            dataset_length=20,
            seed=70,
        ),
    }

    monkeypatch.setattr(
        search_module,
        "load_config_from_yaml",
        lambda path: configs[path],
    )

    train_cfg, val_cfg, test_cfg = _load_split_configs(
        dataset_config=None,
        train_config=train_path,
        val_config=val_path,
        test_config=test_path,
    )

    assert train_cfg == configs[train_path]
    assert val_cfg == configs[val_path]
    assert test_cfg == configs[test_path]

    assert val_cfg.seed == 40
    assert test_cfg.seed == 70


def test_load_split_configs_falls_back_to_train_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use the training config when validation and test configs are absent."""
    train_path = tmp_path / "train.yaml"

    monkeypatch.setattr(
        search_module,
        "load_config_from_yaml",
        lambda path: FakeDatasetConfig(
            dataset_id="train",
            dataset_length=100,
            seed=5,
        ),
    )

    train_cfg, val_cfg, test_cfg = _load_split_configs(
        dataset_config=None,
        train_config=train_path,
        val_config=None,
        test_config=None,
    )

    assert train_cfg.seed == 5
    assert val_cfg.seed == 6
    assert test_cfg.seed == 7


def test_load_split_configs_requires_training_config() -> None:
    """Reject calls that cannot resolve a training configuration."""
    with pytest.raises(
        ValueError,
        match="A training dataset config must be provided",
    ):
        _load_split_configs(
            dataset_config=None,
            train_config=None,
            val_config=None,
            test_config=None,
        )


def test_apply_dataset_overrides() -> None:
    """Apply dataset length and ID overrides."""
    original = FakeDatasetConfig(
        dataset_id="original",
        dataset_length=1_000,
        seed=10,
    )

    updated = _apply_dataset_overrides(
        original,
        dataset_length=100,
        dataset_id="smoke-test",
    )

    assert updated == FakeDatasetConfig(
        dataset_id="smoke-test",
        dataset_length=100,
        seed=10,
    )
    assert updated is not original


def test_apply_dataset_overrides_supports_partial_override() -> None:
    """Only replace fields whose overrides are provided."""
    original = FakeDatasetConfig(
        dataset_id="original",
        dataset_length=1_000,
        seed=10,
    )

    updated = _apply_dataset_overrides(
        original,
        dataset_length=100,
        dataset_id=None,
    )

    assert updated.dataset_id == "original"
    assert updated.dataset_length == 100
    assert updated.seed == 10


def test_apply_dataset_overrides_returns_original_when_unchanged() -> None:
    """Avoid replacing a config when no overrides are supplied."""
    original = FakeDatasetConfig(
        dataset_id="original",
        dataset_length=1_000,
        seed=10,
    )

    result = _apply_dataset_overrides(
        original,
        dataset_length=None,
        dataset_id=None,
    )

    assert result is original


def test_final_metrics_extracts_latest_values() -> None:
    """Extract the final scalar value from each metric history."""
    result = {
        "metrics": SimpleNamespace(
            val_f1s=[0.50, 0.75],
            val_accuracies=[0.60, 0.80],
            train_f1s=[0.70, 0.90],
            train_accuracies=[0.75, 0.95],
        )
    }

    assert _final_metrics(result) == {
        "val_f1": pytest.approx(0.75),
        "val_acc": pytest.approx(0.80),
        "train_f1": pytest.approx(0.90),
        "train_acc": pytest.approx(0.95),
    }


def test_best_trial_summary_uses_best_not_last_trial(tmp_path: Path) -> None:
    """Report parameters from the best trial when the last trial is worse."""
    study = optuna.create_study(direction="maximize")
    distribution = optuna.distributions.FloatDistribution(1e-5, 1e-2)
    study.add_trial(
        optuna.trial.create_trial(
            params={"learning_rate": 0.001},
            distributions={"learning_rate": distribution},
            value=0.9,
        )
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={"learning_rate": 0.009},
            distributions={"learning_rate": distribution},
            value=0.2,
        )
    )

    summary_path, training_params_path = _write_best_trial_summary(
        study,
        "val_f1",
        {
            "model_name": "efficientnet_b0",
            "max_epochs": 30,
            "learning_rate": 0.0005,
            "pretrained": True,
        },
        tmp_path,
    )
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    training_params = yaml.safe_load(training_params_path.read_text(encoding="utf-8"))

    assert summary_path == tmp_path / "best_trial.yaml"
    assert summary == {
        "trial_number": 0,
        "metric_name": "val_f1",
        "metric_value": pytest.approx(0.9),
        "parameters": {"learning_rate": pytest.approx(0.001)},
    }
    assert training_params_path == tmp_path / "best_training_params.yaml"
    assert training_params == {
        "model_name": "efficientnet_b0",
        "max_epochs": 30,
        "learning_rate": pytest.approx(0.001),
        "pretrained": True,
    }


def test_main_configures_and_runs_optimization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Wire dataset configs, training, logging, and Optuna together."""
    dataset_path = tmp_path / "dataset.yaml"
    search_path = tmp_path / "search.yaml"
    output_dir = tmp_path / "optimization"
    dataset_root = tmp_path / "datasets"

    args = argparse.Namespace(
        dataset_config=dataset_path,
        train_config=None,
        val_config=None,
        test_config=None,
        search_config=search_path,
        params=None,
        model="efficientnet_b0",
        dataset_root=dataset_root,
        output_dir=output_dir,
        dataset_length=100,
        dataset_id="overridden-dataset",
        overwrite=True,
        n_trials=3,
        env_file=tmp_path / ".env",
        max_epochs=1,
        enable_mlflow=False,
        mlflow_timeout=7,
        mlflow_max_retries=2,
    )

    monkeypatch.setattr(
        search_module,
        "parse_args",
        lambda: args,
    )

    monkeypatch.setattr(
        search_module,
        "load_search_config",
        lambda path: {
            "metric_name": "val_f1",
            "direction": "maximize",
            "n_trials": 20,
            "experiment_name": "efficientnet-search",
            "run_name": "efficientnet-b0-search",
            "search_space": {
                "learning_rate": {
                    "type": "float",
                    "low": 1e-5,
                    "high": 1e-2,
                    "log": True,
                }
            },
        },
    )

    monkeypatch.setattr(
        search_module,
        "load_config_from_yaml",
        lambda path: FakeDatasetConfig(
            dataset_id="original-dataset",
            dataset_length=10_000,
            seed=10,
        ),
    )

    monkeypatch.setattr(
        search_module,
        "load_training_params",
        lambda model_name, params_path: {
            "model_name": model_name,
            "max_epochs": 50,
            "batch_size": 32,
        },
    )

    monkeypatch.setattr(
        search_module,
        "CSVLogger",
        FakeCSVLogger,
    )

    training_calls: list[dict[str, Any]] = []

    def fake_train_efficientnet_iq(
        **kwargs: Any,
    ) -> dict[str, Any]:
        training_calls.append(kwargs)

        return {
            "metrics": SimpleNamespace(
                val_f1s=[0.80],
                val_accuracies=[0.85],
                train_f1s=[0.90],
                train_accuracies=[0.95],
            ),
            "num_params": 123_456,
        }

    monkeypatch.setattr(
        search_module,
        "train_efficientnet_iq",
        fake_train_efficientnet_iq,
    )

    optimization_arguments: dict[str, Any] = {}

    def fake_run_hyperparameter_optimization(
        **kwargs: Any,
    ) -> SimpleNamespace:
        optimization_arguments.update(kwargs)

        trial_dir = Path(kwargs["output_dir"]) / "trial_0000"
        trial_dir.mkdir(parents=True)

        result = kwargs["train_fn"](
            {
                **kwargs["base_params"],
                "learning_rate": 0.001,
            },
            trial_dir,
            SimpleNamespace(number=0),
        )

        assert result["val_f1"] == pytest.approx(0.80)
        assert result["val_acc"] == pytest.approx(0.85)
        assert result["train_f1"] == pytest.approx(0.90)
        assert result["train_acc"] == pytest.approx(0.95)

        return SimpleNamespace(
            best_value=0.80,
            best_params={"learning_rate": 0.001},
            best_trial=SimpleNamespace(
                number=0,
                value=0.80,
                params={"learning_rate": 0.001},
            ),
        )

    monkeypatch.setattr(
        search_module,
        "run_hyperparameter_optimization",
        fake_run_hyperparameter_optimization,
    )

    # Avoid coupling this test to logging formatting.
    monkeypatch.setattr(
        search_module,
        "logger",
        Mock(),
    )

    main()

    assert optimization_arguments["base_params"] == {
        "model_name": "efficientnet_b0",
        "max_epochs": 1,
        "batch_size": 32,
    }
    assert optimization_arguments["metric_name"] == "val_f1"
    assert optimization_arguments["direction"] == "maximize"

    # CLI value takes precedence over the YAML value of 20.
    assert optimization_arguments["n_trials"] == 3

    assert (
        optimization_arguments["output_dir"]
        == output_dir / "overridden-dataset" / "efficientnet_b0"
    )
    assert optimization_arguments["mlflow_enabled"] is False
    assert optimization_arguments["mlflow_timeout_seconds"] == 7
    assert optimization_arguments["mlflow_max_retries"] == 2

    assert len(training_calls) == 1

    training_call = training_calls[0]

    assert training_call["train_cfg"] == FakeDatasetConfig(
        dataset_id="overridden-dataset",
        dataset_length=100,
        seed=10,
    )
    assert training_call["val_cfg"] == FakeDatasetConfig(
        dataset_id="overridden-dataset",
        dataset_length=100,
        seed=11,
    )
    assert training_call["test_cfg"] == FakeDatasetConfig(
        dataset_id="overridden-dataset",
        dataset_length=100,
        seed=12,
    )

    assert training_call["params"] == {
        "max_epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.001,
    }
    assert "model_name" not in training_call["params"]

    assert training_call["dataset_root"] == dataset_root
    assert training_call["overwrite"] is True
    assert training_call["model_name"] == "efficientnet_b0"

    trial_dir = output_dir / "overridden-dataset" / "efficientnet_b0" / "trial_0000"

    assert training_call["checkpoint_dir"] == (trial_dir / "checkpoints")
    assert training_call["metrics_dir"] == (trial_dir / "metrics")

    assert len(FakeCSVLogger.instances) == 1

    csv_logger = FakeCSVLogger.instances[0]

    assert csv_logger.save_dir == trial_dir
    assert csv_logger.name == "lightning_logs"
    assert csv_logger.version == ""

    assert csv_logger.hyperparameters is not None
    assert csv_logger.hyperparameters["trial_number"] == 0
    assert csv_logger.hyperparameters["model_name"] == "efficientnet_b0"
    assert csv_logger.hyperparameters["train_dataset_length"] == 100
    assert csv_logger.hyperparameters["train_seed"] == 10
    assert csv_logger.hyperparameters["val_seed"] == 11
    assert csv_logger.hyperparameters["test_seed"] == 12


def test_main_uses_search_config_trial_count_when_not_overridden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use the YAML trial count when --n-trials is absent."""
    args = argparse.Namespace(
        dataset_config=tmp_path / "dataset.yaml",
        train_config=None,
        val_config=None,
        test_config=None,
        search_config=tmp_path / "search.yaml",
        params=None,
        model="efficientnet_b0",
        dataset_root=tmp_path / "datasets",
        output_dir=tmp_path / "runs",
        dataset_length=None,
        dataset_id=None,
        overwrite=False,
        n_trials=None,
        env_file=tmp_path / ".env",
        max_epochs=None,
        enable_mlflow=False,
        mlflow_timeout=5,
        mlflow_max_retries=0,
    )

    monkeypatch.setattr(search_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        search_module,
        "load_search_config",
        lambda path: {
            "n_trials": 7,
            "search_space": {},
        },
    )
    monkeypatch.setattr(
        search_module,
        "load_config_from_yaml",
        lambda path: FakeDatasetConfig(
            dataset_id="dataset",
            dataset_length=100,
            seed=1,
        ),
    )
    monkeypatch.setattr(
        search_module,
        "load_training_params",
        lambda model_name, params_path: {},
    )

    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)

        return SimpleNamespace(
            best_value=0.5,
            best_params={},
            best_trial=SimpleNamespace(number=0, value=0.5, params={}),
        )

    monkeypatch.setattr(
        search_module,
        "run_hyperparameter_optimization",
        fake_run,
    )
    monkeypatch.setattr(
        search_module,
        "logger",
        Mock(),
    )

    main()

    assert captured["n_trials"] == 7


@pytest.mark.parametrize(
    ("search_config", "expected"),
    [
        ({}, 20),
        ({"n_trials": 8}, 8),
    ],
)
def test_main_trial_count_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    search_config: dict[str, Any],
    expected: int,
) -> None:
    """Use 20 trials when neither the CLI nor YAML specifies a value."""
    args = argparse.Namespace(
        dataset_config=tmp_path / "dataset.yaml",
        train_config=None,
        val_config=None,
        test_config=None,
        search_config=tmp_path / "search.yaml",
        params=None,
        model="efficientnet_b0",
        dataset_root=tmp_path / "datasets",
        output_dir=tmp_path / "runs",
        dataset_length=None,
        dataset_id=None,
        overwrite=False,
        n_trials=None,
        env_file=tmp_path / ".env",
        max_epochs=None,
        enable_mlflow=False,
        mlflow_timeout=5,
        mlflow_max_retries=0,
    )

    monkeypatch.setattr(search_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        search_module,
        "load_search_config",
        lambda path: {
            **search_config,
            "search_space": {},
        },
    )
    monkeypatch.setattr(
        search_module,
        "load_config_from_yaml",
        lambda path: FakeDatasetConfig(
            dataset_id="dataset",
            dataset_length=100,
            seed=1,
        ),
    )
    monkeypatch.setattr(
        search_module,
        "load_training_params",
        lambda model_name, params_path: {},
    )

    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)

        return SimpleNamespace(
            best_value=0.5,
            best_params={},
            best_trial=SimpleNamespace(number=0, value=0.5, params={}),
        )

    monkeypatch.setattr(
        search_module,
        "run_hyperparameter_optimization",
        fake_run,
    )
    monkeypatch.setattr(
        search_module,
        "logger",
        Mock(),
    )

    main()

    assert captured["n_trials"] == expected


@pytest.mark.parametrize(
    ("enable_mlflow", "env_exists", "should_load"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
    ],
)
def test_main_loads_dotenv_only_when_mlflow_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    enable_mlflow: bool,
    env_exists: bool,
    should_load: bool,
) -> None:
    """Load the environment file only for enabled MLflow tracking."""
    env_file = tmp_path / ".env"

    if env_exists:
        env_file.write_text(
            "MLFLOW_TRACKING_URI=http://example.test\n",
            encoding="utf-8",
        )

    args = argparse.Namespace(
        dataset_config=tmp_path / "dataset.yaml",
        train_config=None,
        val_config=None,
        test_config=None,
        search_config=tmp_path / "search.yaml",
        params=None,
        model="efficientnet_b0",
        dataset_root=tmp_path / "datasets",
        output_dir=tmp_path / "runs",
        dataset_length=None,
        dataset_id=None,
        overwrite=False,
        n_trials=1,
        env_file=env_file,
        max_epochs=None,
        enable_mlflow=enable_mlflow,
        mlflow_timeout=5,
        mlflow_max_retries=0,
    )

    load_dotenv = Mock()

    monkeypatch.setattr(search_module, "parse_args", lambda: args)
    monkeypatch.setattr(search_module, "load_dotenv", load_dotenv)
    monkeypatch.setattr(
        search_module,
        "load_search_config",
        lambda path: {
            "search_space": {},
        },
    )
    monkeypatch.setattr(
        search_module,
        "load_config_from_yaml",
        lambda path: FakeDatasetConfig(
            dataset_id="dataset",
            dataset_length=100,
            seed=1,
        ),
    )
    monkeypatch.setattr(
        search_module,
        "load_training_params",
        lambda model_name, params_path: {},
    )
    monkeypatch.setattr(
        search_module,
        "run_hyperparameter_optimization",
        lambda **kwargs: SimpleNamespace(
            best_value=0.5,
            best_params={},
            best_trial=SimpleNamespace(number=0, value=0.5, params={}),
        ),
    )
    monkeypatch.setattr(
        search_module,
        "logger",
        Mock(),
    )

    main()

    if should_load:
        load_dotenv.assert_called_once_with(env_file)
    else:
        load_dotenv.assert_not_called()
