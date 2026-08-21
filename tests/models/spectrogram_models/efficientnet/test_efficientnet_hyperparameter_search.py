"""Tests for EfficientNet-2D hyperparameter optimization."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

import torchsig_models.models.spectrogram_models.efficientnet.efficientnet_hyperparameter_search as search_module
from torchsig_models.models.spectrogram_models.efficientnet.efficientnet_hyperparameter_search import (
    DEFAULT_OPTIMIZATION_CONFIG,
    _apply_dataset_overrides,
    _create_trial_logger,
    _final_metrics,
    _load_split_configs,
    main,
    parse_args,
)


@dataclass(frozen=True)
class DummyConfig:
    """Minimal immutable dataset configuration for testing."""

    dataset_id: str = "dummy_dataset"
    dataset_length: int = 100
    seed: int = 42


def test_parse_args_uses_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify required arguments and optimization defaults."""
    config_path = tmp_path / "dataset.yaml"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "efficientnet_hyperparameter_search.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.config == config_path
    assert args.optimization_config == DEFAULT_OPTIMIZATION_CONFIG
    assert args.params is None
    assert args.model == "efficientnet_b0"
    assert args.dataset_root == Path("datasets")
    assert args.output_dir == Path("runs/optimization")
    assert args.dataset_length is None
    assert args.dataset_id is None
    assert args.overwrite is False
    assert args.n_trials is None
    assert args.env_file == Path(".env")
    assert args.max_epochs is None
    assert args.signal_generators == "all"


def test_parse_args_reads_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify all supported command-line overrides are parsed."""
    config_path = tmp_path / "dataset.yaml"
    optimization_path = tmp_path / "optimization.yaml"
    params_path = tmp_path / "params.yaml"
    env_path = tmp_path / "mlflow.env"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "efficientnet_hyperparameter_search.py",
            "--config",
            str(config_path),
            "--optimization-config",
            str(optimization_path),
            "--params",
            str(params_path),
            "--model",
            "efficientnet_b4",
            "--dataset-root",
            "custom-datasets",
            "--output-dir",
            "custom-runs",
            "--dataset-length",
            "500",
            "--dataset-id",
            "custom_dataset",
            "--overwrite",
            "--n-trials",
            "12",
            "--env-file",
            str(env_path),
            "--max-epochs",
            "7",
            "--signal-generators",
            "fm-data",
        ],
    )

    args = parse_args()

    assert args.config == config_path
    assert args.optimization_config == optimization_path
    assert args.params == params_path
    assert args.model == "efficientnet_b4"
    assert args.dataset_root == Path("custom-datasets")
    assert args.output_dir == Path("custom-runs")
    assert args.dataset_length == 500
    assert args.dataset_id == "custom_dataset"
    assert args.overwrite is True
    assert args.n_trials == 12
    assert args.env_file == env_path
    assert args.max_epochs == 7
    assert args.signal_generators == "fm-data"


def test_load_split_configs_assigns_distinct_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify train, validation, and test configs use consecutive seeds."""
    config_path = tmp_path / "dataset.yaml"
    load_config = MagicMock(
        side_effect=[
            DummyConfig(seed=10),
            DummyConfig(seed=10),
            DummyConfig(seed=10),
        ]
    )
    monkeypatch.setattr(
        search_module,
        "load_config_from_yaml",
        load_config,
    )

    train_cfg, val_cfg, test_cfg = _load_split_configs(config_path)

    assert load_config.call_args_list == [
        call(config_path),
        call(config_path),
        call(config_path),
    ]
    assert train_cfg.seed == 10
    assert val_cfg.seed == 11
    assert test_cfg.seed == 12


def test_apply_dataset_overrides_updates_requested_values() -> None:
    """Verify dataset length and identifier overrides are applied."""
    cfg = DummyConfig()

    result = _apply_dataset_overrides(
        cfg,
        dataset_length=250,
        dataset_id="overridden_dataset",
    )

    assert result.dataset_length == 250
    assert result.dataset_id == "overridden_dataset"
    assert result.seed == cfg.seed
    assert result is not cfg


def test_apply_dataset_overrides_preserves_unset_values() -> None:
    """Verify a partial override leaves other values unchanged."""
    cfg = DummyConfig()

    result = _apply_dataset_overrides(
        cfg,
        dataset_length=250,
        dataset_id=None,
    )

    assert result.dataset_length == 250
    assert result.dataset_id == cfg.dataset_id


def test_apply_dataset_overrides_returns_same_config_without_updates() -> None:
    """Verify no replacement occurs when no overrides are supplied."""
    cfg = DummyConfig()

    result = _apply_dataset_overrides(
        cfg,
        dataset_length=None,
        dataset_id=None,
    )

    assert result is cfg


def test_create_trial_logger_uses_mlflow_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify trial logger construction uses the MLflow tracking URI."""
    logger = MagicMock()
    logger_cls = MagicMock(return_value=logger)

    monkeypatch.setattr(
        search_module,
        "MLFlowLogger",
        logger_cls,
    )
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "http://mlflow.example",
    )

    result = _create_trial_logger(
        experiment_name="efficientnet-tests",
        trial_number=4,
    )

    logger_cls.assert_called_once_with(
        experiment_name="efficientnet-tests",
        run_name="trial_4",
        tracking_uri="http://mlflow.example",
    )
    assert result is logger


def test_final_metrics_adds_latest_scalar_metrics() -> None:
    """Verify final metric-history values are added to the result."""
    metrics = SimpleNamespace(
        val_f1s=[0.5, 0.8],
        val_accuracies=[0.6, 0.85],
        train_f1s=[0.7, 0.9],
        train_accuracies=[0.75, 0.95],
    )
    result = {
        "model": object(),
        "metrics": metrics,
    }

    final_result = _final_metrics(result)

    assert final_result["model"] is result["model"]
    assert final_result["metrics"] is metrics
    assert final_result["val_f1"] == pytest.approx(0.8)
    assert final_result["val_acc"] == pytest.approx(0.85)
    assert final_result["train_f1"] == pytest.approx(0.9)
    assert final_result["train_acc"] == pytest.approx(0.95)


def test_main_runs_optimization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify the complete optimization workflow and trial callback."""
    config_path = tmp_path / "dataset.yaml"
    optimization_path = tmp_path / "optimization.yaml"
    params_path = tmp_path / "params.yaml"
    env_path = tmp_path / ".env"
    env_path.touch()

    dataset_root = tmp_path / "datasets"
    output_dir = tmp_path / "runs"

    args = SimpleNamespace(
        config=config_path,
        optimization_config=optimization_path,
        params=params_path,
        model="efficientnet_b2",
        dataset_root=dataset_root,
        output_dir=output_dir,
        dataset_length=500,
        dataset_id="overridden_dataset",
        overwrite=True,
        n_trials=7,
        env_file=env_path,
        max_epochs=3,
        signal_generators="fm-data",
    )
    monkeypatch.setattr(
        search_module,
        "parse_args",
        MagicMock(return_value=args),
    )

    load_dotenv = MagicMock()
    monkeypatch.setattr(
        search_module,
        "load_dotenv",
        load_dotenv,
    )

    optimization_config = {
        "metric_name": "val_acc",
        "direction": "maximize",
        "n_trials": 20,
        "experiment_name": "efficientnet2d_test",
        "run_name": "test_search",
        "search_space": {
            "learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-3,
            }
        },
    }
    load_search_config = MagicMock(
        return_value=optimization_config
    )
    monkeypatch.setattr(
        search_module,
        "load_search_config",
        load_search_config,
    )

    train_cfg = DummyConfig(seed=20)
    val_cfg = DummyConfig(seed=21)
    test_cfg = DummyConfig(seed=22)
    load_split_configs = MagicMock(
        return_value=(train_cfg, val_cfg, test_cfg)
    )
    monkeypatch.setattr(
        search_module,
        "_load_split_configs",
        load_split_configs,
    )

    base_params = {
        "learning_rate": 1e-3,
        "max_epochs": 10,
    }
    load_training_params = MagicMock(
        return_value=base_params
    )
    monkeypatch.setattr(
        search_module,
        "load_training_params",
        load_training_params,
    )

    logger = MagicMock()
    create_trial_logger = MagicMock(return_value=logger)
    monkeypatch.setattr(
        search_module,
        "_create_trial_logger",
        create_trial_logger,
    )

    metrics = SimpleNamespace(
        val_f1s=[0.81],
        val_accuracies=[0.84],
        train_f1s=[0.91],
        train_accuracies=[0.94],
    )
    training_result = {
        "model": object(),
        "metrics": metrics,
    }
    train_efficientnet = MagicMock(
        return_value=training_result
    )
    monkeypatch.setattr(
        search_module,
        "train_efficientnet_2d",
        train_efficientnet,
    )

    study = SimpleNamespace(
        best_value=0.84,
        best_params={
            "learning_rate": 5e-4,
            "batch_size": 32,
        },
    )

    def fake_optimize_params(**kwargs):
        trial_params = {
            "model_name": "efficientnet_b2",
            "learning_rate": 5e-4,
            "batch_size": 32,
        }
        trial_dir = tmp_path / "trial_3"
        trial = SimpleNamespace(number=3)

        result = kwargs["train_fn"](
            trial_params,
            trial_dir,
            trial,
        )

        assert result["val_f1"] == pytest.approx(0.81)
        assert result["val_acc"] == pytest.approx(0.84)
        assert result["train_f1"] == pytest.approx(0.91)
        assert result["train_acc"] == pytest.approx(0.94)

        # The callback must not mutate the dictionary supplied by Optuna.
        assert trial_params["model_name"] == "efficientnet_b2"

        return study

    optimize_params = MagicMock(
        side_effect=fake_optimize_params
    )
    monkeypatch.setattr(
        search_module,
        "optimize_params",
        optimize_params,
    )

    main()

    load_dotenv.assert_called_once_with(env_path)
    load_search_config.assert_called_once_with(
        optimization_path
    )
    load_split_configs.assert_called_once_with(config_path)
    load_training_params.assert_called_once_with(
        "efficientnet_b2",
        params_path=params_path,
    )

    assert base_params["max_epochs"] == 3

    create_trial_logger.assert_called_once_with(
        experiment_name="efficientnet2d_test",
        trial_number=3,
    )

    logger.log_hyperparams.assert_has_calls(
        [
            call(
                {
                    "learning_rate": 5e-4,
                    "batch_size": 32,
                }
            ),
            call(
                {
                    "trial_number": 3,
                    "model_name": "efficientnet_b2",
                    "dataset_id": "overridden_dataset",
                    "dataset_length": 500,
                    "train_seed": 20,
                    "val_seed": 21,
                    "test_seed": 22,
                }
            ),
        ]
    )

    train_efficientnet.assert_called_once_with(
        train_cfg=DummyConfig(
            dataset_id="overridden_dataset",
            dataset_length=500,
            seed=20,
        ),
        val_cfg=DummyConfig(
            dataset_id="overridden_dataset",
            dataset_length=500,
            seed=21,
        ),
        test_cfg=DummyConfig(
            dataset_id="overridden_dataset",
            dataset_length=500,
            seed=22,
        ),
        params={
            "learning_rate": 5e-4,
            "batch_size": 32,
        },
        checkpoint_dir=tmp_path / "trial_3" / "checkpoints",
        metrics_dir=tmp_path / "trial_3" / "metrics",
        dataset_root=dataset_root,
        overwrite=True,
        model_name="efficientnet_b2",
        signal_generators="fm-data",
        logger=logger,
    )

    optimize_params.assert_called_once()
    optimize_kwargs = optimize_params.call_args.kwargs

    assert optimize_kwargs["base_params"] == {
        "learning_rate": 1e-3,
        "max_epochs": 3,
    }
    assert (
        optimize_kwargs["search_space"]
        == optimization_config["search_space"]
    )
    assert optimize_kwargs["metric_name"] == "val_acc"
    assert optimize_kwargs["direction"] == "maximize"
    assert optimize_kwargs["n_trials"] == 7
    assert (
        optimize_kwargs["experiment_name"]
        == "efficientnet2d_test"
    )
    assert optimize_kwargs["run_name"] == "test_search"
    assert optimize_kwargs["output_dir"] == (
        output_dir
        / "overridden_dataset"
        / "efficientnet_b2"
    )
    assert callable(optimize_kwargs["train_fn"])

    output = capsys.readouterr().out

    assert "Optimization complete." in output
    assert "Best val_acc: 0.8400" in output
    assert "learning_rate: 0.0005" in output
    assert "batch_size: 32" in output


def test_main_skips_missing_env_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify dotenv loading is skipped when the file does not exist."""
    args = SimpleNamespace(
        config=tmp_path / "dataset.yaml",
        optimization_config=tmp_path / "optimization.yaml",
        params=None,
        model="efficientnet_b0",
        dataset_root=tmp_path / "datasets",
        output_dir=tmp_path / "runs",
        dataset_length=None,
        dataset_id=None,
        overwrite=False,
        n_trials=None,
        env_file=tmp_path / "missing.env",
        max_epochs=None,
        signal_generators="all",
    )
    monkeypatch.setattr(
        search_module,
        "parse_args",
        MagicMock(return_value=args),
    )

    load_dotenv = MagicMock()
    monkeypatch.setattr(
        search_module,
        "load_dotenv",
        load_dotenv,
    )

    config = DummyConfig()
    monkeypatch.setattr(
        search_module,
        "_load_split_configs",
        MagicMock(return_value=(config, config, config)),
    )
    monkeypatch.setattr(
        search_module,
        "load_search_config",
        MagicMock(
            return_value={
                "search_space": {},
            }
        ),
    )
    monkeypatch.setattr(
        search_module,
        "load_training_params",
        MagicMock(return_value={}),
    )
    monkeypatch.setattr(
        search_module,
        "optimize_params",
        MagicMock(
            return_value=SimpleNamespace(
                best_value=0.5,
                best_params={},
            )
        ),
    )

    main()

    load_dotenv.assert_not_called()
