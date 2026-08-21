"""Optuna and MLflow hyperparameter optimization for EfficientNet-2D."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import optuna
from dotenv import load_dotenv
from pytorch_lightning.loggers import MLFlowLogger

from torchsig.datasets.datasets import TorchSigDatasetConfig
from torchsig.utils.yaml import load_config_from_yaml

from torchsig_models.models.spectrogram_models.efficientnet.efficientnet_train import (
    EfficientNet2DModelName,
    load_training_params,
    train_efficientnet_2d,
)
from torchsig_models.utils.optimization import (
    load_search_config,
    optimize_params,
)


DEFAULT_OPTIMIZATION_CONFIG = (
    Path(__file__).parent
    / "optimization_configs"
    / "efficientnet_optimization.yaml"
)


def parse_args() -> argparse.Namespace:
    """Parse EfficientNet-2D optimization command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Optimize EfficientNet-2D hyperparameters with Optuna and MLflow."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TorchSig dataset configuration YAML.",
    )
    parser.add_argument(
        "--optimization-config",
        type=Path,
        default=DEFAULT_OPTIMIZATION_CONFIG,
        help="Path to the Optuna search configuration YAML.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        help="Optional training-parameter YAML.",
    )
    parser.add_argument(
        "--model",
        choices=[
            "efficientnet_b0",
            "efficientnet_b2",
            "efficientnet_b4",
        ],
        default="efficientnet_b0",
        help="EfficientNet-2D architecture to optimize.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets"),
        help="Root directory for generated static datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/optimization"),
        help="Root directory for optimization artifacts.",
    )
    parser.add_argument(
        "--dataset-length",
        type=int,
        help="Override the configured dataset length.",
    )
    parser.add_argument(
        "--dataset-id",
        help="Override the configured dataset identifier.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated datasets.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        help="Override the configured number of Optuna trials.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional file containing MLflow environment variables.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Override the maximum epochs for each Optuna trial.",
    )
    parser.add_argument(
        "--signal-generators",
        default="all",
        help="Signal generators used to create static datasets.",
    )

    return parser.parse_args()


def _load_split_configs(
    config_path: Path,
) -> tuple[
    TorchSigDatasetConfig,
    TorchSigDatasetConfig,
    TorchSigDatasetConfig,
]:
    """Load train, validation, and test configs with distinct seeds."""
    train_cfg = load_config_from_yaml(config_path)
    val_cfg = load_config_from_yaml(config_path)
    test_cfg = load_config_from_yaml(config_path)

    base_seed = train_cfg.seed

    val_cfg = replace(
        val_cfg,
        seed=base_seed + 1,
    )
    test_cfg = replace(
        test_cfg,
        seed=base_seed + 2,
    )

    return train_cfg, val_cfg, test_cfg


def _apply_dataset_overrides(
    cfg: TorchSigDatasetConfig,
    *,
    dataset_length: int | None,
    dataset_id: str | None,
) -> TorchSigDatasetConfig:
    """Apply command-line overrides to a dataset configuration."""
    updates: dict[str, Any] = {}

    if dataset_length is not None:
        updates["dataset_length"] = dataset_length

    if dataset_id is not None:
        updates["dataset_id"] = dataset_id

    if not updates:
        return cfg

    return replace(
        cfg,
        **updates,
    )


def _create_trial_logger(
    *,
    experiment_name: str,
    trial_number: int,
) -> MLFlowLogger:
    """Create an MLflow logger for one Optuna trial."""
    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=f"trial_{trial_number}",
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
    )


def _final_metrics(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Add final scalar training and validation metrics to a result."""
    metrics = result["metrics"]

    return {
        **result,
        "val_f1": float(metrics.val_f1s[-1]),
        "val_acc": float(metrics.val_accuracies[-1]),
        "train_f1": float(metrics.train_f1s[-1]),
        "train_acc": float(metrics.train_accuracies[-1]),
    }


def main() -> None:
    """Run EfficientNet-2D hyperparameter optimization."""
    args = parse_args()

    if args.env_file.exists():
        load_dotenv(args.env_file)

    optimization_config = load_search_config(
        args.optimization_config
    )

    train_cfg, val_cfg, test_cfg = _load_split_configs(
        args.config
    )

    train_cfg = _apply_dataset_overrides(
        train_cfg,
        dataset_length=args.dataset_length,
        dataset_id=args.dataset_id,
    )
    val_cfg = _apply_dataset_overrides(
        val_cfg,
        dataset_length=args.dataset_length,
        dataset_id=args.dataset_id,
    )
    test_cfg = _apply_dataset_overrides(
        test_cfg,
        dataset_length=args.dataset_length,
        dataset_id=args.dataset_id,
    )

    model_name: EfficientNet2DModelName = args.model

    base_params = load_training_params(
        model_name,
        params_path=args.params,
    )

    if args.max_epochs is not None:
        base_params["max_epochs"] = args.max_epochs

    metric_name = optimization_config.get(
        "metric_name",
        "val_f1",
    )
    direction = optimization_config.get(
        "direction",
        "maximize",
    )
    n_trials = (
        args.n_trials
        if args.n_trials is not None
        else optimization_config.get("n_trials", 20)
    )
    experiment_name = optimization_config.get(
        "experiment_name",
        "efficientnet_optimization",
    )
    run_name = optimization_config.get(
        "run_name",
        f"{model_name}_optimization",
    )
    search_space = optimization_config["search_space"]

    def train_fn(
        params: dict[str, Any],
        trial_dir: Path,
        trial: optuna.Trial,
    ) -> dict[str, Any]:
        """Train and evaluate one Optuna trial."""
        logger = _create_trial_logger(
            experiment_name=experiment_name,
            trial_number=trial.number,
        )

        trial_params = params.copy()
        trial_params.pop("model_name", None)

        logger.log_hyperparams(trial_params)
        logger.log_hyperparams(
            {
                "trial_number": trial.number,
                "model_name": model_name,
                "dataset_id": train_cfg.dataset_id,
                "dataset_length": train_cfg.dataset_length,
                "train_seed": train_cfg.seed,
                "val_seed": val_cfg.seed,
                "test_seed": test_cfg.seed,
            }
        )

        result = train_efficientnet_2d(
            train_cfg=train_cfg,
            val_cfg=val_cfg,
            test_cfg=test_cfg,
            params=trial_params,
            checkpoint_dir=trial_dir / "checkpoints",
            metrics_dir=trial_dir / "metrics",
            dataset_root=args.dataset_root,
            overwrite=args.overwrite,
            model_name=model_name,
            signal_generators=args.signal_generators,
            logger=logger,
        )

        return _final_metrics(result)

    study = optimize_params(
        base_params=base_params,
        search_space=search_space,
        train_fn=train_fn,
        metric_name=metric_name,
        direction=direction,
        n_trials=n_trials,
        experiment_name=experiment_name,
        run_name=run_name,
        output_dir=(
            args.output_dir
            / train_cfg.dataset_id
            / model_name
        ),
    )

    print("Optimization complete.")
    print(f"Best {metric_name}: {study.best_value:.4f}")
    print("Best params:")

    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()