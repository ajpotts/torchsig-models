"""Optuna/MLflow hyperparameter optimization for EfficientNet-1D."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any
import logging

import optuna
from dotenv import load_dotenv
from pytorch_lightning.loggers import CSVLogger

from torchsig.utils.yaml import load_config_from_yaml
from torchsig_models.models.iq_models.efficientnet.efficientnet1d_train import (
    EfficientNetModelName,
    load_training_params,
    train_efficientnet_iq,
)
from torchsig_models.utils.hyperparameter_search import (
    load_search_config,
    run_hyperparameter_optimization,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Optimize EfficientNet-1D hyperparameters with Optuna and "
            "optional MLflow tracking."
        )
    )

    parser.add_argument(
        "--dataset-config",
        type=Path,
        help=(
            "Dataset config used for train, validation, and test unless a "
            "split-specific config is provided."
        ),
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        help="Optional training dataset config.",
    )
    parser.add_argument(
        "--val-config",
        type=Path,
        help="Optional validation dataset config.",
    )
    parser.add_argument(
        "--test-config",
        type=Path,
        help="Optional test dataset config.",
    )
    parser.add_argument(
        "--search-config",
        type=Path,
        default=(
            Path(__file__).parent
            / "search_configs"
            / "efficientnet_b0_search_config.yaml"
        ),
    )
    parser.add_argument("--params", type=Path)
    parser.add_argument(
        "--model",
        choices=[
            "efficientnet_b0",
            "efficientnet_b2",
            "efficientnet_b4",
        ],
        default="efficientnet_b0",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/optimization"),
    )
    parser.add_argument("--dataset-length", type=int)
    parser.add_argument("--dataset-id")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-trials", type=int)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional .env file containing MLflow environment variables.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Override the maximum epochs for each Optuna trial.",
    )
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        help="Enable optional MLflow tracking in addition to local trial logs.",
    )
    parser.add_argument(
        "--mlflow-timeout",
        type=int,
        default=5,
        help="MLflow HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--mlflow-max-retries",
        type=int,
        default=0,
        help="Maximum retries for failed MLflow HTTP requests.",
    )

    args = parser.parse_args()

    if args.dataset_config is None and args.train_config is None:
        parser.error("one of --dataset-config or --train-config is required")

    return args


def _load_split_configs(
    *,
    dataset_config: Path | None,
    train_config: Path | None,
    val_config: Path | None,
    test_config: Path | None,
) -> tuple[Any, Any, Any]:
    """Load train, validation, and test dataset configurations.

    Split-specific configs take precedence over the shared dataset config.
    Missing validation and test configs fall back to the training config and
    receive deterministic seed offsets.
    """
    resolved_train_path = train_config or dataset_config

    if resolved_train_path is None:
        raise ValueError("A training dataset config must be provided.")

    resolved_val_path = val_config or dataset_config or resolved_train_path
    resolved_test_path = test_config or dataset_config or resolved_train_path

    train_cfg = load_config_from_yaml(resolved_train_path)
    val_cfg = load_config_from_yaml(resolved_val_path)
    test_cfg = load_config_from_yaml(resolved_test_path)

    if val_config is None:
        val_cfg = replace(
            val_cfg,
            seed=train_cfg.seed + 1,
        )

    if test_config is None:
        test_cfg = replace(
            test_cfg,
            seed=train_cfg.seed + 2,
        )

    return train_cfg, val_cfg, test_cfg


def _apply_dataset_overrides(
    cfg: Any,
    *,
    dataset_length: int | None,
    dataset_id: str | None,
) -> Any:
    """Apply optional dataset configuration overrides."""
    updates: dict[str, Any] = {}

    if dataset_length is not None:
        updates["dataset_length"] = dataset_length

    if dataset_id is not None:
        updates["dataset_id"] = dataset_id

    if not updates:
        return cfg

    return replace(cfg, **updates)


def _final_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Extract final scalar metrics from a training result."""
    metrics = result["metrics"]

    return {
        "val_f1": float(metrics.val_f1s[-1]),
        "val_acc": float(metrics.val_accuracies[-1]),
        "train_f1": float(metrics.train_f1s[-1]),
        "train_acc": float(metrics.train_accuracies[-1]),
    }


def main() -> None:
    """Run EfficientNet-1D hyperparameter optimization."""
    args = parse_args()

    if args.enable_mlflow and args.env_file.exists():
        load_dotenv(args.env_file)

    search_config = load_search_config(args.search_config)

    train_cfg, val_cfg, test_cfg = _load_split_configs(
        dataset_config=args.dataset_config,
        train_config=args.train_config,
        val_config=args.val_config,
        test_config=args.test_config,
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

    model_name: EfficientNetModelName = args.model

    base_params = load_training_params(
        model_name,
        params_path=args.params,
    )

    if args.max_epochs is not None:
        base_params["max_epochs"] = args.max_epochs

    metric_name = search_config.get(
        "metric_name",
        "val_f1",
    )
    direction = search_config.get(
        "direction",
        "maximize",
    )
    n_trials = (
        args.n_trials
        if args.n_trials is not None
        else search_config.get("n_trials", 20)
    )
    experiment_name = search_config.get(
        "experiment_name",
        "efficientnet1d_optimization",
    )
    run_name = search_config.get(
        "run_name",
        f"{model_name}_optimization",
    )
    search_space = search_config["search_space"]

    optimization_dir = args.output_dir / train_cfg.dataset_id / model_name

    def train_fn(
        params: dict[str, Any],
        trial_dir: Path,
        trial: optuna.Trial,
    ) -> dict[str, Any]:
        """Train one Optuna trial using local Lightning logging."""
        logger.info(
            f"Starting trial {trial.number} in {trial_dir.resolve()}",
            flush=True,
        )

        trial_params = params.copy()
        trial_params.pop("model_name", None)

        training_logger = CSVLogger(
            save_dir=trial_dir,
            name="lightning_logs",
            version="",
        )

        training_logger.log_hyperparams(
            {
                **trial_params,
                "trial_number": trial.number,
                "model_name": model_name,
                "train_dataset_id": train_cfg.dataset_id,
                "val_dataset_id": val_cfg.dataset_id,
                "test_dataset_id": test_cfg.dataset_id,
                "train_dataset_length": train_cfg.dataset_length,
                "val_dataset_length": val_cfg.dataset_length,
                "test_dataset_length": test_cfg.dataset_length,
                "train_seed": train_cfg.seed,
                "val_seed": val_cfg.seed,
                "test_seed": test_cfg.seed,
            }
        )

        result = train_efficientnet_iq(
            train_cfg=train_cfg,
            val_cfg=val_cfg,
            test_cfg=test_cfg,
            params=trial_params,
            checkpoint_dir=trial_dir / "checkpoints",
            metrics_dir=trial_dir / "metrics",
            dataset_root=args.dataset_root,
            overwrite=args.overwrite,
            model_name=model_name,
            logger=training_logger,
        )

        return {
            **result,
            **_final_metrics(result),
        }

    study = run_hyperparameter_optimization(
        base_params=base_params,
        search_space=search_space,
        train_fn=train_fn,
        metric_name=metric_name,
        direction=direction,
        n_trials=n_trials,
        experiment_name=experiment_name,
        run_name=run_name,
        output_dir=optimization_dir,
        mlflow_enabled=args.enable_mlflow,
        mlflow_timeout_seconds=args.mlflow_timeout,
        mlflow_max_retries=args.mlflow_max_retries,
    )

    logger.info("Optimization complete.")
    logger.info(f"Best {metric_name}: {study.best_value:.4f}")
    logger.info("Best params:")

    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
