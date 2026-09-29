"""Optuna/MLflow hyperparameter optimization for DETR wideband detection."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import optuna
import yaml
from dotenv import load_dotenv
from pytorch_lightning.loggers import CSVLogger
from torchsig.utils.yaml import load_config_from_yaml

from torchsig_models.models.spectrogram_models.detr.detr_train import (
    DETRModelName,
    MODEL_FACTORY,
    load_training_params,
    train_detr,
)
from torchsig_models.utils.hyperparameter_search import (
    load_search_config,
    run_hyperparameter_optimization,
)


def parse_args() -> argparse.Namespace:
    """Parse DETR tuning command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--val-config", type=Path)
    parser.add_argument("--test-config", type=Path)
    parser.add_argument(
        "--search-config",
        type=Path,
        default=Path(__file__).parent / "search_configs" / "detr_b0_nano_search_config.yaml",
    )
    parser.add_argument("--params", type=Path)
    parser.add_argument("--model", choices=list(MODEL_FACTORY), default="detr_b0_nano")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/optimization"))
    parser.add_argument("--dataset-length", type=int)
    parser.add_argument("--dataset-id")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--signal-generators", nargs="+", default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--mlflow-timeout", type=int, default=5)
    parser.add_argument("--mlflow-max-retries", type=int, default=0)
    args = parser.parse_args()
    if args.dataset_config is None and args.train_config is None:
        parser.error("one of --dataset-config or --train-config is required")
    return args


def _configs(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    train_path = args.train_config or args.dataset_config
    assert train_path is not None
    train_cfg = load_config_from_yaml(train_path)
    val_cfg = load_config_from_yaml(args.val_config or args.dataset_config or train_path)
    test_cfg = load_config_from_yaml(args.test_config or args.dataset_config or train_path)
    if args.val_config is None:
        val_cfg = replace(val_cfg, seed=train_cfg.seed + 1)
    if args.test_config is None:
        test_cfg = replace(test_cfg, seed=train_cfg.seed + 2)
    updates = {
        key: value
        for key, value in {"dataset_length": args.dataset_length, "dataset_id": args.dataset_id}.items()
        if value is not None
    }
    if updates:
        train_cfg, val_cfg, test_cfg = (replace(cfg, **updates) for cfg in (train_cfg, val_cfg, test_cfg))
    return train_cfg, val_cfg, test_cfg


def _write_summary(
    study: optuna.Study, metric_name: str, base_params: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "best_trial.yaml").write_text(
        yaml.safe_dump(
            {
                "trial_number": study.best_trial.number,
                "metric_name": metric_name,
                "metric_value": study.best_value,
                "parameters": study.best_params,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "best_training_params.yaml").write_text(
        yaml.safe_dump({**base_params, **study.best_params}, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    """Run DETR hyperparameter optimization."""
    args = parse_args()
    if args.enable_mlflow and args.env_file.exists():
        load_dotenv(args.env_file)
    search_config = load_search_config(args.search_config)
    train_cfg, val_cfg, test_cfg = _configs(args)
    model_name: DETRModelName = args.model
    base_params = load_training_params(model_name, args.params)
    if args.max_epochs is not None:
        base_params["max_epochs"] = args.max_epochs
    metric_name = search_config.get("metric_name", "val_loss")
    output_dir = args.output_dir / train_cfg.dataset_id / model_name

    def train_fn(
        params: dict[str, Any], trial_dir: Path, trial: optuna.Trial
    ) -> dict[str, Any]:
        training_logger = CSVLogger(trial_dir, name="lightning_logs", version="")
        training_logger.log_hyperparams({**params, "trial_number": trial.number})
        return train_detr(
            train_cfg,
            val_cfg,
            test_cfg,
            params,
            trial_dir / "checkpoints",
            dataset_root=args.dataset_root,
            overwrite=args.overwrite,
            model_name=model_name,
            signal_generators=args.signal_generators,
            logger=training_logger,
        )

    study = run_hyperparameter_optimization(
        base_params=base_params,
        search_space=search_config["search_space"],
        train_fn=train_fn,
        metric_name=metric_name,
        direction=search_config.get("direction", "minimize"),
        n_trials=args.n_trials or search_config.get("n_trials", 20),
        experiment_name=search_config.get("experiment_name", "detr_optimization"),
        run_name=search_config.get("run_name", f"{model_name}_optimization"),
        output_dir=output_dir,
        mlflow_enabled=args.enable_mlflow,
        mlflow_timeout_seconds=args.mlflow_timeout,
        mlflow_max_retries=args.mlflow_max_retries,
    )
    _write_summary(study, metric_name, base_params, output_dir)


if __name__ == "__main__":
    main()
