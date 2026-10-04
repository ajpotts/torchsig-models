"""Optuna/MLflow hyperparameter optimization for Ultralytics RT-DETR."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import optuna
import yaml
from dotenv import load_dotenv
from torchsig.utils.yaml import load_config_from_yaml

from torchsig_models.models.spectrogram_models.detr.detr_train import (
    DETRModelName,
    load_training_params,
    train_detr,
)
from torchsig_models.utils.hyperparameter_search import (
    load_search_config,
    run_hyperparameter_optimization,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse RT-DETR tuning arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--val-config", type=Path)
    parser.add_argument("--test-config", type=Path)
    parser.add_argument(
        "--search-config",
        type=Path,
        default=Path(__file__).parent / "search_configs" / "rtdetr_l_search_config.yaml",
    )
    parser.add_argument("--params", type=Path)
    parser.add_argument("--model", choices=["rtdetr_l"], default="rtdetr_l")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/optimization"))
    parser.add_argument("--dataset-length", type=int)
    parser.add_argument("--dataset-id")
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--signal-generators", nargs="+", default="all")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-timeout", type=int, default=5)
    parser.add_argument("--mlflow-max-retries", type=int, default=0)
    args = parser.parse_args()
    if args.dataset_config is None and args.train_config is None:
        parser.error("one of --dataset-config or --train-config is required")
    return args


def _load_split_configs(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    train_path = args.train_config or args.dataset_config
    if train_path is None:
        raise ValueError("A training dataset config must be provided.")
    train_cfg = load_config_from_yaml(train_path)
    val_cfg = load_config_from_yaml(args.val_config or args.dataset_config or train_path)
    test_cfg = load_config_from_yaml(args.test_config or args.dataset_config or train_path)
    if args.val_config is None:
        val_cfg = replace(val_cfg, seed=train_cfg.seed + 1)
    if args.test_config is None:
        test_cfg = replace(test_cfg, seed=train_cfg.seed + 2)
    updates = {
        key: value
        for key, value in {
            "dataset_length": args.dataset_length,
            "dataset_id": args.dataset_id,
        }.items()
        if value is not None
    }
    if updates:
        train_cfg, val_cfg, test_cfg = (
            replace(cfg, **updates) for cfg in (train_cfg, val_cfg, test_cfg)
        )
    return train_cfg, val_cfg, test_cfg


def _write_best_trial_summary(
    study: optuna.Study,
    metric_name: str,
    base_params: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write best-trial provenance and reusable RT-DETR parameters."""
    best_trial = study.best_trial
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "best_trial.yaml"
    summary_path.write_text(
        yaml.safe_dump(
            {
                "trial_number": best_trial.number,
                "metric_name": metric_name,
                "metric_value": best_trial.value,
                "parameters": best_trial.params,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    params_path = output_dir / "best_training_params.yaml"
    params_path.write_text(
        yaml.safe_dump({**base_params, **best_trial.params}, sort_keys=False),
        encoding="utf-8",
    )
    return summary_path, params_path


def main() -> None:
    """Run RT-DETR hyperparameter optimization."""
    args = parse_args()
    if args.enable_mlflow and args.env_file.exists():
        load_dotenv(args.env_file)
    search_config = load_search_config(args.search_config)
    train_cfg, val_cfg, test_cfg = _load_split_configs(args)
    model_name: DETRModelName = args.model
    base_params = load_training_params(model_name, args.params)
    if args.max_epochs is not None:
        base_params["max_epochs"] = args.max_epochs

    metric_name = search_config.get("metric_name", "val_map_50")
    direction = search_config.get("direction", "maximize")
    n_trials = args.n_trials or search_config.get("n_trials", 20)
    optimization_dir = args.output_dir / train_cfg.dataset_id / model_name

    def train_fn(
        params: dict[str, Any], trial_dir: Path, trial: optuna.Trial
    ) -> dict[str, Any]:
        logger.info("Starting trial %s in %s", trial.number, trial_dir.resolve())
        result = train_detr(
            train_cfg,
            val_cfg,
            test_cfg,
            params,
            trial_dir,
            dataset_root=args.dataset_root,
            overwrite=args.overwrite,
            model_name=model_name,
            signal_generators=args.signal_generators,
            device=args.device,
            workers=args.workers,
            evaluate_test=False,
        )
        return {
            **result,
            "val_map": result["val_summary"]["map"],
            "val_map_50": result["val_summary"]["map_50"],
            "val_precision": result["val_summary"]["precision"],
            "val_recall": result["val_summary"]["recall"],
        }

    study = run_hyperparameter_optimization(
        base_params=base_params,
        search_space=search_config["search_space"],
        train_fn=train_fn,
        metric_name=metric_name,
        direction=direction,
        n_trials=n_trials,
        experiment_name=search_config.get("experiment_name", "rtdetr_optimization"),
        run_name=search_config.get("run_name", f"{model_name}_optimization"),
        output_dir=optimization_dir,
        mlflow_enabled=args.enable_mlflow,
        mlflow_timeout_seconds=args.mlflow_timeout,
        mlflow_max_retries=args.mlflow_max_retries,
    )
    summary, params = _write_best_trial_summary(
        study, metric_name, base_params, optimization_dir
    )
    logger.info("Best-trial summary saved to %s", summary)
    logger.info("Best training parameters saved to %s", params)


if __name__ == "__main__":
    main()
