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
    _detection_collate,
    _spectrogram_transforms,
    load_training_params,
    train_detr,
)
from torchsig_models.utils.datasets import (
    default_wideband_config,
    prepare_torchsig_datasets,
)
from torchsig_models.utils.hyperparameter_search import (
    load_search_config,
    run_hyperparameter_optimization,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse RT-DETR optimization arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config", type=Path,
        help="Shared dataset config unless a split-specific config is provided.",
    )
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--val-config", type=Path)
    parser.add_argument("--test-config", type=Path)
    parser.add_argument(
        "--search-config", type=Path,
        default=Path(__file__).parent / "search_configs" / "rtdetr_l_search_config.yaml",
    )
    parser.add_argument("--params", type=Path)
    parser.add_argument("--model", choices=["rtdetr_l"], default="rtdetr_l")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/optimization"))
    parser.add_argument(
        "--dataset-length", type=int,
        help="Override generated dataset length or limit existing splits.",
    )
    parser.add_argument("--dataset-id")
    parser.add_argument(
        "--storage-backend", choices=["legacy", "packed", "homogeneous"],
        help="Override the storage backend configured in YAML.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--existing-datasets", action="store_true",
        help="Require existing static splits; never generate datasets.",
    )
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--signal-generators", nargs="+", default="all")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-timeout", type=int, default=5)
    parser.add_argument("--mlflow-max-retries", type=int, default=0)

    args = parser.parse_args()
    if args.dataset_config is None:
        args.train_config = args.train_config or default_wideband_config("train")
        args.val_config = args.val_config or default_wideband_config("val")
        args.test_config = args.test_config or default_wideband_config("val")
    if args.overwrite and args.existing_datasets:
        parser.error("--overwrite cannot be combined with --existing-datasets")
    return args


def _load_split_configs(
    *,
    dataset_config: Path | None,
    train_config: Path | None,
    val_config: Path | None,
    test_config: Path | None,
) -> tuple[Any, Any, Any]:
    """Load split configs, prioritizing explicit split paths over the shared path."""
    resolved_train_path = train_config or dataset_config
    if resolved_train_path is None:
        raise ValueError("A training dataset config must be provided.")
    resolved_val_path = val_config or dataset_config or resolved_train_path
    resolved_test_path = test_config or dataset_config or resolved_train_path

    train_cfg = load_config_from_yaml(resolved_train_path)
    val_cfg = load_config_from_yaml(resolved_val_path)
    test_cfg = load_config_from_yaml(resolved_test_path)
    if val_config is None:
        val_cfg = replace(val_cfg, seed=train_cfg.seed + 1)
    if test_config is None:
        test_cfg = replace(test_cfg, seed=train_cfg.seed + 2)
    return train_cfg, val_cfg, test_cfg


def _apply_dataset_overrides(
    cfg: Any, *, dataset_length: int | None, dataset_id: str | None
) -> Any:
    """Apply command-line dataset overrides without mutating the source config."""
    updates: dict[str, Any] = {}
    if dataset_length is not None:
        updates["dataset_length"] = dataset_length
    if dataset_id is not None:
        updates["dataset_id"] = dataset_id
    return replace(cfg, **updates) if updates else cfg


def _final_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Expose RT-DETR validation metrics to the Optuna objective."""
    summary = result["val_summary"]
    return {
        "val_map": float(summary["map"]),
        "val_map_50": float(summary["map_50"]),
        "val_precision": float(summary["precision"]),
        "val_recall": float(summary["recall"]),
    }


def _write_best_trial_summary(
    study: optuna.Study,
    metric_name: str,
    base_params: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write best-trial provenance and reusable training parameters."""
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
    train_cfg, val_cfg, test_cfg = _load_split_configs(
        dataset_config=args.dataset_config,
        train_config=args.train_config,
        val_config=args.val_config,
        test_config=args.test_config,
    )
    train_cfg = _apply_dataset_overrides(
        train_cfg, dataset_length=args.dataset_length, dataset_id=args.dataset_id
    )
    val_cfg = _apply_dataset_overrides(
        val_cfg, dataset_length=args.dataset_length, dataset_id=args.dataset_id
    )
    test_cfg = _apply_dataset_overrides(
        test_cfg, dataset_length=args.dataset_length, dataset_id=args.dataset_id
    )

    model_name: DETRModelName = args.model
    base_params = load_training_params(model_name, args.params)
    if args.max_epochs is not None:
        base_params["max_epochs"] = args.max_epochs

    metric_name = search_config.get("metric_name", "val_map_50")
    direction = search_config.get("direction", "maximize")
    n_trials = (
        args.n_trials if args.n_trials is not None
        else search_config.get("n_trials", 20)
    )
    experiment_name = search_config.get("experiment_name", "rtdetr_optimization")
    run_name = search_config.get("run_name", f"{model_name}_optimization")
    search_space = search_config["search_space"]
    optimization_dir = args.output_dir / train_cfg.dataset_id / model_name

    # Generate or resolve datasets only once; trials must reuse static splits.
    prepare_torchsig_datasets(
        train_cfg,
        val_cfg,
        test_cfg,
        dataset_root=args.dataset_root,
        batch_size=int(base_params.get("batch_size", 2)),
        overwrite=args.overwrite,
        signal_generators=args.signal_generators,
        transforms=_spectrogram_transforms(train_cfg),
        target_labels=["yolo_label"],
        collate_fn=_detection_collate,
        dataset_mode="existing" if args.existing_datasets else "auto",
        storage_backend=args.storage_backend,
    )

    def train_fn(
        params: dict[str, Any], trial_dir: Path, trial: optuna.Trial
    ) -> dict[str, Any]:
        """Train one trial against the existing TorchSig splits."""
        logger.info("Starting trial %s in %s", trial.number, trial_dir.resolve())
        trial_params = params.copy()
        trial_params.pop("model_name", None)
        result = train_detr(
            train_cfg,
            val_cfg,
            test_cfg,
            trial_params,
            trial_dir,
            dataset_root=args.dataset_root,
            overwrite=args.overwrite,
            model_name=model_name,
            signal_generators=args.signal_generators,
            dataset_mode="existing",
            storage_backend=args.storage_backend,
            device=args.device,
            workers=args.workers,
            evaluate_test=False,
        )
        return {**result, **_final_metrics(result)}

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
    summary_path, params_path = _write_best_trial_summary(
        study, metric_name, base_params, optimization_dir
    )
    logger.info("Best-trial summary saved to %s", summary_path)
    logger.info("Best training parameters saved to %s", params_path)


if __name__ == "__main__":
    main()
