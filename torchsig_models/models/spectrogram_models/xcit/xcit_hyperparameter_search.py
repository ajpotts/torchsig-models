"""Optuna hyperparameter search for XCiT spectrogram classification."""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import optuna
import yaml
from dotenv import load_dotenv
from pytorch_lightning.loggers import CSVLogger
from torchsig.utils.yaml import load_config_from_yaml

from torchsig_models.models.spectrogram_models.xcit.xcit_train import (
    load_training_params,
    train_xcit_2d,
)
from torchsig_models.utils.hyperparameter_search import (
    load_search_config,
    run_hyperparameter_optimization,
)

DEFAULT_SEARCH_CONFIG = (
    Path(__file__).parent / "search_configs" / "xcit_nano_search_config.yaml"
)


def parse_args() -> argparse.Namespace:
    """Parse XCiT hyperparameter-search arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--val-config", type=Path)
    parser.add_argument("--test-config", type=Path)
    parser.add_argument("--search-config", type=Path, default=DEFAULT_SEARCH_CONFIG)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/optimization"))
    parser.add_argument("--dataset-length", type=int)
    parser.add_argument("--dataset-id")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--signal-generators", nargs="+", default="all")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-timeout", type=int, default=5)
    parser.add_argument("--mlflow-max-retries", type=int, default=0)
    args = parser.parse_args()
    if args.dataset_config is None and args.train_config is None:
        parser.error("one of --dataset-config or --train-config is required")
    return args


def _load_split_configs(
    dataset_config: Path | None,
    train_config: Path | None,
    val_config: Path | None,
    test_config: Path | None,
) -> tuple[Any, Any, Any]:
    """Load split configs, applying deterministic seeds to shared configs."""
    train_path = train_config or dataset_config
    if train_path is None:
        raise ValueError("A training dataset config must be provided")
    train_cfg = load_config_from_yaml(train_path)
    val_cfg = load_config_from_yaml(val_config or dataset_config or train_path)
    test_cfg = load_config_from_yaml(test_config or dataset_config or train_path)
    if val_config is None:
        val_cfg = replace(val_cfg, seed=train_cfg.seed + 1)
    if test_config is None:
        test_cfg = replace(test_cfg, seed=train_cfg.seed + 2)
    return train_cfg, val_cfg, test_cfg


def _apply_overrides(cfg: Any, length: int | None, dataset_id: str | None) -> Any:
    """Apply optional dataset length and identifier overrides."""
    updates = {}
    if length is not None:
        updates["dataset_length"] = length
    if dataset_id is not None:
        updates["dataset_id"] = dataset_id
    return replace(cfg, **updates) if updates else cfg


def _final_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Extract final scalar metrics from a training result."""
    metrics = result["metrics"]
    return {
        "val_f1": float(metrics.val_f1s[-1]),
        "val_acc": float(metrics.val_accuracies[-1]),
        "train_f1": float(metrics.train_f1s[-1]),
        "train_acc": float(metrics.train_accuracies[-1]),
    }


def _write_best_trial(
    study: optuna.Study,
    metric_name: str,
    base_params: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write search provenance and reusable best training parameters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "trial_number": study.best_trial.number,
        "metric_name": metric_name,
        "metric_value": study.best_trial.value,
        "parameters": study.best_params,
    }
    (output_dir / "best_trial.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "best_training_params.yaml").write_text(
        yaml.safe_dump({**base_params, **study.best_params}, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    """Run XCiT-Nano hyperparameter optimization."""
    args = parse_args()
    if args.env_file.exists():
        load_dotenv(args.env_file)
    config = load_search_config(args.search_config)
    train_cfg, val_cfg, test_cfg = _load_split_configs(
        args.dataset_config,
        args.train_config,
        args.val_config,
        args.test_config,
    )
    train_cfg = _apply_overrides(train_cfg, args.dataset_length, args.dataset_id)
    val_cfg = _apply_overrides(val_cfg, args.dataset_length, args.dataset_id)
    test_cfg = _apply_overrides(test_cfg, args.dataset_length, args.dataset_id)
    base_params = load_training_params(args.params)
    if args.max_epochs is not None:
        base_params["max_epochs"] = args.max_epochs

    metric_name = config.get("metric_name", "val_f1")
    output_dir = args.output_dir / train_cfg.dataset_id / "xcit_nano"

    def train_fn(
        params: dict[str, Any], trial_dir: Path, trial: optuna.Trial
    ) -> dict[str, float]:
        logger = CSVLogger(save_dir=trial_dir, name="logs", version=trial.number)
        result = train_xcit_2d(
            train_cfg,
            val_cfg,
            test_cfg,
            params,
            trial_dir / "checkpoints",
            metrics_dir=trial_dir / "metrics",
            dataset_root=args.dataset_root,
            overwrite=args.overwrite,
            signal_generators=args.signal_generators,
            logger=logger,
        )
        return _final_metrics(result)

    study = run_hyperparameter_optimization(
        base_params=base_params,
        search_space=config["search_space"],
        train_fn=train_fn,
        metric_name=metric_name,
        direction=config.get("direction", "maximize"),
        n_trials=args.n_trials or config.get("n_trials", 20),
        experiment_name=config.get("experiment_name", "xcit_nano_optimization"),
        run_name=config.get("run_name", "xcit_nano_search"),
        output_dir=output_dir,
        mlflow_enabled=args.enable_mlflow,
        mlflow_timeout_seconds=args.mlflow_timeout,
        mlflow_max_retries=args.mlflow_max_retries,
    )
    _write_best_trial(study, metric_name, base_params, output_dir)
    print(f"Best {metric_name}: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
