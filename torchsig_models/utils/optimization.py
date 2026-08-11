"""Hyperparameter optimization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import mlflow
import optuna
import yaml


ObjectiveTrainFn = Callable[[dict[str, Any], Path, optuna.Trial], dict[str, Any]]


def load_search_config(path: str | Path) -> dict[str, Any]:
    """Load an optimization/search YAML config."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Search config not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def suggest_params(
    trial: optuna.Trial,
    base_params: dict[str, Any],
    search_space: dict[str, Any],
) -> dict[str, Any]:
    """Return training params with Optuna-suggested overrides applied."""
    params = base_params.copy()

    for name, spec in search_space.items():
        param_type = spec["type"]

        if param_type == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif param_type == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(
                name,
                spec["choices"],
            )
        else:
            raise ValueError(f"Unsupported search parameter type: {param_type}")

    return params


def optimize_params(
    *,
    base_params: dict[str, Any],
    search_space: dict[str, Any],
    train_fn: ObjectiveTrainFn,
    metric_name: str = "val_f1",
    direction: str = "maximize",
    n_trials: int = 20,
    experiment_name: str | None = None,
    run_name: str = "optimization",
    output_dir: str | Path = "runs/optimization",
) -> optuna.Study:
    """Run Optuna hyperparameter optimization with optional MLflow logging.

    Args:
        base_params:
            Default training parameters.
        search_space:
            Dictionary describing Optuna search parameters.
        train_fn:
            Callable that accepts ``params`` and ``trial_dir`` and returns a
            result dictionary containing metrics.
        metric_name:
            Metric to optimize. The train function result must contain this
            key directly or under ``result["metrics"]``.
        direction:
            Optuna direction, either ``"maximize"`` or ``"minimize"``.
        n_trials:
            Number of Optuna trials.
        experiment_name:
            Optional MLflow experiment name.
        run_name:
            Parent MLflow run name.
        output_dir:
            Directory where trial artifacts/checkpoints should be written.

    Returns:
        The completed Optuna study.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if experiment_name is not None:
        mlflow.set_experiment(experiment_name)

    study = optuna.create_study(direction=direction)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, base_params, search_space)
        trial_dir = output_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        with mlflow.start_run(run_name=f"trial_{trial.number:04d}", nested=True):
            mlflow.log_params(params)
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_param("trial_dir", str(trial_dir))

            result = train_fn(params, trial_dir, trial)

            metric_value = _extract_metric(result, metric_name)

            mlflow.log_metric(metric_name, metric_value)

            if "num_params" in result:
                mlflow.log_metric("num_params", result["num_params"])

            return metric_value

    with mlflow.start_run(run_name=run_name):
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_metric(f"best_{metric_name}", study.best_value)
        mlflow.log_param("best_trial_number", study.best_trial.number)
        mlflow.log_params(
            {f"best_{key}": value for key, value in study.best_params.items()}
        )

    return study


def _extract_metric(result: dict[str, Any], metric_name: str) -> float:
    """Extract an optimization metric from a training result dictionary."""
    if metric_name in result:
        return float(result[metric_name])

    metrics = result.get("metrics")

    if metrics is not None:
        plural_name = f"{metric_name}s"

        if hasattr(metrics, plural_name):
            values = getattr(metrics, plural_name)
            return float(values[-1])

        if hasattr(metrics, metric_name):
            value = getattr(metrics, metric_name)
            if isinstance(value, list):
                return float(value[-1])
            return float(value)

    raise KeyError(f"Could not find metric '{metric_name}' in training result.")
