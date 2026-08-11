"""Hyperparameter optimization utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any

import optuna
import yaml
import logging

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]


ObjectiveTrainFn = Callable[
    [dict[str, Any], Path, optuna.Trial],
    dict[str, Any],
]


logger = logging.getLogger(__name__)


class _MLflowLogger:
    """Provide best-effort MLflow logging.

    MLflow failures disable tracking for the remainder of the optimization.
    Training and Optuna failures are not suppressed.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled and mlflow is not None
        self._warning_emitted = False

        if enabled and mlflow is None:
            self._disable(
                "initialization",
                RuntimeError("the mlflow package is not installed"),
            )

    def _disable(self, operation: str, error: Exception) -> None:
        """Disable MLflow and emit at most one warning."""
        self.enabled = False

        if self._warning_emitted:
            return

        logger.warning(
            "MLflow is unavailable during %s. "
            "Hyperparameter optimization will continue without MLflow: %s",
            operation,
            error,
        )
        self._warning_emitted = True

    def call(
        self,
        operation: str,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any | None:
        """Call an MLflow function if tracking is enabled."""
        if not self.enabled:
            return None

        try:
            return function(*args, **kwargs)
        except Exception as error:
            self._disable(operation, error)
            return None

    @contextmanager
    def run(
        self,
        *,
        run_name: str,
        nested: bool = False,
    ) -> Iterator[None]:
        """Open an MLflow run when tracking is available."""
        if not self.enabled:
            yield
            return

        assert mlflow is not None

        run_started = False

        try:
            mlflow.start_run(
                run_name=run_name,
                nested=nested,
            )
            run_started = True
        except Exception as error:
            self._disable("starting a run", error)

        try:
            yield
        finally:
            if run_started:
                try:
                    mlflow.end_run()
                except Exception as error:
                    self._disable("ending a run", error)

    def set_experiment(self, experiment_name: str) -> None:
        """Set the active MLflow experiment."""
        if mlflow is not None:
            self.call(
                "setting the experiment",
                mlflow.set_experiment,
                experiment_name,
            )

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log parameters when tracking is available."""
        if mlflow is not None:
            self.call(
                "logging parameters",
                mlflow.log_params,
                dict(params),
            )

    def log_param(self, name: str, value: Any) -> None:
        """Log one parameter when tracking is available."""
        if mlflow is not None:
            self.call(
                f"logging parameter {name!r}",
                mlflow.log_param,
                name,
                value,
            )

    def log_metric(self, name: str, value: float | int) -> None:
        """Log one metric when tracking is available."""
        if mlflow is not None:
            self.call(
                f"logging metric {name!r}",
                mlflow.log_metric,
                name,
                value,
            )


@contextmanager
def _mlflow_http_settings(
    *,
    timeout_seconds: int,
    max_retries: int,
) -> Iterator[None]:
    """Temporarily configure MLflow HTTP requests to fail quickly."""
    settings = {
        "MLFLOW_HTTP_REQUEST_TIMEOUT": str(timeout_seconds),
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": str(max_retries),
    }
    previous_values = {name: os.environ.get(name) for name in settings}

    try:
        for name, value in settings.items():
            os.environ.setdefault(name, value)

        yield
    finally:
        for name, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value


def load_search_config(path: str | Path) -> dict[str, Any]:
    """Load a hyperparameter search configuration from YAML."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Search config not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Search config must contain a YAML mapping: {path}")

    return config


def suggest_params(
    trial: optuna.Trial,
    base_params: dict[str, Any],
    search_space: dict[str, Any],
) -> dict[str, Any]:
    """Return training parameters with Optuna suggestions applied."""
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


def run_hyperparameter_optimization(
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
    show_progress_bar: bool = True,
    mlflow_enabled: bool = True,
    mlflow_timeout_seconds: int = 5,
    mlflow_max_retries: int = 0,
) -> optuna.Study:
    """Run Optuna hyperparameter optimization.

    MLflow logging is best-effort. Connection, authentication, storage, and
    logging failures disable MLflow but do not interrupt optimization.

    Errors from the training function, metric extraction, and Optuna are
    allowed to propagate.

    Args:
        base_params:
            Default training parameters.
        search_space:
            Optuna search-space configuration.
        train_fn:
            Function accepting parameters, a trial directory, and an Optuna
            trial. It must return a dictionary containing the target metric.
        metric_name:
            Name of the metric to optimize.
        direction:
            Optimization direction: ``"maximize"`` or ``"minimize"``.
        n_trials:
            Number of trials to run.
        experiment_name:
            Optional MLflow experiment name.
        run_name:
            Name of the optimization and parent MLflow run.
        output_dir:
            Directory in which trial outputs are written.
        show_progress_bar:
            Whether Optuna should display its progress bar.
        mlflow_enabled:
            Whether to attempt MLflow tracking.
        mlflow_timeout_seconds:
            Default MLflow HTTP request timeout.
        mlflow_max_retries:
            Default number of MLflow HTTP request retries.

    Returns:
        The completed Optuna study.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    if mlflow_timeout_seconds < 1:
        raise ValueError("mlflow_timeout_seconds must be at least 1.")

    if mlflow_max_retries < 0:
        raise ValueError("mlflow_max_retries cannot be negative.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking = _MLflowLogger(enabled=mlflow_enabled)
    study = optuna.create_study(direction=direction)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(
            trial,
            base_params,
            search_space,
        )

        trial_name = f"trial_{trial.number:04d}"
        trial_dir = output_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        with tracking.run(
            run_name=trial_name,
            nested=True,
        ):
            tracking.log_params(params)
            tracking.log_param(
                "trial_number",
                trial.number,
            )
            tracking.log_param(
                "trial_dir",
                str(trial_dir),
            )

            result = train_fn(
                params,
                trial_dir,
                trial,
            )
            metric_value = _extract_metric(
                result,
                metric_name,
            )

            tracking.log_metric(
                metric_name,
                metric_value,
            )

            if "num_params" in result:
                tracking.log_metric(
                    "num_params",
                    result["num_params"],
                )

        return metric_value

    with _mlflow_http_settings(
        timeout_seconds=mlflow_timeout_seconds,
        max_retries=mlflow_max_retries,
    ):
        if experiment_name is not None:
            tracking.set_experiment(experiment_name)

        with tracking.run(run_name=run_name):
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=show_progress_bar,
            )

            tracking.log_metric(
                f"best_{metric_name}",
                study.best_value,
            )
            tracking.log_param(
                "best_trial_number",
                study.best_trial.number,
            )
            tracking.log_params(
                {f"best_{name}": value for name, value in study.best_params.items()}
            )

    return study


def _extract_metric(
    result: dict[str, Any],
    metric_name: str,
) -> float:
    """Extract an optimization metric from a training result."""
    if metric_name in result:
        return float(result[metric_name])

    metrics = result.get("metrics")

    if metrics is not None:
        plural_name = f"{metric_name}s"

        if hasattr(metrics, plural_name):
            values = getattr(metrics, plural_name)

            if not values:
                raise ValueError(f"Metric history {plural_name!r} is empty.")

            return float(values[-1])

        if hasattr(metrics, metric_name):
            value = getattr(metrics, metric_name)

            if isinstance(value, list):
                if not value:
                    raise ValueError(f"Metric history {metric_name!r} is empty.")

                return float(value[-1])

            return float(value)

    raise KeyError(f"Could not find metric {metric_name!r} in training result.")
