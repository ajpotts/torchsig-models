"""Locally verify that hyperparameter-search output uses the best trial.

This example avoids model training and dataset generation. It creates a small
in-memory Optuna study in which trial 0 is better than the later trial 1, then
writes and validates the same YAML summaries produced by the EfficientNet-1D
hyperparameter-search command.

Run from the repository root with::

    python examples/verify_best_trial_summary.py

Use ``--output-dir`` to retain the generated files in a specific directory.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import optuna
import yaml

from torchsig_models.models.iq_models.efficientnet.efficientnet1d_hyperparameter_search import (
    _write_best_trial_summary,
)


BEST_PARAMETERS = {
    "learning_rate": 1.0e-4,
    "batch_size": 32,
}
LAST_PARAMETERS = {
    "learning_rate": 9.0e-4,
    "batch_size": 128,
}


def _completed_trial(parameters: dict[str, Any], value: float) -> optuna.trial.FrozenTrial:
    """Create a completed trial using distributions matching the parameters."""
    return optuna.trial.create_trial(
        params=parameters,
        distributions={
            "learning_rate": optuna.distributions.FloatDistribution(
                1.0e-5,
                1.0e-3,
                log=True,
            ),
            "batch_size": optuna.distributions.CategoricalDistribution(
                [32, 64, 128]
            ),
        },
        value=value,
    )


def verify_summary(output_dir: Path) -> tuple[Path, Path]:
    """Generate summary files and verify they describe the best trial."""
    study = optuna.create_study(direction="maximize")
    study.add_trial(_completed_trial(BEST_PARAMETERS, value=0.90))
    study.add_trial(_completed_trial(LAST_PARAMETERS, value=0.20))

    summary_path, training_params_path = _write_best_trial_summary(
        study=study,
        metric_name="val_f1",
        base_params={
            "model_name": "efficientnet_b0",
            "max_epochs": 10,
            "weight_decay": 1.0e-6,
        },
        output_dir=output_dir,
    )

    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    training_params = yaml.safe_load(
        training_params_path.read_text(encoding="utf-8")
    )

    assert study.trials[-1].number == 1
    assert study.best_trial.number == 0
    assert summary["trial_number"] == study.best_trial.number
    assert summary["metric_value"] == study.best_value
    assert summary["parameters"] == BEST_PARAMETERS
    assert summary["parameters"] != LAST_PARAMETERS
    assert training_params["learning_rate"] == BEST_PARAMETERS["learning_rate"]
    assert training_params["batch_size"] == BEST_PARAMETERS["batch_size"]

    return summary_path, training_params_path


def parse_args() -> argparse.Namespace:
    """Parse the optional output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated YAML files; defaults to a temporary directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the local best-trial summary verification."""
    args = parse_args()
    temporary_directory = None
    if args.output_dir is None:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="torchsig-best-trial-"
        )
        output_dir = Path(temporary_directory.name)
    else:
        output_dir = args.output_dir

    summary_path, training_params_path = verify_summary(output_dir)

    print("Verification passed: the final output uses trial 0, not trial 1.")
    print(f"\n{summary_path.name}:\n{summary_path.read_text(encoding='utf-8')}")
    print(
        f"{training_params_path.name}:\n"
        f"{training_params_path.read_text(encoding='utf-8')}"
    )

    if temporary_directory is not None:
        print("Generated files used a temporary directory and are now removed.")
        temporary_directory.cleanup()
    else:
        print(f"Generated files retained in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
