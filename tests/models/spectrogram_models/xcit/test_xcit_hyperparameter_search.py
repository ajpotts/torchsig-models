from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import torchsig_models.models.spectrogram_models.xcit.xcit_hyperparameter_search as search
from torchsig_models.utils.hyperparameter_search import load_search_config


@dataclass(frozen=True)
class DummyConfig:
    dataset_id: str = "demo"
    dataset_length: int = 10
    seed: int = 7


def test_default_search_config_is_packaged() -> None:
    config = load_search_config(search.DEFAULT_SEARCH_CONFIG)

    assert config["metric_name"] == "val_f1"
    assert config["n_trials"] == 20
    assert "learning_rate" in config["search_space"]


def test_load_split_configs_offsets_shared_seeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loader = MagicMock(return_value=DummyConfig())
    monkeypatch.setattr(search, "load_config_from_yaml", loader)
    config_path = tmp_path / "dataset.yaml"

    train, validation, test = search._load_split_configs(
        config_path, None, None, None
    )

    assert train.seed == 7
    assert validation.seed == 8
    assert test.seed == 9


def test_apply_overrides_is_immutable() -> None:
    original = DummyConfig()

    updated = search._apply_overrides(original, 25, "search-demo")

    assert original.dataset_length == 10
    assert updated.dataset_length == 25
    assert updated.dataset_id == "search-demo"


def test_final_metrics_extracts_latest_values() -> None:
    result = {
        "metrics": SimpleNamespace(
            val_f1s=[0.5, 0.8],
            val_accuracies=[0.6, 0.9],
            train_f1s=[0.7, 0.95],
            train_accuracies=[0.75, 1.0],
        )
    }

    assert search._final_metrics(result) == {
        "val_f1": 0.8,
        "val_acc": 0.9,
        "train_f1": 0.95,
        "train_acc": 1.0,
    }


def test_write_best_trial_outputs_yaml(tmp_path: Path) -> None:
    study = SimpleNamespace(
        best_trial=SimpleNamespace(number=2, value=0.91),
        best_params={"learning_rate": 0.001},
    )

    search._write_best_trial(
        study, "val_f1", {"batch_size": 8}, tmp_path
    )

    summary = (tmp_path / "best_trial.yaml").read_text(encoding="utf-8")
    params = (tmp_path / "best_training_params.yaml").read_text(encoding="utf-8")
    assert "trial_number: 2" in summary
    assert "learning_rate: 0.001" in params
    assert "batch_size: 8" in params
