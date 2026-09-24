"""Tests for EfficientNet-1D training orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import torchsig_models.models.iq_models.efficientnet.efficientnet1d_train as training_module
from torchsig_models.models.iq_models.efficientnet.efficientnet1d_train import (
    train_efficientnet_iq,
)


def test_train_efficientnet_iq_orchestrates_training_and_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_cfg = SimpleNamespace(seed=11)
    val_cfg = SimpleNamespace(seed=12)
    test_cfg = SimpleNamespace(seed=13)
    params = {
        "batch_size": 8,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 3,
        "drop_path": 0.1,
        "drop_rate": 0.2,
        "label_smoothing": 0.05,
    }
    train_loader, val_loader, test_loader = [object()], [object()], [object()]
    data_info = {"root": "dataset", "class_names": ["a", "b", "c"]}
    prepare_datasets = MagicMock(
        return_value=(train_loader, val_loader, test_loader, data_info)
    )
    monkeypatch.setattr(training_module, "prepare_torchsig_datasets", prepare_datasets)

    model = torch.nn.Linear(4, 3)
    model_factory = MagicMock(return_value=model)
    monkeypatch.setitem(training_module.MODEL_FACTORY, "efficientnet_b0", model_factory)

    wrapped_model = SimpleNamespace(model=model)
    metrics_callback = object()
    train_validate = MagicMock(return_value=(wrapped_model, metrics_callback))
    monkeypatch.setattr(training_module, "train_validate", train_validate)

    test_metrics = MagicMock()
    evaluate_classifier = MagicMock(return_value=test_metrics)
    monkeypatch.setattr(training_module, "evaluate_classifier", evaluate_classifier)
    set_deterministic = MagicMock()
    monkeypatch.setattr(training_module, "set_deterministic", set_deterministic)
    monkeypatch.setattr(training_module, "compute_num_params", lambda _model: 15)

    checkpoint_dir = tmp_path / "checkpoints"
    metrics_dir = tmp_path / "metrics"
    result = train_efficientnet_iq(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        params=params,
        checkpoint_dir=checkpoint_dir,
        metrics_dir=metrics_dir,
        dataset_root=tmp_path / "datasets",
        overwrite=True,
        model_name="efficientnet_b0",
        signal_generators=["a", "b", "c"],
        logger=False,
        accelerator="cpu",
        devices=1,
    )

    set_deterministic.assert_called_once_with(11)
    assert checkpoint_dir.is_dir()
    prepare_datasets.assert_called_once_with(
        train_cfg,
        val_cfg,
        test_cfg,
        dataset_root=tmp_path / "datasets",
        batch_size=8,
        overwrite=True,
        signal_generators=["a", "b", "c"],
    )
    model_factory.assert_called_once_with(
        num_classes=3,
        drop_path_rate=0.1,
        drop_rate=0.2,
    )

    training_call = train_validate.call_args.kwargs
    assert training_call["train_loader"] is train_loader
    assert training_call["val_loader"] is val_loader
    assert training_call["model"] is model
    assert isinstance(training_call["criterion"], torch.nn.CrossEntropyLoss)
    assert training_call["criterion"].label_smoothing == pytest.approx(0.05)
    assert isinstance(training_call["optimizer"], torch.optim.AdamW)
    assert isinstance(
        training_call["scheduler"], torch.optim.lr_scheduler.SequentialLR
    )
    assert training_call["num_classes"] == 3
    assert training_call["accelerator"] == "cpu"
    assert training_call["devices"] == 1

    evaluate_classifier.assert_called_once_with(
        model=model,
        test_loader=test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_classes=3,
        criterion=training_call["criterion"],
    )
    test_metrics.save_to_csv.assert_called_once_with(metrics_dir / "test")
    assert result["num_classes"] == 3
    assert result["num_params"] == 15
    assert result["metrics"] is metrics_callback
    assert result["data_info"] is data_info
