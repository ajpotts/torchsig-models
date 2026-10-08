"""Tests for EfficientNet-1D training orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import torchsig_models.models.iq_models.efficientnet.efficientnet1d_train as training_module
from torchsig_models.models.iq_models.efficientnet.efficientnet1d_train import (
    parse_args,
    train_efficientnet_iq,
)


def test_parse_args_accepts_loader_performance_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "efficientnet1d_train.py",
            "--num-workers",
            "6",
            "--pin-memory",
            "--no-persistent-workers",
        ],
    )

    args = parse_args()

    assert args.num_workers == 6
    assert args.pin_memory is True
    assert args.persistent_workers is False


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
        "normalization": "none",
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
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
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
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        file_handler=None,
        file_reader=None,
        file_handler_options=None,
        dataset_mode="auto",
    )
    model_factory.assert_called_once_with(
        num_classes=3,
        drop_path_rate=0.1,
        drop_rate=0.2,
        class_names=["a", "b", "c"],
        normalization="none",
        normalization_eps=1e-6,
    )

    training_call = train_validate.call_args.kwargs
    assert training_call["train_loader"] is train_loader
    assert training_call["val_loader"] is val_loader
    assert training_call["model"] is model
    assert isinstance(training_call["criterion"], torch.nn.CrossEntropyLoss)
    assert training_call["criterion"].label_smoothing == pytest.approx(0.05)
    assert isinstance(training_call["optimizer"], torch.optim.AdamW)
    assert isinstance(training_call["scheduler"], torch.optim.lr_scheduler.SequentialLR)
    assert training_call["num_classes"] == 3
    assert training_call["accelerator"] == "cpu"
    assert training_call["devices"] == 1
    assert training_call["precision"] == "32-true"

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


@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
def test_train_efficientnet_iq_forwards_precision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    precision: str,
) -> None:
    """Forward each supported precision mode to the shared trainer."""
    cfg = SimpleNamespace(seed=11)
    params = {
        "batch_size": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 1,
        "normalization": "none",
    }
    loaders = ([object()], [object()], [object()])
    monkeypatch.setattr(
        training_module,
        "prepare_torchsig_datasets",
        MagicMock(return_value=(*loaders, {"class_names": ["a", "b"]})),
    )
    monkeypatch.setitem(
        training_module.MODEL_FACTORY,
        "efficientnet_b0",
        MagicMock(return_value=torch.nn.Linear(4, 2)),
    )
    train_validate = MagicMock(
        return_value=(SimpleNamespace(model=torch.nn.Linear(4, 2)), MagicMock())
    )
    monkeypatch.setattr(training_module, "train_validate", train_validate)
    test_metrics = MagicMock()
    monkeypatch.setattr(
        training_module, "evaluate_classifier", MagicMock(return_value=test_metrics)
    )
    monkeypatch.setattr(training_module, "set_deterministic", MagicMock())

    train_efficientnet_iq(
        train_cfg=cfg,
        val_cfg=cfg,
        test_cfg=cfg,
        params=params,
        checkpoint_dir=tmp_path / "checkpoints",
        model_name="efficientnet_b0",
        accelerator="cpu",
        precision=precision,
    )

    assert train_validate.call_args.kwargs["precision"] == precision


def test_train_efficientnet_iq_uses_precision_from_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use the parameter-file precision when no API override is supplied."""
    cfg = SimpleNamespace(seed=11)
    params = {
        "batch_size": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 1,
        "normalization": "none",
        "precision": "bf16-mixed",
    }
    loaders = ([object()], [object()], [object()])
    monkeypatch.setattr(
        training_module,
        "prepare_torchsig_datasets",
        MagicMock(return_value=(*loaders, {"class_names": ["a", "b"]})),
    )
    monkeypatch.setitem(
        training_module.MODEL_FACTORY,
        "efficientnet_b0",
        MagicMock(return_value=torch.nn.Linear(4, 2)),
    )
    train_validate = MagicMock(
        return_value=(SimpleNamespace(model=torch.nn.Linear(4, 2)), MagicMock())
    )
    monkeypatch.setattr(training_module, "train_validate", train_validate)
    monkeypatch.setattr(
        training_module,
        "evaluate_classifier",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(training_module, "set_deterministic", MagicMock())

    train_efficientnet_iq(
        train_cfg=cfg,
        val_cfg=cfg,
        test_cfg=cfg,
        params=params,
        checkpoint_dir=tmp_path / "checkpoints",
        model_name="efficientnet_b0",
        accelerator="cpu",
    )

    assert train_validate.call_args.kwargs["precision"] == "bf16-mixed"


@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
def test_parse_args_accepts_precision(
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
) -> None:
    """Accept all documented precision modes on the command line."""
    monkeypatch.setattr(
        "sys.argv",
        ["efficientnet1d_train.py", "--precision", precision],
    )

    assert parse_args().precision == precision
