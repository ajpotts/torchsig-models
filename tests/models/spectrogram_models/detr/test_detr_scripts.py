"""Tests for DETR training and inference entry points."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import yaml

import torchsig_models.models.spectrogram_models.detr.detr_train as training_module
from torchsig_models.models.spectrogram_models.detr.detr_inference import (
    _infer_num_classes,
    _strip_lightning_prefix,
    postprocess_detections,
)
from torchsig_models.models.spectrogram_models.detr.detr_train import (
    detr_collate,
    load_training_params,
    train_detr,
)


def test_detr_collate_preserves_multiple_signals() -> None:
    batch = [
        (
            np.ones((8, 8), dtype=np.float32),
            [[0, 0.25, 0.25, 0.1, 0.2], [2, 0.75, 0.5, 0.2, 0.3]],
        ),
        (np.zeros((8, 8), dtype=np.float32), []),
    ]

    images, targets = detr_collate(batch)

    assert images.shape == (2, 2, 8, 8)
    assert targets[0]["labels"].tolist() == [0, 2]
    assert targets[0]["boxes"].shape == (2, 4)
    assert targets[1]["labels"].shape == (0,)
    assert targets[1]["boxes"].shape == (0, 4)


def test_detr_collate_rejects_invalid_channel_count() -> None:
    with pytest.raises(ValueError, match="one or two input channels"):
        detr_collate([(np.ones((3, 8, 8), dtype=np.float32), [])])


def test_load_training_params_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    expected = {"batch_size": 4, "max_epochs": 2}
    path.write_text(yaml.safe_dump(expected), encoding="utf-8")

    assert load_training_params("detr_b0_nano", path) == expected


def test_strip_lightning_prefix_omits_criterion_state() -> None:
    state = {
        "model.linear_class.weight": torch.ones(4, 8),
        "criterion.empty_weight": torch.ones(4),
    }

    stripped = _strip_lightning_prefix(state)

    assert list(stripped) == ["linear_class.weight"]
    assert _infer_num_classes(stripped) == 3


def test_postprocess_detections_filters_background_and_confidence() -> None:
    outputs = {
        "pred_logits": torch.tensor(
            [[[8.0, 0.0, -2.0], [0.0, 0.0, 8.0], [0.1, 0.0, 0.0]]]
        ),
        "pred_boxes": torch.rand(1, 3, 4),
    }

    detections = postprocess_detections(outputs, confidence_threshold=0.75)

    assert detections[0]["labels"].tolist() == [0]
    assert detections[0]["boxes"].shape == (1, 4)
    assert detections[0]["scores"].device.type == "cpu"


def test_postprocess_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        postprocess_detections({}, confidence_threshold=1.1)


def test_train_detr_uses_warn_only_determinism(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaders = (object(), object(), object(), {"class_names": ["signal"]})
    monkeypatch.setattr(
        training_module, "prepare_torchsig_datasets", MagicMock(return_value=loaders)
    )
    monkeypatch.setattr(training_module, "set_deterministic", MagicMock())
    monkeypatch.setattr(
        training_module,
        "_spectrogram_transforms",
        MagicMock(return_value=[]),
    )
    monkeypatch.setitem(
        training_module.MODEL_FACTORY,
        "detr_b0_nano",
        MagicMock(return_value=torch.nn.Linear(1, 1)),
    )
    checkpoint = SimpleNamespace(best_model_path="best.ckpt")
    monkeypatch.setattr(training_module, "ModelCheckpoint", MagicMock(return_value=checkpoint))
    trainer = MagicMock()
    trainer.callback_metrics = {"val_loss": torch.tensor(1.0)}
    trainer.test.return_value = [{"test_loss": 1.25}]
    trainer_factory = MagicMock(return_value=trainer)
    monkeypatch.setattr(training_module.pl, "Trainer", trainer_factory)
    cfg = SimpleNamespace(seed=123)

    train_detr(
        cfg,
        cfg,
        cfg,
        {
            "batch_size": 2,
            "max_epochs": 1,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
        },
        tmp_path,
    )

    assert trainer_factory.call_args.kwargs["deterministic"] == "warn"
