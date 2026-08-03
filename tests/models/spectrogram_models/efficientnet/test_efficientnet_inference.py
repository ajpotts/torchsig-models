"""Unit tests for EfficientNet-2D inference entry points."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import torchsig_models.models.spectrogram_models.efficientnet.efficientnet_inference as inference_module
from torchsig_models.models.spectrogram_models.efficientnet.efficientnet_inference import (
    _strip_lightning_prefix,
    efficientnet_inference,
    parse_args,
)


# =============================================================================
# Checkpoint key normalization
# =============================================================================


def test_strip_lightning_prefix_removes_model_prefix() -> None:
    """Verify Lightning's model prefix is removed from checkpoint keys."""
    weight = torch.ones(2, 2)
    bias = torch.zeros(2)

    state_dict = {
        "model.weight": weight,
        "model.bias": bias,
    }

    result = _strip_lightning_prefix(state_dict)

    assert result == {
        "weight": weight,
        "bias": bias,
    }


def test_strip_lightning_prefix_returns_unwrapped_state_dict() -> None:
    """Verify an ordinary PyTorch state dictionary is unchanged."""
    state_dict = {
        "weight": torch.ones(2, 2),
        "bias": torch.zeros(2),
    }

    result = _strip_lightning_prefix(state_dict)

    assert result is state_dict


def test_strip_lightning_prefix_handles_mixed_keys() -> None:
    """Verify prefixed keys are stripped when any Lightning keys exist."""
    state_dict = {
        "model.weight": torch.ones(2, 2),
        "metadata": torch.tensor(1),
    }

    result = _strip_lightning_prefix(state_dict)

    assert set(result) == {
        "weight",
        "metadata",
    }


# =============================================================================
# Inference pipeline
# =============================================================================


def test_efficientnet_inference_runs_evaluation_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify model loading, dataset preparation, and evaluation."""
    dataset_root = tmp_path / "test"
    dataset_root.mkdir()

    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()

    params_path = tmp_path / "params.yaml"

    params = {
        "drop_path": 0.1,
        "drop_rate": 0.25,
    }
    load_training_params = MagicMock(return_value=params)
    monkeypatch.setattr(
        inference_module,
        "load_training_params",
        load_training_params,
    )

    model = MagicMock()
    model.load_state_dict.return_value = ([], [])
    model_factory = MagicMock(return_value=model)
    monkeypatch.setitem(
        inference_module.MODEL_FACTORY,
        "efficientnet_b0",
        model_factory,
    )

    checkpoint = {
        "state_dict": {
            "model.layer.weight": torch.ones(1),
        }
    }
    torch_load = MagicMock(return_value=checkpoint)
    monkeypatch.setattr(
        inference_module.torch,
        "load",
        torch_load,
    )
    monkeypatch.setattr(
        inference_module.torch.cuda,
        "is_available",
        lambda: False,
    )

    test_loader = object()
    prepare_dataset = MagicMock(return_value=test_loader)
    monkeypatch.setattr(
        inference_module,
        "prepare_torchsig_inference_dataset",
        prepare_dataset,
    )

    tracker = SimpleNamespace(
        history={
            "accuracy": [0.5, 0.875],
        }
    )
    evaluate_classifier = MagicMock(return_value=tracker)
    monkeypatch.setattr(
        inference_module,
        "evaluate_classifier",
        evaluate_classifier,
    )

    result = efficientnet_inference(
        root=dataset_root,
        checkpoint_path=checkpoint_path,
        params_path=params_path,
        batch_size=16,
        num_workers=2,
        num_classes=3,
        model_name="efficientnet_b0",
    )

    load_training_params.assert_called_once_with(
        "efficientnet_b0",
        params_path=params_path,
    )
    model_factory.assert_called_once_with(
        num_classes=3,
        drop_path_rate=0.1,
        drop_rate=0.25,
    )

    torch_load.assert_called_once_with(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.load_state_dict.assert_called_once_with(
        {
            "layer.weight": checkpoint["state_dict"][
                "model.layer.weight"
            ],
        },
        strict=True,
    )
    model.to.assert_called_once_with(
        torch.device("cpu"),
    )
    model.eval.assert_called_once_with()

    prepare_dataset.assert_called_once_with(
        dataset_root,
        batch_size=16,
        num_workers=2,
    )

    evaluate_classifier.assert_called_once_with(
        model=model,
        test_loader=test_loader,
        device=torch.device("cpu"),
        num_classes=3,
    )

    assert result == pytest.approx(0.875)
    assert "Test accuracy: 87.5000%" in capsys.readouterr().out


def test_efficientnet_inference_loads_plain_state_dict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify checkpoints without a state_dict wrapper are supported."""
    dataset_root = tmp_path / "test"
    dataset_root.mkdir()

    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.touch()

    monkeypatch.setattr(
        inference_module,
        "load_training_params",
        MagicMock(return_value={}),
    )

    model = MagicMock()
    model.load_state_dict.return_value = ([], [])
    monkeypatch.setitem(
        inference_module.MODEL_FACTORY,
        "efficientnet_b4",
        MagicMock(return_value=model),
    )

    state_dict = {
        "weight": torch.ones(1),
    }
    monkeypatch.setattr(
        inference_module.torch,
        "load",
        MagicMock(return_value=state_dict),
    )
    monkeypatch.setattr(
        inference_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        inference_module,
        "prepare_torchsig_inference_dataset",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        inference_module,
        "evaluate_classifier",
        MagicMock(
            return_value=SimpleNamespace(
                history={"accuracy": [0.75]}
            )
        ),
    )

    efficientnet_inference(
        root=dataset_root,
        checkpoint_path=checkpoint_path,
    )

    model.load_state_dict.assert_called_once_with(
        state_dict,
        strict=True,
    )


def test_efficientnet_inference_uses_default_model_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify default dropout values are used when absent from YAML."""
    dataset_root = tmp_path / "test"
    dataset_root.mkdir()

    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()

    monkeypatch.setattr(
        inference_module,
        "load_training_params",
        MagicMock(return_value={}),
    )

    model = MagicMock()
    model.load_state_dict.return_value = ([], [])
    model_factory = MagicMock(return_value=model)
    monkeypatch.setitem(
        inference_module.MODEL_FACTORY,
        "efficientnet_b2",
        model_factory,
    )

    monkeypatch.setattr(
        inference_module.torch,
        "load",
        MagicMock(return_value={}),
    )
    monkeypatch.setattr(
        inference_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        inference_module,
        "prepare_torchsig_inference_dataset",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        inference_module,
        "evaluate_classifier",
        MagicMock(
            return_value=SimpleNamespace(
                history={"accuracy": [1.0]}
            )
        ),
    )

    efficientnet_inference(
        root=dataset_root,
        checkpoint_path=checkpoint_path,
        num_classes=10,
        model_name="efficientnet_b2",
    )

    model_factory.assert_called_once_with(
        num_classes=10,
        drop_path_rate=0.2,
        drop_rate=0.3,
    )


def test_efficientnet_inference_uses_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify checkpoint loading and evaluation use CUDA when available."""
    dataset_root = tmp_path / "test"
    dataset_root.mkdir()

    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()

    monkeypatch.setattr(
        inference_module,
        "load_training_params",
        MagicMock(return_value={}),
    )

    model = MagicMock()
    model.load_state_dict.return_value = ([], [])
    monkeypatch.setitem(
        inference_module.MODEL_FACTORY,
        "efficientnet_b4",
        MagicMock(return_value=model),
    )

    torch_load = MagicMock(return_value={})
    monkeypatch.setattr(
        inference_module.torch,
        "load",
        torch_load,
    )
    monkeypatch.setattr(
        inference_module.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        inference_module,
        "prepare_torchsig_inference_dataset",
        MagicMock(return_value=object()),
    )

    evaluate_classifier = MagicMock(
        return_value=SimpleNamespace(
            history={"accuracy": [0.9]}
        )
    )
    monkeypatch.setattr(
        inference_module,
        "evaluate_classifier",
        evaluate_classifier,
    )

    efficientnet_inference(
        root=dataset_root,
        checkpoint_path=checkpoint_path,
    )

    expected_device = torch.device("cuda")

    torch_load.assert_called_once_with(
        checkpoint_path,
        map_location=expected_device,
        weights_only=False,
    )
    model.to.assert_called_once_with(expected_device)

    assert (
        evaluate_classifier.call_args.kwargs["device"]
        == expected_device
    )


def test_efficientnet_inference_rejects_checkpoint_key_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify incompatible checkpoints stop inference."""
    dataset_root = tmp_path / "test"
    dataset_root.mkdir()

    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()

    monkeypatch.setattr(
        inference_module,
        "load_training_params",
        MagicMock(return_value={}),
    )

    model = MagicMock()
    model.load_state_dict.side_effect = RuntimeError(
        "Missing key(s) and unexpected key(s) in state_dict"
    )
    monkeypatch.setitem(
        inference_module.MODEL_FACTORY,
        "efficientnet_b4",
        MagicMock(return_value=model),
    )

    monkeypatch.setattr(
        inference_module.torch,
        "load",
        MagicMock(return_value={}),
    )
    monkeypatch.setattr(
        inference_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        inference_module,
        "prepare_torchsig_inference_dataset",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        inference_module,
        "evaluate_classifier",
        MagicMock(
            return_value=SimpleNamespace(
                history={"accuracy": [0.5]}
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Missing key"):
        efficientnet_inference(
            root=dataset_root,
            checkpoint_path=checkpoint_path,
        )

    model.load_state_dict.assert_called_once_with({}, strict=True)


def test_efficientnet_inference_accepts_matching_checkpoint_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify matching checkpoint keys permit inference."""
    dataset_root = tmp_path / "test"
    dataset_root.mkdir()

    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()

    monkeypatch.setattr(
        inference_module,
        "load_training_params",
        MagicMock(return_value={}),
    )

    model = MagicMock()
    model.load_state_dict.return_value = ([], [])
    monkeypatch.setitem(
        inference_module.MODEL_FACTORY,
        "efficientnet_b4",
        MagicMock(return_value=model),
    )

    monkeypatch.setattr(
        inference_module.torch,
        "load",
        MagicMock(return_value={}),
    )
    monkeypatch.setattr(
        inference_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        inference_module,
        "prepare_torchsig_inference_dataset",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        inference_module,
        "evaluate_classifier",
        MagicMock(
            return_value=SimpleNamespace(
                history={"accuracy": [0.5]}
            )
        ),
    )

    efficientnet_inference(
        root=dataset_root,
        checkpoint_path=checkpoint_path,
    )

    output = capsys.readouterr().out

    assert "Test accuracy: 50.0000%" in output
    model.load_state_dict.assert_called_once_with({}, strict=True)


def test_efficientnet_inference_raises_for_missing_checkpoint(
    tmp_path: Path,
) -> None:
    """Verify a nonexistent checkpoint raises an error."""
    dataset_root = tmp_path / "test"
    dataset_root.mkdir()

    checkpoint_path = tmp_path / "missing.ckpt"

    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint not found",
    ):
        efficientnet_inference(
            root=dataset_root,
            checkpoint_path=checkpoint_path,
        )


def test_efficientnet_inference_delegates_missing_dataset_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify missing dataset validation comes from the dataset utility."""
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()

    missing_root = tmp_path / "missing"

    monkeypatch.setattr(
        inference_module,
        "load_training_params",
        MagicMock(return_value={}),
    )

    model = MagicMock()
    model.load_state_dict.return_value = ([], [])
    monkeypatch.setitem(
        inference_module.MODEL_FACTORY,
        "efficientnet_b4",
        MagicMock(return_value=model),
    )

    monkeypatch.setattr(
        inference_module.torch,
        "load",
        MagicMock(return_value={}),
    )
    monkeypatch.setattr(
        inference_module.torch.cuda,
        "is_available",
        lambda: False,
    )

    prepare_dataset = MagicMock(
        side_effect=FileNotFoundError(
            f"Dataset root not found: {missing_root}"
        )
    )
    monkeypatch.setattr(
        inference_module,
        "prepare_torchsig_inference_dataset",
        prepare_dataset,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Dataset root not found",
    ):
        efficientnet_inference(
            root=missing_root,
            checkpoint_path=checkpoint_path,
        )

    prepare_dataset.assert_called_once_with(
        missing_root,
        batch_size=4,
        num_workers=8,
    )
