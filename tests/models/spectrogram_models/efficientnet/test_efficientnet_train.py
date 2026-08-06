"""Unit tests for EfficientNet-2D training entry points."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import yaml

from torchsig.datasets.datasets import TorchSigDatasetConfig
from torchsig.transforms.transforms import Spectrogram

import torchsig_models.models.spectrogram_models.efficientnet.efficientnet_train as training_module
from torchsig_models.models.spectrogram_models.efficientnet.efficientnet_train import (
    _build_scheduler,
    _resolve_config_path,
    _spectrogram_transforms,
    _validate_single_signal_config,
    load_training_params,
    parse_args,
    train_efficientnet_2d,
)


# =============================================================================
# Helpers
# =============================================================================


def _dataset_config(
    *,
    seed: int = 123,
    fft_size: int = 256,
) -> MagicMock:
    """Create a minimal dataset-config mock."""
    cfg = MagicMock(spec=TorchSigDatasetConfig)
    cfg.seed = seed
    cfg.output_spectrogram_fft = fft_size
    cfg.dataset_metadata = {"fft_size": fft_size}
    cfg.dataset_id = "test_dataset"
    return cfg


def _training_params() -> dict[str, float | int]:
    """Return minimal EfficientNet training parameters."""
    return {
        "batch_size": 8,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 10,
        "drop_path": 0.1,
        "drop_rate": 0.2,
        "label_smoothing": 0.05,
    }


# =============================================================================
# load_training_params
# =============================================================================


def test_load_training_params_reads_explicit_yaml(tmp_path: Path) -> None:
    """Verify an explicitly supplied training-parameter YAML is loaded."""
    params_path = tmp_path / "params.yaml"
    expected = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "max_epochs": 20,
    }

    params_path.write_text(
        yaml.safe_dump(expected),
        encoding="utf-8",
    )

    result = load_training_params(
        "efficientnet_b0",
        params_path=params_path,
    )

    assert result == expected


def test_load_training_params_raises_for_missing_file(
    tmp_path: Path,
) -> None:
    """Verify a missing training-parameter file raises an error."""
    params_path = tmp_path / "missing.yaml"

    with pytest.raises(
        FileNotFoundError,
        match="Training parameter file not found",
    ):
        load_training_params(
            "efficientnet_b0",
            params_path=params_path,
        )


def test_load_training_params_uses_model_default_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the model name determines the default parameter path."""
    module_path = tmp_path / "efficientnet_train.py"
    params_dir = tmp_path / "training_params"
    params_dir.mkdir()

    expected = {
        "batch_size": 16,
        "max_epochs": 5,
    }
    params_path = params_dir / "efficientnet_b2.yaml"
    params_path.write_text(
        yaml.safe_dump(expected),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        training_module,
        "__file__",
        str(module_path),
    )

    result = load_training_params("efficientnet_b2")

    assert result == expected


# =============================================================================
# Spectrogram transforms
# =============================================================================


def test_spectrogram_transforms_uses_config_fft_size() -> None:
    """Verify the spectrogram transform uses the configured FFT size."""
    cfg = _dataset_config(fft_size=128)

    transforms = _spectrogram_transforms(cfg)

    assert len(transforms) == 1
    assert isinstance(transforms[0], Spectrogram)
    assert transforms[0].fft_size == 128


def test_spectrogram_transforms_defaults_to_256() -> None:
    """Verify FFT size defaults to 256 when absent from the config."""
    cfg = SimpleNamespace()

    transforms = _spectrogram_transforms(cfg)

    assert len(transforms) == 1
    assert isinstance(transforms[0], Spectrogram)
    assert transforms[0].fft_size == 256


def test_spectrogram_transforms_falls_back_to_metadata_fft_size() -> None:
    cfg = SimpleNamespace(
        output_spectrogram_fft=None,
        dataset_metadata={"fft_size": 128},
    )

    transforms = _spectrogram_transforms(cfg)

    assert transforms[0].fft_size == 128


def test_validate_single_signal_config_rejects_multi_signal_data() -> None:
    cfg = _dataset_config()
    cfg.dataset_metadata["num_signals_max"] = 2

    with pytest.raises(ValueError, match="num_signals_max=1"):
        _validate_single_signal_config(cfg, "training")


# =============================================================================
# Scheduler
# =============================================================================


@pytest.mark.parametrize(
    "max_epochs,expected_warmup,expected_cosine",
    [
        (1, 1, 1),
        (3, 3, 1),
        (5, 5, 1),
        (10, 5, 5),
        (20, 5, 15),
    ],
)
def test_build_scheduler_uses_warmup_then_cosine(
    max_epochs: int,
    expected_warmup: int,
    expected_cosine: int,
) -> None:
    """Verify scheduler phase lengths for short and long training runs."""
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)

    scheduler = _build_scheduler(
        optimizer,
        max_epochs=max_epochs,
    )

    assert isinstance(
        scheduler,
        torch.optim.lr_scheduler.SequentialLR,
    )

    warmup, cosine = scheduler._schedulers

    assert isinstance(
        warmup,
        torch.optim.lr_scheduler.LinearLR,
    )
    assert isinstance(
        cosine,
        torch.optim.lr_scheduler.CosineAnnealingLR,
    )
    assert warmup.total_iters == expected_warmup
    assert cosine.T_max == expected_cosine
    assert scheduler._milestones == [expected_warmup]


# =============================================================================
# Config path resolution
# =============================================================================


def test_resolve_config_path_prefers_split_config() -> None:
    """Verify a split-specific config overrides the default config."""
    default = Path("default.yaml")
    split = Path("train.yaml")

    result = _resolve_config_path(
        default,
        split,
        "train",
    )

    assert result == split


def test_resolve_config_path_uses_default_config() -> None:
    """Verify the shared config is used when no split config is given."""
    default = Path("default.yaml")

    result = _resolve_config_path(
        default,
        None,
        "val",
    )

    assert result == default


def test_resolve_config_path_raises_when_no_config_is_given() -> None:
    """Verify a missing shared and split-specific config raises an error."""
    with pytest.raises(
        ValueError,
        match=r"Must provide either --dataset-config or --test-config",
    ):
        _resolve_config_path(
            None,
            None,
            "test",
        )


# =============================================================================
# CLI parsing
# =============================================================================


def test_parse_args_uses_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify command-line defaults."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["efficientnet_train.py", "--dataset-config", "dataset.yaml"],
    )

    args = parse_args()

    assert args.dataset_config == Path("dataset.yaml")
    assert args.train_config is None
    assert args.val_config is None
    assert args.test_config is None
    assert args.params is None
    assert args.model == "efficientnet_b0"
    assert args.dataset_length is None
    assert args.dataset_id is None
    assert args.epochs is None
    assert args.batch_size is None
    assert args.overwrite is False
    assert args.accelerator == "auto"
    assert args.devices == "auto"


def test_parse_args_reads_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify supported command-line overrides are parsed."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "efficientnet_train.py",
            "--train-config",
            "train.yaml",
            "--val-config",
            "val.yaml",
            "--test-config",
            "test.yaml",
            "--params",
            "params.yaml",
            "--model",
            "efficientnet_b4",
            "--dataset-length",
            "1000",
            "--dataset-id",
            "spectrogram_test",
            "--epochs",
            "12",
            "--batch-size",
            "64",
            "--overwrite",
            "--accelerator",
            "cpu",
            "--devices",
            "2",
        ],
    )

    args = parse_args()

    assert args.dataset_config is None
    assert args.train_config == Path("train.yaml")
    assert args.val_config == Path("val.yaml")
    assert args.test_config == Path("test.yaml")
    assert args.params == Path("params.yaml")
    assert args.model == "efficientnet_b4"
    assert args.dataset_length == 1000
    assert args.dataset_id == "spectrogram_test"
    assert args.epochs == 12
    assert args.batch_size == 64
    assert args.overwrite is True
    assert args.accelerator == "cpu"
    assert args.devices == 2


# =============================================================================
# train_efficientnet_2d
# =============================================================================


def test_train_efficientnet_2d_runs_training_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the full training pipeline delegates to shared utilities."""
    train_cfg = _dataset_config(seed=11, fft_size=128)
    val_cfg = _dataset_config(seed=12, fft_size=128)
    test_cfg = _dataset_config(seed=13, fft_size=128)
    params = _training_params()
    params["normalize"] = True

    train_loader = object()
    val_loader = object()
    test_loader = object()
    data_info = {"dataset": "info"}

    prepare_datasets = MagicMock(
        return_value=(
            train_loader,
            val_loader,
            test_loader,
            data_info,
        )
    )
    monkeypatch.setattr(
        training_module,
        "prepare_torchsig_datasets",
        prepare_datasets,
    )

    model = torch.nn.Linear(4, 3)
    model_factory = MagicMock(return_value=model)
    monkeypatch.setitem(
        training_module.MODEL_FACTORY,
        "efficientnet_b0",
        model_factory,
    )

    wrapped_model = SimpleNamespace(model=model)
    metrics_callback = SimpleNamespace(val_f1s=[0.75])

    train_validate = MagicMock(
        return_value=(
            wrapped_model,
            metrics_callback,
        )
    )
    monkeypatch.setattr(
        training_module,
        "train_validate",
        train_validate,
    )

    test_metrics = MagicMock()
    evaluate_classifier = MagicMock(
        return_value=test_metrics,
    )
    monkeypatch.setattr(
        training_module,
        "evaluate_classifier",
        evaluate_classifier,
    )

    set_deterministic = MagicMock()
    monkeypatch.setattr(
        training_module,
        "set_deterministic",
        set_deterministic,
    )

    compute_num_params = MagicMock(return_value=15)
    monkeypatch.setattr(
        training_module,
        "compute_num_params",
        compute_num_params,
    )

    monkeypatch.setattr(
        training_module.TorchSigSignalLists,
        "all_signals",
        ["bpsk", "qpsk", "ook"],
    )

    logger = MagicMock()
    checkpoint_dir = tmp_path / "checkpoints"
    metrics_dir = tmp_path / "metrics"

    result = train_efficientnet_2d(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        params=params,
        checkpoint_dir=checkpoint_dir,
        metrics_dir=metrics_dir,
        dataset_root=tmp_path / "datasets",
        overwrite=True,
        model_name="efficientnet_b0",
        signal_generators=["bpsk"],
        logger=logger,
        accelerator="cpu",
        devices=1,
    )

    set_deterministic.assert_called_once_with(11)

    assert checkpoint_dir.is_dir()

    prepare_datasets.assert_called_once()
    prepare_call = prepare_datasets.call_args

    assert prepare_call.args == (
        train_cfg,
        val_cfg,
        test_cfg,
    )
    assert prepare_call.kwargs["dataset_root"] == tmp_path / "datasets"
    assert prepare_call.kwargs["batch_size"] == 8
    assert prepare_call.kwargs["overwrite"] is True
    assert prepare_call.kwargs["signal_generators"] == ["bpsk"]

    transforms = prepare_call.kwargs["transforms"]
    assert len(transforms) == 1
    assert isinstance(transforms[0], Spectrogram)
    assert transforms[0].fft_size == 128

    model_factory.assert_called_once_with(
        num_classes=3,
        drop_path_rate=0.1,
        drop_rate=0.2,
        normalize=True,
    )

    train_validate.assert_called_once()
    training_call = train_validate.call_args.kwargs

    assert training_call["train_loader"] is train_loader
    assert training_call["val_loader"] is val_loader
    assert training_call["model"] is model
    assert isinstance(
        training_call["criterion"],
        torch.nn.CrossEntropyLoss,
    )
    assert isinstance(
        training_call["optimizer"],
        torch.optim.AdamW,
    )
    assert isinstance(
        training_call["scheduler"],
        torch.optim.lr_scheduler.SequentialLR,
    )
    assert training_call["max_epochs"] == 10
    assert training_call["num_classes"] == 3
    assert training_call["metrics_dir"] == metrics_dir
    assert training_call["checkpoint_dir"] == checkpoint_dir
    assert training_call["logger"] is logger
    assert training_call["accelerator"] == "cpu"
    assert training_call["devices"] == 1

    expected_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    evaluate_classifier.assert_called_once_with(
        model=model,
        test_loader=test_loader,
        device=expected_device,
        num_classes=3,
        criterion=training_call["criterion"],
    )

    test_metrics.save_to_csv.assert_called_once_with(
        metrics_dir / "test"
    )
    compute_num_params.assert_called_once_with(model)

    assert result == {
        "pl_model": wrapped_model,
        "model": model,
        "metrics": metrics_callback,
        "test_metrics": test_metrics,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_classes": 3,
        "num_params": 15,
        "data_info": data_info,
    }


def test_train_efficientnet_2d_uses_default_optional_model_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify default dropout parameters are used when omitted."""
    train_cfg = _dataset_config()
    val_cfg = _dataset_config()
    test_cfg = _dataset_config()

    params = {
        "batch_size": 8,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 2,
    }

    monkeypatch.setattr(
        training_module,
        "prepare_torchsig_datasets",
        MagicMock(
            return_value=(
                object(),
                object(),
                object(),
                {},
            )
        ),
    )

    model = torch.nn.Linear(2, 2)
    model_factory = MagicMock(return_value=model)
    monkeypatch.setitem(
        training_module.MODEL_FACTORY,
        "efficientnet_b2",
        model_factory,
    )

    test_metrics = MagicMock()
    monkeypatch.setattr(
        training_module,
        "train_validate",
        MagicMock(
            return_value=(
                SimpleNamespace(model=model),
                MagicMock(),
            )
        ),
    )
    monkeypatch.setattr(
        training_module,
        "evaluate_classifier",
        MagicMock(return_value=test_metrics),
    )
    monkeypatch.setattr(
        training_module,
        "compute_num_params",
        MagicMock(return_value=6),
    )
    monkeypatch.setattr(
        training_module,
        "set_deterministic",
        MagicMock(),
    )
    monkeypatch.setattr(
        training_module.TorchSigSignalLists,
        "all_signals",
        ["bpsk", "qpsk"],
    )

    train_efficientnet_2d(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        params=params,
        checkpoint_dir=tmp_path / "checkpoints",
        model_name="efficientnet_b2",
    )

    model_factory.assert_called_once_with(
        num_classes=2,
        drop_path_rate=0.2,
        drop_rate=0.3,
        normalize=False,
    )


def test_train_efficientnet_2d_uses_checkpoint_metrics_directory_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify metrics default to a subdirectory of the checkpoint directory."""
    train_cfg = _dataset_config()
    val_cfg = _dataset_config()
    test_cfg = _dataset_config()
    params = _training_params()

    monkeypatch.setattr(
        training_module,
        "prepare_torchsig_datasets",
        MagicMock(
            return_value=(
                object(),
                object(),
                object(),
                {},
            )
        ),
    )

    model = torch.nn.Linear(2, 2)
    monkeypatch.setitem(
        training_module.MODEL_FACTORY,
        "efficientnet_b0",
        MagicMock(return_value=model),
    )

    train_validate = MagicMock(
        return_value=(
            SimpleNamespace(model=model),
            MagicMock(),
        )
    )
    monkeypatch.setattr(
        training_module,
        "train_validate",
        train_validate,
    )

    test_metrics = MagicMock()
    monkeypatch.setattr(
        training_module,
        "evaluate_classifier",
        MagicMock(return_value=test_metrics),
    )
    monkeypatch.setattr(
        training_module,
        "compute_num_params",
        MagicMock(return_value=6),
    )
    monkeypatch.setattr(
        training_module,
        "set_deterministic",
        MagicMock(),
    )
    monkeypatch.setattr(
        training_module.TorchSigSignalLists,
        "all_signals",
        ["bpsk", "qpsk"],
    )

    checkpoint_dir = tmp_path / "checkpoints"

    train_efficientnet_2d(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        params=params,
        checkpoint_dir=checkpoint_dir,
        model_name="efficientnet_b0",
    )

    expected_metrics_dir = checkpoint_dir / "metrics"

    assert (
        train_validate.call_args.kwargs["metrics_dir"]
        == expected_metrics_dir
    )
    test_metrics.save_to_csv.assert_called_once_with(
        expected_metrics_dir / "test"
    )


def test_train_efficientnet_2d_builds_expected_loss_and_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify loss smoothing and AdamW parameters come from the config."""
    train_cfg = _dataset_config()
    val_cfg = _dataset_config()
    test_cfg = _dataset_config()
    params = _training_params()

    monkeypatch.setattr(
        training_module,
        "prepare_torchsig_datasets",
        MagicMock(
            return_value=(
                object(),
                object(),
                object(),
                {},
            )
        ),
    )

    model = torch.nn.Linear(4, 3)
    monkeypatch.setitem(
        training_module.MODEL_FACTORY,
        "efficientnet_b4",
        MagicMock(return_value=model),
    )

    train_validate = MagicMock(
        return_value=(
            SimpleNamespace(model=model),
            MagicMock(),
        )
    )
    monkeypatch.setattr(
        training_module,
        "train_validate",
        train_validate,
    )

    monkeypatch.setattr(
        training_module,
        "evaluate_classifier",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        training_module,
        "compute_num_params",
        MagicMock(return_value=15),
    )
    monkeypatch.setattr(
        training_module,
        "set_deterministic",
        MagicMock(),
    )
    monkeypatch.setattr(
        training_module.TorchSigSignalLists,
        "all_signals",
        ["bpsk", "qpsk", "ook"],
    )

    train_efficientnet_2d(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        params=params,
        checkpoint_dir=tmp_path / "checkpoints",
        model_name="efficientnet_b4",
    )

    training_kwargs = train_validate.call_args.kwargs
    criterion = training_kwargs["criterion"]
    optimizer = training_kwargs["optimizer"]
    scheduler = training_kwargs["scheduler"]

    assert isinstance(
        criterion,
        torch.nn.CrossEntropyLoss,
    )
    assert criterion.label_smoothing == pytest.approx(0.05)

    assert isinstance(
        optimizer,
        torch.optim.AdamW,
    )
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(
        1e-3
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        1e-4
    )
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        1e-4
    )

    assert isinstance(
        scheduler,
        torch.optim.lr_scheduler.SequentialLR,
    )
