from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from torchsig_models.models.spectrogram_models.xcit import xcit_nano
from torchsig_models.models.spectrogram_models.xcit.xcit_train import (
    _resolve_config_path,
    _validate_wideband_config,
    _wideband_transforms,
    load_training_params,
    parse_args,
)


@pytest.mark.parametrize(
    ("shape", "num_classes"),
    [
        ((2, 1, 128, 160), 7),
        ((2, 2, 160, 128), 5),
        ((2, 128, 160), 3),
    ],
)
def test_xcit_nano_forward(shape: tuple[int, ...], num_classes: int) -> None:
    input_channels = shape[1] if len(shape) == 4 else 1
    model = xcit_nano(num_classes=num_classes, input_channels=input_channels).eval()

    with torch.no_grad():
        output = model(torch.randn(shape))

    assert output.shape == (shape[0], num_classes)


def test_xcit_nano_rejects_invalid_input_shape() -> None:
    model = xcit_nano()

    with pytest.raises(ValueError, match="Expected spectrogram input"):
        model(torch.randn(2, 128))


def test_xcit_nano_cpu_smoke_training_step() -> None:
    model = xcit_nano(num_classes=4, input_channels=2, drop_path_rate=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randn(2, 2, 64, 80)
    targets = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
    )

    optimizer.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(inputs), targets
    )
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_wideband_config_requires_multiple_signals() -> None:
    cfg = SimpleNamespace(
        output_representation="spectrogram",
        dataset_metadata={"num_signals_max": 1},
    )

    with pytest.raises(ValueError, match="allow multiple signals"):
        _validate_wideband_config(cfg, "training")


def test_wideband_transforms_create_multi_hot_targets() -> None:
    cfg = SimpleNamespace(dataset_metadata={"fft_size": 64})

    transforms = _wideband_transforms(cfg, num_classes=5)

    assert transforms[0].fft_size == 64
    assert transforms[1].num_classes == 5
    assert transforms[1].output_key == "multi_hot_label"


def test_xcit_nano_checkpoint_round_trip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "xcit.pt"
    model = xcit_nano(num_classes=2, input_channels=2)
    torch.save(model.state_dict(), checkpoint_path)

    restored = xcit_nano(
        num_classes=2,
        input_channels=2,
        checkpoint_path=checkpoint_path,
    )

    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_default_training_params_are_available() -> None:
    params = load_training_params()

    assert params["model_name"] == "xcit_nano"
    assert params["input_channels"] == 1
    assert params["experiment_name"] == "spectrogram_wideband_multilabel"
    assert params["decision_threshold"] == 0.5


def test_load_training_params_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Training parameter file not found"):
        load_training_params(tmp_path / "missing.yaml")


def test_training_cli_matches_spectrogram_conventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xcit_train.py",
            "--train-config",
            "train.yaml",
            "--val-config",
            "val.yaml",
            "--test-config",
            "test.yaml",
            "--dataset-root",
            "custom-datasets",
            "--output-dir",
            "custom-runs",
            "--dataset-length",
            "100",
            "--dataset-id",
            "xcit-test",
            "--devices",
            "2",
        ],
    )

    args = parse_args()

    assert args.dataset_config is None
    assert args.train_config == Path("train.yaml")
    assert args.val_config == Path("val.yaml")
    assert args.test_config == Path("test.yaml")
    assert args.dataset_root == Path("custom-datasets")
    assert args.output_dir == Path("custom-runs")
    assert args.dataset_length == 100
    assert args.dataset_id == "xcit-test"
    assert args.devices == 2


def test_resolve_config_path_matches_efficientnet_behavior() -> None:
    shared = Path("shared.yaml")
    split = Path("train.yaml")

    assert _resolve_config_path(shared, split, "train") == split
    assert _resolve_config_path(shared, None, "train") == shared
    with pytest.raises(ValueError, match="--train-config"):
        _resolve_config_path(None, None, "train")
