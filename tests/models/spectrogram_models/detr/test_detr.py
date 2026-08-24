from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchsig_models.models.spectrogram_models.detr.detr as detr_module
from torchsig_models.models.spectrogram_models.detr import detr_b0_nano
from torchsig_models.models.spectrogram_models.detr.utils import (
    format_preds,
    format_targets,
)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_detr_b0_nano_forward_preserves_batch_dimension(batch_size: int) -> None:
    model = detr_b0_nano(num_classes=3).eval()

    with torch.no_grad():
        output = model(torch.randn(batch_size, 2, 128, 128))

    assert output["pred_logits"].shape[0] == batch_size
    assert output["pred_logits"].shape[-1] == 4
    assert output["pred_boxes"].shape == (
        batch_size,
        output["pred_logits"].shape[1],
        4,
    )
    assert torch.all((0 <= output["pred_boxes"]) & (output["pred_boxes"] <= 1))


@pytest.mark.parametrize(
    ("factory_name", "default_num_classes"),
    [
        ("detr_b0_nano", 1),
        ("detr_b2_nano", 1),
        ("detr_b4_nano", 1),
        ("detr_b0_nano_mod_family", 6),
        ("detr_b2_nano_mod_family", 6),
        ("detr_b4_nano_mod_family", 6),
    ],
)
def test_factory_includes_no_object_class(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    default_num_classes: int,
) -> None:
    monkeypatch.setattr(
        detr_module,
        "create_detr",
        lambda **kwargs: SimpleNamespace(
            linear_class=nn.Linear(8, kwargs["num_classes"] + 1)
        ),
    )

    factory = getattr(detr_module, factory_name)
    default_out_features = factory().linear_class.out_features
    custom_out_features = factory(num_classes=4).linear_class.out_features

    assert default_out_features == default_num_classes + 1
    assert custom_out_features == 5


def test_b4_mod_family_uses_matching_pretrained_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(linear_class=nn.Linear(8, 7))
    loaded_model_names: list[str] = []
    monkeypatch.setattr(detr_module, "create_detr", lambda **kwargs: model)
    monkeypatch.setattr(
        detr_module,
        "_load_pretrained_weights",
        lambda model, *, path, model_name: loaded_model_names.append(model_name),
    )

    detr_module.detr_b4_nano_mod_family(pretrained=True)

    assert loaded_model_names == ["detr_b4_nano_mod_family"]


def test_format_predictions_stays_on_input_device_and_handles_empty() -> None:
    predictions = {
        "pred_logits": torch.tensor([[[0.0, 10.0], [0.0, 10.0]]]),
        "pred_boxes": torch.full((1, 2, 4), 0.5),
    }

    formatted = format_preds(predictions)

    assert formatted[0]["boxes"].shape == (0, 4)
    assert formatted[0]["scores"].shape == (0,)
    assert formatted[0]["labels"].shape == (0,)
    assert formatted[0]["boxes"].device == predictions["pred_boxes"].device


def test_format_targets_stays_on_input_device_and_handles_empty() -> None:
    targets = [
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty((0,), dtype=torch.int64),
        }
    ]

    formatted = format_targets(targets)

    assert formatted[0]["boxes"].shape == (0, 4)
    assert formatted[0]["labels"].shape == (0,)
    assert formatted[0]["boxes"].device == targets[0]["boxes"].device
