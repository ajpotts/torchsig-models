"""Tests for the Ultralytics RT-DETR model factory."""

from __future__ import annotations

from pathlib import Path

import pytest

import torchsig_models.models.spectrogram_models.detr.detr as detr_module


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "rtdetr-l.pt"),
        ({"pretrained": False}, "rtdetr-l.yaml"),
        ({"path": Path("custom.pt")}, "custom.pt"),
    ],
)
def test_rtdetr_l_selects_requested_model_source(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    expected: str,
) -> None:
    sources: list[str] = []
    monkeypatch.setattr(
        detr_module,
        "RTDETR",
        lambda source: sources.append(source) or object(),
    )

    detr_module.rtdetr_l(**kwargs)

    assert sources == [expected]
