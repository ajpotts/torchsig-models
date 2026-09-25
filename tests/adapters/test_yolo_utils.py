"""Tests for YOLO dataset conversion and model downloading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import requests
import yaml

import torchsig_models.adapters.yolo_utils as yolo_utils


class _StaticDataset:
    def __init__(self) -> None:
        self.items = [
            (
                np.arange(16, dtype=np.float32).reshape(4, 4),
                [[2, 0.5, 0.5, 0.25, 0.75]],
            )
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> object:
        return self.items[index]


def test_static_to_yolo_writes_image_label_and_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(yolo_utils.TorchSigSignalLists, "all_signals", ["a", "b", "c"])

    yolo_utils.static_to_yolo(_StaticDataset(), str(tmp_path), train=True)

    label_path = tmp_path / "labels" / "train" / "0000000000.txt"
    image_path = tmp_path / "images" / "train" / "0000000000.png"
    config = yaml.safe_load((tmp_path / "dataset_yolo_config.yaml").read_text())

    assert label_path.read_text() == "2 0.5 0.5 0.25 0.75"
    assert image_path.is_file()
    assert image_path.stat().st_size > 0
    assert config == {
        "path": str(tmp_path),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": {0: "a", 1: "b", 2: "c"},
    }


def test_iterable_to_yolo_consumes_requested_number_of_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample = (np.ones((3, 3), dtype=np.float32), [])
    dataset = iter([sample, sample])
    monkeypatch.setattr(yolo_utils.TorchSigSignalLists, "all_signals", ["signal"])

    yolo_utils.iterable_to_yolo(dataset, str(tmp_path), train=False, length=2)

    assert len(list((tmp_path / "images" / "val").glob("*.png"))) == 2
    assert len(list((tmp_path / "labels" / "val").glob("*.txt"))) == 2


class _InterruptedResponse:
    def __enter__(self) -> _InterruptedResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        assert chunk_size == 8192
        yield b"partial checkpoint"
        raise requests.ConnectionError("connection interrupted")


@pytest.mark.xfail(
    strict=True,
    reason="YOLO downloads are not yet written atomically to a temporary file",
)
def test_get_yolo_model_does_not_leave_partial_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "models" / "yolo11n.pt"
    monkeypatch.setattr(
        yolo_utils.requests,
        "get",
        lambda *_args, **_kwargs: _InterruptedResponse(),
    )

    with pytest.raises(requests.ConnectionError):
        yolo_utils.get_yolo_model(destination)

    assert not destination.exists()
