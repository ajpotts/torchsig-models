"""Tests for the Ultralytics RT-DETR comparison adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

import torchsig_models.models.spectrogram_models.detr.ultralytics_rtdetr_train as training_module
from torchsig_models.models.spectrogram_models.detr.ultralytics_rtdetr_train import (
    export_ultralytics_dataset,
    spectrogram_to_uint8,
    train_ultralytics_rtdetr,
)


class _Dataset:
    def __init__(self, image: np.ndarray, labels: list[list[float]]) -> None:
        self.item = image, labels

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[np.ndarray, list[list[float]]]:
        assert index == 0
        return self.item


def test_spectrogram_to_uint8_uses_sample_normalization() -> None:
    image = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)

    encoded = spectrogram_to_uint8(
        image, normalization="sample", clip_sigma=1.0
    )

    assert encoded.dtype == np.uint8
    assert encoded.shape == image.shape
    assert encoded[0, 0] == 0
    assert encoded[0, 2] == 255


def test_spectrogram_to_uint8_requires_dataset_statistics() -> None:
    with pytest.raises(ValueError, match="requires mean and std"):
        spectrogram_to_uint8(np.ones((2, 2)), normalization="dataset")


def test_export_writes_matching_splits_labels_and_class_order(tmp_path: Path) -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    datasets = {
        split: _Dataset(image, [[1, 0.25, 0.5, 0.1, 0.2]])
        for split in ("train", "val", "test")
    }

    config_path = export_ultralytics_dataset(
        datasets,
        tmp_path / "export",
        ["tone", "ofdm-64"],
        normalization="dataset",
        mean=float(image.mean()),
        std=float(image.std()),
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["names"] == {0: "tone", 1: "ofdm-64"}
    assert config["test"] == "images/test"
    for split in ("train", "val", "test"):
        label = config_path.parent / "labels" / split / "0000000000.txt"
        exported_image = cv2.imread(
            str(config_path.parent / "images" / split / "0000000000.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        assert label.read_text(encoding="utf-8") == "1 0.25 0.5 0.1 0.2"
        assert exported_image is not None
        assert exported_image.shape == (4, 4)


def test_export_refuses_to_replace_existing_dataset(tmp_path: Path) -> None:
    root = tmp_path / "export"
    root.mkdir()

    with pytest.raises(FileExistsError, match="--overwrite"):
        export_ultralytics_dataset(
            {}, root, ["tone"], normalization="sample", overwrite=False
        )


def test_train_uses_best_checkpoint_and_held_out_test_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = _Dataset(np.ones((4, 4), dtype=np.float32), [])
    loader = SimpleNamespace(dataset=dataset)
    monkeypatch.setattr(
        training_module,
        "prepare_torchsig_datasets",
        lambda *_args, **_kwargs: (
            loader,
            loader,
            loader,
            {"class_names": ["tone", "ofdm-64"]},
        ),
    )
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("names: [tone, ofdm-64]\n", encoding="utf-8")
    monkeypatch.setattr(
        training_module,
        "export_ultralytics_dataset",
        lambda *_args, **_kwargs: dataset_yaml,
    )

    calls: list[tuple[str, dict[str, object]]] = []

    class _Boxes:
        map = 0.1
        map50 = 0.2
        mp = 0.3
        mr = 0.4

    class _FakeRTDETR:
        def __init__(self, model: str) -> None:
            calls.append(("init", {"model": model}))
            self.trainer = None

        def train(self, **kwargs: object) -> str:
            calls.append(("train", kwargs))
            checkpoint = tmp_path / "run" / "train" / "weights" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            self.trainer = SimpleNamespace(best=checkpoint)
            return "trained"

        def val(self, **kwargs: object) -> SimpleNamespace:
            calls.append(("val", kwargs))
            return SimpleNamespace(box=_Boxes())

    monkeypatch.setattr(training_module, "RTDETR", _FakeRTDETR)
    cfg = SimpleNamespace(seed=7, output_spectrogram_fft=512)

    result = train_ultralytics_rtdetr(
        cfg,
        cfg,
        cfg,
        {
            "batch_size": 2,
            "max_epochs": 3,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "normalization": "sample",
        },
        tmp_path / "run",
        model="rtdetr-l.pt",
    )

    assert calls[0] == ("init", {"model": "rtdetr-l.pt"})
    assert calls[1][0] == "train"
    assert calls[1][1]["deterministic"] is False
    assert calls[1][1]["mosaic"] == 0.0
    assert calls[2][0] == "val"
    assert calls[2][1]["split"] == "test"
    assert result["summary"]["map_50"] == pytest.approx(0.2)
