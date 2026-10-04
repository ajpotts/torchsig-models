"""Tests for RT-DETR training and inference entry points."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch
import yaml

import torchsig_models.models.spectrogram_models.detr.detr_inference as inference_module
import torchsig_models.models.spectrogram_models.detr.detr_train as training_module
from torchsig_models.models.spectrogram_models.detr.detr_inference import (
    detr_inference,
    evaluate_detr,
)
from torchsig_models.models.spectrogram_models.detr.detr_train import (
    export_ultralytics_dataset,
    load_training_params,
    spectrogram_to_uint8,
    train_detr,
)


class _Dataset:
    def __init__(self, image: np.ndarray, labels: list[list[float]]) -> None:
        self.item = image, labels

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[np.ndarray, list[list[float]]]:
        assert index == 0
        return self.item


def test_load_training_params_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    expected = {"batch_size": 2, "max_epochs": 3}
    path.write_text(yaml.safe_dump(expected), encoding="utf-8")

    assert load_training_params("rtdetr_l", path) == expected


@pytest.mark.parametrize("normalization", ["sample", "dataset", "none"])
def test_spectrogram_to_uint8_supports_normalization_modes(
    normalization: str,
) -> None:
    image = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    kwargs = {"mean": 0.0, "std": 1.0} if normalization == "dataset" else {}

    encoded = spectrogram_to_uint8(
        image,
        normalization=normalization,
        clip_sigma=1.0,
        image_min=-1.0,
        image_max=1.0,
        **kwargs,
    )

    assert encoded.dtype == np.uint8
    assert encoded.tolist() == [[0, 128, 255]]


def test_export_writes_all_splits_and_preserves_class_order(tmp_path: Path) -> None:
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
        image_path = config_path.parent / "images" / split / "0000000000.png"
        assert label.read_text(encoding="utf-8") == "1 0.25 0.5 0.1 0.2"
        assert cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).shape == (4, 4)


def test_train_uses_best_checkpoint_and_test_split(
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

    class _Boxes:
        map = 0.7
        map50 = 0.8
        mp = 0.9
        mr = 0.85

    model = MagicMock()
    model.metrics = SimpleNamespace(box=_Boxes())
    model.trainer.best = tmp_path / "train" / "weights" / "best.pt"
    model.val.return_value = SimpleNamespace(box=_Boxes())
    factory = MagicMock(return_value=model)
    monkeypatch.setitem(training_module.MODEL_FACTORY, "rtdetr_l", factory)
    cfg = SimpleNamespace(seed=7, output_spectrogram_fft=512)

    result = train_detr(
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
    )

    train_call = model.train.call_args.kwargs
    assert train_call["deterministic"] is False
    assert train_call["mosaic"] == 0.0
    assert model.val.call_args.kwargs["split"] == "test"
    assert result["best_checkpoint"].endswith("train/weights/best.pt")
    assert result["test_summary"]["map_50"] == pytest.approx(0.8)


def test_inference_returns_normalized_detections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    boxes = SimpleNamespace(
        xywhn=torch.tensor([[0.5, 0.5, 0.1, 0.2]]),
        conf=torch.tensor([0.9]),
        cls=torch.tensor([1.0]),
    )
    model = MagicMock()
    model.predict.return_value = [
        SimpleNamespace(path="sample.png", boxes=boxes, names={0: "tone", 1: "ofdm"})
    ]
    monkeypatch.setattr(inference_module, "rtdetr_l", lambda **_kwargs: model)

    detections = detr_inference("images", checkpoint)

    assert detections[0]["labels"].tolist() == [1]
    assert detections[0]["boxes"].shape == (1, 4)
    assert detections[0]["scores"].item() == pytest.approx(0.9)


def test_evaluate_returns_detection_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "best.pt"
    dataset_yaml = tmp_path / "dataset.yaml"
    checkpoint.touch()
    dataset_yaml.touch()
    boxes = SimpleNamespace(map=0.7, map50=0.8, mp=0.9, mr=0.85)
    model = MagicMock()
    model.val.return_value = SimpleNamespace(box=boxes)
    monkeypatch.setattr(inference_module, "rtdetr_l", lambda **_kwargs: model)

    summary = evaluate_detr(dataset_yaml, checkpoint)

    assert summary == {
        "map": 0.7,
        "map_50": 0.8,
        "precision": 0.9,
        "recall": 0.85,
    }
    assert model.val.call_args.kwargs["split"] == "test"
