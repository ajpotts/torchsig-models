"""Tests for TorchSig-to-Anomalib adapters and transforms."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import torchsig_models.adapters.anomalib_utils as anomalib_utils
from torchsig_models.adapters.anomalib_utils import (
    AnomalyLabel,
    SpectrogramRescale,
    TorchSigAnomalibDataModule,
    TorchSigAnomalibDataset,
)


class _FakeStaticDataset:
    samples: list[object] = [np.zeros((4, 5), dtype=np.float32) for _ in range(6)]

    def __init__(self, **_kwargs: object) -> None:
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> object:
        return self.samples[index]


@pytest.fixture
def fake_static_dataset(monkeypatch: pytest.MonkeyPatch) -> type[_FakeStaticDataset]:
    monkeypatch.setattr(anomalib_utils, "StaticTorchSigDataset", _FakeStaticDataset)
    _FakeStaticDataset.samples = [
        np.zeros((4, 5), dtype=np.float32) for _ in range(6)
    ]
    return _FakeStaticDataset


def test_anomalib_dataset_respects_inclusive_bounds(
    fake_static_dataset: type[_FakeStaticDataset],
    tmp_path,
) -> None:
    dataset = TorchSigAnomalibDataset(root=str(tmp_path), index_bounds=[1, 4])

    assert len(dataset) == 4
    assert dataset[0].image.shape == (3, 4, 5)
    assert dataset[3].image.shape == (3, 4, 5)


def test_anomalib_dataset_maps_anomaly_label(
    fake_static_dataset: type[_FakeStaticDataset],
    tmp_path,
) -> None:
    labels = ["signal", 0.0, 1.0, -0.5, 0.5, True]
    fake_static_dataset.samples = [(np.ones((3, 4), dtype=np.float32), labels)]
    dataset = TorchSigAnomalibDataset(
        root=str(tmp_path),
        index_bounds=[0, 0],
        use_anomaly_labels=True,
    )

    item = dataset[0]

    assert item.gt_label.item() is True
    assert torch.all(item.gt_mask == 1)


def test_anomaly_label_marks_only_configured_classes() -> None:
    transform = AnomalyLabel(anomaly_class_names=["interferer"])
    anomalous = {"class_name": "interferer"}
    normal = {"class_name": "wanted"}

    assert transform.__apply__(anomalous)["anomaly"] is True
    assert transform.__apply__(normal)["anomaly"] is False


def test_spectrogram_rescale_fixed_range_clips_and_scales() -> None:
    transform = SpectrogramRescale(clip_db_range=(-100.0, 0.0))
    signal = SimpleNamespace(
        data=np.array([[-120.0, -50.0, 10.0]], dtype=np.float32)
    )

    result = transform.__apply__(signal)

    assert result is signal
    assert np.allclose(signal.data, [[0.0, 0.5, 1.0]])
    assert signal.data.dtype == np.float32


@pytest.mark.xfail(
    strict=True,
    reason="Anomalib subset size is not yet checked against the source dataset",
)
def test_anomalib_datamodule_rejects_oversized_subset(
    fake_static_dataset: type[_FakeStaticDataset],
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="size"):
        TorchSigAnomalibDataModule(root=str(tmp_path), size=7)


@pytest.mark.xfail(
    strict=True,
    reason="Anomalib split proportions are not yet validated",
)
@pytest.mark.parametrize("splits", [(0.8, 0.4, -0.2), (0.5, 0.2, 0.2)])
def test_anomalib_datamodule_rejects_invalid_splits(
    fake_static_dataset: type[_FakeStaticDataset],
    tmp_path,
    splits: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match="split"):
        TorchSigAnomalibDataModule(root=str(tmp_path), splits=splits)

