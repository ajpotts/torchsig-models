"""Focused tests for the XCiT model and inference helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import torchsig_models.models.iq_models.xcit.xcit1d as xcit_module
from torchsig_models.models.iq_models.xcit.xcit1d import FocalLoss, XCiT1d
from torchsig_models.models.iq_models.xcit.xcit1d_inference import (
    _to_single_class_index,
    narrowband_classifier_collate,
)
from torchsig_models.utils.classifier_metrics_tracker import (
    ClassifierMetricsTrackerCallback,
)


class _PositionEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, batch: int, height: int, width: int) -> torch.Tensor:
        return torch.zeros(batch, self.channels, height, width)


class _IdentityBlock(nn.Module):
    def forward(
        self,
        inputs: torch.Tensor,
        *_shape: int,
    ) -> torch.Tensor:
        return inputs


class _FakeBackbone(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.num_features = channels
        self.patch_embed: nn.Module = nn.Identity()
        self.pos_embed = _PositionEmbedding(channels)
        self.blocks = nn.ModuleList([_IdentityBlock()])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.cls_attn_blocks = nn.ModuleList([_IdentityBlock()])
        self.norm = nn.LayerNorm(channels)
        self.head: nn.Module = nn.Identity()


def test_xcit1d_forward_returns_one_logit_vector_per_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the adapted XCiT forward path without constructing a large model."""
    monkeypatch.setattr(
        xcit_module.timm,
        "create_model",
        lambda *_args, **_kwargs: _FakeBackbone(channels=8),
    )

    model = XCiT1d(
        input_channels=2,
        n_features=3,
        ds_method="chunk",
        ds_rate=2,
    )

    output = model(torch.randn(4, 2, 16))

    assert output.shape == (4, 3)
    assert torch.isfinite(output).all()


def test_focal_loss_matches_cross_entropy_when_gamma_is_zero() -> None:
    """Gamma zero reduces focal loss to ordinary cross entropy."""
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
    targets = torch.tensor([0, 1])

    actual = FocalLoss(gamma=0.0)(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, targets)

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "target",
    [
        2,
        np.int64(2),
        np.array([2]),
        torch.tensor([2]),
        [np.int64(2)],
        {"class_index": np.int64(2)},
    ],
)
def test_to_single_class_index_accepts_torchsig_target_variants(target: object) -> None:
    assert _to_single_class_index(target) == 2


def test_narrowband_classifier_collate_converts_complex_iq_and_labels() -> None:
    batch = [
        (np.array([1 + 2j, 3 + 4j]), [np.int64(1)]),
        (np.array([5 + 6j, 7 + 8j]), {"class_index": np.int64(0)}),
    ]

    inputs, labels = narrowband_classifier_collate(batch)

    assert inputs.shape == (2, 2, 2)
    assert inputs.dtype == torch.float32
    assert torch.equal(inputs[0], torch.tensor([[1.0, 3.0], [2.0, 4.0]]))
    assert torch.equal(labels, torch.tensor([1, 0]))


def test_xcit_classifier_runs_one_epoch_with_metrics_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a CPU epoch with the metrics callback used by XCiT training."""
    import pytorch_lightning as pl

    class TinyClassifier(nn.Module):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__()
            self.network = nn.Sequential(nn.Flatten(), nn.Linear(8, 2))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.network(inputs)

    monkeypatch.setattr(xcit_module, "XCiT1d", TinyClassifier)
    model = xcit_module.XCiTClassifier(input_channels=2, num_classes=2)
    loader = DataLoader(
        TensorDataset(torch.randn(4, 2, 4), torch.tensor([0, 1, 0, 1])),
        batch_size=2,
    )
    callback = ClassifierMetricsTrackerCallback(n_classes=2)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_train_batches=1,
        limit_val_batches=1,
    )

    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)

    assert len(callback.train_metrics.history["loss"]) == 1
    assert len(callback.val_metrics.history["loss"]) == 1
