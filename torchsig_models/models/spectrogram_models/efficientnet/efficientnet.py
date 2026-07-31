"""2D EfficientNet models for spectrogram classification."""

import timm
import torch
from torch import nn
import logging

__all__ = [
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_b4",
    "NormalizedModel",
    "SpectrogramNormalization",
]


logger = logging.getLogger(__name__)

PRETRAINED_CHECKPOINTS = {
    # Reserved for future local checkpoint support.
}


class SpectrogramNormalization(nn.Module):
    """Normalize spectrogram inputs per example and channel."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + self.eps)


class NormalizedModel(nn.Module):
    """Wrap a model with spectrogram normalization."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.normalize = SpectrogramNormalization()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))


def _load_pretrained_if_requested(
    model: nn.Module,
    model_name: str,
    pretrained: bool,
    checkpoint_path: str | None,
) -> None:
    _ = model
    _ = checkpoint_path

    if not pretrained:
        return

    raise NotImplementedError(
        f"Pretrained loading for {model_name} is intentionally disabled for now. "
        "Pass checkpoint_path/load logic here later."
    )


def _create_effnet_2d(
    model_name: str,
    num_classes: int,
    input_channels: int,
    drop_path_rate: float,
    drop_rate: float,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
) -> nn.Module:
    """Create and configure a 2D EfficientNet model."""
    model = timm.create_model(
        model_name,
        num_classes=num_classes,
        in_chans=input_channels,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
    )

    _load_pretrained_if_requested(
        model=model,
        model_name=model_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )

    return NormalizedModel(model)


def efficientnet_b0(
    num_classes: int = 57,
    input_channels: int = 1,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
) -> nn.Module:
    """Construct a 2D EfficientNet-B0 model for spectrogram classification."""
    return _create_effnet_2d(
        "efficientnet_b0",
        num_classes=num_classes,
        input_channels=input_channels,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )


def efficientnet_b2(
    num_classes: int = 57,
    input_channels: int = 1,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
) -> nn.Module:
    """Construct a 2D EfficientNet-B2 model for spectrogram classification."""
    return _create_effnet_2d(
        "efficientnet_b2",
        num_classes=num_classes,
        input_channels=input_channels,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )


def efficientnet_b4(
    num_classes: int = 57,
    input_channels: int = 1,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
) -> nn.Module:
    """Construct a 2D EfficientNet-B4 model for spectrogram classification."""
    return _create_effnet_2d(
        "efficientnet_b4",
        num_classes=num_classes,
        input_channels=input_channels,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )
