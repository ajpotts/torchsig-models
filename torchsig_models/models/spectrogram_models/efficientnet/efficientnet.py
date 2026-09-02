"""2D EfficientNet models for spectrogram classification."""

import timm
import torch
from torch import nn

__all__ = [
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_b4",
    "NormalizedModel",
    "SpectrogramNormalization",
]


DEFAULT_NUM_CLASSES = 72

PRETRAINED_NUM_CLASSES = {
    "efficientnet_b0": DEFAULT_NUM_CLASSES,
    "efficientnet_b2": DEFAULT_NUM_CLASSES,
    "efficientnet_b4": DEFAULT_NUM_CLASSES,
}


class SpectrogramNormalization(nn.Module):
    """Normalize spectrogram inputs per example and channel."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True, correction=0)
        return (x - mean) / (std + self.eps)


class NormalizedModel(nn.Module):
    """Wrap a model with input-shape handling and optional normalization."""

    def __init__(self, model: nn.Module, normalize: bool = True):
        """Initialize the wrapper.

        Args:
            model: Spectrogram model receiving four-dimensional tensors.
            normalize: Whether to standardize each sample and channel.
        """
        super().__init__()
        self.normalize = SpectrogramNormalization() if normalize else nn.Identity()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(
                "Expected spectrogram input with shape [batch, frequency, time] "
                "or [batch, channels, frequency, time], "
                f"got {tuple(x.shape)}."
            )
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
    normalize: bool = False,
) -> nn.Module:
    """Create and configure a 2D EfficientNet model."""
    model_num_classes = (
        PRETRAINED_NUM_CLASSES[model_name] if pretrained else num_classes
    )
    model = timm.create_model(
        model_name,
        num_classes=model_num_classes,
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

    if num_classes != model_num_classes:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return NormalizedModel(model, normalize=normalize)


def efficientnet_b0(
    num_classes: int = DEFAULT_NUM_CLASSES,
    input_channels: int = 1,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
    normalize: bool = False,
) -> nn.Module:
    """Construct a 2D EfficientNet-B0 spectrogram classifier.

    Args:
        normalize: Whether to standardize each input sample and channel.
    """
    return _create_effnet_2d(
        "efficientnet_b0",
        num_classes=num_classes,
        input_channels=input_channels,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        normalize=normalize,
    )


def efficientnet_b2(
    num_classes: int = DEFAULT_NUM_CLASSES,
    input_channels: int = 1,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
    normalize: bool = False,
) -> nn.Module:
    """Construct a 2D EfficientNet-B2 spectrogram classifier.

    Args:
        normalize: Whether to standardize each input sample and channel.
    """
    return _create_effnet_2d(
        "efficientnet_b2",
        num_classes=num_classes,
        input_channels=input_channels,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        normalize=normalize,
    )


def efficientnet_b4(
    num_classes: int = DEFAULT_NUM_CLASSES,
    input_channels: int = 1,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
    normalize: bool = False,
) -> nn.Module:
    """Construct a 2D EfficientNet-B4 spectrogram classifier.

    Args:
        normalize: Whether to standardize each input sample and channel.
    """
    return _create_effnet_2d(
        "efficientnet_b4",
        num_classes=num_classes,
        input_channels=input_channels,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        normalize=normalize,
    )
