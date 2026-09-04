"""2D EfficientNet models for spectrogram classification."""

import timm
import torch
from torch import nn

from torchsig_models.utils.normalization import (
    DatasetNormalization,
    NormalizationMode,
    resolve_normalization_mode,
)

__all__ = [
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_b4",
    "NormalizedModel",
    "SpectrogramNormalization",
]


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

    def __init__(
        self,
        model: nn.Module,
        normalization: NormalizationMode | None = None,
        normalization_mean: torch.Tensor | list[float] | None = None,
        normalization_std: torch.Tensor | list[float] | None = None,
        normalization_eps: float = 1e-6,
        normalize: bool | None = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            model: Spectrogram model receiving four-dimensional tensors.
            normalization: Input normalization strategy. Defaults to dataset.
            normalization_mean: Training-dataset channel means.
            normalization_std: Training-dataset channel standard deviations.
            normalization_eps: Numerical stability constant.
            normalize: Deprecated boolean alias for sample/none normalization.
        """
        super().__init__()
        self.normalization_mode = resolve_normalization_mode(normalization, normalize)
        if self.normalization_mode == "dataset":
            if normalization_mean is None or normalization_std is None:
                raise ValueError(
                    "Dataset normalization requires normalization_mean and "
                    "normalization_std."
                )
            self.normalize = DatasetNormalization(
                normalization_mean,
                normalization_std,
                eps=normalization_eps,
            )
        elif self.normalization_mode == "sample":
            self.normalize = SpectrogramNormalization(eps=normalization_eps)
        elif self.normalization_mode == "none":
            self.normalize = nn.Identity()
        else:
            raise ValueError(
                f"Unsupported normalization mode: {self.normalization_mode}"
            )
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
    normalization: NormalizationMode | None = None,
    normalization_mean: torch.Tensor | list[float] | None = None,
    normalization_std: torch.Tensor | list[float] | None = None,
    normalization_eps: float = 1e-6,
    normalize: bool | None = None,
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

    return NormalizedModel(
        model,
        normalization=normalization,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        normalization_eps=normalization_eps,
        normalize=normalize,
    )


def efficientnet_b0(
    num_classes: int = 72,
    input_channels: int = 1,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
    normalization: NormalizationMode | None = None,
    normalization_mean: torch.Tensor | list[float] | None = None,
    normalization_std: torch.Tensor | list[float] | None = None,
    normalization_eps: float = 1e-6,
    normalize: bool | None = None,
) -> nn.Module:
    """Construct a 2D EfficientNet-B0 spectrogram classifier.

    Args:
        normalization: Normalization mode; defaults to training-dataset
            normalization.
        normalization_mean: Training-dataset channel means for dataset mode.
        normalization_std: Training-dataset channel standard deviations.
        normalization_eps: Numerical stability constant.
        normalize: Deprecated boolean alias for sample/none mode.
    """
    return _create_effnet_2d(
        "efficientnet_b0",
        num_classes=num_classes,
        input_channels=input_channels,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        normalization=normalization,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        normalization_eps=normalization_eps,
        normalize=normalize,
    )


def efficientnet_b2(
    num_classes: int = 72,
    input_channels: int = 1,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
    normalization: NormalizationMode | None = None,
    normalization_mean: torch.Tensor | list[float] | None = None,
    normalization_std: torch.Tensor | list[float] | None = None,
    normalization_eps: float = 1e-6,
    normalize: bool | None = None,
) -> nn.Module:
    """Construct a 2D EfficientNet-B2 spectrogram classifier.

    Args:
        normalization: Normalization mode; defaults to training-dataset
            normalization.
        normalization_mean: Training-dataset channel means for dataset mode.
        normalization_std: Training-dataset channel standard deviations.
        normalization_eps: Numerical stability constant.
        normalize: Deprecated boolean alias for sample/none mode.
    """
    return _create_effnet_2d(
        "efficientnet_b2",
        num_classes=num_classes,
        input_channels=input_channels,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        normalization=normalization,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        normalization_eps=normalization_eps,
        normalize=normalize,
    )


def efficientnet_b4(
    num_classes: int = 72,
    input_channels: int = 1,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
    normalization: NormalizationMode | None = None,
    normalization_mean: torch.Tensor | list[float] | None = None,
    normalization_std: torch.Tensor | list[float] | None = None,
    normalization_eps: float = 1e-6,
    normalize: bool | None = None,
) -> nn.Module:
    """Construct a 2D EfficientNet-B4 spectrogram classifier.

    Args:
        normalization: Normalization mode; defaults to training-dataset
            normalization.
        normalization_mean: Training-dataset channel means for dataset mode.
        normalization_std: Training-dataset channel standard deviations.
        normalization_eps: Numerical stability constant.
        normalize: Deprecated boolean alias for sample/none mode.
    """
    return _create_effnet_2d(
        "efficientnet_b4",
        num_classes=num_classes,
        input_channels=input_channels,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        normalization=normalization,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        normalization_eps=normalization_eps,
        normalize=normalize,
    )
