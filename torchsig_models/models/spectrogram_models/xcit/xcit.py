"""Two-dimensional XCiT classifiers for spectrogram inputs."""

from pathlib import Path

import timm
import torch
from torch import nn

from torchsig_models.models.spectrogram_models.efficientnet.efficientnet import (
    NormalizedModel,
)

__all__ = ["xcit_nano"]


def _load_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    """Load a PyTorch or Lightning checkpoint into an XCiT model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    target_keys = set(model.state_dict())
    if set(state_dict) != target_keys:
        stripped_state_dict = {
            key.removeprefix("model."): value for key, value in state_dict.items()
        }
        if set(stripped_state_dict) == target_keys:
            state_dict = stripped_state_dict
    model.load_state_dict(state_dict, strict=True)


def xcit_nano(
    num_classes: int = 72,
    input_channels: int = 1,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    pretrained: bool = False,
    checkpoint_path: str | Path | None = None,
    normalize: bool = False,
) -> nn.Module:
    """Construct an XCiT-Nano spectrogram classifier.

    Args:
        num_classes: Number of classifier output classes.
        input_channels: Number of spectrogram channels. One- and two-channel
            inputs are supported directly without RGB channel duplication.
        drop_path_rate: Stochastic-depth rate.
        drop_rate: Classifier dropout rate.
        pretrained: Whether to request compatible timm pretrained weights.
        checkpoint_path: Optional TorchSIG Models or Lightning checkpoint. A
            local checkpoint takes precedence over timm pretrained weights.
        normalize: Whether to standardize each input sample and channel.

    Returns:
        An XCiT-Nano classifier accepting ``[B, C, F, T]`` or single-channel
        ``[B, F, T]`` spectrogram batches.
    """
    model = timm.create_model(
        "xcit_nano_12_p16_224",
        pretrained=pretrained and checkpoint_path is None,
        num_classes=num_classes,
        in_chans=input_channels,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
    )
    wrapped_model = NormalizedModel(model, normalize=normalize)

    if checkpoint_path is not None:
        _load_checkpoint(wrapped_model, checkpoint_path)

    return wrapped_model
