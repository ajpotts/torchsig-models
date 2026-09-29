"""Normalization utilities shared by TorchSig models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal
import warnings

import torch
from torch import nn


NormalizationMode = Literal["dataset", "sample", "none"]


def resolve_normalization_mode(
    normalization: NormalizationMode | None,
    normalize: bool | None,
) -> NormalizationMode:
    """Resolve the normalization mode and deprecated boolean alias."""
    if normalize is not None:
        if normalization is not None:
            raise ValueError("Pass either normalization or normalize, not both.")
        warnings.warn(
            "The normalize argument is deprecated; use normalization instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return "sample" if normalize else "none"
    return "dataset" if normalization is None else normalization


class DatasetNormalization(nn.Module):
    """Standardize channels using statistics from a training dataset."""

    def __init__(
        self,
        mean: torch.Tensor | list[float],
        std: torch.Tensor | list[float],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std_tensor = torch.as_tensor(std, dtype=torch.float32).flatten()

        if mean_tensor.numel() == 0 or mean_tensor.shape != std_tensor.shape:
            raise ValueError("Normalization mean and std must have matching channels.")
        if (
            not torch.isfinite(mean_tensor).all()
            or not torch.isfinite(std_tensor).all()
        ):
            raise ValueError("Normalization statistics must be finite.")
        if eps <= 0:
            raise ValueError("Normalization epsilon must be positive.")
        if torch.any(std_tensor <= eps):
            raise ValueError("Normalization standard deviations must exceed epsilon.")

        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise standardization to a batched tensor."""
        if x.ndim < 3:
            raise ValueError(
                "Expected a batched tensor shaped [batch, channels, ...], "
                f"got {tuple(x.shape)}."
            )
        if x.shape[1] != self.mean.numel():
            raise ValueError(
                f"Expected {self.mean.numel()} input channels, got {x.shape[1]}."
            )

        shape = (1, self.mean.numel(), *((1,) * (x.ndim - 2)))
        mean = self.mean.to(dtype=x.dtype).view(shape)
        std = self.std.to(dtype=x.dtype).view(shape)
        eps = self.eps.to(dtype=x.dtype)
        return (x - mean) / (std + eps)


def compute_dataset_channel_stats(
    data: Iterable[object],
    *,
    add_channel_dim: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute channel-wise population statistics over batched model inputs.

    Each item may be an input tensor or a sequence whose first item is the
    input tensor. Accumulation uses the parallel-variance form of Welford's
    algorithm in float64 for numerical stability.
    """
    count = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    for item in data:
        x = item[0] if isinstance(item, (tuple, list)) else item
        if not isinstance(x, torch.Tensor) or x.ndim < 3:
            raise ValueError("Expected batches shaped [batch, channels, ...].")
        if add_channel_dim and x.ndim == 3:
            x = x.unsqueeze(1)

        channels = x.shape[1]
        values = x.detach().to(device="cpu", dtype=torch.float64)
        values = values.transpose(0, 1).reshape(channels, -1)
        batch_count = values.shape[1]
        if batch_count == 0:
            continue

        batch_mean = values.mean(dim=1)
        batch_m2 = ((values - batch_mean[:, None]) ** 2).sum(dim=1)

        if mean is None:
            mean = batch_mean
            m2 = batch_m2
            count = batch_count
            continue
        if channels != mean.numel():
            raise ValueError("All batches must have the same number of channels.")

        new_count = count + batch_count
        delta = batch_mean - mean
        mean = mean + delta * (batch_count / new_count)
        m2 = m2 + batch_m2 + delta.square() * count * batch_count / new_count
        count = new_count

    if mean is None or m2 is None or count == 0:
        raise ValueError("Cannot compute normalization statistics from empty data.")

    return mean.float(), torch.sqrt(m2 / count).float()


def normalization_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    normalization: NormalizationMode | None,
    *,
    legacy_mode: NormalizationMode,
    eps: float = 1e-6,
) -> dict[str, object]:
    """Build model normalization arguments from checkpoint state."""
    means = [
        value for key, value in state_dict.items() if key.endswith("normalize.mean")
    ]
    stds = [value for key, value in state_dict.items() if key.endswith("normalize.std")]
    checkpoint_eps = [
        value for key, value in state_dict.items() if key.endswith("normalize.eps")
    ]

    if len(means) == 1 and len(stds) == 1:
        if normalization not in (None, "dataset"):
            raise ValueError(
                "The checkpoint contains dataset normalization statistics and "
                f"cannot be loaded with normalization={normalization!r}."
            )
        return {
            "normalization": "dataset",
            "normalization_mean": means[0],
            "normalization_std": stds[0],
            "normalization_eps": (
                float(checkpoint_eps[0].item()) if len(checkpoint_eps) == 1 else eps
            ),
        }

    if means or stds:
        raise ValueError("Checkpoint contains incomplete normalization statistics.")
    if normalization == "dataset":
        raise ValueError(
            "Dataset normalization was requested, but the checkpoint does not "
            "contain training statistics."
        )

    selected_mode = legacy_mode if normalization is None else normalization
    if normalization is None:
        warnings.warn(
            f"Checkpoint has no dataset statistics; using legacy {legacy_mode!r} "
            "normalization.",
            UserWarning,
            stacklevel=2,
        )
    return {"normalization": selected_mode, "normalization_eps": eps}


def resolve_checkpoint_normalization_mode(
    checkpoint: object,
    normalization: NormalizationMode | None,
) -> NormalizationMode | None:
    """Resolve a normalization override against saved checkpoint metadata."""
    saved_mode = None
    if isinstance(checkpoint, dict):
        hyperparameters = checkpoint.get("hyper_parameters", {})
        if isinstance(hyperparameters, dict):
            metadata = hyperparameters.get("normalization", {})
            if isinstance(metadata, dict):
                saved_mode = metadata.get("mode")

    if saved_mode not in (None, "dataset", "sample", "none"):
        raise ValueError(f"Unsupported checkpoint normalization mode: {saved_mode!r}")
    if (
        normalization is not None
        and saved_mode is not None
        and normalization != saved_mode
    ):
        raise ValueError(
            f"normalization={normalization!r} conflicts with checkpoint mode "
            f"{saved_mode!r}."
        )
    return normalization if normalization is not None else saved_mode
