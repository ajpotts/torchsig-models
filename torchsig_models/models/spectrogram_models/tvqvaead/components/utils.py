# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utility functions and layers for TimeVQVAE-AD.

Source: TimeVQVAE-AD utils/__init__.py (lines 1-434)

This module contains utility functions ported from the original TimeVQVAE-AD
implementation. Functions specific to time-series processing (STFT, time-frequency
conversion) or UCR dataset handling are omitted as this implementation targets
image data with Anomalib.

Omitted functions (not applicable to image-based anomaly detection):
- get_root_dir() - lines 22-23: Path handling for original repo
- load_yaml_param_settings() - lines 28-34: Anomalib has own config system
- preprocess() - lines 37-58: Time series preprocessing
- minibatch_slices_iterator() - lines 61-83: Time series batching
- BatchSlidingWindow - lines 86-180: Time series windowing
- zero_pad_high_freq() - lines 299-305: Time-frequency processing
- zero_pad_low_freq() - lines 308-322: Time-frequency processing
- compute_downsample_rate() - lines 346-349: Time-frequency specific
- set_window_size() - lines 372-376: UCR dataset specific
- compute_receptive_field() - lines 379-389: Analysis utility
"""

from __future__ import annotations

import numpy as np
import torch
# import torch.jit as jit
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR


# =============================================================================
# Model Freezing Utilities
# Source: utils/__init__.py lines 183-190
# =============================================================================


def freeze(model: nn.Module) -> None:
    """Freeze all parameters in a module.

    Source: utils/__init__.py lines 183-185

    Args:
        model: PyTorch module to freeze.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model: nn.Module) -> None:
    """Unfreeze all parameters in a module.

    Source: utils/__init__.py lines 188-190

    Args:
        model: PyTorch module to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True


# =============================================================================
# Vector Quantization Utility
# Source: utils/__init__.py lines 281-296
# =============================================================================


def quantize(
    z: Tensor,
    vq_model: nn.Module,
    transpose_channel_length_axes: bool = False,
    svq_temp: float | None = None,
) -> tuple[Tensor, Tensor, dict, Tensor]:
    """Quantize latent representations using VQ model.

    Source: utils/__init__.py lines 281-296

    Handles both 1D (time series) and 2D (image) inputs by reshaping
    to (B, N, C) format expected by VQ model, then reshaping back.

    Args:
        z: Latent tensor. Shape (B, C, H, W) for 2D or (B, C, L) for 1D.
        vq_model: Vector quantization module.
        transpose_channel_length_axes: For 1D input, whether to transpose.
        svq_temp: Stochastic VQ temperature for sampling.

    Returns:
        Tuple of (z_q, indices, vq_loss, perplexity).
    """
    # Source: lines 282-283
    input_dim = len(z.shape) - 2

    if input_dim == 2:
        # Source: lines 283-287
        h, w = z.shape[2:]
        z = z.contiguous().permute(0, 2, 3, 1).reshape(z.shape[0], -1, z.shape[1]) # TODO: note that .contiguous() is necessary here
        z_q, indices, vq_loss, perplexity = vq_model(z, svq_temp=svq_temp)
        z_q = z_q.contiguous().reshape(z_q.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous() # TODO: note that .contiguous() is necessary here
    elif input_dim == 1:
        # Source: lines 288-293
        if transpose_channel_length_axes:
            z = z.contiguous().permute(0, 2, 1).reshape(z.shape[0], -1, z.shape[1])
        z_q, indices, vq_loss, perplexity = vq_model(z, svq_temp=svq_temp)
        if transpose_channel_length_axes:
            z_q = z_q.contiguous().permute(0, 2, 1)
    else:
        # Source: lines 294-295
        raise ValueError(f"Input must be 1D or 2D, got {input_dim}D")

    # Source: line 296
    return z_q, indices, vq_loss, perplexity


# =============================================================================
# Embedding Loss for Codebook Training
# Source: utils/__init__.py lines 324-343
# =============================================================================


def compute_emb_loss(
    codebook: nn.Module,
    x: Tensor,
    use_cosine_sim: bool,
    esm_max_codes: int,
) -> Tensor:
    """Compute embedding space matching loss between input and codebook.

    Source: utils/__init__.py lines 324-343

    Encourages the codebook distribution to match the encoder output
    distribution by minimizing MSE between covariances and means.

    Args:
        codebook: Codebook module with .embed attribute.
        x: Input tensor to match against codebook.
        use_cosine_sim: Whether to L2-normalize before computing stats.
        esm_max_codes: Max number of codebook entries to sample.

    Returns:
        ESM loss tensor.
    """
    # Source: lines 325-326
    embed: Tensor = codebook.embed  # type: ignore[attr-defined]
    flatten = x.reshape(-1, x.shape[-1])

    # Source: lines 328-330
    if use_cosine_sim:
        flatten = F.normalize(flatten, p=2, dim=-1)
        embed = F.normalize(embed, p=2, dim=-1)

    # Source: lines 332-334
    # N samples can be sampled from embed for memory efficiency
    num_codes: int = embed.shape[0]
    ind = torch.randint(0, num_codes, size=(min(esm_max_codes, num_codes),))
    embed = embed[ind]

    # Source: lines 336-340
    cov_embed = torch.cov(embed.t())  # (D, D)
    cov_x = torch.cov(flatten.t())  # (D, D)

    mean_embed = torch.mean(embed, dim=0)
    mean_x = torch.mean(flatten, dim=0)

    # Source: lines 342-343
    esm_loss = F.mse_loss(cov_x.detach(), cov_embed) + F.mse_loss(
        mean_x.detach(), mean_embed
    )
    return esm_loss


# =============================================================================
# Learning Rate Scheduler
# Source: utils/__init__.py lines 392-411
# =============================================================================


def linear_warmup_cosine_annealingLR(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    linear_warmup_rate: float = 0.1,
    min_lr: float = 1e-6,
) -> SequentialLR:
    """Create LR scheduler with linear warmup followed by cosine annealing.

    Source: utils/__init__.py lines 392-411

    Args:
        optimizer: PyTorch optimizer.
        max_steps: Total number of training steps.
        linear_warmup_rate: Fraction of steps for linear warmup (0 < rate < 1).
        min_lr: Minimum learning rate at end of cosine annealing.

    Returns:
        SequentialLR scheduler combining warmup and cosine annealing.
    """
    # Source: lines 393-395
    assert 0.0 < linear_warmup_rate < 1.0, "0 < linear_warmup_rate < 1."
    warmup_steps = int(max_steps * linear_warmup_rate)  # n% of max_steps

    # Source: lines 397-401
    def warmup_lambda(current_step: int) -> float:
        if current_step >= warmup_steps:
            return 1.0
        return float(current_step) / float(max(1, warmup_steps))

    # Source: lines 403-404
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Source: lines 406-407
    cosine_scheduler = CosineAnnealingLR(
        optimizer, max_steps - warmup_steps, eta_min=min_lr
    )

    # Source: lines 409-411
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return scheduler


# =============================================================================
# Snake Activation Function
# Source: utils/__init__.py lines 414-434
# =============================================================================


class SnakeActivation(nn.Module):
    """Snake activation: x + (1/a) * sin^2(a*x).

    Source: utils/__init__.py lines 414-434

    This version allows multiple values of `a` for different channels/num_features.
    Periodic activation that helps capture frequency patterns in data.

    Reference: "Neural networks fail to learn periodic functions and how to fix it"
               Ziyin et al., NeurIPS 2020

    Args:
        num_features: Number of input channels.
        dim: Input dimensionality (1 for time series, 2 for images).
        a_base: Lower bound for frequency parameter initialization.
        learnable: Whether frequency parameter is learnable.
        a_max: Upper bound for frequency parameter initialization.
    """

    def __init__(
        self,
        num_features: int,
        dim: int,
        a_base: float = 0.2,
        learnable: bool = True,
        a_max: float = 0.5,
    ) -> None:
        # Source: utils/__init__.py lines 418-430
        super().__init__()
        assert dim in [1, 2], "`dim` supports 1D and 2D inputs."

        if learnable:
            if dim == 1:  # (b d l); like time series
                # Source: line 424
                a = np.random.uniform(a_base, a_max, size=(1, num_features, 1))  # (1 d 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            elif dim == 2:  # (b d h w); like 2d images
                # Source: lines 426-428
                a = np.random.uniform(
                    a_base, a_max, size=(1, num_features, 1, 1)
                )  # (1 d 1 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        else:
            # Source: line 430
            self.register_buffer("a", torch.tensor(a_base, dtype=torch.float32))

    # @jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        """Apply Snake activation.

        Source: utils/__init__.py lines 432-434
        """
        return x + (1 / self.a) * torch.sin(self.a * x) ** 2