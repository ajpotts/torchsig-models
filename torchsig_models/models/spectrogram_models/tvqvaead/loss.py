# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loss functions for the TimeVQVAE-AD model implementation.

This module implements the loss functions used to train the two stages of the
TimeVQVAE-AD model:

1. Stage 1 (VQ-VAE): Reconstruction loss + VQ commitment/regularization loss
2. Stage 2 (MaskGIT): Cross-entropy loss on masked token predictions

Source: TimeVQVAE-AD experiments/exp_stage1.py (lines 44, 67)
        TimeVQVAE-AD experiments/exp_stage2.py (lines 33-34)
        TimeVQVAE-AD models/stage2/maskgit.py (lines 127-131)

NOTE: The source repository does not contain a standalone loss module. All loss
computations are inlined in the experiment files. This module consolidates them
into ``nn.Module`` classes for consistency with the Anomalib framework.

Example:
    >>> import torch
    >>> from torchsig_models.models.spec_models.tvqvae_ad.loss import Stage1Loss
    >>> criterion = Stage1Loss()
    >>> x = torch.randn(2, 1, 64, 128)
    >>> x_rec = torch.randn(2, 1, 64, 128)
    >>> vq_loss_dict = {"loss": torch.tensor(0.1, requires_grad=True)}
    >>> loss = criterion(x, x_rec, vq_loss_dict)

    >>> from torchsig_models.models.spec_models.tvqvae_ad.loss import Stage2Loss
    >>> criterion = Stage2Loss()
    >>> logits = torch.randn(2, 32, 128)
    >>> targets = torch.randint(0, 128, (2, 32))
    >>> mask = torch.ones(2, 32, dtype=torch.bool)
    >>> mask[:, :16] = False
    >>> loss = criterion(logits, targets, mask)

See Also:
    :class:`torchsig_models.models.spec_models.tvqvae_ad.torch_model.TimeVQVAEAD`:
        PyTorch implementation of the TimeVQVAE-AD model architecture.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class Stage1Loss(nn.Module):
    """Loss function for Stage 1 (VQ-VAE) training of TimeVQVAE-AD.

    Combines MSE reconstruction loss with VQ commitment/regularization losses.
    The VQ losses (commitment loss, orthogonal regularization) are computed
    internally by ``VectorQuantize.forward()`` and passed in as a dict.

    Source: experiments/exp_stage1.py lines 44, 66-67

    Example:
        >>> criterion = Stage1Loss()
        >>> x = torch.randn(2, 1, 64, 128)
        >>> x_rec = torch.randn(2, 1, 64, 128)
        >>> vq_loss_dict = {"loss": torch.tensor(0.1, requires_grad=True)}
        >>> loss = criterion(x, x_rec, vq_loss_dict)
        >>> loss.shape
        torch.Size([])
    """

    @staticmethod
    def forward(
        x: Tensor,
        x_rec: Tensor,
        vq_loss_dict: dict[str, Tensor],
    ) -> Tensor:
        """Compute the Stage 1 loss.

        Args:
            x: Original input spectrogram (B, C, H, W).
            x_rec: Reconstructed spectrogram (B, C, H, W).
            vq_loss_dict: Loss dict returned by ``VectorQuantize.forward()``.
                Must contain key ``"loss"`` with the total VQ loss scalar.

        Returns:
            Tensor: Combined scalar loss (reconstruction + VQ).
        """
        floor = x.detach().flatten(2).median(-1).values[:, :, None, None]
        weight = 1 + 4 * ((x.detach()-floor).clamp(min=0)/0.25).clamp(max=1)
        recons_loss = (weight*(x-x_rec).square()).sum()/weight.sum()
        return recons_loss + vq_loss_dict["loss"]


class Stage2Loss(nn.Module):
    """Loss function for Stage 2 (MaskGIT) training of TimeVQVAE-AD.

    Computes cross-entropy loss on masked token positions only. During training,
    a subset of VQ token indices are replaced with a mask token. The transformer
    predicts the original token identities at masked positions.

    Source: models/stage2/maskgit.py lines 127-131,
            experiments/exp_stage2.py lines 33-34

    Example:
        >>> criterion = Stage2Loss()
        >>> logits = torch.randn(2, 32, 128)
        >>> targets = torch.randint(0, 128, (2, 32))
        >>> mask = torch.ones(2, 32, dtype=torch.bool)
        >>> mask[:, :16] = False
        >>> loss = criterion(logits, targets, mask)
        >>> loss.shape
        torch.Size([])
    """

    @staticmethod
    def forward(
        logits: Tensor,
        targets: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Compute the Stage 2 masked prediction loss.

        Args:
            logits: Transformer output logits (B, N, codebook_size).
            targets: Ground truth token indices (B, N).
            mask: Boolean tensor (B, N) where ``True`` = unmasked (known),
                ``False`` = masked (to predict). Loss is computed on masked
                positions (``~mask``).

        Returns:
            Tensor: Cross-entropy loss scalar over masked positions.
        """
        return F.cross_entropy(logits[~mask].float(), targets[~mask].long())
