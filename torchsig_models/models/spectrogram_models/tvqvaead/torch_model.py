# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model implementation for TimeVQVAE-AD.

This module implements the ``TimeVQVAEAD`` class that composes Stage 1
(VQ-VAE) and Stage 2 (MaskGIT) components into a single ``nn.Module``. It
supports two training phases and a mask-prediction-based anomaly scoring
inference path.

**Architecture overview:**

1. **Stage 1 (VQ-VAE)** – An encoder maps input spectrograms to continuous
   latent representations, a vector quantizer discretizes them into codebook
   indices, and a decoder reconstructs the original spectrogram. Trained with
   MSE reconstruction + VQ commitment loss.

2. **Stage 2 (MaskGIT)** – A bidirectional transformer learns to predict
   randomly masked VQ tokens. Only the transformer is trained; Stage 1
   components are frozen.

3. **Inference** – Anomaly scores are computed per latent position via
   *mask-prediction negative log-likelihood* (NLL). For each latent column, a
   local window of tokens is masked, the transformer predicts them, and the NLL
   of the true token under the predicted distribution serves as anomaly score.
   Higher NLL indicates less typical patterns → higher anomaly likelihood.

Source: TimeVQVAE-AD evaluate.py ``detect()`` / ``mask_prediction()``
        TimeVQVAE-AD experiments/exp_stage1.py, experiments/exp_stage2.py

Example:
    >>> import torch
    >>> from torchsig_models.models.spec_models.tvqvae_ad.torch_model import (
    ...     TimeVQVAEAD,
    ... )
    >>> model = TimeVQVAEAD(
    ...     in_channels=1,
    ...     dim=64,
    ...     downsample_rate=4,
    ...     n_resnet_blocks=4,
    ...     codebook_size=64,
    ...     frequency_independence=True,
    ...     input_size=(4, 64),
    ... )
    >>> x = torch.randn(2, 1, 4, 64)  # (B, C, freq_bins, time_steps)
    >>> # Stage 1 training
    >>> x_rec, vq_loss = model.forward_stage1(x)
    >>> # Stage 2 training (requires encoder to have seen data first)
    >>> logits, targets, mask = model.forward_stage2(x)

See Also:
    :class:`torchsig_models.models.spec_models.tvqvae_ad.loss.Stage1Loss`:
        Loss function for Stage 1 training.
    :class:`torchsig_models.models.spec_models.tvqvae_ad.loss.Stage2Loss`:
        Loss function for Stage 2 training.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

try:
    from anomalib.data import InferenceBatch
except ImportError:
    from collections import namedtuple
    InferenceBatch = namedtuple("InferenceBatch", ["pred_score", "anomaly_map"])

from .components.utils import freeze, quantize
from .stage1 import VectorQuantize, VQVAEDecoder, VQVAEEncoder
from .stage2 import BidirectionalTransformer, MaskGIT, gamma_func


class TimeVQVAEAD(nn.Module):
    """PyTorch model for TimeVQVAE-AD anomaly detection.

    Composes Stage 1 (VQ-VAE) and Stage 2 (MaskGIT) into a unified model.
    The ``lightning_model`` controls which training phase is active by
    freezing/unfreezing parameters.

    Source: TimeVQVAE-AD experiments/exp_stage1.py, experiments/exp_stage2.py,
            evaluation/__init__.py ``detect()`` and ``mask_prediction()``

    Args:
        in_channels: Number of input channels (default ``1`` for univariate).
        dim: Latent feature dimension for VQ-VAE encoder/decoder.
        downsample_rate: Temporal downsampling factor (power of 2).
        n_resnet_blocks: Number of ResNet blocks per encoder/decoder stage.
        codebook_size: Number of VQ codebook entries.
        frequency_independence: If True, process frequency bins independently
            in encoder/decoder convolutions.
        dropout: Dropout rate for encoder/decoder ResBlocks.
        choice_temperature: Temperature for mask-by-random-topk during
            MaskGIT iterative decoding.
        T: Number of iterative unmasking steps in MaskGIT inference.
        mask_scheduling_func: Mask scheduling mode for MaskGIT
            (``"cosine"``, ``"linear"``, ``"square"``, ``"cubic"``).
        prior_model_config: Keyword arguments for ``BidirectionalTransformer``
            (``hidden_dim``, ``depth``, ``heads``, ``attn_dim_head``,
            ``ff_mult``, ``use_rmsnorm``, ``dropout``).
        latent_window_size: Size of the local masking kernel for inference
            anomaly scoring. Must be odd. If ``None``, defaults to ``1``
            (single-column masking).
        hybrid_symmetric_blocks: Number of leading encoder/decoder blocks that
            use (2,2) stride to downsample/upsample both frequency and time axes.
            Set to 0 (default) for original temporal-only downsampling. For large
            spectrograms, set to 1-3 to reduce frequency dimension early.
        input_size: Expected input spatial dimensions (H, W). Used to
            analytically compute latent token count for transformer init.

    Example:
        >>> model = TimeVQVAEAD(input_size=(4, 64))
        >>> x = torch.randn(2, 1, 4, 64)
        >>> x_rec, vq_loss = model.forward_stage1(x)
        >>> x_rec.shape
        torch.Size([2, 1, 4, 64])
    """

    def __init__(
        self,
        in_channels: int = 1,
        dim: int = 64,
        downsample_rate: int = 4,
        n_resnet_blocks: int = 4,
        codebook_size: int = 64,
        frequency_independence: bool = True,
        dropout: float = 0.3,
        choice_temperature: float = 4.0,
        T: int = 20,
        mask_scheduling_func: str = "cosine",
        prior_model_config: dict[str, Any] | None = None,
        latent_window_size: int | None = None,
        hybrid_symmetric_blocks: int = 0,
        input_size: tuple[int, int] = (256, 256),
        score_mode: str = "hybrid",
        top_fraction: float = 0.02,        
    ) -> None:
        super().__init__()

        # Store config for serialization / reconstruction
        self.in_channels = in_channels
        self.dim = dim
        self.downsample_rate = downsample_rate
        self.codebook_size = codebook_size
        self.latent_window_size = latent_window_size if latent_window_size is not None else 1

        if self.latent_window_size < 1 or self.latent_window_size % 2 != 1:
            raise ValueError("latent_window_size must be positive and odd")
        if score_mode not in {"nll", "reconstruction", "hybrid"} or not 0 < top_fraction <= 1:
            raise ValueError("Invalid scoring configuration")
        self.score_mode, self.top_fraction = score_mode, top_fraction
        self.register_buffer("score_scale", torch.ones(2))
        self.register_buffer("score_threshold", torch.tensor(float("nan")))
        self.calibrating = False
        self.calibration_scores = []

        # ---- Stage 1: VQ-VAE components ----
        # Source: experiments/exp_stage1.py lines 30-34
        # num_channels = 2 * in_channels for complex spectrogram (real + imag); when dealing with spectrograms directly,
        # num_channels = in_channels
        num_channels = in_channels

        self.encoder = VQVAEEncoder(
            dim=dim,
            num_channels=num_channels,
            downsample_rate=downsample_rate,
            n_resnet_blocks=n_resnet_blocks,
            frequency_independence=frequency_independence,
            dropout=dropout,
            hybrid_symmetric_blocks=hybrid_symmetric_blocks,
        )
        self.vq_model = VectorQuantize(dim=dim, codebook_size=codebook_size, kmeans_init=True, threshold_ema_dead_code=1)
        self.decoder = VQVAEDecoder(
            dim=dim,
            num_channels=num_channels,
            downsample_rate=downsample_rate,
            n_resnet_blocks=n_resnet_blocks,
            frequency_independence=frequency_independence,
            dropout=dropout,
            hybrid_symmetric_blocks=hybrid_symmetric_blocks,
        )

        # Pre-compute latent dimensions so MaskGIT/transformer can be
        # initialised with the correct num_tokens before any forward pass.
        latent_h, latent_w = self.encoder.compute_latent_shape(*input_size)
        self.encoder.H_prime = torch.tensor(latent_h)
        self.encoder.W_prime = torch.tensor(latent_w)
        self.encoder.num_tokens = torch.tensor(latent_h * latent_w)
        self.encoder.is_num_tokens_updated = True

        # ---- Stage 2: MaskGIT (transformer + masking logic) ----
        # Source: experiments/exp_stage2.py line 20
        # MaskGIT takes references to the same encoder/decoder/vq_model.
        # It will freeze them internally; the lightning_model unfreezes
        # for Stage 1 training.
        _prior_config = prior_model_config if prior_model_config is not None else {
            "hidden_dim": 128,
            "depth": 4,
            "heads": 4,
            "attn_dim_head": 64,
            "ff_mult": 4,
            "use_rmsnorm": True,
            "dropout": 0.3,
        }

        self.maskgit = MaskGIT(
            encoder=self.encoder,
            decoder=self.decoder,
            vq_model=self.vq_model,
            codebook_size=codebook_size,
            embed_dim=dim,
            choice_temperature=choice_temperature,
            T=T,
            mask_scheduling_func=mask_scheduling_func,
            prior_model_config=_prior_config,
        )

        self.maskgit.mask_window_size = self.latent_window_size

    # ------------------------------------------------------------------
    # Stage 1 training
    # Source: experiments/exp_stage1.py lines 40-45
    # ------------------------------------------------------------------

    def forward_stage1(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Stage 1 (VQ-VAE) training forward pass.

        Encodes input, quantizes, and decodes. Returns the reconstruction
        and VQ loss dict for use with ``Stage1Loss``.

        Args:
            x: Input spectrogram (B, C, H, W).

        Returns:
            Tuple of (reconstructed spectrogram, VQ loss dict).
        """
        z = self.encoder(x)  # pyrefly: ignore
        z_q, _s, vq_loss, _perplexity = quantize(z, self.vq_model)
        x_rec = self.decoder(z_q)  # pyrefly: ignore
        return x_rec, vq_loss

    # ------------------------------------------------------------------
    # Stage 2 training
    # Source: experiments/exp_stage2.py lines 28-40
    # ------------------------------------------------------------------

    def forward_stage2(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Stage 2 (MaskGIT) training forward pass.

        Encodes input to VQ tokens (with frozen encoder/VQ), randomly
        masks tokens, predicts via transformer, and returns the raw
        components for ``Stage2Loss``.

        Args:
            x: Input spectrogram (B, C, H, W).

        Returns:
            Tuple of (logits, target_tokens, mask) where mask is True
            for unmasked (known) positions and False for masked positions.
        """
        device = x.device

        # Encode to tokens (frozen stage 1)
        # Frozen gradients do not freeze VQ EMA, BatchNorm or dropout
        self.encoder.eval()
        self.decoder.eval()
        self.vq_model.eval()
        with torch.no_grad():
            z_q, s = self.maskgit.encode_to_z_q(
                x, self.encoder, self.vq_model,
            )  # s: (B, N)

        # Random masking + transformer prediction
        s_m, mask = self.maskgit._randomly_mask_tokens(
            s, self.maskgit.mask_token_id, device,
        )
        logits = self.maskgit.transformer(s_m)  # pyrefly: ignore  # (B, N, codebook_size+1)

        return logits, s, mask

    # ------------------------------------------------------------------
    # Inference: mask-prediction anomaly scoring
    # Source: evaluation/__init__.py detect() lines 84-160,
    #         mask_prediction() lines 53-64
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]] | tuple[Tensor, Tensor, Tensor] | InferenceBatch:
        """Main forward dispatch.

        During training, the ``lightning_model`` should call
        ``forward_stage1`` or ``forward_stage2`` directly. This ``forward``
        defaults to Stage 2 during training and anomaly scoring during
        inference.

        Args:
            batch: Input spectrogram (B, C, H, W).

        Returns:
            Training: Stage 2 output ``(logits, targets, mask)``.
            Inference: ``InferenceBatch`` with ``pred_score`` and
                ``anomaly_map``.
        """
        if self.training:
            return self.forward_stage2(batch)
        return self._forward_inference(batch)

    @torch.inference_mode() ##@torch.no_grad()
    def _forward_inference(
        self,
        x: Tensor,
    ) -> InferenceBatch:
        """Compute per-position anomaly scores via mask-prediction NLL.

        Source: evaluation/__init__.py ``detect()`` lines 121-141,
                ``mask_prediction()`` lines 53-64

        For each latent column, masks a local window of tokens, predicts
        them with the transformer, and computes the negative log-likelihood
        of the true tokens. Higher NLL → more anomalous.

        Args:
            x: Input spectrogram (B, C, H, W).

        Returns:
            ``InferenceBatch`` with:
                - ``anomaly_map``: (B, H, W) anomaly heatmap interpolated
                  to input spatial size.
                - ``pred_score``: (B,) per-sample anomaly score (max of map).
        """
        batch_size = x.shape[0]
        input_h, input_w = x.shape[2], x.shape[3]

        # Encode to quantized tokens
        z_q, s = self.maskgit.encode_to_z_q(
            x, self.encoder, self.vq_model,
        )  # z_q: (B, D, H', W'), s: (B, N)
        latent_height = z_q.shape[2]
        latent_width = z_q.shape[3]

        # Compute NLL anomaly scores per latent position
        # Source: evaluation/__init__.py lines 126-139
        a_tilde = self._compute_nll_scores(
            s, latent_height, latent_width,
        )  # (B, H', W')

        # Interpolate anomaly map to input spatial size
        # Source: evaluation/__init__.py line 140
        anomaly_map = F.interpolate(
            a_tilde.unsqueeze(1),  # (B, 1, H', W')
            size=(input_h, input_w),
            mode="nearest",
        ).squeeze(1)  # (B, H, W)

        # Per-sample anomaly score
        residual = (x - self.decoder(z_q)).square().mean(1)
        residual = F.avg_pool2d(residual[:, None], 3, stride=1, padding=1)[:, 0]
        components = torch.stack([self._pool(anomaly_map), self._pool(residual)], dim=1)
        if self.calibrating:
            self.calibration_scores.append(components.detach().cpu())
        scaled = components / self.score_scale.clamp_min(1e-8)
        if self.score_mode == "nll":
            pred_score = scaled[:, 0]
            anomaly_map = anomaly_map / self.score_scale[0]
        elif self.score_mode == "reconstruction":
            pred_score = scaled[:, 1]
            anomaly_map = residual / self.score_scale[1]
        else:
            pred_score = scaled.mean(1)
            anomaly_map = (anomaly_map / self.score_scale[0] + residual / self.score_scale[1]) / 2

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def _pool(self, values):
        import math
        flat = values.flatten(1)
        return flat.topk(max(1, math.ceil(flat.shape[1] * self.top_fraction)), dim=1).values.mean(1)

    def finish_calibration(self, false_positive_rate=0.05):
        """Use held-out NORMAL samples only; no test fitting."""
        if not self.calibration_scores or not 0 < false_positive_rate < 1:
            raise ValueError("Need calibration samples and FPR in (0,1)")
        c = torch.cat(self.calibration_scores)
        if len(c) < 20 or not torch.isfinite(c).all():
            raise ValueError("Need >=20 finite normal calibration scores")
        scale = c.median(0).values.clamp_min(1e-8)
        self.score_scale.copy_(scale.to(self.score_scale.device))
        normalized = c / scale
        score = normalized.mean(1) if self.score_mode == "hybrid" else normalized[:, 0 if self.score_mode == "nll" else 1]
        threshold = torch.quantile(score, 1-false_positive_rate, interpolation="higher")
        self.score_threshold.copy_(threshold.to(self.score_threshold.device))
        self.calibration_scores.clear()
        self.calibrating = False
        return {"normal_samples": len(c), "scale": scale.tolist(), "threshold": threshold.item(),
                "target_calibration_fpr": false_positive_rate, "score_mode": self.score_mode}

    def _compute_nll_scores(
        self,
        s: Tensor,
        latent_height: int,
        latent_width: int,
    ) -> Tensor:
        """Compute NLL anomaly scores for all latent positions.

        Source: evaluation/__init__.py ``detect()`` lines 126-139,
                ``mask_prediction()`` lines 53-64

        For each latent column ``w``, masks a local window of columns
        around ``w``, runs the transformer to get predicted logits, then
        computes the NLL of the true token at each position within the
        window. The NLL is averaged across the window width.

        Args:
            s: Token indices (B, N) where N = H' * W'.
            latent_height: Height of latent grid.
            latent_width: Width of latent grid.

        Returns:
            NLL scores tensor (B, H', W').
        """
        batch_size = s.shape[0]
        device = s.device
        lws = self.latent_window_size

        a_tilde = torch.zeros(
            batch_size, latent_height, latent_width, device=device,
        )

        s_2d = rearrange(s, "b (h w) -> b h w", h=latent_height)

        for w in range(latent_width):
            # Compute kernel range around column w
            # Source: evaluation/__init__.py line 128
            if lws == 1:
                kernel_rng = slice(w, w + 1)
            else:
                half = (lws - 1) // 2
                kernel_rng = slice(max(0, w - half), w + half + 1)

            # Mask-predict
            logits_prob = self._mask_prediction(
                s, latent_height, kernel_rng,
            )  # (B, H', W', K)

            # Gather probability of true tokens in kernel range
            # Source: evaluation/__init__.py lines 134-137
            #s_2d = rearrange(s, "b (h w) -> b h w", h=latent_height)
            p = torch.gather(
                logits_prob, -1, s_2d.unsqueeze(-1),
            )  # (B, H', W', 1)
            p = p[:, :, kernel_rng, 0]  # (B, H', r)

            # NLL averaged across kernel width
            a_w = -1 * torch.log(p + 1e-30).mean(dim=-1)  # (B, H')
            a_tilde[:, :, w] = a_w

        return a_tilde

    def _mask_prediction(
        self,
        s: Tensor,
        height: int,
        slice_rng: slice,
    ) -> Tensor:
        """Mask a slice of latent columns and predict with transformer.

        Source: evaluation/__init__.py ``mask_prediction()`` lines 53-64

        Args:
            s: Token indices (B, N) where N = H' * W'.
            height: Latent grid height H'.
            slice_rng: Column range to mask.

        Returns:
            Softmax probabilities (B, H', W', K) over codebook.
        """
        # Mask specified columns
        # Source: lines 55-57
        s_m = s.detach().clone()  #s_m = copy.deepcopy(s)  # (B, N)
        s_m = rearrange(s_m, "b (h w) -> b h w", h=height)  # (B, H', W')
        s_m[:, :, slice_rng] = self.maskgit.mask_token_id

        # Predict
        # Source: lines 60-62
        logits = self.maskgit.transformer(  # pyrefly: ignore
            rearrange(s_m, "b h w -> b (h w)"),
        )  # (B, N, K)
        logits = rearrange(
            logits, "b (h w) k -> b h w k", h=height,
        )  # (B, H', W', K)
        logits_prob = F.softmax(logits, dim=-1)  # (B, H', W', K)

        return logits_prob
