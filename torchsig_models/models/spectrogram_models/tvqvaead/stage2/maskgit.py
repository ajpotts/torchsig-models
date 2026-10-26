# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""MaskGIT for TimeVQVAE-AD.

Source: TimeVQVAE-AD models/stage2/maskgit.py (lines 1-265)
Reference: https://github.com/dome272/MaskGIT-pytorch

ADAPTATION NOTES (Pre-computed Spectrogram Input):
This implementation expects pre-computed spectrogram input (B, C, H, W).

Removed from source:
- ExpStage1.load_from_checkpoint (line 79-82) - stage1 components passed directly
- config dict initialization - parameters passed explicitly
- n_fft parameter (line 74)
- dataset_idx, input_length parameters (lines 64-65)
- time_to_timefreq / timefreq_to_time (not used directly in this file)
- torchtune RotaryPositionalEmbeddings import (unused in source)
- Unused imports: os, copy, Path, tempfile, deque, Rearrange

Preserved from source:
- All masking logic (_randomly_mask_tokens, mask_by_random_topk)
- Parallel unmasking (_parallel_unmasking)
- Iterative decoding and explainable sampling
- cov_loss_fn, var_loss_fn, gamma_func
- decode_token_ind_to_timeseries renamed to decode_token_ind
  (outputs spectrogram directly, no timefreq_to_time conversion)
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor

from ..components.utils import freeze, quantize
from ..stage1.vq import VectorQuantize
from ..stage1.vq_vae_encdec import VQVAEDecoder, VQVAEEncoder
from .bidirectional_transformer import BidirectionalTransformer


# =============================================================================
# Loss Functions
# Source: models/stage2/maskgit.py lines 26-44
# =============================================================================


def cov_loss_fn(codebook: Tensor) -> Tensor:
    """Covariance loss for codebook diversity.

    Source: models/stage2/maskgit.py lines 26-35

    Args:
        codebook: Codebook embeddings (K, dim).

    Returns:
        Covariance loss scalar.
    """
    # Source: lines 30-35
    norm_cb = codebook - codebook.mean(dim=1, keepdim=True)  # (K, D)
    norm_cb = F.normalize(norm_cb, p=2, dim=1)  # (K, D)
    corr_mat = torch.mm(norm_cb, norm_cb.T)  # (K, K)
    corr_mat.fill_diagonal_(0.0)
    cov_loss = (corr_mat**2).mean()
    return cov_loss


def var_loss_fn(codebook: Tensor, min_scale: float = 1.0) -> Tensor:
    """Variance loss for codebook utilization.

    Source: models/stage2/maskgit.py lines 38-44

    Args:
        codebook: Codebook embeddings (K, dim).
        min_scale: Minimum standard deviation target.

    Returns:
        Variance loss scalar.
    """
    # Source: lines 42-44
    std_cb = torch.sqrt(codebook.var(dim=1) + 1e-4)
    var_loss = torch.mean(F.relu(min_scale - std_cb))
    return var_loss


# =============================================================================
# Mask Scheduling
# Source: models/stage2/maskgit.py lines 46-56
# =============================================================================

def _gamma_linear(r: Any) -> Any:
    return 1 - r

def _gamma_cosine(r: Any) -> Any:
    return np.cos(r * np.pi / 2)

def _gamma_square(r: Any) -> Any:
    return 1 - r**2

def _gamma_cubic(r: Any) -> Any:
    return 1 - r**3

_GAMMA = {
    "linear": _gamma_linear,
    "cosine": _gamma_cosine,
    "square": _gamma_square,
    "cubic": _gamma_cubic,
}

def gamma_func(mode: str = "cosine") -> Callable[[Any], Any]:
    """Mask scheduling function for MaskGIT.
    
    Note: changed from problematic lambda functions that are 
    not logged properly in mlflow to named functions in a dict lookup. 

    Source: models/stage2/maskgit.py lines 46-56

    Args:
        mode: Scheduling mode ("linear", "cosine", "square", "cubic").

    Returns:
        Scheduling function mapping ratio -> mask proportion.
    """    
    try:
        return _GAMMA[mode]
    except KeyError as e:
        raise NotImplementedError(f"Unknown gamma mode: {mode}") from e


# =============================================================================
# MaskGIT
# Source: models/stage2/maskgit.py lines 58-265
# =============================================================================


class MaskGIT(nn.Module):
    """MaskGIT for masked generative modeling on VQ token sequences.

    Source: models/stage2/maskgit.py lines 58-265
    Reference: https://github.com/dome272/MaskGIT-pytorch

    Adaptation: Stage1 components (encoder, decoder, vq_model) are passed
    directly instead of being loaded from a checkpoint. Parameters are
    passed explicitly instead of via config dict.

    Args:
        encoder: Pre-trained VQ-VAE encoder.
        decoder: Pre-trained VQ-VAE decoder.
        vq_model: Pre-trained vector quantizer.
        codebook_size: VQ codebook size.
        embed_dim: Token embedding dimension for transformer.
        choice_temperature: Temperature for mask-by-random-topk during inference.
        T: Number of iterative decoding steps.
        mask_scheduling_func: Mask scheduling mode ("cosine", "linear", etc.).
        prior_model_config: Dict of kwargs for BidirectionalTransformer
            (hidden_dim, depth, heads, attn_dim_head, ff_mult, use_rmsnorm, dropout).
    """

    def __init__(
        self,
        encoder: VQVAEEncoder,
        decoder: VQVAEDecoder,
        vq_model: VectorQuantize,
        codebook_size: int,
        embed_dim: int,
        choice_temperature: float,
        T: int,
        mask_scheduling_func: str = "cosine",
        prior_model_config: dict[str, Any] | None = None,
    ) -> None:
        # Source: lines 63-98
        super().__init__()
        self.choice_temperature = choice_temperature
        self.T = T
        self.mask_token_id = codebook_size
        self.gamma = gamma_func(mask_scheduling_func)

        # Source: lines 79-88 - Stage1 components (frozen)
        # Adaptation: passed directly instead of ExpStage1.load_from_checkpoint
        self.encoder = encoder
        self.decoder = decoder
        self.vq_model = vq_model
        freeze(self.encoder)
        freeze(self.decoder)
        freeze(self.vq_model)
        self.encoder.eval()
        self.decoder.eval()
        self.vq_model.eval()

        # Source: lines 90-91
        self.num_tokens: int = int(self.encoder.num_tokens.item())
        self.H_prime = self.encoder.H_prime
        self.W_prime = self.encoder.W_prime

        # Source: lines 93-98 - Transformer
        _prior_config = prior_model_config if prior_model_config is not None else {}
        self.transformer = BidirectionalTransformer(
            embed_dim=embed_dim,
            num_tokens=self.num_tokens,
            codebook_size=codebook_size,
            **_prior_config,  # pyrefly: ignore
        )

    @torch.no_grad()
    def encode_to_z_q(
        self,
        x: Tensor,
        encoder: VQVAEEncoder,
        vq_model: VectorQuantize,
        svq_temp: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode input to quantized latent tokens.

        Source: models/stage2/maskgit.py lines 100-107

        Args:
            x: Input spectrogram (B, C, H, W).
            encoder: VQ-VAE encoder.
            vq_model: Vector quantizer.
            svq_temp: Stochastic VQ temperature.

        Returns:
            (quantized_latent, token_indices).
        """
        # Source: lines 105-107
        z = encoder(x)
        zq, s, _, _ = quantize(z, vq_model, svq_temp=svq_temp)  # (B,C,H,W), (B,N), ...
        return zq, s

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: random masking + cross-entropy loss on masked tokens.

        Source: models/stage2/maskgit.py lines 109-131

        Args:
            x: Input spectrogram (B, C, H, W).

        Returns:
            Mask prediction loss (scalar).
        """
        # Source: lines 114-120
        device = x.device
        self.encoder.eval()
        self.decoder.eval()
        self.vq_model.eval()

        with torch.no_grad():
            zq, s = self.encode_to_z_q(x, self.encoder, self.vq_model)  # (B, N)

        # Source: lines 122-131 - Masked prediction
        s_M, mask = self._randomly_mask_tokens(s, self.mask_token_id, device)
        logits = self.transformer(s_M)  # (B, N, codebook_size)

        # Loss on masked positions only
        logits_on_mask = logits[~mask]
        s_on_mask = s[~mask]
        mask_pred_loss = F.cross_entropy(logits_on_mask.float(), s_on_mask.long())

        return mask_pred_loss

    def _randomly_mask_tokens(
        self,
        s: Tensor,
        mask_token_id: int,
        device: torch.device | str,
    ) -> tuple[Tensor, Tensor]:
        """Randomly mask tokens with gamma scheduling.

        Source: models/stage2/maskgit.py lines 133-153

        Args:
            s: Token indices (B, N).
            mask_token_id: Index used for mask tokens.
            device: Torch device.

        Returns:
            (masked_tokens, unmask_boolean_mask) where mask=True means unmasked.
        """
        # Source: lines 137-153
        b, n = s.shape

        # Include the same structured masks used by detection.
        if torch.rand((), device=device) < 0.5:
            h, w = int(self.H_prime), int(self.W_prime)
            half = (getattr(self, "mask_window_size", 1)-1)//2
            centers = torch.randint(w, (b, 1, 1), device=device)
            cols = torch.arange(w, device=device)[None, None, :]
            hidden = ((cols-centers).abs() <= half).expand(b,h,w).reshape(b,n)
            return s.masked_fill(hidden, mask_token_id), ~hidden

        # Sample masking ratios per batch element
        ratio = np.random.uniform(0, 1, (b,))  # (B,)
        n_unmasks = np.floor(self.gamma(ratio) * n)  # (B,)
        # Ensures at least one masked token
        n_unmasks = np.clip(n_unmasks, a_min=0, a_max=n - 1).astype(int)  # (B,)

        rand = torch.rand((b, n), device=device)  # (B, N)
        mask = torch.zeros((b, n), dtype=torch.bool, device=device)  # (B, N)

        for i in range(b):
            ind = rand[i].topk(n_unmasks[i], dim=-1).indices
            mask[i].scatter_(dim=-1, index=ind, value=True)

        # Mask the token set
        masked_indices = mask_token_id * torch.ones((b, n), device=device)  # (B, N)
        s_M = mask * s + (~mask) * masked_indices  # (B, N)
        return s_M.long(), mask

    def _create_input_tokens_normal(
        self,
        num: int,
        num_tokens: int,
        mask_token_id: int,
        device: torch.device | str,
    ) -> Tensor:
        """Create fully masked token sequence for inference.

        Source: models/stage2/maskgit.py lines 155-161

        Args:
            num: Batch size.
            num_tokens: Sequence length.
            mask_token_id: Mask token index.
            device: Torch device.

        Returns:
            All-mask token tensor (num, num_tokens).
        """
        # Source: lines 159-161
        blank_tokens = torch.ones((num, num_tokens), device=device)
        masked_tokens = mask_token_id * blank_tokens
        return masked_tokens.to(torch.int64)

    def mask_by_random_topk(
        self,
        mask_len: Tensor,
        probs: Tensor,
        temperature: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Mask tokens by lowest confidence with Gumbel noise.

        Source: models/stage2/maskgit.py lines 163-187

        Args:
            mask_len: Number of tokens to mask (B, 1).
            probs: Confidence scores (B, N).
            temperature: Gumbel noise temperature.
            device: Torch device.

        Returns:
            Boolean mask where True = masked (B, N).
        """

        # Source: lines 170-178 - Local helpers
        def _log(t: Tensor, eps: float = 1e-20) -> Tensor:
            return torch.log(t.clamp(min=eps))

        def _gumbel_noise(t: Tensor) -> Tensor:
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -_log(-_log(noise))

        # Source: lines 180-187
        confidence = torch.log(probs + 1e-5) + temperature * _gumbel_noise(probs).to(
            device
        )  # (B, N)
        mask_len_unique = int(mask_len.unique().item())
        masking_ind = torch.topk(
            confidence, k=mask_len_unique, dim=-1, largest=False
        ).indices  # (B, K)
        masking = torch.zeros_like(confidence).to(device)  # (B, N)
        for i in range(masking_ind.shape[0]):
            masking[i, masking_ind[i].long()] = 1.0
        masking = masking.bool()
        return masking

    def _parallel_unmasking(
        self,
        s: Tensor,
        unknown_number_in_the_beginning: Tensor,
        device: torch.device | str,
        init_t: int = 0,
    ) -> Tensor:
        """Iterative parallel unmasking for MaskGIT decoding.

        Source: models/stage2/maskgit.py lines 189-219

        Args:
            s: Token indices with masks (B, N).
            unknown_number_in_the_beginning: Count of masked tokens per sample (B,).
            device: Torch device.
            init_t: Starting iteration (for explainable sampling).

        Returns:
            Decoded token indices (B, N).
        """
        # Source: lines 196-218
        for t in range(init_t, self.T):
            logits = self.transformer(s)  # (B, N, K)
            sampled_ids = torch.distributions.categorical.Categorical(
                logits=logits
            ).sample()  # (B, N)
            unknown_map = s == self.mask_token_id  # (B, N)
            sampled_ids = torch.where(unknown_map, sampled_ids, s)  # (B, N)

            # Create masking according to t
            ratio = 1.0 * (t + 1) / self.T
            mask_ratio = self.gamma(ratio)

            probs = F.softmax(logits, dim=-1)  # (B, N, K)
            selected_probs = torch.gather(
                probs, dim=-1, index=sampled_ids[:, :, None]
            )[:, :, 0]  # (B, N)
            _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
            selected_probs = torch.where(
                unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS
            )  # (B, N)

            mask_len = torch.unsqueeze(
                torch.floor(unknown_number_in_the_beginning * mask_ratio), 1
            )  # (B, 1)
            mask_len = torch.clip(mask_len, min=0.0)

            # Adds noise for randomness
            masking = self.mask_by_random_topk(
                mask_len,
                selected_probs,
                temperature=self.choice_temperature * (1.0 - ratio),
                device=device,
            )

            # Masks tokens with lower confidence
            s = torch.where(masking, self.mask_token_id, sampled_ids)  # (B, N)

        return s

    @torch.no_grad()
    def iterative_decoding(
        self,
        num: int = 1,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Iterative decoding: sample token indices from scratch.

        Source: models/stage2/maskgit.py lines 221-233

        Args:
            num: Number of samples to generate.
            device: Torch device.

        Returns:
            Sampled token indices (num, num_tokens).
        """
        # Source: lines 228-233
        s = self._create_input_tokens_normal(
            num, self.num_tokens, self.mask_token_id, device
        )  # (B, N)

        unknown_number_in_the_beginning = torch.sum(
            s == self.mask_token_id, dim=-1
        )  # (B,)

        s = self._parallel_unmasking(s, unknown_number_in_the_beginning, device)
        return s

    @torch.no_grad()
    def explainable_sampling(
        self,
        t_star: int,
        s_star: Tensor,
    ) -> Tensor:
        """Explainable sampling: generate counterfactual from partial tokens.

        Source: models/stage2/maskgit.py lines 235-246

        Args:
            t_star: Starting iteration index.
            s_star: Partially masked token indices (B, N).

        Returns:
            Completed token indices (B, N).
        """
        # Source: lines 241-246
        b, n = s_star.shape
        device = s_star.device

        unknown_number_in_the_beginning = (
            torch.ones((b,), device=device) * self.mask_token_id
        )  # (B,)
        s = self._parallel_unmasking(
            s_star, unknown_number_in_the_beginning, device, init_t=t_star
        )  # (B, N)
        return s

    def decode_token_ind(
        self,
        s: Tensor,
        return_representations: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Decode token indices to spectrogram.

        Source: models/stage2/maskgit.py lines 248-265
        (Originally decode_token_ind_to_timeseries)

        Adaptation: Removed timefreq_to_time conversion (source line not
        present in decode path, but method renamed for clarity).
        Outputs spectrogram directly.

        Args:
            s: Token indices (B, N).
            return_representations: Also return quantized latent.

        Returns:
            Reconstructed spectrogram (B, C, H, W), or tuple of
            (spectrogram, quantized_latent) if return_representations=True.
        """
        # Source: lines 255-258
        zq = F.embedding(s, self.vq_model._codebook.embed)  # (B, N, D)
        zq = self.vq_model.project_out(zq)  # (B, N, C)
        zq = rearrange(zq, "b n c -> b c n")  # (B, C, N) == (B, C, (H W))
        zq = rearrange(
            zq, "b c (h w) -> b c h w", h=self.H_prime, w=self.W_prime
        )

        # Source: line 260
        xhat = self.decoder(zq)  # (B, C, H, W)

        # Source: lines 262-265
        if return_representations:
            return xhat, zq
        else:
            return xhat