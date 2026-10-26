# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Vector Quantization module for TimeVQVAE-AD.

Source: TimeVQVAE-AD models/stage1/vq.py (lines 1-421)

The code is adapted from https://github.com/lucidrains/vector-quantize-pytorch
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.distributed as distributed
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, Tensor
from torch.amp import autocast
from torch.distributions.categorical import Categorical


# =============================================================================
# Helper Functions
# Source: models/stage1/vq.py lines 19-75
# =============================================================================


def exists(val: object) -> bool:
    """Check if value is not None.

    Source: models/stage1/vq.py lines 19-20
    """
    return val is not None


def noop(*args: object, **kwargs: object) -> None:
    """No-op function for replacing distributed ops when not using DDP.

    Source: models/stage1/vq.py lines 27-28
    """
    pass


def l2norm(t: Tensor) -> Tensor:
    """L2 normalize tensor along last dimension.

    Source: models/stage1/vq.py lines 31-32
    """
    return F.normalize(t, p=2, dim=-1)


def log(t: Tensor, eps: float = 1e-20) -> Tensor:
    """Safe log with clamping.

    Source: models/stage1/vq.py lines 35-36
    """
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t: Tensor) -> Tensor:
    """Generate Gumbel noise for Gumbel-softmax sampling.

    Source: models/stage1/vq.py lines 39-41
    """
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t: Tensor, temperature: float = 1.0, dim: int = -1) -> Tensor:
    """Sample from Gumbel-softmax distribution.

    Source: models/stage1/vq.py lines 44-48
    """
    if temperature == 0:
        return t.argmax(dim=dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def softmax_sample(t: Tensor, temperature: float = 1.0, dim: int = -1) -> Tensor:
    """Sample from categorical distribution defined by logits.

    Source: models/stage1/vq.py lines 51-56
    """
    if temperature == 0:
        return t.argmax(dim=dim)

    m = Categorical(logits=t / temperature)
    return m.sample()


def ema_inplace(moving_avg: Tensor, new: Tensor, decay: float) -> None:
    """Exponential moving average update in-place.

    Source: models/stage1/vq.py lines 59-60
    """
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x: Tensor, n_categories: int, eps: float = 1e-5) -> Tensor:
    """Laplace smoothing for cluster sizes.

    Source: models/stage1/vq.py lines 63-64
    """
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples: Tensor, num: int) -> Tensor:
    """Randomly sample `num` vectors from batch.

    Source: models/stage1/vq.py lines 67-75
    """
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


# =============================================================================
# K-Means Clustering
# Source: models/stage1/vq.py lines 78-104
# =============================================================================


def kmeans(
    samples: Tensor,
    num_clusters: int,
    num_iters: int = 10,
    use_cosine_sim: bool = False,
) -> tuple[Tensor, Tensor]:
    """K-means clustering for codebook initialization.

    Source: models/stage1/vq.py lines 78-104

    Args:
        samples: Data points (N, D).
        num_clusters: Number of clusters.
        num_iters: Number of iterations.
        use_cosine_sim: Use cosine similarity instead of L2 distance.

    Returns:
        (cluster_means, cluster_sizes).
    """
    # Source: lines 79-81
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
    means = sample_vectors(samples, num_clusters)
    bins = torch.zeros(num_clusters, dtype=torch.long, device=device)

    for _ in range(num_iters):
        # Source: lines 84-88
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
            dists = -(diffs**2).sum(dim=-1)

        # Source: lines 90-93
        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        # Source: lines 95-97
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        # Source: lines 99-100
        if use_cosine_sim:
            new_means = l2norm(new_means)

        # Source: line 102
        means = torch.where(zero_mask[..., None], means, new_means)

    # Source: line 104
    return means, bins


# =============================================================================
# Regularization Losses
# Source: models/stage1/vq.py lines 107-115
# =============================================================================


def orthgonal_loss_fn(t: Tensor) -> Tensor:
    """Orthogonal regularization loss encouraging codebook diversity.

    Source: models/stage1/vq.py lines 109-115

    Note: Function name preserves original typo "orthgonal" for compatibility.
    Equation (2) from https://arxiv.org/abs/2112.00384
    """
    # Source: lines 111-115
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = torch.eye(n, device=t.device)
    cosine_sim = torch.einsum("i d, j d -> i j", normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n**2)


# =============================================================================
# EuclideanCodebook
# Source: models/stage1/vq.py lines 160-276
# =============================================================================


class EuclideanCodebook(nn.Module):
    """EMA-updated codebook with Euclidean distance and stochastic sampling.

    Source: models/stage1/vq.py lines 160-276
    Reference: https://github.com/lucidrains/vector-quantize-pytorch

    Args:
        dim: Codebook vector dimension.
        codebook_size: Number of codebook entries.
        kmeans_init: Use k-means initialization.
        kmeans_iters: K-means iterations.
        decay: EMA decay rate.
        eps: Numerical stability.
        threshold_ema_dead_code: Threshold for replacing dead codes.
        use_ddp: Use distributed data parallel.
        sample_codebook_temp: Temperature for stochastic codebook selection.
    """

    # Buffer type declarations for pyrefly
    initted: Tensor
    cluster_size: Tensor
    embed_avg: Tensor

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.8,
        eps: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        use_ddp: bool = False,
        sample_codebook_temp: float = 0.0,
    ) -> None:
        # Source: lines 164-200
        super().__init__()
        self.decay = decay
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        # Source: line 187
        self.all_reduce_fn: Callable[..., Any] = distributed.all_reduce if use_ddp else noop

        # Source: lines 189-195
        self.register_buffer("initted", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())
        self.embed = nn.Parameter(embed)

        # Source: lines 199-200
        self.embed_onehot: Tensor | None = None
        self.perplexity: Tensor | None = None

    @torch.jit.ignore
    def init_embed_(self, data: Tensor) -> None:
        """Initialize codebook with k-means on first batch.

        Source: models/stage1/vq.py lines 202-211
        """
        # Source: lines 204-205
        if self.initted.item():
            return

        # Source: lines 207-211
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples: Tensor, mask: Tensor) -> None:
        """Replace codebook entries indicated by mask.

        Source: models/stage1/vq.py lines 213-219
        """
        # Source: lines 214-219
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed,
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples: Tensor) -> None:
        """Expire and replace dead codes based on cluster usage.

        Source: models/stage1/vq.py lines 221-229
        """
        # Source: lines 222-223
        if self.threshold_ema_dead_code == 0:
            return

        # Source: lines 225-229
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace(batch_samples, mask=expired_codes)

    @autocast(device_type="cuda", enabled=False)
    def forward(self, x: Tensor, svq_temp: float | None = None) -> tuple[Tensor, Tensor]:
        """Quantize input vectors.

        Source: models/stage1/vq.py lines 231-276

        Args:
            x: Input tensor (..., dim).
            svq_temp: Override sampling temperature for inference.

        Returns:
            (quantized, embed_indices).
        """
        # Source: lines 233-236
        shape, dtype = x.shape, x.dtype
        flatten = x.reshape(-1, x.shape[-1])

        self.init_embed_(flatten)

        # Source: lines 238-245
        embed = self.embed.detach()
        embed = embed.t()

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        # Source: lines 246-250
        if self.training:
            embed_ind = softmax_sample(dist, dim=-1, temperature=self.sample_codebook_temp)
        else:
            temp = 0 if isinstance(svq_temp, type(None)) else svq_temp
            embed_ind = softmax_sample(dist, dim=-1, temperature=temp)

        # Source: lines 251-253
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.reshape(*shape[:-1]) # TODO: embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        # Source: lines 255-268
        if self.training:
            cluster_size = embed_onehot.sum(0)
            self.all_reduce_fn(cluster_size)

            ema_inplace(self.cluster_size, cluster_size, self.decay)

            embed_sum = flatten.t() @ embed_onehot
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size_smoothed = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.eps)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size_smoothed.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        # Source: lines 270-274 - perplexity
        avg_probs = torch.mean(embed_onehot, dim=0)  # (K,)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        self.embed_onehot = embed_onehot.detach()
        self.perplexity = perplexity.detach()

        # Source: line 276
        return quantize, embed_ind


# =============================================================================
# VectorQuantize Main Class
# Source: models/stage1/vq.py lines 280-420
# =============================================================================


class VectorQuantize(nn.Module):
    """Vector Quantization layer with EMA updates and stochastic sampling.

    Source: models/stage1/vq.py lines 280-420

    Args:
        dim: Input dimension.
        codebook_size: Number of codebook vectors.
        codebook_dim: Codebook vector dimension (defaults to dim).
        heads: Number of VQ heads (for multi-headed VQ).
        decay: EMA decay rate.
        eps: Numerical stability epsilon.
        kmeans_init: Use k-means initialization.
        kmeans_iters: K-means iterations for initialization.
        threshold_ema_dead_code: Threshold for replacing dead codes.
        channel_last: Whether input is channel-last format.
        accept_image_fmap: Whether to accept image feature maps (B, C, H, W).
        commitment_weight: Weight for commitment loss.
        orthogonal_reg_weight: Weight for orthogonal regularization.
        orthogonal_reg_active_codes_only: Only regularize active codes.
        orthogonal_reg_max_codes: Max codes to use for regularization.
        sample_codebook_temp: Temperature for stochastic sampling.
        sync_codebook: Sync codebook across DDP processes.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        heads: int = 1,
        decay: float = 0.8,
        eps: float = 1e-5,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 0,
        channel_last: bool = True,
        accept_image_fmap: bool = False,
        commitment_weight: float = 1.0,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_active_codes_only: bool = False,
        orthogonal_reg_max_codes: int | None = None,
        sample_codebook_temp: float = 0.0,
        sync_codebook: bool = False,
        **kwargs: object,
    ) -> None:
        # Source: lines 305-342
        super().__init__()
        self.heads = heads
        _codebook_dim: int = codebook_dim if codebook_dim is not None else dim
        codebook_input_dim = _codebook_dim * heads

        # Source: lines 310-312
        requires_projection = codebook_input_dim != dim
        self.project_in: nn.Module = (
            nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        )
        self.project_out: nn.Module = (
            nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        )

        # Source: lines 314-322
        self.eps = eps
        self.commitment_weight = commitment_weight
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        # Source: lines 324-337
        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_ddp=sync_codebook,
            sample_codebook_temp=sample_codebook_temp,
        )

        # Source: lines 339-342
        self.codebook_size = codebook_size
        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self) -> Tensor:
        """Access codebook embeddings.

        Source: models/stage1/vq.py lines 344-346
        """
        return self._codebook.embed

    def forward(
        self, x: Tensor, svq_temp: float | None = None
    ) -> tuple[Tensor, Tensor, dict[str, Tensor | float], Tensor | None]:
        """Quantize input tensor.

        Source: models/stage1/vq.py lines 348-420

        Args:
            x: Input (B, N, D) or (B, C, H, W) if accept_image_fmap.
            svq_temp: Override sampling temperature for inference.

        Returns:
            (quantized, indices, vq_loss_dict, perplexity).
        """
        # Source: lines 352-358
        shape, device, heads = x.shape, x.device, self.heads
        is_multiheaded = self.heads > 1
        need_transpose = not self.channel_last and not self.accept_image_fmap

        vq_loss: dict[str, Tensor | float] = {
            "loss": torch.tensor(0.0, device=device, requires_grad=self.training),
            "commit_loss": 0.0,
            "orthogonal_reg_loss": 0.0,
        }

        # Source: lines 360-365
        height: int = 0
        width: int = 0
        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1]) # TODO:

        if need_transpose:
            x = x.permute(0, 2, 1) # TODO:

        # Source: lines 367-372
        x = self.project_in(x)

        if is_multiheaded:
            b, n, _ = x.shape; x = x.reshape(b, n, heads, -1).permute(0, 2, 1, 3).reshape(b * heads, n, -1) # TODO:

        quantize, embed_ind = self._codebook(x, svq_temp=svq_temp)

        # Source: lines 374-375 - straight-through estimator
        if self.training:
            quantize = x + (quantize - x).detach()

        # Source: lines 377-398 - compute losses
        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                vq_loss["commit_loss"] = commit_loss
                vq_loss["loss"] = vq_loss["loss"] + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook

                if self.orthogonal_reg_active_codes_only:
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if (
                    self.orthogonal_reg_max_codes is not None
                    and num_codes > self.orthogonal_reg_max_codes
                ):
                    rand_ids = torch.randperm(num_codes, device=device)[
                        : self.orthogonal_reg_max_codes
                    ]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthgonal_loss_fn(codebook)
                vq_loss["orthogonal_reg_loss"] = orthogonal_reg_loss
                vq_loss["loss"] = (
                    vq_loss["loss"] + orthogonal_reg_loss * self.orthogonal_reg_weight
                )

        # Source: lines 407-418 - reshape outputs
        if is_multiheaded:
            quantize = quantize.reshape(-1, heads, quantize.shape[1], quantize.shape[2]).permute(0, 2, 1, 3).reshape(quantize.shape[0] // heads, quantize.shape[1], -1) # TODO:
            embed_ind.reshape(-1, heads, embed_ind.shape[1]).permute(0, 2, 1)

        quantize = self.project_out(quantize)

        if need_transpose:
            quantize = quantize.permute(0, 2, 1) # TODO:

        if self.accept_image_fmap:
            quantize = quantize.reshape(quantize.shape[0], height, width, -1).permute(0, 3, 1, 2) # TODO:
            embed_ind = embed_ind.reshape(embed_ind.shape[0], height, width, *embed_ind.shape[2:]) # TODO:

        # Source: line 420
        return quantize, embed_ind, vq_loss, self._codebook.perplexity