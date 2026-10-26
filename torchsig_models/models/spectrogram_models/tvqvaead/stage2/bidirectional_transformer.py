# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Bidirectional Transformer for MaskGIT in TimeVQVAE-AD.

Source: TimeVQVAE-AD models/stage2/bidirectional_transformer.py (lines 1-107)

ADAPTATION NOTES:
The original implementation uses x_transformers library (ContinuousTransformerWrapper,
TFEncoder). This implementation provides equivalent functionality using native PyTorch,
implementing the key features:
- Pre-norm transformer architecture
- Optional RMSNorm (vs LayerNorm)
- Configurable attention head dimension
- Layer dropout (stochastic depth)
- Attention and feedforward dropout
- Post-embedding normalization

CRITICAL: Selective dropout (source lines 94-97) is preserved - dropout is only
applied to non-mask token embeddings during training, keeping mask token embeddings
intact for stable learning.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor


# =============================================================================
# Helper Function
# Source: models/stage2/bidirectional_transformer.py lines 11-19
# =============================================================================


def load_pretrained_tok_emb(
    pretrained_tok_emb: Tensor | None,
    tok_emb: nn.Embedding,
) -> None:
    """Load pretrained token embeddings from stage 1.

    Source: models/stage2/bidirectional_transformer.py lines 11-19

    Args:
        pretrained_tok_emb: Pretrained embeddings from VQ-VAE codebook.
        tok_emb: Token embedding layer to initialize.
    """
    # Source: lines 17-19
    with torch.no_grad():
        if pretrained_tok_emb is not None:
            tok_emb.weight[:-1, :] = pretrained_tok_emb


# =============================================================================
# RMSNorm
# Equivalent to x_transformers RMSNorm when use_rmsnorm=True
# =============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Reference: https://arxiv.org/abs/1910.07467
    Equivalent to x_transformers RMSNorm implementation.
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g


# =============================================================================
# Transformer Components
# Equivalent to x_transformers TFEncoder components
# =============================================================================


class FeedForward(nn.Module):
    """Feedforward network with GELU activation.

    Equivalent to x_transformers feedforward with ff_mult.
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim * ff_mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention with configurable head dimension.

    Equivalent to x_transformers attention with attn_dim_head.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        attn_dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = heads * attn_dim_head
        self.heads = heads
        self.scale = attn_dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        h = self.heads

        # Source equivalent: x_transformers attention
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, -1).transpose(1, 2), qkv)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block.

    Equivalent to x_transformers TFEncoder layer with pre_norm=True.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        attn_dim_head: int,
        ff_mult: int,
        use_rmsnorm: bool,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        layer_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer_dropout = layer_dropout

        # Pre-norm architecture (source line 69: pre_norm=True)
        norm_class = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.attn_norm = norm_class(dim)
        self.ff_norm = norm_class(dim)

        self.attn = Attention(
            dim=dim,
            heads=heads,
            attn_dim_head=attn_dim_head,
            dropout=attn_dropout,
        )
        self.ff = FeedForward(
            dim=dim,
            ff_mult=ff_mult,
            dropout=ff_dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Stochastic depth (layer dropout) - source line 76
        if self.training and self.layer_dropout > 0:
            if torch.rand(1).item() < self.layer_dropout:
                return x

        # Pre-norm attention
        x = x + self.attn(self.attn_norm(x))
        # Pre-norm feedforward
        x = x + self.ff(self.ff_norm(x))
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder stack.

    Equivalent to x_transformers ContinuousTransformerWrapper + TFEncoder.

    Args:
        dim_in: Input dimension.
        dim_out: Output dimension.
        hidden_dim: Transformer hidden dimension.
        depth: Number of transformer layers.
        heads: Number of attention heads.
        attn_dim_head: Dimension per attention head.
        ff_mult: Feedforward expansion factor.
        use_rmsnorm: Use RMSNorm instead of LayerNorm.
        layer_dropout: Stochastic depth dropout rate.
        attn_dropout: Attention dropout rate.
        ff_dropout: Feedforward dropout rate.
        post_emb_norm: Apply norm after input projection (source line 67).
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        attn_dim_head: int,
        ff_mult: int,
        use_rmsnorm: bool,
        layer_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        post_emb_norm: bool = True,
    ) -> None:
        super().__init__()

        # Input projection if dimensions differ
        self.proj_in = (
            nn.Linear(dim_in, hidden_dim) if dim_in != hidden_dim else nn.Identity()
        )

        # Post-embedding norm (source line 67: post_emb_norm=True)
        norm_class = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.post_emb_norm = norm_class(hidden_dim) if post_emb_norm else nn.Identity()

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                heads=heads,
                attn_dim_head=attn_dim_head,
                ff_mult=ff_mult,
                use_rmsnorm=use_rmsnorm,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                layer_dropout=layer_dropout,
            )
            for _ in range(depth)
        ])

        # Output projection if dimensions differ
        self.proj_out = (
            nn.Linear(hidden_dim, dim_out) if hidden_dim != dim_out else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj_in(x)
        x = self.post_emb_norm(x)

        for layer in self.layers:
            x = layer(x)

        return self.proj_out(x)


# =============================================================================
# BidirectionalTransformer
# Source: models/stage2/bidirectional_transformer.py lines 22-107
# =============================================================================


class BidirectionalTransformer(nn.Module):
    """Bidirectional transformer for MaskGIT token prediction.

    Source: models/stage2/bidirectional_transformer.py lines 22-107

    This transformer predicts masked tokens in a bidirectional manner,
    used for iterative parallel decoding in MaskGIT.

    Args:
        num_tokens: Number of tokens in the sequence (H' * W').
        codebook_size: Size of the VQ codebook.
        embed_dim: Token embedding dimension.
        hidden_dim: Transformer hidden dimension.
        depth: Number of transformer layers.
        heads: Number of attention heads.
        attn_dim_head: Dimension per attention head.
        ff_mult: Feedforward expansion factor.
        use_rmsnorm: Use RMSNorm instead of LayerNorm.
        dropout: Dropout rate for attention, feedforward, and layer dropout.
    """

    def __init__(
        self,
        num_tokens: int,
        codebook_size: int,
        embed_dim: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        attn_dim_head: int,
        ff_mult: int,
        use_rmsnorm: bool,
        dropout: float,
        **kwargs: object,
    ) -> None:
        # Source: lines 35-85
        super().__init__()
        self.num_tokens = num_tokens
        self.dropout = dropout
        # Source: line 56 - mask token is last index in codebook
        self.mask_token_idx = codebook_size

        # Source: line 59 - token embeddings (+1 for mask token)
        self.tok_emb = nn.Embedding(codebook_size + 1, embed_dim)

        # Source: line 62 - position embeddings (+1 for potential CLS token)
        self.pos_emb = nn.Embedding(num_tokens + 1, embed_dim)

        # Source: lines 63-79 - transformer encoder
        # Replaces ContinuousTransformerWrapper + TFEncoder with native implementation
        self.blocks = TransformerEncoder(
            dim_in=embed_dim,
            dim_out=embed_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            heads=heads,
            attn_dim_head=attn_dim_head,
            ff_mult=ff_mult,
            use_rmsnorm=use_rmsnorm,
            layer_dropout=dropout,  # Source: line 76
            attn_dropout=dropout,   # Source: line 77
            ff_dropout=dropout,     # Source: line 78
            post_emb_norm=True,     # Source: line 67
        )

        # Source: lines 80-84 - prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim, eps=1e-12),
        )

        # Source: line 85 - output bias
        self.bias = nn.Parameter(torch.zeros(num_tokens, codebook_size + 1))

    def forward(self, s_M: Tensor) -> Tensor:
        """Forward pass predicting token logits.

        Source: models/stage2/bidirectional_transformer.py lines 87-107

        CRITICAL: Selective dropout (lines 94-97) - dropout is only applied
        to non-mask token embeddings. Mask token embeddings remain intact
        for stable learning.

        Args:
            s_M: Token indices with some masked (B, N). Mask token = codebook_size.

        Returns:
            Logits over codebook (B, N, codebook_size).
        """
        # Source: lines 91-93
        b, n = s_M.shape
        token_embeddings = self.tok_emb(s_M)  # (B, N, embed_dim)

        # Source: lines 94-97 - CRITICAL selective dropout
        # Only apply dropout to non-mask tokens during training
        if self.training:
            mask_ind = (s_M == self.mask_token_idx)[:, :, None]  # (B, N, 1)
            token_embeddings_dropout = F.dropout(
                token_embeddings, p=self.dropout
            )  # (B, N, D)
            token_embeddings = torch.where(
                mask_ind, token_embeddings, token_embeddings_dropout
            )  # (B, N, D)

        # Source: lines 99-100
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = token_embeddings + position_embeddings  # (B, N, embed_dim)

        # Source: lines 102-103
        embed = self.blocks(embed)  # (B, N, embed_dim)
        embed = self.pred_head(embed)  # (B, N, embed_dim)

        # Source: lines 105-106 - compute logits
        logits = torch.matmul(embed, self.tok_emb.weight.T) + self.bias  # (B, N, K+1)
        logits = logits[:, :, :-1]  # (B, N, K) - exclude mask token from predictions

        # Source: line 107
        return logits
