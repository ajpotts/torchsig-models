# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""VQ-VAE Encoder and Decoder for TimeVQVAE-AD (Spectrogram Input).

Source: TimeVQVAE-AD models/stage1/vq_vae_encdec.py (lines 1-213)
Reference: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py

ADAPTATION NOTES (Pre-computed Spectrogram Input):
This implementation expects pre-computed spectrogram input (B, C, freq_bins, time_steps),
removing the internal STFT conversion from the original source.

Removed from source:
- time_to_timefreq() / timefreq_to_time() conversions (lines 131, 192)
- n_fft parameter (lines 95, 108, 153, 166)
- input_length parameter and interpolation layers (lines 152, 186-188)
- x_channels parameter (line 154)

Preserved from source:
- frequency_independence parameter: controls kernel shapes
  - True: (1,3)/(1,4) kernels, (1,2) strides - processes freq bins independently
  - False: (3,3)/(3,4) kernels - allows cross-frequency interaction
- Asymmetric strides (1,2): only downsamples temporal axis, preserves frequency axis
- Channel progression: 4 -> 8 -> 16 -> ... -> dim
- ResBlock with Snake activation and skip connections
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, Tensor

from ..components.utils import SnakeActivation


# =============================================================================
# ResBlock
# Source: models/stage1/vq_vae_encdec.py lines 11-35
# =============================================================================


class ResBlock(nn.Module):
    """Residual block with Snake activation.

    Source: models/stage1/vq_vae_encdec.py lines 11-35

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        frequency_independence: If True, use (1,3) kernels for frequency-independent
            processing. If False, use (3,3) kernels.
        mid_channels: Middle channels (defaults to out_channels).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frequency_independence: bool,
        mid_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        # Source: lines 12-32
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        # Source: lines 18-19
        kernel_size = (1, 3) if frequency_independence else (3, 3)
        padding = (0, 1) if frequency_independence else (1, 1)

        # Source: lines 21-31
        layers = [
            SnakeActivation(in_channels, 2),
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
            ),
            nn.Dropout(dropout),
        ]
        self.convs = nn.Sequential(*layers)

        # Source: line 32
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Source: models/stage1/vq_vae_encdec.py lines 34-35
        """
        return self.proj(x) + self.convs(x)


# =============================================================================
# VQVAEEncBlock
# Source: models/stage1/vq_vae_encdec.py lines 38-59
# =============================================================================


class VQVAEEncBlock(nn.Module):
    """Encoder downsampling block.

    Source: models/stage1/vq_vae_encdec.py lines 38-59

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        frequency_independence: If True, use (1,4) kernel with (1,2) stride for
            temporal-only downsampling. If False, use (3,4) kernel.
        dropout: Dropout rate.
        symmetric_stride: If True, use (2,2) stride to downsample both axes.
            Used in hybrid mode for early blocks to reduce frequency dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frequency_independence: bool,
        dropout: float = 0.0,
        symmetric_stride: bool = False,
    ) -> None:
        # Source: lines 39-55
        super().__init__()

        # Source: lines 47-48
        if symmetric_stride:
            kernel_size = (4, 4)
            padding = (1, 1)
            stride = (2, 2)
        elif frequency_independence:
            kernel_size = (1, 4)
            padding = (0, 1)
            stride = (1, 2)
        else:
            kernel_size = (3, 4)
            padding = (1, 1)
            stride = (1, 2)

        # Source: lines 50-55
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="zeros",
            ),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Source: models/stage1/vq_vae_encdec.py lines 57-59
        """
        return self.block(x)


# =============================================================================
# VQVAEDecBlock
# Source: models/stage1/vq_vae_encdec.py lines 62-82
# =============================================================================


class VQVAEDecBlock(nn.Module):
    """Decoder upsampling block.

    Source: models/stage1/vq_vae_encdec.py lines 62-82

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        frequency_independence: If True, use (1,4) kernel with (1,2) stride for
            temporal-only upsampling. If False, use (3,4) kernel.
        dropout: Dropout rate.
        symmetric_stride: If True, use (2,2) stride to upsample both axes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frequency_independence: bool,
        dropout: float = 0.0,
        symmetric_stride: bool = False,
    ) -> None:
        # Source: lines 63-78
        super().__init__()

        if symmetric_stride:
            kernel_size = (4, 4)
            padding = (1, 1)
            stride = (2, 2)
        elif frequency_independence:
            kernel_size = (1, 4)
            padding = (0, 1)
            stride = (1, 2)
        else:
            kernel_size = (3, 4)
            padding = (1, 1)
            stride = (1, 2)

        # Source: lines 74-78
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Source: models/stage1/vq_vae_encdec.py lines 80-82
        """
        return self.block(x)


# =============================================================================
# VQVAEEncoder
# Source: models/stage1/vq_vae_encdec.py lines 85-139
# Adaptation: Removed STFT conversion (time_to_timefreq), n_fft parameter.
#             Expects pre-computed spectrogram input (B, C, H, W).
# =============================================================================


class VQVAEEncoder(nn.Module):
    """VQ-VAE Encoder for spectrogram input.

    Source: models/stage1/vq_vae_encdec.py lines 85-139

    Adaptation: Removed time_to_timefreq() conversion (line 131) and
    n_fft parameter (lines 95, 108). Expects pre-computed spectrogram
    input (B, C, freq_bins, time_steps).

    Args:
        dim: Output latent dimension.
        num_channels: Input spectrogram channels.
        downsample_rate: Temporal downsampling factor (power of 2: 2, 4, 8, ...).
        n_resnet_blocks: Number of ResNet blocks per stage.
        frequency_independence: If True, process frequency bins independently.
        dropout: Dropout rate.
        hybrid_symmetric_blocks: Number of leading encoder blocks that use (2,2)
            stride to downsample both frequency and time axes. Remaining blocks
            use the standard (1,2) stride (temporal-only). Set to 0 for original
            behavior. For large spectrograms, set to 1-3 to reduce frequency
            dimension early while preserving resolution in later layers.
    """

    num_tokens: Tensor
    H_prime: Tensor
    W_prime: Tensor

    def __init__(
        self,
        dim: int,
        num_channels: int,
        downsample_rate: int,
        n_resnet_blocks: int,
        frequency_independence: bool,
        dropout: float = 0.3,
        hybrid_symmetric_blocks: int = 0,
        **kwargs: object,
    ) -> None:
        # Source: lines 90-124
        super().__init__()

        # Removed: self.n_fft = n_fft (source line 108)

        # Source: lines 110-118 - Build encoder layers
        # Channel progression: 4 -> 8 -> 16 -> ... -> dim
        total_blocks = int(round(np.log2(downsample_rate)))
        d = 4

        # First block (block index 0)
        use_symmetric = hybrid_symmetric_blocks > 0
        enc_layers: list[nn.Module] = [
            VQVAEEncBlock(num_channels, d, frequency_independence, symmetric_stride=use_symmetric),
        ]
        d *= 2

        # Remaining downsample blocks (block indices 1 .. total_blocks-1)
        for block_idx in range(1, total_blocks):
            use_symmetric = block_idx < hybrid_symmetric_blocks
            enc_layers.append(
                VQVAEEncBlock(d // 2, d, frequency_independence, symmetric_stride=use_symmetric),
            )
            for _ in range(n_resnet_blocks):
                enc_layers.append(ResBlock(d, d, frequency_independence, dropout=dropout))
            d *= 2

        # Source: line 118
        enc_layers.append(ResBlock(d // 2, dim, frequency_independence, dropout=dropout))
        self.encoder = nn.Sequential(*enc_layers)

        # Store stride config for analytical latent size computation
        self._total_blocks = total_blocks
        self._hybrid_symmetric_blocks = hybrid_symmetric_blocks

        # Source: lines 121-124 - Track spatial dimensions
        self.is_num_tokens_updated = False
        self.register_buffer("num_tokens", torch.tensor(0))
        self.register_buffer("H_prime", torch.tensor(0))
        self.register_buffer("W_prime", torch.tensor(0))

    def compute_latent_shape(self, input_h: int, input_w: int) -> tuple[int, int]:
        """Analytically compute latent (H', W') from input dimensions.

        Args:
            input_h: Input height.
            input_w: Input width.

        Returns:
            Tuple of (latent_height, latent_width).
        """
        h, w = input_h, input_w
        for block_idx in range(self._total_blocks):
            if block_idx < self._hybrid_symmetric_blocks:
                h //= 2
            w //= 2
        return h, w

    def forward(self, x: Tensor) -> Tensor:
        """Encode spectrogram to latent representation.

        Source: models/stage1/vq_vae_encdec.py lines 126-139

        Adaptation: Removed time_to_timefreq() call (source line 131).
        Input is pre-computed spectrogram (B, C, freq_bins, time_steps).

        Args:
            x: Input spectrogram tensor (B, C, H, W).

        Returns:
            Encoded latent (B, dim, H', W').
        """
        # Removed: x = time_to_timefreq(x, self.n_fft, in_channels)

        # Source: line 133
        out = self.encoder(x)  # (B, dim, H', W')

        # Source: lines 134-138 - Update spatial dimension tracking
        if not self.is_num_tokens_updated:
            self.H_prime = torch.tensor(out.shape[2])
            self.W_prime = torch.tensor(out.shape[3])
            self.num_tokens = self.H_prime * self.W_prime  # pyrefly: ignore
            self.is_num_tokens_updated = True

        return out


# =============================================================================
# VQVAEDecoder
# Source: models/stage1/vq_vae_encdec.py lines 142-196
# Adaptation: Removed STFT conversion (timefreq_to_time), n_fft, input_length.
#             Outputs spectrogram directly (B, C, H, W).
# =============================================================================


class VQVAEDecoder(nn.Module):
    """VQ-VAE Decoder for spectrogram output.

    Source: models/stage1/vq_vae_encdec.py lines 142-196

    Adaptation: Removed timefreq_to_time() conversion (line 192),
    n_fft parameter (lines 153, 166), input_length parameter (line 152),
    and interpolation layers (lines 186-188). Outputs spectrogram directly.

    Args:
        dim: Input latent dimension.
        num_channels: Output spectrogram channels.
        downsample_rate: Temporal upsample factor (must match encoder).
        n_resnet_blocks: Number of ResNet blocks per stage.
        frequency_independence: If True, process frequency bins independently.
        dropout: Dropout rate.
        hybrid_symmetric_blocks: Number of trailing decoder blocks that use (2,2)
            stride. Must match encoder's hybrid_symmetric_blocks to ensure
            the decoder mirrors the encoder's spatial dimensions.
    """

    def __init__(
        self,
        dim: int,
        num_channels: int,
        downsample_rate: int,
        n_resnet_blocks: int,
        frequency_independence: bool,
        dropout: float = 0.3,
        hybrid_symmetric_blocks: int = 0,
        **kwargs: object,
    ) -> None:
        # Source: lines 147-187
        super().__init__()

        # Removed: self.n_fft, self.x_channels, input_length params

        total_blocks = int(round(np.log2(downsample_rate)))

        # Source: lines 172-174 - Compute starting dimension
        d = int(4 * 2 ** (total_blocks - 1))
        if total_blocks == 0:
            d = int(4 * 2 ** total_blocks)

        # Source: lines 176-184 - Build decoder layers
        dec_layers: list[nn.Module] = [ResBlock(dim, d, frequency_independence, dropout=dropout)]

        # VQVAEDecBlock layers (block indices 0 .. total_blocks-2)
        # These mirror encoder blocks in reverse: encoder block [total_blocks-1]
        # corresponds to decoder block [0], etc.
        for block_idx in range(total_blocks - 1):
            for _ in range(n_resnet_blocks):
                dec_layers.append(ResBlock(d, d, frequency_independence, dropout=dropout))
            d //= 2
            # Mirror encoder: encoder's last blocks are temporal-only,
            # encoder's first blocks are symmetric. In reverse, decoder's
            # last VQVAEDecBlocks should be symmetric.
            encoder_mirror_idx = total_blocks - 1 - block_idx
            use_symmetric = encoder_mirror_idx < hybrid_symmetric_blocks
            dec_layers.append(
                VQVAEDecBlock(2 * d, d, frequency_independence, symmetric_stride=use_symmetric),
            )

        # Source: lines 182-183 - Final upsampling layer (mirrors encoder block 0)
        use_symmetric = hybrid_symmetric_blocks > 0
        if use_symmetric:
            kernel_size = (4, 4)
            padding = (1, 1)
            stride = (2, 2)
        elif frequency_independence:
            kernel_size = (1, 4)
            padding = (0, 1)
            stride = (1, 2)
        else:
            kernel_size = (3, 4)
            padding = (1, 1)
            stride = (1, 2)

        dec_layers.append(
            nn.ConvTranspose2d(
                d,
                num_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )

        self.decoder = nn.Sequential(*dec_layers)

        # Removed: self.interp and self.linear for time-series length matching

    def forward(self, x: Tensor) -> Tensor:
        """Decode latent to spectrogram.

        Source: models/stage1/vq_vae_encdec.py lines 190-196

        Adaptation: Removed timefreq_to_time() call (source line 192)
        and interpolation (lines 194-195). Output is spectrogram directly.

        Args:
            x: Latent tensor (B, dim, H', W').

        Returns:
            Reconstructed spectrogram (B, C, H, W).
        """
        # Source: line 191
        out = self.decoder(x)  # (B, C, H, W)

        # Removed: out = timefreq_to_time(out, self.n_fft, self.x_channels)
        # Removed: out = self.interp(out); out = out + self.linear(out)

        return out