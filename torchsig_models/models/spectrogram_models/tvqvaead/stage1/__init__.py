# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Stage 1 components for TimeVQVAE-AD: VQ-VAE.

Source: TimeVQVAE-AD models/stage1/
"""

from .vq import EuclideanCodebook, VectorQuantize
from .vq_vae_encdec import (
    ResBlock,
    VQVAEDecBlock,
    VQVAEDecoder,
    VQVAEEncBlock,
    VQVAEEncoder,
)

__all__ = [
    "EuclideanCodebook",
    "ResBlock",
    "VectorQuantize",
    "VQVAEDecBlock",
    "VQVAEDecoder",
    "VQVAEEncBlock",
    "VQVAEEncoder",
]