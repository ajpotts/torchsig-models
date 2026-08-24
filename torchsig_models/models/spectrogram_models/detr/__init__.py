"""DETR-based wideband spectrogram detection models."""

from .detr import (
    detr_b0_nano,
    detr_b0_nano_mod_family,
    detr_b2_nano,
    detr_b2_nano_mod_family,
    detr_b4_nano,
    detr_b4_nano_mod_family,
)

__all__ = [
    "detr_b0_nano",
    "detr_b2_nano",
    "detr_b4_nano",
    "detr_b0_nano_mod_family",
    "detr_b2_nano_mod_family",
    "detr_b4_nano_mod_family",
]
