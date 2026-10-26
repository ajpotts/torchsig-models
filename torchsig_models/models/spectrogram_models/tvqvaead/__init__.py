"""TimeVQVAE-AD model for time-series anomaly detection via spectrograms.

TimeVQVAE-AD is a two-stage anomaly detection model that learns to reconstruct
normal spectrogram patterns using a VQ-VAE (Stage 1) and a MaskGIT
bidirectional transformer (Stage 2). Anomalies are detected by measuring the
negative log-likelihood of masked token predictions.

Example:
    >>> from torchsig_models.models import TimeVQVAEAD
    >>> model = TimeVQVAEAD()

The model can be used with any of the supported datasets and task modes in
anomalib.

Notes:
    The model implementation is available in the ``lightning_model`` module.

See Also:
    :class:`torchsig_models.models.spec_models.tvqvae_ad.lightning_model.TimeVQVAEAD`:
        Lightning implementation of the TimeVQVAE-AD model.
"""

from .lightning_model import TimeVQVAEAD
from .torch_model import TimeVQVAEAD as TimeVQVAEADModel

__all__ = ["TimeVQVAEAD", "TimeVQVAEADModel"]
