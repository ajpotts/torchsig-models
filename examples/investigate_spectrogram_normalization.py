"""Demonstrate information retained by spectrogram normalization strategies.

Run from the repository root with::

    python examples/investigate_spectrogram_normalization.py

This is a deterministic, CPU-only diagnostic; it does not train a model.
"""

from __future__ import annotations

import numpy as np
import torch

from torchsig.utils.dsp import compute_spectrogram

from torchsig_models.models.spectrogram_models.efficientnet.efficientnet import (
    SpectrogramNormalization,
)


FFT_SIZE = 64
NUM_SAMPLES = FFT_SIZE * 32


def _spectrogram(iq: np.ndarray) -> torch.Tensor:
    values = compute_spectrogram(iq, fft_size=FFT_SIZE, fft_stride=FFT_SIZE)
    return torch.from_numpy(values).unsqueeze(0).unsqueeze(0)


def _summary(name: str, values: torch.Tensor) -> None:
    print(
        f"{name:24s} mean={values.mean().item():8.3f} "
        f"std={values.std(correction=0).item():7.3f}"
    )


def main() -> None:
    """Compare raw, per-sample, and training-dataset normalization."""
    rng = np.random.default_rng(25)
    time = np.arange(NUM_SAMPLES)
    tone = np.exp(2j * np.pi * 0.125 * time)
    noise = rng.normal(size=NUM_SAMPLES) + 1j * rng.normal(size=NUM_SAMPLES)

    reference = _spectrogram(tone + 0.10 * noise)
    lower_power = _spectrogram(0.10 * (tone + 0.10 * noise))
    noisy = _spectrogram(tone + noise)

    per_sample = SpectrogramNormalization()
    reference_per_sample = per_sample(reference)
    lower_power_per_sample = per_sample(lower_power)

    training_pixels = torch.cat([reference.flatten(), noisy.flatten()])
    training_mean = training_pixels.mean()
    training_std = training_pixels.std(correction=0)
    reference_dataset = (reference - training_mean) / training_std
    lower_power_dataset = (lower_power - training_mean) / training_std

    print("Raw TorchSIG dB spectrograms")
    _summary("reference", reference)
    _summary("same sample, -20 dB", lower_power)
    print(
        "raw mean separation:       "
        f"{(reference.mean() - lower_power.mean()).item():.3f} dB"
    )

    print("\nPer-sample standardization")
    _summary("reference", reference_per_sample)
    _summary("same sample, -20 dB", lower_power_per_sample)
    print(
        "maximum element difference: "
        f"{(reference_per_sample - lower_power_per_sample).abs().max().item():.6f}"
    )

    print("\nTraining-dataset standardization")
    _summary("reference", reference_dataset)
    _summary("same sample, -20 dB", lower_power_dataset)
    print(
        "normalized mean separation: "
        f"{(reference_dataset.mean() - lower_power_dataset.mean()).item():.3f}"
    )

    assert 19.9 < (reference.mean() - lower_power.mean()).item() < 20.1
    assert torch.allclose(
        reference_per_sample,
        lower_power_per_sample,
        atol=2e-5,
        rtol=2e-5,
    )
    assert not torch.allclose(reference_dataset, lower_power_dataset)


if __name__ == "__main__":
    main()
