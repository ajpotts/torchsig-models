"""Tests for dataset normalization utilities."""

import pytest
import torch

from torchsig_models.utils.normalization import (
    DatasetNormalization,
    compute_dataset_channel_stats,
    normalization_from_state_dict,
    resolve_checkpoint_normalization_mode,
)


def test_compute_dataset_channel_stats_across_batches():
    samples = torch.tensor(
        [
            [[1.0, 2.0], [10.0, 20.0]],
            [[3.0, 4.0], [30.0, 40.0]],
            [[5.0, 6.0], [50.0, 60.0]],
        ]
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(samples, torch.zeros(3)),
        batch_size=2,
    )

    mean, std = compute_dataset_channel_stats(loader)

    assert torch.allclose(mean, samples.mean(dim=(0, 2)))
    assert torch.allclose(std, samples.std(dim=(0, 2), correction=0))


def test_dataset_normalization_uses_channel_statistics():
    module = DatasetNormalization(mean=[2.0, 20.0], std=[2.0, 10.0])
    x = torch.tensor([[[0.0, 4.0], [10.0, 30.0]]])

    result = module(x)

    assert torch.allclose(
        result,
        torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]),
        atol=1e-6,
    )


def test_compute_dataset_channel_stats_supports_channel_less_spectrograms():
    spectrograms = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    loader = torch.utils.data.DataLoader(spectrograms, batch_size=2)

    mean, std = compute_dataset_channel_stats(loader, add_channel_dim=True)

    assert mean.shape == (1,)
    assert torch.allclose(mean, spectrograms.mean().reshape(1))
    assert torch.allclose(std, spectrograms.std(correction=0).reshape(1))


def test_dataset_normalization_buffers_round_trip():
    source = DatasetNormalization(mean=[1.0, 2.0], std=[3.0, 4.0])
    target = DatasetNormalization(mean=[5.0, 6.0], std=[7.0, 8.0])

    target.load_state_dict(source.state_dict(), strict=True)

    assert torch.equal(target.mean, source.mean)
    assert torch.equal(target.std, source.std)


def test_compute_dataset_channel_stats_rejects_empty_data():
    with pytest.raises(ValueError, match="empty"):
        compute_dataset_channel_stats([])


def test_dataset_normalization_rejects_near_zero_std():
    with pytest.raises(ValueError, match="exceed epsilon"):
        DatasetNormalization(mean=[0.0], std=[0.0])


def test_dataset_normalization_rejects_channel_mismatch():
    module = DatasetNormalization(mean=[0.0, 0.0], std=[1.0, 1.0])

    with pytest.raises(ValueError, match="2 input channels"):
        module(torch.ones(1, 1, 8))


def test_dataset_normalization_preserves_constant_separation():
    module = DatasetNormalization(mean=[10.0], std=[5.0])
    reference = torch.full((1, 1, 8, 8), 20.0)
    lower_power = reference - 10.0

    separation = (module(reference) - module(lower_power)).mean()

    assert separation.item() == pytest.approx(2.0)


def test_normalization_from_state_dict_restores_dataset_statistics():
    mean = torch.tensor([1.0, 2.0])
    std = torch.tensor([3.0, 4.0])

    kwargs = normalization_from_state_dict(
        {"normalize.mean": mean, "normalize.std": std},
        None,
        legacy_mode="none",
    )

    assert kwargs["normalization"] == "dataset"
    assert kwargs["normalization_mean"] is mean
    assert kwargs["normalization_std"] is std


def test_normalization_from_state_dict_uses_legacy_mode_with_warning():
    with pytest.warns(UserWarning, match="legacy"):
        kwargs = normalization_from_state_dict({}, None, legacy_mode="sample")

    assert kwargs["normalization"] == "sample"


def test_normalization_from_state_dict_rejects_missing_dataset_statistics():
    with pytest.raises(ValueError, match="does not contain training statistics"):
        normalization_from_state_dict({}, "dataset", legacy_mode="none")


def test_checkpoint_normalization_mode_uses_saved_metadata():
    checkpoint = {"hyper_parameters": {"normalization": {"mode": "sample"}}}

    assert resolve_checkpoint_normalization_mode(checkpoint, None) == "sample"


def test_checkpoint_normalization_mode_rejects_conflicting_override():
    checkpoint = {"hyper_parameters": {"normalization": {"mode": "none"}}}

    with pytest.raises(ValueError, match="conflicts"):
        resolve_checkpoint_normalization_mode(checkpoint, "sample")
