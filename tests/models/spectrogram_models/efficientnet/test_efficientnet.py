import pytest
import torch
from torch import nn

import torchsig_models.models.spectrogram_models.efficientnet.efficientnet as efficientnet_module
from torchsig_models.models.spectrogram_models.efficientnet.efficientnet import (
    NormalizedModel,
    SpectrogramNormalization,
    efficientnet_b0,
    efficientnet_b2,
    efficientnet_b4,
)


@pytest.mark.parametrize(
    "model_factory",
    [
        efficientnet_b0,
        efficientnet_b2,
        efficientnet_b4,
    ],
)
def test_efficientnet_2d_forward_shape(model_factory):
    model = model_factory(
        num_classes=72,
        input_channels=1,
        drop_path_rate=0.0,
        drop_rate=0.0,
    )
    model.eval()

    x = torch.randn(2, 1, 128, 128)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 72)


@pytest.mark.parametrize(
    "model_factory",
    [
        efficientnet_b0,
        efficientnet_b2,
        efficientnet_b4,
    ],
)
def test_efficientnet_2d_supports_custom_num_classes(model_factory):
    model = model_factory(
        num_classes=11,
        input_channels=1,
        drop_path_rate=0.0,
        drop_rate=0.0,
    )
    model.eval()

    x = torch.randn(2, 1, 128, 128)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 11)


def test_efficientnet_2d_supports_custom_input_channels():
    model = efficientnet_b0(
        num_classes=72,
        input_channels=2,
        drop_path_rate=0.0,
        drop_rate=0.0,
    )
    model.eval()

    x = torch.randn(2, 2, 128, 128)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 72)


def test_efficientnet_2d_disables_normalization_by_default():
    model = efficientnet_b0(
        num_classes=72,
        input_channels=1,
        drop_path_rate=0.0,
        drop_rate=0.0,
    )

    assert isinstance(model, NormalizedModel)
    assert isinstance(model.normalize, nn.Identity)
    assert isinstance(model.model, nn.Module)


def test_efficientnet_2d_supports_per_sample_normalization():
    model = efficientnet_b0(
        num_classes=72,
        input_channels=1,
        drop_path_rate=0.0,
        drop_rate=0.0,
        normalize=True,
    )

    assert isinstance(model, NormalizedModel)
    assert isinstance(model.normalize, SpectrogramNormalization)


def test_efficientnet_2d_accepts_torchsig_spectrogram_batches():
    """TorchSig spectrogram batches do not include a channel dimension."""
    model = efficientnet_b0(
        num_classes=3,
        input_channels=1,
        drop_path_rate=0.0,
        drop_rate=0.0,
    ).eval()

    with torch.no_grad():
        out = model(torch.randn(2, 64, 64))

    assert out.shape == (2, 3)


def test_spectrogram_normalization_preserves_shape():
    normalize = SpectrogramNormalization()

    x = torch.randn(4, 2, 32, 64)

    out = normalize(x)

    assert out.shape == x.shape


def test_spectrogram_normalization_normalizes_per_example_and_channel():
    normalize = SpectrogramNormalization()

    x = torch.randn(4, 2, 32, 64) * 5.0 + 10.0

    out = normalize(x)

    mean = out.mean(dim=(-2, -1))
    std = out.std(dim=(-2, -1), correction=0)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


def test_spectrogram_normalization_is_finite_for_single_pixel_inputs():
    out = SpectrogramNormalization()(torch.ones(1, 1, 1, 1))

    assert torch.isfinite(out).all()


def test_pretrained_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Pretrained loading"):
        efficientnet_b0(pretrained=True)


def test_pretrained_model_loads_source_classifier_before_replacing_it(monkeypatch):
    created_with_num_classes = []
    loaded_with_num_classes = []

    def create_model(_model_name, *, num_classes, **_kwargs):
        created_with_num_classes.append(num_classes)
        model = nn.Module()
        model.classifier = nn.Linear(4, num_classes)
        return model

    def load_pretrained(*, model, **_kwargs):
        loaded_with_num_classes.append(model.classifier.out_features)

    monkeypatch.setattr(efficientnet_module.timm, "create_model", create_model)
    monkeypatch.setattr(
        efficientnet_module,
        "_load_pretrained_if_requested",
        load_pretrained,
    )

    model = efficientnet_b0(num_classes=13, pretrained=True)

    assert created_with_num_classes == [72]
    assert loaded_with_num_classes == [72]
    assert model.model.classifier.out_features == 13
