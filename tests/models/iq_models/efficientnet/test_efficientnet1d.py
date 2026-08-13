import pytest
import torch
from torch import nn

from timm.layers.norm_act import BatchNormAct2d

from torchsig_models.models.iq_models.efficientnet.efficientnet1d import (
    BatchNormAct1d,
    FastGlobalAvgPool1d,
    IQNormalization,
    NormalizedModel,
    SqueezeExcite1d,
    _batch_norm_act_2d_to_1d,
    _batch_norm_2d_to_1d,
    _conv2d_to_conv1d,
    efficientnet_b0,
)


def test_squeeze_excite_1d_preserves_shape():
    module = SqueezeExcite1d(in_chs=8, reduced_base_chs=2)

    x = torch.randn(4, 8, 128)
    y = module(x)

    assert y.shape == x.shape


def test_fast_global_avg_pool_1d_flatten_true():
    module = FastGlobalAvgPool1d(flatten=True)

    x = torch.randn(4, 8, 128)
    y = module(x)

    assert y.shape == (4, 8)
    assert torch.allclose(y, x.mean(dim=-1))


def test_fast_global_avg_pool_1d_flatten_false():
    module = FastGlobalAvgPool1d(flatten=False)

    x = torch.randn(4, 8, 128)
    y = module(x)

    assert y.shape == (4, 8, 1)
    assert torch.allclose(y, x.mean(dim=-1, keepdim=True))


def test_iq_normalization_normalizes_per_sample():
    module = IQNormalization()

    x = torch.randn(4, 2, 128) * 20 + 100
    y = module(x)

    assert y.shape == x.shape
    assert torch.allclose(
        y.mean(dim=(-2, -1)),
        torch.zeros(4),
        atol=1e-5,
    )
    assert torch.allclose(
        y.std(dim=(-2, -1)),
        torch.ones(4),
        atol=1e-5,
    )


def test_batch_norm_act_1d_preserves_shape():
    module = BatchNormAct1d(num_features=8)

    x = torch.randn(4, 8, 64)
    y = module(x)

    assert y.shape == x.shape


def test_batch_norm_2d_to_1d_preserves_batch_norm_params():
    bn2d = nn.BatchNorm2d(8)

    with torch.no_grad():
        bn2d.weight.fill_(2.0)
        bn2d.bias.fill_(3.0)
        bn2d.running_mean.fill_(4.0)
        bn2d.running_var.fill_(5.0)

    bn1d = _batch_norm_2d_to_1d(bn2d)

    assert isinstance(bn1d, nn.BatchNorm1d)
    assert torch.allclose(bn1d.weight, bn2d.weight)
    assert torch.allclose(bn1d.bias, bn2d.bias)
    assert torch.allclose(bn1d.running_mean, bn2d.running_mean)
    assert torch.allclose(bn1d.running_var, bn2d.running_var)


def test_batch_norm_act_2d_to_1d_preserves_batch_norm_params():
    bn2d = BatchNormAct2d(8)

    with torch.no_grad():
        bn2d.weight.fill_(2.0)
        bn2d.bias.fill_(3.0)
        bn2d.running_mean.fill_(4.0)
        bn2d.running_var.fill_(5.0)

    bn1d = _batch_norm_act_2d_to_1d(bn2d)

    assert isinstance(bn1d, BatchNormAct1d)
    assert torch.allclose(bn1d.bn.weight, bn2d.weight)
    assert torch.allclose(bn1d.bn.bias, bn2d.bias)
    assert torch.allclose(bn1d.bn.running_mean, bn2d.running_mean)
    assert torch.allclose(bn1d.bn.running_var, bn2d.running_var)


def test_conv2d_to_conv1d_preserves_core_parameters_ds_rate_2():
    conv2d = nn.Conv2d(
        in_channels=2,
        out_channels=16,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        groups=1,
        bias=False,
    )

    conv1d = _conv2d_to_conv1d(conv2d, ds_rate=2)

    assert isinstance(conv1d, nn.Conv1d)
    assert conv1d.in_channels == 2
    assert conv1d.out_channels == 16
    assert conv1d.kernel_size == (3,)
    assert conv1d.stride == (2,)
    assert conv1d.padding == (1,)
    assert conv1d.groups == 1
    assert conv1d.bias is None


def test_conv2d_to_conv1d_preserves_adapted_weights():
    conv2d = nn.Conv2d(
        in_channels=2,
        out_channels=4,
        kernel_size=(3, 3),
        bias=False,
    )

    with torch.no_grad():
        conv2d.weight.copy_(
            torch.arange(conv2d.weight.numel(), dtype=torch.float32).view_as(
                conv2d.weight
            )
        )

    conv1d = _conv2d_to_conv1d(conv2d, ds_rate=2)

    expected = conv2d.weight.mean(dim=2)
    # If you chose dim=3 instead, use:
    # expected = conv2d.weight.mean(dim=3)

    assert torch.allclose(conv1d.weight, expected)


def test_conv2d_to_conv1d_uses_bias_flag_not_kernel_size():
    conv2d = nn.Conv2d(
        in_channels=2,
        out_channels=16,
        kernel_size=(5, 5),
        bias=True,
    )

    conv1d = _conv2d_to_conv1d(conv2d, ds_rate=2)

    assert conv1d.bias is not None


def test_conv2d_to_conv1d_ds_rate_not_2_adjusts_non_pointwise_layers():
    conv2d = nn.Conv2d(
        in_channels=2,
        out_channels=16,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=True,
    )

    conv1d = _conv2d_to_conv1d(conv2d, ds_rate=4)

    assert conv1d.kernel_size == (5,)
    assert conv1d.stride == (4,)
    assert conv1d.padding == (2,)


def test_efficientnet_b0_forward_shape():
    model = efficientnet_b0(num_classes=10)
    model.eval()

    x = torch.randn(2, 2, 1024)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 10)


def test_efficientnet_b0_returns_normalized_model_wrapper():
    model = efficientnet_b0(num_classes=72)

    assert isinstance(model, NormalizedModel)


def test_efficientnet_b0_replaces_global_pool():
    model = efficientnet_b0(num_classes=72)

    assert isinstance(model.model.global_pool, FastGlobalAvgPool1d)


def test_efficientnet_b0_replaces_classifier_for_custom_num_classes():
    model = efficientnet_b0(num_classes=13)

    assert model.model.classifier.out_features == 13


def test_efficientnet_b0_default_classifier_has_72_classes():
    model = efficientnet_b0()

    assert model.model.classifier.out_features == 72


def test_pretrained_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        efficientnet_b0(pretrained=True)
