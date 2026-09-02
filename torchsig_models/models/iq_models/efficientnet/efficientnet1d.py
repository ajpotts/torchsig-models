"""1D EfficientNet models for IQ signal classification.

This module adapts timm EfficientNet architectures by replacing
2D operations with 1D equivalents suitable for complex RF/IQ data.
"""

import timm
import torch
from torch import nn
from timm.layers.norm_act import BatchNormAct2d
from timm.models._efficientnet_blocks import SqueezeExcite as TimmSqueezeExcite

__all__ = [
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_b4",
    "NormalizedModel",
    "IQNormalization",
]


PRETRAINED_CHECKPOINTS = {
    # "efficientnet_b0": {
    #     "file_id": "1ZQIBRZJiwwjeP4rB7HxxFzFro7RbxihG",
    #     "path": "efficientnet_b0.pt",
    # },
    # "efficientnet_b2": {
    #     "file_id": "1yaPZS5bbf6npHfUVdswvUnsJb8rDHlaa",
    #     "path": "efficientnet_b2.pt",
    # },
    # "efficientnet_b4": {
    #     "file_id": "1KCoLY5X0rIc_6ArmZRdkxZOOusIHN6in",
    #     "path": "efficientnet_b4.pt",
    # },
}

# The planned pretrained checkpoints were trained on the 72-class dataset.
# Build their original classifier before loading so its weights remain loadable;
# transfer-learning callers can replace that classifier afterward.
PRETRAINED_NUM_CLASSES = {
    "efficientnet_b0": 72,
    "efficientnet_b2": 72,
    "efficientnet_b4": 72,
}


def normalize_iq(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True)
    return (x - mean) / (std + eps)


class SqueezeExcite1d(nn.Module):
    """1D squeeze-and-excitation block."""

    def __init__(
        self,
        in_chs: int,
        se_ratio: float = 0.25,
        reduced_base_chs: int | None = None,
        act_layer=nn.SiLU,
        gate_fn=torch.sigmoid,
        divisor: int = 1,
        **_,
    ):
        super().__init__()
        reduced_chs = (
            reduced_base_chs
            if reduced_base_chs is not None
            else max(1, int(in_chs * se_ratio / divisor))
        )

        self.conv_reduce = nn.Conv1d(in_chs, reduced_chs, kernel_size=1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv1d(reduced_chs, in_chs, kernel_size=1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        """Apply squeeze-and-excitation scaling."""
        x_se = x.mean(dim=2, keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


class FastGlobalAvgPool1d(nn.Module):
    """Global average pooling layer for 1D feature maps."""

    def __init__(self, flatten: bool = False):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        """Compute global average pooling."""
        x = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
        if self.flatten:
            return x
        return x.view(x.size(0), x.size(1), 1)


def _copy_bn_params(src: nn.BatchNorm2d, dst: nn.BatchNorm1d) -> None:
    """Copy affine parameters and running stats from BatchNorm2d to BatchNorm1d."""
    if src.affine:
        dst.weight.data.copy_(src.weight.data)
        dst.bias.data.copy_(src.bias.data)

    if src.track_running_stats:
        dst.running_mean.data.copy_(src.running_mean.data)
        dst.running_var.data.copy_(src.running_var.data)
        dst.num_batches_tracked.data.copy_(src.num_batches_tracked.data)


class BatchNormAct1d(nn.Module):
    """1D replacement for timm BatchNormAct2d."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        act_layer: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(x))


def _batch_norm_act_2d_to_1d(module: BatchNormAct2d) -> BatchNormAct1d:
    replacement = BatchNormAct1d(
        num_features=module.num_features,
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats,
        act_layer=type(module.act),
    )

    with torch.no_grad():
        if module.affine:
            replacement.bn.weight.copy_(module.weight)
            replacement.bn.bias.copy_(module.bias)

        if module.track_running_stats:
            replacement.bn.running_mean.copy_(module.running_mean)
            replacement.bn.running_var.copy_(module.running_var)
            replacement.bn.num_batches_tracked.copy_(module.num_batches_tracked)

    return replacement


def _batch_norm_2d_to_1d(module: nn.BatchNorm2d) -> nn.BatchNorm1d:
    replacement = nn.BatchNorm1d(
        module.num_features,
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats,
    )
    _copy_bn_params(module, replacement)
    return replacement


def _replace_modules(parent: nn.Module, ds_rate: int) -> None:
    """Recursively replace 2D EfficientNet modules with 1D variants."""
    for name, module in parent.named_children():
        if isinstance(module, TimmSqueezeExcite):
            replacement = SqueezeExcite1d(
                in_chs=module.conv_reduce.in_channels,
                reduced_base_chs=module.conv_reduce.out_channels,
            )
            setattr(parent, name, replacement)

        elif isinstance(module, BatchNormAct2d):
            replacement = _batch_norm_act_2d_to_1d(module)
            setattr(parent, name, replacement)

        elif isinstance(module, nn.BatchNorm2d):
            replacement = _batch_norm_2d_to_1d(module)
            setattr(parent, name, replacement)

        elif isinstance(module, nn.Conv2d):
            replacement = _conv2d_to_conv1d(module, ds_rate)
            setattr(parent, name, replacement)

        else:
            _replace_modules(module, ds_rate)


def _conv2d_to_conv1d(module: nn.Conv2d, ds_rate: int) -> nn.Conv1d:
    """Convert a Conv2d layer to a Conv1d layer and preserve initialization."""
    if ds_rate == 2:
        kernel_size = module.kernel_size[0]
        stride = module.stride[0]
        padding = module.padding[0]
    else:
        kernel_size = module.kernel_size[0] if module.kernel_size[0] == 1 else 5
        stride = module.stride[0] if module.stride[0] == 1 else ds_rate
        padding = 0 if not module.padding[0] else 2

    replacement = nn.Conv1d(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=module.groups,
        bias=module.bias is not None,
    )

    with torch.no_grad():
        weight_2d = module.weight.data

        # Collapse the removed spatial dimension.
        weight_1d = weight_2d.mean(dim=2)

        # If kernel sizes differ because of ds_rate logic, crop or pad.
        if weight_1d.shape[-1] > kernel_size:
            start = (weight_1d.shape[-1] - kernel_size) // 2
            weight_1d = weight_1d[..., start : start + kernel_size]

        elif weight_1d.shape[-1] < kernel_size:
            pad_total = kernel_size - weight_1d.shape[-1]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            weight_1d = torch.nn.functional.pad(weight_1d, (pad_left, pad_right))

        replacement.weight.copy_(weight_1d)

        if module.bias is not None:
            replacement.bias.copy_(module.bias.data)

    return replacement


def _load_pretrained_if_requested(
    model: nn.Module,
    model_name: str,
    pretrained: bool,
    checkpoint_path: str | None,
) -> None:
    _ = model
    _ = checkpoint_path

    if not pretrained:
        return

    raise NotImplementedError(
        f"Pretrained loading for {model_name} is intentionally disabled for now. "
        "Pass checkpoint_path/load logic here later."
    )

    # Future implementation:
    #
    # path = checkpoint_path or PRETRAINED_CHECKPOINTS[model_name]["path"]
    # state_dict = torch.load(path, map_location="cpu")
    # model.load_state_dict(state_dict, strict=False)


class IQNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + self.eps)


class NormalizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.normalize = IQNormalization()
        self.model = model

    def forward(self, x):
        return self.model(self.normalize(x))


def _create_effnet_1d(
    model_name: str,
    num_classes: int,
    drop_path_rate: float,
    drop_rate: float,
    ds_rate: int = 2,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
) -> nn.Module:
    """Create and configure a 1D EfficientNet model."""
    model_num_classes = (
        PRETRAINED_NUM_CLASSES[model_name] if pretrained else num_classes
    )
    model = timm.create_model(
        model_name,
        num_classes=model_num_classes,
        in_chans=2,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
    )

    _replace_modules(model, ds_rate)
    model.global_pool = FastGlobalAvgPool1d(flatten=True)

    _load_pretrained_if_requested(
        model=model,
        model_name=model_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )

    if num_classes != model_num_classes:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return NormalizedModel(model)


def efficientnet_b0(
    num_classes: int = 72,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
):
    """Construct a 1D EfficientNet-B0 model for IQ signal classification.

    This model adapts the EfficientNet-B0 architecture from timm by
    replacing 2D convolution, pooling, normalization, and squeeze-and-
    excitation layers with 1D equivalents suitable for RF/IQ data.

    Args:
        num_classes (int):
            Number of output classes. If different from the default,
            the classifier layer is replaced to match the requested
            output dimension.

        drop_path_rate (float):
            Stochastic depth rate applied during training.

        drop_rate (float):
            Dropout rate applied before the classifier layer.

        pretrained (bool):
            If True, attempt to load pretrained weights. Pretrained
            checkpoint loading is currently not implemented and will
            raise a NotImplementedError.

        checkpoint_path (str | None):
            Optional path to a pretrained checkpoint. Reserved for
            future use when pretrained loading support is added.

    Returns:
        nn.Module:
            Configured EfficientNet-B0 model.
    """
    return _create_effnet_1d(
        "efficientnet_b0",
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )


def efficientnet_b2(
    num_classes: int = 72,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
):
    """Construct a 1D EfficientNet-B2 model for IQ signal classification.

    This model adapts the EfficientNet-B2 architecture from timm by
    replacing 2D convolution, pooling, normalization, and squeeze-and-
    excitation layers with 1D equivalents suitable for RF/IQ data.

    Args:
        num_classes (int):
            Number of output classes. If different from the default,
            the classifier layer is replaced to match the requested
            output dimension.

        drop_path_rate (float):
            Stochastic depth rate applied during training.

        drop_rate (float):
            Dropout rate applied before the classifier layer.

        pretrained (bool):
            If True, attempt to load pretrained weights. Pretrained
            checkpoint loading is currently not implemented and will
            raise a NotImplementedError.

        checkpoint_path (str | None):
            Optional path to a pretrained checkpoint. Reserved for
            future use when pretrained loading support is added.

    Returns:
        nn.Module:
            Configured EfficientNet-B2 model.
    """
    return _create_effnet_1d(
        "efficientnet_b2",
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )


def efficientnet_b4(
    num_classes: int = 72,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
):
    """Construct a 1D EfficientNet-B4 model for IQ signal classification.

    This model adapts the EfficientNet-B4 architecture from timm by
    replacing 2D convolution, pooling, normalization, and squeeze-and-
    excitation layers with 1D equivalents suitable for RF/IQ data.

    Args:
        num_classes (int):
            Number of output classes. If different from the default,
            the classifier layer is replaced to match the requested
            output dimension.

        drop_path_rate (float):
            Stochastic depth rate applied during training.

        drop_rate (float):
            Dropout rate applied before the classifier layer.

        pretrained (bool):
            If True, attempt to load pretrained weights. Pretrained
            checkpoint loading is currently not implemented and will
            raise a NotImplementedError.

        checkpoint_path (str | None):
            Optional path to a pretrained checkpoint. Reserved for
            future use when pretrained loading support is added.

    Returns:
        nn.Module:
            Configured EfficientNet-B4 model.
    """
    return _create_effnet_1d(
        "efficientnet_b4",
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )
