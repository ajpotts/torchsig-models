"""Inference script for an XCiT 1D classifier trained on a TorchSig dataset."""

from torchsig_models.models import XCiTClassifier
from torchsig_models.utils.training import configure_determinism
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.datasets.datamodules import TorchSigDataModule
from torchsig.transforms.transforms import ComplexTo2D
from torchsig.utils.yaml import load_config_from_yaml

import numpy as np
import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


torch.set_float32_matmul_precision("high")


def _to_single_class_index(target) -> int:
    """Convert TorchSig class_index target variants to one Python int."""

    if isinstance(target, dict):
        if "class_index" not in target:
            raise KeyError(f"Expected 'class_index' in target dict, got keys={list(target.keys())}")
        return _to_single_class_index(target["class_index"])

    if torch.is_tensor(target):
        if target.numel() != 1:
            raise ValueError(
                f"Expected exactly one class_index per sample, got tensor shape {tuple(target.shape)}"
            )
        return int(target.detach().cpu().reshape(-1)[0].item())

    if isinstance(target, np.ndarray):
        if target.size != 1:
            raise ValueError(
                f"Expected exactly one class_index per sample, got ndarray shape {target.shape}"
            )
        return int(target.reshape(-1)[0].item())

    if isinstance(target, np.generic):
        return int(target.item())

    if isinstance(target, (list, tuple)):
        if len(target) != 1:
            raise ValueError(
                "Expected exactly one class_index per sample for narrowband classification; "
                f"got {len(target)} targets: {target!r}"
            )
        return _to_single_class_index(target[0])

    return int(target)


def _to_model_input_tensor(sample) -> torch.Tensor:
    """Convert sample to float tensor with shape expected by the XCiT classifier."""
    x = sample if torch.is_tensor(sample) else torch.as_tensor(sample)

    # Safety fallback: if a raw complex IQ sample slips through, convert to [2, N].
    # If ComplexTo2D already ran, x should already be real-valued [2, N].
    if torch.is_complex(x):
        x = torch.stack((x.real, x.imag), dim=0)

    return x.to(dtype=torch.float32)


def narrowband_classifier_collate(batch):
    """Collate TorchSig narrowband classification samples.

    Handles class_index labels stored as np.int64, [np.int64], tensors,
    or one-element arrays.
    """
    xs, ys = zip(*batch)

    x_batch = torch.stack([_to_model_input_tensor(x) for x in xs], dim=0)
    y_batch = torch.tensor([_to_single_class_index(y) for y in ys], dtype=torch.long)

    return x_batch, y_batch


def _to_device_tensor(x, device):
    """Move data to device and ensure float tensor input for the model."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return x.to(device=device, dtype=torch.float32, non_blocking=True)


def _to_label_tensor(y, device):
    """Normalize TorchSig/DataLoader labels to a 1D long tensor."""
    if isinstance(y, dict):
        # Not expected for target_labels=["class_index"], but keeps this robust.
        y = y.get("class_index", next(iter(y.values())))

    if isinstance(y, (list, tuple)):
        # For single target label, PyTorch collation should usually return a tensor.
        # This handles the common edge case of a one-item list/tuple.
        if len(y) == 1:
            y = y[0]
        else:
            y = torch.as_tensor(y)

    if not torch.is_tensor(y):
        y = torch.as_tensor(y)

    y = y.to(device=device, dtype=torch.long, non_blocking=True)

    # Expected shape is [batch]. Squeeze [batch, 1] if it appears.
    if y.ndim > 1:
        y = y.squeeze()

    return y


def xcit1d_inference(
    root: str,
    config_file: str,
    checkpoint_path: str,
    batch_size: int = 4,
    num_workers: int = 32,
) -> float:
    """Load an XCiT classifier checkpoint and print overall test accuracy.

    Args:
        root: Path to the TorchSig dataset.
        config_file: Path to yaml config file used for dataset metadata.
        checkpoint_path: Path to a Lightning .ckpt file.
        batch_size: Inference batch size.
        num_workers: Number of dataloader workers.

    Returns:
        Accuracy over the test split.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Metadata/config: same construction as training.
    cfg = load_config_from_yaml(Path(config_file))
    base = TorchSigDefaults().default_dataset_metadata
    dataset_metadata = dict(base)
    dataset_metadata.update(cfg.dataset_metadata)

    # Same class definition as training.
    class_list = TorchSigSignalLists.all_signals
    num_classes = len(class_list)

    # Seed the split for repeatable inference. To exactly match an old training run,
    # use the same seeding before dm.setup() in the training script as well.
    pl.seed_everything(cfg.seed, workers=True)
    configure_determinism()

    # DataModule: same dataset setup as training, but only test loader is used.
    dm = TorchSigDataModule(
        root=root,
        metadata=dataset_metadata,
        dataset_size=cfg.dataset_length,
        dataset_splits=[0.05, 0.05, 0.9],
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=None,
        overwrite=False,
        impairment_level=cfg.impairment_level,
        transforms=[ComplexTo2D()],
        target_labels=["class_index"],
        seed=cfg.seed,
    )
    dm.prepare_data()
    dm.setup(stage="test")

    test_loader = DataLoader(
        dataset=dm.test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=narrowband_classifier_collate,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from Lightning checkpoint.
    model = XCiTClassifier.load_from_checkpoint(
        checkpoint_path,
        input_channels=2,
        num_classes=num_classes,
        map_location=device,
    )
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    
    with torch.inference_mode():
        for x, y in test_loader:
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
    
            logits = model(x)
    
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            elif isinstance(logits, dict):
                logits = logits.get("logits", next(iter(logits.values())))
    
            preds = torch.argmax(logits, dim=1)
    
            correct += (preds == y).sum().item()
            total += y.numel()
    
    accuracy = correct / total if total else 0.0
    return accuracy
