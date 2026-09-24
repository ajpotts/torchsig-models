"""Shared helpers for the quick synthetic classification examples."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = ["low", "middle", "high"]


def make_iq_dataset(samples_per_class: int, length: int, seed: int) -> TensorDataset:
    """Create noisy IQ tones whose frequency determines their class."""
    generator = torch.Generator().manual_seed(seed)
    time = torch.arange(length, dtype=torch.float32) / length
    examples: list[torch.Tensor] = []
    labels: list[int] = []
    for class_index, cycles in enumerate((3.0, 7.0, 13.0)):
        for _ in range(samples_per_class):
            phase = 2 * torch.pi * torch.rand((), generator=generator)
            angle = 2 * torch.pi * cycles * time + phase
            iq = torch.stack((torch.cos(angle), torch.sin(angle)))
            noise = 0.08 * torch.randn(iq.shape, generator=generator)
            examples.append(iq + noise)
            labels.append(class_index)
    return TensorDataset(torch.stack(examples), torch.tensor(labels))


def make_spectrogram_dataset(
    samples_per_class: int, size: int, seed: int
) -> TensorDataset:
    """Create noisy spectrograms with class-specific horizontal bands."""
    generator = torch.Generator().manual_seed(seed)
    examples: list[torch.Tensor] = []
    labels: list[int] = []
    for class_index, frequency_bin in enumerate((size // 4, size // 2, 3 * size // 4)):
        for _ in range(samples_per_class):
            image = 0.05 * torch.randn((1, size, size), generator=generator)
            image[:, frequency_bin - 2 : frequency_bin + 2, :] += 1.0
            examples.append(image)
            labels.append(class_index)
    return TensorDataset(torch.stack(examples), torch.tensor(labels))


def train_steps(
    model: nn.Module,
    dataset: TensorDataset,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Train a model for a small fixed number of optimizer steps."""
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    iterator = iter(loader)
    for step in range(steps):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, targets = next(iterator)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        print(f"step {step + 1}/{steps}: loss={loss.item():.4f}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    model_name: str,
    model_kwargs: dict[str, Any],
) -> None:
    """Save weights and the model/class metadata used by the examples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": {
                "class_names": model.class_names,
                "model_name": model_name,
                "model_kwargs": model_kwargs,
            },
        },
        path,
    )


def reload_checkpoint(path: Path, factory: Callable[..., nn.Module]) -> nn.Module:
    """Reconstruct a model and restore its ordered class names and weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["hyper_parameters"]
    model = factory(
        **metadata["model_kwargs"], class_names=metadata["class_names"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def demonstrate_prediction(model: nn.Module, dataset: TensorDataset) -> None:
    """Print one index-to-name prediction using restored model metadata."""
    example, target = dataset[0]
    with torch.inference_mode():
        predicted_index = int(model(example.unsqueeze(0)).argmax(dim=1).item())
    print(f"stored class order: {model.class_names}")
    print(
        "example prediction: "
        f"index={predicted_index}, name={model.class_names[predicted_index]!r}, "
        f"target={model.class_names[int(target)]!r}"
    )
