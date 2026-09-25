"""Utilities for preserving classifier label metadata."""

from collections.abc import Sequence
from typing import Any


def validate_class_names(
    class_names: Sequence[str] | None,
    num_classes: int,
) -> list[str] | None:
    """Validate and normalize an ordered classifier label sequence.

    Args:
        class_names: Labels ordered by classifier output index, or ``None`` for
            models and legacy checkpoints without label metadata.
        num_classes: Number of classifier outputs.

    Returns:
        A copied list of labels, or ``None`` when metadata was not supplied.

    Raises:
        TypeError: If a label is not a string.
        ValueError: If the label count is wrong or labels are duplicated.
    """
    if class_names is None:
        return None

    names = list(class_names)
    if any(not isinstance(name, str) for name in names):
        raise TypeError("class_names must contain only strings.")
    if len(names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class names, received {len(names)}."
        )
    if len(set(names)) != len(names):
        raise ValueError("class_names must not contain duplicates.")
    return names


def checkpoint_class_names(
    checkpoint: dict[str, Any],
    num_classes: int,
) -> list[str] | None:
    """Read ordered class names from a Lightning checkpoint if present.

    Legacy and weights-only checkpoints return ``None`` rather than guessing a
    mapping from the classifier output count.
    """
    hyperparameters = checkpoint.get("hyper_parameters", {})
    names = hyperparameters.get("class_names")
    return validate_class_names(names, num_classes)
