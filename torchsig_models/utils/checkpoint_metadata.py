"""Utilities for storing and restoring inference-critical checkpoint metadata."""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Any, Literal

CHECKPOINT_METADATA_VERSION = 1
InputRepresentation = Literal["iq", "spectrogram"]


def _package_version() -> str:
    """Return installed package provenance without importing the root package."""
    try:
        return version("torchsig-models")
    except PackageNotFoundError:
        return "unknown"


def build_checkpoint_metadata(
    *,
    model_name: str,
    input_representation: InputRepresentation,
    input_channels: int,
    num_classes: int,
    class_names: list[str],
    preprocessing: dict[str, Any],
    drop_path_rate: float | None = None,
    drop_rate: float | None = None,
    model_family: str = "efficientnet",
    model_parameters: dict[str, Any] | None = None,
    postprocessing: str = "argmax",
) -> dict[str, Any]:
    """Build the versioned metadata needed to reconstruct model inference.

    Args:
        model_name: Registered model factory name.
        input_representation: Model input representation.
        input_channels: Number of channels expected by the model.
        num_classes: Number of classifier outputs.
        class_names: Labels ordered by classifier output index.
        drop_path_rate: Model stochastic-depth rate.
        drop_rate: Model dropout rate.
        preprocessing: Serializable description of required input processing.

    Returns:
        A checkpoint-safe metadata dictionary.
    """
    model = {
        "family": model_family,
        "name": model_name,
        "input_channels": input_channels,
        "num_classes": num_classes,
    }
    if drop_path_rate is not None:
        model["drop_path_rate"] = float(drop_path_rate)
    if drop_rate is not None:
        model["drop_rate"] = float(drop_rate)
    if model_parameters:
        model.update(deepcopy(model_parameters))

    return {
        "schema_version": CHECKPOINT_METADATA_VERSION,
        "model": model,
        "input": {
            "representation": input_representation,
            "dtype": "float32",
            "layout": "batch_channels_length"
            if input_representation == "iq"
            else "batch_channels_frequency_time",
            "preprocessing": deepcopy(preprocessing),
        },
        "output": {
            "class_names": list(class_names),
            "postprocessing": postprocessing,
        },
        "provenance": {"torchsig_models_version": _package_version()},
    }


def checkpoint_metadata(
    checkpoint: object,
    *,
    expected_representation: InputRepresentation | None = None,
) -> dict[str, Any] | None:
    """Read and validate inference metadata from a Lightning checkpoint.

    Weights-only and legacy checkpoints return ``None``. Unsupported schemas
    and incompatible input representations fail explicitly rather than
    silently constructing the wrong model.
    """
    if not isinstance(checkpoint, dict):
        return None
    hyperparameters = checkpoint.get("hyper_parameters", {})
    if not isinstance(hyperparameters, dict):
        return None
    metadata = hyperparameters.get("model_metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint model_metadata must be a dictionary.")
    if metadata.get("schema_version") != CHECKPOINT_METADATA_VERSION:
        raise ValueError(
            "Unsupported checkpoint metadata schema version: "
            f"{metadata.get('schema_version')!r}."
        )

    model = metadata.get("model")
    input_metadata = metadata.get("input")
    output = metadata.get("output")
    if not all(isinstance(value, dict) for value in (model, input_metadata, output)):
        raise ValueError("Checkpoint model_metadata is incomplete.")
    required_model_fields = {
        "family",
        "name",
        "input_channels",
        "num_classes",
    }
    if not required_model_fields.issubset(model):
        raise ValueError("Checkpoint model metadata is incomplete.")
    if not {"representation", "dtype", "layout", "preprocessing"}.issubset(
        input_metadata
    ):
        raise ValueError("Checkpoint input metadata is incomplete.")
    if not {"class_names", "postprocessing"}.issubset(output):
        raise ValueError("Checkpoint output metadata is incomplete.")
    if expected_representation is not None and (
        input_metadata["representation"] != expected_representation
    ):
        raise ValueError(
            "Checkpoint input representation "
            f"{input_metadata['representation']!r} is incompatible with "
            f"{expected_representation!r} inference."
        )
    return deepcopy(metadata)


def resolve_checkpoint_model_name(
    metadata: dict[str, Any] | None,
    model_name: str | None,
    *,
    legacy_default: str,
) -> str:
    """Resolve an optional model-name override against checkpoint metadata."""
    if metadata is None:
        return legacy_default if model_name is None else model_name
    stored_name = metadata["model"]["name"]
    if model_name is not None and model_name != stored_name:
        raise ValueError(
            f"model_name={model_name!r} conflicts with checkpoint model "
            f"{stored_name!r}."
        )
    return stored_name


def metadata_sidecar_path(checkpoint_path: str | Path) -> Path:
    """Return the metadata sidecar path for a framework-owned checkpoint."""
    path = Path(checkpoint_path)
    return path.with_name(f"{path.name}.metadata.json")


def write_checkpoint_metadata_sidecar(
    checkpoint_path: str | Path,
    metadata: dict[str, Any],
) -> Path:
    """Validate and write metadata beside a third-party checkpoint file."""
    validated = checkpoint_metadata({"hyper_parameters": {"model_metadata": metadata}})
    path = metadata_sidecar_path(checkpoint_path)
    path.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def read_checkpoint_metadata_sidecar(
    checkpoint_path: str | Path,
    *,
    expected_representation: InputRepresentation | None = None,
) -> dict[str, Any] | None:
    """Read an optional metadata sidecar for a third-party checkpoint."""
    path = metadata_sidecar_path(checkpoint_path)
    if not path.exists():
        return None
    contents = json.loads(path.read_text(encoding="utf-8"))
    return checkpoint_metadata(
        {"hyper_parameters": {"model_metadata": contents}},
        expected_representation=expected_representation,
    )
