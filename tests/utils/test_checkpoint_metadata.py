"""Tests for versioned checkpoint inference metadata."""

import pytest

from torchsig_models.utils.checkpoint_metadata import (
    CHECKPOINT_METADATA_VERSION,
    build_checkpoint_metadata,
    checkpoint_metadata,
    read_checkpoint_metadata_sidecar,
    resolve_checkpoint_model_name,
    write_checkpoint_metadata_sidecar,
)


def _metadata() -> dict[str, object]:
    return build_checkpoint_metadata(
        model_name="efficientnet_b0",
        input_representation="spectrogram",
        input_channels=1,
        num_classes=2,
        class_names=["bpsk", "qpsk"],
        drop_path_rate=0.1,
        drop_rate=0.2,
        preprocessing={"transform": "Spectrogram", "fft_size": 128},
    )


def test_checkpoint_metadata_round_trip() -> None:
    metadata = _metadata()
    checkpoint = {"hyper_parameters": {"model_metadata": metadata}}

    restored = checkpoint_metadata(
        checkpoint,
        expected_representation="spectrogram",
    )

    assert restored == metadata
    assert restored is not metadata
    assert restored["schema_version"] == CHECKPOINT_METADATA_VERSION


def test_checkpoint_metadata_supports_legacy_checkpoint() -> None:
    assert checkpoint_metadata({"state_dict": {}}) is None
    assert (
        resolve_checkpoint_model_name(None, None, legacy_default="efficientnet_b4")
        == "efficientnet_b4"
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(schema_version=999), "schema version"),
        (lambda value: value.pop("input"), "incomplete"),
        (
            lambda value: value["input"].update(representation="iq"),
            "incompatible",
        ),
    ],
)
def test_checkpoint_metadata_rejects_invalid_values(mutation, match: str) -> None:
    metadata = _metadata()
    mutation(metadata)

    with pytest.raises(ValueError, match=match):
        checkpoint_metadata(
            {"hyper_parameters": {"model_metadata": metadata}},
            expected_representation="spectrogram",
        )


def test_resolve_checkpoint_model_name_rejects_conflicting_override() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        resolve_checkpoint_model_name(
            _metadata(),
            "efficientnet_b4",
            legacy_default="efficientnet_b4",
        )


def test_checkpoint_metadata_sidecar_round_trip(tmp_path) -> None:
    checkpoint_path = tmp_path / "model.pt"

    sidecar_path = write_checkpoint_metadata_sidecar(checkpoint_path, _metadata())

    assert sidecar_path == tmp_path / "model.pt.metadata.json"
    assert read_checkpoint_metadata_sidecar(checkpoint_path) == _metadata()


def test_missing_checkpoint_metadata_sidecar_is_legacy_compatible(tmp_path) -> None:
    assert read_checkpoint_metadata_sidecar(tmp_path / "legacy.pt") is None
