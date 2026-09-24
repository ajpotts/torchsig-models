"""Tests for classifier label metadata helpers."""

import pytest

from torchsig_models.utils.class_names import (
    checkpoint_class_names,
    validate_class_names,
)


def test_validate_class_names_preserves_order() -> None:
    names = ["zeta", "alpha", "beta"]

    result = validate_class_names(names, 3)

    assert result == names
    assert result is not names


@pytest.mark.parametrize(
    ("names", "exception"),
    [
        (["one"], ValueError),
        (["one", "one"], ValueError),
        (["one", 2], TypeError),
    ],
)
def test_validate_class_names_rejects_invalid_metadata(names, exception) -> None:
    with pytest.raises(exception):
        validate_class_names(names, 2)


def test_checkpoint_class_names_reads_lightning_hyperparameters() -> None:
    checkpoint = {"hyper_parameters": {"class_names": ["first", "second"]}}

    assert checkpoint_class_names(checkpoint, 2) == ["first", "second"]


def test_checkpoint_class_names_does_not_guess_legacy_mapping() -> None:
    assert checkpoint_class_names({"state_dict": {}}, 57) is None
