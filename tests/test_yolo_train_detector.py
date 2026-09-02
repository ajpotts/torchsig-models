from pathlib import Path

import pytest

from torchsig_models.adapters import yolo_train_detector


@pytest.mark.parametrize("existing_value", [None, ":16:8"])
def test_yolo_train_configures_cublas_workspace(
    monkeypatch: pytest.MonkeyPatch,
    existing_value: str | None,
) -> None:
    expected_value = existing_value or ":4096:8"
    observed_value = None

    if existing_value is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", existing_value)

    class FakeYOLO:
        def __init__(self, model: object) -> None:
            pass

        def train(self, **kwargs: object) -> None:
            nonlocal observed_value
            observed_value = yolo_train_detector.os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            )

    monkeypatch.setattr(yolo_train_detector, "YOLO", FakeYOLO)
    monkeypatch.setattr(
        yolo_train_detector,
        "get_yolo_model",
        lambda model_filepath: model_filepath,
    )
    monkeypatch.setattr(yolo_train_detector.torch.cuda, "is_available", lambda: False)

    yolo_train_detector.yolo_train(
        model_filepath=Path("model.pt"),
        config=Path("dataset.yaml"),
        output_dir=Path("runs"),
    )

    assert observed_value == expected_value
