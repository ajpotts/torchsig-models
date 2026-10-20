from pathlib import Path

import pytest

from torchsig_models.adapters import yolo_train_detector
from torchsig_models.utils.checkpoint_metadata import read_checkpoint_metadata_sidecar


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


def test_write_yolo_checkpoint_metadata_for_all_weights(tmp_path: Path) -> None:
    weights_dir = tmp_path / "run" / "weights"
    weights_dir.mkdir(parents=True)
    for name in ("best.pt", "last.pt", "epoch1.pt"):
        (weights_dir / name).touch()

    yolo_train_detector._write_yolo_checkpoint_metadata(
        model_filepath=Path("yolo11n.pt"),
        output_dir=tmp_path,
        run_name="run",
        fft_size=512,
    )

    for name in ("best.pt", "last.pt", "epoch1.pt"):
        metadata = read_checkpoint_metadata_sidecar(weights_dir / name)
        assert metadata is not None
        assert metadata["model"]["family"] == "ultralytics-yolo"
        assert metadata["input"]["preprocessing"]["image_size"] == [512, 512]
