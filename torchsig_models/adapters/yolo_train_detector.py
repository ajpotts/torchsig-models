"""Train a YOLO model on provided dataset in YOLO compatiable format."""

import os
from pathlib import Path
import torch
from ultralytics import YOLO

from torchsig_models.adapters.yolo_utils import get_yolo_model
from torchsig_models.utils.checkpoint_metadata import (
    build_checkpoint_metadata,
    write_checkpoint_metadata_sidecar,
)


def _write_yolo_checkpoint_metadata(
    *,
    model_filepath: Path,
    output_dir: Path,
    run_name: str,
    fft_size: int,
) -> None:
    """Write metadata sidecars for every checkpoint produced by Ultralytics."""
    metadata = build_checkpoint_metadata(
        model_family="ultralytics-yolo",
        model_name=model_filepath.stem,
        input_representation="spectrogram",
        input_channels=3,
        num_classes=1,
        class_names=["signal"],
        preprocessing={
            "transform": "TorchSig spectrogram to YOLO image",
            "image_size": [fft_size, fft_size],
            "channel_order": "BGR",
            "value_domain": "uint8_0_255",
            "normalization": "per_image_minmax_then_invert",
        },
        postprocessing="non_max_suppression",
    )
    weights_dir = output_dir / run_name / "weights"
    if not weights_dir.exists():
        return
    for checkpoint_path in weights_dir.glob("*.pt"):
        write_checkpoint_metadata_sidecar(checkpoint_path, metadata)


def yolo_train(
    model_filepath: Path,
    config: Path,
    output_dir: Path,
    run_name: str = "detector_yolo",
    fft_size: int = 512,
    num_workers: int = 1,
    epochs: int = 25,
):
    """Train a YOLO model as a detector.
    Args:
        model_filepath: Path to the YOLO model file to use for training.
        config: Path to the YOLO dataset config file.
        output_dir: Directory to save the trained model and results.
        run_name: Name to label the training run. Default is "detector_yolo".
        fft_size: Image size (FFT size) for training. Default 512.
        num_workers: Number of worker processes. Default 1.
        epochs: Number of epochs to train. Default 25.
    """

    # Ultralytics enables deterministic PyTorch algorithms by default. CUDA
    # matrix multiplication also requires this workspace configuration to be
    # deterministic; it must be set before the first cuBLAS operation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    model = YOLO(
        get_yolo_model(model_filepath)
    )  # get local model, download if necessary

    model.train(
        data=config,
        epochs=epochs,
        batch=32,
        imgsz=fft_size,
        device=0 if torch.cuda.is_available() else "cpu",  # single gpu
        workers=num_workers,
        project=output_dir,
        name=run_name,
        save=True,
        save_period=1,
        single_cls=True,
        cos_lr=True,
        cache=False,
        lr0=0.00001,
        plots=True,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    _write_yolo_checkpoint_metadata(
        model_filepath=model_filepath,
        output_dir=output_dir,
        run_name=run_name,
        fft_size=fft_size,
    )
