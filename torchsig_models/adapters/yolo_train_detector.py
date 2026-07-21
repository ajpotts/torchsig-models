"""Train a YOLO model on provided dataset in YOLO compatiable format."""

from pathlib import Path
import torch
from ultralytics import YOLO

from torchsig_models.adapters.yolo_utils import get_yolo_model


def yolo_train(
        model_filepath: Path,
        config: Path,
        output_dir: Path,
        run_name: str = "detector_yolo",
        fft_size: int = 512,
        num_workers: int = 1,
        epochs: int = 25
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

    model = YOLO(get_yolo_model(model_filepath)) # get local model, download if necessary
    
    results = model.train(
        data=config, 
        epochs=epochs,
        batch=32,
        imgsz=fft_size,
        device=0 if torch.cuda.is_available() else "cpu", # single gpu
        workers=num_workers,
        project=output_dir,
        name=run_name,
        save = True,
        save_period=1,
        single_cls=True,
        cos_lr = True,
        cache = False,
        lr0 = 0.00001,
        plots = True,
        box = 7.5,
        cls = 0.5,
        dfl = 1.5
    )