"""Run inference on a trained YOLO model with provided dataset in YOLO format."""

from pathlib import Path
import torch
from ultralytics import YOLO


def yolo_infer(
    output_dir: Path,
    config: Path,
    run_name: str = "detector_yolo",
    eval_run_name: str = "detector_yolo_val",
    fft: int = 512,
    num_workers: int = 1,
    split: str = "val",  # "val", "test", or "train"
):
    weights = output_dir / run_name / "weights" / "best.pt"  # or "last.pt"

    model = YOLO(weights)

    metrics = model.val(
        data=str(config),
        split=split,
        imgsz=fft,
        batch=32,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=num_workers,
        project=str(output_dir),
        name=eval_run_name,
        single_cls=True,  # match your training call
        plots=True,  # saves PR curves, confusion matrix, etc.
        save_txt=True,  # saves detections as YOLO txt files
        save_conf=True,  # include confidence in saved txt detections
        save_json=True,  # optional JSON output
        # Leave conf low/default for AP-style metrics.
        # Ultralytics default is 0.001 for validation.
        conf=0.001,
        iou=0.7,
    )
    return metrics
