"""Employ adapter for ultralytics YOLO to download a model, if necessary.
Use the normal Ultralytics API to autodownload, unless you want this exact configuration.

Usage:
    python3 get_yolo_model.py <model_filepath>
"""
import argparse
from pathlib import Path

from torchsig_models.adapters.yolo_utils import get_yolo_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_filepath",
        type=Path, 
        default=Path("models/yolo/yolo11n.pt"),
        help="Path for YOLO model file."
        )
    args = parser.parse_args()    

    # Check if the model file exists locally; download if missing
    get_yolo_model(args.model_filepath)

if __name__ == "__main__":
    main()