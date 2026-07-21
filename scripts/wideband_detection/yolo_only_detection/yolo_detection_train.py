"""Script interface for running YOLO detector training.

Example usage:
    python3 yolo_detection_train.py --run_name=test_run --model=<model_file>
    --config=<config_file> --output_dir=<output_dir> --fft_size=512 --num_epochs=50
"""

import argparse
from pathlib import Path


from torchsig_models.adapters.yolo_train_detector import yolo_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True, type=str, help="Name to label model training run.")
    parser.add_argument("--model", required=True, type=Path, help="Path to YOLO model file to use for training.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YOLO dataset config yaml file.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to save YOLO training outputs.")
    parser.add_argument("--fft_size", type=int, default=512, help="FFT size of spectrogram images.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for data loading. Default is 32.")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs to train for. Default is 25.")
    args = parser.parse_args()
    
    yolo_train(
        model_filepath = args.model,
        config = args.config,
        output_dir = args.output_dir,
        run_name = args.run_name,
        fft_size = args.fft_size,
        num_workers = args.num_workers,
        epochs = args.num_epochs
    )

if __name__ == "__main__":
    main()
