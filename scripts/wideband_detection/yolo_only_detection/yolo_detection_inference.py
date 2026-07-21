"""Script interface for running YOLO detector inference on provided data."""

import argparse
from pathlib import Path

from torchsig_models.adapters.yolo_inference_detector import yolo_infer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory of training outputs.")
    parser.add_argument("--config", required=True, type=Path, help="Path to config yaml file.")
    parser.add_argument("--run_name", required=True, type=str, help="Model training run name.")
    parser.add_argument("--eval_run_name", default="detector_yolo_val", type=str, help="Evaluation run name.")
    parser.add_argument("--split", default="val", type=str, help="Train, val, test split.")
    parser.add_argument("--fft", type=int, default=512, help="FFT size of spectrogram images. Default 512.")
    args = parser.parse_args()
    
    metrics = yolo_infer(
        output_dir=args.output_dir,
        config=args.config,
        run_name=args.run_name,
        eval_run_name=args.eval_run_name,
        fft=args.fft,
        split=args.split
    )

    # print key metrics
    print("mAP50-95:", metrics.box.map)
    print("mAP50:   ", metrics.box.map50)
    print("mAP75:   ", metrics.box.map75)
    print("per-class mAP50-95:", metrics.box.maps)

    print("all metrics:")
    print(metrics.results_dict)

    # Optional: per-image precision/recall/F1/TP/FP/FN
    # print("per-image metrics:")
    # print(metrics.box.image_metrics)

if __name__ == "__main__":
    main()
