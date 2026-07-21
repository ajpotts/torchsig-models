"""Script interface for running XCiT classifier training.

Example usage:
    python3 train_nb_xcit1d.py --root=<root_dir> --config=<config_file> 
        --pt_dir=<pt_dir> --metrics_dir=<metrics_dir> --num_epochs=10
"""

import argparse
from pathlib import Path

from torchsig_models.models.iq_models.xcit.xcit1d_train import xcit1d_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, help="Path to dataset.")
    parser.add_argument("--config", required=True, type=Path, help="Path to dataset config yaml.")
    parser.add_argument("--pt_dir", required=True, type=Path, help="Path to place checkpoints.")
    parser.add_argument("--metrics_dir", required=True, type=Path, help="Path to place metric graphs.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train for. Default is 50.")
    args = parser.parse_args()

    xcit1d_trainer(
        root = args.root,
        config_file = args.config,
        pt_dir = args.pt_dir,
        metrics_dir = args.metrics_dir,
        num_epochs = args.num_epochs,
    )


if __name__ == "__main__":
    main()