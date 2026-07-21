"""Script for generating a YOLO-compatible dataset from a StaticTorchSigDataset source.
Stores the generated dataset in a YOLO-compatible subdirectory structure with images and labels, 
and creates a YOLO dataset configuration yaml file.

Example usage:
    python3 generate_yolo_from_torchsig --torchsig_dir=<dir> --train_split=0.7
"""

import argparse
import numpy as np

from torchsig.datasets.datasets import StaticTorchSigDataset

from torchsig_models.adapters.yolo_utils import static_to_yolo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=str, help="Input root directory for StaticTorchSigDataset.")
    parser.add_argument("--train_split", default=0.7, type=float, help="Train ratio of dataset elements (remainder is val data).")
    args = parser.parse_args()
    data_dir = args.dataset_dir
    train_split = args.train_split

    # source TorchSig dataset
    static_dataset = StaticTorchSigDataset(
        root=data_dir, 
        target_labels=["yolo_label"],
    )
    train_last_ind = np.round(train_split * len(static_dataset)).astype(int)

    # generate train dataset and place in subfolders
    if train_last_ind > 0:
        static_to_yolo(
            static_dataset,
            train=True, # train
            yolo_root=data_dir + "/yolo_dataset",
            start_index=0,
            stop_index=train_last_ind
        )
    
    # generate val dataset and place in subfolders
    if train_last_ind < len(static_dataset):
        static_to_yolo(
            static_dataset,
            train=False, # val
            yolo_root=data_dir + "/yolo_dataset",
            start_index=train_last_ind,
            stop_index=None # all remaining indices
        )

if __name__ == "__main__":
    main()
    
