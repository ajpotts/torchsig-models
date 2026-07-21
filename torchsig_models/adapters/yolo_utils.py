"""Utilities for working with ultralytics YOLO models."""

import os
from pathlib import Path
import requests
import numpy as np
from tqdm import tqdm
import cv2
import yaml

from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.signals.signal_lists import TorchSigSignalLists


def get_yolo_model(model_filepath: Path) -> Path:
    """Check if a YOLO model exists locally; download default specific yolo11n.pt if missing.
    Unless you want this specific version, try to use the default autodownload ultralytics api
        model = YOLO("yolo11n.pt") 
    """
    model_filepath = Path(model_filepath)
    if model_filepath.exists():
        print(f"{model_filepath} already exists.")
        return model_filepath
 
    # Create parent directory if it does not exist
    model_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Download the model file from the fixed URL
    url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
    print(f"Downloading {model_filepath.name} from {url}...")

    try:
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(model_filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print(f"File downloaded and saved to: {model_filepath}")
        return model_filepath

    except requests.RequestException as e:
        print(f"Failed to download file: {e}")
        raise


def static_to_yolo(
        dataset: StaticTorchSigDataset, 
        yolo_root: str, 
        train: bool = True, 
        start_index: int = 0, 
        stop_index: int = None
):
    """Given an existing StaticTorchSigDataset, generate a new YOLO-compatible dataset 
    version. This new dataset will be written to disk under the provided root directory,
    with png images under images/ and labels as a txt file under the labels/ subdirectory.

    Args:
        dataset: TorchSigDataset with target_labels=["yolo_label"] source for the new 
            YOLO formatted dataset of images and labels.
        yolo_root: Root directory under which the new YOLO dataset will be written.
        train: Boolean indicating whether the dataset is for training or validation. 
            This will determine the subdirectory. Default is True.
        start_index: Starting index of the source dataset. Default is 0.
        stop_index: Stopping index of the source dataset. If None (Default), all items.
    """
    stop_index = stop_index if stop_index is not None else len(dataset)

    # directories
    train_path = "train" if train else "val"
    label_dir = f"{yolo_root}/labels/{train_path}"
    image_dir = f"{yolo_root}/images/{train_path}"
    os.makedirs(yolo_root, exist_ok = True)
    os.makedirs(label_dir, exist_ok = True)
    os.makedirs(image_dir, exist_ok = True)

    # generate YOLO dataset
    for i in tqdm(range(start_index, stop_index),
                  desc=f"Writing YOLO {train_path.title()} Dataset"):
        image, labels = dataset[i] # static dataset

        filename_base = str(i).zfill(10)
        label_filename = f"{label_dir}/{filename_base}.txt"
        image_filename = f"{image_dir}/{filename_base}.png"

        # YOLO labels are expected to be (class index, x center, y center, width, height)
        # all normalized to zero, with (0,0) being upper left corner
        with open(label_filename, "w") as f:
            line = ""
            f.write("\n".join(f"{x[0]} {x[1]} {x[2]} {x[3]} {x[4]}" for x in labels))    
        img_new = np.zeros((image.shape[0], image.shape[1], 3),dtype=np.float32)    
        img_new = cv2.normalize(image, img_new, 0, 255, cv2.NORM_MINMAX)
        img_new = cv2.cvtColor(img_new.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img_new = cv2.bitwise_not(img_new)
        cv2.imwrite(image_filename, img_new, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # generate YOLO description yaml file 
    class_list = TorchSigSignalLists.all_signals
    num_classes = len(class_list)
    config_name = yolo_root + "/dataset_yolo_config.yaml"
    classes = {v: k for v,k in enumerate(class_list)}

    yolo_config = dict(
        path = yolo_root,
        train = "images/train",
        val = "images/val",
        nc = num_classes,
        names = classes
    )
    with open(config_name, 'w+') as file:
        yaml.dump(yolo_config, file, default_flow_style=False)


def iterable_to_yolo(
        dataset: TorchSigIterableDataset, 
        yolo_root: str, 
        train: bool = True, 
        length: int = 64,
):
    """Given an existing TorchSigIterableDataset, generate a new YOLO-compatible dataset 
    version. This new dataset will be written to disk under the provided root directory,
    with png images under images/ and labels as a txt file under the labels/ subdirectory.

    Args:
        dataset: TorchSigDataset with target_labels=["yolo_label"] source for the new 
            YOLO formatted dataset of images and labels.
        yolo_root: Root directory under which the new YOLO dataset will be written.
        train: Boolean indicating whether the dataset is for training or validation. 
            This will determine the subdirectory. Default is True.
        length: Number of dataset items to generate. Default is 64.
    """

    # directories
    train_path = "train" if train else "val"
    label_dir = f"{yolo_root}/labels/{train_path}"
    image_dir = f"{yolo_root}/images/{train_path}"
    os.makedirs(yolo_root, exist_ok = True)
    os.makedirs(label_dir, exist_ok = True)
    os.makedirs(image_dir, exist_ok = True)

    # generate YOLO dataset
    for i in tqdm(range(0, length),
                  desc=f"Writing YOLO {train_path.title()} Dataset"):
        image, labels = next(dataset) # iterable dataset

        filename_base = str(i).zfill(10)
        label_filename = f"{label_dir}/{filename_base}.txt"
        image_filename = f"{image_dir}/{filename_base}.png"

        # YOLO labels are expected to be (class index, x center, y center, width, height)
        # all normalized to zero, with (0,0) being upper left corner
        with open(label_filename, "w") as f:
            line = ""
            f.write("\n".join(f"{x[0]} {x[1]} {x[2]} {x[3]} {x[4]}" for x in labels))    
        img_new = np.zeros((image.shape[0], image.shape[1], 3),dtype=np.float32)    
        img_new = cv2.normalize(image, img_new, 0, 255, cv2.NORM_MINMAX)
        img_new = cv2.cvtColor(img_new.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img_new = cv2.bitwise_not(img_new)
        cv2.imwrite(image_filename, img_new, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # generate YOLO description yaml file 
    class_list = TorchSigSignalLists.all_signals
    num_classes = len(class_list)
    config_name = yolo_root + "/dataset_yolo_config.yaml"
    classes = {v: k for v,k in enumerate(class_list)}

    yolo_config = dict(
        path = yolo_root,
        train = "images/train",
        val = "images/val",
        nc = num_classes,
        names = classes
    )
    with open(config_name, 'w+') as file:
        yaml.dump(yolo_config, file, default_flow_style=False)
