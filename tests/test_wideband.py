"""Test wideband pipeline, yolo models and utily functions."""

import os
from pathlib import Path
import shutil

from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.transforms.transforms import Spectrogram
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.writer import DatasetCreator, identity_collate_fn
from torchsig.utils.yaml import load_config_from_yaml

from torchsig_models.adapters.yolo_utils import static_to_yolo, iterable_to_yolo
from torchsig_models.adapters.yolo_train_detector import yolo_train


SEED = 1234567890
DATASET_LENGTH = 16

DATADIR = Path.joinpath(Path(__file__).parent, "wb_tests")
data_dir = str(DATADIR)

MODELDIR = Path.joinpath(Path(__file__).parent, "models")
model_dir = str(MODELDIR)

CONFIG_FILE = Path(__file__).with_name("test_wideband_config.yaml").absolute()


def generate_test_dataset():
    """Configure and create a TorchSigIterableDataset for testing"""

    # load dataset configuration from yaml file
    cfg = load_config_from_yaml(CONFIG_FILE)

    # metadata
    base = TorchSigDefaults().default_dataset_metadata
    dataset_metadata = dict(base)
    dataset_metadata.update(cfg.dataset_metadata)
    transforms = [Spectrogram(fft_size=dataset_metadata["fft_size"]), YOLOLabel()]

    # create and write a basic wideband dataset to disk
    ds_iterable = TorchSigIterableDataset(
        metadata=dataset_metadata,
        transforms=transforms,
        target_labels=["yolo_label"],
    )
    return ds_iterable


def setup_module(module):
    """Create a small static wideband dataset on disk for tests."""
    if os.path.exists(DATADIR):
        shutil.rmtree(DATADIR)
    os.makedirs(DATADIR)

    # write dataset to disk
    iter_dataset = generate_test_dataset()
    dl = WorkerSeedingDataLoader(
        iter_dataset, collate_fn=identity_collate_fn, batch_size=16, seed=SEED
    )
    dl.seed(SEED)
    dc = DatasetCreator(
        dataloader=dl,
        root=data_dir,
        dataset_length=DATASET_LENGTH,
        overwrite=True,
    )
    dc.create()


def teardown_module(module):
    """Clean up test data, but keep any models."""
    if os.path.exists(DATADIR):
        shutil.rmtree(DATADIR)

    # if os.path.exists(MODELDIR):
    #     shutil.rmtree(MODELDIR)


def test_yolo_train_pipeline():
    """Test YOLO training pipeline from datasets to outputs."""

    # Test YOLO dataset generation from TorchSig dataset functions

    # train dataset: test StaticTorchSigDataset source
    static_dataset = StaticTorchSigDataset(
        root=data_dir,
        target_labels=["yolo_label"],
    )
    static_to_yolo(
        static_dataset,
        train=True,
        yolo_root=data_dir + "/wideband_yolo",
        start_index=0,
        stop_index=DATASET_LENGTH,
    )

    # val dataset: test TorchSigIterableDataset source
    iter_dataset = iter(generate_test_dataset())
    iterable_to_yolo(
        iter_dataset,
        train=False,
        yolo_root=data_dir + "/wideband_yolo",
        length=DATASET_LENGTH,
    )

    assert os.path.exists(data_dir + "/wideband_yolo") and os.path.isdir(
        data_dir + "/wideband_yolo"
    ), "YOLO root creation failed."
    assert os.path.exists(data_dir + "/wideband_yolo/images") and os.path.isdir(
        data_dir + "/wideband_yolo/images"
    ), "No image directory found"
    assert os.path.exists(data_dir + "/wideband_yolo/labels") and os.path.isdir(
        data_dir + "/wideband_yolo/labels"
    ), "No label directory found"
    assert os.path.exists(data_dir + "/wideband_yolo/images/train") and os.path.isdir(
        data_dir + "/wideband_yolo/images/train"
    )
    assert os.path.exists(data_dir + "/wideband_yolo/images/val") and os.path.isdir(
        data_dir + "/wideband_yolo/images/val"
    )
    assert os.path.exists(data_dir + "/wideband_yolo/labels/train") and os.path.isdir(
        data_dir + "/wideband_yolo/labels/train"
    )
    assert os.path.exists(data_dir + "/wideband_yolo/labels/val") and os.path.isdir(
        data_dir + "/wideband_yolo/labels/val"
    )
    assert os.path.exists(data_dir + "/wideband_yolo/dataset_yolo_config.yaml")

    # Test YOLO model training

    # original dataset metadata
    cfg = load_config_from_yaml(CONFIG_FILE)
    fft_size = cfg.dataset_metadata["fft_size"]

    # train model
    yolo_train(
        model_filepath=Path(model_dir + "/yolo11n.pt"),
        config=Path(data_dir + "/wideband_yolo/dataset_yolo_config.yaml"),
        output_dir=Path(data_dir + "/yolo_model_out"),
        run_name="detector_yolo",
        fft_size=fft_size,
        num_workers=0,  # avoid worker delays in small test
        epochs=1,
    )

    assert os.path.exists(data_dir + "/yolo_model_out/detector_yolo") and os.path.isdir(
        data_dir + "/yolo_model_out/detector_yolo"
    )
    assert os.path.exists(
        data_dir + "/yolo_model_out/detector_yolo/weights"
    ) and os.path.isdir(data_dir + "/yolo_model_out/detector_yolo/weights")
    assert os.path.exists(
        data_dir + "/yolo_model_out/detector_yolo/confusion_matrix_normalized.png"
    )
    assert os.path.exists(
        data_dir + "/yolo_model_out/detector_yolo/confusion_matrix.png"
    )
    # assert os.path.exists(data_dir + "/yolo_model_out/detector_yolo/labels_correlogram.jpg")
    assert os.path.exists(data_dir + "/yolo_model_out/detector_yolo/labels.jpg")
    assert os.path.exists(data_dir + "/yolo_model_out/detector_yolo/results.csv")
    assert os.path.exists(data_dir + "/yolo_model_out/detector_yolo/results.png")
