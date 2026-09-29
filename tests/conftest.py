import os
import hashlib
import pytest
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import logging
import shutil
import yaml
from torchsig.datasets.datasets import StaticTorchSigDataset, TorchSigIterableDataset
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.utils.writer import DatasetCreator
from torchsig.transforms.transforms import ComplexTo2D
from torchsig.utils.defaults import TorchSigDefaults
from torch.utils.data import default_collate


logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption(
        "--test-mode",
        action="store",
        default="fast",
        choices=("fast", "full"),
        help="fast skips selected slow tests; full runs everything",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow_no_gpu: skip in fast mode when no GPU is available",
    )
    config.addinivalue_line(
        "markers",
        "slow: skip in fast mode regardless of GPU availability",
    )


def pytest_collection_modifyitems(config, items):
    test_mode = config.getoption("--test-mode")
    has_gpu = torch.cuda.is_available()

    if test_mode == "full":
        return

    skip_slow = pytest.mark.skip(reason="skipped in fast test mode")
    skip_slow_no_gpu = pytest.mark.skip(
        reason="skipped in fast test mode because no GPU is available"
    )

    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

        if "slow_no_gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_slow_no_gpu)


# ==============================================================
# Test constants
# ==============================================================
@pytest.fixture(scope="session")
def class_names():
    return TorchSigSignalLists.all_signals


@pytest.fixture(scope="session")
def narrowband_config():
    """Load narrowband config from test_narrowband_config.yaml."""
    config_path = Path(__file__).parent / "test_narrowband_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def narrowband_data_dir(narrowband_config):
    dataset_length = int(narrowband_config["dataset_length"])
    seed = int(narrowband_config["seed"])
    config_signature = hashlib.sha1(
        yaml.safe_dump(narrowband_config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    cache_root = Path(
        os.environ.get("TORCHSIG_TEST_DATA_CACHE", ".pytest_cache/torchsig_data")
    )
    data_dir = cache_root / f"narrowband_{config_signature}"

    sentinel = data_dir / ".complete"

    def cache_is_complete() -> bool:
        if not sentinel.is_file():
            return False

        try:
            return (
                len(
                    StaticTorchSigDataset(
                        root=str(data_dir),
                        target_labels=["class_index"],
                    )
                )
                == dataset_length
            )
        except (FileNotFoundError, OSError, KeyError, RuntimeError):
            return False

    if cache_is_complete():
        logger.info("Using cached narrowband dataset: %s", data_dir)
        return data_dir

    logger.info("Creating narrowband dataset: %s", data_dir)

    if data_dir.exists():
        shutil.rmtree(data_dir)

    data_dir.mkdir(parents=True)

    base = TorchSigDefaults().default_dataset_metadata
    dataset_metadata = {
        **base,
        **narrowband_config["dataset_metadata"],
    }

    ds = TorchSigIterableDataset(
        metadata=dataset_metadata,
        transforms=[ComplexTo2D()],
        target_labels=["class_index"],
    )

    dl = WorkerSeedingDataLoader(
        ds,
        collate_fn=default_collate,
        batch_size=16,
        seed=seed,
    )
    dl.seed(seed)

    dc = DatasetCreator(
        dataloader=dl,
        root=data_dir,
        dataset_length=dataset_length,
        overwrite=True,
    )
    dc.create()

    sentinel.write_text("ok\n")

    if not cache_is_complete():
        raise RuntimeError(
            f"Failed to create readable narrowband test dataset at {data_dir}"
        )

    return data_dir


@pytest.fixture(scope="session")
def narrowband_dataloaders(narrowband_data_dir, narrowband_config):
    dataset = StaticTorchSigDataset(
        root=str(narrowband_data_dir),
        target_labels=["class_index"],
    )

    train_len = len(dataset) // 2
    val_len = len(dataset) - train_len

    train_dataset, val_dataset = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(int(narrowband_config["seed"])),
    )

    train_loader = DataLoader(train_dataset, batch_size=4)
    val_loader = DataLoader(val_dataset, batch_size=4)

    return train_loader, val_loader
