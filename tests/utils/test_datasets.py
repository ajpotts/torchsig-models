from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, TensorDataset
from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer
from torchsig.utils.file_handlers.homogeneous_hdf5 import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)


from torchsig_models.utils.datasets import (
    _create_static_dataset,
    _dataset_metadata,
    _transforms,
    prepare_torchsig_datasets,
    prepare_torchsig_inference_dataset,
)


class DummyConfig:
    dataset_id = "dummy_dataset"
    dataset_length = 12
    output_representation = "iq"
    dataset_metadata = {"sample_rate": 1_000_000}
    fft_size = 256
    seed = 123


class SeedableTensorDataset(TensorDataset):
    def seed(self, seed: int) -> None:
        self.seed_value = seed


class HomogeneousSignalDataset(Dataset):
    class_names = ["tone"]

    def seed(self, seed: int) -> None:
        self.seed_value = seed

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> Signal:
        return Signal(
            data=np.full(8, index, dtype=np.complex64),
            class_index=index,
        )


def test_dataset_metadata_merges_defaults():
    cfg = DummyConfig()

    with patch("torchsig_models.utils.datasets.TorchSigDefaults") as defaults_cls:
        defaults = MagicMock()
        defaults.default_dataset_metadata = {
            "sample_rate": 100,
            "num_iq_samples_dataset": 4096,
        }
        defaults_cls.return_value = defaults

        metadata = _dataset_metadata(cfg)

    assert metadata["sample_rate"] == 1_000_000
    assert metadata["num_iq_samples_dataset"] == 4096


def test_transforms_iq_returns_complex_to_2d():
    cfg = DummyConfig()
    cfg.output_representation = "iq"

    transforms = _transforms(cfg)

    assert len(transforms) == 1
    assert transforms[0].__class__.__name__ == "ComplexTo2D"


def test_transforms_spectrogram_returns_spectrogram_and_yolo_label():
    cfg = DummyConfig()
    cfg.output_representation = "spectrogram"
    cfg.fft_size = 512

    transforms = _transforms(cfg)

    assert len(transforms) == 2
    assert transforms[0].__class__.__name__ == "Spectrogram"
    assert transforms[1].__class__.__name__ == "YOLOLabel"


def test_transforms_unknown_output_representation_falls_back_to_spectrogram():
    cfg = DummyConfig()
    cfg.output_representation = "other"
    cfg.fft_size = 128

    transforms = _transforms(cfg)

    assert len(transforms) == 1
    assert transforms[0].__class__.__name__ == "Spectrogram"


@patch("torchsig_models.utils.datasets.StaticTorchSigDataset")
@patch("torchsig_models.utils.datasets.DatasetCreator")
@patch("torchsig_models.utils.datasets.WorkerSeedingDataLoader")
@patch("torchsig_models.utils.datasets.TorchSigIterableDataset")
def test_create_static_dataset_creates_dataset(
    iterable_dataset_cls,
    dataloader_cls,
    dataset_creator_cls,
    static_dataset_cls,
    tmp_path,
):
    cfg = DummyConfig()
    transforms = [MagicMock()]

    iterable_dataset = MagicMock()
    iterable_dataset.class_names = ["class_a", "class_b"]
    iterable_dataset_cls.return_value = iterable_dataset

    creator = MagicMock()
    dataset_creator_cls.return_value = creator

    static_dataset = MagicMock()
    static_dataset_cls.return_value = static_dataset

    result, class_names = _create_static_dataset(
        cfg=cfg,
        split="train",
        root=tmp_path,
        transforms=transforms,
        batch_size=8,
        overwrite=True,
        signal_generators=["fm-data"],
    )

    iterable_dataset_cls.assert_called_once()
    assert iterable_dataset_cls.call_args.kwargs["transforms"] == transforms
    assert iterable_dataset_cls.call_args.kwargs["signal_generators"] == ["fm-data"]

    dataloader_cls.assert_called_once()
    assert dataloader_cls.call_args.kwargs["batch_size"] == 8

    dataset_creator_cls.assert_called_once()
    assert dataset_creator_cls.call_args.kwargs["root"] == str(tmp_path / "train")
    assert dataset_creator_cls.call_args.kwargs["overwrite"] is True
    assert dataset_creator_cls.call_args.kwargs["dataset_length"] == 12

    creator.create.assert_called_once()

    static_dataset_cls.assert_called_once_with(
        root=str(tmp_path / "train"),
        file_handler_class=HDF5Reader,
        target_labels=["class_index"],
    )
    assert result is static_dataset
    assert class_names == ["class_a", "class_b"]


@patch("torchsig_models.utils.datasets.StaticTorchSigDataset")
@patch("torchsig_models.utils.datasets.DatasetCreator")
@patch("torchsig_models.utils.datasets.WorkerSeedingDataLoader")
@patch("torchsig_models.utils.datasets.TorchSigIterableDataset")
def test_prepare_torchsig_datasets_returns_three_loaders_and_info(
    iterable_dataset_cls,
    dataloader_cls,
    dataset_creator_cls,
    static_dataset_cls,
    tmp_path,
):
    cfg = DummyConfig()

    iterable_datasets = [MagicMock(), MagicMock(), MagicMock()]
    for dataset in iterable_datasets:
        dataset.class_names = ["class_a", "class_b"]
    iterable_dataset_cls.side_effect = iterable_datasets

    loaders = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]
    dataloader_cls.side_effect = loaders

    static_datasets = [MagicMock(), MagicMock(), MagicMock()]
    static_dataset_cls.side_effect = static_datasets

    creator = MagicMock()
    dataset_creator_cls.return_value = creator

    train_loader, val_loader, test_loader, info = prepare_torchsig_datasets(
        train_cfg=cfg,
        val_cfg=cfg,
        test_cfg=cfg,
        dataset_root=tmp_path,
        batch_size=16,
        overwrite=False,
        signal_generators="all",
    )

    assert train_loader is loaders[3]
    assert val_loader is loaders[4]
    assert test_loader is loaders[5]
    assert info == {
        "root": str(tmp_path / cfg.dataset_id),
        "class_names": ["class_a", "class_b"],
    }

    assert dataset_creator_cls.call_count == 3
    assert static_dataset_cls.call_count == 3
    assert dataloader_cls.call_count == 6

    roots = [call.kwargs["root"] for call in dataset_creator_cls.call_args_list]
    assert roots == [
        str(tmp_path / cfg.dataset_id / "train"),
        str(tmp_path / cfg.dataset_id / "val"),
        str(tmp_path / cfg.dataset_id / "test"),
    ]

    loader_calls = dataloader_cls.call_args_list[3:]
    assert [call.kwargs["shuffle"] for call in loader_calls] == [
        True,
        False,
        False,
    ]
    assert [call.kwargs["seed"] for call in loader_calls] == [
        123,
        123,
        123,
    ]


def test_prepare_torchsig_datasets_uses_expected_samplers_and_seed(tmp_path):
    cfg = DummyConfig()
    dataset = SeedableTensorDataset(torch.arange(12))
    class_names = ["class_a", "class_b"]

    with patch(
        "torchsig_models.utils.datasets._create_static_dataset",
        return_value=(dataset, class_names),
    ):
        loaders_a = prepare_torchsig_datasets(
            cfg,
            cfg,
            cfg,
            dataset_root=tmp_path / "a",
            batch_size=3,
        )[:3]
        loaders_b = prepare_torchsig_datasets(
            cfg,
            cfg,
            cfg,
            dataset_root=tmp_path / "b",
            batch_size=3,
        )[:3]

    assert isinstance(loaders_a[0].sampler, RandomSampler)
    assert isinstance(loaders_a[1].sampler, SequentialSampler)
    assert isinstance(loaders_a[2].sampler, SequentialSampler)

    train_order_a = torch.cat([batch[0] for batch in loaders_a[0]])
    train_order_b = torch.cat([batch[0] for batch in loaders_b[0]])
    assert torch.equal(train_order_a, train_order_b)
    assert not torch.equal(train_order_a, torch.arange(12))

    expected_order = torch.arange(12)
    for loader in loaders_a[1:]:
        assert torch.equal(
            torch.cat([batch[0] for batch in loader]),
            expected_order,
        )


def test_prepare_torchsig_datasets_creates_root(tmp_path):
    cfg = DummyConfig()
    root = tmp_path / "datasets"

    with (
        patch("torchsig_models.utils.datasets.DatasetCreator") as dataset_creator_cls,
        patch("torchsig_models.utils.datasets.StaticTorchSigDataset"),
        patch("torchsig_models.utils.datasets.TorchSigIterableDataset"),
        patch("torchsig_models.utils.datasets.WorkerSeedingDataLoader"),
    ):
        dataset_creator_cls.return_value = MagicMock()

        prepare_torchsig_datasets(
            train_cfg=cfg,
            val_cfg=cfg,
            test_cfg=cfg,
            dataset_root=root,
        )

    assert (root / cfg.dataset_id).exists()


@patch("torchsig_models.utils.datasets.StaticTorchSigDataset")
@patch("torchsig_models.utils.datasets.DatasetCreator")
@patch("torchsig_models.utils.datasets.WorkerSeedingDataLoader")
@patch("torchsig_models.utils.datasets.TorchSigIterableDataset")
def test_prepare_torchsig_datasets_uses_explicit_transforms(
    iterable_dataset_cls,
    dataloader_cls,
    dataset_creator_cls,
    static_dataset_cls,
    tmp_path,
):
    cfg = DummyConfig()
    transforms = [MagicMock()]
    dataset_creator_cls.return_value = MagicMock()
    static_dataset_cls.side_effect = [MagicMock(), MagicMock(), MagicMock()]
    dataloader_cls.side_effect = [MagicMock() for _ in range(6)]

    prepare_torchsig_datasets(
        train_cfg=cfg,
        val_cfg=cfg,
        test_cfg=cfg,
        dataset_root=tmp_path,
        transforms=transforms,
    )

    creation_calls = iterable_dataset_cls.call_args_list
    assert len(creation_calls) == 3
    assert all(call.kwargs["transforms"] is transforms for call in creation_calls)


@patch("torchsig_models.utils.datasets.StaticTorchSigDataset")
@patch("torchsig_models.utils.datasets.DatasetCreator")
@patch("torchsig_models.utils.datasets.WorkerSeedingDataLoader")
@patch("torchsig_models.utils.datasets.TorchSigIterableDataset")
def test_prepare_torchsig_datasets_forwards_file_handler_configuration(
    iterable_dataset_cls,
    dataloader_cls,
    dataset_creator_cls,
    static_dataset_cls,
    tmp_path,
):
    cfg = DummyConfig()
    writer = MagicMock(name="writer")
    reader = MagicMock(name="reader")
    options = {"compression": "gzip", "chunk_samples": 4}

    iterable_dataset_cls.return_value.class_names = ["class_a"]
    dataset_creator_cls.return_value = MagicMock()
    static_dataset_cls.side_effect = [MagicMock(), MagicMock(), MagicMock()]
    dataloader_cls.side_effect = [MagicMock() for _ in range(6)]

    prepare_torchsig_datasets(
        train_cfg=cfg,
        val_cfg=cfg,
        test_cfg=cfg,
        dataset_root=tmp_path,
        file_handler=writer,
        file_reader=reader,
        file_handler_options=options,
    )

    assert dataset_creator_cls.call_count == 3
    for call in dataset_creator_cls.call_args_list:
        assert call.kwargs["file_handler"] is writer
        assert call.kwargs["file_reader"] is reader
        assert call.kwargs["compression"] == "gzip"
        assert call.kwargs["chunk_samples"] == 4

    assert static_dataset_cls.call_count == 3
    assert all(
        call.kwargs["file_handler_class"] is reader
        for call in static_dataset_cls.call_args_list
    )


@patch("torchsig_models.utils.datasets.StaticTorchSigDataset")
@patch("torchsig_models.utils.datasets.DatasetCreator")
@patch("torchsig_models.utils.datasets.WorkerSeedingDataLoader")
@patch("torchsig_models.utils.datasets.TorchSigIterableDataset")
def test_prepare_torchsig_datasets_defaults_to_legacy_hdf5(
    iterable_dataset_cls,
    dataloader_cls,
    dataset_creator_cls,
    static_dataset_cls,
    tmp_path,
):
    cfg = DummyConfig()
    iterable_dataset_cls.return_value.class_names = ["class_a"]
    dataset_creator_cls.return_value = MagicMock()
    static_dataset_cls.side_effect = [MagicMock(), MagicMock(), MagicMock()]
    dataloader_cls.side_effect = [MagicMock() for _ in range(6)]

    prepare_torchsig_datasets(cfg, cfg, cfg, dataset_root=tmp_path)

    assert all(
        call.kwargs["file_handler"] is HDF5Writer
        and call.kwargs["file_reader"] is HDF5Reader
        for call in dataset_creator_cls.call_args_list
    )


def test_create_static_dataset_round_trips_homogeneous_hdf5(tmp_path):
    cfg = DummyConfig()
    cfg.dataset_length = 3

    with patch(
        "torchsig_models.utils.datasets.TorchSigIterableDataset",
        return_value=HomogeneousSignalDataset(),
    ):
        dataset, class_names = _create_static_dataset(
            cfg=cfg,
            split="train",
            root=tmp_path,
            transforms=[],
            batch_size=2,
            overwrite=True,
            file_handler=HomogeneousHDF5Writer,
            file_reader=HomogeneousHDF5Reader,
            file_handler_options={"compression": None},
        )

    assert class_names == ["tone"]
    assert len(dataset) == 3
    for index in range(3):
        data, label = dataset[index]
        np.testing.assert_array_equal(
            data,
            np.full(8, index, dtype=np.complex64),
        )
        assert data.dtype == np.complex64
        assert label == index
    dataset.reader.teardown()


@patch("torchsig_models.utils.datasets.WorkerSeedingDataLoader")
@patch("torchsig_models.utils.datasets.StaticTorchSigDataset")
def test_prepare_torchsig_inference_dataset_returns_loader(
    static_dataset_cls,
    dataloader_cls,
    tmp_path,
):
    static_dataset = MagicMock()
    static_dataset_cls.return_value = static_dataset

    loader = MagicMock()
    dataloader_cls.return_value = loader

    result = prepare_torchsig_inference_dataset(
        root=tmp_path,
        batch_size=8,
        num_workers=2,
        target_labels=["class_index", "snr_db"],
        pin_memory=True,
    )

    static_dataset_cls.assert_called_once_with(
        root=str(tmp_path),
        target_labels=["class_index", "snr_db"],
    )
    dataloader_cls.assert_called_once_with(
        static_dataset,
        batch_size=8,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )
    assert result is loader


@patch("torchsig_models.utils.datasets.torch.cuda.is_available")
@patch("torchsig_models.utils.datasets.WorkerSeedingDataLoader")
@patch("torchsig_models.utils.datasets.StaticTorchSigDataset")
def test_prepare_torchsig_inference_dataset_uses_defaults(
    static_dataset_cls,
    dataloader_cls,
    cuda_available,
    tmp_path,
):
    cuda_available.return_value = False

    prepare_torchsig_inference_dataset(tmp_path)

    static_dataset_cls.assert_called_once_with(
        root=str(tmp_path),
        target_labels=["class_index"],
    )
    dataloader_cls.assert_called_once_with(
        static_dataset_cls.return_value,
        batch_size=4,
        num_workers=8,
        shuffle=False,
        pin_memory=False,
    )


def test_prepare_torchsig_inference_dataset_raises_for_missing_root(
    tmp_path,
):
    missing_root = tmp_path / "missing"

    try:
        prepare_torchsig_inference_dataset(missing_root)
    except FileNotFoundError as error:
        assert str(missing_root) in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError")
