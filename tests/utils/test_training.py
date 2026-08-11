import os
import random

import numpy as np
import pytest
import tempfile
import torch
from pathlib import Path


from torchsig_models.utils.training import (
    train_validate,
    compute_class_weights_tensor,
    compute_num_params,
    set_deterministic,
    SignalClassifier,
    train_validate,
    evaluate_classifier,
)
from torchsig_models.utils.classifier_metrics_tracker import (
    ClassifierMetricsTracker,
    ClassifierMetricsTrackerCallback,
)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, labels):
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.zeros(2), self.labels[idx]


def test_set_deterministic_reproducible_random_state():
    set_deterministic(123)

    python_rand_1 = random.random()
    numpy_rand_1 = np.random.rand()
    torch_rand_1 = torch.rand(3)

    set_deterministic(123)

    python_rand_2 = random.random()
    numpy_rand_2 = np.random.rand()
    torch_rand_2 = torch.rand(3)

    assert python_rand_1 == python_rand_2
    assert numpy_rand_1 == numpy_rand_2
    assert torch.equal(torch_rand_1, torch_rand_2)


def test_set_deterministic_sets_environment_and_cudnn_flags():
    set_deterministic(42)

    assert os.environ["PYTHONHASHSEED"] == "42"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_compute_class_weights_tensor_returns_inverse_frequency_weights():
    dataset = SimpleDataset(labels=[0, 0, 1, 2, 2, 2])
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    weights = compute_class_weights_tensor(loader, num_classes=3)

    expected = torch.tensor([1.0, 2.0, 2.0 / 3.0], dtype=torch.float)

    assert weights.shape == (3,)
    assert weights.dtype == torch.float
    assert torch.allclose(weights, expected)


def test_compute_class_weights_tensor_includes_missing_classes():
    dataset = SimpleDataset(labels=[0, 0, 2])
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    weights = compute_class_weights_tensor(loader, num_classes=4)

    expected = torch.tensor([0.375, 0.75, 0.75, 0.75], dtype=torch.float)

    assert weights.shape == (4,)
    assert torch.allclose(weights, expected)


def test_compute_class_weights_tensor_handles_batched_labels():
    dataset = SimpleDataset(labels=[0, 1, 1, 1])
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    weights = compute_class_weights_tensor(loader, num_classes=2)

    expected = torch.tensor([2.0, 2.0 / 3.0], dtype=torch.float)

    assert torch.allclose(weights, expected)


def test_compute_class_weights_tensor_accumulates_batches_with_bincount(
    monkeypatch,
):
    dataset = SimpleDataset(labels=[0, 1, 1, 2, 2])
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    bincount = torch.bincount
    observed_batches = []

    def recording_bincount(labels, *, minlength):
        observed_batches.append(labels.clone())
        return bincount(labels, minlength=minlength)

    monkeypatch.setattr(torch, "bincount", recording_bincount)

    weights = compute_class_weights_tensor(loader, num_classes=3)

    assert [batch.tolist() for batch in observed_batches] == [[0, 1], [1, 2], [2]]
    assert torch.allclose(weights, torch.tensor([5 / 3, 5 / 6, 5 / 6]))


@pytest.mark.parametrize("label", [-1, 3])
def test_compute_class_weights_tensor_rejects_out_of_range_labels(label):
    loader = torch.utils.data.DataLoader(
        SimpleDataset(labels=[0, label]),
        batch_size=2,
        shuffle=False,
    )

    with pytest.raises(ValueError, match="Labels must be in the range"):
        compute_class_weights_tensor(loader, num_classes=3)


def test_compute_num_params_counts_only_trainable_parameters():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),  # 4 * 3 + 3 = 15
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),  # 3 * 2 + 2 = 8
    )

    model[2].weight.requires_grad = False
    model[2].bias.requires_grad = False

    assert compute_num_params(model) == 15


def test_compute_num_params_counts_all_trainable_parameters():
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),  # 100 * 50 + 50 = 5050
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),  # 50 * 10 + 10 = 510
    )

    assert compute_num_params(model) == 5560


def test_compute_num_params_returns_zero_when_all_parameters_frozen():
    model = torch.nn.Linear(4, 3)

    for param in model.parameters():
        param.requires_grad = False

    assert compute_num_params(model) == 0


@pytest.mark.slow_no_gpu
def test_narrowband_training(
    narrowband_dataloaders,
    narrowband_data_dir,
    narrowband_config,
    class_names,
):
    """Test full training pipeline with narrowband data creates expected metrics."""
    train_loader, val_loader = narrowband_dataloaders
    num_classes = len(class_names)

    max_epochs = 2
    learning_rate = 0.001

    model = torch.nn.Sequential(
        torch.nn.Conv1d(2, 16, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(16, num_classes),
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pl_model, metrics = train_validate(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=max_epochs,
            num_classes=num_classes,
            metrics_dir=tmpdir,
            checkpoint_dir=tmpdir,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="32-true",
            use_distributed_sampler=False,
            gradient_clip_val=0.5,
            enable_progress_bar=False,
            clamp_logits=False,
        )

        assert isinstance(pl_model, SignalClassifier)
        assert pl_model.clamp_logits is False

        assert isinstance(metrics, ClassifierMetricsTrackerCallback)

        for phase_tracker in [metrics.train_metrics, metrics.val_metrics]:
            assert set(phase_tracker.history.keys()) == {
                "loss",
                "accuracy",
                "f1 score",
                "precision",
                "recall",
            }

            for key, values in phase_tracker.history.items():
                assert len(values) == max_epochs, (
                    f"{key} should have {max_epochs} entries, got {len(values)}"
                )
                assert all(isinstance(v, (int, float)) for v in values), (
                    f"Metric {key} should contain only numeric values"
                )

            assert len(phase_tracker.conf_mats) == max_epochs

            for conf_mat in phase_tracker.conf_mats:
                assert isinstance(conf_mat, list)
                assert len(conf_mat) == num_classes
                assert all(len(row) == num_classes for row in conf_mat)

        metrics.save_to_csv(tmpdir)
        metrics.plot(save_dir=tmpdir, show=False, close=True)

        checkpoint_files = list(Path(tmpdir).glob("best-epoch*.ckpt"))
        assert checkpoint_files, "Expected at least one validation-F1 checkpoint"

        for phase in ["train", "val"]:
            phase_dir = Path(tmpdir) / phase
            conf_dir = phase_dir / "conf_mats"

            assert phase_dir.is_dir()
            assert conf_dir.is_dir()
            assert (phase_dir / "metrics_table.csv").is_file()

            for filename in [
                "loss.png",
                "accuracy.png",
                "f1 score.png",
                "precision.png",
                "recall.png",
            ]:
                assert (phase_dir / filename).is_file()

            for epoch in range(max_epochs):
                assert (conf_dir / f"epoch_{epoch}.csv").is_file()


@pytest.mark.slow_no_gpu
def test_narrowband_testing(
    narrowband_config,
    narrowband_dataloaders,
    class_names,
):
    """Test evaluation pipeline with narrowband data."""

    _, test_loader = narrowband_dataloaders

    model = torch.nn.Sequential(
        torch.nn.Conv1d(2, 16, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(16, len(class_names)),
    )

    # Deliberately leave model in train mode to verify evaluate_classifier restores it.
    model.train()
    assert model.training is True

    criterion = torch.nn.CrossEntropyLoss()

    tracker = evaluate_classifier(
        model=model,
        test_loader=test_loader,
        device=torch.device("cpu"),
        num_classes=len(class_names),
        criterion=criterion,
    )

    assert model.training is True
    assert isinstance(tracker, ClassifierMetricsTracker)

    assert set(tracker.history.keys()) == {
        "loss",
        "accuracy",
        "f1 score",
        "precision",
        "recall",
    }

    for metric_name, values in tracker.history.items():
        assert len(values) == 1
        assert isinstance(values[0], float)

    assert tracker.history["loss"][0] > 0.0
    assert len(tracker.conf_mats) == 1

    conf_mat = tracker.conf_mats[0]

    assert isinstance(conf_mat, list)
    assert len(conf_mat) == len(class_names)
    assert all(len(row) == len(class_names) for row in conf_mat)