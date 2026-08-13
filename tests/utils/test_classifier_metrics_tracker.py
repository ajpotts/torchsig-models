import csv
from unittest.mock import MagicMock

import pytest
import torch

from matplotlib.figure import Figure

from torchsig_models.utils.classifier_metrics_tracker import (
    ClassifierMetricsTracker,
    ClassifierMetricsTrackerCallback,
)


def test_classifier_metrics_tracker_compute_and_store():
    tracker = ClassifierMetricsTracker(n_classes=3, device="cpu")

    preds = torch.tensor([0, 1, 2, 1])
    targets = torch.tensor([0, 2, 2, 1])
    loss = torch.tensor(0.75)

    tracker.update(preds, targets, loss)
    metrics = tracker.compute_and_store()

    assert set(metrics) == {"loss", "accuracy", "f1 score", "precision", "recall"}
    assert metrics["loss"] == pytest.approx(0.75)

    for value in metrics.values():
        assert isinstance(value, float)

    assert len(tracker.history["accuracy"]) == 1
    assert len(tracker.conf_mats) == 1
    assert tracker.conf_mats[0] == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
    ]


def test_classifier_metrics_tracker_resets_after_compute():
    tracker = ClassifierMetricsTracker(n_classes=2, device="cpu")

    tracker.update(
        preds=torch.tensor([0, 1]),
        targets=torch.tensor([0, 1]),
        loss=torch.tensor(0.5),
    )

    tracker.compute_and_store()

    assert tracker.batch_losses == []

    tracker.update(
        preds=torch.tensor([1]),
        targets=torch.tensor([0]),
        loss=torch.tensor(1.5),
    )

    metrics = tracker.compute_and_store()

    assert metrics["loss"] == pytest.approx(1.5)
    assert len(tracker.history["loss"]) == 2


def test_classifier_metrics_tracker_save_to_csv(tmp_path):
    tracker = ClassifierMetricsTracker(n_classes=2, device="cpu")

    tracker.update(
        preds=torch.tensor([0, 1]),
        targets=torch.tensor([0, 1]),
        loss=torch.tensor(0.25),
    )
    tracker.compute_and_store()

    tracker.save_to_csv(tmp_path)

    metrics_file = tmp_path / "metrics_table.csv"
    conf_mat_file = tmp_path / "conf_mats" / "epoch_0.csv"

    assert metrics_file.is_file()
    assert conf_mat_file.is_file()

    with metrics_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["loss", "accuracy", "f1 score", "precision", "recall"]
    assert len(rows) == 2


def test_classifier_metrics_tracker_plot_saves_expected_files(tmp_path):
    tracker = ClassifierMetricsTracker(n_classes=2, device="cpu")

    tracker.update(
        preds=torch.tensor([0, 1]),
        targets=torch.tensor([0, 1]),
        loss=torch.tensor(0.25),
    )
    tracker.compute_and_store()

    tracker.plot(save_dir=tmp_path, show=False, close=True)

    expected_files = [
        "loss.png",
        "accuracy.png",
        "f1 score.png",
        "precision.png",
        "recall.png",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).is_file()


def test_classifier_metrics_tracker_plot_old_signature_single_metric(tmp_path):
    tracker = ClassifierMetricsTracker(n_classes=2, device="cpu")

    tracker.update(
        preds=torch.tensor([0, 1]),
        targets=torch.tensor([0, 1]),
        loss=torch.tensor(0.25),
    )
    tracker.compute_and_store()

    save_file = tmp_path / "accuracy_plot.png"

    fig, axes = tracker.plot(
        metrics=["accuracy"],
        save=True,
        save_file=save_file,
        show=False,
        close=True,
    )

    assert save_file.is_file()
    assert fig is not None
    assert len(axes) == 1


def test_classifier_metrics_tracker_plot_rejects_save_file_with_multiple_metrics(
    tmp_path,
):
    tracker = ClassifierMetricsTracker(n_classes=2, device="cpu")

    with pytest.raises(ValueError, match="save_file"):
        tracker.plot(
            metrics=["accuracy", "recall"],
            save=True,
            save_file=tmp_path / "plot.png",
            show=False,
            close=True,
        )


def test_classifier_metrics_tracker_sync_on_compute_is_configurable():
    tracker = ClassifierMetricsTracker(
        n_classes=2,
        device="cpu",
        sync_on_compute=True,
    )

    assert tracker.sync_on_compute is True

    for metric in tracker.metrics.values():
        assert metric.sync_on_compute is True


def test_classifier_metrics_tracker_callback_extracts_preds_outputs():
    callback = ClassifierMetricsTrackerCallback(n_classes=2)

    outputs = {
        "loss": torch.tensor(0.5),
        "preds": torch.tensor([0, 1]),
        "targets": torch.tensor([0, 1]),
    }
    batch = (torch.randn(2, 2, 8), torch.tensor([1, 0]))

    preds, targets, loss = callback._extract_outputs(outputs, batch)

    assert torch.equal(preds, torch.tensor([0, 1]))
    assert torch.equal(targets, torch.tensor([0, 1]))
    assert loss.item() == pytest.approx(0.5)


def test_classifier_metrics_tracker_callback_extracts_logits_outputs():
    callback = ClassifierMetricsTrackerCallback(n_classes=3)

    logits = torch.tensor(
        [
            [2.0, 0.1, 0.0],
            [0.0, 0.2, 3.0],
        ]
    )
    batch_targets = torch.tensor([0, 2])

    outputs = {
        "loss": torch.tensor(0.5),
        "logits": logits,
    }
    batch = (torch.randn(2, 2, 8), batch_targets)

    preds, targets, loss = callback._extract_outputs(outputs, batch)

    assert torch.equal(preds, torch.tensor([0, 2]))
    assert torch.equal(targets, batch_targets)
    assert loss.item() == pytest.approx(0.5)


def test_classifier_metrics_tracker_callback_rejects_missing_predictions():
    callback = ClassifierMetricsTrackerCallback(n_classes=2)
    batch = (torch.randn(2, 2, 8), torch.tensor([0, 1]))

    with pytest.raises(ValueError, match="preds.*logits"):
        callback._extract_outputs({"loss": torch.tensor(0.5)}, batch)


def test_classifier_metrics_tracker_callback_updates_and_stores_train_metrics():
    callback = ClassifierMetricsTrackerCallback(n_classes=2, sync_on_compute=False)

    trainer = MagicMock()
    pl_module = MagicMock()
    pl_module.device = torch.device("cpu")

    callback.on_train_epoch_start(trainer, pl_module)

    outputs = {
        "loss": torch.tensor(0.5),
        "preds": torch.tensor([0, 1]),
        "targets": torch.tensor([0, 1]),
    }
    batch = (torch.randn(2, 2, 8), torch.tensor([0, 1]))

    callback.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)
    callback.on_train_epoch_end(trainer, pl_module)

    assert callback.train_losses == [pytest.approx(0.5)]
    assert len(callback.train_conf_mats) == 1
    assert callback.train_accuracies[0] == pytest.approx(1.0)

    pl_module.log.assert_any_call(
        "train_acc",
        pytest.approx(1.0),
        on_epoch=True,
        prog_bar=True,
        sync_dist=True,
    )


def test_classifier_metrics_tracker_callback_skips_sanity_check_validation():
    callback = ClassifierMetricsTrackerCallback(n_classes=2)

    trainer = MagicMock()
    trainer.sanity_checking = True

    pl_module = MagicMock()
    pl_module.device = torch.device("cpu")

    callback.on_validation_epoch_start(trainer, pl_module)

    outputs = {
        "loss": torch.tensor(0.5),
        "preds": torch.tensor([0, 1]),
        "targets": torch.tensor([0, 1]),
    }
    batch = (torch.randn(2, 2, 8), torch.tensor([0, 1]))

    callback.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)
    callback.on_validation_epoch_end(trainer, pl_module)

    assert callback.val_losses == []
    assert callback.val_conf_mats == []


def test_classifier_metrics_tracker_callback_save_and_plot(tmp_path):
    callback = ClassifierMetricsTrackerCallback(
        n_classes=2,
        metrics_dir=tmp_path,
        sync_on_compute=False,
    )

    trainer = MagicMock()
    trainer.sanity_checking = False

    pl_module = MagicMock()
    pl_module.device = torch.device("cpu")

    outputs = {
        "loss": torch.tensor(0.5),
        "preds": torch.tensor([0, 1]),
        "targets": torch.tensor([0, 1]),
    }
    batch = (torch.randn(2, 2, 8), torch.tensor([0, 1]))

    callback.on_train_epoch_start(trainer, pl_module)
    callback.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)
    callback.on_train_epoch_end(trainer, pl_module)

    callback.on_validation_epoch_start(trainer, pl_module)
    callback.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)
    callback.on_validation_epoch_end(trainer, pl_module)

    callback.save_to_csv()
    callback.plot(show=False, close=True)

    for phase in ["train", "val"]:
        phase_dir = tmp_path / phase

        assert (phase_dir / "metrics_table.csv").is_file()
        assert (phase_dir / "conf_mats" / "epoch_0.csv").is_file()
        assert (phase_dir / "accuracy.png").is_file()
        assert (phase_dir / "f1 score.png").is_file()
        assert (phase_dir / "loss.png").is_file()
        assert (phase_dir / "precision.png").is_file()
        assert (phase_dir / "recall.png").is_file()


def test_classifier_metrics_tracker_plot_confusion_matrix(tmp_path):
    tracker = ClassifierMetricsTracker(n_classes=3, device="cpu")
    tracker.conf_mats.append(
        [
            [2, 1, 0],
            [0, 3, 1],
            [1, 0, 4],
        ]
    )

    save_file = tmp_path / "confusion_matrix.png"

    fig = tracker.plot_confusion_matrix(
        save_file=save_file,
        class_names=["a", "b", "c"],
        normalize=False,
        show=False,
        close=True,
    )

    assert isinstance(fig, Figure)
    assert save_file.exists()
    assert save_file.stat().st_size > 0

    ax = fig.axes[0]
    assert ax.get_xlabel() == "Predicted"
    assert ax.get_ylabel() == "True"
    assert ax.get_title() == "Confusion Matrix"
