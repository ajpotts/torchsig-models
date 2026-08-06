"""Utilities for tracking and visualizing classifier performance.

This module provides two classes:

* ``ClassifierMetricsTracker`` accumulates classification metrics for a
  single phase (training, validation, or testing), stores per-epoch
  histories, exports metrics to CSV, and generates plots.
* ``ClassifierMetricsTrackerCallback`` integrates metric tracking into a
  PyTorch Lightning training loop by automatically collecting predictions,
  computing epoch metrics, logging them, and optionally saving results to
  disk.
"""

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall


ENCODING = "utf-8"


class ClassifierMetricsTracker:
    """Track metrics for one classifier phase, e.g. train, val, or test."""

    def __init__(
        self,
        n_classes: int,
        averaging: dict[str, str] | None = None,
        device: torch.device | str | None = None,
        sync_on_compute: bool = False,
    ) -> None:
        """Initialize a classifier metrics tracker.

        Args:
            n_classes: Number of target classes.
            averaging: Averaging strategy for each metric. Keys may include
                ``"accuracy"``, ``"f1 score"``, ``"precision"``, and
                ``"recall"``. Any omitted metrics default to ``"macro"``.
            device: Device on which torchmetrics objects are allocated. If
                ``None``, uses CUDA when available, otherwise CPU.
            sync_on_compute: Whether to synchronize metric state across
                distributed workers when computing metric values.

        Raises:
            TypeError: If ``n_classes`` is not an integer.
        """
        if not isinstance(n_classes, int):
            raise TypeError(f"`n_classes` must be an int, got {type(n_classes)}")

        self.n_classes = n_classes
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.sync_on_compute = sync_on_compute

        averaging = averaging or {}
        self.averaging = {
            "accuracy": averaging.get("accuracy", "macro"),
            "f1 score": averaging.get("f1 score", "macro"),
            "precision": averaging.get("precision", "macro"),
            "recall": averaging.get("recall", "macro"),
        }

        self.history: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "f1 score": [],
            "precision": [],
            "recall": [],
        }
        self.losses: list[float] = []
        self.conf_mats: list[list[list[int]]] = []
        self.batch_losses: list[torch.Tensor] = []

        self.metrics: dict[str, torch.nn.Module] = {}
        self.initialize_holders(self.device)

    def initialize_holders(self, device: torch.device | str | None = None) -> None:
        """Initialize or reset the metric accumulators.

        Creates fresh torchmetrics objects for the configured device and
        clears any batch losses accumulated during the current epoch.

        Args:
            device: Device on which the metric objects should be created.
                If ``None``, the previously configured device is used.
        """
        if device is not None:
            self.device = torch.device(device)

        self.metrics = {
            "cm": ConfusionMatrix(
                task="multiclass",
                num_classes=self.n_classes,
                sync_on_compute=self.sync_on_compute,
            ).to(self.device),
            "accuracy": Accuracy(
                task="multiclass",
                num_classes=self.n_classes,
                average=self.averaging["accuracy"],
                sync_on_compute=self.sync_on_compute,
            ).to(self.device),
            "f1 score": F1Score(
                task="multiclass",
                num_classes=self.n_classes,
                average=self.averaging["f1 score"],
                sync_on_compute=self.sync_on_compute,
            ).to(self.device),
            "precision": Precision(
                task="multiclass",
                num_classes=self.n_classes,
                average=self.averaging["precision"],
                sync_on_compute=self.sync_on_compute,
            ).to(self.device),
            "recall": Recall(
                task="multiclass",
                num_classes=self.n_classes,
                average=self.averaging["recall"],
                sync_on_compute=self.sync_on_compute,
            ).to(self.device),
        }

        self.batch_losses.clear()

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor | None = None,
    ) -> None:
        """Update all metrics with a single batch.

        Args:
            preds: Predicted class indices for the batch.
            targets: Ground-truth class labels.
            loss: Optional batch loss to include when computing the epoch
                average loss.
        """
        preds = preds.detach().to(self.device)
        targets = targets.detach().to(self.device)

        for metric in self.metrics.values():
            metric.update(preds, targets)

        if loss is not None:
            self.batch_losses.append(loss.detach())

    def compute_and_store(self) -> dict[str, float]:
        """Compute epoch metrics and store them in the history.

        Computes the current values for all tracked metrics, appends them to
        the per-epoch history, stores the confusion matrix, resets the
        internal metric state, and returns the computed values.

        Returns:
            Dictionary containing the epoch loss, accuracy, F1 score,
            precision, and recall.
        """
        loss = self._mean_loss()

        values = {
            "loss": loss,
            "accuracy": float(self.metrics["accuracy"].compute().detach().cpu()),
            "f1 score": float(self.metrics["f1 score"].compute().detach().cpu()),
            "precision": float(self.metrics["precision"].compute().detach().cpu()),
            "recall": float(self.metrics["recall"].compute().detach().cpu()),
        }

        conf_mat = self.metrics["cm"].compute().detach().cpu().int().tolist()

        for key, value in values.items():
            self.history[key].append(value)

        self.losses.append(loss)
        self.conf_mats.append(conf_mat)

        self.reset()

        return values

    def reset(self) -> None:
        """Reset all metric accumulators for a new epoch."""
        for metric in self.metrics.values():
            metric.reset()

        self.batch_losses.clear()

    def _mean_loss(self) -> float:
        """Compute the mean batch loss for the current epoch.

        Returns:
            Average loss across all recorded batches. Returns ``0.0`` if no
            losses have been recorded.
        """
        if not self.batch_losses:
            return 0.0

        return float(
            torch.stack([loss.to(self.device) for loss in self.batch_losses])
            .mean()
            .detach()
            .cpu()
        )

    def save_to_csv(self, metrics_dir: str | Path) -> None:
        """Write stored metrics to CSV files.

        Saves scalar metrics to ``metrics_table.csv`` and writes one confusion
        matrix CSV per epoch in a ``conf_mats`` subdirectory.

        Args:
            metrics_dir: Directory in which the CSV files are written.
        """
        metrics_dir = Path(metrics_dir)
        conf_dir = metrics_dir / "conf_mats"
        conf_dir.mkdir(parents=True, exist_ok=True)

        with (metrics_dir / "metrics_table.csv").open(
            "w",
            newline="",
            encoding=ENCODING,
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["loss", "accuracy", "f1 score", "precision", "recall"])

            rows = zip(
                self.history["loss"],
                self.history["accuracy"],
                self.history["f1 score"],
                self.history["precision"],
                self.history["recall"],
            )
            writer.writerows(rows)

        for epoch, conf_mat in enumerate(self.conf_mats):
            with (conf_dir / f"epoch_{epoch}.csv").open(
                "w",
                newline="",
                encoding=ENCODING,
            ) as csvfile:
                csv.writer(csvfile).writerows(conf_mat)

    def plot(
        self,
        metrics: list[str] | None = None,
        save: bool = False,
        save_file: str | Path | None = None,
        save_dir: str | Path | None = None,
        prefix: str = "",
        show: bool = True,
        close: bool = False,
    ) -> tuple[Figure, list[Axes]]:
        """Plot one or more tracked metrics.

        Generates a separate figure for each requested metric.

        Args:
            metrics: Metrics to plot. If omitted, all tracked metrics are
                plotted. Common aliases such as ``"acc"`` and ``"f1"`` are
                accepted.
            save: Whether to save generated figures.
            save_file: Output filename when plotting a single metric.
            save_dir: Directory in which all figures should be saved.
            prefix: Prefix added to each plot title.
            show: Whether to display the figures.
            close: Whether to close figures after plotting.

        Returns:
            A tuple containing the first generated figure and the list of
            axes corresponding to all generated plots.

        Raises:
            ValueError: If ``save_file`` is provided while plotting multiple
                metrics.
        """
        metric_map = {
            "loss": "loss",
            "accuracy": "accuracy",
            "acc": "accuracy",
            "f1 score": "f1 score",
            "f1": "f1 score",
            "precision": "precision",
            "prec": "precision",
            "recall": "recall",
            "rec": "recall",
        }

        if metrics is None or metrics == []:
            metrics = ["loss", "accuracy", "f1 score", "precision", "recall"]

        metrics = [metric_map[metric] for metric in metrics]

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        if save and save_file is not None and len(metrics) > 1:
            raise ValueError("`save_file` can only be used when plotting one metric.")

        figs: list[Figure] = []
        axes: list[Axes] = []

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(8, 5))
            figs.append(fig)
            axes.append(ax)

            values = self.history[metric]
            ax.plot(range(1, len(values) + 1), values, marker="o")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.set_title(f"{prefix}{metric} over epochs")
            ax.grid(True)
            fig.tight_layout()

            filename = f"{metric}.png"

            if save_dir is not None:
                fig.savefig(save_dir / filename, bbox_inches="tight")

            if save:
                if save_file is not None:
                    fig.savefig(Path(save_file), bbox_inches="tight")
                else:
                    fig.savefig(Path(filename), bbox_inches="tight")

            if show:
                plt.show()

            if close:
                plt.close(fig)

        return figs[0], axes

    def plot_confusion_matrix(
        self,
        epoch: int = -1,
        save_file: str | Path | None = None,
        class_names: list[str] | None = None,
        normalize: bool = False,
        show: bool = True,
        close: bool = False,
    ) -> Figure:
        """Plot a stored confusion matrix.

        Args:
            epoch: Epoch index of the confusion matrix to display. Defaults
                to the most recent epoch.
            save_file: Optional output image filename.
            class_names: Optional class labels for the axes.
            normalize: Whether to normalize each row by the number of
                ground-truth samples.
            show: Whether to display the figure.
            close: Whether to close the figure after plotting.

        Returns:
            The generated Matplotlib figure.
        """

        cm = np.array(self.conf_mats[epoch], dtype=float)

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(
                cm,
                row_sums,
                out=np.zeros_like(cm),
                where=row_sums != 0,
            )

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(cm)

        plt.colorbar(im, ax=ax)

        if class_names is not None:
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=90)
            ax.set_yticklabels(class_names)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(
            "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"
        )

        fig.tight_layout()

        if save_file is not None:
            fig.savefig(save_file, bbox_inches="tight")

        if show:
            plt.show()

        if close:
            plt.close(fig)

        return fig


class ClassifierMetricsTrackerCallback(Callback):
    """Lightning callback that tracks train and validation classifier metrics."""

    def __init__(
        self,
        n_classes: int,
        metrics_dir: str | Path | None = None,
        averaging: dict[str, str] | None = None,
        save_on_epoch_end: bool = True,
        sync_on_compute: bool = True,
    ) -> None:
        """Initialize a Lightning callback for classifier metric tracking.

        Args:
            n_classes: Number of target classes.
            metrics_dir: Directory used for saving metrics and plots.
            averaging: Averaging strategy for each tracked metric.
            save_on_epoch_end: Whether metrics should automatically be written
                to disk after each epoch.
            sync_on_compute: Whether torchmetrics should synchronize state
                across distributed workers.
        """
        super().__init__()

        self.metrics_dir = Path(metrics_dir) if metrics_dir is not None else None
        self.save_on_epoch_end = save_on_epoch_end
        self.sync_on_compute = sync_on_compute

        self.train_metrics = ClassifierMetricsTracker(
            n_classes,
            averaging=averaging,
            sync_on_compute=sync_on_compute,
        )
        self.val_metrics = ClassifierMetricsTracker(
            n_classes,
            averaging=averaging,
            sync_on_compute=sync_on_compute,
        )

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Initialize training metric accumulators for the current epoch.

        Recreates the underlying torchmetrics objects on the Lightning module's
        current device before training begins.
        """
        self.train_metrics.initialize_holders(pl_module.device)

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Initialize validation metric accumulators for the current epoch.

        Recreates the underlying torchmetrics objects on the Lightning module's
        current device before validation begins.
        """
        self.val_metrics.initialize_holders(pl_module.device)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: tuple[Any, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Update training metrics using the outputs from one training batch.

        Extracts predictions, targets, and loss from the Lightning step output
        and updates the training metric tracker.
        """
        preds, targets, loss = self._extract_outputs(outputs, batch)
        self.train_metrics.update(preds, targets, loss)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: tuple[Any, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update validation metrics using the outputs from one validation batch.

        Extracts predictions, targets, and loss from the Lightning step output
        and updates the validation metric tracker.
        """
        preds, targets, loss = self._extract_outputs(outputs, batch)
        self.val_metrics.update(preds, targets, loss)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Compute, log, and optionally save training metrics for the epoch."""
        metrics = self.train_metrics.compute_and_store()
        self._log_metrics(pl_module, "train", metrics)

        if self.save_on_epoch_end and self.metrics_dir is not None:
            self.save_to_csv()

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Compute, log, and optionally save validation metrics for the epoch.

        Metrics are not computed during Lightning's validation sanity check.
        """
        if trainer.sanity_checking:
            return

        metrics = self.val_metrics.compute_and_store()
        self._log_metrics(pl_module, "val", metrics)

        if self.save_on_epoch_end and self.metrics_dir is not None:
            self.save_to_csv()

    def _extract_outputs(
        self,
        outputs: dict[str, Any],
        batch: tuple[Any, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Extract predictions, targets, and loss from a Lightning step.

        Supports step outputs containing either ``preds`` or ``logits``.
        Targets default to those provided by the batch unless overridden by
        the step output.

        Returns:
            Tuple containing predictions, targets, and optional loss.

        Raises:
            ValueError: If the step output is missing predictions.
        """
        if outputs is None:
            raise ValueError(
                "Expected Lightning step to return a dict containing `preds` or `logits`."
            )

        _, batch_targets = batch

        targets = outputs.get("targets", batch_targets)
        loss = outputs.get("loss")

        if "preds" in outputs:
            preds = outputs["preds"]
        elif "logits" in outputs:
            preds = outputs["logits"].argmax(dim=1)
        else:
            raise ValueError("Expected step output to contain `preds` or `logits`.")

        return preds, targets, loss

    def _log_metrics(
        self,
        pl_module: LightningModule,
        phase: str,
        metrics: dict[str, float],
    ) -> None:
        """Log computed epoch metrics to the Lightning module.

        Args:
            pl_module: Lightning module used for logging.
            phase: Metric prefix (for example, ``"train"`` or ``"val"``).
            metrics: Dictionary of computed metric values.
        """
        pl_module.log(f"{phase}_loss", metrics["loss"], on_epoch=True, prog_bar=True, sync_dist=True)
        pl_module.log(f"{phase}_acc", metrics["accuracy"], on_epoch=True, prog_bar=True, sync_dist=True)
        pl_module.log(f"{phase}_f1", metrics["f1 score"], on_epoch=True, prog_bar=True, sync_dist=True)
        pl_module.log(f"{phase}_prec", metrics["precision"], on_epoch=True, sync_dist=True)
        pl_module.log(f"{phase}_rec", metrics["recall"], on_epoch=True, sync_dist=True)


    def save_to_csv(self, metrics_dir: str | Path | None = None) -> None:
        """Save both training and validation metrics to disk.

        Args:
            metrics_dir: Output directory. If omitted, uses the directory
                provided during initialization.

        Raises:
            ValueError: If no output directory has been configured.
        """
        output_dir = Path(metrics_dir) if metrics_dir is not None else self.metrics_dir

        if output_dir is None:
            raise ValueError("`metrics_dir` must be set before saving.")

        self.train_metrics.save_to_csv(output_dir / "train")
        self.val_metrics.save_to_csv(output_dir / "val")

    def plot(
        self,
        save_dir: str | Path | None = None,
        show: bool = True,
        close: bool = False,
    ) -> None:
        """Generate plots for both training and validation metrics.

        Args:
            save_dir: Directory in which figures should be saved.
            show: Whether to display the figures.
            close: Whether to close figures after plotting.

        Raises:
            ValueError: If no output directory is available.
        """
        output_dir = Path(save_dir) if save_dir is not None else self.metrics_dir

        if output_dir is None:
            raise ValueError("`save_dir` or `metrics_dir` must be provided.")

        self.train_metrics.plot(
            save_dir=output_dir / "train",
            prefix="train ",
            show=show,
            close=close,
        )
        self.val_metrics.plot(
            save_dir=output_dir / "val",
            prefix="val ",
            show=show,
            close=close,
        )

    @property
    def train_losses(self) -> list[float]:
        """Training loss recorded for each completed epoch."""
        return self.train_metrics.history["loss"]

    @property
    def val_losses(self) -> list[float]:
        """Validation loss recorded for each completed epoch."""
        return self.val_metrics.history["loss"]

    @property
    def train_conf_mats(self) -> list[list[list[int]]]:
        """Training confusion matrix for each completed epoch."""
        return self.train_metrics.conf_mats

    @property
    def val_conf_mats(self) -> list[list[list[int]]]:
        """Validation confusion matrix for each completed epoch."""
        return self.val_metrics.conf_mats

    @property
    def train_accuracies(self) -> list[float]:
        """Training accuracy recorded for each completed epoch."""
        return self.train_metrics.history["accuracy"]

    @property
    def val_accuracies(self) -> list[float]:
        """Validation accuracy recorded for each completed epoch."""
        return self.val_metrics.history["accuracy"]

    @property
    def train_f1s(self) -> list[float]:
        """Training F1 score recorded for each completed epoch."""
        return self.train_metrics.history["f1 score"]

    @property
    def val_f1s(self) -> list[float]:
        """Validation F1 score recorded for each completed epoch."""
        return self.val_metrics.history["f1 score"]
