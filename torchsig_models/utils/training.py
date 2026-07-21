import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)

from torchsig_models.utils.classifier_metrics_tracker import (
    ClassifierMetricsTracker,
    ClassifierMetricsTrackerCallback,
)


# ==============================================================
#  Deterministic‑seed helper
# ==============================================================

def set_deterministic(seed: int) -> None:
    """Configure common libraries for reproducible execution.

    Sets random seeds for Python, NumPy, and PyTorch, and configures
    CuDNN to use deterministic algorithms when available. This helps
    reduce run-to-run variability during experiments, debugging, and
    benchmarking.

    The following components are configured:

    - Python's ``random`` module
    - NumPy's global random number generator
    - PyTorch CPU random number generator
    - PyTorch CUDA random number generators
    - CuDNN deterministic mode
    - CuDNN benchmark mode (disabled)

    Note:
        This function improves reproducibility but does not guarantee
        complete determinism. Some PyTorch operations, hardware
        configurations, and distributed execution environments may
        still exhibit nondeterministic behavior.

    Args:
        seed (int):
            Random seed used to initialize all supported libraries.

    Returns:
        None
    """
    # PYTHONHASHSEED is only applied when the Python interpreter starts,
    # but setting it here ensures child processes inherit the same value.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    configure_determinism()


def configure_determinism() -> None:
    """Configure PyTorch to prefer deterministic execution.

    Enables deterministic algorithm selection where supported and
    configures CuDNN to use deterministic implementations while
    disabling benchmark mode. These settings help reduce run-to-run
    variability during training and inference.

    Note:
        Deterministic execution is not guaranteed for every operation.
        Some PyTorch operators, hardware platforms, and distributed
        execution environments may still exhibit nondeterministic
        behavior. Unsupported nondeterministic operations will emit
        warnings rather than raising exceptions because
        ``warn_only=True`` is enabled.

    Returns:
        None
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights_tensor(
    loader: torch.utils.data.DataLoader,
    num_classes: int,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from a dataset.

    Iterates through a data loader, counts the occurrences of each class,
    and returns weights proportional to the inverse of the class counts.
    These weights can be passed to loss functions such as
    ``torch.nn.CrossEntropyLoss`` to reduce the impact of class imbalance.

    Classes that do not appear in the dataset are assigned a weight of
    zero because their frequency cannot be estimated from the available
    data.

    Args:
        loader (torch.utils.data.DataLoader):
            Data loader that yields ``(input, label)`` pairs, where labels
            are integer class indices.
        num_classes (int):
            Total number of classes. Used as the minimum length of the
            class-count vector, ensuring weights are returned for every
            class even if some are absent from the dataset.

    Returns:
        torch.Tensor:
            One-dimensional tensor of shape ``(num_classes,)`` containing
            inverse-frequency class weights.

    Example:
        >>> weights = compute_class_weights_tensor(train_loader, num_classes=10)
        >>> criterion = torch.nn.CrossEntropyLoss(weight=weights)
    """
    all_labels: list[int] = []

    dataset = getattr(loader, "dataset", None)

    if dataset is not None:
        for _, (_, label) in enumerate(dataset):

            if isinstance(label, torch.Tensor):
                label = label.detach().cpu().reshape(-1).tolist()[0]
            elif isinstance(label, np.ndarray):
                label = label.reshape(-1).tolist()[0]
            elif isinstance(label, (list, tuple)):
                label = label[0]

            all_labels.append(int(label))
    else:
        for batch in loader:
            _, labels = batch

            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().reshape(-1).tolist()
            else:
                labels = np.asarray(labels, dtype=object).reshape(-1).tolist()

            for label in labels:
                if isinstance(label, (list, tuple)):
                    label = label[0]
                all_labels.append(int(label))

    class_counts = np.bincount(all_labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)

    class_weights = len(all_labels) / (num_classes * class_counts)

    return torch.tensor(class_weights, dtype=torch.float32)


def compute_num_params(model: torch.nn.Module) -> int:
    """Compute the number of trainable parameters in a model.

    Counts all parameters in the model with ``requires_grad=True`` and
    returns the total number of scalar values they contain.

    This can be useful for comparing model sizes, estimating training
    memory requirements, or reporting model complexity.

    Args:
        model (torch.nn.Module):
            Model whose trainable parameters will be counted.

    Returns:
        int:
            Total number of trainable parameters.

    Note:
        Parameters with ``requires_grad=False`` are excluded from the
        count.

    Example:
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(100, 50),
        ...     torch.nn.ReLU(),
        ...     torch.nn.Linear(50, 10),
        ... )
        >>> compute_num_params(model)
        5560
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SignalClassifier(pl.LightningModule):
    """Generic PyTorch Lightning wrapper for multiclass classification models.

    This module provides a lightweight Lightning interface around a standard
    PyTorch classification model. It handles forward propagation, loss
    computation, optimizer and scheduler configuration, and training/validation
    step execution.

    Metric computation and tracking are intentionally delegated to
    ``ClassifierMetricsTrackerCallback`` to keep training logic separate from
    evaluation and reporting functionality.

    Args:
        model:
            Neural network that produces classification logits from input
            tensors.

        criterion:
            Loss function used during training and validation, such as
            ``torch.nn.CrossEntropyLoss``.

        optimizer:
            Optimizer used to update model parameters.

        scheduler:
            Optional learning-rate scheduler. If provided, it is configured
            automatically during trainer initialization. Special handling is
            included for ``ReduceLROnPlateau``.

        clamp_logits:
            If ``True``, clamp model logits to the range [-100, 100] before
            computing loss. This can help avoid numerical instability for
            certain loss functions and extreme model outputs.

    Note:
        This class assumes a standard multiclass classification workflow where
        batches are provided as ``(inputs, targets)`` tuples and the model
        outputs raw logits.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        clamp_logits: bool = True,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            ignore=["model", "criterion", "optimizer", "scheduler"]
        )

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clamp_logits = clamp_logits

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the wrapped model.

        Args:
            x:
                Input tensor containing one or more samples.

        Returns:
            Tensor of classification logits produced by the model.
        """
        return self.model(x)

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        phase: str,
    ) -> dict[str, torch.Tensor]:
        """Execute common training and validation step logic.

        Performs a forward pass, optionally clamps logits, computes loss,
        logs the phase-specific loss metric, and returns outputs required by
        downstream callbacks.

        Args:
            batch:
                Tuple containing input tensors and target labels.

            phase:
                Phase name used for logging, typically ``"train"`` or ``"val"``.

        Returns:
            Dictionary containing:

            - ``loss``: Scalar loss tensor.
            - ``logits``: Detached model logits.
            - ``targets``: Detached target labels.
        """
        x, y = batch
        logits = self(x)

        if self.clamp_logits:
            logits = torch.clamp(logits, min=-100, max=100)

        loss = self.criterion(logits, y)

        self.log(f"{phase}_loss", loss, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "logits": logits.detach(),
            "targets": y.detach(),
        }

    # pylint: disable=arguments-differ
    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        *args,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Execute a single training step.

        Args:
            batch:
                Tuple containing inputs and target labels.

            batch_idx:
                Index of the current batch within the epoch.

        Returns:
            Dictionary containing loss, logits, and targets for use by
            callbacks and Lightning internals.
        """
        del batch_idx, args, kwargs
        return self._shared_step(batch, "train")

    # pylint: disable=arguments-differ
    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        *args,
        **kwargs,
    )  -> dict[str, torch.Tensor]:
        """Execute a single validation step.

        Args:
            batch:
                Tuple containing inputs and target labels.

            batch_idx:
                Index of the current validation batch.

        Returns:
            Dictionary containing loss, logits, and targets for use by
            callbacks and Lightning internals.
        """
        del batch_idx, args, kwargs
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        """Configure optimizer and optional learning-rate scheduler.

        Returns:
            One of the formats accepted by PyTorch Lightning:

            - Optimizer only when no scheduler is configured.
            - Optimizer and scheduler configuration dictionary when a
              scheduler is provided.
            - Special ``ReduceLROnPlateau`` configuration that monitors
              ``val_f1``.
        """
        if self.scheduler is None:
            return self.optimizer

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "val_f1",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def train_validate(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    max_epochs: int,
    num_classes: int,
    metrics_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    accelerator: str = "auto",
    devices: str | int | list[int] = "auto",
    precision: str = "32-true",
    use_distributed_sampler: bool = False,
    gradient_clip_val: float = 1.0,
    enable_progress_bar: bool = True,
    clamp_logits: bool = True,
) -> tuple[SignalClassifier, ClassifierMetricsTrackerCallback]:
    """Train and validate a multiclass classifier using PyTorch Lightning.

    This function wraps a PyTorch model in ``SignalClassifier``, configures
    metric tracking via ``ClassifierMetricsTrackerCallback``, optionally enables
    checkpointing, and executes a complete training and validation run using a
    Lightning ``Trainer``.

    Training metrics, validation metrics, confusion matrices, plots, and CSV
    exports are managed by ``ClassifierMetricsTrackerCallback`` rather than the
    Lightning module itself.

    Args:
        train_loader:
            DataLoader providing training batches.

        val_loader:
            DataLoader providing validation batches.

        model:
            PyTorch model that produces classification logits.

        criterion:
            Loss function used during training and validation.

        optimizer:
            Optimizer used to update model parameters.

        scheduler:
            Optional learning-rate scheduler. If provided, it is configured
            automatically through ``SignalClassifier``.

        max_epochs:
            Maximum number of training epochs.

        num_classes:
            Number of target classes used for metric tracking and confusion
            matrix generation.

        metrics_dir:
            Optional directory used by
            ``ClassifierMetricsTrackerCallback`` for saving plots, metric
            histories, and confusion matrices.

        checkpoint_dir:
            Optional directory for model checkpoints. When provided, the best
            checkpoint is selected using validation F1 score.

        accelerator:
            Lightning accelerator configuration. Examples include ``"cpu"``,
            ``"gpu"``, ``"mps"``, or ``"auto"``.

        devices:
            Device specification passed directly to the Lightning trainer.

        precision:
            Numerical precision setting used by Lightning. Examples include
            ``"32-true"``, ``"16-mixed"``, and ``"bf16-mixed"``.

        use_distributed_sampler:
            Whether Lightning should automatically replace dataloader samplers
            when using distributed training.

        gradient_clip_val:
            Maximum gradient norm used for gradient clipping.

        enable_progress_bar:
            Whether Lightning should display a progress bar during training.

        clamp_logits:
            If ``True``, logits are clamped to the range [-100, 100] before
            loss computation to reduce the risk of numerical instability.

    Returns:
        Tuple containing:

        - ``SignalClassifier``: The trained Lightning module.
        - ``ClassifierMetricsTrackerCallback``: Callback containing training
          and validation metric histories, confusion matrices, plotting
          utilities, and CSV export functionality.

    Notes:
        If ``checkpoint_dir`` is provided, a ``ModelCheckpoint`` callback is
        configured to monitor ``val_f1`` and retain the best-performing model
        checkpoint.
    """

    pl_model = SignalClassifier(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        clamp_logits=clamp_logits,
    )

    metrics_callback = ClassifierMetricsTrackerCallback(
        n_classes=int(num_classes),
        metrics_dir=metrics_dir,
    )

    callbacks: list[Callback] = [
        metrics_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best-epoch={epoch:03d}-val_f1={val_f1:.4f}",
                save_top_k=1,
                monitor="val_f1",
                mode="max",
            )
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        accelerator=accelerator,
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        enable_progress_bar=enable_progress_bar,
        precision=precision,
        use_distributed_sampler=use_distributed_sampler,
    )

    trainer.fit(
        pl_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    return pl_model, metrics_callback


def evaluate_classifier(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: torch.nn.Module | None = None,
) -> ClassifierMetricsTracker:
    """Evaluate a multiclass classifier on a test dataset.

    Runs inference on all samples provided by ``test_loader`` and computes
    aggregate classification metrics using ``ClassifierMetricsTracker``.
    Metrics include accuracy, precision, recall, F1 score, confusion matrix,
    and optionally average loss when a criterion is supplied.

    The model is temporarily switched to evaluation mode and moved to the
    requested device. If the model was originally in training mode, its
    training state is restored before returning.

    Args:
        model:
            PyTorch model that produces classification logits.

        test_loader:
            DataLoader providing evaluation batches as
            ``(inputs, targets)`` tuples.

        device:
            Device on which inference should be performed.

        num_classes:
            Number of target classes used when constructing evaluation
            metrics and confusion matrices.

        criterion:
            Optional loss function used to compute and record average test
            loss. If ``None``, classification metrics are still computed but
            loss history will contain a default value.

    Returns:
        ClassifierMetricsTracker containing the computed evaluation metrics,
        metric histories, and confusion matrices.

    Notes:
        - Evaluation is performed under ``torch.no_grad()``.
        - Predictions are generated using ``argmax`` over model logits.
        - Distributed metric synchronization is disabled
          (``sync_on_compute=False``) because evaluation is performed in a
          single process.
        - The returned tracker contains a single metric entry representing
          the aggregate results over the entire test dataset.
    """
    was_training = model.training

    model.eval()
    model.to(device)

    if criterion is not None:
        criterion = criterion.to(device)

    tracker = ClassifierMetricsTracker(
        n_classes=num_classes,
        device=device,
        sync_on_compute=False,
    )

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)
            loss = criterion(logits, y) if criterion is not None else None

            tracker.update(preds, y, loss)

    tracker.compute_and_store()

    if was_training:
        model.train()

    return tracker
