"""Training entry point for DETR on TorchSig wideband spectrogram datasets."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import Logger
from torch import nn
from torchsig.datasets.datasets import TorchSigDatasetConfig
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.yaml import load_config_from_yaml

from torchsig_models.models.spectrogram_models.detr import (
    detr_b0_nano,
    detr_b2_nano,
    detr_b4_nano,
)
from torchsig_models.models.spectrogram_models.detr.modules import SetCriterion
from torchsig_models.models.spectrogram_models.efficientnet.efficientnet import (
    SpectrogramNormalization,
)
from torchsig_models.utils.datasets import prepare_torchsig_datasets
from torchsig_models.utils.training import compute_num_params, set_deterministic

DETRModelName = Literal["detr_b0_nano", "detr_b2_nano", "detr_b4_nano"]
MODEL_FACTORY = {
    "detr_b0_nano": detr_b0_nano,
    "detr_b2_nano": detr_b2_nano,
    "detr_b4_nano": detr_b4_nano,
}

__all__ = [
    "DETRModelName",
    "MODEL_FACTORY",
    "DETRDetector",
    "detr_collate",
    "load_training_params",
    "train_detr",
]


def load_training_params(
    model_name: DETRModelName,
    params_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load DETR training parameters from YAML."""
    path = (
        Path(params_path)
        if params_path is not None
        else Path(__file__).parent / "training_params" / f"{model_name}.yaml"
    )
    if not path.exists():
        raise FileNotFoundError(f"Training parameter file not found: {path}")
    with path.open("r", encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)
    if not isinstance(params, dict):
        raise ValueError(f"Training parameter file must contain a mapping: {path}")
    return params


def _spectrogram_transforms(cfg: TorchSigDatasetConfig) -> list[Any]:
    """Create spectrogram and multi-object label transforms."""
    fft_size = getattr(cfg, "output_spectrogram_fft", None)
    if fft_size is None:
        fft_size = getattr(cfg, "dataset_metadata", {}).get("fft_size", 256)
    return [Spectrogram(fft_size=int(fft_size)), YOLOLabel()]


def _image_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(image, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(
            "Expected a spectrogram with shape [frequency, time] or "
            f"[channels, frequency, time], got {tuple(tensor.shape)}."
        )
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(2, 1, 1)
    if tensor.shape[0] != 2:
        raise ValueError(f"DETR expects one or two input channels, got {tensor.shape[0]}.")
    return tensor


def detr_collate(
    batch: Sequence[tuple[np.ndarray | torch.Tensor, Iterable[Sequence[float]]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Collate every signal in each sample into a DETR target set.

    TorchSig YOLO labels have the form ``[class, center_time, center_frequency,
    duration, bandwidth]``. No object is discarded, so configurations with
    ``num_signals_max > 1`` train as genuine multi-signal wideband detection.
    """
    images, yolo_targets = zip(*batch)
    image_batch = torch.stack([_image_tensor(image) for image in images])
    image_batch = SpectrogramNormalization()(image_batch)

    targets: list[dict[str, torch.Tensor]] = []
    for objects in yolo_targets:
        values = torch.as_tensor(list(objects), dtype=torch.float32).reshape(-1, 5)
        targets.append(
            {
                "labels": values[:, 0].to(dtype=torch.int64),
                "boxes": values[:, 1:],
            }
        )
    return image_batch, targets


def _weighted_loss(
    losses: dict[str, torch.Tensor], criterion: SetCriterion
) -> torch.Tensor:
    return sum(
        losses[name] * weight
        for name, weight in criterion.weight_dict.items()
        if name in losses
    )


class DETRDetector(pl.LightningModule):
    """Lightning wrapper for DETR set-based wideband detection."""

    def __init__(
        self,
        model: nn.Module,
        *,
        num_classes: int,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        class_loss_coef: float = 1.0,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        eos_coef: float = 0.1,
        model_name: str | None = None,
        class_names: list[str] | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = SetCriterion(
            num_classes=num_classes,
            class_loss_coef=class_loss_coef,
            bbox_loss_coef=bbox_loss_coef,
            giou_loss_coef=giou_loss_coef,
            eos_coef=eos_coef,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.save_hyperparameters(ignore=["model"])

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict class logits and normalized boxes."""
        return self.model(images)

    def _step(self, batch: Any, stage: str) -> torch.Tensor:
        images, targets = batch
        losses = self.criterion(self(images), targets)
        loss = _weighted_loss(losses, self.criterion)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        for name, value in losses.items():
            if name in self.criterion.weight_dict:
                self.log(f"{stage}_{name}", value, on_epoch=True, batch_size=images.shape[0])
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Compute one training batch loss."""
        return self._step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Compute one validation batch loss."""
        return self._step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Compute one test batch loss."""
        return self._step(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW with cosine annealing."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.max_epochs, 1)
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train_detr(
    train_cfg: TorchSigDatasetConfig,
    val_cfg: TorchSigDatasetConfig,
    test_cfg: TorchSigDatasetConfig,
    params: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    dataset_root: str | Path = "datasets",
    overwrite: bool = False,
    model_name: DETRModelName = "detr_b0_nano",
    signal_generators: str | list[str] = "all",
    logger: Logger | bool | None = True,
    accelerator: str = "auto",
    devices: int | str | list[int] = "auto",
) -> dict[str, Any]:
    """Train and evaluate DETR on multi-signal TorchSig wideband data."""
    set_deterministic(int(train_cfg.seed))
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    loaders = prepare_torchsig_datasets(
        train_cfg,
        val_cfg,
        test_cfg,
        dataset_root=dataset_root,
        batch_size=int(params["batch_size"]),
        overwrite=overwrite,
        signal_generators=signal_generators,
        transforms=_spectrogram_transforms(train_cfg),
        target_labels=["yolo_label"],
        collate_fn=detr_collate,
    )
    train_loader, val_loader, test_loader, data_info = loaders
    class_names = data_info["class_names"]
    num_classes = len(class_names)
    model_params = {
        "drop_rate_backbone": params.get("drop_rate_backbone", 0.2),
        "drop_path_rate_backbone": params.get("drop_path_rate_backbone", 0.2),
        "drop_path_rate_transformer": params.get("drop_path_rate_transformer", 0.1),
    }
    detector = DETRDetector(
        MODEL_FACTORY[model_name](num_classes=num_classes, **model_params),
        num_classes=num_classes,
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        max_epochs=int(params["max_epochs"]),
        class_loss_coef=float(params.get("class_loss_coef", 1.0)),
        bbox_loss_coef=float(params.get("bbox_loss_coef", 5.0)),
        giou_loss_coef=float(params.get("giou_loss_coef", 2.0)),
        eos_coef=float(params.get("eos_coef", 0.1)),
        model_name=model_name,
        class_names=class_names,
        model_params=model_params,
    )
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=int(params["max_epochs"]),
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[checkpoint],
        # CUDA's nll_loss2d kernel, used by DETR's set criterion, does not
        # currently provide a deterministic implementation. Keep deterministic
        # algorithms enabled where available, but warn instead of aborting the
        # entire training or tuning run for that operation.
        deterministic="warn",
    )
    trainer.fit(detector, train_loader, val_loader)
    val_loss = float(trainer.callback_metrics["val_loss"].detach().cpu())
    test_results = trainer.test(detector, test_loader)
    return {
        "pl_model": detector,
        "model": detector.model,
        "trainer": trainer,
        "test_results": test_results,
        "val_loss": val_loss,
        "num_classes": num_classes,
        "num_params": compute_num_params(detector.model),
        "data_info": data_info,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "best_checkpoint": checkpoint.best_model_path,
    }


def _parse_devices(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    """Parse DETR training command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--val-config", type=Path)
    parser.add_argument("--test-config", type=Path)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--model", choices=list(MODEL_FACTORY), default="detr_b0_nano")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--dataset-length", type=int)
    parser.add_argument("--dataset-id")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--signal-generators", nargs="+", default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--accelerator", default="auto", choices=["auto", "cpu", "gpu", "mps"])
    parser.add_argument("--devices", type=_parse_devices, default="auto")
    args = parser.parse_args()
    if args.dataset_config is None and args.train_config is None:
        parser.error("one of --dataset-config or --train-config is required")
    return args


def _load_configs(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    train_path = args.train_config or args.dataset_config
    assert train_path is not None
    train_cfg = load_config_from_yaml(train_path)
    val_cfg = load_config_from_yaml(args.val_config or args.dataset_config or train_path)
    test_cfg = load_config_from_yaml(args.test_config or args.dataset_config or train_path)
    if args.val_config is None:
        val_cfg = replace(val_cfg, seed=train_cfg.seed + 1)
    if args.test_config is None:
        test_cfg = replace(test_cfg, seed=train_cfg.seed + 2)
    updates = {
        name: value
        for name, value in {"dataset_length": args.dataset_length, "dataset_id": args.dataset_id}.items()
        if value is not None
    }
    if updates:
        train_cfg, val_cfg, test_cfg = (replace(cfg, **updates) for cfg in (train_cfg, val_cfg, test_cfg))
    return train_cfg, val_cfg, test_cfg


def main() -> None:
    """Run DETR training from the command line."""
    args = parse_args()
    train_cfg, val_cfg, test_cfg = _load_configs(args)
    params = load_training_params(args.model, args.params)
    if args.epochs is not None:
        params["max_epochs"] = args.epochs
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    run_dir = args.output_dir / train_cfg.dataset_id / args.model
    result = train_detr(
        train_cfg,
        val_cfg,
        test_cfg,
        params,
        run_dir / "checkpoints",
        dataset_root=args.dataset_root,
        overwrite=args.overwrite,
        model_name=args.model,
        signal_generators=args.signal_generators,
        accelerator=args.accelerator,
        devices=args.devices,
    )
    print(f"Final validation loss: {result['val_loss']:.6f}")
    print(f"Best checkpoint: {result['best_checkpoint']}")


if __name__ == "__main__":
    main()
