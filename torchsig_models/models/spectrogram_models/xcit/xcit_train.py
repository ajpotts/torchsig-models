"""Training entry point for XCiT spectrogram classifiers."""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml
from pytorch_lightning.loggers import Logger
from torchsig.datasets.datasets import TorchSigDatasetConfig
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.utils.yaml import load_config_from_yaml

from torchsig_models.models.spectrogram_models.efficientnet.efficientnet_train import (
    _build_scheduler,
    _spectrogram_transforms,
    _validate_single_signal_config,
)
from torchsig_models.models.spectrogram_models.xcit import xcit_nano
from torchsig_models.utils.datasets import prepare_torchsig_datasets
from torchsig_models.utils.training import (
    compute_num_params,
    evaluate_classifier,
    set_deterministic,
    train_validate,
)

__all__ = ["load_training_params", "train_xcit_2d"]


def load_training_params(params_path: str | Path | None = None) -> dict[str, Any]:
    """Load XCiT training parameters from YAML."""
    path = (
        Path(params_path)
        if params_path is not None
        else Path(__file__).parent / "training_params" / "xcit_nano.yaml"
    )
    if not path.exists():
        raise FileNotFoundError(f"Training parameter file not found: {path}")
    with path.open("r", encoding="utf-8") as params_file:
        return yaml.safe_load(params_file)


def train_xcit_2d(
    train_cfg: TorchSigDatasetConfig,
    val_cfg: TorchSigDatasetConfig,
    test_cfg: TorchSigDatasetConfig,
    params: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    metrics_dir: str | Path | None = None,
    dataset_root: str | Path = "datasets",
    overwrite: bool = False,
    signal_generators: str | list[str] = "all",
    logger: Logger | bool | None = True,
    accelerator: str = "auto",
    devices: int | str | list[int] = "auto",
) -> dict[str, Any]:
    """Train and evaluate XCiT-Nano on TorchSig spectrogram datasets."""
    for cfg, split in ((train_cfg, "training"), (val_cfg, "validation"), (test_cfg, "test")):
        _validate_single_signal_config(cfg, split)
    set_deterministic(int(train_cfg.seed))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(metrics_dir) if metrics_dir else checkpoint_dir / "metrics"
    train_loader, val_loader, test_loader, data_info = prepare_torchsig_datasets(
        train_cfg,
        val_cfg,
        test_cfg,
        dataset_root=dataset_root,
        batch_size=params["batch_size"],
        overwrite=overwrite,
        signal_generators=signal_generators,
        transforms=_spectrogram_transforms(train_cfg),
    )
    num_classes = (
        len(signal_generators)
        if isinstance(signal_generators, list)
        else len(TorchSigSignalLists.all_signals)
    )
    model = xcit_nano(
        num_classes=num_classes,
        input_channels=params.get("input_channels", 1),
        drop_path_rate=params.get("drop_path", 0.2),
        drop_rate=params.get("drop_rate", 0.3),
        normalize=params.get("normalize", False),
    )
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=params.get("label_smoothing", 0.0)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = _build_scheduler(optimizer, params["max_epochs"])
    pl_model, metrics = train_validate(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        max_epochs=params["max_epochs"],
        num_classes=num_classes,
        metrics_dir=metrics_dir,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_metrics = evaluate_classifier(
        model=pl_model.model,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        criterion=criterion,
    )
    test_metrics.save_to_csv(metrics_dir / "test")
    return {
        "pl_model": pl_model,
        "model": pl_model.model,
        "metrics": metrics,
        "test_metrics": test_metrics,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_classes": num_classes,
        "num_params": compute_num_params(pl_model.model),
        "data_info": data_info,
    }


def parse_args() -> argparse.Namespace:
    """Parse XCiT training command-line arguments."""
    parser = argparse.ArgumentParser(description="Train XCiT-Nano on TorchSig spectrograms.")
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--val-config", type=Path)
    parser.add_argument("--test-config", type=Path)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--dataset-length", type=int)
    parser.add_argument("--dataset-id")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--accelerator", default="auto", choices=["auto", "cpu", "gpu", "mps"])
    parser.add_argument("--devices", type=_parse_devices, default="auto")
    return parser.parse_args()


def _parse_devices(value: str) -> int | str:
    """Parse a Lightning device count or symbolic selection."""
    return int(value) if value.isdigit() else value


def _resolve_config_path(
    dataset_config: Path | None,
    split_config: Path | None,
    split: str,
) -> Path:
    """Resolve a shared or split-specific dataset configuration path."""
    if split_config is not None:
        return split_config
    if dataset_config is not None:
        return dataset_config
    raise ValueError(f"Must provide either --dataset-config or --{split}-config.")


def main() -> None:
    """Train XCiT-Nano from command-line configuration."""
    args = parse_args()
    train_path = _resolve_config_path(
        args.dataset_config, args.train_config, "train"
    )
    val_path = _resolve_config_path(args.dataset_config, args.val_config, "val")
    test_path = _resolve_config_path(args.dataset_config, args.test_config, "test")
    train_cfg = load_config_from_yaml(train_path)
    val_cfg = load_config_from_yaml(val_path)
    test_cfg = load_config_from_yaml(test_path)
    if train_path == val_path == test_path:
        val_cfg = replace(val_cfg, seed=train_cfg.seed + 1)
        test_cfg = replace(test_cfg, seed=train_cfg.seed + 2)
    if args.dataset_length is not None:
        train_cfg = replace(train_cfg, dataset_length=args.dataset_length)
        val_cfg = replace(val_cfg, dataset_length=args.dataset_length)
        test_cfg = replace(test_cfg, dataset_length=args.dataset_length)
    if args.dataset_id is not None:
        train_cfg = replace(train_cfg, dataset_id=args.dataset_id)
        val_cfg = replace(val_cfg, dataset_id=args.dataset_id)
        test_cfg = replace(test_cfg, dataset_id=args.dataset_id)
    params = load_training_params(args.params)
    if args.epochs is not None:
        params["max_epochs"] = args.epochs
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    run_dir = args.output_dir / train_cfg.dataset_id / "xcit_nano"
    train_xcit_2d(
        train_cfg,
        val_cfg,
        test_cfg,
        params,
        run_dir / "checkpoints",
        metrics_dir=run_dir / "metrics",
        dataset_root=args.dataset_root,
        overwrite=args.overwrite,
        accelerator=args.accelerator,
        devices=args.devices,
    )


if __name__ == "__main__":
    main()
