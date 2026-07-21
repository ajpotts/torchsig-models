"""Training entry points for EfficientNet-1D classifiers on TorchSig IQ datasets.

This module loads TorchSig dataset configurations, prepares train/validation/test
data loaders, builds an EfficientNet-1D classifier, and trains it with the shared
TorchSig Lightning training utilities.

It can be imported as a library or executed as a command-line training script.
"""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from torchsig.datasets.datasets import TorchSigDatasetConfig
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.utils.yaml import load_config_from_yaml

from torchsig_models.models.iq_models.efficientnet import (
    efficientnet_b0,
    efficientnet_b2,
    efficientnet_b4,
)
from torchsig_models.utils.datasets import prepare_torchsig_datasets
from torchsig_models.utils.training import (
    compute_num_params,
    evaluate_classifier,
    set_deterministic,
    train_validate,
)

__all__ = [
    "EfficientNetModelName",
    "MODEL_FACTORY",
    "load_training_params",
    "train_efficientnet_iq",
]


MODEL_FACTORY = {
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b4": efficientnet_b4,
}

EfficientNetModelName = Literal[
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_b4",
]


def load_training_params(
    model_name: EfficientNetModelName,
    params_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load training parameters for an EfficientNet model.

    Args:
        model_name: EfficientNet model name used to select the default
            parameter file.
        params_path: Optional path to a training parameter YAML. If omitted,
            ``training_params/<model_name>.yaml`` is loaded.

    Returns:
        Parsed training parameters.

    Raises:
        FileNotFoundError: If the parameter file does not exist.
    """
    if params_path is None:
        params_path = (
            Path(__file__).parent
            / "training_params"
            / f"{model_name}.yaml"
        )
    else:
        params_path = Path(params_path)

    if not params_path.exists():
        raise FileNotFoundError(
            f"Training parameter file not found: {params_path}"
        )

    with params_path.open("r", encoding="utf-8") as params_file:
        return yaml.safe_load(params_file)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Build the learning-rate schedule used for EfficientNet training.

    The schedule applies a short linear warmup followed by cosine annealing.

    Args:
        optimizer: Optimizer whose learning rate will be scheduled.
        max_epochs: Total number of training epochs.

    Returns:
        Sequential learning-rate scheduler containing warmup and cosine phases.
    """
    warmup_epochs = min(5, max_epochs)
    cosine_epochs = max(max_epochs - warmup_epochs, 1)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def train_efficientnet_iq(
    train_cfg: TorchSigDatasetConfig,
    val_cfg: TorchSigDatasetConfig,
    test_cfg: TorchSigDatasetConfig,
    params: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    metrics_dir: str | Path | None = None,
    dataset_root: str | Path = "datasets",
    overwrite: bool = False,
    model_name: EfficientNetModelName = "efficientnet_b4",
    signal_generators: str | list[str] = "all",
) -> dict[str, Any]:
    """Train and evaluate an EfficientNet-1D IQ classifier.

    Args:
        train_cfg: TorchSig dataset configuration for the training split.
        val_cfg: TorchSig dataset configuration for the validation split.
        test_cfg: TorchSig dataset configuration for the test split.
        params: Training parameters, including batch size, learning rate,
            weight decay, and maximum epochs.
        checkpoint_dir: Directory where model checkpoints are written.
        metrics_dir: Optional directory where metric CSV files are written. If
            omitted, metrics are written under ``checkpoint_dir / "metrics"``.
        dataset_root: Root directory where generated/static datasets are stored.
        overwrite: Whether to regenerate existing TorchSig datasets.
        model_name: EfficientNet architecture to train.
        signal_generators: Signal generator selection passed to dataset
            preparation.

    Returns:
        Dictionary containing the trained Lightning model, wrapped PyTorch model,
        metric trackers, data loaders, class count, parameter count, and dataset
        preparation metadata.
    """
    set_deterministic(int(train_cfg.seed))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = Path(metrics_dir) if metrics_dir is not None else checkpoint_dir / "metrics"

    train_loader, val_loader, test_loader, data_info = prepare_torchsig_datasets(
        train_cfg,
        val_cfg,
        test_cfg,
        dataset_root=dataset_root,
        batch_size=params["batch_size"],
        overwrite=overwrite,
        signal_generators=signal_generators,
    )

    class_list = TorchSigSignalLists.all_signals
    num_classes = len(class_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODEL_FACTORY[model_name](
        num_classes=num_classes,
        drop_path_rate=params.get("drop_path", 0.2),
        drop_rate=params.get("drop_rate", 0.3),
    )

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=params.get("label_smoothing", 0.0),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    scheduler = _build_scheduler(optimizer, params["max_epochs"])

    pl_model, metrics_callback = train_validate(
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
    )

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
        "metrics": metrics_callback,
        "test_metrics": test_metrics,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_classes": num_classes,
        "num_params": compute_num_params(pl_model.model),
        "data_info": data_info,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for EfficientNet-1D training.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a 1D EfficientNet on a TorchSig dataset."
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Default TorchSig dataset config YAML used for train/val/test unless split-specific configs are provided.",
    )

    parser.add_argument(
        "--train-config",
        type=Path,
        help="TorchSig dataset config YAML for the training split.",
    )

    parser.add_argument(
        "--val-config",
        type=Path,
        help="TorchSig dataset config YAML for the validation split.",
    )

    parser.add_argument(
        "--test-config",
        type=Path,
        help="TorchSig dataset config YAML for the test split.",
    )

    parser.add_argument(
        "--params",
        type=Path,
        help=(
            "Training params YAML. Defaults to "
            "training_params/<model>.yaml."
        ),
    )

    parser.add_argument(
        "--model",
        choices=["efficientnet_b0", "efficientnet_b2", "efficientnet_b4"],
        default="efficientnet_b0",
        help="EfficientNet model to train.",
    )

    parser.add_argument(
        "--dataset-length",
        type=int,
        help="Override dataset length.",
    )

    parser.add_argument(
        "--dataset-id",
        help="Override dataset ID.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Override maximum training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate datasets.",
    )

    return parser.parse_args()


def _resolve_config_path(
    default_config: Path | None,
    split_config: Path | None,
    split: str,
) -> Path:
    """Resolve the dataset config path for a train, validation, or test split.

    Args:
        default_config: Shared config path supplied with ``--config``.
        split_config: Split-specific config path supplied with
            ``--<split>-config``.
        split: Split name used in the error message.

    Returns:
        Resolved dataset config path.

    Raises:
        ValueError: If neither a shared nor split-specific config path is
            provided.
    """
    if split_config is not None:
        return split_config

    if default_config is not None:
        return default_config

    raise ValueError(f"Must provide either --config or --{split}-config.")


if __name__ == "__main__":
    args = parse_args()

    train_cfg_path = _resolve_config_path(args.config, args.train_config, "train")
    val_cfg_path = _resolve_config_path(args.config, args.val_config, "val")
    test_cfg_path = _resolve_config_path(args.config, args.test_config, "test")

    train_config = load_config_from_yaml(train_cfg_path)
    val_config = load_config_from_yaml(val_cfg_path)
    test_config = load_config_from_yaml(test_cfg_path)

    if train_cfg_path == val_cfg_path == test_cfg_path:
        base_seed = train_config.seed
        train_config = replace(train_config, seed=base_seed)
        val_config = replace(val_config, seed=base_seed + 1)
        test_config = replace(test_config, seed=base_seed + 2)

    if args.dataset_length is not None:
        train_config = replace(train_config, dataset_length=args.dataset_length)
        val_config = replace(val_config, dataset_length=args.dataset_length)
        test_config = replace(test_config, dataset_length=args.dataset_length)

    if args.dataset_id is not None:
        train_config = replace(train_config, dataset_id=args.dataset_id)
        val_config = replace(val_config, dataset_id=args.dataset_id)
        test_config = replace(test_config, dataset_id=args.dataset_id)

    params = load_training_params(
        args.model,
        params_path=args.params,
    )

    if args.epochs is not None:
        params["max_epochs"] = args.epochs

    if args.batch_size is not None:
        params["batch_size"] = args.batch_size

    run_dir = Path("runs") / train_config.dataset_id / args.model

    result = train_efficientnet_iq(
        train_cfg=train_config,
        val_cfg=val_config,
        test_cfg=test_config,
        params=params,
        checkpoint_dir=run_dir / "checkpoints",
        metrics_dir=run_dir / "metrics",
        model_name=args.model,
        overwrite=args.overwrite,
    )

    print(f"Final Val F1: {result['metrics'].val_f1s[-1]:.4f}")


