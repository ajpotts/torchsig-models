"""Lightning model implementation for TimeVQVAE-AD.

This module implements the ``TimeVQVAEAD`` Lightning module that wraps
``TimeVQVAEAD`` for use with the Anomalib framework. It supports
two-stage training via manual optimization (following the DSR pattern):

1. **Stage 1 (VQ-VAE)** – Trains encoder, VQ codebook, and decoder with
   MSE reconstruction + VQ commitment loss.
2. **Stage 2 (MaskGIT)** – Freezes Stage 1, trains a bidirectional
   transformer to predict randomly masked VQ tokens via cross-entropy.

The stage transition is epoch-based, controlled by ``stage2_start_ratio``.

Source: TimeVQVAE-AD experiments/exp_stage1.py, experiments/exp_stage2.py

Example:
    >>> from torchsig_models.models.spec_models.tvqvae_ad import TimeVQVAEAD
    >>> model = TimeVQVAEAD()
    >>> model.trainer_arguments
    {'num_sanity_val_steps': 0}

See Also:
    :class:`torchsig_models.models.spec_models.tvqvae_ad.torch_model.TimeVQVAEAD`:
        PyTorch implementation.
    :class:`torchsig_models.models.spec_models.tvqvae_ad.loss.Stage1Loss`:
        Loss for Stage 1.
    :class:`torchsig_models.models.spec_models.tvqvae_ad.loss.Stage2Loss`:
        Loss for Stage 2.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchvision.transforms.v2 import Compose, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .components.utils import freeze, linear_warmup_cosine_annealingLR, unfreeze
from .loss import Stage1Loss, Stage2Loss
from .torch_model import TimeVQVAEAD as TimeVQVAEADModel

__all__ = ["TimeVQVAEAD"]

logger = logging.getLogger(__name__)


class TimeVQVAEAD(AnomalibModule):
    """TimeVQVAE-AD Lightning Module for anomaly detection.

    Two-stage training with manual optimization following the Anomalib DSR
    pattern. Stage 1 trains the VQ-VAE; Stage 2 trains the MaskGIT
    transformer. Inference uses mask-prediction NLL anomaly scoring.

    Source: TimeVQVAE-AD experiments/exp_stage1.py, experiments/exp_stage2.py,
            configs/config.yaml

    Args:
        in_channels: Input channels (default ``1``).
        dim: Latent feature dimension.
        downsample_rate: Temporal downsampling factor (power of 2).
        n_resnet_blocks: ResNet blocks per encoder/decoder stage.
        codebook_size: Number of VQ codebook entries.
        frequency_independence: Process frequency bins independently.
        dropout: Dropout rate for encoder/decoder.
        choice_temperature: MaskGIT inference temperature.
        T: MaskGIT iterative decoding steps.
        mask_scheduling_func: Mask scheduling mode.
        prior_model_config: BidirectionalTransformer kwargs.
        latent_window_size: Masking kernel size for inference.
        lr_stage1: Learning rate for Stage 1.
        lr_stage2: Learning rate for Stage 2.
        stage2_start_ratio: Fraction of total epochs at which Stage 2
            begins. E.g., ``0.3`` means first 30% of epochs train Stage 1.
        linear_warmup_rate: Fraction of total steps used for LR warmup.
        pre_processor: Pre-processor instance or flag.
        post_processor: Post-processor instance or flag.
        evaluator: Evaluator instance or flag.
        visualizer: Visualizer instance or flag.

    Example:
        >>> model = TimeVQVAEAD()
        >>> model.learning_type
        <LearningType.ONE_CLASS: 'one_class'>
    """

    def __init__(
        self,
        in_channels: int = 1,
        dim: int = 64,
        downsample_rate: int = 4,
        n_resnet_blocks: int = 4,
        codebook_size: int = 64,
        frequency_independence: bool = True,
        dropout: float = 0.3,
        choice_temperature: float = 4.0,
        T: int = 20,
        mask_scheduling_func: str = "cosine",
        prior_model_config: dict[str, Any] | None = None,
        latent_window_size: int | None = None,
        hybrid_symmetric_blocks: int = 0,
        input_size: tuple[int, int] = (256, 256),
        lr_stage1: float = 1e-3,
        lr_stage2: float = 1e-3,
        stage2_start_ratio: float = 0.3,
        linear_warmup_rate: float = 0.1,
        score_mode: str = "hybrid",
        top_fraction: float = 0.02,        
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        # Manual optimization for two-stage training (DSR pattern)
        self.automatic_optimization = False

        # Hyperparameters
        self.lr_stage1 = lr_stage1
        self.lr_stage2 = lr_stage2
        self.stage2_start_ratio = stage2_start_ratio
        self.linear_warmup_rate = linear_warmup_rate

        # Stage boundary (set in configure_optimizers)
        self.stage2_start_epoch: int = 0

        # Loss functions
        self.stage1_loss = Stage1Loss()
        self.stage2_loss = Stage2Loss()

        # PyTorch model
        self.model = TimeVQVAEADModel(  # pyrefly: ignore
            in_channels=in_channels,
            dim=dim,
            downsample_rate=downsample_rate,
            n_resnet_blocks=n_resnet_blocks,
            codebook_size=codebook_size,
            frequency_independence=frequency_independence,
            dropout=dropout,
            choice_temperature=choice_temperature,
            T=T,
            mask_scheduling_func=mask_scheduling_func,
            prior_model_config=prior_model_config,
            latent_window_size=latent_window_size,
            hybrid_symmetric_blocks=hybrid_symmetric_blocks,
            input_size=input_size,
            score_mode=score_mode,
            top_fraction=top_fraction,            
        )

    # ------------------------------------------------------------------
    # Optimizer configuration
    # Source: exp_stage1.py lines 102-105, exp_stage2.py lines 84-87
    # Pattern: DSR configure_optimizers (manual, two optimizers)
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure separate optimizers for Stage 1 and Stage 2.

        Source: experiments/exp_stage1.py line 103 (AdamW + cosine annealing),
                experiments/exp_stage2.py line 85 (AdamW + cosine annealing)

        Returns:
            Tuple of two optimizer/scheduler dicts.
        """
        # Determine total epochs — use datamodule to estimate steps
        datamodule = self.trainer.datamodule  # pyrefly: ignore
        steps_per_epoch: int = len(datamodule.train_dataloader())
        max_epochs = self.trainer.max_epochs or 0
        num_epochs = max(
            self.trainer.max_steps // steps_per_epoch,
            max_epochs,
        )
        if num_epochs < 2 or not 0 < self.stage2_start_ratio < 1:
            raise ValueError("Need >=2 epochs and stage2_start_ratio in (0,1)")
        self.stage2_start_epoch = min(num_epochs-1, max(1, int(num_epochs*self.stage2_start_ratio)))

        # Estimate steps per stage for scheduler
        stage1_steps = self.stage2_start_epoch * steps_per_epoch
        stage2_steps = (num_epochs - self.stage2_start_epoch) * steps_per_epoch

        # Typed local to expose subclass attributes to pyrefly
        model: TimeVQVAEADModel = self.model  # pyrefly: ignore

        # Stage 1 optimizer: encoder + decoder + vq_model
        opt_stage1 = torch.optim.AdamW(
            params=(
                list(model.encoder.parameters())
                + list(model.decoder.parameters())
                + list(model.vq_model.parameters())
            ),
            lr=self.lr_stage1,
        )
        sched_stage1 = linear_warmup_cosine_annealingLR(
            opt_stage1,
            max(stage1_steps, 1),
            self.linear_warmup_rate,
        )

        # Stage 2 optimizer: transformer only
        opt_stage2 = torch.optim.AdamW(
            params=model.maskgit.transformer.parameters(),
            lr=self.lr_stage2,
        )
        sched_stage2 = linear_warmup_cosine_annealingLR(
            opt_stage2,
            max(stage2_steps, 1),
            self.linear_warmup_rate,
        )

        return (  # pyrefly: ignore
            {"optimizer": opt_stage1, "lr_scheduler": sched_stage1},
            {"optimizer": opt_stage2, "lr_scheduler": sched_stage2},
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        """Log stage transitions and freeze/unfreeze components.

        At the start of Stage 2, freezes Stage 1 components and unfreezes
        the transformer.
        """
        if self.current_epoch >= self.stage2_start_epoch:
            logger.info(
                "TimeVQVAE-AD: Transitioning to Stage 2 (MaskGIT) at epoch %d.",
                self.current_epoch,
            )
            # Typed local to expose subclass attributes to pyrefly
            model: TimeVQVAEADModel = self.model  # pyrefly: ignore
            # Freeze Stage 1
            freeze(model.encoder)
            freeze(model.decoder)
            freeze(model.vq_model)
            model.encoder.eval()
            model.decoder.eval()
            model.vq_model.eval()

    def on_train_start(self) -> None:
        """Ensure Stage 1 components are trainable at the start.

        MaskGIT.__init__ freezes encoder/decoder/vq_model; we must
        unfreeze them for Stage 1 training.
        """
        # Typed local to expose subclass attributes to pyrefly
        model: TimeVQVAEADModel = self.model  # pyrefly: ignore
        for module in (model.encoder, model.decoder, model.vq_model):
            if self.current_epoch < self.stage2_start_epoch:
                unfreeze(module)
                module.train()
            else:
                freeze(module)
                module.eval()

    # ------------------------------------------------------------------
    # Training step
    # Source: exp_stage1.py lines 63-82, exp_stage2.py lines 28-46
    # Pattern: DSR training_step with manual optimization
    # ------------------------------------------------------------------

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Training step dispatching to Stage 1 or Stage 2.

        Args:
            *args: Positional arguments; first is the input ``Batch``.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dict containing the loss value.
        """
        del kwargs
        batch: Batch = args[0]
        if getattr(batch, "gt_label", None) is not None and batch.gt_label.bool().any():
            raise ValueError("One-class training requires normal-only data")        
        stage1_opt, stage2_opt = self.optimizers()

        if self.current_epoch < self.stage2_start_epoch:
            # ---- Stage 1: VQ-VAE ----
            x: torch.Tensor = batch.image  # pyrefly: ignore
            x_rec, vq_loss_dict = self.model.forward_stage1(x)  # pyrefly: ignore
            loss = self.stage1_loss(x, x_rec, vq_loss_dict)  # pyrefly: ignore

            stage1_opt.zero_grad()
            self.manual_backward(loss)
            stage1_opt.step()

            # Step scheduler
            scheds = self.lr_schedulers()
            if isinstance(scheds, (list, tuple)):
                scheds[0].step()  # pyrefly: ignore

            self.log("stage1_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("stage", 1.0, on_epoch=True, prog_bar=True, logger=True)

        else:
            # ---- Stage 2: MaskGIT ----
            x: torch.Tensor = batch.image  # pyrefly: ignore
            logits, targets, mask = self.model.forward_stage2(x)  # pyrefly: ignore
            loss = self.stage2_loss(logits, targets, mask)  # pyrefly: ignore

            stage2_opt.zero_grad()
            self.manual_backward(loss)
            stage2_opt.step()

            # Step scheduler
            scheds = self.lr_schedulers()
            if isinstance(scheds, (list, tuple)):
                scheds[1].step()  # pyrefly: ignore

            self.log("stage2_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("stage", 2.0, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    # ------------------------------------------------------------------
    # Validation step
    # Pattern: DSR validation_step
    # ------------------------------------------------------------------

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Validation step using inference (anomaly scoring) path.

        Args:
            *args: Positional arguments; first is the input ``Batch``.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Updated batch with predictions.
        """
        del kwargs
        batch: Batch = args[0]
        predictions = self.model(batch.image)  # pyrefly: ignore
        return batch.update(
            pred_score=predictions.pred_score,
            anomaly_map=predictions.anomaly_map,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Required trainer arguments.

        Returns:
            Dict with ``num_sanity_val_steps: 0`` to skip validation
            before Stage 1 components have seen data.
        """
        return {"num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type.

        Returns:
            ``LearningType.ONE_CLASS`` for unsupervised anomaly detection.
        """
        return LearningType.ONE_CLASS

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
    ) -> PreProcessor:
        """Configure default pre-processor.

        No ImageNet normalization is applied (model trains from scratch).

        Args:
            image_size: Target image size. Defaults to ``(256, 256)``.

        Returns:
            Configured ``PreProcessor`` with resize transform.
        """
        image_size = image_size or (256, 256)
        transform = Compose([Resize(image_size, antialias=True)])
        return PreProcessor(transform=transform)
