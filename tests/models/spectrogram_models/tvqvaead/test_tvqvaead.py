"""Small CPU tests for the public TimeVQVAE-AD integration."""

import copy

import pytest
import torch
from anomalib import LearningType
from anomalib.data import ImageBatch
from anomalib.models.components import AnomalibModule
from lightning.pytorch import LightningDataModule, Trainer
from torch.utils.data import DataLoader

import torchsig_models
from torchsig_models.models import TimeVQVAEAD, TimeVQVAEADModel
from torchsig_models.models.spectrogram_models import tvqvaead
from torchsig_models.models.spectrogram_models.tvqvaead.loss import Stage1Loss, Stage2Loss


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    """Avoid excessive thread overhead for these deliberately tiny models."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return {
        "in_channels": 1,
        "dim": 8,
        "downsample_rate": 2,
        "n_resnet_blocks": 1,
        "codebook_size": 8,
        "input_size": (4, 8),
        "dropout": 0.0,
        "prior_model_config": {
            "hidden_dim": 16,
            "depth": 1,
            "heads": 2,
            "attn_dim_head": 4,
            "ff_mult": 2,
            "use_rmsnorm": True,
            "dropout": 0.0,
        },
    }


@pytest.fixture
def model(config):
    # Explicit CPU allocation also keeps these tests safe on GPU-equipped hosts.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return TimeVQVAEADModel(**config).cpu()


@pytest.fixture
def images():
    return torch.rand(2, 1, 4, 8, generator=torch.Generator().manual_seed(42))


def test_public_imports():
    assert torchsig_models.TimeVQVAEAD is TimeVQVAEAD
    assert torchsig_models.TimeVQVAEADModel is TimeVQVAEADModel
    assert tvqvaead.TimeVQVAEAD is TimeVQVAEAD
    assert tvqvaead.TimeVQVAEADModel is TimeVQVAEADModel
    assert issubclass(TimeVQVAEAD, AnomalibModule)
    assert issubclass(TimeVQVAEADModel, torch.nn.Module)


@pytest.mark.parametrize("symmetric_blocks", [0, 1])
def test_stage1_reconstruction_and_backward(config, images, symmetric_blocks):
    model = TimeVQVAEADModel(**config, hybrid_symmetric_blocks=symmetric_blocks).cpu()
    # MaskGIT starts with Stage 1 frozen; the wrapper enables it for Stage 1.
    for module in (model.encoder, model.decoder, model.vq_model):
        module.requires_grad_(True)
        module.train()
    reconstructed, vq_loss = model.forward_stage1(images)
    loss = Stage1Loss()(images, reconstructed, vq_loss)
    assert reconstructed.shape == images.shape
    assert reconstructed.device.type == "cpu"
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for module in (model.encoder, model.decoder):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients
        assert all(torch.isfinite(grad).all() for grad in gradients)
        assert any(grad.abs().sum() > 0 for grad in gradients)


def test_stage2_masked_prediction_and_backward(model, images):
    model.train()
    logits, targets, known = model(images)
    assert logits.shape == (2, 16, 8)
    assert targets.shape == known.shape == (2, 16)
    assert targets.dtype == torch.long
    assert known.dtype == torch.bool
    assert (~known).any(dim=1).all()
    assert targets.min() >= 0 and targets.max() < model.codebook_size
    loss = Stage2Loss()(logits, targets, known)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [
        p.grad for p in model.maskgit.transformer.parameters() if p.grad is not None
    ]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)
    assert all(p.grad is None for p in model.encoder.parameters())
    assert not model.encoder.training and not model.vq_model.training


@pytest.mark.parametrize("score_mode", ["nll", "reconstruction", "hybrid"])
def test_inference_scores_and_state_dict(config, images, score_mode):
    model = TimeVQVAEADModel(**config, score_mode=score_mode).cpu().eval()
    prediction = model(images)
    assert prediction.pred_score.shape == (2,)
    assert prediction.anomaly_map.shape == (2, 4, 8)
    for tensor in (prediction.pred_score, prediction.anomaly_map):
        assert tensor.device.type == "cpu"
        assert torch.isfinite(tensor).all()
        assert (tensor >= 0).all()
        assert not tensor.requires_grad
    restored = TimeVQVAEADModel(**config, score_mode=score_mode).cpu().eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    restored_prediction = restored(images)
    torch.testing.assert_close(restored_prediction.pred_score, prediction.pred_score)
    torch.testing.assert_close(restored_prediction.anomaly_map, prediction.anomaly_map)


def test_stage2_loss_ignores_known_tokens():
    logits = torch.tensor([[[2.0, -1.0], [0.0, 1.0]]], requires_grad=True)
    targets = torch.tensor([[0, 1]])
    known = torch.tensor([[True, False]])
    loss = Stage2Loss()(logits, targets, known)
    expected = torch.nn.functional.cross_entropy(logits[:, 1], targets[:, 1])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, 1]) > 0


class TinyDataModule(LightningDataModule):
    """One normal spectrogram batch per epoch; no dataset generation or files."""

    def __init__(self, images):
        super().__init__()
        self.images = images

    def train_dataloader(self):
        return DataLoader(
            [
                ImageBatch(
                    image=self.images,
                    gt_label=torch.zeros(len(self.images), dtype=torch.long),
                )
            ],
            batch_size=None,
            num_workers=0,
        )


def test_lightning_two_stage_training_and_validation(config, images, tmp_path):
    # Anomalib ImageBatch requires RGB-like spectrograms.
    config = {**config, "in_channels": 3}
    images = images.repeat(1, 3, 1, 1)
    module = TimeVQVAEAD(
        **config,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    ).cpu()
    assert module.learning_type == LearningType.ONE_CLASS
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        default_root_dir=tmp_path,
    )
    encoder_before = [p.detach().clone() for p in module.model.encoder.parameters()]
    transformer_before = [
        p.detach().clone() for p in module.model.maskgit.transformer.parameters()
    ]
    trainer.fit(module, datamodule=TinyDataModule(images))
    assert module.stage2_start_epoch == 1
    assert trainer.global_step == 2
    assert any(
        not torch.equal(before, after)
        for before, after in zip(encoder_before, module.model.encoder.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            transformer_before, module.model.maskgit.transformer.parameters()
        )
    )
    assert all(not p.requires_grad for p in module.model.encoder.parameters())
    assert all(p.device.type == "cpu" for p in module.parameters())
    module.eval()
    output = module.validation_step(ImageBatch(image=images), 0)
    assert output.pred_score.shape == (2,)
    assert output.anomaly_map.shape[-2:] == images.shape[-2:]
    assert torch.isfinite(output.pred_score).all()
