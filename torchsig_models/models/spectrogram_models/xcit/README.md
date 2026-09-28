# Spectrogram XCiT

`xcit_nano` wraps timm's `xcit_nano_12_p16_224` for multi-label signal-presence
classification in wideband TorchSig spectrograms. Inputs may be
`[batch, frequency, time]` for single-channel spectrograms or
`[batch, channels, frequency, time]`. Outputs are independent class logits;
apply sigmoid and a decision threshold to identify every class present.

```python
from torchsig_models.models.spectrogram_models.xcit import xcit_nano

model = xcit_nano(num_classes=72, input_channels=2)
present = torch.sigmoid(model(spectrograms)) >= 0.5
```

Training configurations must allow at least two signals per sample. Dataset
creation adds TorchSig's `MultiHotLabel`, and training uses
`BCEWithLogitsLoss` with multilabel accuracy, precision, recall, and F1.

Example CPU training invocation:

```bash
python -m torchsig_models.models.spectrogram_models.xcit.xcit_train \
    --dataset-config /path/to/dataset.yaml \
    --accelerator cpu \
    --devices 1 \
    --epochs 1
```

Evaluate a saved checkpoint with:

```bash
python -m torchsig_models.models.spectrogram_models.xcit.xcit_inference \
    --root /path/to/static/test \
    --checkpoint runs/example/xcit_nano/checkpoints/best.ckpt
```

Pass `pretrained=True` to request compatible timm weights, or
`checkpoint_path=...` to load a TorchSIG Models checkpoint.

Run the packaged TorchSig hyperparameter search with:

```bash
python -m torchsig_models.models.spectrogram_models.xcit.xcit_hyperparameter_search \
    --dataset-config /path/to/dataset.yaml \
    --n-trials 20
```

For a small TorchSig tone-versus-LFM-radar search demo that invokes the same
production entry point:

```bash
python examples/scripts/spectrogram_xcit/hyperparameter_search_demo.py --trials 3
```
