# RT-DETR spectrogram detection

This module uses Ultralytics RT-DETR-L for multi-signal wideband detection. It
follows the neighboring EfficientNet module layout:

- `detr.py`: model factory
- `detr_train.py`: TorchSig dataset preparation, training, and test evaluation
- `detr_inference.py`: checkpoint inference and dataset evaluation
- `detr_hyperparameter_search.py`: Optuna tuning
- `training_params/` and `search_configs/`: packaged YAML defaults

TorchSig floating-point spectrograms and YOLO labels are exported into an
Ultralytics dataset beneath each run directory. The original train, validation,
and test seeds, class order, and every signal box are preserved. Generic image
augmentations are disabled because photographic color and geometry transforms
are not representative of RF spectrograms.

## Training

```bash
python torchsig_models/models/spectrogram_models/detr/detr_train.py \
  --dataset-config path/to/wideband.yaml \
  --signal-generators tone ofdm-64 \
  --dataset-length 1000 \
  --batch-size 2
```

## Evaluation

```bash
python torchsig_models/models/spectrogram_models/detr/detr_inference.py \
  --checkpoint runs/<dataset>/rtdetr_l/train/weights/best.pt \
  --dataset-yaml runs/<dataset>/rtdetr_l/dataset/dataset.yaml \
  --split test
```

RT-DETR's deformable attention uses a nondeterministic CUDA `grid_sample`
backward operation. Training therefore uses `deterministic=False`, while the
configured seed still controls initialization, data order, and augmentation.
