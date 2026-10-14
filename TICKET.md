# Default model CLIs to TorchSig impaired dataset configs

## Problem

Model training, tuning, and inference entry points require callers to locate and
pass TorchSig dataset configuration files. This adds boilerplate to the common
workflow and makes otherwise valid commands fail when a config flag is omitted.

## Requested change

Use the installed configs under `torchsig/datasets/default_configs` when callers
do not provide explicit dataset configs:

- training: `narrowband_impaired_train_all.yaml`
- validation and inference: `narrowband_impaired_val_all.yaml`
- test fallback: `narrowband_impaired_val_all.yaml`

Explicit shared and split-specific configuration flags must continue to take
precedence.

## Acceptance criteria

- EfficientNet 1D and 2D training commands run without a dataset-config flag.
- EfficientNet 1D and 2D hyperparameter searches run without a dataset-config
  flag.
- XCiT training and inference wrappers have impaired narrowband defaults.
- Paths resolve from the installed TorchSig package rather than the working
  directory.
- Tests cover default resolution and explicit overrides remain supported.
