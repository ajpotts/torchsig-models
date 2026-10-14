# Summary

Add a versioned, inference-critical metadata contract to EfficientNet
checkpoints so new models can be reconstructed and validated without relying on
caller knowledge of the training configuration.

## Changes

- Add checkpoint metadata helpers for building, validating, and resolving model
  configuration.
- Save architecture, input contract, preprocessing, output labels and behavior,
  and package provenance through the shared Lightning classifier.
- Populate metadata from both IQ and spectrogram EfficientNet training paths.
- Restore the EfficientNet variant and dropout construction values during
  inference.
- Validate schema versions, input representations, channel counts, class
  counts, and conflicting model overrides.
- Preserve legacy checkpoint behavior with an EfficientNet-B4 fallback.
- Allow class-name restoration from the new metadata block.
- Add focused metadata and checkpoint-hyperparameter tests.

## Testing

- `pytest -q tests/utils/test_checkpoint_metadata.py tests/utils/test_training.py tests/utils/test_class_names.py tests/models/spectrogram_models/efficientnet/test_efficientnet_train.py tests/models/spectrogram_models/efficientnet/test_efficientnet_inference.py tests/models/iq_models/efficientnet/test_efficientnet1d_train.py`

## Compatibility

Legacy Lightning and weights-only checkpoints remain supported. The public
inference functions still accept an explicit model name; the default is now
resolved after reading the checkpoint so embedded metadata takes precedence.

No new dependencies are introduced.
