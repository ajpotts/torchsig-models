# Summary

Add a versioned, inference-critical metadata contract across EfficientNet,
XCiT, Ultralytics, and Anomalib models so checkpoints can be reconstructed and
validated without relying on caller knowledge of the training configuration.

## Changes

- Add checkpoint metadata helpers for building, validating, and resolving model
  configuration.
- Save architecture, input contract, preprocessing, output labels and behavior,
  and package provenance through the shared Lightning classifier.
- Populate metadata from both IQ and spectrogram EfficientNet training paths.
- Embed the same metadata contract in XCiT Lightning checkpoints.
- Write JSON metadata sidecars for Ultralytics and Anomalib checkpoints, whose
  framework-owned checkpoint internals should not be rewritten.
- Restore the EfficientNet variant and dropout construction values during
  inference.
- Validate schema versions, input representations, channel counts, class
  counts, and conflicting model overrides.
- Preserve legacy checkpoint behavior with an EfficientNet-B4 fallback.
- Allow class-name restoration from the new metadata block.
- Add focused native-checkpoint and sidecar tests.

## Testing

- `NUMBA_CACHE_DIR=/tmp/torchsig-numba-cache pytest -q tests/utils/test_checkpoint_metadata.py tests/utils/test_training.py tests/utils/test_class_names.py tests/models/spectrogram_models/efficientnet tests/models/iq_models/efficientnet tests/models/iq_models/xcit/test_xcit1d.py tests/adapters/test_yolo_train_detector.py tests/adapters/test_anomalib_utils.py`

## Compatibility

Legacy Lightning, weights-only, and third-party checkpoints without sidecars
remain supported. The public inference functions still accept an explicit
model name; the default is now resolved after reading the checkpoint so
embedded metadata takes precedence.

No new dependencies are introduced.
