# Store inference-critical metadata in model checkpoints

## Problem

Model weights do not fully describe how to reconstruct and use a trained model.
EfficientNet inference currently depends on caller-selected architecture and
external parameter files. A checkpoint can therefore load with the wrong input
contract or model configuration, producing errors or invalid predictions.

Normalization statistics and ordered class names are already preserved, but
architecture, preprocessing, input layout, output behavior, and provenance are
not represented by a single versioned checkpoint contract.

## Scope

- Add versioned model metadata to new EfficientNet Lightning checkpoints.
- Record model name, input channels, class count, dropout configuration, input
  representation, dtype, layout, preprocessing, ordered labels, output
  postprocessing, and package provenance.
- Describe IQ inputs as real/imaginary channels.
- Describe spectrogram inputs with the transform name and FFT size.
- Restore the model name and construction parameters during inference.
- Reject conflicting overrides, unsupported schemas, incomplete metadata, and
  incompatible input representations.
- Continue supporting legacy and weights-only checkpoints with existing
  defaults and explicit overrides.

## Acceptance criteria

- Newly trained 1D and 2D EfficientNet checkpoints contain `model_metadata`
  with a supported schema version.
- Inference reconstructs new checkpoints without requiring `--model`.
- Explicit model overrides that conflict with checkpoint metadata fail clearly.
- Legacy checkpoints retain the previous EfficientNet-B4 fallback.
- Ordered class names can be restored from the versioned metadata block.
- Tests cover round-trip, legacy, boundary, and invalid metadata cases.

## Out of scope

- Automatically transforming raw IQ into spectrograms during inference.
- Adding calibration thresholds for models that do not currently calibrate
  outputs.
- Changing the checkpoint format used by non-EfficientNet model families.
