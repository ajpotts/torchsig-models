# Training-Dataset Normalization Implementation Plan

## Goal

Make training-dataset, per-channel standardization the default normalization
strategy for the 1D IQ and 2D spectrogram EfficientNet models. Validation,
test, and inference data must reuse statistics calculated from the training
split so that absolute power differences remain available to the model and no
evaluation data leaks into training.

For each channel, normalization will be:

```text
(input - training_mean) / (training_std + epsilon)
```

## Supported modes

The model API will support three explicit modes:

- `dataset`: Use channel-wise statistics calculated from the training split.
  This will be the default for newly trained models.
- `sample`: Standardize every sample using its own statistics. This preserves
  the existing per-sample behavior where required for compatibility.
- `none`: Do not normalize model inputs.

## Implementation phases

### 1. Add reusable normalization components

Add a shared normalization module that:

- Stores channel-wise means and standard deviations as registered PyTorch
  buffers.
- Supports IQ tensors shaped `[batch, channels, time]`.
- Supports spectrogram tensors shaped
  `[batch, channels, frequency, time]`.
- Preserves the input tensor's dtype and device.
- Validates the number of channels and the supplied statistics.
- Handles near-zero standard deviations safely using a configured epsilon.
- Is included automatically in model state dictionaries and checkpoints.

Retain dedicated per-sample normalization for the `sample` mode.

### 2. Add streaming training-statistics calculation

Add a utility that computes channel-wise means and standard deviations from a
training data loader using numerically stable streaming accumulation with
`float64` intermediate values.

The implementation must:

- Traverse only the training split.
- Avoid loading the complete dataset into memory.
- Support both IQ and spectrogram batch shapes.
- Return deterministic results for deterministic input data.
- Reject empty datasets and malformed tensors with clear errors.

Statistics should be computed from the representation that reaches the model.
Random augmentations that alter power should not be included unless the
intended statistic is explicitly the augmented training distribution.

### 3. Update the EfficientNet model APIs

Add model arguments equivalent to:

```python
normalization="dataset"
normalization_mean=...
normalization_std=...
normalization_eps=1e-6
```

Constructing a model in `dataset` mode without valid statistics should fail
clearly instead of silently using placeholder values.

Preserve the existing `normalize` argument temporarily as a deprecated
compatibility alias:

- `normalize=True` maps to `sample`.
- `normalize=False` maps to `none`.
- Supplying both APIs inconsistently raises an error.
- Using the legacy argument emits a deprecation warning.

Apply equivalent behavior to the 1D and 2D EfficientNet wrappers.

### 4. Update both training pipelines

After static datasets and loaders have been prepared:

1. Calculate normalization statistics from the training split.
2. Construct the selected EfficientNet with those statistics.
3. Train using the model-resident normalization module.
4. Evaluate validation and test data using the same stored statistics.
5. Return the statistics in the training result metadata.
6. Log the normalization mode, mean, standard deviation, and epsilon.

No validation or test samples may participate in the statistics calculation.

### 5. Persist normalization state

Store the numerical mean and standard deviation as registered model buffers so
they are saved with normal model weights.

Also include descriptive checkpoint or training metadata such as:

```yaml
normalization:
  mode: dataset
  mean: [...]
  std: [...]
  eps: 1.0e-6
```

The registered buffers will be authoritative during model restoration. The
metadata is intended for checkpoint inspection and reconstruction decisions.

### 6. Update inference reconstruction

Both inference entry points should inspect the checkpoint before constructing
the model:

- If dataset-normalization buffers are present, construct the model in
  `dataset` mode and load the buffers strictly with the other weights.
- Use checkpoint metadata to validate the normalization mode and channel
  count when metadata is available.
- Never calculate normalization statistics from inference data.
- Allow an explicit `--normalization {dataset,sample,none}` override only when
  it is compatible with the checkpoint, or require an explicit opt-in to
  override checkpoint behavior.
- Produce clear errors for missing statistics, incompatible channel counts,
  and conflicting configuration.

### 7. Change training configuration defaults

Update the EfficientNet training parameter YAML files to use:

```yaml
normalization: dataset
```

Remove or deprecate existing `normalize: false` entries. Apply the new default
to EfficientNet-B0, B2, and B4 for both IQ and spectrogram training.

## Checkpoint compatibility

Existing checkpoints do not contain training-dataset statistics and must not
silently adopt the new behavior.

- Existing 1D checkpoints should retain their original per-sample
  normalization behavior.
- Existing 2D checkpoints should follow their saved legacy `normalize`
  setting, which is normally disabled.
- New checkpoints should default to dataset normalization and contain the
  required buffers and metadata.
- When legacy behavior must be inferred, emit an informational warning.
- If the original behavior cannot be determined reliably, require the caller
  to select a normalization mode explicitly.

## Tests

Add deterministic pytest coverage for:

- Correct channel-wise mean and standard-deviation calculation.
- Numerically stable streaming calculation over multiple batches.
- Training-only statistics with no validation or test leakage.
- Preservation of absolute power offsets under dataset normalization.
- Loss of constant power offsets under per-sample normalization.
- IQ and spectrogram input shapes.
- Multiple input channels.
- Registered-buffer state-dictionary and checkpoint round trips.
- Consistent normalization during training, validation, test, and inference.
- `dataset`, `sample`, and `none` modes.
- Empty data, zero variance, invalid mode, missing statistics, and channel
  mismatch failures.
- Legacy checkpoint behavior.
- CPU behavior and CUDA/device movement where available.
- Floating-point dtype preservation where applicable.

Run the narrow model and utility tests first, followed by all IQ and
spectrogram EfficientNet tests and the complete test suite if practical.

## Expected file scope

Expected changes include:

- A shared normalization/statistics utility under `torchsig_models/utils`.
- The 1D EfficientNet model, training, and inference modules.
- The 2D EfficientNet model, training, and inference modules.
- EfficientNet-B0, B2, and B4 training parameter YAML files for IQ and
  spectrogram models.
- The nearest model, training, inference, and utility pytest modules.

No new third-party dependency should be required.

## Delivery recommendation

Implement this in a separate merge request from the class-count changes. The
normalization work changes model input semantics and introduces checkpoint
compatibility considerations that should be reviewed and tested independently.
