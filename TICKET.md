# Remove unnormalized inputs from model hyperparameter searches

## Problem

The packaged EfficientNet hyperparameter-search configurations include `none`
as a normalization candidate. Trials without normalization are not part of the
desired search space and spend optimization budget on an unsupported experiment
choice.

## Requested change

Limit the `normalization` categorical search space in every packaged
EfficientNet search configuration to:

- `dataset`
- `sample`

Apply the change to the B0, B2, and B4 configurations for both IQ/1D and
spectrogram/2D models. This change is limited to hyperparameter-search defaults;
it does not remove the model API's backwards-compatible `none` mode.

## Acceptance criteria

- All six packaged search configurations contain exactly `dataset` and `sample`
  as normalization choices.
- No packaged search configuration offers `none` normalization.
- Tests validate the allowed normalization search space for every configuration.
