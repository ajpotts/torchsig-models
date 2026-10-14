## Summary

- remove `none` from normalization choices in all EfficientNet search configs
- retain `dataset` and `sample` normalization for B0, B2, and B4 searches
- update the packaged-config test to enforce the narrowed search space

## Scope

This updates the six IQ/1D and spectrogram/2D hyperparameter-search YAML files.
The underlying model and inference APIs continue to support `none` for existing
callers and checkpoints.

## Testing

- `.venv/bin/pytest -q tests/utils/test_hyperperameter_search.py` (37 passed)
- verified all six packaged search YAMLs use `choices: [dataset, sample]` and
  none contains `none` as a normalization choice
- `git diff --check` (passed)

## Dependencies

No new dependencies.
