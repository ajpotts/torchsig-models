## Summary

- resolve model dataset defaults from TorchSig's installed `default_configs`
  directory
- default training to the impaired narrowband training config
- default validation, test fallback, and inference to the impaired narrowband
  validation config
- retain explicit shared and split-specific config overrides

## Testing

- `NUMBA_DISABLE_JIT=1 pytest -q tests/utils/test_datasets.py tests/models/iq_models/efficientnet/test_efficientnet1d_train.py tests/models/iq_models/efficientnet/test_efficientnet1d_hyperparameter_search.py tests/models/spectrogram_models/efficientnet/test_efficientnet_train.py tests/models/spectrogram_models/efficientnet/test_efficientnet_hyperparameter_search.py` (98 passed)
- `make test` (253 passed, 3 skipped, 4 xfailed)
- `ruff check` on all changed Python files (passed)

## Notes

- No new dependencies are introduced.
- The validation config is used as the test fallback because TorchSig does not
  ship a separate impaired narrowband test config.
- Default-path tests isolate the TorchSig package layout instead of depending on
  whether the active development environment has installed optional YAML package
  data.
