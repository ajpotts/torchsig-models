# Make EfficientNet checkpoint loading safe by default

## Summary

This change prevents spectrogram EfficientNet inference from performing general
pickle deserialization by default. Standard state dictionaries and Lightning
checkpoint dictionaries are now loaded with `torch.load(weights_only=True)`.

For trusted legacy checkpoints that require arbitrary Python objects, callers
may explicitly set `allow_unsafe_checkpoint=True` or pass
`--allow-unsafe-checkpoint` on the command line. Enabling that mode emits a
`RuntimeWarning` explaining that pickle deserialization can execute arbitrary
code. Safe-load failures are not automatically retried unsafely.

## Changes

- Added a small centralized checkpoint-loading helper.
- Changed the default inference path to `weights_only=True`.
- Added an explicit trusted-legacy compatibility argument and CLI flag.
- Documented the security implications in API and CLI help text.
- Updated existing checkpoint assertions and added coverage for the unsafe
  opt-in warning and `weights_only=False` call.

## Testing

- EfficientNet-2D inference tests: 14 passed.
- Complete EfficientNet-2D test group: 72 passed.
- Ruff checks passed for the modified source and inference test files.
- CLI help exposes and documents `--allow-unsafe-checkpoint`.
- `git diff --check` passed.

## Risk and compatibility

Normal project checkpoints containing tensors and primitive metadata remain
supported. Legacy checkpoints containing non-allowlisted Python objects now
fail safely unless the user explicitly enables unsafe loading. This is an
intentional security boundary and may require a command-line change for those
legacy files.
