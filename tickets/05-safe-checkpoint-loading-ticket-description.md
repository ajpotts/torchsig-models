# Default EfficientNet inference to safe checkpoint loading

Type: Security  
Priority: Medium

## Problem

Spectrogram EfficientNet inference explicitly loads checkpoints with
`torch.load(..., weights_only=False)`. That enables general pickle
deserialization before checkpoint contents are validated. A malicious or
compromised checkpoint could therefore execute arbitrary Python code during
inference startup.

## Requested change

Load ordinary PyTorch state dictionaries and Lightning checkpoint dictionaries
with `weights_only=True` by default. Retain compatibility with trusted legacy
checkpoints through an explicit API and CLI opt-in. The unsafe path must warn
that general pickle deserialization can execute arbitrary code and should only
be used with a trusted checkpoint.

## Acceptance criteria

- Spectrogram EfficientNet inference passes `weights_only=True` by default.
- Plain state dictionaries and standard Lightning checkpoint dictionaries
  continue to load.
- General pickle loading is unavailable unless explicitly requested.
- The unsafe compatibility mode is exposed through a clearly named CLI option.
- Enabling unsafe loading emits a prominent trust warning.
- Tests cover both the safe default and explicit unsafe compatibility path.

## Security note

The unsafe option is a compatibility escape hatch, not a fallback. The loader
must never automatically retry a failed safe load with `weights_only=False`.
