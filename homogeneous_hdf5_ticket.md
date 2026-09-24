# Add opt-in homogeneous HDF5 support for TorchSig 2.2 datasets

## Summary

Add support for generating static TorchSig datasets with the homogeneous HDF5
backend introduced in TorchSig 2.2. Storage selection should be exposed through
the shared dataset utilities so every model training workflow can use the new
backend without duplicating dataset-creation logic.

The existing HDF5 backend must remain the default to preserve compatibility
with current callers and previously generated datasets.

## Motivation

TorchSig's homogeneous HDF5 format stores fixed-shape, fixed-dtype top-level
samples in a native HDF5 array. This layout is suitable for the fixed-length IQ
and fixed-size spectrogram datasets used by this repository and can improve
storage and random-read behavior compared with the legacy object-per-record
layout.

## Proposed changes

- Add an optional file-handler/backend argument to
  `prepare_torchsig_datasets()` in `torchsig_models/utils/datasets.py`.
- Forward the selected handler through `_create_static_dataset()` to
  `DatasetCreator`.
- Default to TorchSig's legacy `HDF5Writer` so this is not a breaking change.
- Allow homogeneous-writer settings, such as compression and `chunk_samples`,
  to be passed through without coupling the utility to one specific writer.
- Update an appropriate dataset-generation or training example to demonstrate
  use of `HomogeneousHDF5Writer`.
- Document the homogeneous format's fixed top-level shape and dtype
  requirements.

The generated files should continue to live below the caller-supplied
`dataset_root` (currently `datasets/<dataset_id>/<split>` by default). Dataset
artifacts must not be committed to the package source tree.

## Compatibility and validation

- Existing calls that do not select a backend must behave exactly as before.
- Existing legacy HDF5 datasets must remain readable.
- Homogeneous HDF5 should be opt-in and should fail with a clear upstream error
  when samples do not share a common top-level shape and dtype.
- Complex-valued NumPy and PyTorch dtypes must be preserved.
- Writer selection must apply consistently to train, validation, and test
  splits.

## Acceptance criteria

- `prepare_torchsig_datasets()` can create all three dataset splits using
  TorchSig 2.2's `HomogeneousHDF5Writer`.
- The selected handler and its options are forwarded to each
  `DatasetCreator` invocation.
- Omitting the new arguments still selects the legacy HDF5 backend.
- A generated homogeneous dataset can be loaded through
  `StaticTorchSigDataset` and yields the expected samples and labels.
- Tests cover the default backend, explicit homogeneous backend, option
  forwarding, and at least one incompatible-input case.
- Public docstrings and one runnable example explain how to enable the backend
  and state its homogeneity constraints.

## Suggested tests

- Unit-test backend and keyword forwarding in `tests/utils/test_datasets.py`.
- Generate a small deterministic fixed-shape IQ dataset with the homogeneous
  backend and read it back through `StaticTorchSigDataset`.
- Confirm that shape, complex dtype, sample count, and class labels survive the
  round trip.
- Confirm that existing dataset utility tests pass unchanged with the default
  backend.

## Out of scope

- Changing the default storage backend.
- Migrating existing HDF5 datasets to the homogeneous format.
- Defining a new HDF5 schema in this repository.
- Supporting variable-shaped top-level samples in the homogeneous backend.
