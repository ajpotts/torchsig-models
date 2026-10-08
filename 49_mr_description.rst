Summary
=======

- Add a CLI and Python API override for legacy, packed, and homogeneous
  TorchSig dataset storage backends.
- Deterministically limit every existing train, validation, and test split to
  its configured dataset length without modifying the stored dataset.
- Apply the behavior to both IQ EfficientNet-1D and spectrogram EfficientNet-2D
  training and hyperparameter searches.
- Reuse identical seeded subsets during Optuna preparation and every trial.

Motivation
==========

Pre-generated datasets can use a storage backend that differs from the YAML
configuration. Previously, loading those datasets required editing the YAML,
and --dataset-length affected generation only. This prevented small,
repeatable experiments against large existing datasets such as homogeneous
HDF5 collections.

Implementation
==============

The shared dataset utility now owns the mapping from storage backend names to
reader and writer classes. Its optional storage_backend argument overrides the
YAML selection while preserving the configured behavior when omitted.

When static datasets are loaded, each split is wrapped in a seeded subset with
at most cfg.dataset_length samples. The split's configured seed determines its
indices, requests larger than the stored split retain all available samples,
and a seed-compatible subset wrapper preserves TorchSig data-loader behavior.
Generated datasets continue to use dataset_length during creation as before.

Both EfficientNet training CLIs expose --storage-backend, and both Optuna CLIs
forward it during initial dataset preparation and subsequent trials. The
--dataset-length help text documents its behavior for all three existing
splits.

Testing
=======

- Verify a homogeneous backend override selects HomogeneousHDF5Reader even
  when the configuration selects legacy storage.
- Verify existing train, validation, and test lengths are limited, oversized
  requests are capped, zero-length requests work, and seeded selections are
  reproducible.
- Verify the new CLI option is accepted by IQ and spectrogram training.
- Verify IQ and spectrogram Optuna workflows forward the backend override to
  both dataset preparation and individual trials.
- Preserve coverage for YAML-selected backends and generated datasets.


