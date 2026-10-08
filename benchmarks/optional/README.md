# Optional training data-loader benchmark

This benchmark measures the training loader returned by
`prepare_torchsig_datasets`. It uses a deterministic synthetic dataset with a
small configurable read delay, avoiding TorchSig dataset generation and model
compute in the timing.

The script is compatible with both the base branch and the loader-performance
branch:

- On main, where the loader options are absent, it measures the legacy
  zero-worker loader.
- On this branch, it requests `--num-workers` and enables pinned memory on CUDA
  plus persistent workers when the worker count is positive.

Run the same command from separate worktrees for an unbiased comparison:

```bash
python benchmarks/optional/benchmark_training_dataloader.py \
  --num-workers 4 \
  --json-output /tmp/loader-main.json
```

```bash
python benchmarks/optional/benchmark_training_dataloader.py \
  --num-workers 4 \
  --json-output /tmp/loader-feature.json
```

Because the benchmark file does not exist on main until this change is merged,
either keep it as an uncommitted file while switching branches or invoke the
copy from this branch/worktree by absolute path. Python imports the project from
the current working directory, so run it with the desired worktree as the
working directory.

Compare `median_samples_per_second` and `median_epoch_seconds`. Also verify that
main reports `loader_options_supported: false` and `effective_num_workers: 0`,
while this branch reports `loader_options_supported: true` and the requested
worker count.

Useful tuning options:

```text
--num-workers       Worker count requested on supported revisions (default: 4)
--samples           Samples per epoch (default: 2048)
--sample-length     Float IQ sample length (default: 4096)
--batch-size        Samples per batch (default: 64)
--delay-ms          Simulated read latency per sample (default: 1.0 ms)
--warmup-epochs     Untimed warmup epochs (default: 1)
--epochs            Timed epochs (default: 3)
--device            auto, cpu, or cuda (default: auto)
```

The synthetic delay makes worker parallelism visible and reproducible; it is
not a prediction of the exact speedup for every storage device. After finding a
promising worker count, confirm it with full EfficientNet-1D epoch timings on
the target dataset and hardware.
