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

# Optional mixed-precision benchmark

`benchmark_mixed_precision.py` trains the real EfficientNet-1D model through
the shared Lightning training utility on a deterministic synthetic IQ dataset.
It compares `32-true`, `16-mixed`, and `bf16-mixed` in one invocation and
reports median fit throughput, peak allocated CUDA memory, final validation
accuracy, and relative results against `32-true`.

Run it on the target deployment GPU:

```bash
python benchmarks/optional/benchmark_mixed_precision.py \
  --model efficientnet_b0 \
  --json-output /tmp/mixed-precision-results.json
```

The default run includes one discarded warm-up training run and three measured
runs per supported precision. `bf16-mixed` is recorded as skipped on GPUs that
do not support bf16. The benchmark intentionally requires CUDA because
`16-mixed` GPU behavior is one of the configurations under comparison.

Useful tuning options:

```text
--precisions       Precision modes to compare (default: all three)
--model            efficientnet_b0, efficientnet_b2, or efficientnet_b4
--train-samples    Synthetic training samples per epoch (default: 2048)
--val-samples      Synthetic validation samples per epoch (default: 512)
--sample-length    Float IQ sample length (default: 4096)
--batch-size       Training and validation batch size (default: 64)
--epochs           Epochs per training trial (default: 3)
--warmup-runs      Discarded trials per precision (default: 1)
--runs             Measured trials per precision (default: 3)
--num-workers      Data-loader workers (default: 4)
```

Synthetic validation accuracy is useful for detecting precision-related
divergence, but it is not a substitute for task accuracy. Before changing
production defaults, repeat the comparison with the representative TorchSig
dataset and training duration used for deployment.
