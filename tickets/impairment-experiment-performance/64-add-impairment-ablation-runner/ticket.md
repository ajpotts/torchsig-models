---
id: "64"
title: Add an impairment ablation experiment runner
depends_on: ["61", "62", "63"]
---

# Problem

Users must write custom orchestration to run clean controls, one-at-a-time
impairments, severity sweeps, replicates, and paired evaluation while preserving
provenance.

Confidence: **Medium-high**. Effort: **Medium**. It provides substantial user
value after the underlying dataset and cache contracts are stable.

# Proposed solution

Add a model-agnostic runner that expands an impairment matrix, prepares or
reuses datasets, exposes train/evaluate callbacks, and records manifests and
paired result tables. Keep framework-specific training outside TorchSIG core.

# Acceptance criteria

- Supports clean, single-impairment, severity, replicate, and selected
  interaction conditions.
- Resumes completed conditions using validated manifests.
- Records configuration, seeds, source identity, artifact fingerprints, timing,
  and user-supplied metrics.
- Produces per-condition and paired per-sample result tables.
- A smoke test runs without requiring a GPU or network service.

# Compatibility considerations

The runner is additive and does not prescribe a particular model framework.
