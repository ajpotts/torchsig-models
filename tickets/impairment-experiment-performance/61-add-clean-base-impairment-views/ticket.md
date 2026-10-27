---
id: "61"
title: Reuse clean base signals across impairment conditions
depends_on: []
---

# Problem

Generating a separate synthetic dataset for every impairment condition repeats
signal construction, placement, noise-floor creation, metadata work, and file
reads even when only the impairment differs.

Confidence: **High**. Effort: **Medium**. This is expected to provide the
largest general speedup for multi-condition experiments.

# Proposed solution

Add a dataset wrapper that reads or generates a clean `Signal` once and applies
a keyed impairment view without mutating the stored base object. Preserve
component metadata and define whether impairment metadata is nominal or
realized. Support online views over static and deterministic synthetic sources.

# Acceptance criteria

- Multiple conditions reuse an identical clean sample identified by a stable ID.
- Applying one view cannot mutate data or metadata observed by another view.
- Online views work with static datasets and a documented deterministic
  synthetic source.
- Clean and impaired labels remain aligned.
- Benchmarks demonstrate reduced end-to-end work for at least three conditions.

# Compatibility considerations

Existing datasets and transform application remain unchanged when wrappers are
not used. Copy and ownership behavior must be explicit.
