---
id: "445"
title: Cache and selectively materialize impairment variants
depends_on: ["61"]
---

# Problem

Expensive impairment views are recomputed across epochs and repeated runs, but
blindly materializing every condition can waste more time and storage than it
saves.

Confidence: **Medium-high**. Effort: **Medium-high**. Benefit is strong for
repeated expensive transforms but must be selected using measured break-even.

# Proposed solution

Add configuration-addressed artifact fingerprints and a policy that keeps cheap
views online while materializing variants whose measured reuse exceeds their
break-even point. Fingerprints include source identity, canonical transform
plan, seed contract, dtype, shape, and relevant TorchSIG implementation version.

# Acceptance criteria

- Equivalent configurations reuse complete compatible artifacts.
- Relevant source or transform changes invalidate the fingerprint.
- Incomplete artifacts are never treated as valid cache entries.
- Dry-run output estimates creation time, size, and break-even epochs.
- Materialized and online views are numerically and metadata equivalent within
  documented tolerances.

# Compatibility considerations

Caching is opt-in and must not change uncached numerical behavior.
