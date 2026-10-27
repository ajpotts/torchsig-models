---
id: "61"
title: Add paired clean and impairment multi-view datasets
depends_on: ["61"]
---

# Problem

Paired evaluation currently requires independent dataset passes, repeated base
reads, and application-specific code to align clean and impaired samples.

Confidence: **High**. Effort: **Medium**. It directly improves scientific
comparisons and can reduce repeated I/O for small condition sets.

# Proposed solution

Provide a dataset returning a clean view, selected impairment views, shared
labels, sample identity, and realized impairment metadata from one base read.
Allow callers to request one condition per item or several views together.

# Acceptance criteria

- Clean and impaired views always share the same base sample identity.
- A multi-view item reads or generates its base signal once.
- Collation supports fixed condition sets without losing condition identity.
- View mutation is isolated and keyed replay is preserved.
- Tests cover classification and component-level metadata.

# Compatibility considerations

The new dataset is additive and does not change existing dataset return types.
