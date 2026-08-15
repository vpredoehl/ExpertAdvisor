---
title: "Phase 4B Evaluation Run Large Canonical Identity Fix"
document_type: "architecture review"
status: "final"
generated_from: "Phase4B_Evaluation_Run_LargeCanonical_Identity_Fix_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Phase 4B Evaluation Run Large Canonical Identity Fix

Implemented and deployed the Phase 4B canonical-identity scale fix.

- Root cause: migration 034 put full canonical `TEXT` identity into B-tree `UNIQUE` keys. Scan 129 produced a 410,898-byte run canonical, exceeding PostgreSQL’s 8,191-byte index tuple limit.
- Migration: `066_phase4b_canonical_identity_btree_scale.sql` (production already had unrelated migrations 060–065).
  - Drops unsafe run/result/ranking-canonical unique constraints.
  - Adds bounded hash checks (≤128 bytes).
  - Adds result hash lookup index; preserves/reasserts run and ranking hash indexes.
- Concurrency/idempotency:
  - Transaction advisory locks are derived from complete canonical text.
  - Hashes only narrow lookup candidates; exact canonical comparison decides identity.
  - Hash/lock collisions only serialize requests; they never merge identities.
  - Results retain run+recommendation uniqueness; duplicate canonical identities across distinct recommendations fail explicitly.
- Additional defect found/fixed: Phase 4B ranking snapshot canonical unique index had the same scale risk. Downstream campaign planning already uses bounded hash/collision-safe patterns; no change needed.

Validation passed:

- Phase 4B evaluation/ranking migration SQL regressions.
- Disposable-db C++ repository regressions compiled with `-Werror`.
  - >42 KiB run canonical persists.
  - >42 KiB result canonical persists.
  - >42 KiB ranking snapshot canonical persists.
  - Exact retry converges.
  - Forced same-hash/different-canonical values remain distinct.
  - Concurrent identical runs converge.
  - Existing small-run behavior and scheduler/experiment immutability remain covered.
- Requested clean Release build: succeeded.
- Production migration: succeeded (`066`).
- Scan 129 reevaluation: succeeded.
  - `evaluation_run_id=5`
  - `status=completed`
  - `100/100` evaluated, `0` errors
  - canonical size `410898` bytes
  - no oversized B-tree error
  - output confirmed `experiment_created=false`, `experiment_queued=false`, `scheduler_modified=false`.

Files changed:

- `Database/migrations/066_phase4b_canonical_identity_btree_scale.sql`
- `Sources/ExperimentRecommendationEvaluationRepository.cpp`
- `Sources/ExperimentRecommendationRankingRepository.cpp`
- Phase 4B migration/repository tests
- `docs/Phase4BExperimentRecommendationEvaluation.rst`
- `docs/architecture/Volume_VIII_Recommendation_Engine.md`

Remaining blocker: none. An unrelated `OtherTests` MetaNN scheme still has pre-existing duplicate-header failures, but the required `LSTM Release` clean build passed.

`git status --short`

```text
 M Sources/ExperimentRecommendationEvaluationRepository.cpp
 M Sources/ExperimentRecommendationRankingRepository.cpp
 M Tests/ExperimentRecommendationEvaluationMigrationTests.sql
 M Tests/ExperimentRecommendationEvaluationRepositoryTests.cpp
 M Tests/ExperimentRecommendationRankingMigrationTests.sql
 M Tests/ExperimentRecommendationRankingRepositoryTests.cpp
 M docs/Phase4BExperimentRecommendationEvaluation.rst
 M docs/architecture/Volume_VIII_Recommendation_Engine.md
?? Database/migrations/066_phase4b_canonical_identity_btree_scale.sql
```

`git diff --stat`

```text
8 files changed, 258 insertions(+), 65 deletions(-)
```

(The new untracked migration is not included in Git’s diff stat until staged.)