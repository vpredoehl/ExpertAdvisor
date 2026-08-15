---
title: "LSTM Configurable Donchian Lookback Increment"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Configurable_Donchian_Lookback_Increment_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Configurable Donchian Lookback Increment

Implemented configurable Donchian lookback; no commit made.

- Default: `20`; valid range: `1..10000`.
- CLI/queue/sweep: `--donchian-lookback=N`; invalid input fails explicitly.
- Mode remains independent: zero-ablation still emits zero Donchian channels regardless of lookback.
- Causal calculation still uses completed predecessor bars only; current/future bars excluded.

Semantic compatibility: added recommendation semantic configuration `v6`, which includes `donchian_lookback`. Historical v3–v5 canonical forms remain unchanged and reconstruct lookback `20`; hashes are not reinterpreted.

Database/provenance:

- Added [068_donchian_lookback.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/068_donchian_lookback.sql), forward-only and rerunnable.
- Backfills/defaults existing experiments to `20`, enforces bounds, and adds lookback to `experiment_unique_identity_uidx`.
- Lookback now flows through experiment scheduling, model metadata, recommendation semantic/invocation identities, duplicate detection, evaluation, conversion execution, and campaign materialization/reconstruction.

Files changed include the Donchian calculator/Tensor, model metadata, LSTM CLI/runtime, scheduler, recommendation repositories/semantic configuration, focused tests, and [warmup documentation](/Volumes/Developer%20SSD/ExpertAdvisor/docs/LSTMStatefulFeatureWarmup.rst).

Verification passed:

- `Tests/Donchian20FeatureIncrementTests.sh`
- `Tests/LSTMFeatureVectorParityTests.sh`
- `Tests/LSTMStatefulFeatureWarmupTests.sh`
- Standalone recommendation semantic test with `-Wall -Wextra -Wpedantic -Werror`
- Isolated DB-backed recommendation repository, conversion-execution, and campaign-materialization suites against `ea_warmup_provenance_test`, compiled with warnings enabled and current libpqxx flags
- Full isolated scratch Release build:
  `xcodebuild ... -derivedDataPath DerivedData/DonchianLookbackScratch build`
- Migration applied twice inside a rolled-back transaction on a temporary table: passed default/backfill, constraint, identity-index, and rerun checks.
- `git diff --check`: passed.

Release status: the normal Release build was deferred. A live scheduler and training worker are executing `DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release`, so rebuilding that path could replace an active executable.

Remaining risk: run the normal no-clean Release build once the scheduler/worker is idle, then apply migration 068 in staging before production rollout.

`git status --short`: 20 modified/new implementation and test files; no unrelated changes observed.
`git diff --stat`: 20 tracked files changed, `344 insertions(+), 52 deletions(-)`, plus new migration and header.

Recommendation: ready for final assurance, contingent on the deferred normal Release build and staged migration validation.