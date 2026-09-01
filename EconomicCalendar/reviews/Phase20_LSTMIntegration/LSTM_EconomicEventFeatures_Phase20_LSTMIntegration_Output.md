# Phase 20 LSTM integration output

Date: 2026-09-01

## Continuation recovery

Phase 20 resumed from the authoritative uncommitted worktree on branch
`lstm-feature-development` at
`c0e9cd9c771360b61b1b21f1dc2c9f734ed1b35a`. The starting continuation diff
contained 10 modified tracked files and five untracked Phase 20 artifacts. No
change was reset, restored, checked out, discarded, or overwritten. The
existing pre-implementation audit was preserved.

Review of the interrupted implementation found useful, correct work in all
four gap areas, plus four items requiring repair:

- migration 089's CHECK expression allowed partial NULL identity because a
  PostgreSQL CHECK accepts UNKNOWN;
- meta-recommendation and recommendation-conversion experiment insertions had
  not been updated;
- continuation's custom equivalent-child lookup bypassed the new identity
  dimensions;
- the focused persistence test used invalid Tensor/layout APIs and lacked a
  fresh width-75 save/reload assertion.

Only those issues and their focused regression coverage were repaired.

## Established layout and original audit findings

The preserved current-state audit remains authoritative:

- 49 pre-economic Tensor columns;
- occurrence/recency at Tensor 49-58;
- consensus-era columns at Tensor 59-66, with provider-surprise positions
  63-66 permanently closed zero;
- authoritative initial-release surprise at Tensor 67-70;
- four model-only causal return columns at model inputs 71-74;
- 71 physical Tensor columns, 75 total model inputs, semantic layout V5.

`Tensor::Add` remains the sole append-only economic-feature construction path.
Training, final inference, checkpoint inference, and ordinary inference still
use the common `ModelInputContract`. The established 71-to-75 append-only
expansion remains the sole expansion mechanism. No scheduler-only economic
configuration and no event/consensus/actual schema migration were added.

## Four confirmed gaps and implementation

### 1. Experiment model-input identity

Migration `089_lstm_model_input_identity.sql` adds nullable historical columns
`model_input_width` and `model_input_semantic_layout_version` without
backfilling or reinterpreting pre-089 experiments. A trigger requires both
values on every new INSERT and rejects any later UPDATE that changes either
value. The shape CHECK rejects partial identity and non-positive values.

The migration recreates `experiment_unique_identity_uidx` with every
immediately preceding migration-079 dimension, then appends width and semantic
layout before `duplicate_nonce`. A disposable migration test compares all 21
ordered key expressions and the partial-index predicate exactly. It also
proves historical NULL/NULL retention, new NULL/NULL rejection, partial
rejection, immutability, width distinction, layout distinction, and exact
duplicate rejection.

All three production experiment INSERT authorities now write identity:

- scheduler queue/sweep/resume/explicit expansion/manual and automatic
  continuation through `InsertExperimentRecord`;
- meta-recommendation queueing;
- recommendation-conversion and Campaign Manager materialization.

Fresh experiments use width 75/layout V5. Ordinary resumes validate the source
model and persist its structural width with current compatible V5 prefix
semantics. Explicit expansion persists 75/V5. Recommendation conversion uses
the same source-model structural rule for ordinary resumes. The custom
continuation-equivalence lookup includes width and layout, so it cannot treat
a legacy or differently bound child as the current binary's equivalent.

Scheduler-launched training, final-inference, and checkpoint-inference workers
read the persisted experiment identity. A mismatch between that identity and
the resume/inference model width fails before model-input materialization.
Model saves linked to a post-089 experiment also fail before parameter writes
when runtime model width or layout is inconsistent with the experiment.
Historical NULL/NULL experiments retain the pre-089 compatibility path.

### 2. Ordinary-load semantic validation

`PgModelIO::loadAll` now validates `model_input_semantics_meta` before loading
ordinary resume or inference parameters. Marker-bearing models must have a
registered, append-only-compatible semantic generation for their structural
width; corrupt or incompatible markers fail closed. The same validator is
used by queue-time ordinary resume and explicit expansion, eliminating the
previous contradiction. Marker-less historical models still use the
registered-width compatibility path and remain loadable.

Read-only metadata validation accepts `pqxx::transaction_base`, allowing the
recommendation/campaign materialization transaction to reuse authoritative
`PgModelIO` validation rather than duplicating model-meta parsing.

### 3. Empty economic-event corpus

`LoadEconomicEventsForFeatureRange` now checks for any row in the requested
currency corpus before executing the causal range query. A wholly unavailable
USD corpus throws
`economic_event_feature_source_history_unavailable:USD`. An interval with no
events remains valid when USD history exists elsewhere. The range query,
seed-selection logic, consensus/actual semantics, and point-in-time cutoffs
are unchanged.

### 4. Width-75 nonzero event smoke

The focused persistence integration test now creates a proven PCE event with
selected consensus and a provenance-bearing authoritative initial actual. It
asserts nonzero occurrence and initial-surprise Tensor values, constructs a
fresh width-75 LSTM, calculates finite inference probabilities, performs a
real training update, saves the model against a width-75/V5 experiment,
reloads it, verifies exact persisted parameters and width metadata, and
repeats inference with identical probabilities. A mismatched width-51 save
against a width-75 experiment is rejected before parameter persistence.

## Causality and leakage assessment

Phase 20 does not alter feature values or temporal selection. Existing focused
tests reconfirm future-event exclusion, completed-bar release boundaries,
causal consensus exposure, strict authoritative-actual `available_at`,
revision exclusion, provider-actual exclusion, and deterministic missing-value
availability/zero behavior. Malformed and provenance-invalid data still fail.
The only new source behavior distinguishes a globally unavailable currency
corpus from a legitimate event-free interval.

## Files changed

- `Database/README.md`
- `Database/migrations/089_lstm_model_input_identity.sql`
- `EconomicCalendar/reviews/Phase20_LSTMIntegration/CurrentStateAudit.md`
  (preserved audit artifact)
- `EconomicCalendar/reviews/Phase20_LSTMIntegration/LSTM_EconomicEventFeatures_Phase20_LSTMIntegration_Output.md`
- `Headers/ModelInputExpansion.hpp`
- `Headers/PgModelIO.hpp`
- `LSTM/ExperimentMetaAnalyzer.cpp`
- `LSTM/main.cpp`
- `Sources/EconomicEventFeatureLayout.hpp`
- `Sources/EconomicEventRepository.cpp`
- `Sources/ExperimentRecommendationConversionExecutionRepository.cpp`
- `Sources/ExperimentScheduler.cpp`
- `Tests/EconomicEventFeatureRangeRepositoryTests.cpp`
- `Tests/EconomicEventTensorIntegrationTests.cpp`
- `Tests/ExperimentRecommendationConversionExecutionRepositoryTests.cpp`
- `Tests/LSTMInputWidthExpansionPersistenceTests.cpp`
- `Tests/LSTMInputWidthExpansionPersistenceTests.sh`
- `Tests/LSTMInputWidthExpansionSchedulerIntegrationTests.sh`
- `Tests/LSTMModelInputIdentityMigrationTests.sh`
- `Tests/LSTMModelInputIdentityMigrationTests.sql`

## Validation results

Completed safe validation:

- `bash Tests/LSTMModelInputIdentityMigrationTests.sh` — PASS; disposable DB
  dropped.
- `bash Tests/LSTMModelInputCompatibilityTests.sh` — PASS.
- `bash Tests/LSTMInputWidthExpansionTests.sh` — PASS.
- `bash Tests/EconomicEventFeaturesTests.sh` — PASS,
  `ECONOMIC_EVENT_FEATURES_TEST_PASS`.
- `bash Tests/EconomicEventActualPointInTimeTests.sh` — PASS,
  `PHASE12_POINT_IN_TIME_FEATURES=PASS`; disposable DB dropped.
- `bash Tests/EconomicEventFeatureRangeRepositoryTests.sh` — PASS twice after
  the final event-free-interval assertion; disposable DB dropped each time.
- `bash Tests/EconomicEventTensorIntegrationTests.sh` — PASS.
- syntax-only compilation of the modified recommendation-conversion repository
  and its focused repository test with warnings-as-errors — PASS.
- syntax-only compilation of the modified width-expansion persistence test with
  warnings-as-errors — PASS after repairing the interrupted invalid API uses.

Final Metal persistence, isolated scheduler, and Release-build results are
recorded below after the active production workers release resources.

## Production safety

Before resource-sensitive validation, the canonical scheduler status was read
without mutation. Scheduler authority was active, global execution was
running, and exactly two managed training workers occupied both configured
training slots: experiment 609 (historical width) and experiment 610 (the first
71-to-75 continuation from model 1702). No worker, experiment, scheduler row,
production model, or production economic-event row was stopped, paused,
resumed, requeued, cancelled, or modified. Migration 089 was not applied to
production. Every database test used a disposable database.

## Build and remaining validation

Pending final isolated Release build and resource-sensitive focused tests.

## Remaining risks and readiness assessment

Historical pre-089 experiments intentionally remain NULL/NULL and therefore
cannot be retroactively bound to the queuing binary; this is the explicit
compatibility tradeoff that avoids reinterpreting production history. No
production scheduler cutover or migration application was performed in Phase
20 validation.

Subject to the final isolated Release build and remaining focused tests, the
implementation is production-ready for a separately authorized migration and
binary deployment. The recommended next development step is an operational
deployment plan that applies migration 089 transactionally, verifies the
schema ledger and new-insert trigger, deploys the matching Release binary, and
queues one non-production canary experiment before any production scheduler
cutover.
