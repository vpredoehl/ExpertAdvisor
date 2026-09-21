---
title: "Phase 23A2 Deterministic Managed-Inference End-to-End Fixture and Compatibility/Standalone A/B Validation"
document_type: "validation report"
status: "final"
---

# Phase 23A2: GO

Phase 23A2 closes the Phase 23A1 validation gap with a deterministic, disposable managed-inference fixture. The validation source baseline was `d54b89b Fix standalone managed inference optional symbol fallback`; the fixture and its Release validation were performed at `1e5228c Add deterministic managed inference A/B fixture`.

## Fixture and isolation

Each of the four runs (compatibility A and standalone B, twice each) creates fresh LSTM and Forex databases. Their schemas are read-only schema-only clones of `LSTM` and `forex`; the test writes only its uniquely prefixed disposable databases. A temporary semantic registry copies the runtime and existing train artifact into a `mktemp` directory and supplies the selected test executable as a layout-7 infer worker. It never reads or writes a production registry entry.

The fixture uses 2,304 deterministic five-minute `phase23a2audrmp` Forex bars beginning 2024-01-01, covering full-history warmup and the 2024-01-05 through 2024-01-06 inference interval. It persists causal USD CPI, WEEKLY_CLAIMS, PCE, and subsequent WEEKLY_CLAIMS history; a selected Myfxbook pre-release Weekly Claims consensus; and DOL ETA authoritative release/actual provenance. It calls the real calendar snapshot implementation, then creates the experiment before its model, avoiding an invalid FK/trigger ordering. The model contains the production-required 78x4 parameter matrix, metadata, return-direction head, full-history/donchian metadata, and `train_symbol_meta=phase23a2audrmp`.

The scheduler—not the fixture—creates each exact worker attempt. The test waits with kqueue `KQ_NOTE_EXIT` and runs supported orphan/result reconciliation. Cleanup restricts `dropdb` to the generated names and removes every temporary registry, log directory, fixture executable, and database.

## A/B result

Release executable SHA-256 values used by the post-build run:

- Compatibility `LSTM_Release`: `b98d2b134fa25e4d3cf86bc4a771b86016f21398666f1ff6e7f307d31463ebd1`
- Standalone `lstm-infer-worker`: `1a32e91fed1aa9be405a8b6103a4c135617cd5863863972cdd177aa9b94391c4`
- Scheduler: `bff21df38d46965383e2a4f9b8e91436250ab2b55afc2f36171a82327bdb6ae4`

Both paths succeeded twice via the supported lifecycle and emitted `managed_application_success_return`; standalone had no signal termination or SIGTRAP. Each run ended with `experiment=pending:analyze`, a scheduler-created attempt in `completed:process_missing_result_recovered:NULL`, exactly one completed final `inference_eval_result`, and exactly one linked final profitability observation.

The normalized durable semantic digest was exactly equal in A1, B1, A2, and B2:

`60f42065747e58843ffc11763fd5c7da7c1ce53820c5735b7344f132bc6ef1f8`

The comparison is exact (no numeric tolerance). It includes persisted symbol, horizon, threshold, window, label/target identities, range, completed epochs, accuracy, acceptance/rejection, directional fractions, profitability counts, sums, per-actionable average, metric definition/hash, and source hash. It excludes only independently allocated surrogate model IDs; timestamps, PIDs, worker-attempt IDs, and executable paths are not queried.

## Lifecycle and atomicity evidence

For each path the worker log asserted exact registration and, in order, detached RR/RO materialization start/completion, its commit, evaluation start/completion, fresh RW persistence start, scheduler identity/state revalidation, result persistence, profitability persistence, combined persistence completion, transaction commit, and successful application return. The durable count assertions after reconciliation prove that the committed final result and linked profitability observation are both present; the existing orphan recovery regression also passed.

## Regression and build validation

Passed:

- `Tests/LSTMPhase23A2ManagedInferenceEndToEndABTests.sh` (post-build; A/B twice)
- `Tests/LSTMPhase23A1StandaloneManagedInferenceSIGTRAPTests.sh`
- `Tests/LSTMPhase22Z1InferenceEvaluationFactsTests.sh`
- `Tests/LSTMPhase22Z3InferenceRuntimeCompositionBoundaryTests.sh`
- `Tests/LSTMPhase22Z4ManagedInferenceApplicationBoundaryTests.sh`
- `Tests/LSTMPhase22Z5ThinStandaloneInferenceWorkerTests.sh`
- `Tests/SemanticWorkerRegistryTests.sh` and `Tests/SemanticWorkerPublisherTests.sh`
- `Tests/WorkerAttemptLifecycleServiceTests.sh`
- `Tests/WorkerAttemptReconciliationIntegrationTests.sh` (with the current required isolated helper build)
- `Tests/SchedulerInferenceOrphanResultRecoveryTests.sh`
- `Tests/EconomicCalendarSnapshotTests.sh` and `Tests/EconomicEventFeaturesRealInputIntegrationTests.sh`

`LSTM Release`, `LSTM Infer Worker`, and `LSTM Scheduler Bundle` Release builds all succeeded with `-derivedDataPath DerivedData/ExpertAdvisor`. The economic calendar test was narrowly updated so an already-current schema clone does not replay migration 091's intentionally non-idempotent `ALTER TABLE`; it still tests migration 092 replay, snapshot immutability, deterministic ordering, CLI creation/reuse, and binding behavior.

## Production-safety audit

The audit found no Phase 23A2 disposable databases remaining. Production experiment 648 remains `failed:infer` with model 1914 and 649 remains `failed:infer` with model 1916. Historical attempts 1129 and 1130 remain `failed:parent_observed_exit:-5`. The train-only production scheduler remained `--max-infer-procs=0 --max-analyze-procs=0`; experiments 650/651 stayed running/train under their pre-existing worker PIDs 95926/95985.

The operational registry SHA-256 remains `23f8109ddba8cc3e0a4346b8c42d599dcceacd83587f404214368b3913d96d92`, and the published layout-7 worker remains byte-identical at SHA-256 `ac5023fa912b93c34679e984c4fb5d76bc6d83f6d44c84d5429159da210f3a60`. No executable below the live scheduler was replaced.

## Boundary and next action

This is validation only. Do not publish the tested standalone executable, modify the operational registry, requeue 648/649, or enable production inference/analyze. The next explicit phase should independently authorize publication/cutover of a newly immutable standalone layout-7 worker, verify its registry/runtime provenance, and only then decide recovery using fresh scheduler-created attempts.
