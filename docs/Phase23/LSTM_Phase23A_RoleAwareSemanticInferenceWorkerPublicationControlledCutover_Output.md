# LSTM Phase 23A — Role-Aware Semantic Inference Worker Publication and Controlled Cutover

## Final disposition

Phase 23A role-aware semantic inference-worker publication and controlled cutover: IMPLEMENTATION COMPLETE — PENDING POST-COMMIT RELEASE VALIDATION

## Baseline, candidate, and safety

- Branch: `lstm-feature-development`; baseline HEAD: `bdb4d905b5badedfcaa3706bed9a00ec03762800`; baseline worktree: clean.
- Recent commits: `bdb4d90`, `7050298`, `8538d33`, `87e0d9a`, `d2fcb8d`.
- Two active production `LSTM_Release` training workers (scheduler attempts 1127 and 1128) were detected before edits and validation. Neither was signalled, stopped, resumed, preempted, replaced, or otherwise disturbed. No production experiment row was changed.
- Candidate evidence was independently verified before publication and from the immutable copy: `artifact_role=lstm-infer-worker`, `source_commit=bdb4d905b5badedfcaa3706bed9a00ec03762800`, and `executable_sha256=sha256:11f822ee12497781eca6fc942ddb422224cd9f5ed6e41291d0cf2a79dd778941`.

## Audit architecture

| Contract | Location | Finding |
| --- | --- | --- |
| Registry/parser | `Sources/SchedulerCore/SemanticWorkerRegistry.hpp/.cpp`; `docs/semantic-worker-registry.schema.json` | Operational registry was v2 and keyed only by layout. The transitional v3 role field could not hold two artifacts for layout 7 because the C++ map remained layout-keyed. |
| Publication | `Scripts/PublishSemanticWorker.py` | Existing locked staging, canonical path, manifest, SHA, runtime-link, fsync, rename, and atomic registry-replace protocol is authoritative and was extended rather than duplicated. |
| Existing layout 7 | Prior `Builds/SemanticWorkers/registry.json` | Training/reference `LSTM_Release`: `layout7/5bae79ef0ebc5ac42cb5e489262d3971ee629968/ac5023fa912b93c34679e984c4fb5d76bc6d83f6d44c84d5429159da210f3a60/LSTM_Release`; commit `5bae79ef0ebc5ac42cb5e489262d3971ee629968`; SHA `ac5023fa912b93c34679e984c4fb5d76bc6d83f6d44c84d5429159da210f3a60`. |
| Final/checkpoint selection | `ProductionSchedulerDaemon.cpp`: `LoadInferenceWorkerSelection` (1910), `ReserveExperimentWorkerAttempt` (2841), `ReserveCheckpointWorkerAttempt` (2930) | Both use `SelectInferenceWorker`; no direct or hard-coded infer-worker path exists. |
| Command construction | `BuildInferCommand` (3435), `BuildCheckpointEvalInferCommand` (3490) | Selected executable is argv[0]; all other argv fields are unchanged. |
| Training/resume/reference | non-infer `ReserveExperimentWorkerAttempt`; `SemanticWorkerRegistry::currentWorker` | Training remains `options.currentWorkerExecutablePath`. v4 `currentWorker()` resolves training/reference; v2/v3 retain the legacy inference fallback. |
| Attempts/process/recovery | `WorkerAttemptLifecycleService.cpp`, `WorkerProcessController.cpp`, `ReconciliationService.*`, `SchedulerInferenceResultRecovery.hpp`, `SchedulerChildCompletionService.*` | Exact reservation/attempt identity and canonical-executable checks are unchanged. SIGSTOP/SIGCONT remain `WorkerProcessSignal::{Stop,Resume}`. |
| Managed inference lifecycle | `ManagedInferenceApplication.cpp`, `ManagedInferenceWorkerCli.cpp` | One application-owned registration, RR/RO commit-before-compute, fresh RW lock/revalidation, final idempotency, checkpoint dedupe, and atomic result/profitability persistence are unchanged. |

Semantic layout is persisted model input layout; executable role is `train` or `infer`; artifact identity is immutable path plus manifest/commit/SHA; scheduler protocol identity is invocation/fencing; worker-attempt identity is exact reserved attempt/launch identity. They are not conflated.

## Implementation and publication

Schema v4 keys workers by `(semantic_layout, worker_role)`. Legacy v2/v3 records still parse. Publisher migration makes the old current v2 layout-7 `LSTM_Release` explicit for both established roles, then replaces only `(7,infer)`. No role is inferred from filename.

```text
layout 7, train -> existing immutable LSTM_Release
layout 7, infer -> immutable lstm-infer-worker
```

The loader fails closed for malformed/duplicate role bindings, absent requested role, capability mismatch, invalid manifest, SHA/commit mismatch, noncanonical paths, runtime-link failures, and unsupported layouts. The publisher refuses an inference-only bootstrap or a current-layout change without an existing training/reference binding.

The immutable artifact was published from a clean detached worktree at the candidate commit, while implementation source remained dirty:

```text
Builds/SemanticWorkers/layout7/infer/bdb4d905b5badedfcaa3706bed9a00ec03762800/
  11f822ee12497781eca6fc942ddb422224cd9f5ed6e41291d0cf2a79dd778941/
  lstm-infer-worker
```

Its schema-v2 manifest has role `infer`, layout `7`, width `77`, the exact candidate commit/SHA, and `lstm-infer-worker` identity. The real registry is now v4, preserves layout 6 unchanged, retains the layout-7 training/reference artifact above, and selects the new inference artifact. Artifacts and registry are ignored operational files, not Git source changes.

Final and checkpoint argv differ before/after only at argv[0]; model, final/checkpoint binding, dates, Donchian, warmup, profile, and scheduler-attempt options remain identical. Worker reservation and application registration remain unchanged.

## Validation

Passed:

- Debug builds: `LSTM Debug`, `LSTM Release`, and `LSTM Infer Worker` (incremental `xcodebuild`, no clean).
- Phase regression scripts: 22Z1, 22Z3, 22Z4, 22Z5, 22U, 22V, and 22W.
- `EA_SEMANTIC_REGISTRY_UNDER_TEST="$PWD/Builds/SemanticWorkers/registry.json" Tests/SemanticWorkerRegistryTests.sh`.
- `Tests/SemanticWorkerPublisherTests.sh`, `Tests/SchedulerSemanticAdmissionTests.sh`, and `Tests/SchedulerChildCompletionServiceTests.sh`.
- `python3 -m py_compile Scripts/PublishSemanticWorker.py` and `git diff --check`.

Focused Phase 23A tests cover v2 compatibility, v4 dual-role parsing/selection, training/reference preservation, selected-inference separation, actual published-registry loading, runtime links, malformed/tampered identity rejection, atomic replacement failure, immutable path refusal, and role-preserving publication. Phase 22Z5 confirms argv shape, exact registration ownership, status 125 on registration failure, detached materialization/persistence structural boundaries, and unrelated-mode rejection.

Deferred or not passed:

- DB-backed scheduler semantic admission, detached-materialization concurrency, orphan/result recovery, and SIGSTOP/SIGCONT integration were deliberately not run while production training was active.
- `SchedulerPriorityPreemptionMigrationTests.sh` was started before inspecting its target. It created a uniquely named disposable DB, failed its first fixture insert on current-schema `experiment_scheduler_resume_state_check`, then exited; it did not target or change production rows and was not retried.
- `SchedulerCanonicalPathTests.sh` has a pre-existing include-path failure for `SchedulerCore/SchedulerAuthorityService.hpp`.
- `InferenceProfitabilityTests.sh` compiled but its assertion binary exited nonzero without a diagnostic; it was not altered because the failure is unrelated to role selection.

## Cutover and rollback

No scheduler was running at publication. Registry is loaded once at scheduler startup (`SemanticWorkerRegistryFor`), therefore a restart is required for an already-running scheduler to consume v4. Publication itself does not alter active workers; their persisted canonical path survives reconciliation.

Before operator restart, inspect scheduler and worker processes and preserve the exact scheduler executable, `--schedule-experiments`, `--semantic-worker-registry`, worker-limit, log-path, and environment arguments. Do not signal active workers as part of this phase. Start the committed scheduler binary with those preserved arguments. Confirm first-cycle final/checkpoint selection logs the immutable `lstm-infer-worker` path and training remains on immutable `LSTM_Release`.

Rollback is a validated registry-only selection change: restore `(layout=7, role=infer)` to the existing immutable layout-7 `LSTM_Release` identity above, retain `(7,train)`, validate, then operator-restart with the same arguments. Never overwrite, rename, or delete either artifact. The role fixture proves the rollback selection without artifact mutation.

## Files changed

- `Scripts/PublishSemanticWorker.py`
- `Sources/SchedulerCore/SemanticWorkerRegistry.hpp`
- `Sources/SchedulerCore/SemanticWorkerRegistry.cpp`
- `Tests/SemanticWorkerPublisherTests.py`
- `Tests/SemanticWorkerRegistryTests.cpp`
- `Tests/LSTMPhase22Z5ThinStandaloneInferenceWorkerTests.sh`
- `docs/semantic-worker-registry.schema.json`
- `docs/semantic-layout-inference-worker-routing.md`
- this report

## Required post-commit commands

1. `git add Scripts/PublishSemanticWorker.py Sources/SchedulerCore/SemanticWorkerRegistry.hpp Sources/SchedulerCore/SemanticWorkerRegistry.cpp Tests/SemanticWorkerPublisherTests.py Tests/SemanticWorkerRegistryTests.cpp Tests/LSTMPhase22Z5ThinStandaloneInferenceWorkerTests.sh docs/semantic-worker-registry.schema.json docs/semantic-layout-inference-worker-routing.md docs/Phase23/LSTM_Phase23A_RoleAwareSemanticInferenceWorkerPublicationControlledCutover_Output.md`
2. `git commit -m "Publish role-aware semantic inference worker"`
3. `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build`
4. `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Infer Worker" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build`
5. `DerivedData/ExpertAdvisor/Build/Products/Release/lstm-infer-worker --build-identity`
6. `shasum -a 256 Builds/SemanticWorkers/layout7/infer/bdb4d905b5badedfcaa3706bed9a00ec03762800/11f822ee12497781eca6fc942ddb422224cd9f5ed6e41291d0cf2a79dd778941/lstm-infer-worker`
7. `EA_SEMANTIC_REGISTRY_UNDER_TEST="$PWD/Builds/SemanticWorkers/registry.json" Tests/SemanticWorkerRegistryTests.sh`

## Completion worktree evidence

`git status --short`:

```text
 M Scripts/PublishSemanticWorker.py
 M Sources/SchedulerCore/SemanticWorkerRegistry.cpp
 M Sources/SchedulerCore/SemanticWorkerRegistry.hpp
 M Tests/LSTMPhase22Z5ThinStandaloneInferenceWorkerTests.sh
 M Tests/SemanticWorkerPublisherTests.py
 M Tests/SemanticWorkerRegistryTests.cpp
 M docs/semantic-layout-inference-worker-routing.md
 M docs/semantic-worker-registry.schema.json
?? docs/Phase23/
```

`git diff --stat` (tracked source changes; the report directory is untracked
until staged):

```text
 Scripts/PublishSemanticWorker.py                   | 105 ++++++++++++++------
 Sources/SchedulerCore/SemanticWorkerRegistry.cpp   | 106 +++++++++++++++------
 Sources/SchedulerCore/SemanticWorkerRegistry.hpp   |  11 ++-
 Tests/LSTMPhase22Z5ThinStandaloneInferenceWorkerTests.sh | 8 +-
 Tests/SemanticWorkerPublisherTests.py              |  76 +++++++++++----
 Tests/SemanticWorkerRegistryTests.cpp              |  91 ++++++++++++++++++
 docs/semantic-layout-inference-worker-routing.md   |  32 ++++---
 docs/semantic-worker-registry.schema.json          |   8 +-
 8 files changed, 342 insertions(+), 95 deletions(-)
```
