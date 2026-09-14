---
title: "LSTM Phase 20O Scheduler Architectural Convergence Audit"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase20O_SchedulerArchitecturalConvergenceAudit_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Phase 20O Scheduler Architectural Convergence Audit

## Phase 20O result

Architectural convergence has been reached at the service-orchestration level. I did not create another service. I removed only proven dead pre-extraction implementations and their unreachable dependency tail.

Phase 20P recommendation: **GO WITH PREREQUISITES**. The next phase should implement the semantic-worker registry/publish lifecycle and separate scheduler identity from the current worker identity; it should not create `lstm-scheduler` yet.

### 1. Starting state

- Branch: `lstm-feature-development`
- HEAD: `c38838c2311b3984ac7f2c4c5663c10deb6cbf6a`
- Subject: `Route inference workers by semantic layout`
- Initial worktree: clean.
- Active production state:
  - Scheduler PID 29004, fencing token 125.
  - Two managed training workers, experiments 628 and 629.
  - No unmanaged, mismatched, or missing workers.
- No production process was signaled, stopped, resumed, preempted, or otherwise changed.
- The only executable invocation was the read-only `--scheduler-status` command.

### 2. Architecture discovered

Current flow:

```text
LSTM/main.cpp
  -> SchedulerEngine
  -> ExperimentScheduler compatibility entrypoints
  -> SchedulerCycleService
  -> dispatch/orchestration services
  -> SchedulerRepository / PostgresSchedulerRepository
  -> PostgreSQL and process adapters
```

The established services now coherently own:

- Admission/capacity: `SchedulerAdmissionService`
- Authority/lease/fencing: `SchedulerAuthorityService`
- Attempt reservation/spawn persistence: `WorkerAttemptLifecycleService`
- Process operations: `WorkerProcessController` / `WorkerControlService`
- Orphan sequencing: `ReconciliationService`
- Poll sequencing: `SchedulerCycleService`
- Final train/infer/analyze dispatch: `FinalExperimentDispatchService`
- Child completion: `SchedulerChildCompletionService`
- Checkpoint analysis: `CheckpointAnalysisOrchestrationService`
- Checkpoint policy evaluation: `CheckpointEvaluationService`
- Continuation automation: `ContinuationOrchestrationService`
- Operator state transitions: `ExperimentTransitionService`

`SchedulerEngine` remains only a facade over the compatibility entrypoints, not yet an independent composition root: [SchedulerEngine.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/SchedulerCore/SchedulerEngine.cpp:15).

### 3. Remaining responsibility map

| Responsibility | Current ownership | Audit verdict |
|---|---|---|
| Broad CLI parsing and validation | `ExperimentScheduler.cpp` | Correct for current monolithic application edge, but too broad for a standalone scheduler. |
| Scheduler configuration and authority acquisition | Adapter plus `SchedulerAuthorityService` | Correct composition-edge code. |
| Poll preparation and phase order | `SchedulerCycleService` with transaction callbacks | Correct boundary. |
| Final experiment dispatch | `FinalExperimentDispatchService` with DB/command callbacks | Correct boundary. |
| Checkpoint inference | Straight adapter loop using semantic selection, admission, reservation, and launch services | Deliberately retain until executable/runtime adapter split; another callback service would add little isolation. |
| Checkpoint analysis | `CheckpointAnalysisOrchestrationService` with transaction adapters | Correct boundary. |
| Reservation, launch, exact spawn persistence | `WorkerAttemptLifecycleService` plus adapter | Correct boundary. |
| Child completion | `SchedulerChildCompletionService` plus exact-attempt persistence callbacks | Correct boundary. |
| Semantic admission and inference worker routing | `SchedulerSemanticAdmission` / `InferenceWorkerSelection`; DB identity loading remains adapter code | Correct Phase 20N boundary. |
| Preemption and stopped-worker resumption | Policy/admission/control services plus transaction/process adapter | Correct adapter composition, although strongly database-coupled. |
| Missing/orphan reconciliation | `ReconciliationService` owns sequencing; adapter owns evidence and phase-specific persistence | Correct boundary. |
| Continuation and checkpoint policy | Services own decisions/sequencing; adapter owns PostgreSQL workflows | Correct boundary. |
| Operator cancel/retry/requeue | `ExperimentTransitionService` plus CLI/transaction adapter | Correct boundary. |
| Raw SQL and transactions | `PostgresSchedulerRepository` plus substantial compatibility-adapter SQL | Acceptable today, but prevents a clean executable link boundary. |
| Command construction | Application adapter; process launch in process controller | Correct edge responsibility. |
| Scheduler logs/status/reporting | Adapter | Correct edge responsibility. |
| Historical no-PID and semantic compatibility | Active recovery/status paths | Reachable and deliberately retained. |

No remaining scheduler responsibility justified another narrow Phase 20O service.

### 4. Dead/duplicate cleanup

Changed only [ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/SchedulerCore/ExperimentScheduler.cpp).

Removed definition-only, internal pre-extraction implementations:

- Legacy final train/infer/analyze dispatch loops.
- Legacy checkpoint-inference dispatch loop.
- Legacy orphan reconciliation and train/infer completion paths.
- Their now-unreachable process discovery, direct launch/wait, PID persistence, inference-log recovery, and model-ID diagnostic helpers.
- Several already `[[maybe_unused]]` reporting/control helpers with no callers.

Each initial legacy function had exactly one occurrence: its definition. Subsequent removals were driven by compiler `-Wunused-function` diagnostics after those callers disappeared.

The active checkpoint-analysis adapter was renamed from `RunCheckpointEvalAnalyzeJobsLegacy` to `RunCheckpointEvalAnalyzeJobs`, removing its one-line compatibility wrapper. Its implementation and behavior are unchanged.

No historical compatibility path was removed. The tracked `Sources/ExperimentScheduler.cpp` symlink remains because tests and historical tooling consume it.

Line count:

- Before: 27,144
- After: 25,298
- Net: −1,846 lines

This is incidental to removing proven unreachable duplication, not the success criterion.

### 5. Dependency and build graph findings

The Xcode target graph is correctly isolated:

- SchedulerCore implementations are compiled only in the `SchedulerCore` static-library target.
- `LSTM Release` and `LSTM Debug` depend on and link `libSchedulerCore.a`.
- Neither executable target directly compiles `ExperimentScheduler.cpp` or the other SchedulerCore implementations.
- The compatibility symlink is not separately compiled.
- No duplicate implementation membership was found.

However, the library boundary is not yet clean enough for a lightweight executable:

- `ExperimentScheduler.cpp` includes 73 project headers.
- Its dependency file names 137 project paths; the next-heaviest SchedulerCore unit names seven.
- It contains approximately 466 `pqxx::` references and 247 SQL execution calls.
- Pulling `ExperimentScheduler.o` introduces at least 131 unique unresolved dependencies on campaign, recommendation, profitability, economic-calendar, and other non-scheduler application components.
- `SchedulerInferenceResultRecovery.hpp` exposes `pqxx::transaction_base` and inline SQL. This should eventually move behind a PostgreSQL adapter.
- `InferenceWorkerSelection.hpp` carries POSIX filesystem implementation and the large semantic-expansion header, although current fan-out is only two consumers.
- `SchedulerEngine` depends inward on the broad compatibility implementation, rather than exposing a typed scheduler-runtime composition API.

No speculative header rewrite was made.

### 6. Build and tests

Final focused build:

```bash
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme SchedulerCore \
  -configuration Release \
  -derivedDataPath "$PWD/DerivedData/ExpertAdvisor" \
  build
```

Result: `BUILD SUCCEEDED`.

Fan-out:

- `CompileC`: 1 — `ExperimentScheduler.cpp`
- `Libtool`: 1 — `libSchedulerCore.a`
- Executable links: 0
- Unrelated compilation: 0

Immediate unchanged repeat:

- `BUILD SUCCEEDED`
- `CompileC`: 0
- `Libtool`: 0
- `Ld`: 0

The prompt’s example using `-target SchedulerCore` was initially attempted, but Xcode 26.5 rejects `-derivedDataPath` without `-scheme`; it was replaced with the equivalent SchedulerCore scheme command.

All passed:

```text
FinalExperimentDispatchServiceTests
OrphanedRunningExperimentReconciliationServiceTests
SchedulerCycleServiceTests
CheckpointAnalysisOrchestrationServiceTests
SchedulerChildCompletionServiceTests
WorkerAttemptLifecycleServiceTests
SchedulerSemanticAdmissionTests
SchedulerCoreBoundaryTests
SchedulerInternalSeamTests
LegacyLayout6WorkerCompatibilityTests
```

The layout-6 check reconfirmed:

- Commit `7645265bca0c2529523e1d2cdb37e7d023dfd559`
- SHA-256 `945225dd2a42f87a2a8dfbfe47b006708e3d90c88a858d25787e5a2237c62dd7`
- Semantic layout 6
- Exact worker-attempt registration
- Final and checkpoint inference persistence

The final build still reports the existing `-Ofast` and libpqxx `exec_params` deprecation debt. It has no unused-function warning.

A full `LSTM Release` build was not warranted: no public interface, Xcode linkage, target membership, or active behavior changed. It would also encounter the deliberate dirty-worktree provenance barrier. Database/process integration suites were not rerun while production workers were active.

### 7. Phase 20N invariants

Phase 20N-sensitive files were unchanged, and focused admission/lifecycle/layout-6 tests passed.

Confirmed:

- Scheduler authority still uses the scheduler executable identity.
- Layout 7 selects the current worker.
- Layout 6 selects only the explicitly configured canonical legacy worker.
- Missing, incomplete, unknown, unsupported, and width-mismatched identities fail closed.
- No `layout <= current` shortcut exists.
- Train and analyze remain on the current executable.
- Final and checkpoint inference share the same selector.
- Selected identity is persisted before launch and matched exactly during spawn registration and reconciliation.
- The legacy flag is not propagated to child argv.
- Capacity, priority, preemption, stopped-worker, and authoritative-result rules are unchanged.
- No production-specific routing for IDs 620, 622, or 623 exists in SchedulerCore.

### 8. Semantic-worker artifact lifecycle design

Proposed durable root:

```text
/Volumes/Developer SSD/ExpertAdvisor/Builds/SemanticWorkers/
  layout<N>/<git-commit>/<sha256>/LSTM_Release
  layout<N>/<git-commit>/<sha256>/manifest.json
  registry.json
```

`Builds/` is already ignored. Binary artifacts and the operational registry should not be source-controlled. Source-control only:

- Registry/manifest JSON schema.
- Publish/archive implementation.
- Validation logic and tests.

Registry entries should contain:

```text
schema version
semantic layout
supported input widths
allowed phases/capabilities
Git commit
SHA-256
artifact-root-relative executable path
compiler/Xcode/SDK provenance
publication timestamp
```

Exact publish sequence:

1. Complete the normal clean Release build in stable `DerivedData/ExpertAdvisor`.
2. Determine layout, width, capabilities, embedded commit, Xcode, SDK, and compiler provenance.
3. Hash the built executable.
4. Copy it to a same-filesystem staging directory under the durable root.
5. Verify executable permissions, copied hash, embedded commit, and capability metadata.
6. Atomically rename into the content-addressed final directory; never overwrite a conflicting artifact.
7. Write and validate an updated registry temporary file.
8. Retain the previous current entry unchanged, mark the new entry current, then atomically rename the registry.
9. Only afterward update any convenience symlink.

A failed build, copy, hash check, manifest check, or registry validation must leave the active registry unchanged.

The existing `Publish LSTM Canonical` aggregate target is the correct explicit post-build hook, but its current `DerivedData/Canonical/LSTM_Release` symlink is not durable. Replace its inline symlink script with a source-controlled publishing tool; ordinary compile targets must not mutate archived artifacts.

At rollover:

- Layout 7’s immutable entry does not move or change.
- `current_layout` changes to layout 8.
- New attempts select the layout-8 artifact.
- Existing layout-7 attempts keep their already-persisted exact path.
- Reconciliation, registration, pause/resume, preemption, and orphan recovery continue matching that exact immutable path.

Retention must be operator-controlled. No artifact may be deleted while any persisted model/checkpoint, queued experiment, replay workflow, or active/terminal retained attempt can require its layout.

Startup diagnostics should distinguish:

```text
semantic_worker_registry_missing
semantic_worker_artifact_missing
semantic_worker_hash_mismatch
semantic_worker_layout_unsupported
semantic_worker_capability_mismatch
```

Migration of the Phase 20N layout-6 path:

1. Copy the already validated binary byte-for-byte into the durable root.
2. Verify its known commit/hash.
3. Create an immutable layout-6 inference-only manifest.
4. Add it to `registry.json`.
5. Keep `--legacy-layout6-infer-worker` temporarily as an explicit fallback; reject conflicting flag/registry identities.
6. Remove the flag after operational migration is validated.

### 9. Phase 20P readiness

**GO WITH PREREQUISITES**

Positive evidence:

- Scheduler services are testable without linking the monolith.
- Repository and process abstractions exist.
- Scheduler authority can store a distinct canonical executable.
- Phase 20N established exact selected-worker identity for inference.
- The Xcode target graph already links SchedulerCore as a library.
- A standalone executable would materially reduce build/link scope once the giant compatibility object is no longer pulled in.

Blocking prerequisites:

1. Replace the one-off layout-6 flag with `SemanticWorkerRegistry` and durable immutable publication.
2. Split `SchedulerOptions::selfPath` into:
   - scheduler canonical executable;
   - current published worker executable;
   - registry-selected worker executable.
3. Apply selected worker identity to train and analyze as well as inference. Today train/analyze still use `selfPath`.
4. Extract the scheduler-daemon-only parser/runtime adapter from the broad 25K-line CLI object.
5. Make `SchedulerEngine` a true typed composition API rather than a facade calling `RunExperimentSchedulerCli`.
6. Move `gSchedulerOwnedChildren` into a scheduler runtime context. The signal flag may remain a process-local signal-safe bridge.
7. Move the inline pqxx inference-result recovery helper behind the PostgreSQL adapter.

Recommended Phase 20P scope: implement prerequisites 1–3 as one cohesive semantic-worker registry/publish and executable-identity separation increment. Do not create `lstm-scheduler` in that phase. A following executable phase can then extract the narrow daemon parser/runtime adapter and add the new target without reopening worker-selection semantics.

Initially, `LSTM_Release` should retain train/infer/analyze execution, exact worker self-registration, model/Tensor/training code, and existing operator/recommendation/campaign commands.

### 10. Final repository state

- Schema migrations: none.
- Files changed: one.
- Changes are unstaged and uncommitted.
- `git diff --check`: passed.
- `git diff --stat`:

```text
 Sources/SchedulerCore/ExperimentScheduler.cpp | 1882 +------------------------
 1 file changed, 18 insertions(+), 1864 deletions(-)
```

- `git status --short`:

```text
 M Sources/SchedulerCore/ExperimentScheduler.cpp
```

Remaining risks are the broad compatibility object inside SchedulerCore, incomplete PostgreSQL isolation, dual-use `selfPath`, existing compiler deprecation warnings, and the current layout-6 artifact’s dependence on disposable DerivedData until the proposed registry migration is implemented.