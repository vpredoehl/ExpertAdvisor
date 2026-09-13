---
title: "LSTM Phase 20K Scheduler Final Experiment Phase Dispatch Decomposition"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase20K_SchedulerFinalExperimentPhaseDispatchDecomposition_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Phase 20K Scheduler Final Experiment Phase Dispatch Decomposition

Implemented Phase 20K as exactly one extraction: `FinalExperimentDispatchService`, covering final train/infer/analyze phase-dispatch orchestration.

1. Baseline

- Starting branch: `lstm-feature-development`
- Starting/current HEAD: `57db05d69987ae2da17905013e3e819101d4f50c`
- HEAD subject: `Archive Phase 20 architecture reviews`
- `99f8524` is confirmed in history.
- Starting worktree: clean.
- Starting `ExperimentScheduler.cpp`: 27,112 lines.

2. Inventory and ownership

Common orchestration formerly owned by the three active paths:

- Load phase candidates and initial capacity.
- Iterate candidates in repository order.
- Apply phase-specific semantic, preemption, stopped-worker, eligibility, and dry-run guards.
- Reserve worker attempts.
- Distinguish capacity exhaustion from claim changes.
- Prepare and launch workers.
- Aggregate launch failures and phase statistics.

Phase-specific behavior preserved:

- TRAIN: semantic preflight; preemption except cancellation-only; stopped-worker admission; dry-run before resume-model validation; resume-model validation; cancellation-aware reservation; start diagnostic before launch.
- INFER: semantic preflight; preemption; stopped-worker admission; model/result/forced-rerun eligibility before dry-run; reservation and launch.
- ANALYZE: requires `last_model_id`; no semantic/preemption/stopped-worker path; reservation and launch.

Called, but not semantically owned:

- Admission/capacity and candidate ordering: `SchedulerAdmissionService`
- Authority/fencing: `SchedulerAuthorityService`
- Reservation/spawn persistence: `WorkerAttemptLifecycleService`
- Process launch/control: existing process/control services
- Semantic admission and legacy identity resolution: existing semantic admission code
- Database mutations and result transitions: retained transaction-scoped adapters
- Cycle sequencing: `SchedulerCycleService`

3. Selected responsibility

`SELECTED PHASE 20K RESPONSIBILITY: final-experiment phase dispatch orchestration across train, infer, and analyze candidates.`

This was not already owned by another service: `SchedulerCycleService` chooses when phases run, while subordinate services own individual admission, authority, lifecycle, and process operations. No existing service owned the per-candidate dispatch protocol tying those operations together.

4. Control flow

Before:

```text
TRAIN
load/capacity transaction
→ semantic preflight
→ optional preemption
→ stopped-worker admission
→ dry-run
→ resume-model validation transaction
→ reserve transaction
→ build/start diagnostic
→ launch/spawn persistence
→ failure aggregation
→ stats

INFER
load/capacity transaction
→ semantic preflight
→ preemption
→ stopped-worker admission
→ model/result eligibility transaction
→ dry-run
→ reserve transaction
→ launch/spawn persistence
→ failure aggregation
→ stats

ANALYZE
load/capacity transaction
→ last-model guard
→ dry-run
→ reserve transaction
→ launch/spawn persistence
→ failure aggregation
→ stats
```

After:

```text
SchedulerCycleService
→ RunTrainJobs / RunInferJobs / RunAnalyzeJobs adapter
→ FinalExperimentDispatchService owns candidate-loop sequencing
→ transaction/authority/admission/lifecycle/process callbacks
   execute in the same phase-specific order as before
```

All SQL, transactions, command construction, authority checks, lifecycle operations, and process operations remain in the compatibility adapter.

5. New boundary and files

Added:

- [FinalExperimentDispatchService.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/FinalExperimentDispatchService.hpp>)
- [FinalExperimentDispatchService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/FinalExperimentDispatchService.cpp>)
- [FinalExperimentDispatchServiceTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/FinalExperimentDispatchServiceTests.cpp>)
- [FinalExperimentDispatchServiceTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/FinalExperimentDispatchServiceTests.sh>)

Modified:

- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/ExperimentScheduler.cpp>)
- [project.pbxproj](</Volumes/Developer SSD/ExpertAdvisor/ExpertAdvisor.xcodeproj/project.pbxproj>)

`ExperimentScheduler.cpp`: 27,112 → 27,000 lines, net reduction 112.

6. Repository and public-header boundary

- No repository interface or implementation changes.
- No unrelated SQL moved.
- Existing SQL remains inside `ExperimentScheduler.cpp` transaction adapters.
- The new public header exposes no pqxx, PostgreSQL types, raw SQL, Postgres implementation types, `ExperimentScheduler.hpp`, or umbrella headers.
- Direct header includes:
  - Service implementation
  - `ExperimentScheduler.cpp`
  - Focused test
- Production fan-out is therefore limited to two translation units.

7. Tests

Passed:

```text
Tests/FinalExperimentDispatchServiceTests.sh
Tests/SchedulerCycleServiceTests.sh
Tests/SchedulerInternalSeamTests.sh
Tests/SchedulerOrchestrationServiceTests.sh
Tests/SchedulerSemanticAdmissionTests.sh
Tests/WorkerAttemptLifecycleServiceTests.sh
Tests/SchedulerAuthorityServiceTests.sh
Tests/SchedulerCoreBoundaryTests.sh
Tests/SchedulerZeroWorkerLimitsTests.sh
```

The new suite covers:

- No-candidate/no-op behavior
- Launch-disabled no-op
- Capacity rejection and stop
- Successful train delegation
- Train/infer/analyze distinctions
- Dry-run ordering differences
- Stopped-worker capacity rejection
- Eligibility failure propagation
- Launch-error propagation and exact diagnostic
- Important callback ordering
- Incomplete adapter rejection

`Tests/SchedulerAdmissionServiceTests.sh` does not exist; admission coverage was exercised through `SchedulerOrchestrationServiceTests.sh`.

8. Build

Command:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme SchedulerCore \
  -configuration Release \
  -derivedDataPath "$PWD/DerivedData/Release" \
  build
```

Result: `** BUILD SUCCEEDED **`

- Initial focused build: `CompileC=2` (`FinalExperimentDispatchService.cpp`, `ExperimentScheduler.cpp`), `Libtool=1`, `Ld=0`.
- Final incremental build: `CompileC=1` (`ExperimentScheduler.cpp`), `Libtool=1`, `Ld=0`.
- Full `LSTM Release` build was not run.

9. Risk assessment

- TRANSACTION-BOUNDARY RISK: Low. No transaction crosses the service API. Transaction construction, authority refresh, locking, commit timing, reservation commits, launch, and spawn-persistence behavior remain in existing adapters.
- ORDERING RISK: Low. Candidate → semantic → preemption → stopped-worker → eligibility/dry-run → reservation → preparation → launch ordering is preserved per phase and covered by focused tests.
- AUTHORITY/FENCING RISK: None identified. Authority responsibilities were not moved, cached, recreated, or made bypassable. Existing callbacks still invoke `RequireAndRefreshSchedulerAuthority`.
- BUILD-FAN-OUT RISK: Low. The narrow header uses only standard-library value/callback types and has two direct production consumers.

10. Preservation and diagnostics

- No CLI, schema, or migration changes.
- Migration-089 `model_input_width` and `model_input_semantic_layout_version` behavior is untouched.
- Legacy model-identity resolution remains on the existing semantic-preflight path.
- No changes to priority, admission, preemption, pause/resume, attempts, checkpoint, continuation, transition, or worker-command semantics.
- Diagnostic identifiers and dispatch reason strings were mechanically compared against HEAD; no additions, removals, or renames were found.
- The only new text is an internal invalid-adapter exception.

11. Warnings and remaining concerns

The focused build emits the existing `-Ofast` deprecation warning and the monolith’s existing libpqxx `exec_params` deprecation warnings (`342 warnings generated` for `ExperimentScheduler.cpp`). The new service produced no code-specific warning. These were not cleaned because that would be unrelated scope.

Database/process integration suites were not run because they can interact with live scheduler workers. Final human-operated `LSTM Release` validation remains pending.

12. Standalone scheduler assessment

Not justified yet. At 27,000 lines, `ExperimentScheduler.cpp` still contains substantial CLI compatibility, PostgreSQL adapters, child reaping/recovery, queue materialization, and reporting behavior.

Recommended Phase 20L candidate: inventory and potentially extract scheduler-owned child completion/reaping orchestration around `ReapSchedulerOwnedChildren`, while delegating exact-attempt lifecycle, process control, reconciliation, and authority semantics to their existing owners.

13. Final checks

`git diff --check`: passed.

Exact `git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/SchedulerCore/ExperimentScheduler.cpp
?? Sources/SchedulerCore/FinalExperimentDispatchService.cpp
?? Sources/SchedulerCore/FinalExperimentDispatchService.hpp
?? Tests/FinalExperimentDispatchServiceTests.cpp
?? Tests/FinalExperimentDispatchServiceTests.sh
```

Exact `git diff --stat`—excluding untracked additions until staged:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj        |   6 +
 Sources/SchedulerCore/ExperimentScheduler.cpp | 784 +++++++++++---------------
 2 files changed, 342 insertions(+), 448 deletions(-)
```

Nothing was staged, committed, amended, or pushed.