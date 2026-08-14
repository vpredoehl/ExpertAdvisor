---
title: "Return Feature Exact Attempt Reconciliation Command Implementation and Independent Test"
document_type: "architecture review"
status: "final"
generated_from: "ReturnFeature_ExactAttemptReconciliationCommand_ImplementationAndIndependentTest_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Return Feature Exact Attempt Reconciliation Command Implementation and Independent Test

# Executive Result

**BLOCKED**

The narrow reconciliation command is implemented and both frozen attempts pass live dry-run validation. Completion is blocked because the required normal Release build and full focused integration suite could not pass in the current dirty-worktree environment.

# Existing Exact-Attempt State Model

- `identity_ambiguous` is active/capacity-consuming but intentionally non-signalable.
- `observed` is active and signalable.
- Operator stop continues to require `LockAndVerifyExactActiveAttempt(...)` and `IsSignalableAttemptState(...)`.
- The new command restores only `identity_ambiguous → observed`; it creates no signaling bypass.

# Command Design

CLI:

```text
--reconcile-worker-attempt=WORKER_ATTEMPT_ID [--dry-run | --yes]
```

- One exact attempt only.
- `--dry-run` performs all locking and identity checks but no durable update.
- Durable application requires `--yes`.
- Only `worker_kind=experiment`, phase/capacity `train|infer|analyze`, and source state `identity_ambiguous` are accepted.
- Checkpoint workers are explicitly rejected.
- The command takes the global coordination lock but never claims, refreshes, replaces, or releases scheduler ownership.

# Implementation

Changed:

- [ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp)
  - CLI parsing, validation, help, and dispatcher.
- [GlobalExperimentControl.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/GlobalExperimentControl.hpp)
  - Narrow reconciliation command API.
- [GlobalExperimentControl.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp)
  - Locked reconciliation service and dry-run report.
- [SchedulerOwnershipRepository.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/SchedulerOwnershipRepository.hpp)
  - Explicit required-source-state support for exact-attempt locking.
- [SchedulerOwnershipProcessIntegrationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/SchedulerOwnershipProcessIntegrationTests.sh)
  - Isolated fixture scenario for dry-run, guarded transition, and non-ambiguous rejection.

# Identity Validation Contract

Before eligibility, the service verifies:

- attempt ID, experiment ID, worker kind, phase, capacity class;
- source lifecycle state;
- worker PID, PGID, start identity, canonical executable, and complete command line;
- command identity;
- active experiment binding;
- running experiment status and matching phase;
- lifecycle PID/PGID/start identity/executable/command equality;
- live PID/PGID/start identity/executable/command equality;
- exact `--scheduler-worker-attempt-id`;
- exact experiment/phase command identity;
- native executable observation, including the corrected kernel executable-path fallback.

# Fencing and Concurrency Analysis

The command:

1. Takes the global coordination advisory lock.
2. Locks the exact attempt row.
3. Uses `LockAndVerifyExactActiveAttempt` to lock and verify the authoritative lifecycle row.
4. Observes the process immediately before the transition.
5. Uses an equality-guarded update covering attempt state, immutable identity fields, historical scheduler invocation/fence identity, and the active lifecycle binding.

It neither acquires scheduler authority nor changes the scheduler lease. Historical scheduler invocation/fence fields are treated as immutable exact-attempt identity predicates; current lease ownership is not required because this command cannot dispatch work.

# Automated Test Results

Passed:

- `bash -n Tests/SchedulerOwnershipProcessIntegrationTests.sh`
- `./Tests/SchedulerCanonicalPathTests.sh`
- `SchedulerOwnershipPolicyTests`
- Syntax-only compilation of changed C++ sources.
- Debug Xcode build with `ENABLE_USER_SCRIPT_SANDBOXING=NO`.

Blocked/failed:

- The full `SchedulerOwnershipProcessIntegrationTests.sh` exited `1` before reaching the added reconciliation scenario, during existing cloned-schema fixture setup.
- The normal required Release build failed in the provenance phase because the repository is dirty, including a pre-existing untracked report.

# Independent Adversarial Review

One defect was found and corrected: the scheduler-command discriminator initially recognized only the spaced form of `--reconcile-worker-attempt`, causing `--reconcile-worker-attempt=623` to route to the general CLI parser. It now recognizes both forms.

The reviewed transition has no call to `SignalProcessGroup`, `PauseWorker`, `ResumeWorker`, `CancelWorker`, scheduler dispatch, or lifecycle mutation. The update is not authorized by `worker_attempt_id` alone.

# Release Build Result

Required command run:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
```

Result: **failed before compilation** because `GenerateBuildProvenance.py` requires a clean source tree.

# Frozen Attempt 623 Dry-Run Validation

Command actually run with the newly built Debug administrative binary:

```text
DerivedData/ExpertAdvisor/Build/Products/Debug/LSTM_Release --reconcile-worker-attempt=623 --dry-run
```

Result: `eligible=true`, with exact PID `68338`, PGID `68338`, start identity, executable, command, and lifecycle binding matched.

# Frozen Attempt 629 Dry-Run Validation

```text
DerivedData/ExpertAdvisor/Build/Products/Debug/LSTM_Release --reconcile-worker-attempt=629 --dry-run
```

Result: `eligible=true`, with exact PID `32973`, PGID `32973`, start identity, executable, command, and lifecycle binding matched.

# Non-Mutation Proof

After both dry-runs:

- Attempts 623 and 629 remain `identity_ambiguous`.
- Their persisted PID/PGID/start identity/executable/command/reconciliation fields are unchanged.
- Experiments 549 and 554 remain `running/train` with the same active attempt bindings.
- Both cancellation request IDs and completion timestamps remain null.
- Running checkpoint-evaluation count remains `0`.

# Frozen Process Proof

Final states:

```text
32973  ... 32973  Ts  ... --scheduler-worker-attempt-id=629
68338  ... 68338  Ts  ... --scheduler-worker-attempt-id=623
42864  ... 42863  T+  LSTM_Release --schedule-experiments ...
```

No frozen process was signaled, resumed, replaced, or reaped.

# Remaining Cutover Boundary

This work does **not** authorize durable reconciliation of 623/629, worker retirement, scheduler retirement/restart, experiment cancellation, runtime cutover, or scheduler resumption.

# Exact Next Recommended Command Boundary

Authorize a clean-worktree Release build and completion of the isolated reconciliation integration suite review. Do not run `--yes` against attempts 623 or 629 as part of that step.

`git status --short`

```text
 M Sources/ExperimentScheduler.cpp
 M Sources/GlobalExperimentControl.cpp
 M Sources/GlobalExperimentControl.hpp
 M Sources/SchedulerOwnershipRepository.hpp
 M Tests/SchedulerOwnershipProcessIntegrationTests.sh
?? ReturnFeature_ObserveCorrection_ImplementationAndFrozenValidation_Output.md
```

`git diff --stat`

```text
 Sources/ExperimentScheduler.cpp                    |  29 +++
 Sources/GlobalExperimentControl.cpp                | 244 +++++++++++++++++++++
 Sources/GlobalExperimentControl.hpp                |  23 ++
 Sources/SchedulerOwnershipRepository.hpp           |   9 +-
 Tests/SchedulerOwnershipProcessIntegrationTests.sh |  77 +++++++
 5 files changed, 381 insertions(+), 1 deletion(-)
```