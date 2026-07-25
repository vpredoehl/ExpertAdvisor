---
title: "Pause Resume Cancel Final Staged Commit Verification"
document_type: "architecture review"
status: "final"
generated_from: "PauseResumeCancelAllExperiments_FinalStagedCommitVerification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Pause Resume Cancel Final Staged Commit Verification

## 1. Executive Summary

The exact staged diff is not ready to commit.

Production process-operation injection is safely constrained, native signaling remains unchanged, and the real-process readiness handshake is correctly ordered. However:

- Several deterministic crash-window fixtures contain states the production transactions cannot create.
- The scheduler-restart fixture omits mandatory durable recovery fields and relies on an unrelated running row that cannot coexist with the active cancellation as seeded.
- Emergency cleanup tracks only numeric PIDs/PGIDs, creating a narrow unrelated-process signaling risk after identifier reuse.

No file, working-tree content, or Git index state was changed by this review.

## 2. Findings by Severity

### Critical

None.

### High

1. Deterministic crash fixtures are not production-reachable as seeded.

   - [`SeedRequest`](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/GlobalExperimentControlProcessTests.cpp:526) leaves `requester_identity` and `scheduler_running_observed` null, while production always persists both when inserting the request at [GlobalExperimentControl.cpp:1402](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:1402).
   - The cancellation-before-accounting fixture uses `SeedRequest` without setting `experiment.cancellation_request_id`. Production always assigns that link in the planning transaction before signaling at [GlobalExperimentControl.cpp:1476](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:1476).
   - The accounted-before-reconciliation fixture seeds an `immediate` cancellation containing a `pending_checkpoint` outcome at [GlobalExperimentControlProcessTests.cpp:823](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/GlobalExperimentControlProcessTests.cpp:823). Production creates `pending_checkpoint` only for `after_next_checkpoint` cancellation at [GlobalExperimentControl.cpp:1498](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:1498).
   - Its completed, validated/signaled outcome omits the worker PID, process group, and start identity that production planning would have persisted and accounting would not remove.
   - Its pending recovery experiment lacks the durable restart model, `last_model_id`, `current_operation`, and recovery error state required by production requeue paths.

2. The active-cancellation scheduler-restart fixture is not production-reachable.

   - Experiment `800002` is seeded `pending` with a checkpoint cancellation but without a durable model or `last_model_id` at [GlobalExperimentControlIntegrationTests.sh:287](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/GlobalExperimentControlIntegrationTests.sh:287). Production requeues such a row only after finding a durable restart model and persists that model and recovery metadata at [ExperimentScheduler.cpp:12111](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:12111). Without a model, production cancels it as partial instead.
   - Experiment `800003` is seeded as unrelated `running` work solely to occupy capacity. A running experiment present when `cancel_all` plans would be linked to the request; after the gate is active, ordinary scheduling cannot launch it. The seeded combination therefore cannot arise through the reviewed workflows.

These defects invalidate the claimed crash/restart guarantees even though the tests previously passed.

### Medium

1. Emergency cleanup can signal an unrelated process after PID/PGID reuse.

   [`EmergencyProcessGroupRegistry`](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/GlobalExperimentControlProcessTests.cpp:88) stores only numeric IDs. Cleanup unconditionally sends group signals and falls back to the positive PID. If a launcher has been reaped or a worker group disappears before its registry entry is removed, that numeric ID can be reused and cleanup can target the new process/group.

2. Caller identifiers are not explicitly rejected by the cleanup registry.

   `Add()` rejects values `<=1`, but does not reject `getpid()` or `getpgrp()`. Current call sites supply direct test-fork results, making caller values structurally unlikely, but the requested invariant is not enforced locally.

### Low

None.

## 3. Test-Injection Seam Assessment

Passed.

- `RunCommand` constructs `PosixProcessOperations` on its own stack and passes it synchronously at [GlobalExperimentControl.cpp:1166](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:1166).
- The only non-production invocation of `RunCommandWithProcessOperationsForTesting` is in the staged process test.
- No CLI option, environment variable, database field, factory registry, or global setter enables injection.
- The injected object is a non-owning reference used only during the synchronous call and is never stored.
- There is no shared mutable global process-operations state.
- Concurrent production commands each construct independent native operations.
- Exceptions and early returns require no restoration because no injection state is installed.
- Native observation, identity validation, and signaling remain the same implementation.

Production behavior change caused specifically by the seam: none beyond a synchronous function-call indirection.

## 4. Crash-State Production-Reachability Assessment

### A. Plan committed before signaling

Not fully production-reachable as seeded.

The running experiment, worker identity, planned outcome, active global request, desired pause state, and expired lease timing are plausible. The request nevertheless omits `requester_identity` and `scheduler_running_observed`, which production always persists.

### B. Signal applied before outcome accounting

Not fully production-reachable as seeded.

- Paused process with database control state still `running`: physically valid.
- Resumed process with database control state still `paused`: physically valid.
- Terminated/absent process while the database still says `running`: physically valid.
- The pause/resume durable requests retain the request-field omissions above.
- The cancel fixture additionally omits the production-mandatory `experiment.cancellation_request_id`.

Thus the physical signal states are realistic, but the complete durable states are not.

### C. Outcome accounted before request reconciliation

Not production-reachable.

An immediate-cancel request cannot contain a `pending_checkpoint` outcome. The completed signaled outcome also omits persisted worker identity fields, while the pending experiment lacks the durable model and recovery metadata required to become `pending`.

### D. Scheduler restart with active cancellation

Not production-reachable.

The active request and global scheduling gate are plausible, and unrelated queued work could be enqueued after cancellation began. The claimed recovery row lacks its required durable restart model and recovery metadata. The unrelated running row used to occupy capacity could not remain unlinked through cancel-all planning or be launched after the gate became active.

## 5. Handshake and Pipe-Failure Assessment

The readiness protocol is correctly ordered:

- `setsid()` occurs before `exec`.
- The final test executable and arguments are active before readiness.
- `SIGTERM` handling or `SIG_IGN` is installed before readiness.
- Additional members are forked after signal policy installation and inherit the intended process group and policy.
- The ready payload verifies magic, PID, PGID, and additional-member PID.
- Parent-side observation revalidates executable arguments, process group, and process identity before signaling.
- Intended pipe ends are closed in the launcher, worker, group member, and parent.

Failure coverage is adequate:

- Pipe and fork failures cause deterministic cleanup-aware test failure.
- `setsid`, member-fork, and `exec` failures close the readiness pipe through process exit.
- PID and readiness reads handle partial reads and `EINTR`.
- Writes handle short writes and `EINTR`.
- EOF, malformed payload, child exit, readiness timeout, and launcher failure cannot be mistaken for readiness.
- The fixed readiness structure is smaller than `PIPE_BUF`, so the post-poll blocking read does not present a realistic partial-payload hang with the actual writer.
- No realistic unhandled pipe failure capable of indefinitely hanging the tests was found.

## 6. Emergency Cleanup Safety Assessment

Passed aspects:

- Only launcher and worker IDs created by the process-test binary are registered.
- Registration occurs before subsequent fallible parent checks.
- Values `<=1` and duplicates are rejected or harmless.
- Cleanup uses bounded waits, `SIGCONT`, `SIGTERM`, then `SIGKILL`.
- Group and process disappearance are checked.
- Repeated cleanup and cleanup after partial initialization are structurally safe.
- `FailCheck` uses `_Exit`, preventing duplicate `atexit` cleanup; normal completion leaves the registry empty.
- The self-test deliberately invokes `FailCheck`, so it exercises failure cleanup rather than normal teardown.
- No production scheduler or database-discovered worker is added to the registry.

Failed aspects:

- Caller PID/PGID rejection is not explicit.
- Registry entries carry no start identity or equivalent reuse proof.
- A stale registered numeric ID can therefore signal an unrelated reused process or group.

## 7. Exact Staged File Assessment

The staged set is exactly the expected 13 files:

- `Database/README.md`
- `Database/migrations/046_global_experiment_control.sql`
- `ExpertAdvisor.xcodeproj/project.pbxproj`
- `LSTM/main.cpp`
- `Sources/ExperimentScheduler.cpp`
- `Sources/GlobalExperimentControl.cpp`
- `Sources/GlobalExperimentControl.hpp`
- `Tests/GlobalExperimentControlIntegrationTests.sh`
- `Tests/GlobalExperimentControlMigrationTests.sql`
- `Tests/GlobalExperimentControlProcessTests.cpp`
- `Tests/GlobalExperimentControlTests.cpp`
- `docs/GlobalExperimentControls.rst`
- `docs/architecture/Volume_XI_Scheduler.md`

Assessment:

- No generated output, transcript, build product, dump, or unrelated campaign file is staged.
- Migration `046` follows tracked migration `045`; documentation consistently references `046`.
- The project contains both production source/header references and builds `GlobalExperimentControl.cpp` in both applicable targets.
- The integration script is staged executable.
- Index and working-tree versions of every tracked file are identical.
- `git diff --cached --check`: passed.
- `plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj`: passed.
- `bash -n Tests/GlobalExperimentControlIntegrationTests.sh`: passed.
- No build or executable suite was rerun, as instructed.

Staged stat:

```text
13 files changed, 5356 insertions(+), 169 deletions(-)
```

## 8. Unstaged and Untracked File Assessment

Unstaged tracked files: none.

Untracked feature-relevant files:

- `PauseResumeCancelAllExperiments_CEE_Verification_Output.md`
- `PauseResumeCancelAllExperiments_FinalTestCorrections_Output.md`
- `PauseResumeCancelAllExperiments_FinalVerificationOutput.md`
- `PauseResumeCancelExperiments_VerificationCorrections_Output.md`

These are review reports, not product dependencies. None contains an authoritative patch required by the staged implementation. The final-corrections report contains a readiness claim now contradicted by this inspection, but it is not staged.

The proposed commit is reproducible from the index alone.

Final `git status --short` remains the 13 staged files plus those four untracked reports. `git diff --name-status` and `git diff --stat` are empty.

## 9. Review Artifact Disposition

Untracked:

- The four Markdown outputs listed above.

Ignored:

- `PauseResumeCancelAllExperiments_CEE_VerificationPrompt.txt`
- `PauseResumeCancelAllExperiments_CEE_Verification_Transcript.txt`
- `PauseResumeCancelAllExperiments_FinalStagedCommitVerification_Prompt.txt`
- `PauseResumeCancelAllExperiments_FinalStagedCommitVerification_Transcript.txt`
- `PauseResumeCancelAllExperiments_FinalTestCorrections_Prompt.txt`
- `PauseResumeCancelAllExperiments_FinalTestCorrections_Transcript.txt`
- `PauseResumeCancelAllExperiments_FinalVerificationPrompt.txt`
- `PauseResumeCancelAllExperiments_FinalVerificationTranscript.txt`
- `PauseResumeCancelAllExperiments_ImplementationPrompt.txt`
- `PauseResumeCancelAllExperiments_ImplementationTranscript.txt`
- `PauseResumeCancelExperiments_VerificationCorrections_Prompt.txt`
- `PauseResumeCancelExperiments_VerificationCorrections_Transcript.txt`
- `investigate_child_exit_127_prompt_and_reasoning.txt` — unrelated historical scheduler investigation.

Tracked historical artifacts:

- All 92 files under `ArchitectureReviews/` are unchanged and unstaged.

Staged review artifacts: none.

## 10. Commit Recommendation

Before committing:

1. Make every crash fixture reproduce the exact fields and transaction shape production persists, including request audit fields and cancellation links.
2. Replace the impossible immediate-plus-pending-checkpoint fixture with a valid after-next-checkpoint state and include the durable worker/recovery identity fields.
3. Give the scheduler-restart recovery row its genuine durable model and recovery metadata, and remove the impossible unrelated running capacity row.
4. Make emergency cleanup reject caller identifiers and revalidate registered process identity before signaling, so stale numeric IDs cannot target reused processes/groups.
5. Rerun the focused process/crash tests, integration suite, and staged integrity checks.

Files changed by this review: none.

READY AFTER SPECIFIED FIXES