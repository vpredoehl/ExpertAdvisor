---
title: "Pause Resume Cancel Final Corrected Staged Verification"
document_type: "architecture review"
status: "final"
generated_from: "PauseResumeCancelAllExperiments_FinalCorrectedStagedVerification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Pause Resume Cancel Final Corrected Staged Verification

## 1. Executive Summary

The staged candidate is not ready to commit. The two-file correction is correctly limited in scope and improves most fixtures, but concrete blockers remain:

- The accounted-before-reconciliation fixture still contains a completed experiment state production cannot write.
- The scheduler-restart request has accounting totals inconsistent with production reconciliation.
- The launcher can release or retain an unregistered disposable worker on realistic failure paths.
- Cleanup does not always validate exact executable/start identity and has an `atexit` lifetime defect.

No production file was changed by the correction.

## 2. Findings by Severity

### Critical

None.

### High

1. **Accounted fixture remains unreachable.**
   Experiment `700010` retains PID/PGID and uses detail `process_group_terminated` at [GlobalExperimentControlProcessTests.cpp:1109](</Volumes/Developer SSD/ExpertAdvisor/Tests/GlobalExperimentControlProcessTests.cpp:1109>) and [GlobalExperimentControlProcessTests.cpp:1143](</Volumes/Developer SSD/ExpertAdvisor/Tests/GlobalExperimentControlProcessTests.cpp:1143>). Production clears PID/PGID during cancellation accounting at [GlobalExperimentControl.cpp:1819](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:1819>) and persists the signal outcome’s existing detail—`validated` on normal success—not `process_group_terminated`, through [GlobalExperimentControl.cpp:949](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:949>) and [GlobalExperimentControl.cpp:609](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:609>). Its fixed 2020 completion timestamp also cannot belong to the same accounting commit as the outcome’s default current timestamp.

2. **Launcher/worker registration is not fail-closed.**
   `SpawnWorker` writes the launcher release byte before checking that launcher registration succeeded at [GlobalExperimentControlProcessTests.cpp:552](</Volumes/Developer SSD/ExpertAdvisor/Tests/GlobalExperimentControlProcessTests.cpp:552>). If start-identity capture or observation fails, the launcher is released, forks the worker, and `FailCheck` has no worker identity to clean safely.

   The worker also performs `setsid`, `exec`, and possibly creates a group member before the parent successfully registers it at [GlobalExperimentControlProcessTests.cpp:521](</Volumes/Developer SSD/ExpertAdvisor/Tests/GlobalExperimentControlProcessTests.cpp:521>) and [GlobalExperimentControlProcessTests.cpp:577](</Volumes/Developer SSD/ExpertAdvisor/Tests/GlobalExperimentControlProcessTests.cpp:577>). A registration failure can leave it running unregistered.

3. **Pending-worker cleanup can leak a group member.**
   A managed worker creates its group member before sending readiness at [GlobalExperimentControlProcessTests.cpp:447](</Volumes/Developer SSD/ExpertAdvisor/Tests/GlobalExperimentControlProcessTests.cpp:447>). Until readiness, the registry owns only the positive PID. A timeout, malformed message, or parent failure in that interval kills only the leader; the group member may survive.

4. **Scheduler-restart request accounting is unreachable.**
   The fixture seeds a `pending` request with a `process_missing` outcome but `missing_count=0` in `result_summary` at [GlobalExperimentControlIntegrationTests.sh:297](</Volumes/Developer SSD/ExpertAdvisor/Tests/GlobalExperimentControlIntegrationTests.sh:297>). Production `UpdateRequestAccounting` counts that outcome as missing and atomically writes `missing_count=1` and the corresponding summary at [GlobalExperimentControl.cpp:624](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:624>). There is no production boundary yielding the seeded combination.

### Medium

1. Emergency registry matching uses a command-line substring, not exact `ProcessObservation::executable`, at [GlobalExperimentControlProcessTests.cpp:259](</Volumes/Developer SSD/ExpertAdvisor/Tests/GlobalExperimentControlProcessTests.cpp:259>).

2. The shell scheduler fixture never captures or revalidates process-start identity. Its cleanup checks command substrings and signals a positive PID at [GlobalExperimentControlIntegrationTests.sh:11](</Volumes/Developer SSD/ExpertAdvisor/Tests/GlobalExperimentControlIntegrationTests.sh:11>) and [GlobalExperimentControlIntegrationTests.sh:500](</Volumes/Developer SSD/ExpertAdvisor/Tests/GlobalExperimentControlIntegrationTests.sh:500>). PID reuse therefore is not excluded.

3. `EmergencyCleanupAtExit` is registered before the function-local registry is constructed at [GlobalExperimentControlProcessTests.cpp:1498](</Volumes/Developer SSD/ExpertAdvisor/Tests/GlobalExperimentControlProcessTests.cpp:1498>). Reverse destruction order can destroy the registry before the callback invokes it, resulting in use-after-destruction on normal exit.

4. The ordinary running-experiment fixtures omit `current_operation` and `worker_started_at`, while the production scheduler populates them when marking a worker running at [ExperimentScheduler.cpp:11656](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11656>). Thus the administrative-control slice is reachable, but the entire seeded row is not production-shaped.

### Low

- The actual branch is `campaign-operations`, not the `phase6` branch stated in the supplied project instructions. HEAD is `748c4d4e8c12068b0104d6ce6b2608e4e5369dd4`.

## 3. Corrected Two-File Diff Assessment

The preceding staged baseline blobs remain available and match the correction transcript:

- Integration: `e366bf…` → staged `9d7ec…`: 183 insertions, 20 deletions.
- Process tests: `8019cc…` → staged `a2a9d…`: 515 insertions, 70 deletions.

The delta is confined to:

- Request and cancellation fixtures.
- Durable model/matrix recovery metadata.
- Scheduler-restart setup and assertions.
- Emergency registry identities.
- Launcher synchronization.
- Caller identity rejection.
- Focused cleanup/replay assertions.

No production behavior or unrelated test area was changed.

## 4. Production-Reachability Assessment

### A. Plan committed before signal

- **Boundary:** Planning transaction committed before `SIGSTOP`.
- **Reachability:** Administrative state is reachable; entire experiment row is not.
- **Supporting path:** Request insertion, global state update, and `InsertOutcome` at [GlobalExperimentControl.cpp:1402](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:1402>).
- **Discrepancy:** Running experiment metadata omits scheduler-populated operational fields.

### B. Pause signal sent before accounting

- **Boundary:** `SIGSTOP` succeeds while database control state remains running.
- **Reachability:** Physical/database divergence and replay behavior are reachable.
- **Supporting path:** `PauseWorker`, followed later by accounting at [GlobalExperimentControl.cpp:1716](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:1716>).
- **Discrepancy:** Same whole-row running metadata omission as A. No duplicate `SIGSTOP` is sent.

### C. Resume signal sent before accounting

- **Boundary:** Database remains paused after successful `SIGCONT`.
- **Reachability:** Physical/database divergence and replay behavior are reachable.
- **Supporting path:** `ResumeWorker` and subsequent accounting at [GlobalExperimentControl.cpp:1719](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:1719>).
- **Discrepancy:** Same whole-row metadata omission. No duplicate `SIGCONT` is sent.

### D. Immediate cancellation signal sent before accounting

- **Boundary:** Cancellation link and planned outcome commit, process terminates, accounting has not committed.
- **Reachability:** Control boundary is reachable.
- **Supporting path:** Cancellation link at [GlobalExperimentControl.cpp:1476](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:1476>), planned immediate outcome at [GlobalExperimentControl.cpp:1567](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:1567>), accounting at [GlobalExperimentControl.cpp:1740](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:1740>).
- **Discrepancy:** Same pre-signal experiment metadata omission. Replay correctly sends no duplicate termination signal.

### E. Accounted before request reconciliation

- **Boundary:** One immediate infer cancellation accounted; one checkpoint worker requeued; request not yet reconciled.
- **Reachability:** **No.**
- **Supporting path:** Missing-worker requeue at [GlobalExperimentControl.cpp:1661](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:1661>) and checkpoint terminal transaction at [main.cpp:3756](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:3756>).
- **Discrepancies:** Completed experiment PID/PGID, detail, and timestamps do not match production accounting. The pending recovery row, model selection metadata, cancellation links, first replay, terminal checkpoint transition, and second reconciliation are otherwise consistent.

## 5. Scheduler-Restart Assessment

The test does prove the database-authoritative cancellation gate:

- `LoadPendingExperiments(..., cancellationOnly=true)` requires `cancellation_request_id=active_request_id` and a checkpoint target at [ExperimentScheduler.cpp:4923](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:4923>).
- Exactly one authorized row is examined with one free slot.
- Both unrelated pending rows remain unlinked and excluded.
- `already_running` is reached before capacity rejection.
- No child command or administrative signal is emitted.
- Request/outcome identities and `active_request_id` remain stable.

However, the request accounting totals are unreachable, and the disposable process teardown does not safely revalidate process-start identity or exact PGID ownership.

## 6. Emergency Registry Safety Assessment

Confirmed safe aspects:

- PID, intended PGID, start identity, role, and command marker are stored.
- PID/PGID values `0`, `1`, caller PID, and caller PGID are rejected.
- PID and PGID reuse fail closed.
- A disappeared group leader does not cause signaling of a surviving numeric group.
- There is no numeric group-to-positive-PID fallback.
- Duplicate promotion and empty cleanup are idempotent.
- Permission/inspection ambiguity prevents signaling.

Unresolved:

- Exact executable identity is not compared.
- Pending group ownership can leak members before promotion.
- Normal `atexit` ordering can access the destroyed registry.
- The shell-managed scheduler fixture lacks equivalent start-identity validation.

## 7. Release-Pipe Ordering and Failure Assessment

Safe paths include initial pipe/fork failure, launcher read/fork failure before a worker exists, and worker `setsid`/`exec` failure after successful pending registration.

Blocking paths:

- Launcher release occurs even when registration failed.
- Worker execution is not gated on successful parent registration.
- PID-pipe `WriteAll` failure is ignored by the launcher.
- A worker registration failure leaves no safe registry ownership.
- Ready timeout/malformed readiness after member creation can leave a group member.
- Parent failure before group promotion has only PID-level cleanup.

Consequently, the protocol does not satisfy the required “no unregistered disposable process” or “no disposable process after emergency cleanup” guarantees.

## 8. Verification Evidence Assessment

Transcript evidence confirms successful execution of:

- `GlobalExperimentControlTests`.
- Real-process tests.
- Emergency-cleanup self-test with expected exit `1`.
- Deterministic database crash-window tests.
- Full integration suite.
- Migration apply, assertions, reapply, and repeated assertions.
- Scheduler cancellation recovery.
- Advisory-lock regression.
- CLI validation.
- Shell syntax and project-file validation.
- Cached and uncached diff checks.

The supplied current Release build succeeded. The integration suite used that production binary, and the correction changed no production source. Known libpqxx deprecation warnings are not blockers.

These passes do not overcome the defects because the unreachable values are manually seeded and the release failure paths are not injected or exercised.

## 9. Exact Staged-Status Assessment

Exact staged files:

1. `Database/README.md`
2. `Database/migrations/046_global_experiment_control.sql`
3. `ExpertAdvisor.xcodeproj/project.pbxproj`
4. `LSTM/main.cpp`
5. `Sources/ExperimentScheduler.cpp`
6. `Sources/GlobalExperimentControl.cpp`
7. `Sources/GlobalExperimentControl.hpp`
8. `Tests/GlobalExperimentControlIntegrationTests.sh`
9. `Tests/GlobalExperimentControlMigrationTests.sql`
10. `Tests/GlobalExperimentControlProcessTests.cpp`
11. `Tests/GlobalExperimentControlTests.cpp`
12. `docs/GlobalExperimentControls.rst`
13. `docs/architecture/Volume_XI_Scheduler.md`

Status:

- Unstaged tracked files: none.
- Cached diff: 13 files, 5,964 insertions, 169 deletions.
- `git diff --cached --check`: passed.
- Initial/final cached hash: `fec93ede…`, unchanged.
- Initial/final index-entry hash: `3a5bc458…`, unchanged.
- Cached tree `abe26da3…` exactly matches the index.
- No staged prompt, transcript, output, log, dump, or build product.

Relevant untracked files:

- `PauseResumeCancelAllExperiments_CEE_Verification_Output.md`
- `PauseResumeCancelAllExperiments_FinalFixtureAndCleanupCorrections_Output.md`
- `PauseResumeCancelAllExperiments_FinalStagedCommitVerification_Output.md`
- `PauseResumeCancelAllExperiments_FinalTestCorrections_Output.md`
- `PauseResumeCancelAllExperiments_FinalVerificationOutput.md`
- `PauseResumeCancelExperiments_VerificationCorrections_Output.md`

They are non-product review artifacts.

The index, tracked working tree, and six listed untracked files did not change during review. The ignored live verification transcript grew automatically as this inspection was recorded; it is not staged.

## 10. Commit Recommendation

Before committing:

1. Make the accounted fixture persist production’s cleared experiment PID/PGID, actual successful detail, and transaction-consistent timestamps.
2. Seed scheduler-restart accounting with `missing_count=1` and matching summary.
3. Prevent launcher release unless registration succeeds, and gate worker continuation until worker registration succeeds.
4. Handle PID-pipe failure and pre-promotion group-member cleanup without unsafe numeric signaling.
5. Validate exact executable/start/PGID identity in both C++ and shell cleanup.
6. Correct the registry/`atexit` lifetime ordering.

READY AFTER SPECIFIED FIXES