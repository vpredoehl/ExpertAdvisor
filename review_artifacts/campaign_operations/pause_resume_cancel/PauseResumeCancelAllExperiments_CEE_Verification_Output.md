---
title: "Pause/Resume/Cancel Experiments Independent Verification Review"
document_type: "architecture review"
status: "superseded"
generated_from: "PauseResumeCancelAllExperiments_CEE_VerificationPrompt.txt"
reasoning_effort: "high"
model: "default"
---

# Pause/Resume/Cancel Experiments Independent Verification Review

> **Superseded on 2026-07-24.** This document records the independent
> verification findings before the focused correction pass. The open
> process-start-identity and destructive-regression-test findings below have
> now been resolved. The implementation persists the macOS kernel process
> start timestamp for launched and adopted workers; validates PID, process
> group, executable, experiment, phase, and that exact start identity before
> every administrative signal and escalation; and rejects incomplete or
> mismatched identity. Isolated tests now cover real STOP/CONT, graceful TERM,
> KILL escalation, process groups, mismatched start identity, lease
> acquisition/expiry/takeover, restart recovery, checkpoint-inference
> serialization, and the repaired checkpoint queue race. Migration 046 was
> applied, asserted, reapplied, and reasserted in a disposable database. The
> final-source clean Release build and all affected tests passed. The stale
> implementation report was removed from the pending commit. The historical
> assessment and file listings below are retained only as review provenance
> and are not the current commit-readiness conclusion.

## 1. Executive summary

The database-authoritative architecture, scheduler gating, transaction ordering, advisory locking, checkpoint cancellation, and restart persistence are generally sound.

I found and fixed one concrete race: periodic checkpoint inference could be queued outside the coordination lock after immediate cancellation had already committed its suppression plan. The queue path now takes the shared lock and suppresses ordinary checkpoint inference whenever the experiment has a persisted cancellation request ([main.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:3372>)).

One release-blocking process-identity gap remains: PID, PGID, executable, phase, and argv are validated, but no kernel process-start token is persisted. Therefore identical-argv PID/PGID reuse cannot be distinguished from the original worker, despite documentation claiming PID reuse is rejected.

## 2. Findings by severity

### Critical

None.

### Major

1. **Process identity cannot conclusively reject identical-identity PID reuse — open.**

   [ManagedWorker](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.hpp:51>) and migration 046 persist PID, PGID, executable, and command line, but no kernel process birth/start token. [ValidateManagedWorker](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:733>) therefore accepts a process with the same PID/PGID, executable, phase, and experiment argv even if it is a later process.

   Required fix:

   - Persist a macOS process-start identity for experiment and checkpoint workers.
   - Capture it during launch/adoption.
   - Require an exact match before every signal and escalation.
   - Add a test where every existing identity field matches but the start token differs.
   - Amend the unconditional PID-reuse guarantee in [GlobalExperimentControls.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/GlobalExperimentControls.rst:51>) until fixed.

2. **Destructive concurrency/restart coverage is insufficient — open.**

   Current tests use fake process operations and nonexistent PIDs. They do not exercise:

   - scheduler launch versus pause/cancel transaction serialization;
   - actual isolated process-group pause, resume, TERM, and KILL;
   - unrelated-process decoys;
   - crash points after plan commit, after signaling, and before accounting;
   - lease expiry and retry takeover;
   - scheduler restart during checkpoint cancellation;
   - the repaired opportunistic-checkpoint queue race.

   These should be isolated-process and isolated-database tests; the live scheduler must remain untouched.

### Minor

1. **Checkpoint inference cancellation race — fixed during review.**

   Periodic checkpoint inference insertion previously did not take the shared advisory lock. Immediate cancellation could miss a row inserted immediately after its suppression transaction. The corrected path serializes insertion with cancellation and skips ordinary insertion after cancellation assignment.

2. **Compiler warnings remain.**

   The clean build succeeds, but emits repository-wide libpqxx deprecation warnings and existing non-deprecation diagnostics such as unused/unreachable code, the invalid LLVM22 toolchain metadata warning, and a symbol-free Metal object warning. The new control module also uses deprecated `exec_params` extensively. No warning was introduced by the focused race fix itself.

3. **The generated implementation report is stale as a commit artifact.**

   [PauseResumeCancelAllExperiments_ImplementationOutput.md](</Volumes/Developer SSD/ExpertAdvisor/PauseResumeCancelAllExperiments_ImplementationOutput.md:1>) states unconditional readiness, does not record the race found here, and its embedded `git status` omits the report itself. Exclude it from the commit or regenerate it after the remaining fixes.

### Suggestions

- Add an operational downgrade runbook. The schema is additive and an older binary should ignore it, but physically removing migration 046 requires ordered removal of dependent foreign keys and columns.
- Clarify whether `worker_command_line` is audit-only; it is persisted but not compared as a complete identity value.
- Add the migration assertion SQL to the integration script so it cannot accidentally be skipped.

## 3. Migration 046 review

Migration 046 is structurally reasonable:

- Singleton global-control row is constrained correctly.
- Administrative action, mode, status, identity-result, signal-result, inference-action, and outcome values have appropriate checks.
- Foreign keys protect request, experiment, checkpoint-evaluation, and model identities.
- Partial indexes support active cancellation and checkpoint reconciliation.
- Defaults are compatible with existing rows.
- Audit identity columns are protected while runtime accounting columns remain mutable.
- Cancellation-specific persistence is justified; no clearly redundant authoritative control path was found.
- The repository migration runner wraps each migration and ledger update in one transaction, so failure rolls back atomically.
- Reapplication passed.
- A production-schema-only clone upgraded successfully and passed the migration assertions.
- The live production database remains untouched and reports `migration_046=false`.

Downgrade concern: there is no down migration. Binary rollback is likely safe because the schema changes are additive, but schema rollback is not trivial.

## 4. Build verification

Final exact post-fix clean build:

```bash
nice -n 10 xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath /tmp/expertadvisor-pause-review.Cmk1eZ \
  -jobs 1 \
  clean build
```

Result: `CLEAN SUCCEEDED`, `BUILD SUCCEEDED`.

The isolated derived-data path prevented removal or replacement of the executable used by the active scheduler.

## 5. Test results

Passed:

- `GlobalExperimentControlTests`
- `GlobalExperimentControlIntegrationTests.sh` after the fix and again after the final clean build
- Migration 046 first application, assertions, reapplication, and repeated assertions
- Production-schema-clone migration upgrade and repeat migration
- `SchedulerChildStatusTests`
- `ContinuationPolicyInheritanceTests`
- CLI help verification
- `plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj`
- `git diff --check`

Not run against production:

- Live pause, resume, or cancellation
- Tests capable of launching scheduler work
- Production-backed destructive persistence tests

The active scheduler and seven training workers remained active and production migration 046 was not applied.

## 6. Overall assessment

The authoritative database design and most recovery/concurrency mechanisms are well constructed. No reversed advisory-lock ordering or clear database deadlock path was found. Persistent pause and cancellation gates survive scheduler restart, while interrupted one-shot signal application requires same-shaped command replay after lease expiry, as documented.

The implementation is not yet ready to commit because its stated PID-reuse safety guarantee is stronger than the actual identity contract, and the missing real process/restart concurrency coverage is material for destructive administrative controls.

## 7. Commit recommendation

**READY AFTER SPECIFIED FIXES**

Required before commit:

1. Add and validate a kernel process-start identity token.
2. Add isolated real-process and crash/restart concurrency tests listed above.
3. Update the PID-reuse documentation and stale implementation report.
4. Preferably remove new-module libpqxx deprecation warnings.

Files currently modified:

- [Database/README.md](</Volumes/Developer SSD/ExpertAdvisor/Database/README.md>)
- [project.pbxproj](</Volumes/Developer SSD/ExpertAdvisor/ExpertAdvisor.xcodeproj/project.pbxproj>)
- [main.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp>)
- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp>)
- [Volume_XI_Scheduler.md](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_XI_Scheduler.md>)

Untracked feature files include migration 046, the control module, three tests, operator documentation, and the implementation report.

`git status --short`:

```text
 M Database/README.md
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M LSTM/main.cpp
 M Sources/ExperimentScheduler.cpp
 M docs/architecture/Volume_XI_Scheduler.md
?? Database/migrations/046_global_experiment_control.sql
?? PauseResumeCancelAllExperiments_ImplementationOutput.md
?? Sources/GlobalExperimentControl.cpp
?? Sources/GlobalExperimentControl.hpp
?? Tests/GlobalExperimentControlIntegrationTests.sh
?? Tests/GlobalExperimentControlMigrationTests.sql
?? Tests/GlobalExperimentControlTests.cpp
?? docs/GlobalExperimentControls.rst
```

`git diff --stat` (untracked files excluded):

```text
 Database/README.md                       |   8 +
 ExpertAdvisor.xcodeproj/project.pbxproj  |   8 +
 LSTM/main.cpp                            | 214 +++++++-
 Sources/ExperimentScheduler.cpp          | 902 ++++++++++++++++++++++++++-----
 docs/architecture/Volume_XI_Scheduler.md |   8 +
 5 files changed, 978 insertions(+), 162 deletions(-)
```
