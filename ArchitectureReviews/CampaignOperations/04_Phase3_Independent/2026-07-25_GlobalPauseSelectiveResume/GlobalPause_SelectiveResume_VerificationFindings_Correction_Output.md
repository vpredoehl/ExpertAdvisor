---
title: "Global Pause Selective Resume Verification Findings Correction"
document_type: "architecture review"
status: "final"
generated_from: "GlobalPause_SelectiveResume_VerificationFindings_Correction_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Global Pause Selective Resume Verification Findings Correction

### 1. Executive summary

Implemented the focused Global Pause / Selective Resume corrections. Reconciliation is now owner-fenced, selective replay recovers before lifecycle validation, resume-all audits and cleans the full pause generation, and CLI replay output reflects persisted accounting.

Focused tests, migration tests, controlled-process tests, crash recovery tests, the full isolated integration suite, and the worktree-safe Release build pass. No changes were committed or staged.

### 2. Exact defects corrected

- Post-signal mutations require ownership of the exact persisted administrative request.
- Expired requests can be claimed by a new invocation; superseded owners cannot reconcile afterward.
- Exact selective-resume recovery occurs before new-command lifecycle validation.
- Departed primary and checkpoint workers are reconciled without signaling replacement processes.
- Resume-all accounts for every applicable pause-generation member, including earlier selective releases.
- Resume-all clears stale generation associations only when no unresolved stopped member remains.
- Selective replay CLI output uses authoritative persisted status, signal result, and counters.
- Process identity validation now includes exact command-line evidence alongside executable, process group, and start identity.
- Cancellation, migration, crash-window, primary/child, and realistic process-identity coverage was expanded.

### 3. Files modified and why

- [GlobalExperimentControl.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp>) — ownership fencing, replay recovery, generation-wide accounting, departed-member cleanup, identity validation, and CLI reporting.
- [GlobalExperimentControl.hpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.hpp>) — selective-resume command and pause-generation interfaces.
- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/ExperimentScheduler.cpp>) — CLI routing and scheduler-status pause-generation reporting.
- [GlobalExperimentControlProcessTests.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlProcessTests.cpp>) — deterministic ownership, crash, cancellation, primary/child, accounting, and identity fixtures.
- [GlobalExperimentControlIntegrationTests.sh](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlIntegrationTests.sh>) — isolated migration, CLI, replay, and crash integration coverage.
- [GlobalExperimentControlMigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlMigrationTests.sql>) — mixed-state backfill and idempotence assertions.
- [GlobalExperimentControls.rst](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/docs/GlobalExperimentControls.rst>) — updated ownership, replay, accounting, identity, and cleanup contracts.
- [047_global_pause_selective_resume.sql](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Database/migrations/047_global_pause_selective_resume.sql>) — reviewed and exercised extensively; no concrete corrective SQL change was required during this pass. It remains an untracked part of the existing implementation.

### 4. Ownership and replay invariants

Post-signal accounting now locks the request and global-control rows and verifies the exact `application_owner` in the same transaction as guarded mutations.

A stale invocation cannot:

- complete or rewrite outcomes;
- mutate worker control state;
- clear the lease or active request;
- clear or replace the current pause generation;
- interfere with a newer pause-all generation.

Exact active selective-resume requests are reconstructed and claimed before ordinary lifecycle checks. Replay therefore handles workers reaped into pending, completed, failed, or cancelled states after successful `SIGCONT`.

### 5. Resume-all cleanup and accounting

Resume-all targets the full successfully paused generation:

- `target_count`: every applicable generation member.
- `successful_count`: members actually signaled successfully.
- `already_satisfied_count`: selectively released or previously reconciled members.
- `missing_count`: safely departed members.
- `failed_count`/`rejected_count`: unresolved signaling or identity failures.

Departed members receive durable `process_missing` outcomes without counting as failures. Only live, still-stopped unresolved members preserve the pause generation. Primary and checkpoint-child stale associations are otherwise cleared.

Tests verify a three-member generation with two earlier selective releases reports `target=3`, `successful=1`, `already_satisfied=2`, while sending only one `SIGCONT`.

### 6. CLI accounting and exit codes

Selective replay now reports:

- original request ID;
- `replay`;
- invocation-local `signal_attempted`;
- persisted `signal_result`;
- authoritative status and all request counters.

Completed fresh requests, successful replay, and safely departed recovery return exit code `0`. Partial, failed, rejected, identity-invalid, and unsafe recovery outcomes return `1`.

### 7. Migration and backfill verification

Verified:

- paused checkpoint-child backfill;
- mixed paused/running/missing/rejected pause outcomes;
- upgrade during an incomplete active pause;
- paused state without a usable completed generation;
- migration idempotence;
- end-to-end migration-046 evidence backfilled by 047, followed by production CLI selective resume, replay, and resume-all.

### 8. Primary/child, cancellation, and crash verification

Added coverage for:

- selective primary release while its checkpoint child remains stopped;
- no accidental child `SIGCONT`;
- checkpoint-child departure before resume-all;
- immediate and after-next-checkpoint cancellation with mixed released/stopped workers;
- cancellation lease takeover and replay accounting;
- crash after `SIGCONT` but before accounting;
- ownership takeover after lease expiry;
- stale-owner reconciliation attempts;
- worker lifecycle transitions before replay;
- newer concurrent pause generation protection;
- resume-all departure before accounting;
- realistic start-identity, executable, command-line, process-group, exited-process, PID-replacement, and stale-PID failures.

All process tests used disposable controlled processes. No production worker was signaled.

### 9. Commands run and results

Passed:

```text
/tmp/GlobalExperimentControlTests.focused
GlobalExperimentControlTests passed

/tmp/GlobalExperimentControlProcessTests.focused
GlobalExperimentControlProcessTests passed

Tests/GlobalExperimentControlIntegrationTests.sh \
  DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release \
  /tmp/GlobalExperimentControlProcessTests.focused
GlobalExperimentControlCrashWindowTests passed
GlobalExperimentControlIntegrationTests passed

git diff --check
git diff --cached --check
bash -n Tests/GlobalExperimentControlIntegrationTests.sh
```

The required Release build was also run. The unmodified invocation encountered the project’s pre-existing absolute main-checkout header path and produced a `CanonicalSymbol.hpp` redefinition. The established worktree-safe invocation overriding `USER_HEADER_SEARCH_PATHS` and `LIBRARY_SEARCH_PATHS` succeeded.

### 10. Remaining limitations or blockers

- The plain Xcode invocation remains sensitive to the project’s absolute main-checkout search path. The worktree-safe Release build succeeds.
- Xcode reports the established libpqxx 7.10 `exec_params` deprecation diagnostics. No new non-deprecation compile diagnostics were observed; the warning-strict focused compile suppresses only that known API deprecation.
- Production database and active production workers were intentionally not exercised.

### 11. Commit-readiness assessment

Ready for final independent verification. No implementation or test failure remains.

No commit was created. Nothing is staged. The generated review/CEE files and candidate transcripts were explicitly verified as not staged; no new corrective CEE artifact was generated.

### 12. Final git status and diff summary

```text
 M Sources/ExperimentScheduler.cpp
 M Sources/GlobalExperimentControl.cpp
 M Sources/GlobalExperimentControl.hpp
 M Tests/GlobalExperimentControlIntegrationTests.sh
 M Tests/GlobalExperimentControlMigrationTests.sql
 M Tests/GlobalExperimentControlProcessTests.cpp
 M docs/GlobalExperimentControls.rst
?? Database/migrations/047_global_pause_selective_resume.sql
?? GlobalPause_SelectiveResume_Defect_Fix_CEE_Output.md
?? GlobalPause_SelectiveResume_FocusedVerificationReview_Output.md
```

```text
 Sources/ExperimentScheduler.cpp                  |   46 +
 Sources/GlobalExperimentControl.cpp              | 1119 +++++++++++++++++-
 Sources/GlobalExperimentControl.hpp              |   24 +
 Tests/GlobalExperimentControlIntegrationTests.sh |  355 ++++++
 Tests/GlobalExperimentControlMigrationTests.sql  |   47 +
 Tests/GlobalExperimentControlProcessTests.cpp    | 1349 +++++++++++++++++++++-
 docs/GlobalExperimentControls.rst                |   86 +-
 7 files changed, 2963 insertions(+), 63 deletions(-)
```

The stat excludes the untracked 134-line migration and untracked generated review artifacts.