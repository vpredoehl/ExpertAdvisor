---
title: "Global Pause Selective Resume Final Verification Corrections"
document_type: "architecture review"
status: "final"
generated_from: "GlobalPause_SelectiveResume_FinalVerification_Corrections_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Global Pause Selective Resume Final Verification Corrections

## 1. Executive summary

Completed the bounded corrective pass on `fix/global-pause-selective-resume`.

The implementation now:

- Fences scheduler cancellation reconciliation with the exact persisted request owner and live lease.
- Uses immutable worker-outcome evidence for generic replay.
- Preserves pause generation and active gate for every unresolved worker.
- Treats replacement, identity, permission, and inspection failures as unresolved.
- Checks predicate-sensitive post-signal mutations and retains the lease after predicate mismatch.
- Emits final CLI accounting from committed request/outcome rows.
- Adds real-process crash, replacement, takeover, cancellation, and inspection-failure coverage.
- Runs every required migration scenario from an isolated migration-046 schema.
- Passes focused tests, the full isolated integration suite, and the Release build.

No production process or database was modified.

## 2. Root causes corrected

1. Scheduler reconciliation previously lacked an explicit invocation owner; request shape and active request state could authorize mutations without proving lease ownership.
2. Several post-signal updates accepted zero affected rows and continued accounting.
3. Cleanup could clear generation/gate state before every stopped member was positively reconciled.
4. Identity ambiguity and inspection failure were conflated with process departure.
5. Generic replay rebuilt targets from mutable worker rows instead of frozen outcomes.
6. There was no complete signal-success/crash/replacement/takeover production-path test.
7. Some migration cases reused a schema already upgraded by migration 047.
8. CLI summaries used local counters or abbreviated action names and could be emitted before final commit.
9. Cancellation coverage omitted owner fencing, replacement, permission/inspection failure, and mixed primary/checkpoint-child suspension cases.

## 3. Scheduler cancellation ownership model

Each scheduler invocation derives a stable owner identity from its PID, kernel process-start identity, and executable.

`ReconcileActiveCancellation()` now:

- Requires a non-empty application owner.
- Locks the exact active `cancel_all` request and global control row.
- Rejects a foreign live lease without mutation.
- Claims only an absent or expired lease, or refreshes the same owner.
- Verifies request ID, action, active-request relationship, owner, and live lease in every guarded mutation.
- Prevents stale owners from updating outcomes, workers, accounting, generation, or gate after takeover.
- Retains the existing coordination advisory lock and request-shape validation.

## 4. Post-signal mutation audit

Checked or authoritatively reread mutations include:

- Primary and checkpoint-child pause state and generation association.
- Resume-all and selective-resume control-state transitions.
- Immediate cancellation lifecycle changes.
- After-checkpoint resume, restart, and terminal transitions.
- Process-missing reconciliation.
- Per-worker signal/outcome accounting.
- Checkpoint inference assignment and reconciliation.
- Request totals and status.
- Pause-generation member cleanup.
- `current_pause_request_id` cleanup.
- Active-request gate cleanup.
- Lease refresh and release.

Exact predicates cover worker identity, lifecycle, PID, process group, start identity, executable, command, cancellation request, pause generation, action, active request, owner, and live lease.

Unexpected row counts either throw and roll back or trigger a locked authoritative reread. A post-signal predicate mismatch is persisted as unresolved, cannot receive successful accounting, keeps the active gate, and now retains its live lease until expiry/takeover.

## 5. Unresolved worker behavior

Unresolved members preserve:

- `worker_global_pause_request_id`
- `current_pause_request_id`
- `active_request_id`
- stopped worker state and recoverable frozen plan

Cancellation work that remains actionable stays `pending`; signal or identity failures in otherwise terminal pause/resume work remain `partial` behind the active gate.

Even when resume-all has set `desired_state=running`, normal train, inference, analysis, checkpoint inference, continuation, and replacement dispatch remain blocked until every applicable member is safely reconciled.

## 6. Frozen-evidence replay

Generic pause, resume, cancellation, and cancellation recovery now reconstruct targets from `experiment_admin_worker_outcome`, including:

- Worker identity and kind
- Experiment and checkpoint-evaluation IDs
- Phase and frozen lifecycle
- PID and process group
- Process-start identity
- Executable and exact command line
- Source pause generation
- Cancellation checkpoint and model
- Inference plan and state

Current lifecycle rows are compared separately. Missing rows do not erase the plan, and replacement rows or reused lifecycle slots are never signaled or credited. Immutable identity fields use insert-once behavior and are not overwritten during replay.

Retryable unresolved `failed` or `partial` outcomes can now be safely reconsidered after authoritative evidence becomes reconcilable.

## 7. Replacement-process replay test

The new test uses real disposable process groups:

1. Pauses an original worker.
2. Sends resume successfully.
3. Blocks the original invocation before accounting.
4. Terminates the original worker.
5. Starts a real replacement in the same persisted lifecycle slot.
6. Expires the original request lease.
7. Replays with a new owner.
8. Releases the stale original invocation.

It verifies:

- No replacement signal.
- Frozen original PID/start identity/command remain unchanged.
- Replacement identity is persisted as unresolved.
- Request remains partial behind the gate.
- Pause generation remains.
- Replacement receives no successful accounting.
- Stale owner loses all mutation authority.
- Machine output and exit status agree with SQL state.

Numeric PID reuse was not forced because macOS does not provide a safe deterministic PID allocation fixture; the required real replacement lifecycle-slot case and separate start-identity/PID-reuse validation are covered.

## 8. Authentic pre-047 migration fixtures

Each required case now receives its own schema:

- `gp_mig_primary`
- `gp_mig_child`
- `gp_mig_mixed`
- `gp_mig_active`
- `gp_mig_no_usable`
- `gp_mig_idempotent`
- `gp_mig_selective`

Every scenario:

1. Creates the migration-045-era base tables.
2. Applies migration 046.
3. Asserts 047-only columns, indexes, and `resume_experiment` shape are absent.
4. Inserts only migration-046-era evidence.
5. Applies migration 047.
6. Asserts the upgraded state.
7. Drops the isolated schema.

The selective-consumption scenario launches a real stopped disposable worker, applies migration 047, then runs the production selective-resume CLI against that isolated schema.

Migration 047 itself was left unchanged because the authentic fixtures did not reveal a migration defect.

## 9. CLI and exit-code contract

Final applied/replay output is buffered until accounting commits.

Summary fields are:

- `request_id`
- `action`
- `status`
- `result`
- `replay`
- `signal_attempted`
- `target_count`
- `successful_count`
- `already_satisfied_count`
- `missing_count`
- `rejected_count`
- `failed_count`

Worker lines contain:

- `worker_identity`
- `identity_result`
- `outcome_status`
- `signal_result`
- `requested_signal`
- `detail`

Actions use persisted names, including `resume_experiment`.

Exit behavior:

| Persisted state | Exit |
|---|---:|
| `completed` | 0 |
| `pending` | 0 |
| `partial` | 1 |
| `failed` | 1 |
| rejected invocation | 1 |

The status mapper now has focused assertions for completed, pending, partial, failed, and applying states.

## 10. Cancellation regression coverage

Coverage now includes:

- Foreign live scheduler owner rejection.
- Lease expiry and authoritative takeover.
- Stale-owner mutation rejection.
- Selectively released primary with stopped checkpoint child.
- Immediate and after-next-checkpoint modes.
- Real replacement process.
- Identity mismatch.
- Permission denial and inspection failure around real stopped workers.
- Positively missing process.
- Generation/gate preservation for unresolved suspension.
- Cleanup only after all members reconcile.
- Exact request/outcome accounting and CLI fields.
- Pending, completed, partial, failed exit mapping.
- Newer-generation protection.
- Foreign `cancellation_request_id` protection.
- Checkpoint and infer-before-cancel behavior.
- Disposable-worker cleanup proving no stopped worker is stranded.

Independent child-only selective release was not invented because the production selective-resume command targets the primary worker; the inverse permutation is covered only where production-reachable.

## 11. Files changed

- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/ExperimentScheduler.cpp>) — stable scheduler reconciliation owner and owner-aware calls.
- [GlobalExperimentControl.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp>) — ownership fencing, frozen replay, mutation checks, unresolved behavior, accounting, and CLI.
- [GlobalExperimentControl.hpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.hpp>) — owner-aware reconciliation API and shared exit-status contract.
- [GlobalExperimentControlTests.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlTests.cpp>) — exit-code contract assertions.
- [GlobalExperimentControlProcessTests.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlProcessTests.cpp>) — real-process crash, takeover, replacement, predicate-race, and cancellation tests.
- [GlobalExperimentControlIntegrationTests.sh](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlIntegrationTests.sh>) — isolated authentic migration scenarios and CLI/SQL verification.
- [GlobalExperimentControlMigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlMigrationTests.sql>) — upgraded-schema assertions.
- [GlobalExperimentControls.rst](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/docs/GlobalExperimentControls.rst>) — final ownership, replay, unresolved, CLI, and exit contracts.
- [047_global_pause_selective_resume.sql](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Database/migrations/047_global_pause_selective_resume.sql>) — inspected and tested unchanged; remains untracked.

## 12. Verification performed

Passed:

- `git diff --check`
- `bash -n Tests/GlobalExperimentControlIntegrationTests.sh`
- Focused unit build/run with C++20, `-Wall -Wextra -Werror`
  - `GlobalExperimentControlTests passed`
- Native process build/run with C++20, `-Wall -Wextra -Werror`
  - `GlobalExperimentControlProcessTests passed`
- Full isolated suite:
  - `GlobalExperimentControlCrashWindowTests passed`
  - `GlobalExperimentControlIntegrationTests passed`
- Worktree-safe Release build:

```sh
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor \
  'USER_HEADER_SEARCH_PATHS=$(PROJECT_DIR)/Headers' \
  'LIBRARY_SEARCH_PATHS=$(inherited) "/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release"' \
  build
```

Result: `** BUILD SUCCEEDED **`

The production scheduler and seven production training workers were detected before testing and remained running afterward. All database tests used uniquely named temporary databases and disposable process groups.

## 13. Remaining risks

- The Release build still reports existing libpqxx `exec_params` deprecation warnings and the local multiple-destination/toolchain warning. Focused corrected sources compile under `-Werror` with the project’s established deprecation suppression. Converting the repository’s broader libpqxx usage was intentionally outside this bounded pass.
- Forced same-numeric-PID reuse was not attempted; real replacement-slot and frozen start-identity coverage provides the safe deterministic equivalent.

## 14. Final repository state

`git status --short`:

```text
 M Sources/ExperimentScheduler.cpp
 M Sources/GlobalExperimentControl.cpp
 M Sources/GlobalExperimentControl.hpp
 M Tests/GlobalExperimentControlIntegrationTests.sh
 M Tests/GlobalExperimentControlMigrationTests.sql
 M Tests/GlobalExperimentControlProcessTests.cpp
 M Tests/GlobalExperimentControlTests.cpp
 M docs/GlobalExperimentControls.rst
?? Database/migrations/047_global_pause_selective_resume.sql
?? GlobalPause_SelectiveResume_Defect_Fix_CEE_Output.md
?? GlobalPause_SelectiveResume_FinalIndependentVerification_Output.md
?? GlobalPause_SelectiveResume_FocusedVerificationReview_Output.md
?? GlobalPause_SelectiveResume_VerificationFindings_Correction_Output.md
```

`git diff --stat`:

```text
 Sources/ExperimentScheduler.cpp                  |   67 +-
 Sources/GlobalExperimentControl.cpp              | 2402 +++++++++++++++++++---
 Sources/GlobalExperimentControl.hpp              |   36 +-
 Tests/GlobalExperimentControlIntegrationTests.sh |  773 +++++++
 Tests/GlobalExperimentControlMigrationTests.sql  |   47 +
 Tests/GlobalExperimentControlProcessTests.cpp    | 1921 ++++++++++++++++-
 Tests/GlobalExperimentControlTests.cpp           |    5 +
 docs/GlobalExperimentControls.rst                |  157 +-
 8 files changed, 5058 insertions(+), 350 deletions(-)
```

The untracked 134-line migration is not included in `git diff --stat`.

## 15. Commit hygiene

No commit was created. Nothing was staged (`git diff --cached --stat` is empty). Generated CEE/review artifacts remain untracked and excluded, as do DerivedData, temporary binaries, build logs, dumps, and scratch files.