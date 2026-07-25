---
title: "Global Pause Selective Resume Final Independent Verification"
document_type: "architecture review"
status: "final"
generated_from: "GlobalPause_SelectiveResume_FinalIndependentVerification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Global Pause Selective Resume Final Independent Verification

# Final independent verification report

## 1. Executive summary

The branch is **not ready to commit**.

The existing focused unit, process, crash-window, migration, integration, CLI smoke, and worktree Release build all pass. However, independent tracing found commit-blocking gaps that those tests do not exercise:

- Scheduler-driven cancellation reconciliation is not fenced by `application_owner`.
- Multiple post-signal, predicate-sensitive mutations ignore `affected_rows()`.
- Resume/cancel cleanup can release the active gate—or clear the pause generation—while a live stopped worker remains unresolved.
- Generic replay does not consistently use frozen process evidence.
- Several required migration scenarios are built after migration 047 has already been applied.
- Machine-readable output does not exactly expose or agree with persisted action/outcome data across all paths.

No files were modified, staged, or committed during this review.

## 2. Commit-readiness verdict

**NOT READY TO COMMIT.**

The branch needs a bounded corrective pass in global-control reconciliation and its focused tests. This is not a request to redesign the scheduler or broaden the feature.

## 3. Findings ordered by severity

### Critical — scheduler cancellation reconciliation is owner-unfenced

[ReconcileActiveCancellation](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:1517) directly mutates outcomes, request accounting, and `active_request_id` without verifying the exact persisted `application_owner`.

It is called by the scheduler at [ExperimentScheduler.cpp:13897](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/ExperimentScheduler.cpp:13897) and [ExperimentScheduler.cpp:13948](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/ExperimentScheduler.cpp:13948), where no application-owner identity is supplied or claimed.

Consequences:

- A scheduler can reconcile or clear a request while another invocation owns a live lease.
- A scheduler can act after lease takeover without proving it owns the exact request.
- This contradicts the documented guarantee at [GlobalExperimentControls.rst:43](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/docs/GlobalExperimentControls.rst:43).

### Critical — unresolved stopped workers do not preserve the administrative gate

Resume-all records an identity/permission/inspection failure, preserves the worker association, but unconditionally clears `active_request_id` at [GlobalExperimentControl.cpp:2433](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2433). Since resume-all has already set `desired_state=running`, normal scheduling becomes allowed despite a still-stopped unresolved worker.

Cancellation is more severe:

- Identity mismatch is treated as safely detached at [GlobalExperimentControl.cpp:730](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:730).
- Identity failure can transition the database worker to terminal at [GlobalExperimentControl.cpp:2323](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2323).
- Cancellation then clears every association and `current_pause_request_id` unconditionally at [GlobalExperimentControl.cpp:2378](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2378), including after permission or inspection failures.

A live stopped replacement or uninspectable worker can therefore be forgotten and stranded.

The unresolved-member test at [GlobalExperimentControlProcessTests.cpp:2313](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlProcessTests.cpp:2313) verifies that the generation remains, but does **not** verify that `active_request_id` remains occupied.

### High — predicate failures can silently become successful accounting

Generic post-signal paths do not check affected rows:

- Per-worker outcome: [GlobalExperimentControl.cpp:747](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:747)
- After-checkpoint worker reconciliation: [GlobalExperimentControl.cpp:2140](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2140)
- Pause reconciliation: [GlobalExperimentControl.cpp:2278](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2278)
- Resume-all reconciliation: [GlobalExperimentControl.cpp:2300](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2300)
- Cancellation terminal mutation: [GlobalExperimentControl.cpp:2323](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2323)
- Generation and gate cleanup: [GlobalExperimentControl.cpp:2381](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2381)

For example, `SIGCONT` can succeed, the worker row predicate can subsequently match zero rows, yet the outcome can still become completed and the gate can be released.

Selective resume checks the main success update’s `affected_rows()` at [GlobalExperimentControl.cpp:2856](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2856), but its fallback, missing-worker, outcome, lease, and gate mutations remain unchecked at lines 2894–2957.

### High — frozen replay evidence is incomplete outside selective resume

Selective replay correctly loads its frozen plan from the outcome row at [GlobalExperimentControl.cpp:899](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:899).

Generic pause/resume/cancel replay instead loads current worker rows and correlates them to outcomes using only `worker_identity` at [GlobalExperimentControl.cpp:416](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:416). It does not reconstruct PID, start identity, executable, command, or process group from the frozen outcome.

The tests provide:

- Low-level fake PID-reuse validation in [GlobalExperimentControlTests.cpp:186](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlTests.cpp:186).
- Native identity-component mismatches in [GlobalExperimentControlProcessTests.cpp:3018](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlProcessTests.cpp:3018).
- Selective replay after the original process exits and the database PID is cleared in [GlobalExperimentControlProcessTests.cpp:2481](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlProcessTests.cpp:2481).

There is no full replay test with a live replacement process occupying the persisted lifecycle slot or reused numeric PID. The required crash-window scenario is therefore not fully verified.

### High — migration fixtures are only partially authentic

The initial fixture is genuinely pre-047:

1. Base schema is created.
2. Migration 046 is applied at [GlobalExperimentControlIntegrationTests.sh:192](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlIntegrationTests.sh:192).
3. Primary, checkpoint-child, mixed-outcome, and active-request data are inserted.
4. Migration 047 is first applied at line 281.

That initial block authentically covers several backfills.

However:

- The claimed migration-046-to-selective-resume fixture writes the 047-only `current_pause_request_id` at [line 461](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlIntegrationTests.sh:461), after 047 was already applied, then reapplies 047.
- The “no usable generation” fixture also writes `current_pause_request_id` at [line 350](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlIntegrationTests.sh:350), again after 047.
- [GlobalExperimentControlMigrationTests.sql](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlMigrationTests.sql:1) contains post-migration schema/privilege assertions, not independent pre-047 fixtures.
- All cases reuse one database rather than isolating each upgrade starting state.

Thus genuine pre-047 coverage is missing for the migration-046 evidence later consumed by selective resume and the paused-with-no-usable-generation case.

### High — CLI output is not an exact persisted-row representation

Selective output at [GlobalExperimentControl.cpp:1018](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:1018):

- Prints `action=resume`, while the persisted action is `resume_experiment`.
- Prints `identity=`, not `identity_result=`.
- Omits persisted `outcome_status`.
- Correctly distinguishes `replay` and invocation-local `signal_attempted`.

Generic output at [GlobalExperimentControl.cpp:2470](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2470):

- Omits a machine-readable `result`.
- Omits `replay` and invocation-local `signal_attempted`.
- Emits replay only as a separate retry line.
- Returns based on invocation-local `failed` at [line 2507](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2507), not persisted request status. A replay containing only already-accounted partial outcomes can therefore return zero while the database reports `partial`.

Existing tests mostly use substring `grep`/`find`, not exact field-by-field comparison against a direct SQL row.

## 4. Owner-fencing trace

| Path | Post-signal owner check | Verdict |
|---|---|---|
| Selective `SIGCONT` | Exact active request, action, owner, and pause generation at line 2826 | Main fence is correct |
| Pause-all `SIGSTOP` | Exact active request/action/owner at line 2236 | Fenced, but affected rows unchecked |
| Resume-all `SIGCONT` | Exact active request/action/owner at line 2236 | Fenced, but generation is not part of `OwnsActiveRequest` and predicates are unchecked |
| Immediate cancellation | Exact active request/action/owner at line 2236 | Fenced, but unsafe identity/departure handling |
| After-next-checkpoint `SIGCONT` | Exact active request/action/owner at line 2122 | Fenced, but reconciliation predicates unchecked |
| Final command accounting/cleanup | Owner checked at line 2364 | Fenced inside command invocation |
| Scheduler cancellation reconciliation | No owner check | **Fails requirement** |
| Pause-all replay generation restoration | `current_pause_request_id` is changed at line 1800 before the new owner is persisted at line 1806 | **Fails strict owner-ordering requirement** |

## 5. Affected-row and stale-predicate analysis

Only selective resume’s primary successful worker mutation checks for exactly one affected row.

No equivalent deterministic handling exists for generic worker-state mutations, request accounting, gate release, lease clearing, cancellation terminal transitions, or pause-generation cleanup. Zero-row results can allow later accounting and cleanup to continue.

This is a concrete silent-failure mode, not merely missing defensive validation.

## 6. PID-reuse replay analysis

The low-level validator is strong: PID, process group, start identity, executable, exact command, phase, experiment ID, and checkpoint-evaluation ID are checked.

Selective replay also uses frozen outcome evidence and will not signal a live PID whose start identity or command has changed.

The branch nevertheless lacks:

- Full database replay against a controlled replacement process.
- Generic replay reconstruction from frozen outcomes.
- A test proving that accounting and pause cleanup remain blocked when the replacement is live but does not match the original invocation.

## 7. Pause-generation cleanup analysis

Positive behavior:

- Selectively released workers remain associated with the generation.
- Resume-all audits selectively released and departed members.
- Current generation is not cleared while an associated running/paused row with a non-null PID remains.

Unsafe behavior:

- Identity mismatch is classified as departed without OS inspection in [LoadPauseGenerationTargets](/Volumes/Developer%20SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:525).
- Lifecycle-terminal or null-PID rows are cleaned without proving the process is gone.
- Resume-all releases the active gate after partial/unresolved outcomes.
- Cancellation clears the generation unconditionally, including after permission or inspection failure.

## 8. Cancellation regression analysis

The existing tests successfully exercise:

- Immediate cancellation of selectively released primary plus stopped checkpoint child.
- `SIGCONT` before cancellation for stopped workers.
- After-next-checkpoint behavior.
- Lease-expiry replay.
- Primary/checkpoint-child request accounting.
- Generation cleanup on successful cancellation.

They do not establish:

- Stale-owner rejection for scheduler reconciliation.
- Cancellation behavior after identity mismatch, permission denial, or inspection failure.
- Preservation of the generation and gate when a stopped worker cannot safely be reconciled.
- Exact CLI/SQL equality and exit codes on partial cancellation replay.

Cancellation therefore cannot be certified regression-free.

## 9. Migration-fixture authenticity

Authentic pre-047 coverage exists for the first mixed fixture only. Required authentic coverage remains missing for:

- Paused global state without a usable completed generation.
- Migration-046 pause evidence later consumed by production selective resume.
- Independent uncontaminated idempotence runs.
- An active/incomplete pause generation with production-reachable outcome combinations.

## 10. CLI-versus-database accounting

Selective fresh/replay counters are loaded after accounting and generally agree with their request row. However, field naming/action and outcome coverage are incomplete.

Generic summary counters are read from persisted rows after accounting, but invocation-local facts are absent and the exit code is not derived from persisted status.

No hard-coded counter defect remains in the selective success/replay path, but exact cross-path equality is not proven.

## 11. Tests and builds run

Passed:

```text
git diff --check
git diff --cached --check
bash -n Tests/GlobalExperimentControlIntegrationTests.sh
/tmp/GlobalExperimentControlTests.final_verify
/tmp/GlobalExperimentControlProcessTests.final_verify
Tests/GlobalExperimentControlIntegrationTests.sh \
  DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release \
  /tmp/GlobalExperimentControlProcessTests.final_verify
```

Results:

```text
GlobalExperimentControlTests passed
GlobalExperimentControlProcessTests passed
GlobalExperimentControlCrashWindowTests passed
GlobalExperimentControlIntegrationTests passed
```

The prescribed build command failed because the project hard-codes the main checkout’s headers, producing duplicate `CanonicalSymbol` definitions:

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor build
```

A worktree-safe override succeeded:

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor \
  'USER_HEADER_SEARCH_PATHS=$(PROJECT_DIR)/Headers' \
  'LIBRARY_SEARCH_PATHS=$(inherited) "/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release"' \
  build
```

The production scheduler and seven production workers remained alive and were not signaled. No production database rows were touched.

## 12. Exact staged-file inventory

Nothing is staged.

```text
git diff --cached --name-only
# empty
```

`git diff --cached --check` passed vacuously.

After the defects are corrected, the safe explicit staging command for the current intended files is:

```bash
git add -- \
  Database/migrations/047_global_pause_selective_resume.sql \
  Sources/ExperimentScheduler.cpp \
  Sources/GlobalExperimentControl.cpp \
  Sources/GlobalExperimentControl.hpp \
  Tests/GlobalExperimentControlIntegrationTests.sh \
  Tests/GlobalExperimentControlMigrationTests.sql \
  Tests/GlobalExperimentControlProcessTests.cpp \
  docs/GlobalExperimentControls.rst
```

Do not stage the three `GlobalPause_SelectiveResume_*_Output.md` artifacts.

## 13. Exact files requiring modification

At minimum:

- `Sources/GlobalExperimentControl.cpp`
- `Sources/GlobalExperimentControl.hpp`
- `Sources/ExperimentScheduler.cpp`
- `Tests/GlobalExperimentControlProcessTests.cpp`
- `Tests/GlobalExperimentControlIntegrationTests.sh`
- `Tests/GlobalExperimentControlMigrationTests.sql`
- `docs/GlobalExperimentControls.rst`

No concrete migration-SQL defect was proven yet; `047_global_pause_selective_resume.sql` should change only if authentic pre-047 tests reveal one.

## 14. Remaining risks and blockers

- Owner-safe scheduler cancellation reconciliation needs an explicit authoritative ownership design consistent with the existing request lease.
- Every predicate-sensitive post-signal mutation needs checked affected rows or a locked authoritative reread.
- Unresolved live/stopped workers must preserve both generation and gate.
- Full replacement-process replay coverage is missing.
- Required migration fixtures must be rebuilt from isolated pre-047 schemas.
- Exact prescribed worktree build remains dependent on hard-coded external paths.
- The build emits widespread libpqxx `exec_params` deprecation warnings; these are largely pre-existing but new calls add to them.

## 15. Final repository state

`git status --short`:

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
?? GlobalPause_SelectiveResume_VerificationFindings_Correction_Output.md
```

Staged diff summary: empty.

Tracked unstaged diff summary:

```text
7 files changed, 2963 insertions(+), 63 deletions(-)
```

Additionally untracked:

- Intended 134-line migration 047.
- Three generated/review-output Markdown artifacts that must remain excluded.