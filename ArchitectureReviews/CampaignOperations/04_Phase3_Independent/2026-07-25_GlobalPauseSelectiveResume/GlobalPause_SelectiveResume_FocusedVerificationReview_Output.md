---
title: "Global Pause Selective Resume Focused Verification Review"
document_type: "architecture review"
status: "final"
generated_from: "GlobalPause_SelectiveResume_FocusedVerificationReview_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Global Pause Selective Resume Focused Verification Review

## Executive summary

The implementation is **not ready to commit**. The generation predicates and migration backfill appear structurally sound, and the isolated migration/process tests pass, but two production-reachable crash/concurrency defects can strand control state or allow a stale invocation to undo a newer pause.

No files were modified during this review.

## Findings, ordered by severity

### High — Lease ownership is not fenced during post-signal reconciliation

After an expired selective-resume lease is claimed by a second invocation, the original invocation can still reconcile because the accounting transaction checks only the active request and pause generation—not `application_owner`. See [GlobalExperimentControl.cpp:2480](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2480>) through [GlobalExperimentControl.cpp:2602](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2602>).

A harmful sequence is possible:

1. Invocation A sends `SIGCONT` and stalls past lease expiry.
2. Invocation B claims the expired request and commits ownership.
3. A resumes, accounts, and clears the active gate despite no longer owning it.
4. A new pause-all request stops the worker.
5. B sends its stale `SIGCONT`, leaving the process running while the newer generation is recorded as paused.

Reconciliation and active-gate release must be conditional on the current invocation still owning the request. Add a lease-takeover/stale-owner test.

### High — Selective-resume replay can permanently strand the active request

On replay, lifecycle validation occurs before recovery of the existing active selective request at [GlobalExperimentControl.cpp:2294](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2294>). If the scheduler reaps the resumed worker after `SIGCONT` but before accounting, the experiment can become pending, completed, or failed. Replay then returns `invalid_lifecycle` without reconciling or clearing the active request.

Because scheduler child reaping continues while normal dispatch is gated, this is production-reachable. The administrative gate can remain permanently occupied, contradicting the recovery guarantee in [GlobalExperimentControls.rst:188](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/docs/GlobalExperimentControls.rst:188>).

Add a crash test where the exact resumed worker exits and its lifecycle transition is persisted before lease recovery.

### Medium — Resume-all can retain a stale pause generation after a worker departs

Resume-all updates pause state only for still-running rows at [GlobalExperimentControl.cpp:1992](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:1992>). Final cleanup clears associations only where `worker_control_state='running'`, while generation removal is blocked by any associated `paused` row regardless of lifecycle status at [GlobalExperimentControl.cpp:2082](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2082>).

If a resumed worker completes before accounting or replay, its row can remain `worker_control_state='paused'` with the old association, leaving `current_pause_request_id` populated even though global state is running.

Add primary-worker and checkpoint-child departure tests for resume-all recovery.

### Medium — Resume-all audit omits selectively released workers it reconciles

Resume-all counts and creates outcomes only for workers still marked paused at [GlobalExperimentControl.cpp:1598](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:1598>) and [GlobalExperimentControl.cpp:1632](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:1632>). It later clears associations from selectively released workers without adding outcomes at [GlobalExperimentControl.cpp:2082](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2082>).

Consequently, a three-worker generation with two selective releases reports resume-all `target_count=1`, `already_satisfied_count=0`, despite reconciling all three associations. The current test checks only the remaining `SIGCONT` and cleared associations, not request accounting: [GlobalExperimentControlProcessTests.cpp:1900](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlProcessTests.cpp:1900>).

### Medium — Replay CLI counters contradict persisted request accounting

Successful replay prints the original request ID but hard-codes invocation-local counters and `signal_result=not_attempted` at [GlobalExperimentControl.cpp:2353](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2353>). For an originally signaled request, the persisted row instead has `successful_count=1` and `signal_result=signaled`.

Either report authoritative persisted accounting or label replay-local fields distinctly. The present output is internally inconsistent for machine consumers.

### Low — Test and commit hygiene

- The identity-failure fixture modifies frozen audit evidence directly at [GlobalExperimentControlProcessTests.cpp:2123](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlProcessTests.cpp:2123>), despite runtime update permission being revoked. This can remain as a corruption-defense test, but it does not replace a production-reachable PID/process-identity mismatch fixture.
- `GlobalPause_SelectiveResume_Defect_Fix_CEE_Output.md` is untracked and not ignored. It must be excluded from the intended commit.

## Migration concerns

No concrete migration SQL defect was identified. The primary upgrade/backfill and migration idempotence checks passed.

Missing migration verification:

- Already-paused checkpoint-inference child backfill.
- Mixed partial pause with paused, running, missing, and rejected workers.
- Upgrade during an active/incomplete pause request.
- Paused installation with no previously completed pause generation.
- End-to-end selective resume using evidence created by migration 046 and backfilled by 047.

## Other missing verification

Add focused tests for:

- Stale lease owner versus new owner and a concurrent newer pause generation.
- Selective `SIGCONT`, crash, scheduler lifecycle transition, then replay.
- Resume-all worker departure before accounting.
- Exact resume-all target/success/already-satisfied counters after selective releases.
- Primary selective release while a checkpoint child remains stopped.
- Immediate and after-checkpoint cancellation with mixed selectively released/stopped primary and checkpoint workers.
- Replay CLI output against persisted request accounting.

## Verification performed

- `git diff --check` — passed.
- Migration upgrade/idempotence and `GlobalExperimentControlCrashWindowTests` — passed in an isolated temporary database.
- Full integration CLI stage — not run because no current executable could be linked.
- Required Release build — failed before linking due the project’s absolute header search path mixing worktree and main-repository `CanonicalSymbol.hpp`.
- A diagnostic build with the header path overridden compiled the reviewed sources but then failed linking because `MetalBuffer` was unavailable.

A production scheduler and seven training workers were active, so no production CLI or database-mutating verification was performed.

## Exact files requiring modification

- [Sources/GlobalExperimentControl.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp>)
- [Tests/GlobalExperimentControlProcessTests.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlProcessTests.cpp>)
- [Tests/GlobalExperimentControlIntegrationTests.sh](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlIntegrationTests.sh>)
- [docs/GlobalExperimentControls.rst](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/docs/GlobalExperimentControls.rst>)

Migration tests should also be expanded in [GlobalExperimentControlMigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlMigrationTests.sql>). No migration SQL modification is currently indicated.

## Commit readiness

**Not ready to commit.** Resolve the two high-severity recovery/ownership defects, correct resume-all and replay accounting, add the focused tests above, exclude the CEE artifact, and obtain a successful required Release build and full isolated integration run.

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
```

`git diff --stat`:

```text
 Sources/ExperimentScheduler.cpp                  |  46 ++
 Sources/GlobalExperimentControl.cpp              | 767 ++++++++++++++++++++++-
 Sources/GlobalExperimentControl.hpp              |  24 +
 Tests/GlobalExperimentControlIntegrationTests.sh |  82 +++
 Tests/GlobalExperimentControlMigrationTests.sql  |  47 ++
 Tests/GlobalExperimentControlProcessTests.cpp    | 520 ++++++++++++++-
 docs/GlobalExperimentControls.rst                |  51 +-
 7 files changed, 1487 insertions(+), 50 deletions(-)
```

The stat excludes the two untracked files.