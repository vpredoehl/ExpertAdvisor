---
title: "Phase 5/6 Checkpoint Worker Classification Fix"
document_type: "architecture review"
status: "final"
generated_from: "Phase5_6_Checkpoint_Worker_Classification_Fix_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Phase 5/6 Checkpoint Worker Classification Fix

Implemented and verified the focused checkpoint worker-classification fix. No commit was created.

### Root cause

Scheduler status derived managed workers exclusively from active `experiment` rows. It detected checkpoint inference processes, but never parsed `--scheduler-checkpoint-eval-id` or consulted `experiment_checkpoint_eval`, so completed parent experiments made valid checkpoint workers appear unmanaged.

### Behavioral correction

- Added a bounded lookup covering active experiment and checkpoint-evaluation workers.
- Checkpoint workers are authorized independently of parent experiment status.
- Reused existing PID, process-group, executable, command identity, and process-start validation.
- Checkpoint-tagged processes cannot fall through to ordinary experiment matching.
- Invalid, terminal, mismatched, unknown, or PID-reused workers remain unmanaged.
- Managed/unmanaged counts and CPU/RSS totals count each process exactly once.
- Checkpoint workers now consume `max-infer-procs` slots when ordinary inference is scheduled.
- Added human-readable “Active Checkpoint Inference Jobs” output and `SCHEDULER_STATUS_CHECKPOINT_JOB`.
- Existing machine-readable fields remain unchanged; their values are corrected.
- No schema, lifecycle, scheduling-policy, cancellation, inference, or campaign behavior changes.

### Files changed

- [ExperimentScheduler.hpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/checkpoint-worker-classification/Headers/ExperimentScheduler.hpp:9>)
- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/checkpoint-worker-classification/Sources/ExperimentScheduler.cpp:13055>)
- [GlobalExperimentControl.hpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/checkpoint-worker-classification/Sources/GlobalExperimentControl.hpp:98>)
- [GlobalExperimentControl.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/checkpoint-worker-classification/Sources/GlobalExperimentControl.cpp:859>)
- [GlobalExperimentControlTests.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/checkpoint-worker-classification/Tests/GlobalExperimentControlTests.cpp:288>)
- [Volume_XI_Scheduler.md](</Volumes/Developer SSD/ExpertAdvisor-worktrees/checkpoint-worker-classification/docs/architecture/Volume_XI_Scheduler.md:67>)

### Verification results

Passed:

- Focused `GlobalExperimentControlTests`
- `GlobalExperimentControlProcessTests`
- `SchedulerChildStatusTests`
- Full isolated `GlobalExperimentControlIntegrationTests`
- Release compilation and link with worktree-local header override
- `git diff --check`

Coverage includes normal inference, checkpoint workers with completed/running parents, unknown/missing IDs, PID and phase mismatches, terminal rows, PID reuse, command identity, exact-once counts/resources, infer-slot capacity, and pause/resume/cancellation regression behavior.

The exact prescribed build command still fails because the checked-in Xcode configuration adds the main checkout’s absolute `Headers` path alongside this worktree, producing duplicate `CanonicalSymbol.hpp` definitions. With `USER_HEADER_SEARCH_PATHS=$(SRCROOT)/Headers`, the Release build succeeds. This pre-existing worktree configuration was not changed. Existing unrelated compiler warnings remain.

A live production `--scheduler-status` invocation was intentionally not run because another worktree’s scheduler and seven training workers are active.

### Repository state

```text
 M Headers/ExperimentScheduler.hpp
 M Sources/ExperimentScheduler.cpp
 M Sources/GlobalExperimentControl.cpp
 M Sources/GlobalExperimentControl.hpp
 M Tests/GlobalExperimentControlTests.cpp
 M docs/architecture/Volume_XI_Scheduler.md
```

```text
 Headers/ExperimentScheduler.hpp          |  12 ++
 Sources/ExperimentScheduler.cpp          | 313 ++++++++++++++++++++++++-------
 Sources/GlobalExperimentControl.cpp      | 125 ++++++++++++
 Sources/GlobalExperimentControl.hpp      |  53 ++++++
 Tests/GlobalExperimentControlTests.cpp   | 254 +++++++++++++++++++++++++
 docs/architecture/Volume_XI_Scheduler.md |  16 +-
 6 files changed, 706 insertions(+), 67 deletions(-)
```

The implementation is ready to commit, subject to the documented pre-existing Xcode worktree-path issue.