---
title: "Global Pause Selective Resume Defect Fix"
document_type: "architecture review"
status: "final"
generated_from: "GlobalPause_SelectiveResume_Defect_Fix_CEE_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Global Pause Selective Resume Defect Fix

## 1. Root-cause assessment

Global pause and lifecycle pause used different control models:

- `--pause-all-experiments` left lifecycle status as `running`, recorded runtime suspension, and sent `SIGSTOP`.
- `--resume-experiment=ID` only accepted lifecycle status `paused`.
- No durable link identified which completed global-pause generation suspended a specific worker.

The CLI therefore could not safely distinguish a globally suspended running worker from an ordinary running worker.

## 2. Control semantics implemented

`--resume-experiment=ID [--dry-run] [--yes]` now supports both paths:

- Lifecycle `paused`: preserves the existing transition to `pending`, reported as `lifecycle_resumed`.
- Lifecycle `running`: selectively releases the worker only when it is associated with the current global-pause generation and its frozen pause outcome exactly matches current authoritative identity.

A successful selective release:

- Sends `SIGCONT` only to the validated process group.
- Records a targeted administrative request and outcome.
- Changes `worker_control_state` from `paused` to `running`.
- Retains the source pause association for replay and later reconciliation.
- Leaves `desired_state=paused`, preventing all unrelated dispatch.
- Leaves every unselected worker stopped.

Ordinary running workers without applicable evidence are rejected as `not_globally_suspended`.

## 3. Database/schema changes

Added [047_global_pause_selective_resume.sql](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Database/migrations/047_global_pause_selective_resume.sql:1>):

- `resume_experiment` administrative action.
- `target_experiment_id` request identity.
- `current_pause_request_id` on the singleton global control row.
- `worker_global_pause_request_id` on experiment and checkpoint-inference workers.
- Frozen executable, command-line, and source-pause evidence on worker outcomes.
- Shape constraints and supporting indexes.
- Upgrade backfill for an already-paused installation.
- Revoked `pqxx` update rights on frozen outcome evidence.

Lifecycle status was not overloaded to represent Unix suspension.

## 4. CLI behavior

Representative success:

```text
SCHEDULER_CONTROL_RESULT,action=resume,experiment_id=442,request_id=<request>,source_pause_request_id=3,result=globally_suspended_worker_resumed,identity=validated,signal_result=signaled,global_state=paused,status=completed,target_count=1,successful_count=1,already_satisfied_count=0,missing_count=0,rejected_count=0,failed_count=0
```

Replay:

```text
SCHEDULER_CONTROL_RESULT,action=resume,experiment_id=442,request_id=<original>,result=already_resumed,source_pause_request_id=3,global_state=paused,signal_result=not_attempted,status=completed,target_count=1,successful_count=0,already_satisfied_count=1,missing_count=0,rejected_count=0,failed_count=0
```

Other explicit results include `lifecycle_resumed`, `not_globally_suspended`, `missing_worker`, `identity_validation_failed`, `stale_control_evidence`, `invalid_lifecycle`, and `signal_failed`.

Dry-run performs read-only validation and reports the intended signal. Missing `--yes` retains the existing confirmation behavior.

## 5. Concurrency and replay assessment

[GlobalExperimentControl.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Sources/GlobalExperimentControl.cpp:2184>) now uses the existing advisory coordination lock and active-request gate.

The flow is:

1. Lock, verify authoritative state, freeze the plan, and create the audit request.
2. Commit before signaling.
3. Validate and signal outside the transaction.
4. Reacquire authority, verify the pause generation, and persist the result.

Concurrent resume-all, pause-all, and cancel-all attempts are rejected while selective resume owns the active request. Expired applications retain the existing lease/retry model.

The independent review added a final exact identity predicate: a phase/PID/PGID/start/executable/command change during the signal window cannot mark a replacement row released. Such drift is audited as partial `stale_control_evidence`.

Historical pause requests are ignored; only `current_pause_request_id` is applicable. A later pause-all creates a new generation and can stop a previously released worker again.

## 6. Process identity and signaling safety

Identity validation was not weakened. Selective resume still validates:

- PID and PGID.
- Executable identity.
- Process-start identity.
- Full command identity.
- Experiment ID and phase.
- Process-group safety.

Already-running workers produce an idempotent already-satisfied result without another signal. Missing, reused, mismatched, or unsafe identities do not receive `SIGCONT`.

The operation works without a scheduler process; `scheduler_running_observed` remains audit evidence rather than an execution dependency.

## 7. Scheduler restart/reconciliation

The global scheduler gate remains paused after selective release, so restart and reconciliation cannot launch queued training, inference, analysis, continuation, or replacement work.

`resume-all` now signals only workers still represented as paused under the current generation, clears selective-release associations during reconciliation, and transitions global state to `running`.

Cancellation handles selectively released, still-stopped, and lifecycle-paused workers through the existing authoritative workflows.

A selective resume releases the experiment’s primary worker only. A separate checkpoint-inference child remains globally suspended until resume-all or cancellation; new children cannot dispatch while the gate remains paused. This is documented in [GlobalExperimentControls.rst](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/docs/GlobalExperimentControls.rst:91>).

## 8. Tests added or changed

Coverage in [GlobalExperimentControlProcessTests.cpp](</Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Tests/GlobalExperimentControlProcessTests.cpp:1750>) and the integration/migration suites now proves:

- One-of-many and subset selective resume.
- Global state remains paused with no new dispatch.
- Lifecycle-paused compatibility.
- Ordinary-running rejection.
- Replay without duplicate signaling.
- Missing and identity-invalid workers.
- PID/PGID/executable/command/start-identity validation.
- Resume-all after selective releases.
- A second pause generation.
- Resume/pause/cancel race rejection.
- Post-signal identity drift.
- Restart scheduling gate behavior.
- Cancellation of released and stopped workers.
- Dry-run, confirmation, output, and accounting.
- Migration backfill, constraints, privileges, and idempotence.

Tests used isolated databases and disposable process groups. Cleanup verification found no remaining test workers. Existing live scheduler/workers, including 442–448, were only observed read-only and were not signaled or mutated.

## 9. Commands and results

Passed:

```bash
clang++ -std=c++20 -Wall -Wextra -Wpedantic -Werror \
  -Wno-deprecated-declarations -Wno-c++23-attribute-extensions \
  Tests/GlobalExperimentControlTests.cpp \
  Sources/GlobalExperimentControl.cpp \
  -IHeaders -ISources $(pkg-config --cflags --libs libpqxx) \
  -o /tmp/GlobalExperimentControlTests_selective_final

/tmp/GlobalExperimentControlTests_selective_final
```

```text
GlobalExperimentControlTests passed
```

```bash
clang++ -std=c++20 -Wall -Wextra -Wpedantic -Werror \
  -Wno-deprecated-declarations -Wno-c++23-attribute-extensions \
  Tests/GlobalExperimentControlProcessTests.cpp \
  Sources/GlobalExperimentControl.cpp \
  -IHeaders -ISources $(pkg-config --cflags --libs libpqxx) \
  -o /tmp/GlobalExperimentControlProcessTests_selective
```

```bash
bash Tests/GlobalExperimentControlIntegrationTests.sh \
  DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release \
  /tmp/GlobalExperimentControlProcessTests_selective
```

```text
GlobalExperimentControlCrashWindowTests passed
GlobalExperimentControlIntegrationTests passed
```

Also passed:

```bash
bash -n Tests/GlobalExperimentControlIntegrationTests.sh
git diff --check
git diff --cached --check
```

The prescribed build command was attempted and failed because the project contains an absolute `USER_HEADER_SEARCH_PATHS` pointing at the main checkout, producing conflicting headers in this worktree:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor build
```

The equivalent worktree-corrected Release build succeeded:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor \
  "USER_HEADER_SEARCH_PATHS='/Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/Headers'" \
  "LIBRARY_SEARCH_PATHS='/Volumes/Developer SSD/ExpertAdvisor-worktrees/global-pause-selective-resume/DerivedData/ExpertAdvisor/Build/Products/Release' '/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release' /opt/homebrew/opt/libpqxx@7.10.1/lib /opt/homebrew/opt/libpq/lib /opt/homebrew/lib /opt/homebrew/lib/postgresql@17 /opt/homebrew/opt/libomp/lib" \
  build
```

```text
** BUILD SUCCEEDED **
```

No clean was run.

## 10. Remaining risks or limitations

- Migration 047 must be applied before running the updated binary.
- The project still emits repository-wide libpqxx `exec_params` deprecation warnings. The focused changed source passed strict compilation with deprecation warnings isolated.
- The project’s absolute header-search path remains a pre-existing worktree build issue.
- No destructive test was run against production experiments, by design.

## 11. Git status and files changed

Current branch: `fix/global-pause-selective-resume`

```text
 M Sources/ExperimentScheduler.cpp
 M Sources/GlobalExperimentControl.cpp
 M Sources/GlobalExperimentControl.hpp
 M Tests/GlobalExperimentControlIntegrationTests.sh
 M Tests/GlobalExperimentControlMigrationTests.sql
 M Tests/GlobalExperimentControlProcessTests.cpp
 M docs/GlobalExperimentControls.rst
?? Database/migrations/047_global_pause_selective_resume.sql
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

The untracked migration adds another 134 lines. No files are staged and no commit was created.

## 12. Readiness recommendation

Ready for review and controlled deployment after applying migration 047. The selective-resume defect is corrected without weakening identity validation, resuming unrelated workers, or opening scheduler dispatch.