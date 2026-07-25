---
title: "Phase 5/6 Checkpoint Worker Classification Independent Verification"
document_type: "architecture review"
status: "final"
generated_from: "Phase5_6_Checkpoint_Worker_Classification_Independent_Verification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Phase 5/6 Checkpoint Worker Classification Independent Verification

## 1. Executive assessment

The checkpoint worker-classification fix is correct, fail-closed, regression-safe, and ready to commit on Campaign Operations Phase 3 commit `eb11084`. No production lifecycle behavior, schema, or scheduling policy was changed.

## 2. Root-cause verification

- Ordinary `--scheduler-experiment-id` workers remain authorized by active `experiment` rows.
- Checkpoint-tagged workers are authorized independently by active `experiment_checkpoint_eval` rows.
- Parent experiment completion is not consulted when authorizing an active checkpoint worker.
- Checkpoint-tagged candidates cannot fall through to ordinary experiment matching.
- Authority is loaded with one bounded union query; there is no N+1 lookup.

## 3. Ownership and identity verification

Classification remains fail-closed for:

- Unknown, missing, terminal, failed, wrong-phase, or wrong-PID checkpoint identities.
- Missing or mismatched persisted command identity.
- Executable, process-group, and process-start mismatches.
- PID reuse, inspection failure, permission failure, and unsafe process groups.
- Scheduler commands masquerading as workers.
- Database lookup or hydration exceptions, which abort status generation rather than produce managed classifications.

`worker_control_state` remains orthogonal to ownership: schema-valid `running` and `paused` states retain active ownership without changing pause/resume semantics.

## 4. Accounting and infer capacity

Each detected worker produces exactly one classification and contributes to exactly one managed or unmanaged aggregate.

- Checkpoint inference contributes once to managed inference count, CPU, RSS, and memory.
- Valid checkpoint workers are excluded from unmanaged counts.
- Ordinary train, infer, and analyze classifications remain intact.
- Both ordinary inference scheduling and checkpoint inference scheduling subtract active ordinary plus checkpoint inference from `max-infer-procs`, clamped at zero.

## 5. Status output

- Existing `SCHEDULER_STATUS` and `SCHEDULER_STATUS_RESOURCE` field names are unchanged.
- Their managed/unmanaged values are corrected using authoritative identity validation.
- Valid checkpoint workers no longer emit `SCHEDULER_STATUS_UNMANAGED_WORKER`.
- `SCHEDULER_STATUS_CHECKPOINT_JOB` is additive and keyed by authoritative `checkpoint_eval_id`.
- Human output presents checkpoint inference separately using checkpoint-evaluation identity.
- Ordinary inference log/model fallback explicitly excludes checkpoint-tagged processes.

## 6. Lifecycle and Phase 3 regression review

Classification performs no lifecycle write or signal. Pause, resume, cancellation, continuation, checkpoint policy, Campaign Operations, and experiment transitions are unchanged.

The branch is directly based on committed Campaign Operations Phase 3. No conflict markers, reverted Campaign Operations behavior, or accidental conflict-resolution changes were found.

A production scheduler and seven training workers were observed. Production `--scheduler-status` was not invoked, and no production process was signalled or otherwise disturbed.

## 7. Tests

Passed:

- `GlobalExperimentControlTests` — passed before and after corrections.
- `GlobalExperimentControlProcessTests` — passed.
- `SchedulerChildStatusTests` — passed.
- `GlobalExperimentControlIntegrationTests.sh` — passed, including `GlobalExperimentControlCrashWindowTests`.
- `git diff HEAD --check` — passed.

Representative commands:

```bash
clang++ -std=c++20 -O0 -g -Wall -Wextra -Wpedantic \
  Tests/GlobalExperimentControlTests.cpp \
  Sources/GlobalExperimentControl.cpp $(pkg-config --cflags --libs libpqxx) \
  -o /tmp/.../GlobalExperimentControlTests

clang++ -std=c++20 -O0 -g -Wall -Wextra -Wpedantic \
  Tests/GlobalExperimentControlProcessTests.cpp \
  Sources/GlobalExperimentControl.cpp $(pkg-config --cflags --libs libpqxx) \
  -o /tmp/.../GlobalExperimentControlProcessTests

clang++ -std=c++20 -O0 -g -Wall -Wextra -Wpedantic \
  Tests/SchedulerChildStatusTests.cpp \
  -o /tmp/.../SchedulerChildStatusTests

Tests/GlobalExperimentControlIntegrationTests.sh \
  DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release \
  /tmp/.../GlobalExperimentControlProcessTests
```

## 8. Release build

The prescribed worktree build succeeded:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor \
  'USER_HEADER_SEARCH_PATHS=$(SRCROOT)/Headers' \
  build
```

A fresh temporary DerivedData build compiled all current sources. Its first link lacked the separately prebuilt `libMetalBuffer`; adding the existing worktree product directory to `LIBRARY_SEARCH_PATHS` produced `BUILD SUCCEEDED`.

No diagnostic originated from the new classification lines. Existing unrelated warnings remain: libpqxx `exec_params` deprecations, legacy LSTM format/unused warnings, and a missing local `LLVM22.xctoolchain/Info.plist` warning.

## 9. Corrections made

- Added regression coverage proving checkpoint-tagged candidates cannot fall through to a same-PID ordinary inference row.
- Added ordinary inference and analysis classification/summary coverage.
- Corrected scheduler architecture metadata from version `0.2.0`/revision `0.1.1` to chronological version `0.2.1` dated 2026-07-25.

No production code correction was required.

## 10. Remaining risks

- The Xcode project does not build `libMetalBuffer` as an explicit target dependency in a completely empty DerivedData tree.
- Existing unrelated compiler and toolchain warnings remain.
- The untracked `Phase5_6_Checkpoint_Worker_Classification_Fix_Output.md` is a stale prior report and should not be included in the fix commit.

## 11. Repository state

```text
M  Headers/ExperimentScheduler.hpp
M  Sources/ExperimentScheduler.cpp
M  Sources/GlobalExperimentControl.cpp
M  Sources/GlobalExperimentControl.hpp
M  Tests/GlobalExperimentControlTests.cpp
M  docs/architecture/Volume_XI_Scheduler.md
?? Phase5_6_Checkpoint_Worker_Classification_Fix_Output.md
```

```text
 Headers/ExperimentScheduler.hpp          |  12 ++
 Sources/ExperimentScheduler.cpp          | 313 ++++++++++++++++++++++++-------
 Sources/GlobalExperimentControl.cpp      | 125 ++++++++++++
 Sources/GlobalExperimentControl.hpp      |  53 ++++++
 Tests/GlobalExperimentControlTests.cpp   | 308 ++++++++++++++++++++++++++++++
 docs/architecture/Volume_XI_Scheduler.md |  16 +-
 6 files changed, 760 insertions(+), 67 deletions(-)
```

The six tracked files are staged. No commit was created.

## 12. Readiness

The tracked checkpoint-classification fix is ready to commit. Exclude the stale untracked report.

READY TO COMMIT CHECKPOINT WORKER CLASSIFICATION FIX