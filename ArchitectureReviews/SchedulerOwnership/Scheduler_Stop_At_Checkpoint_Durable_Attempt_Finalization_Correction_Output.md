# Stop-at-Checkpoint Durable-Attempt Finalization Correction

Date: 2026-07-30
Branch: `phase6`
Scope: implementation and independent verification only

## 1. Executive summary

The stop-at-checkpoint ownership defect is corrected. The production checkpoint
workflow now completes the exact active durable train attempt, records auditable
checkpoint evidence, clears only that attempt's lifecycle binding and process
mirrors, and moves the experiment to the established pending next phase or
cancellation destination in one transaction. Any one-row predicate failure
rolls the transaction back, so the formerly malformed state cannot commit.

The old attempt uses the existing terminal state `completed` and reconciliation
result `checkpoint_stop_completed`. It therefore stops consuming train
capacity exactly once. A delayed child reap recognizes that terminal reason,
updates only exact exit evidence on the old attempt, and changes no lifecycle
row. Replay recognizes the exact terminal attempt and checkpoint evidence
without repeating the transition, including after a replacement attempt has
claimed the next phase.

Scenario 37 now has deterministic executable production-path assertions for
terminalization, binding clearing, capacity release, next-phase claim,
delayed reap, restart before reap, replay, wrong-attempt rejection, and
cancellation compatibility. The independently re-audited matrix is now
**40 Full; 0 Partial; 0 Missing; 0 Manual-only**.

No staging, commit, backup, scheduler shutdown, migration, cutover, production
launch, production signal, or production database mutation was performed.

## 2. Independent defect confirmation

The independent review finding was confirmed against the repository. Before
this correction, `GlobalExperimentControl::RecordCheckpointStopReached` was
the unsafe transition: it changed an experiment from `running/train` to
`pending/infer`, `pending/analyze`, or the cancellation destination while
leaving the generation-52 active train attempt and
`active_scheduler_worker_attempt_id` intact.

The mechanically traced production path is:

1. Administrative stop/cancellation persists `stop_after_checkpoint_epoch`.
2. Training saves a periodic checkpoint.
3. Checkpoint inference is queued under the existing policy.
4. `LoadCheckpointStopConfig` locks the experiment and records
   `last_checkpoint_stop_decision_epoch`.
5. The trainer calls `RecordCheckpointStopReached` with its exact
   `--scheduler-worker-attempt-id`.
6. The global-control workflow verifies the checkpoint and exact train attempt.
7. The corrected transaction completes the attempt, clears the exact binding,
   and advances/cancels the lifecycle.
8. The trainer exits zero.
9. The owning scheduler either reaps that child and appends exact exit evidence,
   or a restart observes the old attempt as already terminal.
10. The normal reservation path can claim the pending infer/analyze phase
    because the binding is null and train capacity is no longer consumed.

The cancellation branch was traced through persisted administrative outcome
identity, cancellation checkpoint materialization, optional infer-before-cancel
policy, terminal experiment state, and reconciliation.

## 3. Pre-fix failure reproduction

A fresh disposable PostgreSQL database was cloned from the production schema,
migrations 051/052 were applied, and a production-faithful generation-52
`running/train` experiment plus exact active train attempt was created. Applying
the former lifecycle-only checkpoint transition produced:

```text
reaper_match=false
recovery_rows=0
next_phase_claim_rows=0
train_capacity=1
```

This independently reproduces the reported leak: neither reap nor recovery
could retire the old attempt, the next phase could not claim, and one train
slot remained consumed. The database was dropped.

The corrected production path was then executed twice in the process
integration. The former malformed postcondition is now impossible: each run
asserted old attempt `completed/checkpoint_stop_completed`, train capacity
zero, a cleared binding before claim, and successful infer replacement claim.

## 4. Exact unsafe transition

The exact unsafe function was
`EA::GlobalExperimentControl::RecordCheckpointStopReached` in
`Sources/GlobalExperimentControl.cpp`. Its old ordinary branch updated only the
experiment's checkpoint metadata, status, phase, and current operation. Its
cancellation branch likewise updated the lifecycle without completing and
unbinding the train attempt.

That caused all four legs of the deadlock:

- active attempt still counted as train capacity;
- reaper's exact active-lifecycle match failed after the phase changed;
- recovery did not classify a still-bound attempt as detached;
- next-phase claim required a null active-attempt binding.

## 5. Corrected ownership transition

`RecordCheckpointStopReached` now:

1. takes the canonical global coordination advisory lock;
2. locks the exact attempt before the experiment row;
3. verifies attempt ID, experiment, null checkpoint ID, worker kind
   `experiment`, phase/capacity `train`, active signalable state, complete
   immutable process identity, command identity, and the exact active binding;
4. verifies `running/train`, the exact decision epoch, stop target, checkpoint
   model ownership, and `train_config_meta` epoch;
5. completes the exact attempt with a diagnostic containing checkpoint epoch,
   model ID, next phase, and worker-attempt ID;
6. clears the exact binding and all worker PID/PGID/start/executable/command
   mirrors;
7. writes the existing next-phase or cancellation policy result and canonical
   operation;
8. verifies one returned attempt row and one returned lifecycle row;
9. commits only through the caller's transaction.

If either update affects anything other than exactly one row, an exception
causes the entire transaction to roll back. There is no best-effort repair
after an invalid transition.

## 6. Exact terminal-attempt semantics

The stopped train attempt becomes:

```text
lifecycle_state       = completed
reconciliation_result = checkpoint_stop_completed
```

`completed` is the semantically accurate existing state: the train phase
successfully reached and durably persisted its selected checkpoint. It is not
an abandoned or failed computation. No new lifecycle state was added.

The diagnostic is:

```text
checkpoint_stop;checkpoint_epoch=<epoch>;
checkpoint_model_id=<model>;next_phase=<infer|analyze|cancelled>;
worker_attempt_id=<exact-attempt>
```

The semantic contract is also documented in Volumes VII and XI.

## 7. Delayed-reaper semantics

`ObserveTerminalCheckpointStopAttempt` runs before the ordinary exact-active
reaper path. It resolves the original child attempt by exact attempt,
experiment, invocation, fence, kind, phase, capacity, PID, terminal state, and
reconciliation reason.

It may fill previously absent exit/signal evidence with `COALESCE`, requiring
any existing evidence to match. It updates no experiment or checkpoint
lifecycle row and emits `lifecycle_rows_affected=0`. It therefore cannot clear
or transition a replacement attempt and cannot release capacity a second time.

Duplicate observation with matching evidence is idempotent. Conflicting or
foreign evidence affects zero rows and fails closed.

## 8. Restart and replay semantics

The new terminal-attempt verification helper deliberately does not require a
current lifecycle binding. That narrow mode is valid only for an exact
terminal state and exact reconciliation reason.

Consequences proved by tests:

- restart after checkpoint commit sees the old train attempt as terminal;
- restart before old-child reap consumes no train capacity;
- before next claim the lifecycle remains claimable;
- after next claim only the new attempt is bound;
- the old attempt is not resurrected or classified ambiguous;
- replay after claim recognizes the original terminal evidence and creates no
  attempt or transition;
- checkpoint inference/final inference is not duplicated.

## 9. Next-phase claim semantics

The existing reservation workflow is unchanged. It still:

- revalidates scheduler invocation/fence;
- checks normal/cancellation scheduling policy;
- counts active states in the requested capacity class;
- requires `status='pending'`, exact expected phase, and null binding;
- creates one reserved attempt;
- compare-and-updates the lifecycle to `running` with that new exact binding.

The scenario-37 process test proves train capacity changes from one to zero,
then normal infer reservation creates a different attempt and binds only that
attempt. The old train attempt remains terminal.

## 10. Cancellation compatibility

The established cancellation policy is preserved:

- ordinary stop advances to `infer` when the final inference range exists,
  otherwise to `analyze`;
- after-next-checkpoint cancellation terminalizes the experiment at the
  checkpoint;
- infer-before-cancel uses the existing cancellation checkpoint inference
  materialization and outcome states;
- no-infer cancellation suppresses the matching pending checkpoint evaluation;
- checkpoint epoch/model and stopped-at-checkpoint metadata remain persisted;
- normal final infer/analyze reservation is not introduced into cancellation.

All cancellation variants now additionally complete and unbind the exact old
train attempt in the same transaction.

## 11. Exact SQL predicate table

| Mutation | Authoritative predicate and affected-row proof |
|---|---|
| Complete exact train attempt | `WHERE a.worker_attempt_id=$2 AND a.experiment_id=$3 AND a.checkpoint_eval_id IS NULL AND a.worker_kind='experiment' AND a.lifecycle_phase='train' AND a.capacity_class='train' AND a.scheduler_invocation_id IS NOT DISTINCT FROM $4 AND a.scheduler_fencing_token IS NOT DISTINCT FROM $5 AND a.worker_pid=$6 AND a.worker_process_group_id=$7 AND a.worker_process_start_identity=$8 AND a.canonical_executable_path=$9 AND a.command_line=$10 AND a.command_identity=$11 AND a.lifecycle_state IN ('spawned','running','observed')` plus `EXISTS` for exact active `running/train` binding/decision and exact model/matrix epoch; `RETURNING a.worker_attempt_id`; exactly one required. |
| Clear binding and advance | `WHERE experiment_id=$3 AND status='running' AND phase='train' AND active_scheduler_worker_attempt_id=$5 AND last_checkpoint_stop_decision_epoch=$1 AND stop_after_checkpoint_epoch IS NOT NULL AND stop_after_checkpoint_epoch<=$1 RETURNING experiment_id`; exactly one required. The same statement sets the binding and all process mirrors null and writes `pending/<next-phase>`. |
| Cancellation checkpoint destination | The preceding exact-attempt predicate, followed by `WHERE experiment_id=$3 AND cancellation_request_id=$4 AND status='running' AND phase='train' AND active_scheduler_worker_attempt_id=$5 AND last_checkpoint_stop_decision_epoch=$1 AND stop_after_checkpoint_epoch IS NOT NULL AND stop_after_checkpoint_epoch<=$1 RETURNING experiment_id`; exactly one required. |
| Terminal replay lookup | Exact attempt ID, experiment, null checkpoint ID, kind, train phase/capacity, `lifecycle_state='completed'`, and `reconciliation_result='checkpoint_stop_completed'`, with optional exact invocation/fence; one locked row required. Checkpoint epoch/model/attempt diagnostic evidence must match. No destructive lifecycle update occurs. |
| Delayed reap evidence | `WHERE a.worker_attempt_id=$4 AND a.scheduler_invocation_id=$1 AND a.scheduler_fencing_token=$5 AND a.experiment_id=$6 AND a.checkpoint_eval_id IS NULL AND a.worker_kind='experiment' AND a.lifecycle_phase='train' AND a.capacity_class='train' AND a.worker_pid=$7 AND a.lifecycle_state='completed' AND a.reconciliation_result='checkpoint_stop_completed' AND (a.exit_code IS NULL OR a.exit_code=$2) AND (a.signal_number IS NULL OR a.signal_number IS NOT DISTINCT FROM $3) RETURNING a.worker_attempt_id`; exactly one required; no lifecycle table is updated. |
| Wrong/stale attempt | Active helper first selects only `worker_attempt_id=$1`, then exact identity/lifecycle verification requires the lifecycle binding equal that same ID. Destructive updates repeat that ID. A stale/foreign ID therefore reaches no destructive row; executable assertions prove experiment and replacement unchanged. |
| Next-phase claim | `WHERE experiment_id=$5 AND status='pending' AND phase=$6 AND active_scheduler_worker_attempt_id IS NULL RETURNING experiment_id`; one row required after authority, policy, and capacity checks and creation of the new reserved attempt. |

## 12. Shared helper changes

`Sources/SchedulerOwnershipRepository.hpp` adds
`ExactTerminalAttemptSnapshot` and
`LockAndVerifyExactTerminalAttempt`. This is the narrow shared mode used by
checkpoint replay and delayed reap after the old binding has correctly been
cleared. It verifies immutable attempt identity, optional exact scheduler
invocation/fence, complete process identity, terminal state, and reconciliation
reason. It does not weaken the existing active-attempt helper.

## 13. Warning-as-error correction

The unused `DbTarget` parameter/capture in
`ExactAttemptSignalAuthorization` near the reported
`Sources/GlobalExperimentControl.cpp:1544` location was removed, and its call
sites were narrowed accordingly.

Fresh independent compiles of both corrected global-control and changed
scheduler sources passed with:

```text
-Wall -Wextra -Wpedantic -Werror
-Wno-deprecated-declarations
```

No new non-deprecation warning attributable to this correction remains. The
repository-wide libpqxx deprecation backlog was not changed.

## 14. Files changed by this correction

- `Sources/SchedulerOwnershipRepository.hpp`
- `Sources/GlobalExperimentControl.hpp`
- `Sources/GlobalExperimentControl.cpp`
- `LSTM/main.cpp`
- `Sources/ExperimentScheduler.cpp`
- `Tests/GlobalExperimentControlProcessTests.cpp`
- `Tests/SchedulerOwnershipProcessIntegrationTests.sh`
- `docs/architecture/Volume_VII_Experiment_Lifecycle.md`
- `docs/architecture/Volume_XI_Scheduler.md`
- this report

The worktree already contained extensive uncommitted generation-52 work,
reports, migrations, fixtures, documentation edits, and backup-file changes.
Those pre-existing changes were preserved. The plain Git diff/stat therefore
is not an increment-only view.

## 15. Tests added or changed

`TestProductionCheckpointStopOwnership` now creates production-faithful
generation-52 train attempts and asserts ordinary and cancellation atomic
finalization, mirror clearing, capacity release, replacement claim, replay
after replacement, restart via a new connection, and stale/foreign-attempt
non-mutation.

The scheduler ownership process integration adds deterministic OS-process
barriers:

- a real train worker stops itself with `SIGSTOP` only after the production
  checkpoint transaction commits;
- the scheduler normally claims a real infer replacement, which also stops at
  an explicit barrier;
- the old train child is continued and reaped after replacement claim;
- a second scenario stops/restarts the disposable scheduler between transition
  and reap.

The failpoint is gated by all of:

```text
EA_SCHEDULER_OWNERSHIP_TEST_ENABLE=1
EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY=
  checkpoint_stop_after_transition_before_exit
LSTM_DB_NAME prefix ea_scheduler_process_test_
exact scheduler experiment and worker-attempt IDs
```

It cannot activate against the production database.

## 16. Commands executed

Principal final-source commands:

```bash
ps ...; lsof ...                         # read-only production identity

Tests/SchedulerOwnershipProcessIntegrationTests.sh \
  "$PWD/DerivedData/CheckpointStopCorrection/Build/Products/Release/LSTM_Release" \
  "$PWD/DerivedData/CheckpointStopCorrection/Tests/GlobalExperimentControlProcessTests"

Tests/GlobalExperimentControlIntegrationTests.sh \
  "$PWD/DerivedData/CheckpointStopCorrection/Build/Products/Release/LSTM_Release" \
  "$PWD/DerivedData/CheckpointStopCorrection/Tests/GlobalExperimentControlProcessTests"

Tests/SchedulerOwnershipIntegrationTests.sh
Tests/SchedulerContinuationOwnershipIntegrationTests.sh \
  "$PWD/DerivedData/CheckpointStopCorrection/Build/Products/Release/LSTM_Release"
Tests/SchedulerCanonicalPathTests.sh

ASAN_OPTIONS=halt_on_error=1:detect_leaks=0 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  DerivedData/CheckpointStopCorrection/Tests/SchedulerOwnershipPolicyTests
# The same sanitizer settings were used for SchedulerChildStatusTests,
# ContinuationPolicyInheritanceTests, ExperimentCurrentOperationTests, and
# GlobalExperimentControlTests.

psql -X -v ON_ERROR_STOP=1 -d postgres \
  -f Tests/ExperimentCurrentOperationMigrationTests.sql
# The paired failure fixture was required to reject the unsupported value.

xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" -configuration Release \
  -derivedDataPath DerivedData/CheckpointStopCorrection build

clang++ ... -Wall -Wextra -Wpedantic -Werror \
  -Wno-deprecated-declarations \
  -c Sources/GlobalExperimentControl.cpp
xcrun clang++ ... -Wall -Wextra -Wpedantic -Werror \
  -Wno-deprecated-declarations \
  -c Sources/ExperimentScheduler.cpp

bash -n Tests/*.sh
git diff --check
rg ... waitpid/reaper/finalization/active-attempt/capacity/checkpoint/
       cancellation/lock-order paths
git status --short
git diff --stat
```

## 17. Exact results

Executed and passed on final source:

- focused scenario-37 scheduler process integration: passed twice, including
  the separate independent execution;
- global experiment control integration: implementation execution passed;
  independent immediate rerun passed;
- scheduler ownership migration/repository integration: passed twice;
- scheduler continuation ownership integration: passed twice;
- scheduler canonical path integration: passed;
- fresh ASan/UBSan scheduler ownership policy: passed twice;
- fresh ASan/UBSan scheduler child-status/reaper: passed twice;
- fresh ASan/UBSan continuation inheritance: passed;
- fresh ASan/UBSan current-operation: passed twice;
- fresh ASan/UBSan global-control pure tests: passed;
- global-control process/crash-window tests: passed;
- checkpoint inference, checkpoint analysis, interruption/recovery, and
  cancellation sections embedded in the process/global suites: passed;
- current-operation migration/idempotency fixture: passed;
- current-operation unsupported-value fixture: failed as required with the
  exact unsupported-row diagnostic;
- warning-as-error global-control compile: passed twice;
- warning-as-error scheduler compile: passed twice;
- isolated Release build: passed;
- all shell syntax: passed;
- `git diff --check`: passed twice.

Executed failures retained in the record:

- early development Release builds failed on missing signal declarations in
  the new deterministic barrier; the include was added and the final isolated
  build passed;
- early process-test development runs exposed relative binary paths, required
  production model/matrix fixture columns, and the valid restart transition
  from replacement `running` to `observed`; fixtures/assertions were corrected
  and final plus independent runs passed;
- the first independent global-control rerun passed all checkpoint-stop tests
  but hit the existing selective-resume lease-takeover assertion. The complete
  suite had already passed in implementation validation and passed on the
  immediate independent rerun. This is classified as a one-off unrelated
  timing flake and remains a residual test-harness risk.

Not executed against production:

- migrations, cutover, backup, scheduler stop/start, worker signals, and any
  production write command.

## 18. Scenario 37 traceability

| Acceptance item | Executable evidence | Result |
|---|---|---|
| Exact terminalization | real train attempt becomes `completed/checkpoint_stop_completed` with exact diagnostic | Full |
| Exact binding/mirror clear | pending transition asserts null binding and all PID/PGID/start/executable/command mirrors | Full |
| Capacity release | train count starts 1 and becomes 0 | Full |
| Correct lifecycle | pending infer (and cancellation fixtures for terminal destination), canonical operation asserted | Full |
| Next-phase claim | normal infer reservation creates a distinct attempt and exact binding | Full |
| Delayed reap | old child exits after infer claim; log says lifecycle rows 0; replacement unchanged | Full |
| Restart safety | scheduler restarted before old reap; old remains terminal, train count 0, replacement alone bound/observed | Full |
| Replay | replay before/after new connection and after replacement claim creates no transition or attempt | Full |
| Wrong attempt | stale/foreign and replacement IDs cause no destructive row; lifecycle/replacement unchanged | Full |
| Cancellation | after-next-checkpoint variants complete/unbind old train attempt and preserve infer-before-cancel policy | Full |

Scenario 37 verdict: **Full**.

## 19. Updated numbered 1–40 traceability

| # | Required scenario | Verdict | Reverified evidence |
|---:|---|---|---|
| 1 | Restart with live train | Full | live-process recovery/adoption |
| 2 | Restart with live infer | Full | mixed attempt/status/capacity |
| 3 | Restart with live analyze | Full | analysis recovery |
| 4 | Restart with live checkpoint-infer | Full | checkpoint binding/uniqueness |
| 5 | Duplicate scheduler while lease valid | Full | second scheduler rejected |
| 6 | Duplicate rejection nonmutating | Full | counts/state unchanged |
| 7 | Expired dead-owner takeover | Full | expiry plus owner-death |
| 8 | No takeover of valid owner | Full | fresh live owner retained |
| 9 | Scheduler crash with live workers | Full | restart observes durable workers |
| 10 | Graceful shutdown with live workers | Full | attempts preserved |
| 11 | PID reuse/start mismatch | Full | ambiguous, capacity retained |
| 12 | Executable mismatch | Full | canonical executable rejection |
| 13 | Command/experiment mismatch | Full | exact work identity rejection |
| 14 | Valid prior worker adoption | Full | exact identity observed |
| 15 | Invalid worker/orphan reconciliation | Full | bounded terminal/requeue |
| 16 | Ambiguous identity fails closed | Full | no signal or destructive clear |
| 17 | Train capacity | Full | attempt-state count assertions |
| 18 | Infer capacity | Full | mixed infer assertions |
| 19 | Analyze capacity | Full | exact claim/finalize |
| 20 | Checkpoint capacity | Full | infer/analyze attempt fixtures |
| 21 | Mixed accounting | Full | per-class status totals |
| 22 | Concurrent last slot | Full | reservation/capacity fencing |
| 23 | Duplicate experiment launch | Full | unique active attempt/claim |
| 24 | Duplicate checkpoint launch | Full | unique checkpoint attempt |
| 25 | Claim succeeds/no spawn | Full | reserved capacity/recovery |
| 26 | Spawn then parent crash | Full | gated boundary/recovery |
| 27 | Exit 127/126 | Full | launch failure/status decoding |
| 28 | Canonical executable/foreign CWD | Full | real canonical path |
| 29 | Basename invocation | Full | canonical path test |
| 30 | Symlink invocation | Full | direct/symlink equality |
| 31 | Lease loss during work | Full | fence loss stops claims |
| 32 | Foreign refresh/release | Full | invocation/fence rejection |
| 33 | Destructive exact predicate | Full | stale reaper cannot touch replacement |
| 34 | Replay idempotency | Full | migration/cutover/attempt replay |
| 35 | Interrupted recovery | Full | checkpoint-analysis boundaries |
| 36 | Global pause/resume/cancel | Full | process/global integration |
| 37 | Stop-at-checkpoint compatibility | **Full** | new complete production-path proof |
| 38 | Cancellation inference | Full | exact checkpoint cancellation/replay |
| 39 | Continuation/checkpoint inference | Full | authority/replay integration |
| 40 | Status ownership counts | Full | per-capacity status assertions |

## 20. Coverage totals

```text
Full:        40
Partial:      0
Missing:      0
Manual-only:  0
```

## 21. Independent reverification findings

The separate phase re-read the independent-review defect, re-traced the final
trainer/global-control/reaper/reservation/recovery source, inspected the exact
SQL above, and re-executed the focused process scenario and materially affected
suites.

Independent finding: the defect is closed. The phase transition and attempt
completion share one transaction; the attempt update is exact and capacity
releasing; the lifecycle update repeats the binding; terminal replay is exact
without requiring an obsolete binding; delayed reap is evidence-only; restart
does not resurrect or strand the attempt; no reviewed scenario regressed.

## 22. Production safety verification

Before testing, read-only inspection recorded:

```text
scheduler  PID 94420 PGID 94419 start Sun Jul 26 13:59:02 2026
workers    67171 67818 71349 38330 38400 39259 39540
```

For every process, PID, PGID, start identity, command, and mapped executable
were recorded. Final read-only `ps` and `lsof` inspection found the same
scheduler and all seven workers with identical PID/PGID/start/command and the
same mapped production executable:

```text
/Volumes/Developer SSD/ExpertAdvisor/
DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release
```

The production DerivedData path was not built or cleaned. Validation used
`DerivedData/CheckpointStopCorrection`. The final database audit found no
disposable scheduler/global-control/checkpoint-reproduction database. The final
process audit found no disposable scheduler or worker. Production scheduler and
workers were unchanged.

## 23. Deferred verification and residual risks

- Production migration/cutover/runtime behavior is intentionally unexecuted in
  this implementation-only run.
- The live production executable path is presently mapped by all recorded
  processes; because the earlier review reported that pathname missing at one
  point, those processes must still be treated as non-restartable until the
  later controlled cutover validates the new artifact.
- One independent global-control execution encountered a non-checkpoint
  selective-resume timing assertion; the preceding and immediate subsequent
  complete executions passed. This is a test-harness flake risk, not evidence
  of a checkpoint-stop ownership regression.
- The worktree contains extensive pre-existing, uncurated changes and modified
  backup artifacts. Commit curation must separate and verify intended files
  without assuming every dirty path belongs to this correction.

## 24. Commit-readiness verdict

**Ready to enter commit curation: Yes.**

The bounded implementation and independent verification are complete, but the
dirty worktree requires careful curation. Nothing was staged or committed here.

## 25. Production-cutover-readiness verdict

**Ready to resume the controlled production-cutover workflow after commit
curation: Yes. Ready for an immediate cutover from this run: No.**

Backup, scheduler shutdown, migration, generation-52 cutover, corrected launch,
and post-launch verification remain separate authorized steps.

## 26. Exact next action

Perform a dedicated commit-curation review of the complete generation-52
worktree, explicitly separating pre-existing backup/report/watch artifacts from
source, migration, test, and architecture files. Only after that curated commit
is independently verified should the production-cutover workflow resume.

## 27. `git status --short`

The final status includes extensive pre-existing generation-52 work plus this
correction:

```text
 M Database/README.md
 M Database/backups/LSTM_latest.dump
 M Database/backups/LSTM_latest.dump.json
 M Headers/ExperimentScheduler.hpp
 M LSTM/main.cpp
 M Sources/ExperimentScheduler.cpp
 M Sources/GlobalExperimentControl.cpp
 M Sources/GlobalExperimentControl.hpp
 M Tests/GlobalExperimentControlIntegrationTests.sh
 M Tests/GlobalExperimentControlProcessTests.cpp
 M docs/architecture/Volume_VII_Experiment_Lifecycle.md
 M docs/architecture/Volume_XI_Scheduler.md
 M docs/architecture/adr/ADR-0016-scheduler-atomic-claim-hardening.md
 M docs/architecture/adr/README.md
?? Database/migrations/051_scheduler_ownership_and_worker_attempts.sql
?? Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql
?? Headers/SchedulerExecutablePath.hpp
?? Headers/SchedulerOwnershipPolicy.hpp
?? Scheduler_Ownership_Coordinated_Remaining_Defects_Implementation_Output.md
?? Scheduler_Ownership_Coordinated_Remaining_Defects_Independent_CEE_Cutover_Output.md
?? Scheduler_Ownership_Independent_Review_Commit_Curation_Production_Cutover_Output.md
?? Scheduler_Restart_MultiInstance_WorkerOwnership_Architectural_Correction_Implementation_Output.md
?? Scheduler_Restart_MultiInstance_WorkerOwnership_Correctness_Implementation_Output.md
?? Scheduler_Restart_MultiInstance_WorkerOwnership_Focused_Independent_CEE_Review_Output.md
?? Scheduler_Stop_At_Checkpoint_Durable_Attempt_Finalization_Correction_Output.md
?? Sources/SchedulerOwnershipRepository.hpp
?? Tests/SchedulerCanonicalPathTests.sh
?? Tests/SchedulerContinuationOwnershipIntegrationTests.sh
?? Tests/SchedulerOwnershipIntegrationTests.sh
?? Tests/SchedulerOwnershipMigrationTests.sql
?? Tests/SchedulerOwnershipPolicyTests.cpp
?? Tests/SchedulerOwnershipProcessIntegrationTests.sh
?? Tests/fixtures/
?? docs/architecture/adr/ADR-0018-scheduler-generation-52-exact-attempt-authority.md
?? watch_20260729-142309
```

## 28. `git diff --stat`

Plain `git diff --stat` does not include untracked files such as the shared
repository helper, process integration, migrations, fixtures, ADR, or this
report:

```text
 Database/README.md                                 |   39 +
 Database/backups/LSTM_latest.dump                  |    4 +-
 Database/backups/LSTM_latest.dump.json             |   14 +-
 Headers/ExperimentScheduler.hpp                    |    9 +
 LSTM/main.cpp                                      |  152 +-
 Sources/ExperimentScheduler.cpp                    | 5776 +++++++++++++++++---
 Sources/GlobalExperimentControl.cpp                | 1111 +++-
 Sources/GlobalExperimentControl.hpp                |    7 +
 Tests/GlobalExperimentControlIntegrationTests.sh   |  151 +-
 Tests/GlobalExperimentControlProcessTests.cpp      |  872 ++-
 .../Volume_VII_Experiment_Lifecycle.md             |   41 +-
 docs/architecture/Volume_XI_Scheduler.md           |  193 +-
 .../ADR-0016-scheduler-atomic-claim-hardening.md   |   14 +
 docs/architecture/adr/README.md                    |    2 +
 14 files changed, 7311 insertions(+), 1074 deletions(-)
```

## Final questions

| Question | Answer |
|---|---|
| Was the capacity leak independently reproduced? | **Yes.** |
| Does the correction terminalize the exact old attempt? | **Yes, as `completed/checkpoint_stop_completed`.** |
| Is the exact binding cleared atomically? | **Yes.** |
| Is train capacity released exactly once? | **Yes.** |
| Can the next phase claim a new attempt? | **Yes.** |
| Can delayed old-child reap modify the replacement? | **No.** |
| Is restart before reap deterministic? | **Yes.** |
| Is replay idempotent? | **Yes.** |
| Does wrong-attempt mutation affect zero rows? | **Yes.** |
| Does cancellation-at-checkpoint remain correct? | **Yes.** |
| Is scenario 37 now Full? | **Yes.** |
| Are all 40 scenarios now Full? | **Yes.** |
| Does warning-as-error pass for the unused-capture defect? | **Yes.** |
| Were production scheduler/workers unchanged? | **Yes.** |
| Is the work ready for commit curation? | **Yes.** |
| Is it ready to resume production-cutover workflow? | **Yes, after commit curation; not for immediate cutover in this run.** |
