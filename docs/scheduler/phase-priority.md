# Scheduler phase priority

Migration `099_scheduler_phase_priority.sql` adds a singleton scheduler admission
policy. Its default is `concurrent`, preserving simultaneous phase scheduling.
The setting is independent of `experiment.scheduler_priority`, resume origin,
and the existing selection and preemption policies.

Apply migrations and build the updated scheduler once before using the controls:

```bash
./migrate_lstm_db.sh
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
```

Start a scheduler with all desired capacity classes enabled. For example, add
these options to the usual launch command:

```bash
--phase-priority=train:infer:analyze --max-train-procs=2 --max-infer-procs=1 --max-analyze-procs=1
```

`--phase-priority` persists the startup policy after scheduler authority is
acquired. Omitting the option reuses the stored policy. A dry-run startup uses
its specified policy locally without changing the stored setting.

Change or inspect the policy while that scheduler is running:

```bash
"$BIN" --set-phase-priority=infer:analyze:train
"$BIN" --show-phase-priority
"$BIN" --set-phase-priority=concurrent
```

All six permutations of `train`, `infer`, and `analyze` are accepted. Each phase
must appear exactly once. Duplicate, omitted, additional, and unknown phases
are rejected. Controls also work with `lstm-scheduler`; they exit without
acquiring scheduler authority, reconciling workers, or starting jobs.

The daemon reloads the policy once per normal poll, so a committed change takes
effect on its next poll. `--set-phase-priority=ORDER --dry-run` prints the proposed
setting without persisting it. Worker limits remain startup settings; an already
running scheduler launched with `--max-infer-procs=0` still cannot admit inference.

## Admission behavior

For `train:infer:analyze`, training fills its available slots first. Pending
inference and analysis wait until eligible training is exhausted and active
training attempts have drained. Final and checkpoint inference then share the
inference limit; analysis follows after inference has drained. Final jobs retain
precedence within a phase. In ordered mode, checkpoint jobs can use spare slots
after final dispatch even if blocked final rows remain pending.

Only enabled phases with eligible pending work or active workers demand
admission. Missing-model and incompatible/unavailable semantic-worker candidates
do not hold up later phases. Paused, failed, cancelled, and completed final
experiments are not pending demand. Checkpoint jobs attached to a paused parent
are excluded in ordered mode, and checkpoint analysis requires its completed
inference evidence. Existing dispatch still revalidates claims and eligibility.

Authoritative worker-attempt capacity, rather than only experiment status counts,
provides active-worker evidence. It includes checkpoint workers, reservations,
and identity-ambiguous processes. Managed stopped workers do not consume active
capacity; resumable pending jobs retain their existing ordering.

A priority change, or newly arriving earlier-phase work, changes the desired
phase on the next poll. If workers from another phase remain active, no phase
admits new work until those workers drain. Workers are not stopped or cancelled
by this policy. Workers already active in the selected phase may continue to
fill its slots and use existing same-phase job-priority preemption. Workers in
a phase with a zero limit still block conflicting new admissions until drained.

Global pause/cancellation workflows retain precedence over ordinary scheduling.
Their existing cancellation checkpoint training/inference path bypasses the
normal phase policy. Continuous demand in an earlier phase can delay later
phases indefinitely; change the order when a different queue needs precedence.

Policy and selection changes emit `SCHEDULER_PHASE_ADMISSION` with the stored
order, selected phase, and `draining` flag. Identical lines are suppressed across
polls unless verbose logging is enabled.

## Verification

Portable policy tests exercise all permutations, invalid values, idle and disabled
phases, active-worker draining, live switches, and existing job-priority preemption.
Cycle tests verify that both final and checkpoint dispatch obey admission gates,
that concurrent call order remains intact, and that cancellation retains its path.
Configuration tests cover startup and standalone live controls. PostgreSQL tests
use a newly created disposable database to check missing-migration behavior,
persistence across connections, revision idempotence, rollback, SQL constraints,
runtime grants, and replay preservation.

```bash
bash Tests/SchedulerPhasePriorityTests.sh
bash Tests/SchedulerCycleServiceTests.sh
bash Tests/SchedulerDaemonConfigurationTests.sh
bash Tests/FinalExperimentDispatchServiceTests.sh
bash Tests/SchedulerPhasePriorityPostgresTests.sh
```

The PostgreSQL test needs an existing `pqxx` runtime role and permission to create
and drop its own test database; it does not read or modify production experiments.
