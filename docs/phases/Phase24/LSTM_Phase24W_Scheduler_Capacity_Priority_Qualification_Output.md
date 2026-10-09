# Phase 24W — Scheduler capacity and priority qualification

## Assessment

**PASS for bounded synthetic scheduler capacity and phase-priority
qualification; ready for Phase 24W closure within that scope.** Native18 completed
TRAIN=2, TRAIN/INFER phase exclusion and exact-attempt readmission,
ANALYZE=18, nineteenth-candidate rejection, high-priority ANALYZE displacement,
second-cycle capacity enforcement, exact-identity cleanup, and orphan
reconciliation. Its private PostgreSQL server was stopped and cleanup passed.

The supporting four rollback-compensation and 19 displacement/recovery behavior
checks also passed, with private cleanup passing for every run.

This qualifies scheduler process-control behavior with inert workers. It does
not establish resource safety for 18 real ANALYZE workloads or authorize a
production configuration change. No production worktree, database, worker,
scheduler configuration, or qualified worker artifact was changed.

## Initial restricted-run assessment (historical)

**NO-GO for this qualification.** The required native private PostgreSQL
qualification could not reach scheduler startup in this runner. PostgreSQL 17
`initdb` fails while creating its System V shared-memory segment:
`shmget(...): Operation not permitted`. The failure is environmental and
occurred before any scheduler, worker, database schema, or fixture experiment
was created. No production system was accessed or changed. No scheduler source
change was made.

This result does not establish a scheduler defect, and it does not qualify
TRAIN=2, INFER=1, or ANALYZE=18 for production. It also does not establish
resource safety for 18 real ANALYZE workers.

## Baseline

Repository: `/Volumes/Developer SSD/ExpertAdvisor-Rollover`  
Branch: `dedicated-train-layout-rollover-squashed-v1`  
HEAD: `72f0160dcc94e4c791288ff5e44fd9316542e71d`  

The worktree was clean at the required baseline. The Phase24T/24U scheduler
implementation and reports were preserved. Qualified TRAIN and INFER artifacts,
the MetaNN link, and production were not touched.

## Intended checks

The new inert-worker harness, `Tests/SchedulerCapacityPriorityQualification.py`,
reuses the Phase24T private SCRAM harness and requests:

* two independently observed TRAIN workers at cap 2;
* INFER remaining pending while TRAIN owns the host, then exact-attempt
  recovery after TRAIN drains;
* 18 independently observed ANALYZE workers at cap 18 and a nineteenth
  candidate remaining pending;
* high-priority ANALYZE draining lower-priority TRAIN and INFER workers under
  the existing `train:infer:analyze` policy;
* exact PID/start-identity/attempt cleanup through the existing harness.

These cases were not marked PASS because the private server could not be
initialized. The harness uses synthetic process-control workers only; no Metal,
model, TRAIN, INFER, or ANALYZE workload was launched.

## Commands and results

The targeted Apple-Clang scheduler build completed using the stable Phase24U
build recipe:

```text
python3 DerivedData/ExpertAdvisor/Phase24U/build.py       PASS
```

It rebuilt the scheduler objects and private link target with Apple Clang. The
two existing `semanticWorkerRole` missing-field-initializer warnings remain;
there were no build errors. No qualified worker binary was rebuilt or replaced.

Pure phase-policy regression passed:

```text
bash Tests/SchedulerPhasePriorityTests.sh                 PASS
```

The new harness and existing harness passed Python syntax compilation:

```text
python3 -m py_compile Tests/SchedulerCapacityPriorityQualification.py \
  Tests/SchedulerDisplacementRecoveryTests.py              PASS
```

The required native commands were attempted with bounded execution:

```text
python3 -B Tests/SchedulerCapacityPriorityQualification.py \
  DerivedData/ExpertAdvisor/Phase24T/Qualification24W        BLOCKED at initdb
python3 -B Tests/SchedulerRollbackCompensationTests.py       BLOCKED at initdb
python3 -B Tests/SchedulerDisplacementRecoveryTests.py \
  --phase24t DerivedData/ExpertAdvisor/Phase24T/Qualification24W-displacement
                                                               BLOCKED at initdb
```

Each failure recorded the same `shmget ... Operation not permitted` diagnostic.
The rollback harness's `finally` cleanup receipt is PASS; failed `initdb`
removed its incomplete data directory. No private PostgreSQL service or worker
process remained. The capacity harness failed before `q.started` became true,
so it created no server or worker. The qualification evidence directories are
ignored DerivedData artifacts.

## Findings and limitations

The existing pure policy tests continue to prove the intended global priority
rules, including phase selection, draining, and high-priority ANALYZE winning
against lower-priority work. They do not prove native capacity, process
ownership, signal safety, or cleanup under the requested caps. The requested
capacity and safe-displacement observations therefore remain **unverified**.

No reproducible scheduler defect was found because no native cycle ran. The
environment must provide a disposable PostgreSQL instance whose initialization
can create the required shared-memory resources (or an approved equivalent
private runner) before this phase can be completed. Do not infer production
readiness from the pure policy result or from the successful scheduler build.

`git diff --check` passed. No production repository, database, process,
configuration, registry, qualified binary, or shared MetaNN source was
modified. No commit, merge, push, publication, or deployment was performed.

## Native12 follow-up — scheduler-managed ANALYZE

The harness now contains a separate
`test_scheduler_managed_analyze_capacity()` method. It queues 19 model-backed
pending ANALYZE experiments, runs the production scheduler implementation with
TRAIN=2, INFER=1, ANALYZE=18 and `train:infer:analyze`, and is prepared to
verify exact PID/start-identity/`--analyze-experiment=<id>` bindings, the
nineteenth pending row, a second-cycle cap, identity-verified SIGTERM cleanup,
and orphan reconciliation. The synthetic ANALYZE executable accepts both
`--analyze-experiment` and the scheduler-attempt option. Scheduler-launched
workers are admitted to the private process-observation filter only by their
exact private helper path and argument marker. Cleanup is performed before
`results.json` can report success.

Native12 was attempted with:

```text
python3 -B Tests/SchedulerCapacityPriorityQualification.py \
  DerivedData/ExpertAdvisor/Phase24T/Qualification24W-Native12
```

It failed before `initdb` completed, with PostgreSQL 17 reporting
`shmget(...): Operation not permitted`. Therefore no scheduler-managed worker
was launched, observed, or terminated, and no Native12 capacity invariant is
claimed. The retained evidence is
`DerivedData/ExpertAdvisor/Phase24T/Qualification24W-Native12/001-initdb.log`,
`commands.jsonl`, and `results.json`; the latter records
`qualification=BLOCKED` and `cleanup=PASS`. A second attempt was not made
against the same evidence directory.

The relevant bounded checks passed: Python syntax validation,
`SchedulerPhasePriorityTests.sh`, and `SchedulerAnalyzeWorkerRoutingTests.sh`.
The private Apple-Clang build remained successful with the pre-existing two
warnings. The original Native11 assertions and Phase24T/24U regression files
were not modified. Native12 remains **NO-GO / unqualified** until a private
PostgreSQL runner with the required shared-memory capability is available.

## Native13 — complete synthetic-worker cleanup

The Native12 operator run exposed one cleanup defect: the high-priority
ANALYZE worker admitted by `test_priority()` was not inserted into the
qualification's scheduler-worker tracking map. The operator verified PID
43386 / experiment 980302 / attempt 1 and terminated it manually. This was a
qualification-harness defect, not a scheduler-capacity finding.

Native13 corrected the harness without changing scheduler source, schema,
registry, or any existing Native11 assertion. `cycle_w()` now discovers every
active scheduler-owned synthetic ANALYZE attempt after each scheduler cycle,
including the high-priority worker from `test_priority()`. Discovery requires
the persisted PID/start identity and the exact private helper path plus
`--analyze-experiment=<id>` marker. Cleanup sends SIGTERM only after the
identity matches, waits with a bounded timeout, scans for any remaining exact
synthetic helper processes, and records cleanup failure if one survives.
Private PostgreSQL shutdown remains after worker cleanup. The harness's
`finally` path performs the same identity-gated cleanup when assertions fail,
and `results.json` is written only after cleanup completes.

Validation completed:

```text
python3 -m py_compile Tests/SchedulerCapacityPriorityQualification.py  PASS
bash Tests/SchedulerPhasePriorityTests.sh                              PASS
bash Tests/SchedulerAnalyzeWorkerRoutingTests.sh                        PASS
git diff --check                                                        PASS
```

Native13 execution was not repeated in this restricted runner because the
Native12 private PostgreSQL initialization blocker remains: PostgreSQL cannot
create its required System V shared-memory segment (`shmget: Operation not
permitted`). The corrected harness is therefore source- and syntax-validated,
but its end-to-end cleanup assertion remains to be run by the operator in a
private runner with PostgreSQL shared-memory capability. No production process
was signaled, and no scheduler source was changed.

## Native14 — identity stabilization correction

Native13's PID 46937 failure was investigated against the actual identity
implementations. `Qualification.inspect(pid)` prints five fields in this
order: PID, process-group ID, process-start identity, canonical executable,
and command line. Its process-start field is produced by the same
`proc_pidinfo(PROC_PIDTBSDINFO)` implementation as the scheduler's
`ReadProcessStartIdentity`, formatted as `<seconds>:<microseconds>`. The
persisted `worker_process_start_identity` therefore uses the same
representation; no normalization or production identity change was needed.

The harness defect was premature observation: it performed one inspection
while the scheduler-owned child was still in the bounded spawn/exec-gate
transition (`lifecycle_state='spawned'`). A transient empty or incomplete
process observation was treated as permanent failure even though the worker
then logged `SYNTHETIC_ANALYZE_READY`. This explains the Native13 failure
without weakening PID-reuse protection.

Native14 changes only the qualification harness. It records each database
discovery before validation, persists discovery/failure diagnostics, and polls
for a bounded 25 × 20 ms stabilization window. It requires the exact PID,
persisted start identity, private helper path, experiment option, and
scheduler-attempt option. A different start identity, contradictory command,
or timeout remains a hard failure. Cleanup can retry only previously recorded
database discoveries; it never infers ownership from an arbitrary PID scan.
Before signaling it rechecks the complete identity and uses `killpg` only when
the observed process group equals the verified PID; otherwise it uses the
verified individual PID. An unidentified survivor prevents PostgreSQL shutdown
and prevents `cleanup=PASS`.

Native14 validation passed the requested bounded checks:

```text
python3 -m py_compile Tests/SchedulerCapacityPriorityQualification.py  PASS
bash Tests/SchedulerPhasePriorityTests.sh                              PASS
bash Tests/SchedulerAnalyzeWorkerRoutingTests.sh                        PASS
git diff --check                                                        PASS
```

Native14 itself was not claimed PASS. The private PostgreSQL shared-memory
initialization restriction from Native12 remains, so the full native command
must be run by the operator:

```text
python3 -B Tests/SchedulerCapacityPriorityQualification.py \
  DerivedData/ExpertAdvisor/Phase24T/Qualification24W-Native14
```

## Native15 — process-inspection diagnostics

Native14's stabilization still produced an insufficiently explained identity
failure for experiment 980302 / PID 49609, even though a later direct helper
invocation returned the persisted start identity `1791496203:738121`. Native15
adds `scheduler-analyze-inspection.jsonl` records for every failed inspection
attempt. Each record includes experiment and attempt IDs, PID, expected start
and process-group identities, observed identity when available, subprocess exit
status, stdout, stderr, elapsed time, lifecycle state, and a classification of
subprocess failure, missing observation, start-identity mismatch,
command-line mismatch, or process-group mismatch.

The correction does not increase the 25 × 20 ms stabilization budget, alter
production identity semantics, or weaken cleanup fencing. Cleanup retries only
database-recorded discoveries, verifies the same complete identity before
SIGTERM, and retains diagnostics when verification fails.

Requested validation passed:

```text
python3 -m py_compile Tests/SchedulerCapacityPriorityQualification.py  PASS
bash Tests/SchedulerPhasePriorityTests.sh                              PASS
bash Tests/SchedulerAnalyzeWorkerRoutingTests.sh                        PASS
git diff --check                                                        PASS
```

Native15 remains unqualified; the full private PostgreSQL run has not been
claimed PASS. Operator command:

```text
python3 -B Tests/SchedulerCapacityPriorityQualification.py \
  DerivedData/ExpertAdvisor/Phase24T/Qualification24W-Native15
```

## Native16 — process-observation root-cause diagnostics

Native15 evidence for experiment 980302 / PID 52265 showed 50 normal
inspection attempts returning exit 1 with empty stdout and stderr, followed by
a later successful direct inspection. Native16 adds a development-only
`--diagnose-managed-test-process=<pid>` mode to the synthetic test helper. It
traces the same native observation stages used by the observer: process
existence, first start-identity read, status/command read, process-group
observation, executable path and canonicalization, second start-identity read,
stability, and final validation. Each failure reports its stage and errno;
normal identity output is unchanged.

The qualification harness invokes this diagnostic only after a nonzero normal
inspection result and records the normal result plus diagnostic output in
`scheduler-analyze-inspection.jsonl`. The existing 25 × 20 ms budget and all
identity assertions are unchanged. Diagnostic output never authorizes a
signal. Cleanup continues to require the exact persisted PID/start/group/
command/attempt identity.

The focused diagnostic-record regression and native helper syntax check passed:

```text
python3 -m py_compile Tests/SchedulerCapacityPriorityQualification.py \
  Tests/SchedulerCapacityPriorityInspectionDiagnosticsTests.py              PASS
python3 Tests/SchedulerCapacityPriorityInspectionDiagnosticsTests.py        PASS
/usr/bin/clang++ -std=c++20 -fsyntax-only ... \
  Tests/GlobalExperimentControlProcessTests.cpp                              PASS
bash Tests/SchedulerPhasePriorityTests.sh                                     PASS
bash Tests/SchedulerAnalyzeWorkerRoutingTests.sh                              PASS
git diff --check                                                               PASS
```

Native16 has not been claimed PASS. No private PostgreSQL/native qualification
was run in this session; the prior sandbox shared-memory restriction remains
the blocker. Operator command:

```text
python3 -B Tests/SchedulerCapacityPriorityQualification.py \
  DerivedData/ExpertAdvisor/Phase24T/Qualification24W-Native16
```

## Native17 — observation-filter ordering fixed and capacity qualified

The resumed worktree matched branch
`dedicated-train-layout-rollover-squashed-v1` and HEAD
`72f0160dcc94e4c791288ff5e44fd9316542e71d`. Existing diagnostic source changes,
the qualification harness, its regression, and all previous evidence were
preserved.

### Root cause

Native16-Local's PID 54839 still existed as the inert Python ANALYZE worker for
experiment 980302 / attempt 1. Direct inspection returned the persisted start
identity `1791496830:480537` and group 54839. The private PostgreSQL server on
127.0.0.1:55485 had PID 53107, the exact Native16-Local data directory, and
system identifier `7694420151418838877`.

The diagnostic helper was current: its successful build at 17:00:02 followed
the source update at 16:59:14, it contained the diagnostic option and stage
strings, and the retained inspection records already reported
`stage=process_status_read,errno=10,detail=ps_status_failed_or_unparseable`.

The harness installed its scheduler-owned ANALYZE observation filter only
inside `test_scheduler_managed_analyze_capacity()`. `test_priority()` ran
earlier and launched a worker absent from the base harness's registered-PID
allowlist. Consequently the private `ps` wrapper rejected every status read;
polling could never make that worker visible. Reproduction against the same
PID returned exit 1 with the old filtered environment and exit 0 with direct
private-PID inspection. This was a harness ordering defect, not a spawn
stabilization defect or a stale helper.

### Change and focused validation

`registry()` now installs the existing private ANALYZE observation filter
immediately after creating the synthetic helper, before any scheduler cycle.
The later capacity-only installation was removed. The regression verifies
that an unregistered scheduler-owned priority worker is visible at registry
setup and that an unrelated worker remains excluded. Production scheduler
source and identity semantics, the polling budget, and signal fencing are
unchanged.

Focused commands passed:

```text
python3 -B Tests/SchedulerCapacityPriorityInspectionDiagnosticsTests.py
bash Tests/SchedulerPhasePriorityTests.sh
bash Tests/SchedulerAnalyzeWorkerRoutingTests.sh
git diff --check
```

Only the private C++ process-test helper was rebuilt, with the existing
Apple-Clang/libpqxx source and link recipe; there were no compiler diagnostics.
The exact command and output are retained in
`DerivedData/ExpertAdvisor/Phase24T/Qualification24W-Native16-Recovery/build-command.json`
and `build.log`. No Xcode clean, full application rebuild, or scheduler rebuild
was needed. The rebuilt helper and corrected filter also passed a native
regression against PID 54839, while the unrelated private PostgreSQL PID was
excluded from worker inspection.

### Qualification result

```text
python3 -B Tests/SchedulerCapacityPriorityQualification.py \
  DerivedData/ExpertAdvisor/Phase24T/Qualification24W-Native17    PASS
```

Native17's `results.json` records PASS for:

* two independently observed TRAIN workers at cap 2;
* INFER exclusion while TRAIN runs and admission after TRAIN drains;
* 18 registered ANALYZE attempts and nineteenth-candidate rejection;
* high-priority scheduler-launched ANALYZE displacement of lower TRAIN/INFER;
* 18 scheduler-launched ANALYZE workers with exact PID/start/group/command/attempt
  checks, nineteenth pending, and a second cycle remaining capped;
* identity-verified worker termination and reconciliation clearing active attempts;
* complete worker and private PostgreSQL cleanup.

The native scheduler-managed capacity evidence is retained in
`Qualification24W-Native17/native12-evidence.json`, with commands, scheduler
logs, discoveries, and signal receipts alongside it. Cleanup diagnostic records
report ESRCH for previously terminated discoveries; they are expected absent
process observations and do not indicate a qualification failure.

### Native16 cleanup and remaining scope

Before signaling PID 54839, the rebuilt helper's full current identity was
rechecked against the private persisted experiment/attempt/start/group binding.
Only that PID received SIGTERM; its exit was verified. Private scheduler status
was read using the destination-verified private database environment. `pg_ctl`
then stopped only the verified Native16-Local data directory. The receipt is
`Qualification24W-Native16-Recovery/receipt.json`. Original Native16 results
and diagnostics were preserved; its stopped private data directory was retained.

The Native17 capacity/priority qualification has no remaining failed assertion.
Real workload memory, Metal/model execution, and production ANALYZE=18 resource
safety remain outside this synthetic qualification. No commit, push, publication,
production access, or production configuration change was performed.

## Supporting native regressions

The previously blocked rollback and displacement commands now completed on
fresh, independent private SCRAM clusters, using the existing Phase24U scheduler
binary and newly built inert process fixtures:

```text
python3 -B Tests/SchedulerRollbackCompensationTests.py            PASS
python3 -B DerivedData/ExpertAdvisor/Phase24T/\
  Qualification24W-Native16-Recovery/run-displacement.py          PASS
```

Rollback evidence is in `Phase24U-Rollback-20261008193108/results.json`: ordinary
compensation, supported operator pause after abort, changed start identity, and
lost scheduler fence all passed. Cleanup passed.

Displacement evidence is in `Qualification24W-Native17-displacement/results.json`:
all 19 behavior checks passed, including same-attempt TRAIN/INFER recovery,
stale/conflicting identities, completed-result recovery, rollback compensation,
external over-cap correction, exec-gate races, independent zero caps, ambiguous
excess, in-flight exclusion, owner restart, and stale fencing. Cleanup passed.
The retained launcher explicitly selects the Phase24U CLI while preserving the
existing Phase24T test cases; it reads private scheduler status before tests.
Each subprocess and wait uses the existing bounded harness deadlines. Each
private helper build produced an empty compiler diagnostic log.

## Native18 — exact excluded-INFER attempt qualified; final closure

Review of Native17's assertions found that its TRAIN-drain scenario retired
the excluded INFER fixture and tested admission of a replacement. Native18
strengthens this case: it retains the excluded INFER worker, signals only the
two identity-verified TRAIN fixtures, waits for their exit, and lets the normal
scheduler cycle reconcile and resume the original INFER attempt. It asserts
unchanged full process identity, unchanged active attempt, exactly one persisted
attempt, TRAIN capacity zero, and INFER capacity one. No replacement worker is
created and no new fixture SQL transition bypasses scheduler reconciliation.

The focused diagnostic/filter, phase-priority, and ANALYZE routing regressions
were rerun and passed before the final native command:

```text
python3 -B Tests/SchedulerCapacityPriorityQualification.py \
  DerivedData/ExpertAdvisor/Phase24T/Qualification24W-Native18    PASS
```

Both `train-infer-exclusive.json` and `infer-after-train-drain.json` show
experiment 980112 with PID/group 88339, start identity `1791506050:281398`, and
attempt 1980112. Its state changes from pending/stopped to running without
changing that identity or attempt. Native18 also passed every Native17
capacity/priority assertion, including scheduler-managed ANALYZE=18, nineteenth
candidate rejection, second-cycle enforcement, identity-verified termination,
orphan reconciliation, and complete private cleanup. Its helper build emitted
no diagnostics. The existing scheduler and qualified worker binaries were not
rebuilt or replaced.

**Final result: no remaining qualification failures; Phase 24W is ready for
closure for the bounded synthetic scheduler capacity and phase-priority
contract.** Real ANALYZE workload resource safety remains unverified and no
production ANALYZE=18 configuration was enabled. Native16's stopped data
directory is deliberately retained for evidence; no running worker or private
server cleanup remains required.
