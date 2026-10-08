# Phase 24T — External-resume capacity reconciliation

**Scheduler correctness: GO for bounded reconciliation of verified, pause-safe experiment workers, with fail-closed diagnostics/admission for unsafe cases. Production TRAIN expansion: NO-GO on this evidence alone.** The Phase24S external-resume capacity defect is fixed and the final development qualification passed. Production remains untouched.

## Baseline and preservation

Development only, branch `dedicated-train-layout-rollover-squashed-v1`. Entry HEAD was `baf2e7bc368075f20c568f8f0fa01caf9892087c`. The five expected Phase24S changes were present, with no unexpected work. Read AGENTS.md, Phase24R/S reports, retained Phase24S Qualification7 logs/results/cleanup, and relevant Phase24Q qualification and memory evidence.

Reviewed and preserved the Phase24S tests and historical R/S reports in independent baseline commit **`e98375a595443ad39545223d1321cc83dee61e2e`**, `Phase 24S: Preserve scheduler displacement qualification and regression baseline`. The modified orphan unit suite passed before that commit. No generated evidence, credentials, database data or DerivedData files were staged. Phase24R now remains part of that baseline without further edits; 713 completed, and the later operator-reported suspended 714/resumed 747 observations remain historical. No later completion or live production state is asserted.

## Investigation, invariant and policy

The Phase24S counterexample was reproduced: external SIGCONT of a verified stopped attempt made orphan observation restore its running lifecycle and consume a slot, but no path reduced existing excess capacity. Ordinary admission correctly avoided a duplicate launch; that alone did not repair two active workers under INFER cap 1. This does not identify the cause of production displacement.

The correction runs after authoritative orphan/process observation and before dispatch, at startup and each scheduler cycle. It uses the existing scheduler lease/fence, exact active-attempt locking, native kernel start identity, PGID/executable/managed-argument validation, SIGSTOP path and preemption rollback compensation. It adds no authority system, protocol bypass or worker contract.

**Invariant:** after bounded successful reconciliation, verified, pause-safe experiment workers fit their configured per-phase caps, including zero. Uncertain/unowned identities, reservations, checkpoint workers and in-flight controls remain conservatively counted or block admission; they never become speculative signal targets. If correction cannot be verified, emit an actionable deferral/unresolved-capacity diagnostic and withhold admission. A continuously intervening external actor cannot be made atomic with PostgreSQL or OS signals: re-observation and the next cycle repair/detect subsequent transitions; there is no instantaneous system-wide cap guarantee.

Survivors follow the existing victim order from `PostgresSchedulerRepository::selectPauseSafePreemptionVictim`: lowest priority loses first, then newest `worker_started_at` (`NULLS LAST`), then descending experiment ID. Thus higher priority and older equal-priority workers survive. Unsafe workers are retained ahead of eligible victims regardless of priority. Selection is a deterministic observation snapshot; the signal transaction rechecks exact lifecycle, control eligibility, observed owner, authority and remaining capacity. It does not add fairness/aging or change ordinary strictly-higher-priority preemption, queue order or phase precedence.

Excess workers retain PID, attempt, launch owner/fence, model/checkpoint and semantic identity. The established stopped/pending/preemption transition permits later ordinary admission through the original PID/attempt. Operator-paused rows remain paused, retain their resume metadata, and receive no automatic SIGCONT. A verified external resume of such a row reasserts its pause only when no in-flight control transition conflicts.

| Hazard | Handling / qualification |
| --- | --- |
| PID reuse, start mismatch, wrong attempt or lifecycle binding | Existing exact/native validation refuses signals; ambiguous capacity is retained. Actual kernel PID reuse is represented deterministically by mismatched identity, not forced on the host. |
| Old owner, expired lease, stale fencing | Current authority is renewed under the existing transaction locks before control. Valid expiry permits takeover/adoption while preserving original launch provenance. Stale/live competing owners are rejected. |
| External resumes after observation/reservation | Re-observe exact stopped bindings before admission and at the parent child-exec gate. A gated child receives no exec permission if the census changed. Existing launch-failure workflow reports the rejected fresh attempt; it is not silently retried. |
| Exit or identity change near SIGSTOP | Ten observations, at most 2 ms sleep each, confirm the exact stopped process. Failure withholds admission; existing exact compensation handles pre-commit rollback. ESRCH is process missing, never permission to signal another PID. |
| Operator pause / checkpoint / cancellation / global control | Preserve pause; exclude in-flight transitions from stop targets. Ambiguous control defers explicitly and requires existing control completion or operator recovery. |
| Missing checkpoint after a stopped TRAIN exits | Existing authoritative recovery fails explicitly without a durable checkpoint; it does not restart TRAIN from scratch. A live suspended worker can resume retained memory. |
| Shutdown / commit ambiguity | Stop-request guard prevents correction/new exec. Pre-commit failure compensates the exact non-operator victim; uncertain commit outcome withholds compensation. No forced termination was added. |

## Source and test changes

- `ReconciliationService.hpp/.cpp`: pure deterministic excess-victim planner, using existing priority ranks and conservative total capacity.
- `ProductionSchedulerDaemon.cpp`: bounded capacity correction and diagnostics; preserve operator-paused lifecycle during external-resume observation; exact stopped-attempt admission census and parent exec-gate guard; reuse existing private-database failpoint facility for bounded race/rollback fixtures.
- `GlobalExperimentControl.hpp/.cpp`: reassert an externally resumed authoritative stopped pause through the existing verified control path. Caller must retain current scheduler authority and exact row locks.
- `OrphanedRunningExperimentReconciliationServiceTests.cpp/.sh`: planner priority/tie/null/zero/unsafe cases, keeping Phase24S observation/recovery assertions.
- `GlobalExperimentControlTests.cpp` and new `.sh`: exact stopped-pause validation, wrong attempt/start/binding, idempotence and ESRCH/missing-process cases; targeted Apple Clang runner.
- `GlobalExperimentControlProcessTests.cpp`: optional bounded alarm for inert phase fixtures; no qualified worker change.
- `SchedulerDisplacementRecoveryTests.py`: private native external-resume tests, exec-gate race, rollback, both-phase/zero caps, paused controls, excess-capacity restart and per-worker cleanup receipts. Its scheduler process census is restricted to registered fixture PIDs through a private `ps` wrapper, avoiding production inspection.

## Commands and evidence

All build/test outputs are under ignored **`DerivedData/ExpertAdvisor/Phase24T/`**. No Clean, CONFIGURATION_BUILD_DIR override, dependency change, canonical CLI replacement or TRAIN/INFER rebuild occurred.

Targeted private CLI build: `/usr/bin/python3 -B DerivedData/ExpertAdvisor/Phase24T/build.py`. Apple `/usr/bin/clang++`, C++20, ARM64/macOS 27.0; compile only GlobalExperimentControl, ReconciliationService and ProductionSchedulerDaemon, replace their objects in a private archive/file list, relink and ad-hoc sign the private test CLI. Exact compile/link argument arrays and logs are retained alongside the script. Existing stable DerivedData libraries/objects are reused. Header inputs were archived from the development MetaNN gitlink object into private BuildInputs; neither the shared symlink nor its production source target was traversed or edited. Initial include-order attempts selected incompatible libpqxx 9 headers; corrected compilation explicitly uses unchanged Homebrew libpqxx 7.10.1 and libpq. Final compile/link passed. Three pre-existing warnings remain justified outside this change: unused `CanonicalizeExecutablePath` and two omitted default `semanticWorkerRole` aggregate fields, unchanged in the baseline; no warning originated in the added correction. The focused unit runner suppresses the existing unused helper and otherwise uses `-Werror`.

Regression driver: `/usr/bin/python3 -B DerivedData/ExpertAdvisor/Phase24T/regressions.py`; commands/exits/timeouts in `Regressions/results.json`. It runs `bash Tests/{OrphanedRunningExperimentReconciliationServiceTests,SchedulerOrchestrationServiceTests,SchedulerAuthorityServiceTests,SchedulerPhasePriorityTests,GlobalExperimentControlTests,SchedulerOperationalObservationBoundaryTests,SchedulerTrainingWorkerRoutingTests,SchedulerAnalyzeWorkerRoutingTests}.sh`, manually compiles/runs `SchedulerSemanticAdmissionTests.cpp` under DerivedData, and runs all **14** offline publication-contract tests with their temporary directory relocated to DerivedData. All completed runs passed. No DB-backed shell suite was run against an ambient database.

Native command: `/usr/bin/python3 -B Tests/SchedulerDisplacementRecoveryTests.py --phase24t DerivedData/ExpertAdvisor/Phase24T/Qualification4`. The final CLI is private `Phase24T/build/LSTM_Release`. Complete command arrays, exit expectations, 30-second scheduler bounds, JSON SQL/OS snapshots, native logs and exact fixture signal receipts are retained. Fixture compilation is limited to 60 seconds; readiness and worker cleanup to four seconds; private daemon waits to five seconds; inert workers have a 180-second alarm while executing. Stopped fixtures still require verified harness cleanup.

| Required evidence | Result |
| --- | --- |
| A: INFER external resume, cap 1 | PASS Qualification4: one active OS worker and one active attempt within the first correcting cycle; no duplicate launch. |
| B: TRAIN equivalent | PASS Qualification4, same bounded correction and original attempt. |
| C: Independent TRAIN/INFER caps and zero | PASS Qualification4: both caps, deterministic stop, repeated zero-cap cycle and readmission. |
| D/E: unequal/equal priority, multiple resumes, pause preservation, recovery | PASS Qualification4: original detached PPID=1 identity/attempt, exact paused experiment row unchanged, normal readmission before low. |
| F: stale/reused identity, wrong binding/attempt, exit, in-flight transition | PASS native ambiguity/binding/in-flight exclusion and unit signal guards. Paused conflicting control explicitly defers, blocks admission and requires recovery; it is not claimed automatically corrected. |
| G: crash/restart, valid expiry, stale owner/fence | Qualification4 PASS with externally resumed excess at takeover; no changed launch owner, duplicate attempt or worker signal by stale owner. |
| H: normal displacement, checkpoint failure, rollback, routing/authority/phase priority | PASS final native and targeted regressions, including new capacity rollback and exec-gate race. |
| Committed result reconciliation | Three native invocations preserve one inference result and one attempt; no rerun. This is an authoritative-result fixture, not a new concurrent inference/profitability writer stress test. |

Qualification1 had one harness assertion failure: fence mutation raced startup and safely rejected authority before the expected heartbeat log. Fixture now waits for `SCHEDULER_START`; Qualification2/3/4 passed the heartbeat assertion. Final Qualification4 harness exit was 0; all recorded expected outcomes passed. Each run cleaned its cluster/credentials. Deliberate nonzero CLI outcomes for protocol/ownership rejection and injected failures are expected test assertions, not passing production executions.

## Isolation, artifacts and cleanup

Each run owns PostgreSQL 17 at explicit TCP **127.0.0.1:55485**, Unix sockets disabled, data directory beneath its own Phase24T qualification folder, SCRAM credentials freshly randomized for private admin/runtime roles, test databases `ea_scheduler_phase24s_lstm` and `ea_phase24s_forex` (retained fixture names). The runtime role name `pqxx` is required by the existing connection constructor; its password belongs solely to the independent cluster. Effective host/port/user/database/data directory and server system identifier are verified before writes and retained in results. No ambient PG settings, production credentials/schema dump or production DB dependency is used. Fixture SQL initializes test state only; scheduler operations use existing authoritative application workflows. Protocol pending rejects admission with zero invocations before test-only protocol fixture completion.

Qualified unchanged hashes rechecked:

| Artifact | SHA-256 |
| --- | --- |
| Dedicated TRAIN | `018333a5694971e1dcd97a97d8f4a2018243f3a7ec96ee728dd74f4954a577d9` |
| Dedicated INFER | `a34cb34d111f746e1757366af8dc36a7ffa49d811db17b779ef68c3c8a2e1d35` |

Nested `MetaNN/MetaNN` symlink text is unchanged. Production source, database, registry, logs, runtime, processes and configuration were neither inspected nor mutated. Git operated only on the development worktree and its shared worktree Git metadata. No production signals, publication, merge, push, Qwen reconfiguration/duplicate instance, or production TRAIN activation occurred.

Final cleanup **PASS**: `Qualification4/results.json`, **41** per-worker receipts in `cleanup-workers.jsonl`, exact signal receipts and private `pg_ctl` stop exit 0. Server identifier was **7694393893188944814**. All four run directories were checked: cluster storage and both password files are absent. The harness revalidates each fixture identity before SIGCONT/TERM, reaps direct children or verifies detached exit, shuts only its owned private daemon, stops `pg_ctl -D <verified private pgdata> -m fast -w`, removes cluster storage/password files, and retains logs. Qualification1/2/3/4 `cleanup` results are PASS. Direct child reaping and detached-exit checks occurred inside the private harness; no host-wide or production census was performed.

## Memory and readiness

One bounded host snapshot reports `memory_pressure` **50% free**; `vm_stat` counters are cumulative since boot, not swap rates. A sandboxed `sysctl vm.swapusage` read was denied; one read-only authorized retry reported **11,195.31 MiB used / 12,288 MiB total**. Evidence: `host-memory.json`, `host-swap.log`. No process attribution or production workload inspection was performed. Swap used alone is retained historical allocation and does not establish present pressure or sufficient headroom.

Phase24Q used a 48-GiB host and tiny sequential native jobs: sampled free 41–49%, ablation TRAIN RSS about 56 MiB, continuation about 75 MiB, INFER about 33 MiB, and no swap-out increment over that qualification. These coarse short samples do not establish representative simultaneous TRAIN/INFER/Qwen peaks. Suspended models may retain CPU/Metal allocations. This phase ran no model/GPU workload and did not change Qwen or impose GPU concurrency limits.

**Scheduler correctness: GO for the stated scoped invariant and fail-closed behavior; no claim of unconditional instantaneous control over external actors or unsafe in-flight workers.** Remaining limits: externally stopped active rows conservatively retain capacity and do not receive automatic SIGCONT; endless higher-priority arrivals have no new starvation bound; checkpoint/in-flight/unknown workers can explicitly block correction and require supported recovery; a rejected pre-exec reservation uses the existing terminal launch-failure path and may require an operator retry. OS/DB/external signals cannot form an atomic global transaction. Real kernel PID recycling and concurrent result/profitability writer stress were not requalified.

**Production TRAIN=1 / INFER=1 expansion: NO-GO on this phase alone.** No activation is authorized. Next operator action is review the development correction and separately qualify representative memory headroom and chosen phase policy before requesting controlled production deployment/activation. Do not change current production jobs or scheduling on this evidence alone.

Final `git diff --check` passed. Phase24T source/tests/report and the concise Phase24S follow-up are committed together only after the passing final qualification; the commit ID and post-commit HEAD/status are recorded in the final response and ignored `Phase24T/final-git-review.log`. No report or implementation was pushed.
