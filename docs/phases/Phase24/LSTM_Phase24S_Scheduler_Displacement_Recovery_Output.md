# Phase 24S — Scheduler displacement and recovery qualification

**Supported priority displacement and finite-queue recovery: PASS. NO-GO for declaring all requested invariants satisfied or expanding production to TRAIN 1 / INFER 1 on this qualification alone.** A native isolated counterexample leaves **two active INFER processes after a scheduler cycle under cap 1** when a stopped worker receives an external SIGCONT. No duplicate worker was launched. Scheduler-controlled admission and recovery passed; the unconditional cap/recovery invariant did not. No scheduler implementation fix was made and no production action was taken.

Qualification date: **2026-10-08**, finalized approximately **19:39 UTC**. Development HEAD **`baf2e7bc368075f20c568f8f0fa01caf9892087c`**, branch **`dedicated-train-layout-rollover-squashed-v1`**. Initial worktree contained only the untracked Phase 24R report. Read AGENTS.md, Phase 24Q/R evidence and relevant N/O/P reports. No commit, merge, push, publication, deployment or production TRAIN activation occurred.

## Intended policy and verified boundaries

Reviewed `ProductionSchedulerDaemon.cpp`, `PostgresSchedulerRepository.cpp`, `ReconciliationService.cpp`, `SchedulerPolicy.cpp`, `SchedulerPhasePriority.hpp`, `SchedulerOwnershipRepository.hpp`, `GlobalExperimentControl.cpp`, and their relevant tests.

| Mechanism | Intended behavior / verified result |
| --- | --- |
| Eligibility and order | Pending rows sort by high/normal/low priority, then operator/preemption/ordinary resume origin, then updated_at and experiment ID. This is not FIFO by experiment number. Paused rows are not admission candidates. |
| Same-phase displacement | Requires strictly higher priority and a pause-safe running victim. Victims sort lowest priority first, then newest worker_started_at and descending experiment ID. Equal-priority same-phase work does not preempt through this predicate. |
| Across-phase policy | Highest eligible pending priority controls admission globally. Ordered phase precedence breaks ties and drains losing classes, including equal-priority workers in another phase. Concurrent policy permits eligible classes at the winning priority; enabling both caps does not guarantee both classes will run simultaneously. Phase 24R recorded production policy `train:infer:analyze`; this phase did not change it. |
| Signal and transaction boundary | Exact active attempt, lifecycle binding, PID, PGID, kernel start identity, canonical executable and managed arguments must match before signaling. SIGSTOP precedes transactional stopped/pending/preemption persistence. Verified rollback compensation SIGCONTs the original worker after a pre-commit failure. Ambiguous commit outcome deliberately withholds compensation. |
| Slots | Reserved/spawned/running/observed/identity_ambiguous attempts consume capacity. A verified persisted-stopped, OS-stopped attempt does not. Identity uncertainty consumes capacity and prevents speculative replacement. |
| Recovery | Pending resume_requested stopped attempts can reuse their exact process/attempt when eligible and a slot is available. Launch owner/fence remain historical; the observing scheduler identity changes. PPID 1 alone does not prevent adoption. |
| External transitions | A persisted-stopped worker observed executing is restored to running/observed and consumes capacity. An externally stopped worker whose attempt was running remains observed/capacity-consuming; it is not automatically SIGCONT'd. Neither behavior imposes a post-observation cap repair. |
| Missing worker/result | Exact authoritative completed INFER evidence wins over relaunch. Missing running work without result fails. Missing preempted TRAIN without a usable checkpoint explicitly fails; checkpoint restart planning remains supported. |
| Authority | Existing coordination lock, protocol barrier and lease/fence remain authoritative. A fresh dead-owner lease still rejects acquisition; expired positively dead ownership can be replaced. A stale context loses authority on heartbeat. |
| Fairness | Finite higher-priority demand drained in the test, after which low-priority work resumed. There is no aging, maximum suspension deadline or bounded-wait guarantee under continuing higher-priority demand. Queue/attempt diagnostics expose deferred work; they do not establish an automatic starvation alarm. |

These mechanisms can produce the reported production-shaped states. They do **not** identify which component stopped 714, establish its priority, prove production starvation, or prove duplicate execution. No production PID was used as a test fixture.

## Isolation and native evidence

Final evidence directory: **`DerivedData/ExpertAdvisor/Phase24S/Qualification7/`**. New PostgreSQL 17 cluster, independent storage `Qualification7/pgdata`, TCP **127.0.0.1:55485**, no Unix sockets, fresh SCRAM credentials for private `phase24s_admin` and `pqxx`, test databases **`ea_scheduler_phase24s_lstm` / `ea_phase24s_forex`**. Server identity **`7694383361292401343`**. Destination was positively checked against database, user, address, port and exact data_directory before database creation. No production credential, connection string, record or backup archive was reused.

Environment is constructed afresh with explicit PGHOST, PGPORT, PGUSER, PGDATABASE, PGPASSFILE, PGCONNECT_TIMEOUT and both LSTM_DB/FOREX_DB destinations; ambient connection/service overrides are not inherited. The scheduler's explicit `user=pqxx` connection uses the private PGPORT and password file. The private pqxx role is a disposable fixture superuser, not a production grant change.

Restored **checked-in schema only**, correcting its appended DDL search_path in memory, then migrations **046, 051, 052, 071, 078, 086, 093, 099**. No live pg_dump was used. Native pending-protocol startup rejected with exit **3** before any invocation row. The complete-generation-52 row was then established as an explicit private regression fixture, consistent with existing integration tests. **This is not supported operational protocol-cutover qualification:** no cutover CLI, process-guard bypass or production protocol write was attempted while production is live.

The harness compiles the existing inert managed-process fixture with Apple Clang and constructs a **private synthetic content-addressed registry** for layout9/103 INFER and layout13/171 TRAIN/INFER. Runtime resource bytes are inert fixtures, never loaded by Metal. Actual development scheduler code performs native ownership, preflight, dispatch admission, SIGSTOP/SIGCONT and reconciliation; the fixture workers do not calculate predictions or train models. Phase 24Q remains the native model/Metal qualification evidence. No real TRAIN job was activated here.

| Case / fixture experiments | Result / retained evidence |
| --- | --- |
| INFER displacement: 995000 low/layout9, 995001 high/layout13, 995002 normal/layout9, 995003 operator-paused | **PASS**: low becomes pending/infer with retained PID, stopped attempt and OS Ts; high resumes. After high exits, normal resumes before low; low subsequently resumes with original PID/attempt. Four attempts remain four. `infer-*.json` and native logs. |
| TRAIN analogue: 995100–995103 | **PASS**: same sequence, exact private selected TRAIN executable matches; no model training. Paused full experiment row remains identical. `train-*.json`. |
| Restart and detached workers | **PASS**: displaced low workers have verified PPID 1. Separate native scheduler invocations retain stopped attempts and later resume them without relaunch. |
| Operator pauses | **PASS**: both paused fixtures remain stopped; full experiment rows byte-equivalent across displacement and recovery. No scheduler-issued SIGCONT to those fixtures. Harness-only teardown subsequently removes them. |
| External stop/resume: 995200 | **PASS for conservative accounting**, **recovery limitation**: externally stopped running attempt retains its slot; no automatic resume. Explicit private SIGCONT restores execution without another attempt. |
| PID reuse surrogate: 995210/995211 | **PASS**: wrong persisted kernel start identity becomes identity_ambiguous, consumes one slot, prevents preemption/replacement and leaves candidate stopped. Real PID allocator reuse was not forced. |
| Conflicting lifecycle identity: 995215/995216 | **PASS**: attempt and experiment start identities disagree; exact-attempt preemption fails closed, original process is not signaled. |
| Completed INFER: 995220 | **PASS**: synthetic committed final-result fixture recovered to pending/analyze, attempt completed. Three invocations retain one attempt and one result. This tests replay avoidance, not concurrent real prediction/profitability writers. |
| Rollback and missing TRAIN: 995250/995251 | **PASS**: established private-database failpoint after SIGSTOP compensates with verified SIGCONT. Subsequent real preemption plus disappeared worker yields failed status, cleared binding and `preempted_worker_missing_no_valid_checkpoint`; no fresh training restart. |
| External over-cap: 995230/995231 | **FAIL against unconditional active-INFER cap invariant**: cap1; external SIGCONT of a pending/preempted stopped worker; after native cycle **two OS-active processes, two capacity-consuming attempts, zero new attempts**. `external-over-cap.json`, native log, `results.json`. |
| Real owner crash/restart: 995240 | **PASS**: competing live owner rejected with exit3; own daemon SIGKILL leaves detached worker intact. Fresh dead-owner lease rejected. Only private test expires_at advanced to avoid a 90-second wait; expired dead owner then replaced, fence increased, original launch owner/attempt retained and observing owner updated. |
| Stale fence | **PASS**: private lease token increment invalidates live daemon context; native heartbeat logs ownership loss and exits **4** without signaling the worker or creating a duplicate attempt. |
| Routing | **PASS**: real preflight selects private historical layout9/103 and current layout13/171 role paths. TRAIN resumes only its selected canonical artifact. Missing-capability/semantic rejection covered by the existing pure admission suite; production publication/routing evidence remains Phase 24R. |

## Tests, commands and changes

All final regression commands passed; **the external-cap safety finding remains FAIL and is not hidden by the harness's exit0**. The harness asserts the current counterexample explicitly as a characterization, rather than claiming hard-cap protection.

```bash
bash Tests/OrphanedRunningExperimentReconciliationServiceTests.sh
bash Tests/SchedulerAuthorityServiceTests.sh
bash Tests/SchedulerPhasePriorityTests.sh
bash Tests/SchedulerSemanticAdmissionTests.sh
bash Tests/SchedulerOrchestrationServiceTests.sh
/usr/bin/clang++ -std=c++20 -Wall -Wextra -Werror -IHeaders -ISources \
  Tests/SchedulerOwnershipPolicyTests.cpp Sources/SchedulerCore/SchedulerAuthorityService.cpp \
  -o DerivedData/ExpertAdvisor/Phase24S/SchedulerOwnershipPolicyTests
DerivedData/ExpertAdvisor/Phase24S/SchedulerOwnershipPolicyTests
/usr/bin/python3 -B Tests/SchedulerDisplacementRecoveryTests.py \
  DerivedData/ExpertAdvisor/Phase24S/Qualification7
```

The final harness also ran `Tests/SchedulerOwnershipIntegrationTests.sh` against its private cluster: migration replay, unique active attempts and protocol/exact-binding contracts passed. Its existing minimal model fixture lacked `name`, required by the SQL regression; added that single nullable column. No database migration or application behavior changed.

Native scheduler cycles use the existing Rollover Release `LSTM_Release`, `--schedule-experiments --scheduler-once`, capacities **0/1/0** or synthetic-TRAIN **1/0/0**, the private registry/analyzer paths and private log directory. Automatic continuation flags are omitted, yielding verified false defaults. Crash/fence daemons use **0/1/0**, `--scheduler-poll-seconds=1`. All exact argv, destinations and exits are retained in `commands.jsonl`, `environment.json`, logs and snapshots. Child commands have timeouts; process readiness/transition waits are bounded to 4–5 seconds. No production job completion wait occurred.

Process fixture build links the existing GlobalExperimentControlProcessTests, GlobalExperimentControl, CheckpointPolicy, checkpoint-evaluation service and scheduler repository/policy/observation sources with Homebrew libpqxx. Apple Clang **21.0.0**, ARM64; no emitted compiler warnings in that build. Pure suites use their established `-Wall -Wextra -Werror` patterns. No xcodebuild, Clean, CONFIGURATION_BUILD_DIR override, dependency modification or TRAIN/INFER rebuild.

Earlier setup attempts are retained as Qualification1–6: missing private registry before admission check; detached launcher's inherited output descriptors; incorrect expected rejection exit code; and the existing minimal-schema missing-name failure were corrected. One escaped **private** fixture from the descriptor failure was identified by exact private path/start/arguments and terminated; its receipt is retained. An initial ownership-policy link command omitted SchedulerAuthorityService.cpp and was corrected. These were test setup defects, not scheduler displacement failures. All attempts were cleaned up. A sandbox-denied Python bytecode-cache write was avoided by using `-B`; sandbox-denied swap sysctl was repeated successfully as read-only.

Changes: new `Tests/SchedulerDisplacementRecoveryTests.py`; exhaustive 32-state observation and restart sequencing assertions in `Tests/OrphanedRunningExperimentReconciliationServiceTests.cpp`; minimal fixture correction in `Tests/SchedulerOwnershipIntegrationTests.sh`; preserved and updated Phase24R report; this Phase24S report. **No scheduler implementation or scheduling policy change.** Reports and tests remain uncommitted.

## Protection, cleanup and readiness

Qualified TRAIN/INFER hashes unchanged before/after; current reused CLI hash also recorded:

| Development artifact | SHA-256 |
| --- | --- |
| TRAIN | `018333a5694971e1dcd97a97d8f4a2018243f3a7ec96ee728dd74f4954a577d9` |
| INFER | `a34cb34d111f746e1757366af8dc36a7ffa49d811db17b779ef68c3c8a2e1d35` |
| Reused development CLI | `5d2097f73fbffd363b1fad7d194615d4aa8b74ed10421a49bf33ca45802f0c4a` |

Nested MetaNN link still targets `/Volumes/Developer SSD/ExpertAdvisor/MetaNN`; no shared source was edited. Stable Rollover DerivedData paths preserved. No live production census, DB connection, source/registry/configuration/binary modification, job mutation or production signal was performed. Production evidence came from retained Phase24Q/R records and explicitly labeled operator observations. Qwen configuration/instances were untouched.

All seven private clusters stopped successfully with pg_ctl targeting their owned data directories; directories and credential files removed. Final exact-identity checks found **zero matching live fixture workers** across all attempts, including the rescued detached fixture, and zero matching test daemons. Final run verified **18 known workers and two daemon identities** absent. Cleanup receipts, final hashes and regression summary remain under `DerivedData/ExpertAdvisor/Phase24S/`; generated data is not staged.

One host-only memory sample reported **54% memory_pressure free**; swap **11,598.38 MiB used of 13,312 MiB**. These are snapshots, not swap-rate or production peak measurements. Retained Phase24Q evidence covers a 48-GiB host and tiny sequential native jobs, not full simultaneous TRAIN/INFER/Qwen peaks. SIGSTOP relinquishes scheduling activity without guaranteeing release of model/GPU allocations; stopped workers must be included in a future memory budget. This phase used no GPU workload and imposed no arbitrary GPU concurrency limit.

Remaining gaps: externally resumed over-cap workers have no automatic cap repair; unrecorded external stops have no automatic resume; indefinite higher-priority arrivals have no bounded low-priority wait; real PID reuse is represented by a deterministic start-identity mismatch; real concurrent inference/profitability writers and simultaneous production-scale memory peaks were not requalified. Private protocol completion is fixture initialization, not operational cutover. Ordered cross-phase behavior was inspected and unit tested, while the native displacement harness used concurrent phase policy. No current production state or later 714/747 completion is claimed.

**Next recommended operator action:** leave production configuration and jobs unchanged. Review a narrowly scoped development recovery contract for an exact externally resumed worker when capacity is already full (which process may safely relinquish and how that decision is reported), and agree on explicit overdue-recovery handling without silently changing priority policy. Separately qualify resource headroom and the chosen ordered/concurrent phase policy before authorizing TRAIN 1 / INFER 1. The supported finite-queue recovery path needs no speculative scheduler rewrite; the reproduced external-transition gap remains unfixed and does not establish the cause of the production observation.

Final HEAD unchanged. `git diff --check` passed. Tracked diff: two test files, **57 insertions / 1 deletion**; untracked additions are the new native harness and both phase reports. Exact final status/stat are recorded in `DerivedData/ExpertAdvisor/Phase24S/final-git-review.log`.


Development archival status at Phase 24T entry: this completed historical report and the preserved Phase 24S tests are included in the independent Phase 24S baseline commit. Statements above about uncommitted state describe their original phase closeout. Retained private evidence was reviewed; no later production state was inspected or asserted.


Phase24T follow-up: [external-resume capacity reconciliation qualification](LSTM_Phase24T_External_Resume_Cap_Reconciliation_Output.md) fixes the reproduced capacity gap for verified, pause-safe experiment workers, with deterministic existing victim order, preserved attempts/operator pauses, bounded correction and fail-closed deferrals. Final isolated native and targeted regressions passed. This does not identify the cause of the historical production displacement or qualify representative simultaneous memory demand; production TRAIN expansion remains NO-GO pending separate qualification and authorization. The original Phase24S findings above remain historical evidence.
