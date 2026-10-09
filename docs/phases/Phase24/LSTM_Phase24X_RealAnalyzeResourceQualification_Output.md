# Phase 24X Real ANALYZE Workload Resource Qualification

**Current assessment: two authentic fixtures independently PASS at concurrency 1. Ready to implement a bounded concurrency-2 qualification; not ready to launch it with the existing single-worker harness.** Experiment 681/model 1982/inference 979 reproduces all 41 historical scientific fields and passes resource coverage and cleanup, independently of experiment 746. Validate two-worker identity, overlap, measurement attribution, stop gates and TRAIN/INFER headroom before launching the next level. No concurrent workload ran, and no positive production resource capacity is qualified. Keep production TRAIN=2, INFER=1, ANALYZE=0.

The prerequisite findings appear under **Recovered historical fixture and single worker validation**, followed by **Instrumented single-worker resource qualification** and **Second authentic historical workload qualification**. Earlier assessments are retained as historical evidence and superseded where the later sections establish new results.

## Initial assessment retained

**BLOCKED before real workload execution. No real ANALYZE concurrency limit is qualified. Keep production TRAIN=2, INFER=1, ANALYZE=0.** The available offline backup contains genuine persisted results, but no pending ANALYZE experiments or accessible private TRAIN/INFER logs. A representative private workload and its isolation have not been demonstrated. The requested prerequisite therefore requires stopping before the concurrency ladder.

The reproducible preflight collector and a proposed execution design are delivered below. The full real-workload runner, private database bootstrap, per-worker measurements, and capacity recommendation above zero remain deferred. No synthetic worker result is used as resource evidence.

## Checkout and protected systems

Repository: `/Volumes/Developer SSD/ExpertAdvisor-Rollover`.
Branch: `dedicated-train-layout-rollover-squashed-v1`.
Baseline HEAD: `15d9e9762ac9b7ff12e69ff7009cf391b7e8c715`.
The initial worktree was clean. The root `AGENTS.md` was read; no additional source-tree `AGENTS.md` was found. Its stale branch description does not override the explicitly requested branch.

Phase 24W Native18 passed the bounded synthetic capacity, priority, original INFER-attempt recovery, identity cleanup, and reconciliation assertions. Its report explicitly excludes real resource qualification. Existing evidence directories were retained.

Read-only process observations showed production scheduler PID 55976 configured with `train:infer:analyze`, TRAIN=2, INFER=1, ANALYZE=0. TRAIN PIDs 57072 and 57078 were active; INFER PID 52730 was stopped (`Ts`). Commands and start times are retained in the process logs. No production scheduler CLI or database was contacted. Persisted scheduler status was not queried; process observations establish only the running command configuration and OS state.

No production repository, database, setting, worker artifact, or process was modified. No database server, scheduler, or workload was launched. No worker signal, commit, push, publication, clean, or application build was performed.

## Real ANALYZE execution and resource dependencies

The dedicated entry point is `LSTM/AnalyzeWorkerMain.cpp`. It calls `RunStandaloneAnalyzeWorkerCli` in `Sources/SchedulerCore/ExperimentScheduler.cpp`. The worker requires the exact scheduler attempt, registers through `SchedulerWorkerRegistration.cpp`, and then calls `AnalyzeExperimentById`. Registration verifies experiment, attempt, PID, process group, native process-start identity, canonical executable, and lifecycle binding. Finalization locks and revalidates the exact active attempt and model source. A standalone worker command with an invented attempt is not a valid benchmark.

| Resource | Source finding | Qualification implication |
|---|---|---|
| CPU | `ParseMetricsFromLogs` performs repeated `std::regex` searches over complete log strings. Structured metrics are applied afterwards. Ranking, persistence and optional reports also consume CPU. | Actual log sizes and report population must be represented, even when structured inference results exist. |
| Unified memory | `ReadFileIfExists` materializes each complete log through an `ostringstream`; the parser also constructs a combined TRAIN/INFER string. Regex state and report query results add allocations. | Missing or abbreviated logs can drastically understate memory. RSS sampling alone does not establish lifetime peak or Metal allocations. |
| GPU and Metal | The inspected final-analysis path reads persisted metrics and model metadata; it does not execute inference, construct an LSTM, or invoke a Metal compute workload. | Direct GPU demand is expected to be negligible for this path, but has not been measured. Linked Metal code is not proof of GPU execution. |
| PostgreSQL | Registration, read-only metric extraction, write finalization, and optional report generation open separate scoped libpqxx connections sequentially. Log parsing occurs while the read transaction is open. | Expect approximately one connection per active final worker, plus scheduler/admin observers and PostgreSQL background activity. Verify actual peaks; this is source inference, not a measured bound for the entire scheduler. |
| Disk | Complete input logs are read. Worker output and analysis summaries are written; PostgreSQL writes results/WAL. Optional reporting queries broader history and rewrites report files. | Private input/output paths, database storage, cold/warm cache behavior, and report settings must match the intended workload. |

`AnalyzeExperimentById` reads an experiment's model, symbol/configuration, date range and log paths; it looks for a completed final `inference_eval_result` matching model, symbol, horizon, threshold and dates. It persists `experiment_analysis_result` and completes the experiment. Raw ticks and model matrices are not directly consumed by this final analysis path. They would be required to regenerate missing TRAIN/INFER outputs through their authoritative workflows.

Checkpoint analysis is a distinct workload. `RunCheckpointEvalAnalyzeJobs` in `ProductionSchedulerDaemon.cpp` runs synchronously inside the scheduler process, uses completed checkpoint inference metrics, persists checkpoint analysis and evaluates policy. It shares ANALYZE capacity accounting but does not launch a separate dedicated ANALYZE worker. Final-worker concurrency results must not be generalized to checkpoint policy/report workloads or the separate meta-analysis CLI.

## Admission and phase priority

`SchedulerAdmissionService` admits by durable capacity counts and configured limits, using existing ownership, reservation, and transactional fencing. The inspected dispatch paths do not gate admission on live memory pressure, swap growth, CPU saturation, or disk latency. Resource warnings in `SchedulerStatusService.cpp` are observations, not admission controls.

`SchedulerPhasePriority.hpp` selects the lowest numeric eligible scheduler-priority rank across phases; `train:infer:analyze` breaks ties. Ordered admission drains losing phases before admitting the winner. Higher-priority ANALYZE can therefore displace lower-priority TRAIN/INFER. Phase exclusion does not guarantee immediate release of memory: stopped processes can retain allocations. Preserve these semantics rather than treating the three capacity limits as a host resource budget.

## Available data and missing prerequisites

Offline `pg_restore --list` and public-table data extraction inspected only the existing local `Database/backups/LSTM_latest.dump`. No live database connection was made. Its size is 639,872,182 bytes; its sidecar identifies a 2026-10-03 archive. Extraction confirms migration 096, 469 experiments and 944 completed inference results: 442 final and 502 checkpoint.

The experiments comprise 447 completed/done, 17 cancelled/train, four paused/train, and one failed/train. **None are pending/analyze.** The 938 TRAIN/INFER path fields comprise 633 missing paths inside the checkout, 164 paths resolving outside it, and 141 nulls. No private input log was available. Paths outside the checkout were rejected without inspecting their contents.

The archive contains a completed generation-52 protocol row with historical production cutover identity. That historical record is not a demonstrated private bootstrap or scheduler ownership qualification. It has not been restored or reinterpreted as a private cutover receipt. The backup also predates current migration 099; migration compatibility and persisted identities need explicit review before reuse.

The Phase 24S/24T/24W harnesses provide disposable SCRAM PostgreSQL setup, bounded commands, reservations, process inspection and exact-identity cleanup, but their worker/model fixtures are inert. Their fixture SQL marks protocol completion and seeds experiment/attempt states. Those setup operations must not be copied into real scientific qualification as substitutes for authoritative workflows. Their configured `max_connections=16` is also insufficient for a possible 18 active final workers plus scheduler and observers.

Phase 24Q exercised real worker code on short synthetic market inputs. Its private databases were removed after cleanup, and its report excludes production data distributions and resource peaks. It cannot supply this phase's real representative inputs.

Required before continuing:

1. A scientifically reviewed, frozen corpus of genuine TRAIN/INFER logs and matching model/configuration/inference records, with hashes, lengths, dates, symbols, horizons, semantic identities and expected analysis outputs. Include representative typical and large log/report populations; do not omit logs because structured metrics exist.
2. A validated private cluster and data directory, private credentials/port/catalog, compatible migrations, and all referenced filesystem inputs and outputs inside the qualification directory.
3. An authoritative workflow to create fresh ANALYZE candidates and scheduler-owned attempts from those inputs. Do not reset completed experiments or manufacture active attempts with SQL to create repeatable work. No supported ANALYZE-only replay/bootstrap was established during this investigation.
4. A private scheduler bootstrap that respects protocol and host process protections. Fresh generation-52 completion currently checks host-wide scheduler presence before opening the destination database; production is present. Do not forge absence evidence, stop production, or substitute a fixture SQL update. Establish an applicable supported isolated workflow or use a separate equivalent Mac Studio without production dispatch processes.
5. A validated real-workload measurement and identity-safe cleanup runner, and a host baseline with an explicit TRAIN/INFER resource reserve.

## Retained baseline measurements

Final preflight evidence is under `DerivedData/ExpertAdvisor/Phase24X/Preflight-20261008-02`; the first run remains under `Preflight-20261008-01`. Each directory contains command argument arrays, UTC timestamps, deadlines, return codes, output logs, offline inventories and baseline samples. The final collector also retains its source and SHA-256. All collector commands succeeded; exit 2 is the deliberate BLOCKED qualification result.

The first run sampled 2026-10-09 00:42:23 through 00:42:33 UTC, approximately ten seconds. The final run sampled 00:43:35 through 00:43:46 UTC and again recorded zero swap/pageout growth and no collection failures. These are observational baselines while production continued naturally, not stress tests. The table reports the first run:

| Measurement | Observation |
|---|---|
| Physical unified memory | 48 GiB |
| CPU | 18 physical/logical cores |
| GPU inventory | Apple M5 Max, 40 GPU cores, Metal 4 |
| Memory pressure | Raw kernel level 1; `memory_pressure -Q` free percentage 59% |
| Swap allocated/in use | 22,528 / 20,790.88 MiB |
| Swap/pageout growth during samples | Zero swap-ins, swap-outs and pageouts |
| TRAIN RSS | Approximately 3.36 and 3.37 GiB, about 6.73 GiB combined |
| TRAIN CPU | Approximately 63–67% per process in `ps` samples; 100% denotes one core |
| Stopped INFER RSS | 6,176 KiB; unusable as an active INFER reserve |
| Disk | System-wide iostat; second disk0 interval 49.10 MB/s; no worker attribution |

Existing swap occupancy is not evidence of current swap growth. Conversely, ten seconds without growth and a pressure percentage do not establish available stress headroom. Process RSS is not total physical footprint, does not bound GPU/unified allocations, and these snapshots are not measured TRAIN/INFER peaks. No production work was stimulated to obtain these observations.

GPU utilization, private PostgreSQL connection count, ANALYZE lifetime peak/sustained footprint, execution time, throughput, failures and timeouts are **unmeasured**. There were zero qualification workload launches.

## Proposed bounded execution harness

This is a design for resumption after prerequisites pass, not an implemented or validated workload runner.

Use one fresh evidence directory per run, with a private SCRAM cluster listening only on a unique loopback port and no default socket. Verify server address/port, database name, data directory and cluster system identifier before each administration stage. Sanitize the child environment; set both LSTM/FOREX host and database overrides, explicit `PGPORT`, connection deadline and a private exact-host/port password file. Connection strings fix `user=pqxx` and omit port, so `PGUSER` alone cannot establish isolation. Reject ambient service/credential/routing overrides. Use no production credentials.

Import immutable genuine scientific inputs through a reviewed bootstrap; use authoritative workflows for all lifecycle/materialization changes. Verify that no archived active production ownership is eligible for adoption. Confirm the scheduler process census cannot adopt or signal a production process. The old inert harness's observation filtering is useful infrastructure, but applying it to real scheduler execution requires validation; a wrapper must never falsify protocol absence evidence or weaken identity checks.

Use the actual dedicated worker and private scheduler. Record executable/source/runtime hashes and each persisted invocation/fence, experiment/attempt, PID, group, native start identity and exact command. Reserve and spawn through the scheduler's existing execution gate. Keep TRAIN/INFER dispatch disabled in the private trial; reserve their resources analytically rather than simulating them with production processes or synthetic workers.

Run levels 1, 2, 4, 8, 12, 18 only after the previous level passes. Proposed bounds are 120 seconds per worker, 180 seconds per cohort, at most three independently prepared cohorts per level and a 45-minute overall deadline. Stop after the first failed cohort. These initial deadlines require review against actual log sizes before launch. Do not add sleeps or inert work to artificially extend worker lifetimes. Require observed simultaneous real execution; a cap of 18 with short sequential completions does not demonstrate 18 concurrent workloads.

At 250 ms intervals, capture owned-worker CPU time and physical footprint/RSS with native read-only process APIs; take memory pressure, vm_stat and swap samples each second. Retain per-worker lifetime maximum RSS from wait/rusage where the owning parent can observe it; otherwise label sampled maxima accurately. Measure whole-cohort private PostgreSQL connections, backend memory, latency and write/WAL statistics, plus disk byte/activity deltas and wall/CPU execution time. Validate collectors on the one-worker run before escalation. Treat PostgreSQL/data-directory totals separately from worker memory.

GPU/Metal utilization should use an available read-only GPU counter provider and include whole-host background context. If that provider is unavailable, record the limitation and qualify only the inspected CPU/database/log-processing path; do not fabricate utilization from linkage or wall timing. CPU utilization from `ps` is contextual sampling, not a substitute for per-worker CPU-time deltas.

Run the actual intended automatic-report setting and private report directory topology. Multiple workers rewrite common report paths through `WriteTextFile`; independent directories would hide possible production contention. Count report warnings and verify file completeness as well as analysis completion. Report generation is warn-only, so an exit-zero worker alone cannot establish reporting correctness. Validate persisted scientific metrics against expected outputs and terminalize attempts through scheduler reconciliation.

Proposed conservative stop gates, fixed before launch:

- Require a minimum 60-second acceptable baseline with no swap-out growth and normal pressure; on this already swap-occupied host, defer qualification until the resource reserve is demonstrated. Do not clear swap or terminate background processes.
- Stop immediately on elevated pressure, new swap-out activity, allocation/Metal errors, worker failure, report corruption, lost fencing/identity, unexpected process or database destination, or deadline expiry.
- Stop if measured available memory falls below the TRAIN/INFER reserve plus a 20% physical-memory safety margin (9.6 GiB here); stop if CPU exceeds 85% of total capacity or database/disk latency exceeds twice the one-worker baseline for five consecutive seconds. These are proposed policy choices, not observed safe thresholds.
- Do not escalate if per-worker median duration exceeds twice the one-worker baseline, throughput regresses, actual overlap cannot be verified, or any required measurement is unavailable.

Freeze further admission before abort cleanup. Revalidate the exact persisted PID/start/group/executable/command/experiment/attempt immediately before every signal; use a process-group signal only when its ownership is established. Permit bounded SIGTERM then identity-verified SIGKILL only for owned surviving children. Never signal a scan-discovered PID merely because its name matches. If identity cannot be proved, record failure and preserve the private cluster/evidence for resolution; do not claim cleanup PASS or automatically stop a server still needed by unidentified workers. Verify private postmaster identity/data-directory/system identifier before shutdown. Preserve all scientific inputs, receipts, logs and results.

## Concurrency results and configuration

| Real ANALYZE workers | Result | Resource measurements |
|---|---|---|
| 1 | NOT RUN | None |
| 2 | NOT RUN | None |
| 4 | NOT RUN | None |
| 8 | NOT RUN | None |
| 12 | NOT RUN | None |
| 18 | NOT RUN | None |

Maximum demonstrated stable concurrency is **undetermined**. Zero workers launched is not a demonstrated stable capacity of zero. Whether 18 real workers are practical is **undetermined**; 18 CPU cores and Phase 24W synthetic success cannot answer it.

Retain production `--max-train-procs=2 --max-infer-procs=1 --max-analyze-procs=0 --phase-priority=train:infer:analyze`. This preserves the current configuration; it is not a newly qualified positive capacity.

For future accounting, reserve `2 × worst-case TRAIN footprint + worst-case INFER footprint + PostgreSQL/scheduler/background allowance + 9.6 GiB safety margin` before allocating ANALYZE slots. Use measured footprints including unified GPU allocations and stopped-worker retention. The observed 6.73 GiB TRAIN RSS and tiny stopped INFER RSS cannot supply those peaks. Numeric TRAIN/INFER headroom remains unknown.

No scheduler source change is justified by a reproduced resource-admission defect here. Live pressure/swap admission gating and log-size-aware estimates are candidates for a separately scoped change after real measurements; report write coordination may also require qualification. They must preserve durable reservations, ownership fencing, exact cleanup and phase precedence. Nothing in this report authorizes production rollout.

## Validation and review output

New files:

- `Tests/AnalyzeResourceQualificationPreflight.py`: bounded offline inventory and read-only resource evidence collection; always reports BLOCKED and never launches a workload.
- `Tests/AnalyzeResourceQualificationPreflightTests.py`: six offline safety/parser regressions, including path/symlink escape and existing-evidence rejection.
- This Phase 24X report.

Runtime behavior, schema, scheduler admission, ownership and worker implementation are unchanged. Only the preflight/reporting infrastructure was added.

Commands:

```text
python3 -B Tests/AnalyzeResourceQualificationPreflightTests.py        PASS (6 tests)
bash Tests/SchedulerPhasePriorityTests.sh                            PASS
bash Tests/SchedulerAnalyzeWorkerRoutingTests.sh                     PASS
python3 -B Tests/AnalyzeResourceQualificationPreflight.py \
  DerivedData/ExpertAdvisor/Phase24X/Preflight-20261008-01             BLOCKED (exit 2)
python3 -B Tests/AnalyzeResourceQualificationPreflight.py \
  DerivedData/ExpertAdvisor/Phase24X/Preflight-20261008-02             BLOCKED (exit 2)
git diff --check                                                    PASS
```

The pure phase-policy regression compiled C++20 with `-Wall -Wextra -Werror` without diagnostics. No C++ implementation changed, so the full Xcode application build and database/native-worker regressions were not run. No Xcode Clean ran. The initial preflight regression exposed `/var` versus `/private/var` path canonicalization in its temporary fixture; the boundary now compares canonical roots, and all final tests pass. A second path test prevents a symlinked evidence area escaping the checkout.

`git status --short`:

```text
?? Tests/AnalyzeResourceQualificationPreflight.py
?? Tests/AnalyzeResourceQualificationPreflightTests.py
?? docs/phases/Phase24/LSTM_Phase24X_RealAnalyzeResourceQualification_Output.md
```

`git diff --stat`: empty because the three additions remain untracked. Evidence under DerivedData remains ignored and retained. No existing tracked file was changed or staged.

Remaining risks are representative-input completeness, private bootstrap/lifecycle authority, current-schema compatibility, real cleanup validation, measured TRAIN/INFER reserves, concurrent reporting correctness, collector peak accuracy, and stable host contention. Resolve those requirements before implementing or running the real qualification ladder.

## Recovered historical fixture and single worker validation

The resumed checkout remains at `15d9e9762ac9b7ff12e69ff7009cf391b7e8c715` on the requested branch. Both original preflight Python files remain byte-identical. The original report and SHA-256 values were preserved under `DerivedData/ExpertAdvisor/Phase24X/Prerequisites-20261008-01/original-files` before adding this update. Both preflight evidence directories remain intact.

### Historical artifact locations

Searches covered Rollover DerivedData, artifacts, audit/review evidence, the development checkout on `/Volumes/Developer`, the layout5 checkout and DerivedData, adjacent RunArtifacts and ReviewArtifacts, development archives, and canonical build archives. Development and canonical build archives primarily contain source/build provenance and worker artifacts. Phase 24Q's worker logs describe synthetic market inputs and remain excluded from real-data qualification.

The historical production log directory `/Volumes/Developer SSD/ExpertAdvisor/experiment_logs` contains the genuine completed TRAIN/INFER logs. Only historical files were read and copied; no production file was changed, and no production database connection was made. The canonical and semantic build archives supplied retained worker binaries/manifests for read-only provenance verification. Canonical build archives do not themselves supply the required worker logs.

Offline backup investigation found:

| Development backup | Experiments | Models | Inference results | Matching final results with both logs present | Explicit TRAIN and INFER producer links with both logs present |
|---|---|---|---|---|---|
| `Database/backups/LSTM_latest.dump` | 469 | 1,726 | 944 | 374 | 50 |
| `DerivedData/ExpertAdvisor/Phase24R/backup/LSTM-pre-cutover.dump` | 511 | 1,810 | 967 | 395 | 71 |

These counts are candidate inventory, not qualification of every record. Some candidates have unobserved exit codes or older logging contracts and require individual review. The newer archive is 695,096,386 bytes, has migration 099, and matches its retained SHA-256 `df5b4004aa7f8ad102f90acb05e7b455d05aae9ace07bd3e11815d9e5753d393`. No new production dump was taken.

### Verified experiment 746 evidence

Experiment 746 is a completed `cadchfrmp` experiment with horizon 4, threshold 0.0008, 20 epochs, seed 1002, training dates 2010-01-01 through 2025-01-01, and final inference dates 2025-01-01 through 2026-01-01. It uses semantic layout 13, input width 171, and the persisted feature-ablation, objective and immutable calendar identities in the archive.

| Artifact | Verified evidence |
|---|---|
| TRAIN log | `experiment_746_cadchfrmp_train.log`, 8,203 bytes; exactly 20 epoch accuracy records, checkpoint save and final model 2090 save; registration identifies experiment 746, attempt 1391, PID 25189. |
| INFER log | `experiment_746_cadchfrmp_infer.log`, 6,258 bytes; detached materialization, evaluation, result/profitability persistence, committed transaction and success return; registration identifies attempt 1393, model 2090, PID 98003. |
| Final model | Model 2090 belongs to experiment 746, has no parent, and explicitly references producer attempt 1391. |
| Inference result | Result 1023 is completed/final, references model 2090 and producer attempt 1393, and matches symbol, dates, horizon, threshold and 20 completed epochs. Accuracy is 0.5510282005127366. |
| Worker provenance | TRAIN 1391 and INFER 1393 both completed with observed exit 0. PID/group/start, scheduler invocation/fence, command, source commit, artifact SHA, runtime identity, manifest, layout and input width are retained. Both actual archived executable hashes match their persisted SHA values. |
| Model materialization | All 62,563 actual matrix/metadata rows for model 2090 were streamed from the backup unchanged. Parameter dimensions/cardinality, finite values, model width, layout, symbol, training range/configuration, target type, Donchian mode/lookback, warmup scope and canonical objective/hash agree with experiment/inference metadata. |
| Original analysis | Analysis 1161 records accuracy 0.5510282005127366 and leader score 0.4891791698246286, with rejection reason `pred_up_lt_0.15`. It supplies the comparison baseline for private replay. |

The log SHA-256 values are `9b5303672dc9c46dbbf682266c020757bb0500a1290414472fd378fd087a6d0f` for TRAIN and `a3dfd875ce81ebe90500868b80a0e4e2637b658b26195c9f2024afed09097142` for INFER. Private copies are read-only. The inference log intentionally uses the genuine summary/stage logging contract; numerical inference outcomes are persisted in result 1023 rather than fabricated or inserted into the log.

Model 2088 appears as the genuine intermediate checkpoint in the TRAIN log and remains in the offline selected evidence. The final-worker fixture imports final model 2090 and its full materialization; it does not turn the checkpoint into a new result or claim checkpoint-analysis qualification.

### Repeatable snapshot and authoritative replay

`Tests/AnalyzeHistoricalFixtureQualification.py` prepares an exact selective snapshot from the existing archive. It validates the scientific chain before database startup, freezes the original logs, streams only the final model's matrix rows with bounded buffers and a 90-second deadline, retains source SQL, and records the archive/input hashes. It refuses Python optimization that would disable its safety assertions. All output must be a fresh directory under Rollover Phase24X; existing evidence is never overwritten.

The snapshot imports only experiment 746, final model 2090, final inference 1023, original final analysis 1161, the experiment's three terminal historical attempts and their released scheduler invocation. It also restores genuine reference/calendar tables, sequences, migration ledger, global control, phase policy, completed protocol and released lease from the same archive. It excludes active or stopped production attempts, including experiment 747's stopped attempt 1394. Schema pre-data/data/post-data restoration preserves FK constraints and source identities; no experiment, model or result value is invented.

The completed generation-52 protocol and released lease are restored historical workflow records. No fresh cutover receipt is claimed, no host process census is falsified, and no SQL marks an incomplete protocol complete. The native authority service successfully accepted the restored released lease and generated a new private invocation/fence. This resolves the private-bootstrap question for this reviewed snapshot without invoking the host-wide fresh-cutover workflow or stopping production.

`Tests/AnalyzeHistoricalFixtureWorker.py` creates a fresh SCRAM PostgreSQL instance on `127.0.0.1:55489`, with its own data directory, database name, credentials and no Unix socket. Server address, port, directory and system identifier are checked before restoration. Private snapshot owner roles have no login; the disposable runtime role has a new local password. No production credential is copied. Explicit native connection environment and psql flags target the private cluster.

The existing `--requeue-analysis=746 --yes` command calls `ExperimentTransitionService` and authoritatively changes the restored completed experiment to pending/analyze, clearing its stale worker identity. This workflow was present at baseline and permits completed experiments with a model. The initial report's statement that an ANALYZE-only lifecycle workflow had not been established is therefore superseded. No fixture SQL resets experiment state or fabricates an active attempt.

The native scheduler reserves and launches the actual dedicated ANALYZE executable with TRAIN=0, INFER=0, ANALYZE=1. Its semantic registry contains genuine archived routing artifacts and runtime resources, copied privately with their required relative symlinks preserved. No TRAIN/INFER or synthetic worker is executed. Historical PID/group values remain inert terminal provenance; only new private attempts can enter cleanup. The native process census is unfiltered and read-only; production processes cannot be adopted from the selected private database state.

The single-worker stage has a 120-second deadline, checks normal pressure and swap-out growth, and requires exactly one new ANALYZE attempt, observed worker exit 0, completed/done state, cleared active binding, and matching scientific analysis fields. Every scientific analysis column is compared with the original result, including nulls; timestamps may change during replay. Numeric comparison tolerance is 1e-12. Automatic reports and continuation actions are disabled, matching the currently observed production scheduler command's report setting and keeping the single trial limited to final analysis.

Cleanup rechecks native PID/start/group/executable/command identity before signaling only its private scheduler or a verified surviving private worker. The scheduler reaps the successful worker normally. Private postmaster identity is verified immediately before shutdown, including recovery of a partially started owned server if pg_ctl times out. Identity failure preserves the cluster and reports cleanup failure. No production process is signaled.

Reproduction from the repository root, using a new output directory each time:

```text
# Offline preparation only
python3 -B Tests/AnalyzeHistoricalFixtureQualification.py \
  DerivedData/ExpertAdvisor/Phase24X/Fixture746-NEW

# One real managed worker, after the same offline evidence checks pass
python3 -B Tests/AnalyzeHistoricalFixtureQualification.py \
  DerivedData/ExpertAdvisor/Phase24X/Single746-NEW --single-worker
```

The runner requires PostgreSQL 17 tools, the retained Phase24R backup/routing archive, existing qualified development Release/analyzer products, the Native18 read-only process observer, and readable historical experiment 746 logs. Port 55489 must be available. No build, publication, production dump, or TRAIN/INFER rerun is required. The script is deliberately restricted to a complete fresh-training chain with no model ancestry and the inspected managed-inference log contract; it does not silently accept incomplete or incompatible historical evidence.

### Single worker results and retained failures

Offline preparation passed in `Fixture746-20261008-01`. `Single746-20261008-01` restored and requeued the private snapshot, but native scheduler admission rejected the routing copy because ordinary copytree had followed runtime symlinks. It acquired no authority and launched zero workers. Cleanup passed. The source fix uses `symlinks=True`; no runtime manifest or native validation rule was weakened. The original failed evidence and stopped private data directory remain retained.

`Single746-20261008-02` passed real worker execution and cleanup. The final harness added immediate failure detection and complete scientific-field comparison, then passed again in a fresh independent snapshot:

| Final run `Single746-20261008-03` | Result |
|---|---|
| Verified private restore | PASS |
| Cluster system identifier | 7694474641726813541 |
| Private scheduler identity | PID/group 1451, start `1791509485:739269` |
| New worker attempt | 1398, PID/group 1490, start `1791509486:153698` |
| Native ownership | Fresh invocation and fence 199, acquired from restored released lease |
| Real ANALYZE execution | PASS, observed exit 0 |
| Scientific output equality | PASS for every scientific column of analysis 1161 |
| Terminal state | completed/done; active attempt cleared; attempt completed |
| Launch through reconciliation | 1.9003 seconds, including startup and scheduler/observer polling |
| Worker and private server cleanup | PASS |
| Concurrent ANALYZE execution | None |

The 1.9003 seconds is not isolated worker execution time or a throughput estimate. The worker log proves real parsing/persisted-inference analysis and finalization. The native scheduler log records its exit and reconciliation. Signal receipts name only the identity-verified private scheduler; successful ANALYZE exited without a termination signal. Post-shutdown process inspections returning exit 1 are expected absent-process evidence, not failed worker cleanup.

All source snapshots, command logs, SQL restoration inputs, authentic log copies/hashes, process identities, results and stopped private data directories remain under Phase24X. The private databases can be reproduced from the retained snapshot procedure without resetting production or changing source evidence. Production was neither modified nor connected to through PostgreSQL.

### Next validated execution step and remaining limits

**Phase 24X can proceed past the missing-input/private-fixture prerequisite.** The next step is an instrumented single-worker resource baseline using this proven fixture procedure, followed by individual validation of additional genuine candidates representing typical and larger workloads. No TRAIN/INFER evidence needs to be regenerated for experiment 746; read-only reuse of existing artifacts is the least disruptive source of valid input.

Do not treat the other 70 candidate chains as already qualified. Each needs the same exact log/result/model/provenance validation before inclusion in a concurrency cohort. Review log-size distributions and production report settings explicitly. The current 8 KiB/6 KiB summary logs demonstrate a real workload but do not bound larger historical logs, report-population costs, or long-running behavior.

The harness currently proves isolation, genuine inputs, authoritative replay, scientific output and bounded cleanup. It does not yet collect comprehensive per-worker CPU/physical-footprint peaks, GPU counters, database/disk attribution, sustained resource usage or TRAIN/INFER reserves. Short workers require sufficiently fine sampling or owning-parent rusage to avoid missing peaks. Validate that instrumentation at concurrency 1 before moving to 2, then use the original conservative ladder and stop gates. Concurrent execution at 2/4/8/12/18 remains unperformed, and production ANALYZE=0 remains recommended until resource qualification is complete.

### Resumed validation and changed files

New implementation files are `Tests/AnalyzeHistoricalFixtureQualification.py`, `Tests/AnalyzeHistoricalFixtureWorker.py` and `Tests/AnalyzeHistoricalFixtureQualificationTests.py`. Only this report was updated among the original three files; the two original Python files are unchanged. No C++ implementation, schema, native worker, scheduler setting, archive, production log or production artifact was changed.

```text
python3 -B Tests/AnalyzeHistoricalFixtureQualificationTests.py       PASS (9 tests)
python3 -B Tests/AnalyzeResourceQualificationPreflightTests.py       PASS (6 tests)
bash Tests/ExperimentTransitionServiceTests.sh                      PASS
bash Tests/SchedulerPhasePriorityTests.sh                           PASS
bash Tests/SchedulerAnalyzeWorkerRoutingTests.sh                    PASS
python3 -B Tests/AnalyzeHistoricalFixtureQualification.py \
  DerivedData/ExpertAdvisor/Phase24X/Fixture746-20261008-01           PASS, offline only
python3 -B Tests/AnalyzeHistoricalFixtureQualification.py \
  DerivedData/ExpertAdvisor/Phase24X/Single746-20261008-01 \
  --single-worker                                                  FAIL before dispatch; cleanup PASS
python3 -B Tests/AnalyzeHistoricalFixtureQualification.py \
  DerivedData/ExpertAdvisor/Phase24X/Single746-20261008-02 \
  --single-worker                                                  PASS; cleanup PASS
python3 -B Tests/AnalyzeHistoricalFixtureQualification.py \
  DerivedData/ExpertAdvisor/Phase24X/Single746-20261008-03 \
  --single-worker                                                  PASS; cleanup PASS
git diff --check                                                   PASS
```

The nine regressions use the retained genuine corpus and native result; they reject truncated logs, wrong model/dates, missing/failed producers and changed scientific output, and verify snapshot COPY field preservation. They fabricate no qualification logs/results and mutate only in-memory negative-test inputs. The service/priority regressions compile the existing pure C++ implementation; no compiler diagnostics occurred. No full Xcode build or Clean ran because no C++ implementation changed. No commit or push was performed.

Final offline verification also ran `python3 -B Tests/AnalyzeHistoricalFixtureQualificationTests.py DerivedData/ExpertAdvisor/Phase24X/Single746-20261008-03` successfully against the final replay. `Single746-20261008-03/final-verification.json` retains exact validation commands/results, original-file and input-hash checks, and cleanup checks. Its read-only `final-processes.log` confirms no private worker/server remained and the original production scheduler, two TRAIN workers, stopped INFER worker and PostgreSQL PID remained present. Production ANALYZE capacity was still zero. A verification-helper path assumption about the saved original files was corrected (snapshots use basenames); it affected only offline verification, not fixture construction or execution.

Final `git status --short`:

```text
?? Tests/AnalyzeHistoricalFixtureQualification.py
?? Tests/AnalyzeHistoricalFixtureQualificationTests.py
?? Tests/AnalyzeHistoricalFixtureWorker.py
?? Tests/AnalyzeResourceQualificationPreflight.py
?? Tests/AnalyzeResourceQualificationPreflightTests.py
?? docs/phases/Phase24/LSTM_Phase24X_RealAnalyzeResourceQualification_Output.md
```

`git diff --stat` remains empty because these six files are untracked. All pre-existing and new evidence remains retained in ignored DerivedData. The initial historical assessment's status/stat output describes its earlier scope; this resumed output describes the current worktree.


## Instrumented single-worker resource qualification

### Scope, implementation and preservation

The existing branch and baseline remain unchanged. All six pre-existing untracked Phase 24X files were snapshotted with SHA-256 values before editing under `DerivedData/ExpertAdvisor/Phase24X/Instrumentation-Development-20261008-01/original-files`. The two preflight scripts and the historical-fixture regression script remain byte-identical. Original prerequisite and `Single746-20261008-03` evidence remains intact. This section supersedes the preceding statement that comprehensive single-worker instrumentation had not yet been implemented.

`AnalyzeHistoricalFixtureQualification.py` adds an explicit `--instrument-resources` option requiring `--single-worker`. The existing fixture creation, original scientific-field comparison, native requeue workflow, scheduler ownership, launch gate, worker attempt identity and cleanup remain authoritative. `AnalyzeHistoricalFixtureWorker.py` starts the recorder before scheduler/worker dispatch, checks recorder health during execution, keeps sampling through cleanup, and fails the command if coverage is inadequate even after successful scientific replay. The default uninstrumented option remains available for prerequisite validation.

Three new files implement measurements and regressions:

- `Tests/AnalyzeResourceInstrumentation.py`: bounded private resource observer, fixed requested cadence, raw JSONL samples, coverage gates and machine-readable summary.
- `Tests/AnalyzeResourceProbe.cpp`: test-only Darwin process/host API bridge and private owning-parent exit accounting.
- `Tests/AnalyzeResourceInstrumentationTests.py`: offline counter, identity, coverage, private-destination and census regressions, plus optional replay of retained real measurements.

The C++ probe is compiled privately with C++20, `-O2 -Wall -Wextra -Werror -dynamiclib`. Only the private scheduler launch environment receives `DYLD_INSERT_LIBRARIES`. Its fork and waitpid observations are restricted to the exact private scheduler executable; waitpid delegates to wait4 with the same PID, status and flags. Native launch/reconciliation policy is unchanged. The child receives only a PostgreSQL application-name label for connection attribution before its normal exec. No worker executable bytes, manifest, historical log, persisted scientific input, production setting or production artifact are modified. Probe binary/source hashes and the launch environment are retained.

The owning-parent wait4 receipt supplies kernel peak RSS and total user/system CPU through child exit, including startup and shutdown. This avoids estimating peak memory from polling. Darwin reports ru_maxrss in bytes; wait4 preserves child status semantics and returns child resource accounting. Sources: [Apple getrusage manual](https://github.com/apple/darwin-xnu/blob/main/bsd/man/man2/getrusage.2), [Apple wait4 manual](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/wait4.2.html).

Process samples use proc_pid_rusage v2 for RSS, physical footprint, CPU counters and disk bytes, with native BSD start identity checked before and after each read. Mach CPU ticks are converted using mach_timebase_info; raw ticks and numerator/denominator are retained and checked against exit CPU accounting. The kernel supplies these CPU counters from task timing: [Apple XNU fill_task_rusage](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/bsd_kern.c). PostgreSQL children are enumerated with proc_listchildpids, whose return value is a PID count, not a byte count: [Apple libproc implementation](https://github.com/apple-oss-distributions/xnu/blob/main/libsyscall/wrappers/libproc/libproc.c).

Host samples record pressure, swap/page-out counters, page size and existing swap use. PostgreSQL samples include the private postmaster and its children, with separate process identities, RSS/footprint/CPU/disk counters. A single persistent private observer connection queries pg_stat_activity and records total, worker, scheduler and observer connections. Explicit loopback host/address, port, database and private passfile parameters plus structured destination verification prevent fallback to production. Inherited libpq defaults are cleared in the disposable harness process. No production PostgreSQL connection is opened.

The measurement window starts after private fixture restoration but before scheduler status, requeue and dispatch, and ends after worker exit, scheduler reconciliation and private scheduler/server shutdown. It measures all worker startup and shutdown. PostgreSQL bootstrap/restoration is outside the window; the later server shutdown checkpoint can flush restored buffers and is explicitly included in PostgreSQL disk measurements.

Requested cadence is 5 ms; gaps above 50 ms fail coverage. Missing owning-parent accounting, fewer than two live worker samples, no sample of the actual ANALYZE executable, wrong native identity, missing startup/shutdown coverage, inadequate worker-window PostgreSQL connection observations, missing host/server counters, unsafe pressure or any new swap-out activity also fail. RSS high-water and exit CPU must exist independently of sample count. Worker duration is reported as conservative observed bounds, not as scheduler reconciliation time. Coverage counts and short-lived PostgreSQL read misses are retained. No artificial sleep or workload extension is inserted into ANALYZE.

### Retained setup failures and pilot audit

Three fresh instrumented setups stopped before worker dispatch:

1. `Instrumented746-20261008-01`: libpq installation-path assumption failed. The harness now resolves the library using pg_config before PostgreSQL startup.
2. `Instrumented746-20261008-02`: an empty service parameter selected a nonexistent libpq service. The diagnostic is retained; explicit private parameters and cleared inherited defaults replace it.
3. `Instrumented746-20261008-03`: an overly literal textual inet identity comparison failed. Structured JSON fields with host(inet_server_addr()) now verify the destination; offline tests reject wrong ports/directories.

Each private cluster stopped cleanly, and all evidence remains retained. The native probe's first sandboxed self-check could not read a required host counter; the approved bounded self-check passed. It also verified Mach-time conversion. These self-checks are instrumentation tests, not scientific workload qualification.

`Instrumented746-20261008-04` ran a real worker successfully and retained a provisional PASS summary. Audit found that PostgreSQL enumeration treated a PID count as bytes, undercounting server processes, and that requested-cadence drift needed clearer accounting. Its original files are unchanged. `resource-audit-correction.json` explicitly rejects that pilot as the final resource qualification. The corrected census, fixed cadence and regression tests were validated in a fresh final run. No failed or provisional result was discarded or silently overwritten.

### Accepted final result

Final evidence directory: `DerivedData/ExpertAdvisor/Phase24X/Instrumented746-20261008-05`.

| Measurement / check | Final result |
|---|---|
| Scientific fixture | Authentic experiment 746, model 2090, inference result 1023 |
| Scientific output | Every scientific column matches original analysis 1161; no comparison relaxed |
| Private cluster | 127.0.0.1:55489, private pgdata, system ID 7694500784352112980 |
| Private scheduler | PID/group 16792, native start 1791515572:441986 |
| Real ANALYZE | Attempt 1398, PID/group 16826, native start 1791515572:802869; exit 0 |
| Kernel peak worker RSS | 11,321,344 bytes = **10.796875 MiB** |
| Sampled maximum worker RSS | 11,239,424 bytes = 10.718750 MiB |
| Sampled peak worker physical footprint | 3,555,808 bytes = 3.391083 MiB |
| Worker user / system CPU time | 0.021021 / 0.006978 seconds |
| Worker total CPU time | **0.027999 seconds** |
| Startup-through-shutdown duration | **0.116088–0.121276 seconds**; uncertainty 5.188 ms |
| Average worker CPU utilization bounds | 23.087–24.119% of one CPU core |
| Sampled worker peak CPU utilization | 98.620% of one CPU core |
| Scheduler launch through reconciliation | 1.924329 seconds; distinct from worker execution duration |
| Window | 2026-10-09 03:12:52.137704–03:12:54.608639 UTC (October 8 CDT) |
| Requested / median interval | 5 ms / 5.001708 ms |
| Window samples / nominal missed intervals | 488 samples, 7 nominal intervals missed; 98.586% nominal coverage |
| Maximum observed window gap | 10.325125 ms, below 50 ms limit |
| Live worker / actual ANALYZE executable samples | 22 / 19; remaining startup samples cover fork-to-exec |
| Maximum gap between actual ANALYZE samples | 7.216834 ms |
| PostgreSQL connection observations in worker window | 25 |
| Peak worker connections | 1 |
| Peak total clients / excluding observer | 3 / 2, observed rather than theoretical connection maxima |
| Observer connections | 1; explicitly included in total and excluded from adjusted total |
| Peak sampled private PostgreSQL process RSS sum | 106,512,384 bytes = 101.578125 MiB |
| Peak sampled private PostgreSQL footprint sum | 69,718,192 bytes = 66.488449 MiB |
| PostgreSQL process identities sampled | 61 across the whole window, including transient harness/scheduler backends |
| Missed short-lived PostgreSQL reads | 16 read misses across 15 PIDs; 0 observed backend PIDs entirely unsampled |
| Pressure | Normal level 1 throughout |
| Swap-in / swap-out / page-out deltas | 0 / 0 / 0 pages; swap-used change 0 bytes |
| Pre-existing host swap | 15.6745 GiB; this run did not add swap activity |
| Worker sampled disk counters | 16,384 bytes read, 0 bytes written at last observed high-water; sampled lower bounds |
| Worker exit-accounted block counters | 0 input / 0 output blocks; these zeros do not establish absence of I/O |
| Private PostgreSQL sampled cumulative disk lower bounds | 1,867,776 bytes read / 25,411,584 bytes written; can include pre-window work |
| Private PostgreSQL sampled cumulative CPU lower bound | 0.434969 seconds across observed process lifetimes |
| Private PostgreSQL observed window deltas | 0 bytes read / 24,518,656 bytes written / 0.384449 seconds CPU, all lower bounds; includes shutdown checkpoint and observer/harness work |
| Scientific worker, resources and cleanup | **PASS / PASS / PASS** |
| Concurrent ANALYZE execution | None |

PostgreSQL RSS sums can double-count shared pages and are not exclusive memory use. Its per-process CPU/disk counters are cumulative sampled lower bounds, not complete child-exit totals. For server processes born before the window, cumulative counters can include bootstrap/restoration; `resource-window-attribution.json` subtracts first observed counters for those processes and uses zero baseline for processes born during the window. Its window deltas remain lower bounds. The private database numbers include the observer, scheduler/harness sessions and server shutdown work; they must not be assigned entirely to ANALYZE. Worker byte-I/O counters may miss final unsampled activity; kernel block counters and their limitations are retained separately. CPU utilization uses one core as 100%, allowing values above 100% for multithreaded processes. No GPU utilization was collected: this final ANALYZE path parses logs and consumes persisted inference outcomes rather than executing a Metal model.

`resource-samples.jsonl` contains raw timestamped host, worker, server, connection and interval observations. `resource-lifecycle.jsonl` contains fork identity and owning-parent exit RSS/CPU/status receipts. `resource-summary.json` is the machine-readable qualification result. `resource-coverage-audit.json` independently checks original-file/input hashes, source snapshots, coverage and post-run cleanup. `final-verification.json` records the exact offline/native-policy validation commands and passing CPU-unit, full-census and summary-recomputation regressions. `commands.jsonl`, native worker/scheduler logs, `scheduler-process.json`, `owned-attempts.json`, `signals.jsonl` and `final-processes.log` retain commands, identities and cleanup evidence. The stopped private cluster and original historical SQL/log copies remain retained.

The successful worker exited without a termination signal. Signal receipts identify only private scheduler 16792 after an exact native identity check. Private postmaster identity was checked before pg_ctl shutdown; its PID file disappeared. The sampler stopped and the final process census contains no private scheduler, worker or PostgreSQL process. The original production scheduler 55976, PostgreSQL 3876 and stopped INFER 52730 remained present. Original TRAIN PIDs 57072/57078 were absent in the later read-only census; their lifecycle was not queried through production PostgreSQL, and no production signal was issued. A constant two-TRAIN background load was therefore not established.

### Reproduction and validation

Run from the repository root using a fresh output directory:

```text
python3 -B Tests/AnalyzeHistoricalFixtureQualification.py \
  DerivedData/ExpertAdvisor/Phase24X/Instrumented746-NEW \
  --single-worker --instrument-resources
```

Prerequisites remain the retained archive/logs/routing evidence, PostgreSQL 17 tools, existing qualified development products and native identity observer. The probe additionally requires clang++ and permission to read Darwin host/process counters. Automatic approval granted the bounded private runs; no user permission to change production was sought or used. Existing evidence directories are rejected rather than overwritten. The worker deadline remains 120 seconds; PostgreSQL query statement timeout is 100 ms and connect timeout 5 seconds.

Final validation commands:

```text
python3 -B Tests/AnalyzeResourceInstrumentationTests.py \
  DerivedData/ExpertAdvisor/Phase24X/Instrumented746-20261008-05      PASS (20 tests)
python3 -B Tests/AnalyzeHistoricalFixtureQualificationTests.py \
  DerivedData/ExpertAdvisor/Phase24X/Instrumented746-20261008-05      PASS (9 tests)
python3 -B Tests/AnalyzeResourceQualificationPreflightTests.py      PASS (6 tests)
bash Tests/ExperimentTransitionServiceTests.sh                    PASS
bash Tests/SchedulerPhasePriorityTests.sh                         PASS
bash Tests/SchedulerAnalyzeWorkerRoutingTests.sh                  PASS
/usr/bin/clang++ -std=c++20 -O2 -Wall -Wextra -Werror -dynamiclib \
  Tests/AnalyzeResourceProbe.cpp -o DerivedData/ExpertAdvisor/Phase24X/Instrumented746-20261008-05/resource-probe.dylib
                                                               PASS; no diagnostics
python3 -B Tests/AnalyzeHistoricalFixtureQualification.py \
  DerivedData/ExpertAdvisor/Phase24X/Instrumented746-20261008-05 \
  --single-worker --instrument-resources                         PASS
```

The standalone offline suite has 19 in-memory regressions and one optional retained-native-evidence test. Passing the final evidence directory runs all 20, including exact recomputation of the saved summary, Mach-time conversion, census completeness and CPU-time consistency with owning-parent accounting. The scientific comparison regressions also pass against the final instrumented result. Native policy/service regressions and the probe compile without warnings. No full application Xcode build or Clean ran: no production C++ implementation changed. No commit, push or publication was performed.

### Concurrency 2 recommendation

**Not ready to launch concurrency 2 yet.** The measurement prerequisite at concurrency 1 is now satisfied. The next step is offline selection and full validation of a second genuine completed TRAIN/INFER chain, then extend the collector/coverage gates to an explicitly bounded two-worker private cohort with separate identities, exit receipts and connection attribution. This harness intentionally rejects more than one scheduler child and cannot qualify concurrency 2 unchanged.

Retain pressure/swap stop gates and establish conservative TRAIN/INFER reserve estimates without controlling production. This short, small-log fixture demonstrates correctness and reliable process measurement; it does not demonstrate sustained load, larger-log/report behavior, representative fleet diversity or capacity under two worst-case TRAIN workers plus INFER. Observer overhead and sampled PostgreSQL/disk attribution must remain explicit. No production capacity above zero, numeric TRAIN/INFER headroom or practicality of 18 real workers is qualified. Production TRAIN=2, INFER=1, ANALYZE=0 remains the recommendation.

### Current worktree review output

Updated existing files: `Tests/AnalyzeHistoricalFixtureQualification.py`, `Tests/AnalyzeHistoricalFixtureWorker.py`, and this report. Added files: `Tests/AnalyzeResourceInstrumentation.py`, `Tests/AnalyzeResourceProbe.cpp`, `Tests/AnalyzeResourceInstrumentationTests.py`. Three other pre-existing Phase 24X scripts remain unchanged, and snapshots preserve all six original files.

`git status --short`:

```text
?? Tests/AnalyzeHistoricalFixtureQualification.py
?? Tests/AnalyzeHistoricalFixtureQualificationTests.py
?? Tests/AnalyzeHistoricalFixtureWorker.py
?? Tests/AnalyzeResourceInstrumentation.py
?? Tests/AnalyzeResourceInstrumentationTests.py
?? Tests/AnalyzeResourceProbe.cpp
?? Tests/AnalyzeResourceQualificationPreflight.py
?? Tests/AnalyzeResourceQualificationPreflightTests.py
?? docs/phases/Phase24/LSTM_Phase24X_RealAnalyzeResourceQualification_Output.md
```

`git diff --stat`: empty because all nine Phase 24X files remain untracked. Evidence under DerivedData remains ignored and preserved. No tracked application file, production repository or production scheduler setting changed.

## Second authentic historical workload qualification

This increment selected and independently qualified **experiment 681, model 1982, final inference result 979, original final analysis 1117**. Exactly one real ANALYZE worker executed for this second fixture. Scientific equality, resource instrumentation and identity-safe cleanup all passed. The requested branch and baseline remain unchanged; no production database connection, write, scheduler change or signal occurred. No concurrent ANALYZE workers ran.

### Selection and complete historical provenance

The existing migration-099 offline archive remains `DerivedData/ExpertAdvisor/Phase24R/backup/LSTM-pre-cutover.dump`, SHA-256 `df5b4004aa7f8ad102f90acb05e7b455d05aae9ace07bd3e11815d9e5753d393`. The retained historical table extracts were screened again without contacting a live database. Of 71 candidates with explicit producer links and both logs, 57 candidates other than 746 passed the initial chain/artifact screening. Experiment 681 had the largest combined logs among these initially passing candidates: **14,833 bytes**, versus 14,461 for 746, an increase of 372 bytes or **2.572%**. The inventory and selection receipt are retained in `DerivedData/ExpertAdvisor/Phase24X/SecondFixture-Investigation-20261008-01/candidate-review.json` and `selected-681-evidence.json`. Initial screening is not full qualification of the other candidates.

The increased log size is modest. This fixture adds a layout-8, width-80 feature-ablation case and an accepted inference outcome; it does not establish a heavy or sustained analysis workload. The original historical evidence and the previously qualified 746 evidence were preserved. All nine original untracked Phase 24X files were snapshotted with their full relative paths and SHA-256 hashes under `SecondFixture-Investigation-20261008-01/original-files` before changes.

| Evidence | Verified historical value |
|---|---|
| Experiment / model / final inference / analysis | 681 / 1982 / 979 / 1117 |
| Symbol / horizon / threshold / epochs / seed | cadchfrmp / 4 / 0.0008 / 20 / 46 |
| TRAIN range | 2010-01-01 through 2025-01-01 |
| INFER range | 2025-01-01 through 2026-01-01 |
| Input layout / width / hidden size | 8 / 80 / 64 |
| Feature ablation | tg4_inner_break_any,tg4_source_tg3_structurally_eligible,tg4_source_tg3_confluent; persisted and effective masks match the authentic TRAIN diagnostic |
| Objective hash | fnv1a64:65818f2e1fa1a324 |
| Immutable calendar snapshot / hash | 1 / fnv1a64:67610f94f5c8e7cc |
| Model materialization | 39,267 original matrix rows; complete shapes/cardinality, finite values, unique indices and metadata checked |
| TRAIN attempt / PID / native start / fence | 1228 / 25602 / 1790675819:817785 / 174 |
| INFER attempt / PID / native start / fence | 1238 / 26046 / 1790709821:867458 / 175 |
| Original ANALYZE attempt | 1240, completed, exit 0 |
| TRAIN and INFER lifecycle | Both completed; parent-observed exit 0; registered producer identities bind the original model and inference result |
| Persisted final inference | Completed, accepted=true, reason=none; accuracy 0.5463917525773195 |
| Expected final analysis | Accept count 1, reject count 0, leader score 0.48442980125411833 |

Both TRAIN/INFER owner invocations later ended with status `crashed`; the successful producer attempt exit observations precede those owner failures. Their status was preserved, not converted to a released or successful invocation. The fixture includes the original ended owners and the additional original owner referenced by the released global lease. No live production binding or active attempt was restored.

The authentic logs are read-only private copies (mode 0400). Source and private hashes were checked again after execution:

```text
experiment_681_cadchfrmp_train.log  8578 bytes
SHA-256 fd32004f2b98641b7cee92607a7172e418dbaf013f11ccd8851c7cc8fa53fa83
experiment_681_cadchfrmp_infer.log  6255 bytes
SHA-256 6d6840ffd3aa47daeb52ad03e91ca4f0dac513956e904cc1ed54d85ff413a5b7
```

The TRAIN log contains all 20 epoch diagnostics and final model save 1982. The INFER log binds model 1982 and includes successful evaluation, profitability-result persistence commit and application success. Logs, persisted rows, seed command, model semantics/objective, calendar, producer registration and final-analysis source binding agree. Historical null scientific fields remain null; missing metrics were not fabricated.

Original semantic-worker archives and actual executable hashes were verified read-only, together with source commits, manifests, capabilities, input layout/width and runtime identity:

```text
TRAIN source 59697166dfd2f097f605062515fec4f5ba54e282
TRAIN executable SHA-256 0c9039aac96eb584c5b1d4fc5eb306aeef3af8567fbc8195607c7f6d2500fec1
INFER source cf55915ef43782ad59798bec314cf357e86ced20
INFER executable SHA-256 f6167fcd5bdf30b76f90bce7d52e811668b21b118e6f7a2fec481c80fbcb25cc
Runtime identity 6c8d208a0aae281f3fbb1a2a38fe2c7d6deb92811b4a41d3615fac67defee34e
```

These archives are available under the original `ExpertAdvisor/Builds/SemanticWorkers/layout8` TRAIN and INFER paths. The private routing registry retains the genuine two current-layout-13 routes unchanged and adds the exact two layout-8 producer routes from the retained development publication archive. Source commit, executable hash, role, layout, width and runtime identity must all match; an available substitute cannot satisfy this check. Copied executables are rehashed. TRAIN and INFER dispatch remain disabled; these artifacts establish provenance/admission compatibility and were not executed in qualification.

### Restore dependency correction and authoritative replay

Offline preparation passed under `Fixture681-20261008-01` before any private worker was launched. The first private attempt, `Instrumented681-20261008-01`, stopped during post-data restoration: the released global lease referenced an original ended owner invocation absent from the experiment-only subset. PostgreSQL enforced the original foreign key. **Zero scheduler/ANALYZE workers launched in that attempt**, and identity-safe PostgreSQL cleanup passed. Its SQL, error output, results and stopped cluster remain retained.

The fixture builder now closes invocation dependencies for producer attempts, their observer/reconciler references and the released lease owner, requiring every included invocation to be ended. This adds the original missing invocation without rewriting ownership state or weakening constraints. The second fixture restores three original terminal invocations. Two offline regressions cover inclusion of the genuine lease owner and rejection when it is unavailable. The generic fixture builder also validates the original analysis/inference binding and authentic logged semantic/objective/calendar/seed identities; it compares logged effective ablation when that historical logging contract emits the event. The selected 681 fixture does emit it, and an altered mask is rejected. Existing 746 regression evidence still passes without fabricating an absent historical diagnostic.

The fresh final directory is **`DerivedData/ExpertAdvisor/Phase24X/Instrumented681-20261008-02`**. It restored the original generation-52 complete protocol, released lease, scientific rows, matrices and sequences, then used the existing authoritative `--requeue-analysis=681 --yes` workflow. The private scheduler acquired fresh ownership and dispatched one ANALYZE attempt with TRAIN=0, INFER=0, ANALYZE=1. No direct SQL lifecycle rewrite created work.

| Scientific / isolation check | Final result |
|---|---|
| Private PostgreSQL destination | 127.0.0.1:55489; own credentials, no Unix socket, own pgdata |
| Private system identifier | 7694507271438819468 |
| Private database | ea_phase24x_historical_lstm; full schema constraints restored |
| Private scheduler identity | PID/group 20728; native start 1791517082:938641 |
| Real ANALYZE attempt / PID / group / native start | 1398 / 20767 / 20767 / 1791517083:314653 |
| Real ANALYZE executable | Private copied lstm-analyze-worker; SHA-256 8f6cc57c8775d4d89055eff1ce03da705f65ff742939154b0aabd4ee3773be74 |
| Worker completion | Exit 0 observed by owning scheduler, completed attempt, completed/done experiment, active binding cleared |
| Scientific equality | **All 41 non-timestamp columns match original analysis 1117**, including every value and null; numeric tolerance remains 1e-12 |
| Inference / accept accuracy | 0.5463917525773195 / 0.5463917525773195 |
| Accept count / rate / reject count | 1 / 1 / 0 |
| Leader score / notes | 0.48442980125411833 / none |
| Worker / resources / cleanup | **PASS / PASS / PASS** |

### Second-fixture resource results

The same native probe, owning-parent fork/wait4 receipts, Darwin sampler, private PostgreSQL observer and single-worker adequacy gates used for 746 were reused unchanged. The sampler/probe and their existing tests remain byte-identical to the original snapshots. Measurement includes worker startup through disappearance and private PostgreSQL shutdown; restoration/bootstrap precedes the measurement window. The 120-second worker deadline, normal-pressure/no-new-swap-out gates, 5 ms requested interval and 50 ms maximum coverage gap remain unchanged.

| Measurement | Experiment 681 result |
|---|---|
| Kernel lifetime peak worker RSS | 11,354,112 bytes = **10.828125 MiB** |
| Sampled maximum worker RSS | 11,255,808 bytes = 10.734375 MiB |
| Sampled peak worker physical footprint | 3,572,168 bytes = 3.406685 MiB |
| Worker user / system / total CPU | 0.020898 / 0.005244 / **0.026142 seconds** |
| Startup-through-shutdown duration bounds | **0.141856–0.146972 seconds**; 5.116 ms uncertainty |
| Average CPU utilization bounds | 17.787–18.429% of one core |
| Sampled peak CPU utilization | 99.995% of one core |
| Scheduler launch through reconciliation | 1.882215 seconds; distinct from worker execution duration |
| Window | 2026-10-09 03:38:02.595245–03:38:05.061174 UTC (October 8 CDT) |
| Requested / median sampling interval | 5 ms / 4.999292 ms |
| Samples / nominal missed intervals / coverage | 490 / 4 / **99.1903%**; three gaps skipped a complete interval |
| Maximum observed window gap | 10.502792 ms; below 50 ms limit |
| Live worker / actual ANALYZE executable samples | 28 / 26; two startup samples precede exec |
| Maximum actual ANALYZE sample gap | 7.073208 ms |
| Worker-window PostgreSQL observations | 30 |
| Peak worker connections | 1 |
| Peak total clients / excluding observer | **2 / 1**; observed, not an upper bound on all possible transient overlap |
| Peak scheduler / observer clients | 1 / 1; per-role peaks do not necessarily coincide |
| Peak sampled private PostgreSQL RSS sum | 113,016,832 bytes = **107.781250 MiB** |
| Peak sampled private PostgreSQL footprint sum | 74,502,512 bytes = 71.051132 MiB |
| Sampled PostgreSQL process identities | 61 across the complete window |
| Missed short-lived PostgreSQL reads | 11 reads across 10 PIDs; zero observed backend PIDs entirely unsampled |
| Private PostgreSQL sampled cumulative CPU | 0.416290 seconds, lower bound; can include pre-window work |
| Private PostgreSQL observed window CPU | 0.365167 seconds, lower bound |
| Private PostgreSQL cumulative / window disk bytes | Read 0 / 0; write 25,280,512 / 24,731,648; sampled lower bounds |
| Worker sampled disk bytes / exit block counters | 4,096 read, 0 written / 0 input and 0 output blocks; not proof of no I/O |
| Pressure | Normal kernel level 1 throughout |
| Swap-in / swap-out / page-out activity | **0 / 0 / 0 pages**; swap-used change 0 bytes |
| Pre-existing swap use | 16,830,365,696 bytes; no swap growth observed in this window |
| GPU/Metal utilization | Not collected; this final-analysis path does not execute a Metal model |

As for 746, RSS sums may double-count shared PostgreSQL pages. PostgreSQL measurements include observer/harness/scheduler sessions and shutdown checkpoint work; they cannot all be attributed to the worker. Counter deltas subtract initial observations for server processes born before the window; missing final transient-process observations make the results lower bounds. Filesystem caching and short duration limit disk attribution and sustained-memory conclusions. Nominal coverage is sample count divided by sample count plus skipped nominal intervals, not a claim that all process resource peaks were sampled; worker lifetime RSS/CPU additionally come from owning-parent wait4.

Raw evidence includes `resource-samples.jsonl`, `resource-lifecycle.jsonl`, `resource-summary.json`, `resource-coverage-audit.json`, `resource-window-attribution.json`, `single-worker-results.json`, exact-command `commands.jsonl`, identities, signals and worker/scheduler logs. `offline-final-audit.py` preserves the independent read-only recomputation procedure. `final-verification.json` records exact regression command arrays, outputs, return codes and timing.

### Cleanup and preservation

The worker exited normally without a termination signal. The sole signal receipt is SIGTERM to private scheduler 20728 after exact PID/group/start/executable/command revalidation. The private postmaster 20635 was similarly revalidated before `pg_ctl -D <private-pgdata> stop -m fast`; the subsequent identity observation and final process census establish its absence. All 61 sampled private PostgreSQL identities, the private scheduler and worker are absent from the final census. Both 681 attempt directories and accepted 746 evidence have no `postmaster.pid`. No sampler or private database process remains.

The archive, original logs, private historical input hashes, all nine original file snapshots, and final harness source snapshots passed independent verification. Six original scripts/probe files remain unchanged; only the fixture builder, private worker wrapper and report were updated. Historical model/inference rows and logs were not edited. The stopped private clusters and failed-restore evidence remain retained. Production activity continued naturally; a fixed two-TRAIN/one-INFER background workload or worst-case resource reserve was not demonstrated.

### Reproduction, regressions and current review output

Use a new, nonexistent evidence directory for replay; the runner rejects existing paths:

```text
python3 -B Tests/AnalyzeHistoricalFixtureQualification.py \
  DerivedData/ExpertAdvisor/Phase24X/Instrumented681-NEW \
  --experiment=681 --single-worker --instrument-resources
```

The complete historical archive, read-only logs and semantic artifacts, retained development routing archive, existing development binaries, PostgreSQL 17 tools and native identity observer are prerequisites. No historical TRAIN/INFER computation is regenerated. Scientific meaning comes from the immutable original rows/logs and exact producer provenance. Scientific comparison remains unchanged; timestamp differences alone are excluded.

Commands executed successfully for the accepted run and final validation:

```text
python3 -B Tests/AnalyzeHistoricalFixtureQualification.py \
  DerivedData/ExpertAdvisor/Phase24X/Fixture681-20261008-01 \
  --experiment=681                                                 PASS (offline preparation)
python3 -B Tests/AnalyzeHistoricalFixtureQualification.py \
  DerivedData/ExpertAdvisor/Phase24X/Instrumented681-20261008-02 \
  --experiment=681 --single-worker --instrument-resources           PASS (exactly one real worker)
python3 -B Tests/AnalyzeSecondHistoricalFixtureQualificationTests.py \
  DerivedData/ExpertAdvisor/Phase24X/Instrumented681-20261008-02      PASS (17 tests)
python3 -B Tests/AnalyzeHistoricalFixtureQualificationTests.py \
  DerivedData/ExpertAdvisor/Phase24X/Instrumented681-20261008-02      PASS (9 tests)
python3 -B Tests/AnalyzeHistoricalFixtureQualificationTests.py \
  DerivedData/ExpertAdvisor/Phase24X/Instrumented746-20261008-05      PASS (9 tests)
python3 -B Tests/AnalyzeResourceInstrumentationTests.py \
  DerivedData/ExpertAdvisor/Phase24X/Instrumented681-20261008-02      PASS (20 tests)
python3 -B Tests/AnalyzeResourceQualificationPreflightTests.py      PASS (6 tests)
bash Tests/SchedulerSemanticAdmissionTests.sh                      PASS
bash Tests/ExperimentTransitionServiceTests.sh                     PASS
bash Tests/SchedulerPhasePriorityTests.sh                          PASS
bash Tests/SchedulerAnalyzeWorkerRoutingTests.sh                   PASS
python3 -B DerivedData/ExpertAdvisor/Phase24X/Instrumented681-20261008-02/offline-final-audit.py
                                                                  PASS
/usr/bin/clang++ -std=c++20 -O2 -Wall -Wextra -Werror -dynamiclib \
  Tests/AnalyzeResourceProbe.cpp \
  -o DerivedData/ExpertAdvisor/Phase24X/Instrumented681-20261008-02/resource-probe.dylib
                                                                  PASS; no diagnostics
git diff --check                                                  PASS
```

The 17 new regressions cover genuine layout-8 provenance and the accepted result, exact current/historical routing selection, rejection of wrong/missing provenance/runtime, analysis-source mismatch, changed ablation/semantic/calendar/seed evidence, incomplete persistence, original lease-owner dependency closure and every native scientific output field. Negative tests alter only in-memory copies. Resource regressions recompute the retained native summary and exercise counter units, lifecycle coverage, PostgreSQL census and failure gates. No tracked application C++ changed, so no full Xcode application build or Clean was necessary or run. Probe and relevant native policy/service tests compile without warnings.

Files changed in this increment:

- `Tests/AnalyzeHistoricalFixtureQualification.py`: exact producer routing helper, original analysis/log-chain validation, terminal invocation dependency closure and explicit producer mapping in receipts.
- `Tests/AnalyzeHistoricalFixtureWorker.py`: select and rehash the exact genuine historical producer routes alongside preserved current routes; keep single-worker isolation/measurement/cleanup.
- `Tests/AnalyzeSecondHistoricalFixtureQualificationTests.py`: new offline and retained-native-evidence regressions.
- This report: second-fixture selection, measurements, cleanup and next-step assessment.

`git status --short`:

```text
?? Tests/AnalyzeHistoricalFixtureQualification.py
?? Tests/AnalyzeHistoricalFixtureQualificationTests.py
?? Tests/AnalyzeHistoricalFixtureWorker.py
?? Tests/AnalyzeResourceInstrumentation.py
?? Tests/AnalyzeResourceInstrumentationTests.py
?? Tests/AnalyzeResourceProbe.cpp
?? Tests/AnalyzeResourceQualificationPreflight.py
?? Tests/AnalyzeResourceQualificationPreflightTests.py
?? Tests/AnalyzeSecondHistoricalFixtureQualificationTests.py
?? docs/phases/Phase24/LSTM_Phase24X_RealAnalyzeResourceQualification_Output.md
```

`git diff --stat`: empty, because all ten Phase 24X files remain untracked. No tracked source file changed. All nine prior files and existing evidence were preserved, with original snapshots retained. No commit, push or publication occurred.

### Whether concurrency 2 is ready

**The second-fixture prerequisite is satisfied. Phase 24X can proceed to implementing and validating a bounded concurrency-2 harness.** The current harness deliberately rejects more than one child; it is not ready to launch two workers unchanged. The next validated execution step is offline two-fixture snapshot/routing dependency closure and per-worker instrumented identity/exit/connection attribution, followed by a fresh private destination and resource-headroom preflight. Only after those gates pass should exactly two real workers be dispatched, with actual overlap demonstrated rather than inferred from configured capacity.

Retain conservative pressure/swap/deadline stops and identity-safe cleanup. Neither the slightly larger logs nor two independent short runs establish sustained contention, worst-case TRAIN/INFER reserves, concurrent report behavior, numeric production headroom or practicality of 18 workers. Automatic reports remain disabled in these qualifications. No additional scheduler admission change is established by this increment. Production configuration remains TRAIN=2, INFER=1, ANALYZE=0; concurrency 2 and all higher levels are **NOT RUN**.
