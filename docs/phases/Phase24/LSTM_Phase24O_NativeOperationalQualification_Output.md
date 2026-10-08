# Phase 24O — Isolated native operational qualification

**NO-GO for a future controlled production cutover. Operational qualification is blocked at scheduler protocol initialization.** Database isolation, native identities, isolated runtime connectivity, pending-protocol rejection, and negative native worker admission were verified. Successful native TRAIN, masked TRAIN, checkpoint continuation, and checkpoint-backed INFER were **not** executed and are **not** passing results.

The supported protocol-cutover CLI requires absence of scheduler dispatch processes across the host. Production scheduler PID **30886** remains active and its command contains `--schedule-experiments`. Completing that CLI's precondition would conflict with the absolute production-protection requirement. No production process was interrupted, no protocol state was forced through SQL, and no experiment or attempt was fabricated to bypass admission. This is an operational blocker on this shared host, not evidence that the qualified workers cannot train or infer.

## Baseline and artifact evidence

Qualification date: **2026-10-08**. Repository: `/Volumes/Developer SSD/ExpertAdvisor-Rollover`.

| Check | Result |
| --- | --- |
| Branch | `dedicated-train-layout-rollover-squashed-v1` |
| Initial and final HEAD | `3f59667acbaa997a71351ea415a5a40bbea6d795`, matching expected `3f59667a` |
| Initial working tree | Clean; `git status --short` empty |
| Required reading | Root `AGENTS.md`, Phase 24M and Phase 24N reports |
| Delta from TRAIN source commit | `git diff --stat 12e31c02 HEAD`: only the Phase 24N report, 254 inserted lines |
| Native host | macOS 27.0.1, build `26A434` |
| Installed Xcode | 27.0, build `27A266a`; preserved, despite historical AGENTS.md wording |
| Compiler and dependencies | Existing Apple `/usr/bin/clang++` and Homebrew paths preserved; no installation, replacement, relinking, or compiler switch |
| Native products | Existing stable `DerivedData/ExpertAdvisor/Build/Products/Release` artifacts reused |

Both executable identities were queried natively with `--build-identity` from `/`. Each returned exit **0**, version **1**, the correct dedicated role, semantic layout **13**, model input width **171**, its canonical Rollover path, and its independently verified SHA-256. `file` reports thin ARM64 Mach-O for both. `otool -l` reports minimum macOS **27.0** and SDK **27.0**. Strict `codesign --verify --strict` passes for both.

| Artifact | Embedded source commit | SHA-256 |
| --- | --- | --- |
| `lstm-train-worker` | `12e31c0299193e4198907701182ccbaad0dcf67a` | `018333a5694971e1dcd97a97d8f4a2018243f3a7ec96ee728dd74f4954a577d9` |
| `lstm-infer-worker` | `d99449d44f4473639c7fab1b831f058d6ac4386b` | `a34cb34d111f746e1757366af8dc36a7ffa49d811db17b779ef68c3c8a2e1d35` |
| `MetaNN_metal.metallib` | Existing paired resource | `9c894e9e02b3dfafb69d639535ebc064c7b5ef30ce72edd5cf628f220e16f759` |
| `default.metallib` | Existing paired resource | `a13694e6940e8287c1b3ca696edbcc291e85de2fd514ade52e431054f1d537d5` |

These four hashes match Phase 24N and remain unchanged after all probes. AGENTS.md and both prior report hashes also remain unchanged. No TRAIN or INFER rebuild was necessary. No `xcodebuild build` or Clean action ran. Small native qualification helpers were compiled under ignored Phase24O evidence using the existing archives/objects and established C++ composition-test pattern; they are not deployment artifacts.

## Independent PostgreSQL isolation

Isolation was established before database creation or worker execution. A socket bind check proved the proposed private port was available. Existing PostgreSQL server PID **3876** and production scheduler/workers were inspected read-only; none was controlled. The new instance used the existing PostgreSQL **17.11** installation, without changing its Homebrew configuration or production service.

| Boundary | Evidence |
| --- | --- |
| Private data directory | `DerivedData/ExpertAdvisor/Phase24O/pgdata` under Rollover |
| Private server | PID **83667**, separate from production PostgreSQL PID 3876 |
| Listen address / port | `127.0.0.1:55479`; no external interface |
| Socket directory | `DerivedData/ExpertAdvisor/Phase24O/socket` |
| Cluster system identifier | **7694270663846376713** from this instance's `pg_control_system()` |
| Test databases | `ea_phase24o_lstm`, `ea_phase24o_forex` |
| Administrator | `phase24o_admin`, independent random password |
| Runtime principal | A separately created `pqxx` role with a new independent random password; the worker connection contract fixes this role name |
| Authentication | SCRAM for local/TCP connections; runtime HBA entry restricted to the two test databases; remaining network clients rejected |
| Credentials | Private 0600 files; Phase24O directory restricted to 0700; credentials removed during cleanup |
| Resource limits | 12 connections, 32 MB shared buffers; probes sequential; `OMP_NUM_THREADS=1` |

The native probe calls the **actual** `EA::RuntimeDatabaseConnection::ForexConnectionString()` and `LstmConnectionString()` from the existing TRAIN object. Server queries independently confirmed:

```text
ISOLATED_RUNTIME_CONNECTION,database=ea_phase24o_forex,user=pqxx,port=55479
ISOLATED_RUNTIME_CONNECTION,database=ea_phase24o_lstm,user=pqxx,port=55479
```

Connection settings were supplied through a fresh subprocess environment, without inherited PostgreSQL service, password, options, or external-registry overrides:

```text
PGHOST=127.0.0.1
PGPORT=55479
PGCONNECT_TIMEOUT=5
PGPASSFILE=<private absolute Phase24O file>
LSTM_DB_HOST=127.0.0.1
LSTM_DB_NAME=ea_phase24o_lstm
FOREX_DB_HOST=127.0.0.1
FOREX_DB_NAME=ea_phase24o_forex
OMP_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1
CXX=/usr/bin/clang++
TMPDIR=<absolute Rollover Phase24O/tmp>
```

The fixed `user=pqxx` in the connection strings takes precedence over the helper environment's `PGUSER`. Omitted connection-string ports inherit the explicitly supplied `PGPORT`; native verification confirms that behavior. Production processes do not inherit this task's environment and query their own configured database rather than enumerate other clusters. This instance has a separate port, catalog, password authority, storage, and database names. Production's scheduler therefore cannot discover these test experiments through its configured connection. No production PostgreSQL connection, schema dump, data copy, or write occurred.

Schema/reference-data investigation identified the required families: raw ticks (`time`, `ask`, `vol`) and checked-in `candlestick`/`cst` functions for market input; experiment configuration/identity, ownership/protocol/lease/attempt state; model and matrix materializations, including optimizer/runtime configuration; economic-event and immutable calendar-snapshot read contracts; and inference-result/profitability persistence. Empty economic reference data is distinct from a populated authoritative production calendar and would limit eventual canary coverage.

The setup reused the checked-in `Database/LSTM_schema.sql` schema-only superset and `Database/forex/candlestick.plpgsql`, rather than inventing a reduced persisted contract. It applied checked-in migration **052** to initialize the protocol singleton. It granted isolated runtime read access for the probes. It did not seed raw ticks, economic events, experiments, models, attempts, calendar snapshots, or workflow decisions. No sensitive production records or historical checkpoints were cloned. This is not a claim that a complete minimal native-workload bootstrap or migration ledger was qualified.

### Schema replay finding

Direct replay of `Database/LSTM_schema.sql` exits **3** at line **31770**: `no schema has been selected to create in`. The dump sets an empty search path, then appends unqualified controlled-family DDL after its “database dump complete” marker. The preceding dumped objects had already loaded. The retained temporary tail was applied in a session with `SET search_path TO public`; its SQL created the appended objects, but its copied trailing `\unrestrict` produced another exit **3** because that separate session was not in restricted mode. A final catalog check found all **12** controlled-family tables. These are recorded setup failures, not successful full-file restore commands. No checked-in schema was edited, and no experiment workflow state was changed by the recovery.

A future bootstrap should use a regenerated, self-contained schema export or a reviewed migration bootstrap with correct search-path and psql restriction boundaries. This finding is separate from the scheduler admission blocker.

## Scheduler protocol blocker

Migration 052 authoritatively creates the singleton as generation **52**, state **`pending`**. The native probe uses the real `PostgresSchedulerRepository` and `SchedulerAuthorityService::acquire`, with actual process identity, against the isolated database. It returns:

```text
ISOLATED_PROTOCOL_REJECTION,generation=52,state=pending,accepted=0,authority_acquired=0,reason=explicit_safe_cutover_required
```

The probe exits **0** because the expected rejection was verified. Database counts before and after remain **zero** for experiments, models, scheduler invocations, and worker attempts. It does not call protocol completion, reserve workers, or run the scheduler dispatch loop.

The supported completion path is `ExperimentScheduler.cpp::CompleteSchedulerProtocolCutover` at line **5219**. It calls `ProductionSchedulerDaemon.cpp::InspectAllSchedulerDispatchProcesses` at line **1051**, which runs a host-wide `ps -axo pid=,command=` and accepts no process whose command contains `--schedule-experiments`. It rejects those processes **before opening the configured database**. Its parser requires `--yes` and rejects `--dry-run` (line **4878**). The final read-only process snapshot confirms production PID **30886** still has that flag.

Therefore the CLI would reject this isolated cluster's completion while production remains active. This conclusion is from inspected source plus observed process evidence; the full completion CLI was **not executed**, and no native CLI rejection log is claimed. Native rejection of the pending state by the authoritative service **was** executed. Calling completion directly below its CLI safety guard, forging absence evidence, updating the singleton through SQL, or stopping production was excluded by the task's protection and workflow requirements.

Prefer repeating qualification on a separate supported macOS host with no production scheduler. If qualification must share this host, seek a separately authorized, narrowly scoped isolated-database bootstrap workflow that proves cluster/credential isolation and uses the existing authority service while retaining production's safety gate. No selector, persisted lifecycle, or completed-phase redesign is proposed or implemented.

## Native worker findings and qualification coverage

The exact qualified workers were exercised only for identity and missing-ownership rejection. TRAIN reached native scheduler registration and returned **125**, reporting `worker_attempt_identity_missing`. INFER reached `managed_application_entry`, then the same registration rejection with exit **125**. Neither entered model construction, materialization, training, or prediction.

| Requested milestone | Finding |
| --- | --- |
| TRAIN startup / database connectivity | Native registration path and isolated runtime connection builders exercised; successful managed workload startup not qualified |
| TRAIN feature materialization, forward/backward, optimizer | Not run; no native pass claimed |
| Native Metal initialization / GPU kernels | Not reached; metallib hashes are integrity evidence only |
| Native control / nonempty persisted-mask treatment | No experiment created; no native column suppression or unmasked-column invariance observed |
| Checkpoint creation / experiment transitions | No model/checkpoint or experiment exists; no lifecycle pass claimed |
| Checkpoint continuation / optimizer restoration | Blocked by absent newly trained checkpoint; no historical production checkpoint used |
| Checkpoint-backed INFER / valid predictions / result persistence | Not run; unowned-admission refusal is not inference success |
| Dedicated TRAIN/ablation/INFER scheduler dispatch | No scheduler dispatch loop launched; routing evidence remains CPU-fixture scope |
| Failure reporting / cleanup | Native workers emit explicit registration failure and exit 125; no durable attempt/model/experiment side effects; both processes exited |
| Production interference | No production control action; final production scheduler and worker processes remain present; GPU work was not attempted |

Identifiers **1** in the negative worker commands are deliberately nonexistent requested bindings, not created experiment, model, or attempt identifiers. There are **no qualification experiment IDs or checkpoint IDs**. A valid nonempty canonical mask was exercised in CPU regressions only, not in persisted native workload metadata.

## Commands, builds, and regressions

Exact subprocess argument arrays, environments, durations, and exit codes are retained in ignored `DerivedData/ExpertAdvisor/Phase24O/commands.jsonl`; helper link commands are retained in `scheduler-build-command.json` and `protocol-probe-build-command.json`. Full identity output, build logs, database observations, process snapshots, regression output, and cleanup evidence are alongside them. Private passwords are not part of this report or command inventory; the server log's role password was redacted during cleanup.

Principal commands included:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
git diff --stat 12e31c02 HEAD
shasum -a 256 DerivedData/ExpertAdvisor/Build/Products/Release/{lstm-train-worker,lstm-infer-worker,MetaNN_metal.metallib,default.metallib}
file DerivedData/ExpertAdvisor/Build/Products/Release/lstm-*-worker
otool -l DerivedData/ExpertAdvisor/Build/Products/Release/lstm-train-worker
otool -l DerivedData/ExpertAdvisor/Build/Products/Release/lstm-infer-worker
codesign --verify --strict DerivedData/ExpertAdvisor/Build/Products/Release/lstm-train-worker
codesign --verify --strict DerivedData/ExpertAdvisor/Build/Products/Release/lstm-infer-worker
```

Cluster creation used `/opt/homebrew/opt/postgresql@17/bin/initdb -D <Phase24O/pgdata> -U phase24o_admin --auth-host=scram-sha-256 --auth-local=scram-sha-256 --pwfile=<private file> --encoding=UTF8 --locale=C`, followed by `pg_ctl -D <same data directory> -l <Phase24O/postgres.log> -w -t 10 start`. Queries/restores used `psql -X -v ON_ERROR_STOP=1 -d <explicit test database>` with the isolated environment above.

Native negative commands, with the absolute stable Release executable paths and isolated environment:

```text
lstm-train-worker --train --scheduler-experiment-id=1 --scheduler-worker-attempt-id=1 --symbol=eurusdrmp --epochs=1 --window-size=4 --hidden-size=4 --num-layers=1 2025-01-01 2025-01-02
lstm-infer-worker --infer --model=1 --scheduler-experiment-id=1 --scheduler-worker-attempt-id=1 --donchian20-mode=enabled --feature-warmup-scope=full_history_warmup --donchian-lookback=20 --log-level=summary 2025-01-01 2025-01-02
```

Three relevant regressions pass:

```bash
CXX=/usr/bin/clang++ bash Tests/TrainingWorkerFeatureAblationTests.sh
CXX=/usr/bin/clang++ bash Tests/TrainingWorkerPersistedAblationTests.sh
PYTHONDONTWRITEBYTECODE=1 python3 Tests/DedicatedTrainAblationRoutingTests.py
```

The first verifies the actual shared mask/materialization contract on CPU; the second reports pure persisted/continuation validators with no database connection; the third passes its one unittest method covering the existing 26 routing subcases. These results do not substitute for native masked training or native scheduler dispatch.

Setup retries are retained candidly: the initial sandbox socket bind was denied, before initdb/database creation, and the authorized isolated-instance retry succeeded outside that restriction. An incomplete helper link failed because the SchedulerCore archive requires additional existing composition objects/services; the completed scheduler and protocol-probe helper links succeed with empty diagnostic logs, using the established composition-harness dependencies. The broad experiment-CLI helper was not completed or executed. The first CPU test environment omitted Homebrew's `rg` path and exited **127**; adding the existing `/opt/homebrew/bin` to that task-local PATH made it pass. Two initial INFER commands used invalid mode/scope spellings and exited **1** before registration; the corrected supported options produced the expected **125** registration failure. None of these invocation failures was treated as a worker model/GPU defect or hidden as a passing workload.

## Cleanup, risks, and final Git state

Both test databases were dropped using only the verified private cluster. Before deletion, SQL asserted its exact data directory and port. Its catalog afterward contained only `postgres`, `template0`, and `template1`. `pg_ctl -D <Phase24O/pgdata> -m fast -w -t 10 stop` exits **0**; subsequent status exits **3**, `no server running`. Private pgdata and disposable credential files were removed. No production server was stopped or signaled. Final process evidence contains no live Phase24O PostgreSQL or probe worker.

No source, project, shared MetaNN file, production file/database/registry, RepositoryAgent configuration, or Qwen configuration was changed. No worker binary was published, deployed, copied into a publication root, rebuilt, or replaced. No commit, merge, or push occurred. Ignored native helpers and logs remain under Rollover DerivedData.

Remaining cutover risks include the blocked positive native workloads, actual mask suppression in those workloads, checkpoint/model/optimizer restoration, valid prediction/result persistence, scheduler dispatch/reaping/transitions, and workload failure cleanup after GPU startup. Exact macOS 27.0 behavior, deployment-host signing/dependency acceptance, full production-like calendar/market coverage, resource contention, and rollback execution also remain unverified. Operational cutover must wait for a successful protected canary; the identity/CPU evidence alone does not justify GO.

Only source-tree file changed: **this report**, uncommitted. Exact behavioral change: **none to the application**; only isolated qualification evidence was added.

`git status --short`:

```text
?? docs/phases/Phase24/LSTM_Phase24O_NativeOperationalQualification_Output.md
```

`git diff --stat`: **empty**, because the report is untracked. `git diff --check`: passes. Final HEAD remains **3f59667acbaa997a71351ea415a5a40bbea6d795**. Stop after this report; no production cutover is authorized or performed.
