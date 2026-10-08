# Phase 24P — Database-scoped protocol cutover investigation

**NO-GO for implementing an exclusion of the active production scheduler or claiming isolated protocol cutover success.** Investigation did not establish a reliable, immutable effective database binding for that running scheduler using the existing components. A native counterexample proves that the macOS kernel exec-environment snapshot can disagree with current libpq settings. The requested fail-closed policy therefore keeps this scheduler blocking.

Stopped under Parts B/C and the explicit stop condition: no speculative classifier, global bypass, force flag, direct SQL protocol update, or parallel authority mechanism was implemented. No application source or checked-in regression test changed; no implementation commit was made. This report remains uncommitted. Further direction is required for a separate qualification host or a separately scoped verified database-binding design.

## 1. Baseline and production protection

Date: **2026-10-08**. Repository: `/Volumes/Developer SSD/ExpertAdvisor-Rollover`.

| Check | Result |
| --- | --- |
| Branch | `dedicated-train-layout-rollover-squashed-v1` |
| Initial/final HEAD | **`df5836a053cc9b774862c0f4c4e7a7ee4f65dcb8`**, matching `df5836a0` |
| Initial worktree | Clean; `git status --short` empty |
| Required reading | Root AGENTS.md, Phase 24O report, relevant Phase 24M/24N qualification and compatibility findings |
| Delta from Phase 24O execution HEAD | `git diff --stat 3f59667a HEAD`: only the Phase 24O report, 179 insertions |
| Production scheduler | PID **30886**, parent **30885**, state `S+`; active in initial and final read-only snapshots |
| Production command | Dedicated production `lstm-scheduler`, including `--schedule-experiments`, production semantic registry and analyzer paths |
| Observed production workers | TRAIN PIDs 21660, 21666, 55503, 55801; INFER PID 5916; none signaled |
| Production PostgreSQL | Existing server PID **3876**; no connection or control action by this phase |
| Nested MetaNN link | `MetaNN/MetaNN -> /Volumes/Developer SSD/ExpertAdvisor/MetaNN`, unchanged |
| Apple compiler | `/usr/bin/clang++`, Apple clang **21.0.0**, `clang-2100.3.34.2`, ARM64 Darwin 27 |

Existing stable Release artifacts remain unchanged:

| Worker | SHA-256 |
| --- | --- |
| TRAIN | `018333a5694971e1dcd97a97d8f4a2018243f3a7ec96ee728dd74f4954a577d9` |
| INFER | `a34cb34d111f746e1757366af8dc36a7ffa49d811db17b779ef68c3c8a2e1d35` |

Baseline/final hashes also match for inspected connection, cutover, authority, repository and observer sources, AGENTS.md, schema, and both shared MetaNN/MetalSwift project files. No shared source was modified. No TRAIN/INFER executable was run or rebuilt in this phase. No Homebrew dependency, production file/configuration, RepositoryAgent, or Qwen configuration was changed. No publication, deployment, merge, push, Clean, or `CONFIGURATION_BUILD_DIR` override occurred.

## 2. Original blocker and authoritative path

Phase 24O created an independent PostgreSQL cluster on port 55479 and confirmed real runtime connectivity. Migration 052 initialized its protocol as generation **52**, state **pending**. The native authority probe rejected that state without acquiring authority or creating an invocation/attempt. The CLI's host-wide absence requirement prevented completing initialization while production scheduler 30886 remained active. Positive TRAIN, ablation, continuation and checkpoint-backed INFER were consequently not qualified.

Relevant current boundaries:

| Boundary | Location and behavior |
| --- | --- |
| Cutover CLI | `Sources/SchedulerCore/ExperimentScheduler.cpp:5219`, `CompleteSchedulerProtocolCutover`; validates process absence before opening its database connection |
| CLI authorization syntax | Same file, line 4878; requires `--yes`, rejects `--dry-run` |
| Discovery | `Sources/SchedulerCore/ProductionSchedulerDaemon.cpp:1051`, `InspectAllSchedulerDispatchProcesses`; runs `ps -axo pid=,command=`, finds substring `--schedule-experiments`, requires successful `pclose` |
| Reuse outside cutover | Same census is called by orphan recovery around line 7816; changing its global semantics would affect another accepted workflow |
| Authority acquisition | Same file, line 1080, `AcquireSchedulerAuthority`; obtains actual PID/group/start identity and delegates to the existing service |
| Protocol admission | `SchedulerAuthorityService.cpp:183`, `acquire`; coordination lock, protocol-row lock/read, exact generation and `complete` state required before invocation registration |
| Cutover persistence | Same service, line 337, `completeProtocolCutover`; same coordination lock, generation check, idempotent already-complete result, exact repository update |
| PostgreSQL serialization | `PostgresSchedulerRepository.cpp:1847`; transaction advisory lock on the established coordination key, then singleton `FOR UPDATE` |
| Protocol mutation | Same repository, line 2028; conditional update of pending/failed generation-matching singleton; one-row result required |

The existing transaction and service boundaries must remain authoritative. Opening a target transaction, proving absence of a row there, or finding no current backend would not by itself prove that a live scheduler belongs permanently to another instance.

The census is a process-discovery snapshot, not a database-identity observer. Its substring test is also narrower than the existing status recognizer, which can recognize the dedicated scheduler executable without that flag. Production 30886 does carry the flag, so that additional discovery limitation does not explain away the observed blocker. No discovery redesign was implemented.

## 3. Effective connection investigation

The ordinary scheduler connection builder in `ProductionSchedulerDaemon.cpp:174` constructs:

```text
hostaddr=<LSTM_DB_HOST or 127.0.0.1> gssencmode=disable user=pqxx dbname=<LSTM_DB_NAME or LSTM>
```

The experiment CLI uses the same builder through its existing runtime composition. `LSTM/SchedulerMain.cpp:78` and `Sources/RuntimeDatabaseConnection.cpp` have equivalent ordinary connection construction. Campaign-specific principal builders are separate workflows and were not used.

Connections are reconstructed at many scheduler operations, including authority acquisition/renewal, queue reads and polling. The ordinary builder supplies host address, database and runtime user but **omits port**. libpq resolves omitted parameters from its service/environment/default machinery. Installed PostgreSQL 18.6 primary documentation was inspected locally at `/opt/homebrew/opt/libpq/share/doc/postgresql/html/{libpq-connect,libpq-envars,libpq-pgservice}.html`: explicit connection parameters override service values; service values override corresponding environment defaults. `PGPORT` is therefore relevant even when it is absent from argv. Explicit database/user fields supersede `PGDATABASE`/`PGUSER` defaults. Endpoint strings and a different role name alone do not identify a different server instance.

Normal nonempty `hostaddr` selects TCP. Generic libpq Unix-socket connections use the `host` socket directory and port/socket filename; a directory string cannot simply be interpreted as this builder's IP address. Socket-path aliases, multiple listeners, default sockets, service settings and ambiguous/nonliteral connection fragments require actual resolution rather than string comparison. The ordinary builders concatenate unquoted environment values, so a proposed classifier would also need to parse the actual resulting libpq parameters instead of treating each environment value as one already validated field. No parser or connection contract was changed here.

### Existing observation and persisted identity

`SchedulerOperationalObservation.hpp::ProcessObservation` supplies PID, group, start identity, canonical executable, command and inspection status. Its native backend uses `proc_pidpath` and a `KERN_PROCARGS2` executable-path fallback. It rechecks process start identity after observation, protecting that observation against PID reuse. It does **not** expose current libpq options, server-system identity, database OID/name from an actual connection, or an immutable connection binding.

The `experiment_scheduler_invocation` contract persists process identity, executable/command, nonce, protocol generation and lifecycle timestamps. The lease persists ownership and fencing state. Neither supplies a cross-instance, process-bound immutable database identity. These records are inside the connected database; they are not an authenticated host-wide index of every scheduler's future connection destinations. No schema extension or alternate authority registry was added.

### Native environment counterexample

A small ignored C++ investigation helper was compiled with Apple Clang, `-Wall -Wextra -Werror`, the real `Sources/RuntimeDatabaseConnection.cpp`, and the unchanged installed libpq. It makes **no database connection**. It reads its own kernel exec environment, changes only its own environment, reads libpq defaults with `PQconndefaults()`, and calls the real runtime connection builder:

```text
FIXTURE_EXEC_PGPORT=59971
KERNEL_PGPORT_AFTER_SETENV=59971
LIBPQ_DEFAULT_PORT_BEFORE=59971
LIBPQ_DEFAULT_PORT_AFTER=59972
ACTUAL_RUNTIME_CONNECTION_AFTER=hostaddr=127.0.0.1 gssencmode=disable user=pqxx dbname=ea_phase24p_changed
KERNEL_EXEC_ENVIRONMENT_IS_NOT_CURRENT_LIBPQ_IDENTITY=verified
```

Compilation and execution both exit **0**. The same kernel exec-environment mechanism can therefore return stale launch values while effective libpq defaults and the real application builder have changed. This is a controlled counterexample, **not** evidence that production changed its environment. No `setenv`/`unsetenv` call was found in the inspected scheduler implementation; absence of such a call in inspected source does not turn the observer's historical environment into a verified current connection binding for every existing executable/process.

A whitelist-only projection of production PID 30886's kernel exec environment succeeded without logging passwords or full environment contents. `LSTM_DB_HOST`, `LSTM_DB_NAME`, `PGPORT`, `PGSERVICE`, and `PGSERVICEFILE` were absent; `PGDATABASE` was `forex`. Interpreting that last field as the scheduler's database would be incorrect for the inspected ordinary builder, whose explicit `dbname` takes precedence. Defaults suggest `127.0.0.1:5432/LSTM`, but that is **an inference**, not sufficient positive evidence for the required exclusion.

Read-only `lsof -nP -a -p 30886 -iTCP -Fpcfn` returned exit **1** with no matching TCP records and an unrelated mounted-filesystem warning that output may be incomplete. No reliable live-session identity was obtained. The result was not treated as proof of disconnection or unrelatedness. Scheduler connections are transaction-scoped and recreated, so even a successfully attributed backend/socket snapshot would not certify the next connection's destination or an unchanged binding throughout cutover.

### Races and identity changes

A second process census can reveal some changes but cannot prevent a scheduler from starting or its effective connection parameters from changing between the last sample and commit. Existing target-database coordination serializes participating target authority operations; it does not freeze another process's environment, service files or future endpoint resolution, nor does it turn a legacy process into an attested participant. Process start identity protects against PID reuse, not an unchanged database destination within one process.

A robust future design would need a verified actual server/database identity bound to process start identity, a connection destination pinned for the relevant lifetime, complete discovery with inaccessible cases rejected, and startup/inspection coordinated through the established authority boundary. Checking an IP/port, kernel launch environment, argv, a cached authority row, or one TCP session cannot substitute for those obligations. Merely fetching `pg_control_system()` from the inspector's own connection proves that connection's cluster, not the scheduler's effective or future connection.

## 4. Safety policy and implementation decision

The requested policy is retained as a design requirement, **not claimed as implemented**:

| Evidence about a discovered scheduler | Required decision |
| --- | --- |
| Connected to the target database | Block |
| Same server instance, different database name | Conservatively block under the requirement for positive different-instance evidence |
| Different endpoint string, role, argv or kernel environment only | Block; these are not verified server identity |
| Unknown, ambiguous or inaccessible effective identity | Block |
| Identity/destination can change during inspection or cutover | Block |
| Discovery failure, PID/start-identity inconsistency, new unclassified scheduler | Block/restart the safety assessment |
| Positively verified different instance, exact process binding, stable effective connection and coordinated discovery interval | Exclusion could be considered; this evidence was not established for production 30886 |

The current production scheduler belongs to the **unverified effective-binding** category for this proposed exclusion. It was not positively classified as unrelated. Implementing a filter that ignores it based on inferred default port/database, absent current TCP evidence or a successful kernel-environment read would violate the task's fail-closed requirements.

**No implementation correction was made.** The existing guard, protocol admission, ownership semantics, advisory locks, persisted workflows and production behavior/configuration remain unchanged. An immutable verified binding may be feasible for future cooperating scheduler builds, but establishing it for the already-running production process without an existing binding is not a demonstrated small correction within this phase's constraints. No production restart/update was attempted or requested as part of execution.

## 5. Regressions and requested coverage limits

Six existing relevant suites pass, using a task-local PATH that selects Apple's compiler and temporary output under stable Rollover DerivedData where supported:

```bash
bash Tests/SchedulerAuthorityServiceTests.sh
bash Tests/SchedulerStatusProcessRecognitionTests.sh
bash Tests/SchedulerInternalSeamTests.sh
bash Tests/SchedulerSemanticAdmissionTests.sh
bash Tests/SchedulerOperationalObservationBoundaryTests.sh
PYTHONDONTWRITEBYTECODE=1 python3 Tests/SemanticWorkerPublicationContractTests.py
```

The publication suite reports **14 passing tests**. Authority tests exercise coordination, generation rejection, cutover-service completion/idempotency, fencing and owner inspection. Process-recognition, internal-seam, semantic-admission and observer-boundary checks pass. These are unchanged baseline regressions, not qualification of a new database-scoped classifier or native worker execution.

An additional ignored probe extends the existing in-memory authority fixture with a pending-state admission assertion and compiles the actual `SchedulerAuthorityService.cpp`. It verifies `protocolAccepted=false`, no acquired authority, no invocation registration, exactly one coordination-lock acquisition, and no owner-inspector call before rejection. Compile and run exit **0**. It neither completes a real cutover nor writes PostgreSQL.

| Requested new coverage | Phase 24P status |
| --- | --- |
| 1. No active schedulers | No new scoped-cutover integration test; existing authority vacant-owner/cutover-service cases pass |
| 2. Scheduler on target | Required blocking policy documented; new classifier not implemented/tested |
| 3. Scheduler on unrelated database | Positive reliable binding unavailable for active production; no exclusion pass |
| 4–5. Unknown/ambiguous identity | Blocking requirement preserved by declining to add exclusions; no new end-to-end classifier tests claimed |
| 6. Environment host/port overrides | Real builder traced; native stale-kernel/current-libpq counterexample verified; no scheduler classification pass |
| 7. Unix sockets | libpq/socket semantics investigated; no socket-backed cutover integration run |
| 8. Discovery failure | Existing process/observer boundaries pass; lsof observation treated as insufficient, not as clearance |
| 9–10. Startup/identity changes during cutover | Race obligations documented; no synchronization correction or new integration pass |
| 11. Authority locking | Existing tests and additional real-service pending probe pass; existing repository advisory/row locking source-reviewed |
| 12. Admission before successful cutover | Additional pending-state probe passes; existing generation-rejection tests pass; no isolated successful cutover claimed |

No new checked-in regression suite was added because Part C explicitly requires stopping when reliable classification cannot be established. No test is presented as proving a policy that was not implemented. Database/process integration scripts that default to creating databases on the ordinary server or cloning production schema were inspected but **not executed**. No full worker or scheduler product build ran. The native investigation/probe builds have no compiler diagnostics; the authority shell invocation emitted sandbox/Xcode cache-discovery messages but returned success, with no compiler warning/error. No dependency or compiler configuration was changed to address those messages.

## 6. Isolated validation and remaining work

Part E was **not attempted**: its reliable implementation/qualification preconditions were not met. No Phase24P cluster, credentials, database, scheduler authority, experiment or worker attempt was created. No protocol-cutover CLI was invoked against production or a disposable instance. There is therefore no isolated cutover-success or post-cutover authority result to report, and no Phase24P database/server cleanup requirement.

There were **zero PostgreSQL connections or protocol writes** by this phase. Production protocol state was not queried or snapshotted, so no before/after SQL equality claim is made; this phase performed no action capable of changing it. The final read-only process snapshot confirms production scheduler 30886 remains active. Worker hashes, inspected sources/schema/shared projects and nested link are unchanged.

Remaining Phase 24O work remains outstanding: successful protected native TRAIN control/treatment, actual column suppression/invariance, checkpoints, model/optimizer continuation, checkpoint-compatible INFER and result persistence, Metal execution, and scheduler dispatch/lifecycle/reaping/failure cleanup. None was started in this phase.

**Recommendation:** use a separate supported qualification host without production schedulers for the existing cutover workflow. If same-host exclusion remains necessary, request a separately scoped design for an immutable, process-bound effective connection/server identity and coordinated inspection using existing authority components, with explicit treatment of legacy processes as unknown. That work must not assume permission to update/restart production. No approval for production modification is inferred from this report.

**NO-GO for same-host database-scoped cutover under the current evidence; NO-GO for production cutover.** This conclusion is limited to the inability to safely exclude the running production scheduler under the requested conditions, not a claim that database-scoped protection is impossible in a future verified design.

## 7. Evidence, commands and final Git state

Ignored evidence: `DerivedData/ExpertAdvisor/Phase24P/` contains baseline/final hashes and identity metadata; environment probe source/executable/build log and counterexample; the whitelist production projection; TCP observation/error/exit logs; local installed-libpq service documentation text; six regression logs and exact argv/environment/duration/results; pending-protocol authority probe source/build-command/log/result; and final production process evidence. These helpers are investigation artifacts, not production configuration or a parallel authority service.

Environment-probe build:

```bash
/usr/bin/clang++ -std=c++20 -O3 -Wall -Wextra -Werror \
  -ISources -I/opt/homebrew/opt/libpq/include \
  DerivedData/ExpertAdvisor/Phase24P/environment_identity_probe.cpp \
  Sources/RuntimeDatabaseConnection.cpp \
  -L/opt/homebrew/opt/libpq/lib -lpq \
  -o DerivedData/ExpertAdvisor/Phase24P/environment-identity-probe
```

The counterexample runs in a fresh environment with `PGPORT=59971`, `LSTM_DB_HOST=127.0.0.1`, `LSTM_DB_NAME=ea_phase24p_fixture`, task-local TMPDIR and a system PATH. The production projection uses that executable with `--inspect 30886` and prints only the whitelisted nonsecret keys. Read-only process commands include `ps -axo pid,ppid,stat,comm`, `ps -p 30886 -o pid=,ppid=,stat=,command=`, and the lsof command above. `shasum`, `readlink`, compiler version, Git branch/HEAD/status/diff and Python hash comparisons support baseline/final checks. No production executable was launched.

Only source-tree file changed: **this report**. Exact application behavioral change: **none**. The conditional authorization to commit passing implementation/tests was not exercised because no reliable correction was implemented or operationally qualified.

`git status --short`:

```text
?? docs/phases/Phase24/LSTM_Phase24P_DatabaseScopedProtocolCutover_Output.md
```

`git diff --stat`: **empty**, because the report is untracked. `git diff --check`: passes. HEAD remains **df5836a053cc9b774862c0f4c4e7a7ee4f65dcb8**. No commits, publication or deployment. Stop and request further direction.
