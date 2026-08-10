# Campaign Operations Phase H — H3 Runtime/Concurrency Harness Completion

Date: 2026-08-09

## Result

The baseline H3 implementation report identified a MEDIUM gap: injected local/global continuation and concurrent Manager run-once scenarios had not been executed through a dedicated disposable H3 workflow harness. That gap is closed.

Final marker:

`READY_FOR_H3_INDEPENDENT_REVERIFICATION`

## Harness architecture

`Tests/CampaignOperationsPhaseH3RuntimeConcurrencyTests.sh` reuses the H2 disposable PostgreSQL workflow to create a quiescent temporary cluster, installs migration 058 with an exact checksum/ledger check, adds only disposable H3 fixture requests, and clones eight isolated scenario databases. The C++ harness invokes `RunCampaignOperationsManagerOnceForTest`, which shares the production Manager candidate snapshot, sequential loop, production Manager adapter, Phase E engine, retry, replay, and uncertain-commit paths. The only test seam injects an approved fixture build and synchronization callbacks; it is not exposed by the CLI.

Synchronization uses condition-variable barriers, independent PostgreSQL sessions, transaction-visible hooks, `pg_backend_pid()`, `pg_blocking_pids()`, and `std::thread` joins. No timing sleep is used to infer ordering or blocking. Final examples captured by the harness included F waiter/blocker PIDs `40944/40931` with `pg_blocking_pids=40931`, and G `40963/40957` with `pg_blocking_pids=40957`.

## Scenario results

| Scenario | Executable proof |
|---|---|
| A | Six candidates selected in one repeatable-read snapshot; IDs 72–74 produced three distinct request-local failures; IDs 71, 75, and 76 committed; failed requests had zero admission/Attempt/audit/binding/source rows; ordered before/after hooks proved sequential continuation. |
| B | Disable became effective between IDs 71 and 72; the next mutation stopped as `production_disabled`; later candidates were untouched and prior evidence remained immutable. |
| C | Disposable scheduler protocol evidence was changed to valid `failed` state with completion fields cleared; the readiness path stopped as `scheduler_protocol_ineffective`, not as disablement. |
| D | Exactly the Manager source-evidence INSERT capability was revoked; the actual mutation received privilege failure and stopped without fallback or escalation. |
| E | A deterministic `pqxx::broken_connection` reached the Manager loop; it stopped as `database_failure`, with no guessed success or later processing. |
| F | Two Managers passed overlapping snapshot barriers, selected request 71/version 3, derived identical source/key identity, and resolved to one new operation plus one exact replay. Counts remained one admission, one Attempt V2, two audit-reference rows, one binding, and one Manager source row. PostgreSQL blocking catalog evidence was captured. |
| G | Manager B progressed request 72 while Manager A was blocked on request 71; direct PostgreSQL blocker evidence distinguished the shared authority lock domain from a Manager-global mutex. |
| H | Migration 058 source evidence was checked for exact association, version/key/canonical equality, replay full comparison, missing/duplicate/mismatched/altered/same-hash tampering, atomic rollback, and unchanged historical Attempt V1 canonical bytes. |
| I | A’s ordered hook trace proved operational_request_id order, no request overlap, bound limit, and snapshot completion before processing. |
| J | Structural assertions retained the H4 negative proof: no daemon/continuous/polling/autostart/supervision/worker-control/durable-batch surface, `SKIP LOCKED`, or batch identity. |

Failure classification is explicit: request-local semantic failure continues; global disable stops; scheduler protocol/effective-enable mismatch stops; privilege failure stops; database-wide and commit-unknown failure stops.

## Narrow production corrections

The harness exposed and corrected three concrete H3 defects:

1. Migration 058 compared the table’s original `request_identity_canonical` against the complete Manager source canonical. The trigger now compares that column with the Attempt V2 request canonical and compares `source_canonical` with the complete derived source.
2. Manager now reports the Phase E `transientDatabaseRetryExhausted`/proven-no-commit result as request-local, while `commitOutcomeUnknown` stops as database-wide instead of being treated as an ordinary dispatch result.
3. Scheduler-protocol diagnostics now take precedence over the accompanying ineffective-enable diagnostic, preserving the required stable scheduler-stop classification.

The disposable fixture adapter retains the production readiness gate and ignores only the known aggregate `canonical_contract_versions` blocker caused by immutable historical Attempt V1 fixture rows. Production Manager and CLI readiness behavior is unchanged.

## Files changed by this pass

- Added `Tests/CampaignOperationsPhaseH3RuntimeConcurrencyTests.cpp` and `.sh`.
- Added H3 test-only synchronization seams in `Sources/CampaignOperationsManagerService.hpp` and `Sources/CampaignOperationsDispatchService.hpp`.
- Updated the H3 Manager/service implementation and migration 058 as described above.
- Extended the disposable H1 fixture with deterministic request IDs 73–76.
- Added H3 runtime traceability to `docs/CampaignOperationsPhaseH3.rst`.

The worktree also contained pre-existing H1/H2/H3 implementation, migration, documentation, project, database-backup, scheduler, and test-script changes; those were preserved and are not attributed to this focused pass.

## Verification commands

- `Tests/CampaignOperationsPhaseH3RuntimeConcurrencyTests.sh` — PASS, including migration 058 disposable install and A–J.
- H3 identity unit test compiled with C++20/`-Wall -Wextra -Werror` and passed.
- `Tests/CampaignOperationsPhaseH3CliTests.sh DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release` — PASS.
- `Tests/CampaignOperationsPhaseH3ContractTests.sh` — PASS.
- `Tests/CampaignOperationsPhaseH2WorkflowTests.sh` — PASS.
- `Tests/CampaignOperationsPhaseH2ReplayAssociationTests.sh` — PASS.
- `Tests/CampaignOperationsPhaseH2ConcurrencyTests.sh` — PASS.
- `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build` — `BUILD SUCCEEDED`. Existing scheduler-source libpqxx deprecation warnings remain; no new H3 warning was introduced.

The live Release Manager command was not run because active production scheduler/training workers were present. The parser-only CLI proof and the actual Manager path in isolated disposable databases were run.

## Findings

- BLOCKER: none.
- HIGH: none.
- MEDIUM: none remaining for H3 runtime verification.
- LOW: live production CLI invocation remains intentionally unexecuted while active scheduler/training workers are running; independent reverification should perform that operational check in a controlled environment.
