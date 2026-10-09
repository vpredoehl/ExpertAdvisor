# Phase 24Y — Transaction-Safe Global Experiment Pause/Resume

Date: 2026-10-09. Development repository: /Volumes/Developer SSD/ExpertAdvisor-Rollover.
Baseline HEAD: 15d9e9762ac9b7ff12e69ff7009cf391b7e8c715.

## Qualification decision

The transaction-safe correction is implemented. The final private artifact builds,
links, signs, and passes the focused faults, global-control, repository, offline,
capacity and priority checks. The final full native regression run is still in
progress; its final result must be recorded before declaring qualification complete.

A publishable Release remains **BLOCKED by clean Git provenance**. The private
Phase24T/U qualification workflow supplies development runtime evidence; it does
not replace an authorizable Release build. Production publication needs a separate
decision and a clean, reviewed source commit. No production action was performed.

## Independent root-cause review

The pre-fix counterexample is retained at
DerivedData/ExpertAdvisor/Phase24T/Phase24Y-Characterize-20261009001925/global-pause-transaction-tests.log:
a native synthetic worker received SIGSTOP, the open PostgreSQL transaction
failed, and the database retained running status with no committed pause request.
The existing reconciliation planner retained the observed worker and its capacity
rather than blindly continuing it.

The current complete tracked diff and new qualification files were reviewed.
Sources/SchedulerCore/ReconciliationService.cpp/.hpp remain unchanged:
PlanAttemptObservation retains a live worker without relaunch, keeps uncertain
identities capacity-consuming, and restores running lifecycle when a persisted
stopped worker is observed executing. The new pause recovery runs before ordinary
orphan recovery, preventing that planner from obscuring an unfinished pause.

## Before and after state machine

Before: one transaction created the request/gate/outcomes, sent SIGSTOP, wrote
stopped/paused state and committed. Rollback could erase every record of the stop.

After:

1. Transaction A freezes the exact target attempts and process identities in
   planned administrative outcomes and commits the paused global gate and request
   generation **before any OS signal**. Pending members without workers are
   paused in that transaction.
2. Transaction B claims the same durable request, takes the coordination lock,
   locks/verifies each exact active attempt, re-observes PID, process group,
   kernel start identity, executable and complete command, and sends only
   SIGSTOP. It verifies stopped delivery before recording stopped/paused state.
3. Failure or lost acknowledgement leaves either the committed plan or the
   completed pause. Retry replays that generation, never retargets a replacement
   process, and never creates an attempt. Unresolved outcomes retain the active
   request and closed gate. Successful application clears active_request_id
   while retaining current_pause_request_id and desired_state=paused.
4. Resume atomically queues only the completed global generation with
   resume_requested=true and operator origin, retires the generation and opens
   the gate. The CLI sends **no SIGCONT**. An incomplete pause must first recover
   successfully; resume cannot discard its committed request.
5. The existing scheduler winning-class/capacity admission path performs
   SIGCONT only for the exact stopped, pending, resume-requested attempt under
   current fenced authority and the locked global gate.
6. If SIGCONT succeeds but the admission transaction fails, the same durable
   attempt remains. Subsequent native observation accounts for that executing
   worker once and corrects capacity before further admission. If it exits after
   committing final INFER evidence, exact-result recovery now also accepts the
   legitimate pending operator-resume state and completes the original attempt.

A re-pause after selective release carries already globally paused members into
the new generation. Individually paused members remain excluded. This prevents
resume-all from stranding a global member under an obsolete generation. The
legacy regression now asserts that membership and the absence of SIGCONT.

## Authority, identity and priority review

Sources/GlobalExperimentControl.cpp:3772–4068 contains the durable claim,
application and reconciliation logic. Recovery checks completed protocol
generation 52, current invocation ID, nonce, canonical executable, active lease,
fencing token and unexpired lease. It locks the lease/invocation, refuses a live
foreign administrative lease, and repeats current-authority validation immediately
before a stop. Frozen launch ownership remains immutable: a legitimate successor
scheduler may observe an old owner's exact attempt, but a competing or stale
scheduler cannot control it.

Sources/SchedulerCore/ProductionSchedulerDaemon.cpp:2883–3220 retains locked
gate/capacity/attempt/semantic-worker admission and now rechecks authority at the
SIGCONT boundary. The existing service can renew an expired but uncontested
owner's lease under its coordination and row locks. Expiry alone is not fencing
loss; token/owner mismatch is rejected. This established renewal contract was
preserved. Private fault injection proves token loss between the initial check
and SIGCONT sends no continuation signal.

The new global-pause reconciliation path contains no SIGCONT. It cannot open the
pause gate or override a priority winner. Mismatched start identity, PID, group,
executable, command or active attempt prevents signaling; inconclusive evidence
keeps the request/slot rather than resurrecting a worker. Deterministic tests cover
identity change at both the first observation and the immediate stop authorization,
plus a stale stopped worker at resume admission.

Existing cross-phase priority, admission ordering, and preemption rollback
compensation remain intact. Normal ANALYZE displaced low TRAIN in the final
native suite, with the original TRAIN PID/start identity and single attempt
retained. Resume does not bypass winning-class selection.

## Implementation and files

- Sources/GlobalExperimentControl.cpp/.hpp: durable intent/application split,
  idempotent replay, fenced scheduler recovery and deterministic fault seam.
- Sources/SchedulerCore/ProductionSchedulerDaemon.cpp: startup/poll recovery,
  retryable stopped-admission failures, signal-boundary authority recheck and
  private test failpoints.
- Sources/SchedulerCore/PostgresSchedulerRepository.cpp: exact final-INFER result
  recovery accepts stopped pending operator resumes as well as preemption.
  Active-attempt binding, owned model, final scope and completion-time checks
  remain required; paused, ambiguous, unbound and stale-result cases remain rejected.
- Tests/GlobalExperimentControlProcessTests.cpp: transaction fault matrix,
  identity/authority tests, generation assertions and FK-correct fixture cleanup.
- Tests/PostgresSchedulerRepositoryTests.cpp: positive operator/preemption
  recovery and negative state, binding, identity, timestamp and model guards;
  forced-rerun identity remains preserved.
- Tests/GlobalExperimentControlTransactionTests.py: owned isolated PostgreSQL,
  full-schema fixture defaults, original native/legacy/priority/repository suites,
  timeouts and identity-verified cleanup.
- Tests/Phase24YQualification.py: reproducible private object/archive relink,
  source/object hashes, artifact validation and offline/native/priority runners.
- This report.

No schema change or new migration was introduced. Public CLI flags remain
--pause-all-experiments --yes and --resume-all-experiments --yes.

## Fault-injection matrix

The global-control matrix contains **17 named scenarios**, with additional
authority/idempotence assertions inside them. The native resume matrix has **7
named scenarios**. They use synthetic workers and owned PostgreSQL only.

| Failure window | Observed safe recovery |
|---|---|
| Before intent commit | No signal; request/gate roll back |
| Immediately after intent commit | Committed plan/gate survive; same-generation replay |
| Before SIGSTOP | Durable plan retained; later exact stop |
| Immediately after SIGSTOP | Native stop with planned outcome; replay persists it |
| Before worker-state persistence | Stop retained under committed intent |
| Actual PostgreSQL worker-state trigger failure | B rolls back; A survives; exact stop recovered |
| Before pause commit | Same durable request replayed without new attempt |
| Immediately after pause commit | Completed pause survives lost acknowledgement; retry is idempotent |
| Abrupt controller exit after intent / after stop | Parent recovers committed intent; no compensating SIGCONT |
| Before / after resume commit | Atomic queue/gate state; retry sends no signal |
| Actual deferred PostgreSQL resume-commit failure | Pause generation/gate remain intact |
| During reconciliation; wrong nonce/token; lease expires before stop | No unauthorized signal; durable request retained |
| Worker exit after pause intent | Exact absence retires attempt; paused generation retained |
| PID/start identity changes at observation 1 / 2 | No stop; resume withheld; request retained |
| Before SIGCONT | Pending stopped attempt retained; original worker later admitted |
| Fencing loss immediately before SIGCONT | No SIGCONT; supported new owner recovers original attempt |
| SIGCONT delivery failure | Stopped retry intent retained; no duplicate attempt |
| Immediately after SIGCONT / before admission commit | Executing original observed and accounted once |
| Reused stopped PID at resume | No SIGCONT/replacement; ambiguity retained for review |
| Worker exit with final result after SIGCONT/rollback | Original attempt completed; no repeated INFER dispatch |

Checks preserve epoch 17, checkpoint/model references, continuation fields,
policy revisions and scientific date ranges through the injected control failures.
These are isolated fixture observations, not a measurement of production epochs.

## Build result and provenance

Command: python3 -B Tests/Phase24YQualification.py build — **exit 0**.

The retained Phase24T/U recipe compiles GlobalExperimentControl,
ReconciliationService, PostgresSchedulerRepository and ProductionSchedulerDaemon
using Apple clang++, C++20, optimized ARM64/macOS 27.0 objects and explicit
libpqxx 7.10.1 headers/libraries; it replaces only private archive/file-list
objects, relinks and ad-hoc signs a private executable. Stable development
Release objects/libraries are reused. No shared MetaNN source was edited.

Artifact:
/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/Phase24Y/FinalQualification/build/LSTM_Release

SHA-256:
a2a8f998d8ef643d2b9f1519107d72f98fc67e190e8eb1822f970baa706e0f68

Compilation, archive replacement, link, codesign --verify --strict, Mach-O ARM64
inspection, dynamic-library inspection, --help and zero-worker CLI parsing pass.
artifact.json records changed-source/object/archive hashes and explicitly marks
qualification_only=true and publishable_release=false. Reused main/Release objects
retain baseline embedded provenance; the manifest identifies the dirty
qualification source. This executable must not be published as the Phase24Y Release.

Three unchanged warnings are justified by the accepted Phase24T qualification:
the unused CanonicalizeExecutablePath helper and two default semanticWorkerRole
aggregate initializers. No new project compiler warning was introduced.
The repository regression retains -Werror, suppressing only the existing pqxx
deprecation/attribute diagnostics.

The earlier complete supported build command was:

    xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build

It returned **65** at provenance generation. Independent final verification of
Scripts/GenerateBuildProvenance.py with --configuration Release returns **1**:
"Release provenance requires a clean source tree"; no header is generated.
release_commit requires an existing .git, empty git status --porcelain
(including untracked files), and an exact lowercase 40-hex HEAD. This worktree
cannot meet that requirement while retaining the authorized uncommitted work.
No generator, provenance header, ignore rule, index flag or publication check
was changed to hide it. No clean/stash/reset/discard/commit was performed.

## Regression results and evidence

Evidence root: DerivedData/ExpertAdvisor/Phase24Y/FinalQualification.
commands.jsonl retains exact build/offline argument arrays, exit codes and bounds.
Each owned native cluster retains its own commands, SQL/OS snapshots, signal and
cleanup receipts; disposable credentials and pgdata are removed after cleanup.

| Check | Count / final result |
|---|---|
| Offline C++ regressions | 18 suites, 18 pass, 0 fail |
| Global transaction faults | 17 named scenarios, 17 pass, 0 fail |
| Native global-control process suite | 1 suite, pass |
| Legacy database control | 8 cases, 8 pass, 0 fail |
| Priority/selective/cancellation database control | 7 cases, 7 pass, 0 fail |
| PostgreSQL scheduler repository | 1 suite, pass, including new result-state guards |
| Scheduler ownership PostgreSQL contracts | 1 suite, pass on each owned cluster |
| Focused native resume matrix | 7 scenarios, 7 pass, 0 fail |
| Final native scheduler/recovery suite | 27 scenario groups; final full run pending |
| Final capacity/cross-phase priority | 7 groups, 7 pass, 0 fail |
| Preserved terminal adapter, native sandbox enabled | 33 tests, 33 pass, 0 fail, 0 skip |
| Release provenance rejection | Expected exit 1; enforcement preserved |

Counts represent suites/named groups where those runners do not enumerate
individual assertions. Overlapping native/focused cases are not added into an
inflated aggregate count. Offline terminal tests also pass (27 pass / 6 native
cases skipped); the explicit --native run above removes those skips.

Commands:

    python3 -B Tests/Phase24YQualification.py offline
    python3 -B Tests/Phase24YQualification.py native --faults-only
    python3 -B Tests/Phase24YQualification.py native
    python3 -B Tests/Phase24YQualification.py priority
    python3 -B Tests/GlobalExperimentControlTransactionTests.py --cli DerivedData/ExpertAdvisor/Phase24Y/FinalQualification/build/LSTM_Release --regressions
    python3 -B Tests/RepositoryAgentTerminalTests.py --native

Passing global/repository/native-process/legacy/priority database evidence:
DerivedData/ExpertAdvisor/Phase24T/Phase24Y-Transactions-20261009005544.
Passing focused final-artifact resume evidence:
DerivedData/ExpertAdvisor/Phase24T/Phase24Y-FinalNative-20261009010450.
Passing final-artifact priority evidence:
DerivedData/ExpertAdvisor/Phase24T/Phase24Y-FinalPriority-20261009010601.
Final full native evidence:
DerivedData/ExpertAdvisor/Phase24T/Phase24Y-FinalNative-20261009010739.

Qualification iterations exposed fixture/schema defaults, FK cleanup ordering,
obsolete active-request/generation expectations, an overlapped FIFO candidate,
and an incorrect expectation that uncontested lease renewal must fail.
The exact-result recovery gap was corrected in source. Earlier failed logs were
retained; final results above refer only to corrected runs.

## Preservation and production boundary

preservation.json verifies **18 Phase24X/Qwen protected files**, **74 baseline
file hashes**, and **26,640 retained evidence entries** unchanged. Evidence
comparison uses lstat for links/directories and baseline SHA-256 hashes for
hashed regular files. RepositoryAgent source and live MCP configuration match
their baseline. Existing terminal regressions pass; no adapter activation occurred.

A read-only production census showed scheduler 55976, TRAIN 21282/21288 and the
pre-existing stopped INFER 52730 with the same identities and commands.
TRAIN processes were not stopped. Production LSTM_Release SHA-256 remains
8f7280c3a6151b384f32c8ac3adad13b9875c70c8d61945245241445c208d6b9.

No production PostgreSQL connection/write, worker signal, scheduler restart,
source/configuration write, binary replacement or migration was performed.
No live Qwen/MLX model was loaded. No commit, merge, push or deployment occurred.
Runtime scheduler crashes/signals/database writes were confined to owned inert
fixtures and isolated PostgreSQL at explicit 127.0.0.1:55485 with fresh SCRAM
credentials, disabled Unix sockets, verified pgdata and server system identifier.
No production schema dump was used.

## Remaining limits and publication prerequisites

- Recovery requires reachable PostgreSQL, persisted authority contracts and an
  authorized scheduler/controller. A foreign live administrative lease is not
  stolen; recovery latency includes its 30-second expiry and scheduler polling.
- Ambiguous/mismatched identities and inspection/permission failures remain
  fail closed. This phase supplies no blind repair for pre-existing untracked
  stops or deliberately corrupted identities. Missing TRAIN resumes use existing
  checkpoint selection; progress beyond the last durable checkpoint cannot be
  reconstructed after process loss.
- Native observation followed by POSIX group signaling is not an atomic kernel
  identity-bound signal API. Deterministic observable PID reuse is rejected;
  machine-power-loss and arbitrary kernel races were not exhaustively tested.
- Existing global pause scope covers primary TRAIN/INFER. Active ANALYZE and
  checkpoint children are excluded by the existing protocol. SIGSTOP does not
  release model allocations. This artifact does not qualify live Qwen loading,
  real GPU/checkpoint serialization, protocol cutover, or production resources.
- The isolated protocol-generation completion is fixture setup. No new migration
  is needed, but publication must verify the deployment's existing authoritative
  schema/migration sequence and generation-52 cutover.
- The private optimized relink is development evidence, not a complete fresh
  authorizable Release. Standalone scheduler publication remains separate.

Safe reproducible next qualification: review and explicitly authorize a commit
of the selected Phase24Y source/tests/report; create an isolated clean development
checkout of that actual commit, preserve this dirty worktree/evidence, resolve
supported dependencies without production writes, and run the unchanged Release
build/provenance workflow into that checkout's fresh DerivedData. Require empty
porcelain before/after, matching embedded source commit and artifact SHA/signature,
then rerun the isolated matrices against that exact Release artifact. Do not hide
untracked files or copy dirty source beneath a false commit identity.

Before any separate production publication decision, qualify the management CLI
and standalone scheduler together, exclude old controllers that still implement
signal-before-intent behavior, verify registry/bundle provenance and unchanged
priority/capacity limits, and perform staging pause/resume with real checkpoint
continuity and an explicit rollback procedure. This report authorizes no
production command or replacement.

## Final Git review

git status --short:

```text
 M Sources/GlobalExperimentControl.cpp
 M Sources/GlobalExperimentControl.hpp
 M Sources/SchedulerCore/PostgresSchedulerRepository.cpp
 M Sources/SchedulerCore/ProductionSchedulerDaemon.cpp
 M Tests/GlobalExperimentControlProcessTests.cpp
 M Tests/PostgresSchedulerRepositoryTests.cpp
?? Scripts/RepositoryAgentTerminal/
?? Tests/AnalyzeHistoricalFixtureQualification.py
?? Tests/AnalyzeHistoricalFixtureQualificationTests.py
?? Tests/AnalyzeHistoricalFixtureWorker.py
?? Tests/AnalyzeResourceInstrumentation.py
?? Tests/AnalyzeResourceInstrumentationTests.py
?? Tests/AnalyzeResourceProbe.cpp
?? Tests/AnalyzeResourceQualificationPreflight.py
?? Tests/AnalyzeResourceQualificationPreflightTests.py
?? Tests/AnalyzeSecondHistoricalFixtureQualificationTests.py
?? Tests/GlobalExperimentControlTransactionTests.py
?? Tests/Phase24YQualification.py
?? Tests/RepositoryAgentTerminalTests.py
?? Tests/RepositoryAgentTerminalValidation.py
?? docs/phases/Phase24/LSTM_Phase24X_RealAnalyzeResourceQualification_Output.md
?? docs/phases/Phase24/LSTM_Phase24Y_TransactionSafeGlobalPause_Output.md
?? docs/phases/Phase24/LSTM_RepositoryAgent_QwenTerminalIntegration_Output.md
?? docs/phases/Phase24/LSTM_RepositoryAgent_QwenTerminal_LiveQualification_Output.md
?? docs/phases/Phase24/LSTM_RepositoryAgent_QwenTerminal_RealMLXQualification_Output.md
```

git diff --stat (tracked files only; the two new Python runners and this report
are untracked and therefore not included):

```text
 Sources/GlobalExperimentControl.cpp                | 552 +++++++++++++++------
 Sources/GlobalExperimentControl.hpp                |  35 +-
 .../SchedulerCore/PostgresSchedulerRepository.cpp  |   2 +-
 .../SchedulerCore/ProductionSchedulerDaemon.cpp    |  48 +-
 Tests/GlobalExperimentControlProcessTests.cpp      | 390 ++++++++++++++-
 Tests/PostgresSchedulerRepositoryTests.cpp         |  41 ++
 6 files changed, 902 insertions(+), 166 deletions(-)
```
