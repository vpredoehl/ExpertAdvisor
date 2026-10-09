# LSTM Trader — Controlled pause and real Qwen live qualification

**BLOCKED IN PHASE 1, BEFORE PRODUCTION PAUSE. Production remains running.**
The current pause implementation sends SIGSTOP before its database transaction
commits. A subsequent failure/timeout can roll back the pause-generation and
worker-state records while leaving a native worker stopped. The supported
resume-all path queues only members of a persisted pause generation; the
current reconciliation planner does not resume a native-stopped worker whose
persisted attempt remains running. Safe restoration of this failure case could
not be established using the two authorized commands alone.

The user's instruction to stop before pausing when lifecycle safety cannot be
established therefore applied. **Neither production `--pause-all-experiments
--yes` nor `--resume-all-experiments --yes` was issued.** No model was loaded.
The requested real exchange remains unqualified. Production never entered a
pause window, and no manual restoration is needed for this task.

Development HEAD remains `15d9e9762ac9b7ff12e69ff7009cf391b7e8c715`, branch
`dedicated-train-layout-rollover-squashed-v1`. Observations span approximately
23:56 October 8 through 00:02 October 9, 2026 America/Chicago (04:56–05:02 UTC).

## 1. Production preflight and lifecycle review

Read the original Qwen terminal integration report, previous real-MLX blocked
qualification, Phase 24J remediation/restoration, Phase 24U review and root/
production AGENTS.md. Inspected the current production global control,
CLI dispatch, scheduler launch gate and reconciliation implementation. The
current source takes precedence over historical reports and documentation.

Production binary:

```text
/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release
SHA-256: 8f7280c3a6151b384f32c8ac3adad13b9875c70c8d61945245241445c208d6b9
Size: 13,915,248 bytes
Modified: October 8, 2026 17:10:56 local
```

Native symbol inspection finds `RunPriorityQueueGlobalControl`,
`RunCommandWithProcessOperationsForTesting` and `PauseWorkerAuthorized` in the
specified binary. Its documented pause and resume dry runs both succeeded:

```text
"$BIN" --pause-all-experiments --dry-run
GLOBAL_EXPERIMENT_CONTROL_DRY_RUN,action=pause_all,previous_state=running,resulting_state=paused,targets=13,signals=verified_SIGSTOP

"$BIN" --resume-all-experiments --dry-run
GLOBAL_EXPERIMENT_CONTROL_DRY_RUN,action=resume_all,previous_state=running,resulting_state=running,targets=0,signals=none
```

These are dry runs, not execution or restoration receipts. This version's
priority-queue dry run does not individually validate/signal its worker targets.
Native worker identities were independently captured through the existing
read-only process-inspection helper and compared with authoritative attempts.

The broad `"$BIN" --scheduler-status` command exited **2** with
`EXPERIMENT_DATABASE_ERROR,error=ERROR: canceling statement due to statement timeout`.
Its deliberately bounded read-only session used a 3,000 ms statement timeout.
Smaller PostgreSQL snapshots succeeded in repeatable-read read-only transactions
with a 1,000 ms lock timeout and five-second connection timeout. They establish
the control state, active authority, protocol, exact active attempts, experiment
epochs and referenced model metadata without retrying the broad expensive query.
Destination evidence is retained: LSTM, localhost:5432, transaction read-only on.
No SQL UPDATE/INSERT/DELETE or migration was issued.

Scheduler identity:

| Field | Observed value |
| --- | --- |
| PID / process group | 55976 / 55974 |
| Kernel process-start identity | `1791497212:881093` |
| Executable | Production `DerivedData/ExpertAdvisor/Build/Products/Release/lstm-scheduler` |
| Invocation | `scheduler:43602f02a92cd86b2bf906ed43c9bcb67f7170e88beda755` |
| Authority / fencing token | active / 201 |
| Protocol | generation 52, cutover complete |
| Global control | running; active request NULL; pause generation NULL; revision 342 |
| Configuration | train:infer:analyze; TRAIN=2, INFER=1, ANALYZE=0; poll 30 seconds |

The exact native command, executable and start identity are retained in
`preflight.json`. The database invocation command contains an extra historical
`--schedule-experiments` token relative to native argv; this discrepancy is
recorded rather than silently normalized. PID/group/start/executable agree;
authority and heartbeat are verified independently. No scheduler signal,
restart or configuration change occurred.

| Experiment | Phase / status | PID / group | Kernel start identity | Attempt | Epoch |
| --- | --- | --- | --- | --- | --- |
| 732 | TRAIN / running | 21282 / 21282 | `1791517265:69827` | 1410 | 4 |
| 733 | TRAIN / running | 21288 / 21288 | `1791517265:98930` | 1411 | 4 |
| 721 | INFER / pending; native Ts | 52730 / 52730 | `1791496680:811836` | 1405, stopped | 20 |

All three priorities are low. The INFER row already has `resume_requested=true`
and a stopped authoritative attempt; its stopped native process was not caused
by this task. Its OS parent is PID 1, while its identity remains bound to the
durable attempt. Both TRAIN workers use the original published layout-9 worker
`.../e964fa9e335e9ae63918187a7ffee7aa77f32b4b/f56342895009e19cc69751259590cd945d64e5bfb5ffcd330f6aefc8d3fd26e9/LSTM_Release`.
INFER uses the published layout-9 inference worker
`.../dc3c10fecc1990a9eb4194265322820ac2d95a384639603c9070bdaf9e6ede8d/lstm-infer-worker`.
Full executable paths, commands and persisted identities are in the snapshots.
No active ANALYZE or checkpoint worker was found. There were two running,
23 pending and zero paused experiment rows.

The lifecycle review establishes:

- `NormalSchedulingAllowed` requires global desired state running and no active
  administrative request. Scheduler dispatch uses the shared coordination lock
  and this gate. A committed pause blocks new ordinary worker dispatch.
- A successful TRAIN/INFER pause validates exact attempts/process identities,
  SIGSTOPs their groups and retains the same stopped attempts. It preserves
  in-memory progress rather than terminating/saving/restarting the workers.
  SIGSTOP does not guarantee memory release or create a new durable checkpoint.
- Resume queues current-generation paused TRAIN/INFER rows as pending with
  `resume_requested=true`, clears the global generation/gate and sends no eager
  SIGCONT. The scheduler remains responsible for exact-identity, capacity-limited
  admission. It does not reset epochs, replace scientific identities or delete
  checkpoints/continuation policy fields.
- `RunPriorityQueueGlobalControl` filters out ANALYZE and checkpoint workers.
  The global gate blocks new work, but this path does not stop an already-active
  worker in those excluded categories. A live-state guard could exclude them
  from this particular run; universal active-phase coverage is not established.

## 2. Why bounded production pause was rejected before execution

The decisive failure path is visible in production
`Sources/GlobalExperimentControl.cpp`:

1. `RunPriorityQueueGlobalControl` creates one `pqxx::work`, acquires coordination
   and writes the request/global generation inside that transaction (3734–3827).
2. Its worker loop calls `PauseWorker` (3930–3947), which can issue native SIGSTOP
   (1988–2044), before the transaction has committed.
3. Attempt/experiment updates, affected-row checks and request accounting follow
   the signal (3950–4032). Commit occurs at line 4033.
4. This path has no catch/rollback compensation for a signaled worker. Its
   resume filter requires a persisted current generation and matching paused
   member (3763–3777); queuing uses that generation (3830–3849).

Thus a database error, affected-row failure or forced controller timeout after
SIGSTOP but before commit can leave the native stop in place while database
generation/attempt changes roll back. Merely placing the exact resume CLI in a
Python `finally` or independent watchdog cannot recreate missing membership.
No manual SIGCONT or SQL repair is authorized as an alternative.

Checked whether ordinary reconciliation closes that gap. The current
`PlanAttemptObservation` returns `RetainLive`, lifecycle `observed`, capacity
consumed and `restoreRunningLifecycle=false` for an exact live process that is
native-stopped but not persisted stopped. The production daemon persists that
observation; the inspected branch sends no SIGCONT. A pure C++ characterization
using the current production planner confirms this case and two neighboring
cases. It is not a live PostgreSQL rollback reproduction or proof of every
scheduler path. It is sufficient to withhold the required bounded-restoration
qualification until a supported recovery path is established.

The production control documentation says signals occur after request commit.
That description does not match this current priority-queue pause path. No
production document or source was edited to reconcile the discrepancy.
Native symbols and dry runs establish the binary's corresponding path exists;
full disassembly/provenance equivalence was not established. No evidence shows
that the supplied binary independently repairs the observed source-level gap.

An evidence-only draft controller, model invocation and independent watchdog
were prepared before this final lifecycle review. They were never run or armed.
All operational draft entry points now fail closed on `PREFLIGHT_BLOCKED.json`.
The three mocked wrapper tests confirm `finally` calls resume after pause,
inspection, model failure or KeyboardInterrupt and exposes restoration failure.
They do not qualify the application's failed-pause recovery semantics.

## 3. Resources before pause and at final verification

No after-pause measurement exists, because no pause was issued. The final
measurement below is an ordinary live production observation, not evidence of
pause-related memory release.

| Observation | Preflight sample 1 | Preflight sample 2 | Final sample 1 | Final sample 2 |
| --- | --- | --- | --- | --- |
| UTC | 04:56:14 | 04:56:19 | 05:02:06 | 05:02:11 |
| Physical RAM | 48 GiB | 48 GiB | 48 GiB | 48 GiB |
| Pressure level | normal (1) | normal (1) | normal (1) | normal (1) |
| Reported free percentage | 46% | 48% | 46% | 46% |
| Physical free | 1.057 GiB | 1.189 GiB | 0.275 GiB | 0.292 GiB |
| Inactive | 10.113 GiB | 10.512 GiB | 10.863 GiB | 10.842 GiB |
| Wired | 17.429 GiB | 16.493 GiB | 17.509 GiB | 17.444 GiB |
| Compressor occupancy | 7.877 GiB | 7.876 GiB | 7.844 GiB | 7.844 GiB |
| Swap occupied | 15,954.62 MiB | 15,954.62 MiB | 15,938.62 MiB | 15,938.62 MiB |
| Swapins | 9,918,708 | 9,918,708 | 9,919,172 | 9,919,172 |
| Swapouts | 14,526,081 | 14,526,081 | 14,526,081 | 14,526,081 |

Both five-second windows show zero swapin/swapout deltas. Across the overall
observation interval, 464 pages swapped in and none swapped out. Swap occupancy
fell 16 MiB naturally. No causal attribution to this work or capacity
qualification follows from those observations. TRAIN workers each retained
about 2 GiB RSS and continued executing.

The prior cache inspection established approximately 16.001 GiB of existing
Qwen weights, before MLX runtime buffers. No new model-residency measurement was
made. A future admitted run must establish physical/reclaimable headroom,
pressure, swap and remaining-worker memory together; free percentage alone is
insufficient. A SIGSTOP-only pause is not a reliable memory-release mechanism.

## 4–7. Qwen exchange, terminal audit, findings and cleanup

Real model loads: **0**. Real inference turns: **0**. Real terminal exchanges:
**0**. Real Qwen-generated commands/authorization decisions/results/conclusions:
**NONE**. No source-grounded Qwen finding about normal ANALYZE displacing low
TRAIN is claimed. That question remains pending independent model qualification.
The pure lifecycle test above is a Codex-controlled test, not a Qwen finding.

The draft uses the existing interface-owned lazy MLX loader and unchanged
adapter policy; it is disabled and has not been qualified. The original live
MCP still reports its original 26 read-only operations and forbidden shell,
database, build/test and repository-write capabilities. No terminal tools were
activated. No production-facing MCP or RepositoryAgent source was changed.

No controller, watchdog, Qwen or terminal-adapter process was launched. Final
process inspection found no such owned child. There is no armed watchdog,
pause-attempt marker or real-model audit directory. MLX resources require no
cleanup because none were allocated. The offline compiler/test and read-only
observer subprocesses completed. No application worker was signaled by tests.

## 8–9. Resume obligation and final production state

Production was never paused by this task, so no restoration command was needed
or issued. This is the explicit Phase 1 stop branch, not an interrupted pause
window or a failed resume. No request for the user to resume production remains.

Final read-only verification establishes:

- Global control is exactly unchanged: running, active request NULL, current
  pause generation NULL, revision 342.
- Scheduler PID/group/start/executable/command identity is unchanged; owner and
  fencing token 201 are unchanged. Its heartbeat advanced to 00:01:57 local and
  lease expiry to 00:03:27, demonstrating continued scheduler operation.
- The same two TRAIN workers and stopped INFER process remain bound to the same
  three authoritative attempts. No new/duplicate attempt or worker was observed.
- Experiments 732 and 733 advanced naturally from epoch **4 to 5**. Experiment
  721 remains pending INFER at epoch 20 with its existing stopped attempt and
  resume request. No claimed new dispatch test was performed.
- Across all 25 captured experiment rows, only TRAIN epoch values and their
  update times changed. All captured scientific/policy/checkpoint/continuation
  fields and referenced model metadata are unchanged. No checkpoint payload or
  model matrix parity test was run; no production lifecycle operation occurred.

The supported command for a genuine completed pause remains:

```text
"/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release" --resume-all-experiments --yes
```

It was verified by source review and dry run for the normal committed-generation
case. It is not asserted to repair the uncommitted-signal failure case above.
Production currently needs no recovery command from this task.

## 10. Validation, preservation and remaining limitations

Exact validation argv, exits and output are retained in `offline-validation.json`:

```text
python3 -B .../LiveQualification-20261008-01/control.py self-test
python3 -B Tests/RepositoryAgentTerminalTests.py
```

PASS: three mocked restoration tests; 27 deterministic adapter regressions with
the six native tests skipped as designed. The existing unchanged native adapter
and five compatibility suites passed in the preceding real-MLX qualification;
those results are preserved, not relabeled as fresh tests here.

The new pure planner characterization compiled and ran with exit 0 and no
compiler diagnostics. Build command (all output in development evidence):

```text
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk -std=c++20 -Wall -Wextra -Werror "-I/Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore" "/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/RepositoryAgentTerminal/LiveQualification-20261008-01/RollbackPauseObservation.cpp" "/Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/ReconciliationService.cpp" -o "/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/RepositoryAgentTerminal/LiveQualification-20261008-01/rollback-pause-observation"
"/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/RepositoryAgentTerminal/LiveQualification-20261008-01/rollback-pause-observation"
```

It checks three observation cases with explicit runtime failure returns. No
PostgreSQL, worker, model or scheduler is started by this binary. Python syntax
checks pass for all private scripts. No application Xcode build or Clean ran.

All 74 selected source/configuration/binary/registry hashes are unchanged,
including all ten Phase 24X files, existing Qwen integration files/reports,
shared RepositoryAgent Python source and Codex configuration. Production Git
status remains clean; shared RepositoryAgent status retains its initial changes.
All **26,640** existing Phase 24X/Qwen-evidence entries retain names, metadata
and, for regular files at most 1 MiB, SHA-256 hashes. Larger files were checked by
size, inode, mode and nanosecond mtime to avoid archive I/O; this is not a full
cryptographic proof of every large archive. No prior evidence was overwritten.

Only the requested new report and new ignored private evidence were added.
There was no production source/configuration write, direct database mutation,
manual worker signal, scheduler restart, concurrency/priority change, commit,
merge, push, deployment or Ollama interaction. Runtime environment overrides
were scoped to read-only observer subprocesses and changed no live configuration.

Evidence root:
`DerivedData/ExpertAdvisor/RepositoryAgentTerminal/LiveQualification-20261008-01`.
It contains `preflight.json`, `final.json`, bounded observation receipts,
dry-run/status receipts, `lifecycle-source-evidence.json`,
`RollbackPauseObservation.cpp`, compiler/test receipts,
`qualification-summary.json`, `cleanup-verification.json`, preservation
manifests and disabled controller/model/watchdog drafts. `PREFLIGHT_BLOCKED.json`
records the admission rejection and prevents their operational entry points.

Remaining limitations: real Qwen qualification, production pause/resume execution,
all-phase active-worker pause coverage, timeout/crash/rollback recovery, measured
MLX peak headroom and checkpoint-payload parity are not established. The
rollback finding is source/planner evidence, not a deliberately induced
production failure. No activation recommendation is justified. The required
next prerequisite is qualifying a supported restoration path for failed pause
in an isolated environment; this task did not redesign or deploy that workflow.

Final `git status --short`:

```text
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
?? Tests/RepositoryAgentTerminalTests.py
?? Tests/RepositoryAgentTerminalValidation.py
?? docs/phases/Phase24/LSTM_Phase24X_RealAnalyzeResourceQualification_Output.md
?? docs/phases/Phase24/LSTM_RepositoryAgent_QwenTerminalIntegration_Output.md
?? docs/phases/Phase24/LSTM_RepositoryAgent_QwenTerminal_LiveQualification_Output.md
?? docs/phases/Phase24/LSTM_RepositoryAgent_QwenTerminal_RealMLXQualification_Output.md
```

`git diff --stat`: empty; additions are untracked/ignored.
`git diff --check`: PASS.
