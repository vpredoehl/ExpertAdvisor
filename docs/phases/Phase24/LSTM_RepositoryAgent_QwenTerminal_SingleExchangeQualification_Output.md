# LSTM Trader — Real Qwen MLX terminal qualification

**PASS: one real read-only Qwen command-and-response exchange completed after the JSON fence parser correction.** Qwen loaded once, proposed a Markdown-fenced JSON command, the existing parser accepted it, the allowlisted command ran in the existing sandbox, and its exact result was returned to Qwen. Qwen supplied a source-grounded acknowledgement. No retry or additional exchange occurred in this qualification.

Observed October 9, 2026, **2026-10-09T06:24:08.593897+00:00–2026-10-09T06:24:21.711034+00:00**. Branch `dedicated-train-layout-rollover-squashed-v1`; HEAD `15d9e9762ac9b7ff12e69ff7009cf391b7e8c715`.

| Check | Result |
| --- | --- |
| Model/cache | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`; cached revision `6e302ea604ad9ab206367e2c501d1571023e7b6d`, four shards totaling 17,181,071,994 bytes; offline MLX 0.32.3 / mlx-lm 0.31.3 |
| Model load | PASS; sole interface-owned `LazyClaimVerifierRuntime._ensure_loaded()`, 7.89 seconds; same model/tokenizer reused by the adapter |
| Proposal parsing | PASS; one `json`-fenced object parsed and matched the sole host-offered command |
| Authorization/execution | PASS; `read_file`, `Sources/SchedulerCore/SchedulerPolicy.hpp`, lines 1–12; terminal PID 30117, exit 0, empty stderr, no timeout |
| Result delivery | PASS; all 351 bytes exactly match the repository range; no output truncation; receipt and `qwen_result_delivered` audited |
| Qwen acknowledgement | PASS; correctly identified `EA::SchedulerCore`, `PriorityRank`, and `ResumeOriginRank` |
| Exchange count | Exactly one `qwen_terminal_inspect`, one command child, two model generations (proposal and acknowledgement); no retries |
| Duration | 4.09 seconds for exchange; 13.20 seconds including model load and monitoring |
| Memory pressure | Baseline kernel level 1 / 56% free; peak observed level **2**, minimum free **24%**; final level 1 / 57% free |
| Swap | 21,643.62 → 22,918.56 MiB; observed peak 22,918.56 MiB; peak and net growth **1,274.94 MiB (1.245 GiB)** |
| MLX peak allocation | 17,819,278,918 bytes (**16.595 GiB**) |
| Production continuity | Scheduler and both TRAIN identities unchanged throughout 8 retained samples; both TRAIN CPU times advanced; no stopped TRAIN state observed |

Exact Qwen proposal:

````text
```json
{"operation": "read_file", "path": "Sources/SchedulerCore/SchedulerPolicy.hpp", "start": 1, "end": 12}
```
````

Exact terminal argv, launched through `/usr/bin/sandbox-exec` with the existing deny-default profile:

```text
/usr/bin/sed -n '1,12p;12q' '/Volumes/Developer SSD/ExpertAdvisor-Rollover/Sources/SchedulerCore/SchedulerPolicy.hpp'
```

Qwen acknowledgement:

> The namespace identified is `EA::SchedulerCore` and the two ranking function declarations are `PriorityRank` and `ResumeOriginRank`.

The original audit retains the unmodified model proposal, host authorization, executable/profile hashes, sanitized environment, terminal process identity and exit, linked result receipt, acknowledgement and successful MCP result (`isError: false`). All eleven audit events are consecutive. The terminal child and private model invocation exited normally; no cleanup or production signal was sent. Existing 10-second command deadline, 65,536-byte output bound, allowlist and filesystem/network restrictions were unchanged.

Production identities recorded before loading and retained afterwards:

| Role | PID / parent | Start identity | Work identity | CPU time before → after |
| --- | --- | --- | --- | --- |
| Scheduler | 55976 / 55974 | Thu Oct 8 17:06:52 2026 | Existing `lstm-scheduler`; TRAIN=2, INFER=1, ANALYZE=0; `train:infer:analyze` | 0:21.02 → 0:21.05 |
| TRAIN | 21282 / 55976 | Thu Oct 8 22:41:05 2026 | Experiment 732, attempt 1410 | 102:48.57 → 102:53.78 |
| TRAIN | 21288 / 55976 | Thu Oct 8 22:41:05 2026 | Experiment 733, attempt 1411 | 103:01.53 → 103:06.71 |

Complete executable paths, arguments, process states and start identities are retained in `before.json`, `after.json` and `observations.jsonl`. Scheduler log growth was **12,440 bytes** during observation. The two TRAIN workers continued operating. Existing INFER PID 52730 retained its identity and pre-existing stopped state and was untouched. No scheduler CLI or PostgreSQL connection was used.

This qualifies the single exchange and observed process continuity. It does not establish unchanged training throughput or absence of transient memory contention. Approximately two-second sampling can miss shorter pressure peaks; OS identity evidence does not validate database leases/fencing.

Files changed in this qualification: **this existing report only**, plus new ignored private evidence under `DerivedData/ExpertAdvisor/RepositoryAgentTerminal/RealMLXSingleExchange-20261009-02/`. The existing qualification observer was copied unchanged into that fresh directory. Before/after preservation receipts match for the adapter, shared RepositoryAgent Python sources, Codex configuration, pre-existing modified files and inspected Phase 24Y files. No adapter, scheduler, Phase 24Y or production changes were made; live MCP configuration remains unchanged. No production controls/signals, PostgreSQL writes, tests, frameworks, Release builds, commits, merges, pushes or deployments occurred.

Exact qualification command:

```text
/Users/vjp/LLM/mlx-env/bin/python -B DerivedData/ExpertAdvisor/RepositoryAgentTerminal/RealMLXSingleExchange-20261009-02/qualify.py
```

The private child uses the existing adapter's MCP request handler and default MLX generation function. Loading the same interface-owned runtime directly avoids an extra claim-verification exchange. This is a temporary qualification invocation, not live MCP activation. Evidence includes invocation/environment, cached-model manifest, model state, request/reply, audit, resource/process observations and preservation receipts. **No remaining blocker for this one-exchange qualification.**

Previous result: the pre-fix attempt at 06:18 UTC loaded Qwen but rejected the fenced proposal before terminal execution. Its original evidence remains in `RealMLXSingleExchange-20261009-01/`; the previous report is preserved as `RealMLXSingleExchange-20261009-02/report-before.md`. This new, separately authorized qualification supersedes that parsing failure.

Final `git status --short`:

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
?? docs/phases/Phase24/LSTM_RepositoryAgent_QwenTerminal_SingleExchangeQualification_Output.md
```

Final `git diff --stat` (pre-existing tracked changes only; report is untracked):

```text
 Sources/GlobalExperimentControl.cpp                | 552 +++++++++++++++------
 Sources/GlobalExperimentControl.hpp                |  35 +-
 .../SchedulerCore/PostgresSchedulerRepository.cpp  |   2 +-
 .../SchedulerCore/ProductionSchedulerDaemon.cpp    |  48 +-
 Tests/GlobalExperimentControlProcessTests.cpp      | 390 ++++++++++++++-
 Tests/PostgresSchedulerRepositoryTests.cpp         |  41 ++
 6 files changed, 902 insertions(+), 166 deletions(-)
```

`git diff --check`: PASS. Work stopped after recording this result.
