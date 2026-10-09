# LSTM Trader — Qwen terminal real MLX qualification

**BLOCKED: no real model load, inference or terminal exchange was attempted.**
Two production TRAIN workers were executing on a 48 GiB host with substantial
wired/compressed memory and 15.6 GiB of occupied swap. Qwen's existing cached
weights alone occupy 16.001 GiB. Safe headroom for loading those weights plus
runtime/generation buffers alongside production was not established. The user's
resource-safety requirement therefore prevented loading Qwen. Production was
not paused to create a qualification window.

The existing controlled adapter remains unchanged. All 33 native regressions,
five original RepositoryAgent compatibility suites and the 15-request private
stdio validation PASS. These results qualify the existing adapter regression
behavior; they do not qualify a real Qwen exchange or authorize activation.

Repository: `/Volumes/Developer SSD/ExpertAdvisor-Rollover`.
Branch: `dedicated-train-layout-rollover-squashed-v1`.
HEAD: `15d9e9762ac9b7ff12e69ff7009cf391b7e8c715` (unchanged).
Observations: October 8, 2026 local time / October 9 UTC.

## Resource admission decision

The original integration report, Phase 24J restoration report, Phase 24U review,
root/shared AGENTS.md, adapter/policy and both requested validation files were
read. Historical successful MLX reviews were not counted as evidence for this
qualification. No Ollama API, installation or alternate backend was used.

Initial sandboxed process and sysctl observations were denied. Read-only
escalated observations succeeded. A first ten-second `vm_stat -c 6 2` sample
showed no new swapins or swapouts. Two retained snapshots ten seconds apart
then established:

| Observation | Retained sample 1 | Retained sample 2 |
| --- | --- | --- |
| UTC | 2026-10-09 04:23:51 | 2026-10-09 04:24:01 |
| Physical RAM | 48 GiB | 48 GiB |
| `memory_pressure -Q` free percentage | 47% | 47% |
| Kernel pressure level | 1 (normal) | 1 (normal) |
| Swap occupied | 15,970.62 MiB | 15,970.62 MiB |
| Swap free | 1,437.38 MiB | 1,437.38 MiB |
| Physical free pages | 0.640 GiB | 0.619 GiB |
| Inactive pages | 10.525 GiB | 10.400 GiB |
| Speculative pages | 0.569 GiB | 0.580 GiB |
| Wired pages | 16.608 GiB | 16.834 GiB |
| Compressor occupancy | 7.984 GiB | 7.980 GiB |
| Swapin / swapout delta across samples | — | 0 / 0 pages |

Normal pressure and stable swap describe the existing workload. They do not
demonstrate capacity for a new model load. The reported free percentage is not
an independently reserved 22.56 GiB allocation for Qwen. Physical free,
inactive and speculative pages together were approximately 11.6 GiB, and
inactive pages are not all guaranteed reclaimable without work. The 16 GiB
weight size is an on-disk observation, not a measured peak MLX allocation.
Loading/generation may require additional buffers and shared GPU memory.
The decision is conservative resource admission, not a claim that an outage or
current memory-pressure incident was observed.

Process census showed production TRAIN PIDs **21282** and **21288** executing
experiments 732/733, attempts 1410/1411, with roughly 2.0–2.2 GiB RSS each and
about 60–65% CPU in the initial census. Production INFER PID **52730**,
experiment 721/attempt 1405, was already `Ts` (OS-stopped). Scheduler PID
**55976** retained `--phase-priority=train:infer:analyze`, TRAIN=2, INFER=1,
ANALYZE=0. This is OS/process configuration evidence; no database-backed
scheduler status query was made. No ANALYZE worker was observed.

The three existing RepositoryAgent MCP processes had approximately 24 MiB RSS
each. This is consistent with unloaded lazy runtimes, but is not an inspection
of their Python objects. No live process was injected into, restarted or
requested to perform model inference.

The existing offline cache revision is
`6e302ea604ad9ab206367e2c501d1571023e7b6d`. Four existing safetensor shards
were statted without reading/loading their payloads; total size is
**17,181,071,994 bytes (16.001120 GiB)**. The model selection remains
`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, backend MLX / `mlx_lm`.

## Temporary invocation boundary and real acceptance status

The temporary staged stdio adapter was invoked for isolated validation under a
fresh private audit directory. It composes the existing RepositoryAgent server
and interface and advertises its 26 original tools plus four controlled
terminal tools. It never registered a connection in the live MCP configuration.

The existing MLX route was inspected: the interface's `_claim_verifier()` owns
`LazyClaimVerifierRuntime`; its existing `_ensure_loaded()` is the sole lazy
loading path; `QwenTerminalBridge` accepts only that interface's already-loaded
runtime with the existing model identity. The bridge neither loads weights nor
creates a second runtime. The private smoke call to `qwen_terminal_inspect`
correctly rejected an unloaded runtime with `no second loader`.

A model-enabled temporary invocation was **not established or run** beyond this
validated unloaded boundary because resource admission failed. No fallback
loader, alternate backend, monkeypatch or shared implementation change was
introduced to force qualification. The requested scheduler inspection question
was not sent to Qwen.

| Real-exchange requirement | Current evidence |
| --- | --- |
| Real MLX inference | NOT RUN; zero real model loads/turns |
| Model-generated command request | NOT RUN; zero real Qwen commands |
| Authorization of that command | NOT RUN; no real request to authorize |
| Sandboxed execution of that command | NOT RUN |
| Result returned to Qwen | NOT RUN |
| Source-grounded Qwen conclusion | NONE; no model finding claimed |
| Real command/outcome audit | No real command; BLOCKED admission recorded |

Exactly-one-exchange acceptance remains unmet. Zero exchanges were performed
under the explicitly requested BLOCKED safety branch. Mock exchanges inside
the regression suite and historical Phase 24J/24U calls do not satisfy it.

## Regression commands, authorization and audit

The smallest check ran first:

```text
python3 -B Tests/RepositoryAgentTerminalTests.py
```

PASS: 27 deterministic tests, six native tests skipped as designed.

The first full driver invocation used
`DerivedData/ExpertAdvisor/RepositoryAgentTerminal/RealMLXQualification-20261008-01/validation`.
It failed in all six native fixtures because the outer execution sandbox
rejected `sandbox-exec: sandbox_apply: Operation not permitted` (exit 71).
That failure and its exact command/output remain preserved. No adapter policy
or sandbox was weakened. The authorized retry ran outside the outer sandbox,
retaining every child's adapter-managed deny-default sandbox:

```text
python3 -B Tests/RepositoryAgentTerminalValidation.py DerivedData/ExpertAdvisor/RepositoryAgentTerminal/RealMLXQualification-20261008-01/validation-native
```

PASS, exit 0. The existing driver ran these exact module/test commands with its
private HOME/TMPDIR, offline mode, disabled bytecode and scoped PYTHONPATH:

```text
python3 -B Tests/RepositoryAgentTerminalTests.py --native
python3 -B -m Tools.RepositoryAgent.tests.test_repository_agent_mcp
python3 -B -m Tools.RepositoryAgent.tests.test_repository_agent_mcp_profiles
python3 -B -m Tools.RepositoryAgent.tests.test_repository_agent_mcp_claims
python3 -B -m Tools.RepositoryAgent.tests.test_codex_interface
python3 -B -m Tools.RepositoryAgent.tests.test_source_reader
python3 -B -m Scripts.RepositoryAgentTerminal.mcp_adapter --audit-directory "/Volumes/Developer SSD/ExpertAdvisor-Rollover/DerivedData/ExpertAdvisor/RepositoryAgentTerminal/RealMLXQualification-20261008-01/validation-native/mcp-audit" --approve-scheduler-phase-priority-test
```

Actual interpreter in driver command receipts: `/Volumes/Darwin/homebrew/bin/python3`.
All 33 native tests and all five compatibility suites passed. The 15-request
stdio smoke test returned eleven successes and four expected rejections.

| Controller-requested inspection/test | Authorization and outcome |
| --- | --- |
| `git_status` | Authorized; exit 0 |
| `git_diff`, `Sources/SchedulerCore/SchedulerPolicy.cpp` | Authorized; exit 0 |
| `git_log` | Authorized; exit 0 |
| `git_show`, `Sources/SchedulerCore/SchedulerPolicy.hpp`, HEAD | Authorized; exit 0 |
| `read_file`, same header, lines 1–5 | Authorized; exit 0 |
| `search_file`, policy cpp, literal `PriorityRank` | Authorized; exit 0 |
| `run_test`, `scheduler_phase_priority` | Startup-approved hash-pinned copy; compile and test exit 0 |
| `shell` | Rejected before spawn |
| Absolute production `AGENTS.md` path | Rejected before spawn |
| `PGHOST` environment injection | Rejected before spawn |
| `qwen_terminal_inspect` with unloaded runtime | Rejected before any generation/command |

These are controller/test requests, not real Qwen-generated requests. Native
regressions also validate the fixed executable catalog, shell-injection
rejection, destructive operations, symlink/hardlink escapes, unoffered mock
Qwen requests, output bounds, private timeout cleanup and kernel network/read/
write denial. No model bypass attempt occurred in this run; policy bypass
rejection is regression evidence only.

The copied pure C++ test was built using the direct Xcode clang++, explicit SDK,
`-std=c++20 -Wall -Wextra -Werror`, its five pinned source dependencies and a
private output. Exact absolute compiler, source, include, output and sandbox
argument arrays are retained in `validation-native/mcp-audit/audit.jsonl`.
Compilation and execution completed with no stdout/stderr diagnostics.
No full application Xcode build or Clean ran; no application source changed.

The final staged audit has 55 consecutive events, eight child starts and eight
exits (six inspections plus compile/test). All recorded children were absent at
validation completion. Request, policy, launch, executable/profile hashes,
session identities, deadlines and results are retained. No unsandboxed terminal
execution fallback was used. The native timeout fixture signals only its own
private sleep process; it does not signal a scheduler or worker.

## Preservation and production boundary

The configured Rollover MCP `capabilities` was called before and after
validation. Both responses retained protocol
`expertadvisor.repository.readonly.v1`, the same 26 operations and forbidden
shell/repository-write/Git-mutation/database/build/test/arbitrary-filesystem
capabilities. No terminal capability was activated in the live connection.
The original five compatibility suites independently passed.

Before/after hashes match for all shared RepositoryAgent Python sources,
shared instructions/configuration documentation, the existing adapter/tests,
the original terminal-integration report and `/Users/vjp/.codex/config.toml`.
The shared repository's pre-existing modified/untracked status is unchanged.
No shared source or model-selection setting changed.

All **ten Phase 24X files** retain their initial SHA-256 hashes. All **26,428**
entries beneath `DerivedData/ExpertAdvisor/Phase24X` retain their names,
sizes, nanosecond modification times, modes and inode identities. The **25,679**
regular evidence files at most 1 MiB also retain SHA-256 hashes. Larger evidence
was checked by metadata rather than fully streamed to avoid extra archive I/O;
this is not a cryptographic proof of every large archive's contents. No entry
was added, removed or overwritten in Phase 24X.

No production PostgreSQL connection, scheduler action, worker signal, production
configuration change or production file write was performed by this work.
Resource collection uses only OS observation. Tests use private fixtures and
mock claim verifiers. Terminal child sandboxes deny network and production
writes; the existing narrow shared Git metadata read exception is unchanged.
Writes are limited to new Rollover evidence/test scratch and this report.
No global production database/file-activity tracing was performed; naturally
running production workers continue their own normal activity.

## Evidence and remaining limitations

Evidence root:
`DerivedData/ExpertAdvisor/RepositoryAgentTerminal/RealMLXQualification-20261008-01`.

- `resource_preflight.py`, `resource-preflight.json`: exact read-only observation
  argv, timestamps, exits and filtered production process census.
- `qualification-summary.json`: BLOCKED admission, model shard sizes, resource
  counters and zero real inference/exchange counts.
- `preservation-before.json`, `preservation-after.json`: source/configuration,
  Phase 24X and retained-evidence preservation receipts.
- `live-mcp-capabilities-before.json`, `live-mcp-capabilities-after.json`: the
  identical observed live responses; the initial response was transcribed after
  collection and is not a transport-level capture.
- `validation/commands.json`: retained outer-sandbox native failure.
- `validation-native/commands.json`, `summary.json`, `mcp-requests.json`,
  `mcp-replies.json`, `mcp-audit/audit.jsonl`: passing driver and exact launch
  receipts. `validation-command-outcomes.json` summarizes inspected requests.

Only this new report and ignored private evidence were added. No operational
behavior changed. The existing terminal adapter, tests, model/backend,
RepositoryAgent implementation and live MCP configuration remain unchanged.

Real MLX proposal parsing, result consumption, source-grounded conclusion,
latency and contention remain unqualified. The existing bridge retains its
160/80 token limits and 2,048/1,024-character result delivery bounds. Its
acknowledgement prompt and evidence truncation must actually establish the
requested conclusion in a future run; a generic acknowledgement is insufficient.
Existing synchronous MLX generation has no independent wall deadline.

**No activation recommendation is issued because the real exchange did not
pass.** The smallest next development step is a fresh resource preflight during
a naturally safe window, followed, only if admitted, by one temporary bounded
invocation through the same interface/runtime and controlled adapter. Do not
change live MCP configuration or production controls to obtain this result.
A retry needs an owned-process wall deadline/resource monitor and evidence
ranges that fit delivery bounds. No production pause/resume, second backend,
commit, merge, push, publication or deployment is implied.

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
?? docs/phases/Phase24/LSTM_RepositoryAgent_QwenTerminal_RealMLXQualification_Output.md
```

`git diff --stat`: empty (the new report is untracked; evidence is ignored).
`git diff --check`: PASS.
