# LSTM Trader — RepositoryAgent Qwen Terminal Integration

## Result and live integration status

**The development-only adapter and command policy are implemented and validated in Rollover. They are not activated in the configured live MCP connection.** All 33 regressions, five original compatibility suites, and the stdio MCP validation pass. Six real read-only inspections and one copied, hash-pinned pure C++ test complete successfully. No private validation child remains.

**A real Qwen terminal request/result exchange has NOT been demonstrated.** The bridge exchange is tested with a mock of the existing MLX runtime. It never creates or loads a model and requires the interface's existing runtime to be loaded before a real exchange. The current live `expertadvisor-repository-rollover` still advertises its original read-only tools; capability checks before and after implementation agree. No model-backed MCP operation was invoked in this Codex session. Earlier Phase 24J/24U model invocations are historical evidence, not evidence of a terminal exchange in this task.

Activating terminal tools in the configured live connection would require a separately reviewed development-only activation/restart or integration into its shared implementation. Deployment and production/shared-source changes are outside this task. The deliverable therefore remains a concrete, reproducible adapter for review; live activation and real-model acceptance remain unverified. The existing configuration and production boundaries were not changed to obtain a passing result.

Repository: `/Volumes/Developer SSD/ExpertAdvisor-Rollover`; branch `dedicated-train-layout-rollover-squashed-v1`; HEAD `15d9e9762ac9b7ff12e69ff7009cf391b7e8c715`. All ten pre-existing Phase 24X files and their evidence are preserved, with file hashes checked after implementation.

## Phase A: existing architecture and audit

Read the root `AGENTS.md`, shared RepositoryAgent instructions, Phase 24J remediation/restoration report, Phase 24U independent-review report, MCP configuration documentation, server/dispatch, source reader, retrieval, claim-verifier and generation code. No configuration changed during the audit or implementation.

The configured Rollover MCP launches:

```text
/Users/vjp/LLM/mlx-env/bin/python -m Tools.RepositoryAgent.repository_agent_mcp
cwd /Volumes/Developer SSD/ExpertAdvisor-RepositoryAgent
PYTHONPATH /Volumes/Developer SSD/ExpertAdvisor-RepositoryAgent
EXPERTADVISOR_REPOSITORY_ROOT /Volumes/Developer SSD/ExpertAdvisor-Rollover
EXPERTADVISOR_LEDGER_CACHE_NAMESPACE ExpertAdvisor-Rollover
EXPERTADVISOR_CLAIM_EVIDENCE_LEDGER /Users/vjp/Library/Caches/ExpertAdvisor-Rollover/RepositoryAgent/verified_claims.json
HF_HUB_OFFLINE 1
PYTHONDONTWRITEBYTECODE 1
```

The shared implementation is **outside Rollover** and is also used by `expertadvisor-repository` and `expertadvisor-repository-assisted`, whose default source root is production. Its existing six modified and two untracked files predate this task. Their status was preserved. Selected source/configuration files were snapshotted and hashed under `DerivedData/ExpertAdvisor/RepositoryAgentTerminal/Audit-20261008-01/shared-snapshot`; the Codex configuration was hashed without copying credentials or changing it. `preservation-after.json` verifies unchanged source/configuration hashes, shared status and Phase 24X file hashes.

Existing responsibilities:

| Component | Existing responsibility |
|---|---|
| `repository_agent_mcp.py` | Dependency-free stdio JSON-RPC framing, startup-fixed `full` / `codex_assisted` profiles, advertised closed tool schemas and dispatch admission |
| `codex_interface.py` | Deterministic source/index/relationship operations, bounded investigation workflows and dedicated claim-ledger decisions |
| `source_reader.py` | Startup-selected source root, normalized relative paths, source allowlist and resolved-target escape checks |
| `repository_index.py` / `retrieval.py` | Structural navigation and bounded source retrieval; retrieval `execute_tool` supports only list/search/read |
| `claim_verifier.py` | Lazy, reusable `LazyClaimVerifierRuntime` owned by the interface; source-grounded semantic verification |
| `verifier.py` | Existing `mlx_lm.generate` wrapper with bounded generation tokens and stdout isolation |
| `/Users/vjp/LLM/expertadvisor_agent.py` | Shared default production reader, unchanged; exposes allowed file listing/search/read, no terminal subprocess interface |

The model remains **`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`**, backend MLX / `mlx_lm`. The interactive RepositoryAgent CLI has a separate existing model-loading path; it was not launched or modified. No Ollama replacement, additional model load, model-server restart or package installation occurred.

The existing MCP exposes 26 full-profile tools: capabilities; list/search/read; index/symbol/function/relationship/trace/source excerpts; legacy and claim ledgers; source/bundle verification; source, bundle, direct/chain/set/operation relationship, symbol and subsystem investigations; catalog/path discovery and catalog-child navigation. Its protocol remains `expertadvisor.repository.readonly.v1`. It forbids shell, repository writes, Git mutation, database access, builds, tests and arbitrary filesystem access. Dedicated claim-ledger writes are its existing controlled-state exception. These tools are source analysis, not OS terminal execution. Phase 24U's rejected general test-source bundle demonstrated that boundary; no prior test or shell capability was found.

Production scheduler configuration was observed only through an OS process census. Its command remains TRAIN=2, INFER=1, ANALYZE=0. No scheduler CLI, live PostgreSQL connection, worker control, pause/resume or production experiment mutation was used.

## Phases B/C: independent authorization and staged interface

The extension is composed with the existing `StdioMCPServer` and its interface. It does not patch shared modules, change global tool/profile definitions, broaden the existing source reader or replace the existing model runtime. The local adapter refuses startup unless the explicit source root is Rollover. It adds four separate tools to the staged server, retaining all 26 original tools unchanged:

| Staged tool | Controlled behavior |
|---|---|
| `terminal_capabilities` | Reports the separate terminal policy, bounds and startup test approval |
| `terminal_inspect` | Validates one structured inspection request and returns an audited command result |
| `terminal_test` | Selects the single approved pure test by ID; disabled unless trusted startup explicitly approves it |
| `qwen_terminal_inspect` | Reuses the already-loaded interface runtime to select one of 1–4 host-validated read-only requests, execute through the independent policy and receive a bounded structured result |

The read-only catalog is `git_status`, `git_diff`, `git_log`, `git_show`, `read_file`, `search_file`. The caller cannot provide an executable, raw argv, shell string, environment, timeout override, approval assertion or alternate working directory. Each operation builds a fixed absolute executable path and explicit argument array. `Popen` uses `shell=False`, a sanitized environment, closed inherited descriptors and no stdin. The controller's operation catalog, source policy and startup test approval authorize execution; Qwen's response does not.

Example inspection payloads:

```json
{"operation":"git_status"}
{"operation":"git_diff","path":"Sources/SchedulerCore/SchedulerPolicy.cpp"}
{"operation":"git_log"}
{"operation":"git_show","path":"Sources/SchedulerCore/SchedulerPolicy.hpp","revision":"HEAD"}
{"operation":"read_file","path":"Sources/SchedulerCore/SchedulerPolicy.hpp","start":1,"end":12}
{"operation":"search_file","path":"Sources/SchedulerCore/SchedulerPolicy.cpp","pattern":"PriorityRank"}
```

Git status is short format; log is limited to ten commit identities/subjects; show accepts only HEAD or a full lowercase 40-character object identity and an approved file. Diff disables external diff/text conversion and defaults to approved source/evidence paths. No push, merge, reset, clean, checkout, arbitrary revision expressions, submodule commands, hooks, pager, credentials or network operations are admitted. Git optional locks are disabled, fsmonitor/hooks/external diff/credential helper settings are overridden, user/system configuration is suppressed and the child filesystem sandbox prohibits Git metadata writes.

Reads accept normalized existing regular files under `Sources`, `Headers`, `LSTM`, `Tests`, `Scripts`, `docs`, or the exact `AGENTS.md` / Xcode project files. They reject absolute paths, traversal, hidden components, symlinks at every component, multiply-linked files, non-files and files larger than 8 MiB. File reads are limited to 500 lines; literal searches are limited to 100 matches in one explicit file. Reads/searches use fixed `sed`/`grep` arguments, not shell interpretation. The sed line-range expression is constructed only from validated integers; the caller supplies no script. The original MCP source-reader allowlist remains separate and unchanged.

Shell metacharacters, chaining, pipes and redirections in caller input are rejected. No environment-variable injection or privilege-escalation interface exists. Unknown request fields are denied, and terminal inspection cannot switch to test execution. Public request admission is bounded to 8,192 serialized bytes.

### Filesystem, process and network controls

Every child runs under `/usr/bin/sandbox-exec` with **deny default**. No unsupported-platform or unsandboxed fallback exists. Network access is not granted, including loopback PostgreSQL access. Content reads are confined to the approved repository/snapshot, private scratch and explicit system/toolchain runtime locations. Direct file writes are allowed only in fresh private scratch/test directories and `/dev/null`; source and production writes are denied. The approved test binary runs against its copied snapshot, not the live worktree or production tree.

On macOS 27, dyld/libignition must open the literal filesystem root directory as an `openat` root. Apple's installed `/System/Library/Sandbox/Profiles/dyld-support.sb` documents this requirement. The profile grants that literal directory read plus executable mapping at admitted runtime locations; it does **not** grant reads of arbitrary descendants. Unapproved file-content reads, writes and loopback connection attempts were rejected by the kernel in private negative fixtures.

Git needs a narrow read-only exception because Rollover is a linked worktree whose Git metadata physically resides under production `.git`. The adapter verifies the exact Rollover gitdir binding and `commondir`, then admits only its worktree metadata and required common objects/refs/logs/config/packed-refs/shallow/HEAD/info. It does not admit production source/data/registries or production index writes. Caller-supplied production paths remain rejected. This exception is a shared Git dependency, not a production PostgreSQL or scheduler interaction.

Executable hashes are captured by the trusted policy and rechecked; launch receipts retain the resolved executable path, executable and sandbox hashes. Commands have a 10-second default wall deadline and a combined 65,536-byte stdout/stderr retention limit. Output is drained after truncation, with observed byte counts and truncation flags. Failures include exit status and bounded stdout/stderr; no-match grep exit 1 remains a command failure rather than fabricated success.

Each command owns a fresh process session and retained `Popen` child handle. Timeout cleanup verifies its process group before signaling. The PID is not polled/reaped before timeout cleanup, preventing PID reuse in that cleanup window. Only the owned session is signaled; scan-discovered or production PIDs are never signal targets. Signal and exit receipts are audited. A successful command is not evidence of successful Qwen semantic analysis.

### Isolated test authorization

The startup-approved catalog contains only `scheduler_phase_priority`. Its five reviewed dependencies have fixed SHA-256 identities in the policy:

- `Tests/SchedulerPhasePriorityTests.cpp`
- `Sources/SchedulerCore/SchedulerPolicy.cpp`
- `Sources/SchedulerCore/SchedulerPolicy.hpp`
- `Sources/SchedulerCore/SchedulerPhasePriority.hpp`
- `Sources/SchedulerCore/SchedulerPhasePriorityService.hpp`

A changed dependency is rejected rather than approved from its current contents. Test copies use directory-relative `openat`/`O_NOFOLLOW` reads, remain mode 0400 and preserve exact bytes. Compilation and execution use distinct fresh runtime directories and explicit argument arrays. The compiler uses the direct Xcode clang++ path, explicit read-only SDK, C++20 and `-Wall -Wextra -Werror`. Compilation is bounded to 50 seconds and execution to ten seconds. The test has no PostgreSQL, scheduler daemon, worker, network or model dependency. Arbitrary scripts, Python commands and full application builds are not accepted test IDs.

Approval is a trusted process-start flag, `--approve-scheduler-phase-priority-test`; it cannot be asserted by model output or changed by an MCP request. Without it, the test tool rejects execution.

### Qwen request/result bridge and audit

The bridge uses the interface's existing `_claim_runtime`, model and tokenizer. It refuses an unloaded/missing runtime or a different model identity; it calls neither `load` nor `_ensure_loaded`. Host offers are validated before model generation and checked again at execution. Model output must exactly match one offered request. A non-offered request, invalid JSON, altered fields or unavailable runtime fails closed. Only one command can run; the acknowledgement is never interpreted as another command.

The proposal uses at most 160 generation tokens and acknowledgement at most 80. Structured result delivery retains command/exit/identity/timeout/resource-bound metadata, with stdout/stderr prefixes bounded to 2,048/1,024 characters and an explicit additional-delivery-truncation flag. Terminal output is labeled untrusted data. This avoids feeding full terminal output into an unbounded model conversation. These are token bounds, not an independent wall-time guarantee for MLX generation.

The private audit directory is new, mode 0700; `audit.jsonl` is mode 0600 and fsynced. Each request has a unique ID, UTC timestamp and monotonic sequence. Records include request/proposer, policy decision, target/test hashes, exact launch argv, cwd, sanitized environment, sandbox identity, process session identity, deadline, signals, per-process exits and structured final results. Qwen proposals and receipt delivery are linked. Oversized rejected requests retain byte count and digest rather than unbounded payloads. Failed audit admission prevents process launch. No previous evidence directory is overwritten.

## Phase D: validation and retained evidence

Final evidence: **`DerivedData/ExpertAdvisor/RepositoryAgentTerminal/Validation-20261008-02`**. `summary.json` reports PASS; `commands.json` contains exact validation argv, deadlines, timestamps, exit codes and captured output; `mcp-requests.json` / `mcp-replies.json` retain all 15 stdio protocol requests/results. `mcp-audit/audit.jsonl` contains 55 events for the final smoke validation, including all eight child starts and exits. Source hashes and private source/binary/sandbox artifacts are retained.

| Required case | Result |
|---|---|
| Approved read-only commands | All six catalog operations execute successfully against Rollover |
| Destructive commands | Unknown destructive Git/shell/worker/database/network operations rejected before spawn |
| Production paths | Public absolute production paths and production root rejected; narrow fixed Git metadata dependency kept read-only |
| Shell injection | Metacharacters, newlines, substitutions, pipelines and redirects rejected |
| Symlink escape | File/directory/internal symlinks and multiply-linked files rejected; private kernel fixture denies unapproved content reads |
| Timeout | Owned private sleep fixture times out, exits -9, session signal audited, PID absent afterwards |
| Output limit | Private 100,000-byte fixture retains exactly 65,536 bytes and reports truncation/counts |
| Command failure | No-match grep returns failure/exit 1 with structured output; prior failed compilation also remained a failure |
| Audit completeness | Sequence/request/decision/start/signal/exit/result checks pass; all final launches have exits and executable hashes |
| Existing MCP compatibility | Original 26 tools, profiles, capability contract, source reads and mock claim interfaces remain compatible |
| Network / write enforcement | Private loopback discard-port connect, unapproved private write and unapproved private read each fail with PermissionError |
| Second-model prevention | Mock bridge reuses exact model/tokenizer; unloaded runtime rejects and `mlx_lm` is not imported in deterministic/native validation |
| Actual Qwen terminal exchange | **NOT RUN**; mock request/result exchange passes |

The 33-test suite includes six native private fixtures. Five original RepositoryAgent suites additionally pass. The staged MCP advertises 30 tools (26 original plus four terminal tools). Of 15 smoke requests, the first eleven succeed and four reject shell execution, production path, environment injection and unloaded Qwen runtime. Six inspections plus compiler/test produce eight child processes; all exited. No application worker or database was launched.

Retained earlier evidence:

- `NativeValidation-20261008-01`: read-only commands work; a deliberate no-match search is reported as failure and compilation fails without an explicit SDK.
- `NativeValidation-20261008-02`: read-only operations and compilation succeed; execution stops on a private runtime-directory collision.
- `NativeValidation-20261008-03`: all six read-only operations and isolated test pass after explicit SDK and distinct runtime directories.
- `Validation-20261008-01`: earlier complete 33-test/compatibility/stdio PASS before adding explicit executable/test hash fields to audit receipts.

Initial private sandbox probes aborted in dyld before command startup. They established the literal-root runtime requirement above; no network or production permission was added to fix it. A negative Python fixture initially used the xcrun launcher and then tested socket creation instead of outbound connection; the final fixture uses the direct Xcode Python binary and verifies actual kernel connection denial. These setup failures are not claimed as passing validations.

Exact reproducible validation command (use a new nonexistent directory):

```text
python3 -B Tests/RepositoryAgentTerminalValidation.py \
  DerivedData/ExpertAdvisor/RepositoryAgentTerminal/Validation-NEW
```

Executed final driver command used `Validation-20261008-02`. It ran:

```text
python3 -B Tests/RepositoryAgentTerminalTests.py --native                 PASS (33 tests)
python3 -B -m Tools.RepositoryAgent.tests.test_repository_agent_mcp       PASS
python3 -B -m Tools.RepositoryAgent.tests.test_repository_agent_mcp_profiles PASS
python3 -B -m Tools.RepositoryAgent.tests.test_repository_agent_mcp_claims PASS
python3 -B -m Tools.RepositoryAgent.tests.test_codex_interface             PASS
python3 -B -m Tools.RepositoryAgent.tests.test_source_reader               PASS
python3 -B -m Scripts.RepositoryAgentTerminal.mcp_adapter \
  --audit-directory <fresh-private-output>/mcp-audit \
  --approve-scheduler-phase-priority-test                                PASS (stdio validation)
```

The driver supplies per-process `PYTHONPATH` for the existing shared implementation plus Rollover, a Rollover source-root override for the smoke adapter, offline mode and a private HOME/TMPDIR. Original compatibility suites supply their own reader/verifier mocks; their environment omits the source-root override. This does not edit any Codex or MCP configuration. The driver never requests model-backed analysis or loads weights.

Build performed by the approved test: direct Xcode clang++, explicit `-isysroot .../MacOSX.sdk`, `-std=c++20 -Wall -Wextra -Werror`, copied include/source paths and private output, then that private binary. Exact absolute argv appear in launch receipts. Compilation and execution both exit 0 without diagnostics. No application C++ implementation changed; no full Xcode build or Clean ran.

## Files changed and remaining limitations

All additions are confined to Rollover:

- `Scripts/RepositoryAgentTerminal/__init__.py`: package boundary.
- `Scripts/RepositoryAgentTerminal/terminal_policy.py`: fixed command policy, validated roots/targets, approved test manifest, sandbox runner, time/output bounds and durable audit.
- `Scripts/RepositoryAgentTerminal/mcp_adapter.py`: composition with existing MCP interface, separate terminal tools and existing-runtime request/result bridge.
- `Tests/RepositoryAgentTerminalTests.py`: 33 policy, negative, native-sandbox, audit, model-mock and compatibility regressions.
- `Tests/RepositoryAgentTerminalValidation.py`: bounded, fresh-evidence validation driver and stdio acceptance check.
- This report.

No shared RepositoryAgent source, configuration, production resource, model, semantic-worker registry, existing Phase 24X file or tracked application file changed. There was no deployment, publication, merge, push or commit.

Remaining limitations:

1. The configured live integration does not expose these tools yet; real Qwen proposal/result acceptance is unverified. Do not present the mock exchange as a successful live Qwen invocation.
2. Activation must remain development-only and reuse one existing interface/runtime. Do not load weights in the standalone validation server alongside a live model instance. Shared production-facing source or configuration must not be changed implicitly.
3. The runner is macOS/Xcode-path-specific and depends on private sandbox/runtime behavior. A missing sandbox/toolchain, changed test hash or unsupported platform fails closed; no network permission or unsandboxed retry is inferred.
4. MLX proposal/acknowledgement uses bounded tokens but retains the existing synchronous generation behavior; terminal commands have independent wall deadlines. Real-model latency and contention were not measured.
5. Only one approved pure test is provided. Adding tests requires reviewed source/dependency hashes and isolated recipes; database/worker qualification and arbitrary shell scripts remain outside the catalog.
6. No automated resistance to a malicious privileged local user replacing system/toolchain binaries or tampering with the controller is claimed. Model/request authorization, target validation and child sandboxing are enforced independently of prompts.
7. Git metadata is physically shared with production; the documented read-only exception is necessary for this worktree. No production source/data/registry read is admitted as a terminal path.

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
```

`git diff --stat`: empty because all additions remain untracked. `git diff --check`: PASS. Shared RepositoryAgent status and selected hashes match the audit baseline, and the Codex configuration hash is unchanged.
