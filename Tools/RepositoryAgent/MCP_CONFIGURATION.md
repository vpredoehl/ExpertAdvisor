# RepositoryAgent MCP tool profiles

`Tools.RepositoryAgent.repository_agent_mcp` selects its MCP tool profile once
at process startup. A client cannot change the profile through `initialize`,
`tools/list`, or `tools/call` requests.

## Profiles

`full` is the default and retains the complete pre-profile tool surface,
including repository discovery, exact source reads, structural-index tools,
ledger tools, low-level claim-verification primitives, and all investigation
workflows.

`codex_assisted` advertises and admits exactly these planning-facing bounded
operations:

- `list_catalog_children`
- `discover_catalog_targets`
- `discover_relationship_paths`
- `discover_operation_relationship_paths`
- `investigate_source_claim`
- `investigate_source_bundle_claim`
- `investigate_relationship_claim`
- `investigate_operation_relationship_claim`
- `investigate_relationship_chain_claim`
- `investigate_relationship_set_claim`
- `investigate_symbol`
- `investigate_subsystem`

The profile intentionally excludes raw discovery/content operations such as
`list_files`, `search`, `read`, `resolve_symbol`, `function_for_line`,
`relationship`, `trace_calls`, and `source_excerpt`. It also excludes the
lower-level `verify_source_claim` and `verify_source_bundle_claim` primitives:
the admitted `investigate_*` workflows are the Codex planning interface and
retain their deterministic manifests, server-owned evidence selection where
applicable, exact rereads, evidence identities, and ledger behavior. The
lower-level primitives remain available in `full` for development/debugging.

Relationship-chain and relationship-set investigations use a relationship-aware
semantic-verification request after structural selection. The server associates
each admitted `caller -> callee` identity with only the normalized exact source
range(s) it selected and reread for that direct call. The verifier must confirm
each direct source-visible call independently; it does not require an inferred
data-flow/control-flow handoff between separate chain edges. These association
records are server-owned, appear in the bounded investigation manifest, and are
included in a distinct claim-ledger identity, so generic bundle decisions cannot
be reused for relationship-aware verification.

`discover_catalog_targets` is a deterministic, index-metadata-only catalog
selector. It requires one normalized directory scope (or an unambiguous
directory basename) and 1–4 canonicalized identifier-term groups (1–4 terms
per group). Every term in one group must match one filename stem or one
function identity; groups are alternatives. It returns only canonical file
paths and function identities with their container files, marked
`non_evidentiary`; it never returns source text, line contents, semantic
conclusions, source-search matches, or relationships. It does not invoke Qwen.
It accepts any resolved direct directory within catalog safety limits, then
rejects without truncating more than 48 query-matched metadata items, more
than 16 candidate files, more than 16 candidate symbols, or more than 24 total
candidates. A later `investigate_*` call remains required for behavioral claims.

`discover_relationship_paths` is a deterministic, metadata-only shortest-path
selector. It accepts one normalized direct-directory scope, exact or uniquely
resolvable function endpoints, and a required 1–4 hop bound. It returns only
canonical indexed function identities, never source text, source locations,
semantic conclusions, or evidence. Traversal is fail-closed before returning a
result if it would exceed 64 visited nodes, 512 examined call edges, or four
returned shortest paths; it never truncates. Nodes and edges must remain in the
resolved direct scope, descendant directories are excluded, and ambiguous
unqualified call-site names do not create edges. A returned path is not
evidence: use `investigate_relationship_chain_claim` or another bounded
evidentiary investigation before making behavioral claims.

`discover_operation_relationship_paths` retains the existing narrow
`operation_binding` traversal (single-call lambda assignments and declared
positional `CheckpointAnalysisOperations` slots) and one-hop
`operation_invocation` discovery. It also accepts
`relationship_kind="operation_implementation_call"` with an exact `operation`
selector to discover an individual call inside an explicitly assigned callback.
Its `from` endpoint is the callback's lexical owner and `to` is the indexed
callee. This targeted path does not traverse downstream direct calls.

An `operation_invocation` request can additionally supply `binding_owner` and
`operation`, with a concrete callee as `to`. This returns a three-edge contextual
path only when the invocation and explicit callback assignment have the same
uniquely resolved, declared operation type and field. The path retains the
invocation, callback assignment, and implementation call as separate typed
relationships. `max_hops < 3` returns no contextual path. The callback owner
must be explicitly selected; a field-name match alone cannot create a bridge.
Ambiguous types, receivers or repeated assignments fail closed. Only explicit
local declarations and explicit class members are supported; aliases, inferred
types and dynamic callback wiring are not inferred. Static slot compatibility
does **not** establish which callback object is installed at runtime, execution
timing, branch execution, or an unconditional call.

The returned `investigation_targets` provide exact admitted selectors for the
field invocation and implementation claim. The implementation investigation
verifies both the complete callback assignment and the represented call; the
contextual callback identity is not a new function endpoint for a direct-call
or legacy single-call binding investigation.

For example, within `scope="Sources/SchedulerCore"`:

```json
{
  "from": "FinalExperimentDispatchService::runPhase",
  "to": "BuildTrainCommand",
  "relationship_kind": "operation_invocation",
  "binding_owner": "RunFinalExperimentPhase",
  "operation": "operations.prepareReservedLaunch",
  "max_hops": 3
}
```

All discovery results remain metadata-only and `non_evidentiary`. The existing
direct-scope, four-hop, 64-node, 512-examined-edge and four-result limits remain
unchanged. The targeted routes admit at most one contextual path, do not follow
arbitrary callbacks, and never enter direct-call discovery.

`investigate_operation_relationship_claim` accepts the new implementation kind
with `caller`, `callee`, and exact `operation`. It selects and rereads the whole
explicit callback assignment, preserving branch conditions, independently of
the lexical owner's 16-relationship symbol-investigation cap. Repeated or
unavailable assignments and callbacks exceeding the existing 500-line range
limit return an explicit not-run result. The reread must match the indexed
source hash before verification or replay. The semantic request and existing
claim-ledger identity include the relationship kind, owner, field, target and
claim; exact source and verifier identity checks remain in force. Replay still
rereads evidence and requires no model turn on an exact hit.

Existing binding and invocation investigation behavior is unchanged. An
invocation names its operation field (such as
`operations_.runCheckpointAnalysis`), rather than guessing a bound function.
Behavioral claims still require separate bounded verification of each
relationship; contextual discovery is not proof of a complete runtime bridge.

For an admitted positional aggregate binding, investigation uses a distinct
two-range relationship-aware proof: the aggregate declaration (field order)
and the complete initializer (initializer position). Its ledger identity also
contains the operation field and aggregate schema, so a decision for one slot
cannot satisfy another. RepositoryAgent code is loaded from this worktree;
`expertadvisor_agent` remains the configured read-only production-source root.
Tests that need source-sensitive line assertions therefore supply an explicit
reader fixture rather than assuming worktree line numbers are live MCP lines.

## Isolated development repository

`EXPERTADVISOR_REPOSITORY_ROOT` optionally selects an absolute existing directory
at process startup. With no override, the adapter delegates to the existing
`expertadvisor_agent` reader and preserves the production root and permissions.
The shared `/Users/vjp/LLM/expertadvisor_agent.py` is not modified. An empty,
relative, missing, or non-directory override fails startup instead of falling
back to production.

Every RepositoryAgent reader and structural-index consumer imports the same
startup-selected adapter. Requests cannot switch its root or expand its policy.
An overridden non-production root permits C/C++/Metal files under `Headers`,
`Sources`, and `LSTM`, plus exactly these Phase24B evidence files:

- `Tests/DedicatedTrainingWorkerArchitectureTests.sh`
- `Tests/ReleaseWorkerBuildConfigurationTests.sh`
- `Scripts/tests/test_dedicated_train_rollover.py`
- `ExpertAdvisor.xcodeproj/project.pbxproj`

Reads, listings, and searches reject traversal and absolute paths and exclude
symlinks escaping the repository or permitted directories. The additional files
are readable evidence; the structural index still parses only C/C++/Metal.
No shell, write, build, test, or database capability is added. Both existing MCP
profile definitions remain unchanged.

Add a **separate** full-profile connection, retaining the existing two entries:

```toml
[mcp_servers.expertadvisor-repository-rollover]
command = "/Users/vjp/LLM/mlx-env/bin/python"
args = ["-m", "Tools.RepositoryAgent.repository_agent_mcp"]
cwd = "/Volumes/Developer SSD/ExpertAdvisor-RepositoryAgent"

[mcp_servers.expertadvisor-repository-rollover.env]
PYTHONPATH = "/Volumes/Developer SSD/ExpertAdvisor-RepositoryAgent"
PYTHONDONTWRITEBYTECODE = "1"
HF_HUB_OFFLINE = "1"
EXPERTADVISOR_REPOSITORY_ROOT = "/Volumes/Developer SSD/ExpertAdvisor-Rollover"
EXPERTADVISOR_LEDGER_CACHE_NAMESPACE = "ExpertAdvisor-Rollover"
EXPERTADVISOR_CLAIM_EVIDENCE_LEDGER = "/Users/vjp/Library/Caches/ExpertAdvisor-Rollover/RepositoryAgent/verified_claims.json"
```

Use an absolute ledger path: the existing ledger override treats a literal `~`
literally. The distinct path and namespace isolate both persistence and cache
identity. Deterministic operations never create/load a claim verifier or import
`mlx_lm`; model verification is a separate operation and is not needed for
connection validation. No production-source path belongs in this entry's
`PYTHONPATH`. `HF_HUB_OFFLINE=1` prevents model downloads if verification is
requested later; validation does not request verification or load weights.

## Development reader limits and regression qualification

When `EXPERTADVISOR_REPOSITORY_ROOT` selects an isolated development
repository, source discovery and reading use bounded, read-only operations.

| Resource | Limit |
| --- | ---: |
| Directory entries examined | 20,000 |
| Directory nesting depth | 20 |
| Search files | 2,000 |
| Search physical lines | 200,000 |
| Search input bytes | 32 MiB |
| Individual physical line | 64 KiB |
| Read output bytes | 256 KiB |
| Read output lines | 500 |
| Search results | 100 |
| Search pattern length | 256 characters |

Development-root search is **case-insensitive literal substring search**,
not regular-expression search. The production reader retains its original
behavior when no repository-root override is configured.

Directory traversal does not follow symlinked directories. Source-file
resolution rejects paths outside the selected repository and its permitted
source boundary.

Exceeding a resource budget raises an explicit error rather than silently
returning incomplete results.

The Rollover source tree contains 217,792 physical lines, exceeding the
200,000-line search budget. Consequently, an exhaustive search that does
not reach its requested result count may fail explicitly. This is a
documented limitation, not a successful exhaustive-search result.

The Phase 24B regression runner executes test modules in separate Python
processes because some legacy tests use module-level assertions rather
than `unittest.TestCase` classes.

Run from the isolated RepositoryAgent worktree:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:/Users/vjp/LLM" \
python3 Tools/RepositoryAgent/tests/run_isolated_regressions.py
```

The Phase 24D.3 inventory contains 32 passing modules (the previous runner's
29-module inventory assertion was stale). Three pre-existing
legacy modules remain excluded pending separate review:

- `test_ledger.py`
- `test_structural.py`
- `test_verifier.py`

These exclusions are not passing-test claims. The regression runner does
not load Qwen, invoke production training, build ExpertAdvisor, or access
PostgreSQL.

### Scheduler interrupted-launch recovery regressions

The existing RepositoryAgent regression runner supports two explicit
scheduler recovery modes without changing its default 32-module inventory.

Static recovery guard (no PostgreSQL):

    python3 Tools/RepositoryAgent/tests/run_isolated_regressions.py --scheduler-static

Static guard plus disposable-PostgreSQL integration:

    python3 Tools/RepositoryAgent/tests/run_isolated_regressions.py \
      --scheduler-integration \
      "$PWD/DerivedData/Phase24F/Build/Products/Release/LSTM_Release"

Integration requires an explicitly supplied isolated RepositoryAgent
Release executable. It creates a disposable PostgreSQL database, loads
the checked-in schema, and runs recovery-only with dummy workers.

Neither mode changes the default RepositoryAgent regression suite.
The integration test does not access the production LSTM database or
launch training workers.

The new focused regression module can be run independently:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:/Users/vjp/LLM" \
python3 -m unittest Tools.RepositoryAgent.tests.test_bounded_operation_implementation_discovery -v
```

Validate the five final-dispatch callback implementation relationships and the
three corresponding operation invocations against the actual read-only Rollover
source through the local `codex_assisted` MCP adapter:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:/Users/vjp/LLM" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
EXPERTADVISOR_REPOSITORY_ROOT="/Volumes/Developer SSD/ExpertAdvisor-Rollover" \
/Users/vjp/LLM/mlx-env/bin/python -m \
Tools.RepositoryAgent.tests.validate_rollover_operation_discovery --qwen
```

This uses the existing cached Qwen configuration, an in-memory index and a
disposable claim ledger under the temporary directory. It does not change MCP
registrations, source, PostgreSQL, workers or production ledgers. Omit `--qwen`
for deterministic-only validation. The already-running registered MCP process
must reload the updated code before its tools/list schema reflects these changes;
this implementation does not restart or reconfigure it.

## Startup selection

Use the explicit command-line option:

```sh
python3 -m Tools.RepositoryAgent.repository_agent_mcp --profile codex_assisted
```

Or set `EXPERTADVISOR_REPOSITORY_AGENT_MCP_PROFILE=codex_assisted` before the
process starts. If both are given, `--profile` takes precedence. The default is
`full`. Any other value fails process startup; it never falls back to `full`.

The Codex configuration lives outside this repository and is not modified by
this change. Add the following to the trusted project `.codex/config.toml` (or
the user `~/.codex/config.toml`):

```toml
[mcp_servers.repository_agent_codex_assisted]
command = "python3"
args = ["-m", "Tools.RepositoryAgent.repository_agent_mcp", "--profile", "codex_assisted"]
cwd = "/Volumes/Developer SSD/ExpertAdvisor-RepositoryAgent"
required = true
```

The existing full-profile development invocation remains:

```sh
python3 -m Tools.RepositoryAgent.repository_agent_mcp
```
