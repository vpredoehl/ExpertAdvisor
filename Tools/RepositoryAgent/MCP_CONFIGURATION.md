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

`discover_operation_relationship_paths` is a separate metadata-only traversal
for the deliberately narrow `operation_binding` category (assignment of a
single-call lambda to an accepted operation field, or the declared positional
slots of the production `CheckpointAnalysisOperations` aggregate) and one-hop
`operation_invocation` discovery. It has the same scope and admission caps as
direct path discovery, but never reports either relationship as a direct call
and never establishes runtime execution.
`investigate_operation_relationship_claim` independently rereads exact
server-selected ranges for either an `operation_binding` or an
`operation_invocation`. The latter names the invoked operation field (for
example `operations_.runCheckpointAnalysis`) rather than guessing which bound
function will run. A complete runtime bridge must therefore retain the
binding and invocation as separate claims.

For an admitted positional aggregate binding, investigation uses a distinct
two-range relationship-aware proof: the aggregate declaration (field order)
and the complete initializer (initializer position). Its ledger identity also
contains the operation field and aggregate schema, so a decision for one slot
cannot satisfy another. RepositoryAgent code is loaded from this worktree;
`expertadvisor_agent` remains the configured read-only production-source root.
Tests that need source-sensitive line assertions therefore supply an explicit
reader fixture rather than assuming worktree line numbers are live MCP lines.

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
