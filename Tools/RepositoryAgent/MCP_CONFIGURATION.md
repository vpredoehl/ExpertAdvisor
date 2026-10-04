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
- `investigate_source_claim`
- `investigate_source_bundle_claim`
- `investigate_relationship_claim`
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
