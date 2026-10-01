# ExpertAdvisor Investigator

`Scripts/ExpertAdvisorInvestigator.py` is a deterministic, read-only repository
investigator intended for operators and AI-assisted development tools.

It does not provide an AI model and does not make research decisions. Its job is
to expose a small set of current repository facts in stable text or JSON form
without creating another source of truth.

## Commands

```bash
python3 Scripts/ExpertAdvisorInvestigator.py authorities
python3 Scripts/ExpertAdvisorInvestigator.py semantics
python3 Scripts/ExpertAdvisorInvestigator.py workers
python3 Scripts/ExpertAdvisorInvestigator.py status
python3 Scripts/ExpertAdvisorInvestigator.py domains
python3 Scripts/ExpertAdvisorInvestigator.py evidence scheduler
```

Each command accepts `--repo-root <path>` and `--format text|json`. `text` is
the default. `evidence` additionally requires a domain.

### `authorities`

Reports the checked-in source paths that govern architecture, database schema,
model-input semantics, feature ablation, and semantic workers. It also reports
any missing authority paths.

### `semantics`

Reports the current semantic layout, model-input width, Tensor feature width,
return suffix width, and registered predecessor chain. These facts reuse the
Q1 reference generator's validated extraction rather than maintaining a second
set of semantic constants.

### `workers`

Reports the validated current train and inference semantic workers, including
layout, width, capabilities, source commit, SHA-256, and immutable artifact
path.

### `status`

Reports repository status only: Git HEAD, current branch or detached state,
clean/dirty worktree state, and whether `docs/ai/ExpertAdvisorReference.md`
exactly matches current generated content.

`status` deliberately does not claim scheduler, process, database, experiment,
or runtime-worker state.

## Q3 authoritative evidence routing

### `domains`

Lists the deterministic evidence domains known to the investigator and reports
missing routed authority paths.

### `evidence <domain>`

Returns four deliberately separate evidence layers:

1. **constitutional authority** — relevant numbered sections of Volume I;
2. **domain authority** — numbered sections of the owning architecture volume;
3. **accepted decisions** — ADR files routed for investigation of that domain;
4. **implementation authority** — checked-in schema/contracts/registries that
   demonstrate shipped behavior.

Supported domains are:

```text
data-pipeline
labels
model
training
inference
experiment-lifecycle
recommendations
profitability
research-automation
scheduler
database
```

The Markdown section spans are discovered from headings at runtime. The routing
map is navigation metadata; it does not copy normative prose, does not establish
exclusivity, and does not supersede architecture. A routed ADR is a relevant
starting point, not a claim that no other ADR can apply.

The architecture distinction remains controlling: architecture describes
intended contracts, while implementation demonstrates currently shipped
behavior. Neither silently overrides the other.

Example:

```bash
python3 Scripts/ExpertAdvisorInvestigator.py evidence scheduler --format json
```

## Safety boundary

The investigator performs no PostgreSQL connection, process signaling, product
binary invocation, network access, experiment queue/requeue, artifact
publication, semantic-layout mutation, or production cutover.

Its output is evidence for investigation, not authorization for an operational
action. Repository architecture, accepted ADRs, physical schema, semantic
contracts, and normal operator controls remain authoritative. Live operational
state belongs in a separately authorized interface.

## Machine-readable use

JSON output is intended for GPT, Qwen, Codex, shell tooling, and other
investigators that need deterministic repository context before inspecting
deeper authoritative sources.

The investigator should be extended only when a new query can be answered
deterministically from checked-in authoritative sources. Do not add free-text
heuristic routing that can silently choose the wrong authority.


## Deterministic investigation manifests

Q4 adds named investigation profiles that compose the authoritative domain routes
introduced by Q3.  Profiles contain domain names only.  They do not define file
paths, ADR associations, schema authority, or any second evidence map.

List the supported profiles:

```bash
python3 Scripts/ExpertAdvisorInvestigator.py investigations
```

Build an evidence manifest:

```bash
python3 Scripts/ExpertAdvisorInvestigator.py plan semantic-compatibility
python3 Scripts/ExpertAdvisorInvestigator.py plan semantic-compatibility --format json
```

A plan preserves profile order and embeds the exact Q3 `evidence` result for each
selected domain.  Therefore Q4 selects existing Q3 evidence routes; it does not
rediscover, reinterpret, rank, or supersede them.

The initial profiles are:

- `semantic-compatibility`
- `scheduler-dispatch`
- `experiment-identity`
- `training-configuration`
- `inference-evaluation`
- `profitability-evaluation`
- `recommendation-governance`
- `research-automation`

An investigation plan is navigation metadata, not a diagnosis or answer.  It
does not inspect PostgreSQL, processes, scheduler state, logs, experiments, or
other live runtime state, and it grants no operational authorization.  Live
evidence acquisition requires a separately authorized interface.
