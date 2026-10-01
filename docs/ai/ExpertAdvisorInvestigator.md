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
```

Each command also accepts:

```text
--repo-root <path>
--format text|json
```

`text` is the default.

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

Reports repository status only:

- Git HEAD;
- current branch or detached state;
- clean/dirty worktree state; and
- whether `docs/ai/ExpertAdvisorReference.md` exactly matches current generated
  content.

`status` deliberately does not claim scheduler, process, database, experiment,
or runtime-worker state.

## Safety boundary

The investigator performs no PostgreSQL connection, process signaling, product
binary invocation, network access, experiment queue/requeue, artifact
publication, semantic-layout mutation, or production cutover.

Its output is evidence for investigation, not authorization for an operational
action. Repository architecture, accepted ADRs, physical schema, semantic
contracts, and normal operator controls remain authoritative.

## Machine-readable use

JSON output is intended for GPT, Qwen, Codex, shell tooling, and other
investigators that need deterministic repository context before inspecting
deeper authoritative sources.

Example:

```bash
python3 Scripts/ExpertAdvisorInvestigator.py semantics --format json
```

The investigator should be extended only when a new query can be answered
deterministically from checked-in authoritative sources. Live operational state
belongs in a separately authorized interface rather than this tool.
