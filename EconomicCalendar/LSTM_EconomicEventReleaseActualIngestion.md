# Authoritative initial-actual ingestion and coverage audit

Phase 9 supplies a fail-closed operator workflow for migration 088. It does
not change the Phase 8 feature contract or apply migration 088 to production.

## Supported source contract

The initial implementation supports only the Census `RETAIL_SALES` and
`DURABLE_GOODS` advance-release headline month-over-month percentage. Each
candidate must preserve all migration 088 fields and prove:

- the existing canonical `source_event_id`, source agency, family, reference
  period, source URL, and release instant;
- an immutable tracked source artifact, its SHA-256, and the first Git commit
  that admitted the artifact to the repository archive;
- a direct `ADVANCE` marker and a source-text headline actual for the current
  period;
- `publication_state='initial'`, `revision_sequence=0`, and
  `available_at` exactly equal to the canonical event release instant;
- a later explicitly revised/unrevised prior-period observation before
  emitting `publication_state='revision'`, `revision_sequence=1`;
- scalar `percent`, qualifier `m/m`, scale `1`, and exact
  `canonical_value = raw_value * scale` semantics.

The first archive-admission committer timestamp is used as a conservative,
reproducible `retrieved_at`: it proves when the artifact entered the immutable
repository archive, is later than historical publication, and does not rely on
mutable filesystem timestamps. The provenance object records this basis and
the archive commit.

Unsupported document layouts, values, units, or identity evidence are
rejected. The importer does not use OANDA or Myfxbook actuals, infer an initial
from a revision, select among ambiguous events, update an existing row, or
perform network access.

## Deliberately unsupported families

- BLS CPI, Employment, PPI, and JOLTS: local BLS schedules prove event times,
  but the repository has no historical authoritative release-value archive.
- BEA GDP and PCE: the local release archive is broad, but estimate-specific
  GDP and statistic/qualifier-specific PCE initial/revision semantics require
  separate adjudication.
- Federal Reserve FOMC: statements contain target ranges, while Phase 8
  surprise requires a compatible scalar consensus/actual pair.
- DOL/ETA Weekly Claims: acquisition and parser fixtures exist, but no local
  historical source archive is present.

## Workflow

The normal invocation is read-only and writes only deterministic local audit
artifacts:

```sh
python3 EconomicCalendar/import_economic_event_release_actual.py \
  --db LSTM \
  --dry-run-output /tmp/phase9-release-actual.jsonl \
  --coverage-output /tmp/phase9-coverage.json
```

PostgreSQL catalog reads run with `default_transaction_read_only=on`. The JSONL
reports every candidate, match, value, provenance object, decision, and
rejection. The coverage JSON contains 2010-01-01 through 2025-01-01 summaries,
year segments, post-target data, actual-only coverage, and selected-consensus
joint coverage.

Writes require both `--commit` and `--allow-disposable-write`, and the database
name must begin with `ea_`. The importer executes one append-only transaction,
with no `UPDATE`, broad upsert, or conflict suppression. Identical persisted
observations are reported as `duplicate_identical`; any differing observation
or event/revision identity is a visible `conflict` and is not written.

## Validation

The focused suite uses local text fixtures and a uniquely named disposable
PostgreSQL database:

```sh
Tests/EconomicEventReleaseActualImporterTests.sh
```

It exercises initial and revision extraction, immutable identity handling,
duplicate/conflict behavior, deterministic matching and output, value
semantics, SHA-256 and causal timestamp preservation, append-only SQL,
idempotent repeated import, and coverage determinism without network or
production-database dependencies.
