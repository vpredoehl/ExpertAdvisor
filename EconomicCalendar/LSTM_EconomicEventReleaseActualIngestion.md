# Authoritative initial-actual ingestion and coverage audit

Phase 10 extends the fail-closed Phase 9 operator workflow for migration 088.
It does not change the Phase 8 width-75 feature contract, apply migration 088
to production, or write authoritative actuals to production.

Phase 11 adds a PCE-only production-readiness artifact workflow.  It remains
read-only in PostgreSQL and has no database commit option:

```sh
python3 EconomicCalendar/prepare_pce_production_import.py \
  --db LSTM \
  --manifest-output /tmp/pce-production-import.jsonl \
  --audit-output /tmp/pce-production-readiness.json \
  --sql-output /tmp/pce-production-import.sql
```

The JSONL contains exactly the migration-088 insert fields in canonical
availability/event/observation order.  The audit records every current catalog
identity and selected-consensus classification, the exact fail-closed source
exclusions, coverage, and the SHA-256 of the complete manifest bytes.  The SQL
is a single append-only transaction with no update, delete, upsert, or conflict
suppression.  It is review material only; applying migration 088 and importing
PCE remain separate future operator gates.

Phase 12 validates that reviewed package against the unchanged width-75 feature
contract.  Repository queries now exclude an initial actual unless its
persisted `available_at` is strictly before the requested information upper
bound, and the feature engine repeats the same predicate for every completed
bar.  The migration-088 feature view continues to expose only revision zero;
later revisions remain immutable audit evidence and never replace the initial
surprise.  This validation does not deploy migration 088 or import the package
into production.

## Supported source contract

The unchanged Census implementation supports `RETAIL_SALES` and
`DURABLE_GOODS` advance-release headline month-over-month percentage. Each
candidate preserves all migration 088 fields and proves:

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

Phase 10 adds two source-specific BEA contracts using the tracked 393-artifact
canonical HTML corpus:

- `GDP`: real GDP quarter-over-quarter growth at an annual rate, scalar
  `percent`, qualifier `NULL`. Only an `Advance` (or the shutdown-era
  equivalent `Initial`) estimate emits revision 0. A `Second`/`Updated`
  estimate may emit revision 1 and a `Third` estimate may emit revision 2,
  always against the same advance/initial event identity. Later estimates
  never become initial actuals for their separately cataloged events.
- `PCE`: current-dollar personal consumption expenditures month-over-month
  change, scalar `percent`, qualifier `m/m`. This is deliberately neither the
  PCE price index nor core PCE. A source phrase such as “less than 0.1 percent”
  is rejected because it does not prove an exact scalar. Combined multi-month
  releases are rejected because they do not prove one canonical monthly value.

BEA identity is the canonical source agency, exact release timestamp,
`source_event_id`, reference period/estimate type, and official source URL.
HTML evidence, SHA-256, first Git archive admission, and exact headline text
are preserved in provenance. PCE annual-revision material is not assigned a
synthetic sequence; Phase 10 emits only the directly published monthly initial.

Phase 17 adds release-specific BLS contracts backed by the retained official
archive pages in `EconomicCalendar/raw/bls`:

- `CPI`: CPI-U all items, seasonally adjusted month-over-month percent change.
- `EMPLOYMENT`: seasonally adjusted total nonfarm payroll employment change;
  explicit reissues are excluded from the initial-release boundary.
- `PPI`: seasonally adjusted finished-goods month-over-month percent change
  before the historical transition and final-demand month-over-month percent
  change afterward.
- `JOLTS`: seasonally adjusted total-nonfarm job-openings level.

Each BLS candidate requires exact production catalog identity, release-time and
reference-period evidence, retained bytes matching the manifest SHA-256, and a
family-specific headline or table proof. The general Phase 10 importer includes
this corpus only when `--bls-manifest` is supplied. The dedicated
`prepare_bls_production_import.py` workflow defaults to the retained manifest,
reads production with `default_transaction_read_only=on`, and emits review-only
append-only SQL; it does not import BLS actuals.

## Deliberately unsupported families

- Federal Reserve FOMC: statements contain target ranges, while Phase 8
  surprise requires a compatible scalar consensus/actual pair.
- DOL/ETA Weekly Claims: no local authoritative historical source archive is
  present (`raw/dol` is absent).

## Workflow

The normal invocation remains read-only and writes only deterministic local audit
artifacts:

```sh
python3 EconomicCalendar/import_economic_event_release_actual.py \
  --db LSTM \
  --dry-run-output /tmp/phase10-release-actual.jsonl \
  --coverage-output /tmp/phase10-coverage.json
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

The focused suite uses local Census and BEA text fixtures and a uniquely named disposable
PostgreSQL database:

```sh
Tests/EconomicEventReleaseActualImporterTests.sh
```

It exercises Census regression, BEA statistic and estimate-type adjudication,
initial and revision extraction, immutable identity handling,
duplicate/conflict behavior, deterministic matching and output, value
semantics, SHA-256 and causal timestamp preservation, append-only SQL,
idempotent repeated import, and coverage determinism without network or
production-database dependencies.
