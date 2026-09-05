# Phase 7 DOL/ETA Weekly Claims first-release actual readiness

This directory contains the deterministic Phase 7 package generated from the
retained official DOL/ETA Weekly Claims archive and a production catalog read
under `default_transaction_read_only=on`. No production actual was imported.

## Evidence and classification

- Production catalog: 860 `DOL_ETA/WEEKLY_CLAIMS` events from 2010-01-07
  through 2026-08-20; zero existing immutable actual observations and zero
  selected consensus observations.
- Official archive: 862 release-specific URLs from 2010-01-07 through
  2026-09-03; all 862 acquisitions succeeded and are retained locally.
- Artifact formats: 222 HTML releases and 640 PDFs. Each PDF also retains the
  separately hashed `pdftotext -layout -enc UTF-8` parser input.
- Eligible exact releases: 815, producing 815 initial observations and 689
  explicitly labelled prior-week revision observations (1,504 total).
- Fail-closed publication-time exclusions: 45 releases from 2011-11-10 through
  2012-09-13 whose embedded EST/EDT label contradicts historical
  `America/New_York` rules.
- Unmapped archive releases: 2 (2026-08-27 and 2026-09-03), both later than the
  current production catalog endpoint. There are no ambiguous mappings.
- Initial-value, parser/extraction, or acquisition failures: zero.

Every eligible initial maps by exact official source URL and then reconciles
the reporting week and exact embedded publication instant to the canonical
event. The current week's seasonally adjusted advance figure is revision 0.
Only a separately labelled prior-week revised value becomes revision 1 for the
prior event. Values are scalar `count`, scale `1`, qualifier `NULL`.

## Deterministic package

- `dol-eta-weekly-claims-actual-import.jsonl`: 1,504 rows, SHA-256
  `6c565ea11aaea440fac0fc6a3b38fc5a088f1ca896756ac0f3ba6c6469701388`.
- `dol-eta-weekly-claims-actual-import.sql`: append-only transaction, SHA-256
  `76ccee530ca0968c10634a96c737eb40a4c24d9e95f5155b21a6bd6a075e5c81`.
- `dol-eta-weekly-claims-classifications.jsonl`: all 862 archive candidates,
  SHA-256 `865c6525fe6c42cf8d3f0c37b9fb1fedf58cb5be9c1d9a99d4efcbd07bac300d`.
- `dol-eta-weekly-claims-coverage-audit.json`: exact counts and safety state,
  SHA-256 `ea4f9623f13d38e2ffffb0ced72d071611c3cd02ce1ab9d90255e39093f52237`.
- `EconomicCalendar/raw/dol_eta/actual_acquisition_manifest.jsonl`: acquisition
  identity and hashes, SHA-256
  `f3e0fbff080b515cee832a2ed6cbc64b2a87b1a75b4ce3992781da1fa000c95f`.

Two independent read-only regenerations and the retained package were
byte-identical.

## Production safety decision

At final preparation, experiments 619 and 620 were both actively training and
could still generate feature rows from the production economic-event database.
The application backup and production import were therefore deliberately not
performed. The prepared SQL is review evidence only.

Even after a future safe actual import, Weekly Claims causal surprise will
remain unavailable until a separate authoritative consensus phase addresses
the current zero-consensus coverage. Phase 7 does not import consensus.

Decision: `PASS_WITH_PRODUCTION_IMPORT_DEFERRED`
