# Phase Pocket 4 — confirmation execution enablement

This implementation enables only the confirmation execution path frozen in
`Pocket2025ConfirmationFreeze.md`.  It does not execute it.  The command
accepts no scientific date, universe, detector, lookback, horizon, or outcome
options: it must load the exact byte-canonical committed configuration.

```text
PocketResearch_Release --pocket-confirmation-evaluator --execute \
  --config Scripts/pocket_prospective_confirmation_2025_v1.conf \
  --output-dir /previously-absent/confirmation-primary \
  --git-commit <40-hex-commit> --executable-identity <sha256>
```

The shared read-only evaluator uses the confirmation configuration’s
`resolution_end=1767283200` as the sole source-query upper bound.  Source
preflight also rejects any bar at or after that bound.  The primary writer
records and verifies `phase-pocket-4-prospective-confirmation-artifact-v1`.

```text
PocketResearch_Release --verify-pocket-confirmation-artifact /confirmation-primary
PocketResearch_Release --derive-pocket-confirmation-report \
  --artifact-dir /confirmation-primary --output-dir /previously-absent/confirmation-derived \
  --git-commit <40-hex-commit> --executable-identity <sha256>
PocketResearch_Release --verify-pocket-confirmation-derived-report /confirmation-derived
```

The derived path verifies the schema-bearing primary artifact before parsing
observations, then reuses the existing deterministic reporting/aggregation/
bootstrap implementation under
`phase-pocket-4-derived-confirmation-report-v1`.  It does not access market
data or invoke the detector.  No confirmation result was accessed while this
enablement was implemented.
