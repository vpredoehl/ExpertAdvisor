# Phase Pocket 4 — frozen preconfirmation reporting-completeness audit

## Scope and evidence boundary

This audit compares the complete frozen Phase Pocket 3 protocol
(`PocketProspectiveEmpiricalEvaluationProtocolFreeze.md`) with the *schema and
writer implementation* of the sealed preconfirmation artifact.  It did not
open the historical artifact, run an evaluator, contact PostgreSQL, or access
`confirmation_2025`.  Pocket remains the Phase Pocket 2 project operational
detector, not a mathematical transcription of the MTI manual.

The sealed artifact is immutable evidence.  The corrective command added here
first uses the existing immutable artifact verifier, then reads only its
configuration, manifest, observations, and aggregates files.  It does not
invoke a detector, use a market-data adapter, or regenerate observations.

## Complete frozen reporting audit

| Frozen protocol reporting requirement | Classification | Basis |
| --- | --- | --- |
| Canonical configuration, identity, hashes, protocol/detector/source provenance and primary artifact hashes | IMPLEMENTED_IN_EXISTING_ARTIFACT | `configuration.conf` and `manifest.txt` already contain and verify them. |
| Eligible, complete, censored, first-touch, close-fill counts/rates by symbol/lookback/partition/direction/horizon | IMPLEMENTED_IN_EXISTING_ARTIFACT | Existing `aggregates.csv`. |
| Event-weighted and equal-symbol touch/close summaries and touch bootstrap interval | IMPLEMENTED_IN_EXISTING_ARTIFACT | Existing `aggregates.csv`; they remain distinct. |
| Observation identity, symbol, partition, lookback, direction, timestamps, range, horizon, censor reason, future bars, touch/fill time, MFE, MAE, signed return and race label | IMPLEMENTED_IN_EXISTING_ARTIFACT | Existing `observations.csv`. |
| Structural emitted/eligible and bullish/bearish counts/shares; price/pip width; confirmation spacing; simultaneous observations; 64-bar temporal/range overlap; repeated direction; UTC-week clustering; greedy 64-bar thinning | PRESENT_AT_OBSERVATION_LEVEL_BUT_NOT_REPORTED | Unique immutable identities include exact confirmation coordinates; rows contain timestamp, direction and range. |
| Resolved time-to-touch/fill medians and P25/P75, including the H64 “not reached within 64” rule | PRESENT_AT_OBSERVATION_LEVEL_BUT_NOT_REPORTED | Stored first offsets and complete/censor state make this mechanical. |
| MFE/MAE/signed return medians and P25/P75 in price, frozen pips and width multiples | PRESENT_AT_OBSERVATION_LEVEL_BUT_NOT_REPORTED | Stored price-space outcomes and lower/upper range make conversion mechanical. |
| H64 continuation/revisit/same-bar/neither/censored reporting | PRESENT_AT_OBSERVATION_LEVEL_BUT_NOT_REPORTED | Stored race labels plus complete/censor state make censored explicit. |
| Per-censor-reason and complete/unresolved outcome reporting; valid future bars | PRESENT_AT_OBSERVATION_LEVEL_BUT_NOT_REPORTED | Stored labels make it mechanical. |
| Equal-symbol versions of the stored-outcome summaries | PRESENT_AT_OBSERVATION_LEVEL_BUT_NOT_REPORTED | Symbol cohorts and the frozen pip sizes are in the canonical configuration. |
| Counts per structurally evaluable confirmation bar and per 10,000 such bars | NOT_DERIVABLE_FROM_SEALED_ARTIFACT | The denominator (all structurally evaluable confirmation bars) is not present. |
| Duplicate/out-of-order timestamps, malformed/non-finite OHLC, insufficient/discontinuous warmup, cadence-gap timestamps/durations, observations rejected/outside scoring | NOT_DERIVABLE_FROM_SEALED_ARTIFACT | These source/preflight audit populations and rejected candidates were not persisted. |
| Tail bars requested/used as a source-audit quantity | NOT_DERIVABLE_FROM_SEALED_ARTIFACT | Per-label `valid_future_bars` exists, but the source-level request/use audit is not persisted. |
| Frozen hierarchical bootstrap intervals for the newly reported outcome/continuation summaries | IMPLEMENTATION_DEFECT | The protocol freezes 2,000 UTC-week hierarchical resamples; the existing artifact reports only the touch interval.  The sealed rows contain weeks, but the historical artifact cannot be altered. |
| Absolute differences and “meaningful” ratios | NOT_DERIVABLE_FROM_SEALED_ARTIFACT | The protocol does not predeclare comparison pairs or a denominator for a ratio; choosing them after outcomes would be a new analysis plan. |
| Confirmation-2025 reporting | NOT_APPLICABLE_TO_PRECONFIRMATION | The frozen preconfirmation configuration ends before 2025-01-01T00:00:00Z. |

There is no detector, outcome-label, source-read, or sealed-artifact mutation
defect identified by this audit.  The incomplete aggregate reporting and its
limited bootstrap coverage are reporting implementation defects, not a change
to the frozen scientific contract.

## Corrective derived report

After operator review, run only against the already verified sealed artifact:

```text
DerivedData/PocketResearch/Build/Products/Release/PocketResearch_Release \
  --derive-pocket-prospective-report \
  --artifact-dir /absolute/path/to/sealed-preconfirmation-artifact \
  --output-dir /absolute/path/to/new-derived-report \
  --git-commit <40-hex-HEAD> \
  --executable-identity <PocketResearch_Release-SHA-256>
```

The target must not exist.  Publication writes a sibling staging directory,
verifies the derived manifest and hashes, and atomically renames it.  Files are
`outcomes.csv`, `equal_symbol.csv`, `structural.csv`, and `manifest.txt`.
The manifest records the source study/protocol/protocol-document hash/detector,
source configuration/observation/aggregate hashes, Git identity, executable
SHA-256, analyzer schema, and derived file hashes.  Verify it with:

```text
PocketResearch_Release --verify-pocket-prospective-derived-report /absolute/path/to/new-derived-report
```

This reports only deterministic frozen statistics represented by the sealed
rows.  It does not repair unavailable source-audit denominators or choose
undefined comparison ratios.  No market-data reevaluation is required, the
historical evidence remains byte-for-byte unchanged, and
`confirmation_2025` remains sealed.
