# Pocket 2025 confirmation freeze

Status: **frozen before confirmation access**.

Authority is the frozen Phase Pocket 3 prospective empirical evaluation
protocol.  This document is the separately committed confirmation freeze it
requires before opening the 2025 partition.  Pocket remains the Phase Pocket 2
project operational detector, not an exact mathematical transcription of the
MTI manual.

## Immutable identity and Git boundary

- Confirmation study: `pocket-prospective-confirmation-2025-v1`
- Implementation baseline: `eac49434ba105d78f1523820205764f5d16b4641`
  (`Complete Pocket derived uncertainty reporting`)
- Canonical configuration schema:
  `phase-pocket-4-canonical-confirmation-run-configuration-v1`
- Mechanically derived configuration SHA-256:
  `efb408fe8498f67320b302e7abc441d0f4030381321eaa1acf90f835bdbfb32b`
- Primary confirmation artifact schema:
  `phase-pocket-4-prospective-confirmation-artifact-v1`
- Derived confirmation reporting schema:
  `phase-pocket-4-derived-confirmation-report-v1`

The implementation baseline is intentionally the commit from which this
configuration and schema identity were derived.  The Git commit containing
this document is supplied by Git after the document is committed; it is not
invented or embedded self-referentially in the scientific configuration.

## Frozen configuration

The checked-in canonical configuration is
`Scripts/pocket_prospective_confirmation_2025_v1.conf`.  Its byte-stable
rendering is generated/validated by `PocketConfirmationFreeze.hpp`, not chosen
from a handwritten hash. Its exact payload is:

```text
aggregation=event_weighted_and_equal_symbol_separate
bootstrap_block=utc_calendar_week_confirmation_blocks
bootstrap_confidence=0.95
bootstrap_replicates=2000
cadence_seconds=900
censoring=right_censor_first_unusable_bar_boundary_tail_gap_invalid
configuration_schema=phase-pocket-4-canonical-confirmation-run-configuration-v1
derived_report_schema=phase-pocket-4-derived-confirmation-report-v1
detector=causal-pocket-detector-phase2-v1
detector_baseline=41800f4
horizons=4,16,64
lookbacks=10,15,20
ordering=symbol,lookback,confirmation_timestamp,observation_identity
outcomes=inclusive_touch_and_close;bounded_continuation_and_race
partitions=confirmation:1735689600:1767225600
primary_artifact_schema=phase-pocket-4-prospective-confirmation-artifact-v1
protocol=phase-pocket-3-prospective-empirical-evaluation-protocol-v1
protocol_document_sha256=e2852555026098e7c9fcd378874fbc075732f23db5ebb9183bf04a6590f067e1
resolution_end=1767283200
source_adapter=postgresql-candlestick-canonical-absolute-half-open-v1
source_identity=canonical_15m_completed_candlestick
study=pocket-prospective-confirmation-2025-v1
symbols=AUDCAD:audcadrmp:0.0001,AUDUSD:audusdrmp:0.0001,EURUSD:eurusdrmp:0.0001,GBPUSD:gbpusdrmp:0.0001,USDCAD:usdcadrmp:0.0001,USDJPY:usdjpyrmp:0.01
timeframe=15m_completed
warmup_bars=21
configuration_sha256=efb408fe8498f67320b302e7abc441d0f4030381321eaa1acf90f835bdbfb32b
```

It contains exactly one scoring partition:

```text
confirmation:1735689600:1767225600
```

- Scoring start inclusive: `2025-01-01T00:00:00Z`
- Scoring end exclusive: `2026-01-01T00:00:00Z`
- Authorized resolution end exclusive: `2026-01-01T16:00:00Z`
- Historical warmup: the frozen 21 contiguous completed bars before the first
  possible scoring confirmation only.

The canonical file retains the six frozen symbols and tables (AUDCAD/
`audcadrmp`, AUDUSD/`audusdrmp`, EURUSD/`eurusdrmp`, GBPUSD/`gbpusdrmp`,
USDCAD/`usdcadrmp`, USDJPY/`usdjpyrmp`), pip sizes, completed 15-minute/
900-second cadence, `L=10,15,20`, `H=4,16,64`, source identity, protocol,
and fixed 2,000 UTC-week hierarchical bootstrap policy.

Expected eventual primary artifact files are `configuration.conf`,
`manifest.txt`, `observations.csv`, and `aggregates.csv`.  Expected separate
derived-report files are `manifest.txt`, `outcomes.csv`, `equal_symbol.csv`,
`uncertainty.csv`, and `structural.csv`.  The confirmation-specific schemas
above prevent either output from being semantically confused with the
preconfirmation artifact or its derived report.

## Results-free advancement decision

Advance the existing frozen Phase Pocket 2/3 study unchanged into its
predeclared 2025 confirmation partition.  All three predeclared lookbacks
remain separately reported.  No lookback is selected.

This preserves unchanged: detector predicate and one-completed-bar
confirmation; touch, close-fill, MFE, MAE, signed directional-close-return,
and H64 race semantics; censoring, 900-second continuity, and no
interpolation; event-weighted and equal-symbol reporting; greedy 64-bar
thinning; pip and Pocket-width units; deterministic 2,000-replicate UTC-week
bootstrap; and its configuration-SHA-derived seed.

No p-value, composite score, winner, profitability interpretation, trading
rule, filter, detector/outcome/horizon change, or LSTM feature-selection
decision is authorized.  Confirmation is replication of the same frozen
descriptive phenomenon in the untouched partition.

## Offline validation before execution

```text
PocketResearch_Release --validate-pocket-confirmation-configuration \
  --config Scripts/pocket_prospective_confirmation_2025_v1.conf \
  --output-dir /previously-absent/confirmation-output
```

This command validates only the canonical configuration and an absent local
output target.  It opens neither PostgreSQL nor market data and cannot emit an
observation or outcome.  `--print-pocket-confirmation-freeze` prints the
same canonical configuration and identity without an output path.

No 2025 Pocket event, outcome, bar, source table, or result artifact was
accessed to create this freeze.  Preconfirmation and derived historical
artifacts remain unchanged.
