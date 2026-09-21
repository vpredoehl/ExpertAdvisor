# LSTM TG4 historical empirical evaluation output

## Decision

Implementation validation is in progress. No historical TG4 empirical funnel
has been run, and no market result or trading conclusion is claimed.

The clean audited baseline before any TG4 change was:

`02e414a1b9274318d95b73aa85597d6cde1a8b6d`

Branch: `lstm-feature-development`.

Existing stashes were inventoried and left untouched.

## Implemented empirical contract

TG4 now provides a standalone, causal, deterministic, read-only evaluation
path over the unchanged TG1A/TG1B/TG2/TG3 stack. It freezes one record for
each causal Inner break, follows finite retest/Outer/conditioned outcomes,
streams machine evidence, and aggregates the causal funnel without converting
pending, censored, structurally ineligible, or unpaired observations into
failures.

The output includes per-symbol and event-weighted partition/direction/cohort
counts, equal-symbol views, frozen angle moments, Wilson 95% intervals, and
underlying 2x2 counts for the specified comparisons. The raw observation file
retains all causal fields needed for later regrouping.

The CLI is independent of `LSTM_Release`; it cannot start the scheduler,
training, inference, analysis workers, or trading behavior.

## Parameter/configuration provenance

`Scripts/tg4_analysis_config.example.conf` records every effective setting and
copies only source-defined defaults where they exist. It deliberately refuses
to choose the three unsupported methodology inputs:

- TG1B `referenceBarScale`;
- TG3 Fibonacci ratio set;
- TG3 absolute price tolerance.

Those fields remain `REQUIRED_EXPERIMENTAL_VALUE`. A real study requires an
explicit named configuration whose provenance contains `experimental`. The
loader rejects missing/unknown fields and ratio provenance without that label.
No value is selected from 2025 data.

## Read-only data and coverage audit

The PostgreSQL audit used `REPEATABLE READ READ ONLY` and the canonical
`candlestick(..., 15, 'minute', ...)` path. The database has 28 raw FX tables;
the authoritative source list selects six for this study.

Catalog/data audit through 2026-02-15, before any empirical TG evaluation:

| Symbol | First source candle | Last source candle | Bars | Duplicate timestamps | Gaps > 22m30s | Maximum gap | Exploratory bars | Calibration bars | Validation bars | Confirmation bars |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| audcadrmp | 2010-01-03 17:00 NY civil | 2026-02-13 16:45 NY civil | 397,727 | 0 | 1,571 | 3 days 03:15 | 248,632 | 74,701 | 46,522 | 24,896 |
| audusdrmp | 2010-01-03 17:00 NY civil | 2026-02-13 16:45 NY civil | 397,775 | 0 | 1,549 | 3 days 03:15 | 248,622 | 74,735 | 46,543 | 24,899 |
| eurusdrmp | 2010-01-03 17:00 NY civil | 2026-02-13 16:45 NY civil | 397,870 | 0 | 1,561 | 3 days 03:15 | 248,646 | 74,750 | 46,602 | 24,896 |
| gbpusdrmp | 2010-01-03 17:00 NY civil | 2026-02-13 16:45 NY civil | 397,784 | 0 | 1,560 | 3 days 03:15 | 248,634 | 74,738 | 46,540 | 24,896 |
| usdcadrmp | 2010-01-03 17:00 NY civil | 2026-02-13 16:45 NY civil | 397,754 | 0 | 1,566 | 3 days 03:15 | 248,601 | 74,742 | 46,538 | 24,897 |
| usdjpyrmp | 2010-01-03 17:00 NY civil | 2026-02-13 16:45 NY civil | 395,764 | 0 | 1,574 | 32 days 00:15 | 248,657 | 74,742 | 46,506 | 24,899 |

These gap counts include routine market closures and are not automatically
missing-data diagnoses. The 32-day USDJPY span is material and must be reviewed
before interpreting a real study. TG4 does not fill any gap.

The executable repeats this audit in its own snapshot and emits UTC-converted
details to `data_quality.csv` and `data_gaps.csv`.

## Temporal and causal validation

The named study uses 2010-01-01 warmup, scores only by causal break timestamp,
and keeps the four required half-open partitions. Focused tests cover:

- TG4 historical/streaming prefix parity;
- pre-partition warmup without pre-partition scoring;
- exact partition boundaries;
- future outcome updates leaving the frozen cohort unchanged;
- deterministic record order and byte-identical artifact sets;
- separate success/failure/censor/pending/ineligibility accounting;
- zero-denominator rates;
- Wilson known cases;
- confluence/non-confluence, retest/no-retest, and paired/unpaired cohorts;
- duplicate rejection and material-gap reporting;
- bounded representative streaming.

The representative TG4 run processed 50,000 synthetic completed bars and
33,318 Inner-break observations in 77 ms in one observed run, with only four
pending TG4 records at peak. Elapsed time is an environment observation, not a
deterministic artifact or performance guarantee.

## Semantic/model and operational state

TG4 source files contain no `Tensor`, `FeatureLayout`, or model-width
dependency. Model input width and semantic-layout evidence will be recorded
after the required regression suite. No production worker was published or
restarted.

Two active scheduler-managed `LSTM_Release --train` workers were observed
during implementation (experiments 650 and 651). They were not interrupted.
The clean Release provenance gate must not be attempted while that makes the
build unsafe.

## First real study command

First create a separate configuration and replace every
`REQUIRED_EXPERIMENTAL_VALUE` from a decision made without viewing 2025
results:

```bash
cp Scripts/tg4_analysis_config.example.conf /absolute/path/tg4-frozen.conf
${EDITOR:?set EDITOR} /absolute/path/tg4-frozen.conf
```

Then run the six-symbol named study into a new directory:

```bash
Scripts/run_tg4_historical_empirical_evaluation.sh \
  --config /absolute/path/tg4-frozen.conf \
  --output-dir /absolute/path/tg4-study-2010-2025-v1 \
  --all-canonical-symbols \
  --study tg4-2010-2025-v1
```

`FOREX_DB_HOST` and `FOREX_DB_NAME` follow the existing application defaults;
`--connection` can supply an explicit libpq connection string. The output
directory must not already contain TG4 artifacts.

## Limitations and remaining risks

- No canonical ratio set, Fibonacci tolerance, or reference-bar scale exists;
  the empirical study is intentionally blocked until those are explicitly
  frozen and labeled experimental.
- No full market TG4 run has occurred, so there is no evidence yet about
  cross-symbol stability, censoring rates, or incremental Fibonacci
  information beyond TG2.
- Observations overlap and are not independent trials.
- Gap causes, particularly the USDJPY maximum gap, have not been classified.
- Source validation and clean Release provenance closure remain to be recorded
  below; this document must not be read as a Release GO until those gates pass.
