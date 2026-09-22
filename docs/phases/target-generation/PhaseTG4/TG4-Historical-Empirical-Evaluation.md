# TG4 historical empirical evaluation design and semantics

## Scope and baseline

TG4 is read-only diagnostic and research infrastructure over the completed
TG1A, TG1B, TG2, and TG3 contracts. The audited clean baseline is
`02e414a1b9274318d95b73aa85597d6cde1a8b6d` on
`lstm-feature-development`.

TG4 does not change any prior TG definition, model feature, semantic layout,
model width, training behavior, scheduler behavior, worker registry, trading
rule, profitability rule, or production database row. It contains no
parameter search and no 90% target or prior.

## Architecture and read-only boundary

The standalone path is:

```text
TG4 CLI -> historical evaluation service -> TG4 market-data repository
        -> read-only repeatable-read PostgreSQL transaction
```

`Sources/TG4HistoricalMarketDataRepository.cpp` calls the existing canonical
PostgreSQL `candlestick(text, integer, text, timestamp, timestamp)` function.
It parses the returned naive New York civil timestamp through the repository's
authoritative `HistoricalFxTimestamp::ParseNewYorkCivilTimestamp` conversion.
The query is ordered by `dt`, streamed row by row, and never fills a missing
bar. A duplicate canonical candle timestamp excludes the symbol before causal
evaluation.

The canonical study universe is the six symbols in
`SupportedSymbols::TrainingSymbols()`:

- `audcadrmp`
- `audusdrmp`
- `eurusdrmp`
- `gbpusdrmp`
- `usdcadrmp`
- `usdjpyrmp`

The database contains additional raw FX tables, but TG4 does not silently
promote them into the source-defined training universe.

## Temporal contract

Named study `tg4-2010-2025-v1` uses UTC causal event timestamps and half-open
intervals:

- exploratory: `[2010-01-01, 2020-01-01)`;
- calibration/parameter-development: `[2020-01-01, 2023-01-01)`;
- locked-rule validation: `[2023-01-01, 2025-01-01)`;
- untouched confirmation: `[2025-01-01, 2026-01-01)`.

TG4A adds the first-study identity
`tg4-preconfirmation-2010-2025-v1`. It uses the same 2010 warmup and partition
definitions but makes both `score_end` and `outcome_end` exactly
`2025-01-01T00:00:00Z`. It cannot load a confirmation-period bar; unresolved
finite windows at the boundary are censored rather than resolved from 2025.

The named study loads canonical bars from 2010-01-01. A partition-only run
still uses 2010-01-01 as its warmup start so the ATR recursion and every
bounded TG causal state see the same prefix as the full named study. No
pre-partition break is scored. A record belongs to exactly one partition based
only on its Inner-break timestamp.

Outcome-only bars after the score end may resolve finite TG2 windows. The
named study reads them through 2026-02-15. They cannot create scored records or
rewrite pairing, frozen angles/classes, AB selection, or confluence. End-of-
input before resolution is censoring, not failure.

TG4 performs no calibration. Consequently, confirmation data has no code path
that selects ratios, angle scale/bands, tolerances, horizons, or policies.

## Causal observation contract

One machine record is created for every TG2 break whose frozen TG1B class is
`Inner`. At that completed break bar TG4 freezes:

- the TG2 event and deterministic Inner identity;
- Inner class and creation-time calibrated angle;
- paired/unpaired status under TG2's authoritative pairing policy;
- the paired Outer identity, frozen class, and creation-time angle when one
  exists;
- TG3 AB identity and confirmation/availability facts;
- configured ratio provenance, level zones, nearest-level diagnostics, and
  confluence/ineligibility status;
- direction, symbol, timeframe, and temporal partition.

Later bars may update only TG2's retest, Outer-target, and retest-then-Outer
outcomes. They cannot change any frozen causal cohort field. Same-bar retest
and Outer contact keep TG2's existing rule: the conditioned target outcome
starts on a later completed bar because intrabar order is unknown.

An unpaired Inner break is structurally ineligible for both Outer outcomes.
TG3 structural ineligibility remains distinct from non-confluence. Pending,
censored, structurally ineligible, and unpaired observations are never folded
into failures.

## Configuration and provenance

TG4 requires an explicit `tg4-analysis-configuration-v1` file. The loader
rejects missing and unknown keys. It records all effective TG1A/TG1B/TG2/TG3
geometry, policy, tolerance, horizon, state-bound, candle, interval, gap, and
reporting settings in `metadata.json`.

TG1A and TG2 values in the example file mirror the audited implementation
defaults. The repository does not define a canonical TG1B
`referenceBarScale`, Fibonacci ratio set, or Fibonacci price tolerance. The
checked-in example deliberately contains `REQUIRED_EXPERIMENTAL_VALUE` for
those fields and is not runnable. TG4A prospectively freezes the first-study
values in `Scripts/tg4_analysis_config.frozen_v1.conf`; their evidence and
experimental provenance are recorded in
`docs/phases/target-generation/PhaseTG4/TG4A-Pre-Study-Methodology-Freeze.md`.

TG4 expresses Fibonacci proximity in canonical FX pips and deterministically
materializes the effective absolute tolerance per symbol (0.0001 per pip for
the five non-JPY canonical pairs and 0.01 per pip for USDJPY). Metadata records
the convention, pip count, all six effective values, and an effective
configuration fingerprint. TG3 itself retains its inclusive absolute-price
comparison.

TG3's source-direction default remains `source_utl_up_ab_only`. Symmetric
DTL/down-AB measurement is accepted only when explicitly named as the existing
diagnostic hypothesis.

## Statistical contract

Every binary rate is:

```text
successes / (successes + failures)
```

The denominator is emitted explicitly. A zero denominator emits an empty
rate and empty interval. Nonzero binary rates use a two-sided 95% Wilson score
interval with `z = 1.959963984540054`.

`comparisons.csv` emits underlying 2x2 counts for:

- retest versus resolved no-retest;
- Fibonacci-confluent versus eligible non-confluent;
- retest-and-confluent versus other resolved, eligible combinations.

It also emits the descriptive absolute rate difference when both denominators
are nonzero. No p-value or independence claim is made. Events can overlap,
share lines, and belong to the same market episode.

`cohorts.csv` retains symbol, partition, direction, paired status, retest
status, confluence status, their combinations, and frozen class views.
Event-weighted pooled rows use `__event_weighted__`.
`equal_symbol_rates.csv` separately averages per-symbol rates only across
symbols with a nonzero denominator and reports the contributing symbol count,
range, and sample standard deviation. Raw observation rows remain the
authority for regrouping.

The human report suppresses rate-table cells below the configured minimum
resolved N. Machine files retain the counts regardless.

## Data-quality contract

The repository preflights each symbol in the same repeatable-read snapshot as
the stream. `data_quality.csv` contains first/last usable UTC timestamps, row
counts, duplicate/out-of-order counts, warmup/scored/outcome-only counts,
partition coverage, gap counts, and exclusions. `data_gaps.csv` contains every
gap exceeding `expected_interval_seconds * material_gap_multiple` in stable
timestamp order.

Gaps include ordinary market closures as well as possible missing-data spans;
TG4 reports but does not guess their cause. It performs no interpolation.

## Boundedness and deterministic output

TG1A/TG2/TG3 keep their existing deterministic bounds. TG4 retains only an
event-ordered deque of scored records whose finite outcomes are not terminal.
Terminal prefixes stream immediately to `observations.csv`. Exceeding the
explicit TG4 pending bound stops the study instead of discarding evidence.

Stable symbol order, event order, map ordering, canonical numeric formatting,
schema versions, and absence of wall-clock values make completed artifacts
byte-deterministic for the same database snapshot and configuration. Runtime
milliseconds are printed to standard error, not placed in deterministic
evidence files.

## Interpretation limit

TG4 measures historical behavior. An Outer contact success rate is not a
trade return, profitability claim, independent trial probability, or model
feature justification. Results may motivate a separately designed locked
validation or later feature ablation; TG4 never performs model integration.
