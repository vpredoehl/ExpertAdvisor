# TG4A pre-confirmation findings and confirmation freeze

## Status

Pre-confirmation evidence reviewed. 2025 confirmation remains untouched.

This document was created after inspection of the completed
`tg4-preconfirmation-2010-2025-v1` study and before inspection of any TG4
outcome from the reserved `[2025-01-01, 2026-01-01)` confirmation partition.

It records the findings produced by the prospectively frozen TG4A methodology
and freezes the questions that will subsequently be evaluated on 2025.

This document does not modify the original TG4A methodology freeze.

## Prospective methodology provenance

Original methodology freeze:

`docs/phases/target-generation/PhaseTG4/TG4A-Pre-Study-Methodology-Freeze.md`

Audited TG4 implementation baseline:

`8cc823239ed34b068009c92c50a185216db01158`

Frozen configuration identity:

`tg4a-first-study-preconfirmation-frozen-v1`

Frozen configuration SHA-256:

`78187108b908b4298eb9ae2f1e9fa1de63eadb1aceb29d09ab5261c42a7a9444`

Configuration fingerprint:

`fnv1a64:bf809ce38a4a444a`

Pre-confirmation study identity:

`tg4-preconfirmation-2010-2025-v1`

The original prospective freeze established these half-open partitions:

- exploratory `[2010-01-01, 2020-01-01)`
- calibration `[2020-01-01, 2023-01-01)`
- locked-rule validation `[2023-01-01, 2025-01-01)`
- untouched confirmation `[2025-01-01, 2026-01-01)`

The completed pre-confirmation study ended both scoring and outcome resolution
exclusively at `2025-01-01`. No 2025 bar was available to resolve a
pre-2025 observation.

## Frozen experimental conventions

The following remain unchanged from the prospective TG4A freeze:

- `referenceBarScale=14`
- `tg3_retracement_ratios=0.6180339887498949`
- `tg3_price_tolerance_convention=canonical_fx_pips`
- inclusive tolerance of one canonical FX pip
- `0.0001` effective tolerance for AUDCAD, AUDUSD, EURUSD, GBPUSD, USDCAD
- `0.01` effective tolerance for USDJPY

The source-defined angle bands, implementation-defined TG1/TG2/TG3 semantics,
causal timing, pairing rules, outcome definitions, and comparison semantics
also remain unchanged.

## Completed pre-confirmation study

The study completed successfully across all six canonical symbols.

Total observations:

`1,292,602`

Per-symbol observations:

- AUDCAD: 174,887
- AUDUSD: 203,480
- EURUSD: 234,469
- GBPUSD: 246,768
- USDCAD: 216,121
- USDJPY: 216,877

The study reported:

- `read_only=true`
- `parameter_optimization=false`
- `scheduler_started=false`
- `workers_started=false`

No parameter search or automatic calibration was performed.

## Artifact provenance

Artifact directory:

`Artifacts/tg4-preconfirmation-2010-2025-v1`

SHA-256:

- `angle_distributions.csv`
  `1fd7e159e72dc7ff2689b5fac3e2bd6d5516740f7c92da874b1d426c820d24f6`
- `observations.csv`
  `339ac950226c0c90e5ba502da52853ef2d2a98412fc6779d403d8ce2ac6a563f`
- `metadata.json`
  `35292f8539026a68cb84ecfca1aa5da9f64218dd069a85033380da4c7501da57`
- `data_quality.csv`
  `69445845344881b64727ebd89f3764005a924626409157c4cfcf8889888bf06b`
- `equal_symbol_rates.csv`
  `845a1dd2bd88da2969cde8c2870a40146618fd09ad09ea91525c7e3db959f847`
- `report.md`
  `b37f3c52e81dbe78d43ae37bf2e2cc646baf647b9e4b769a5b15285186bbb120`
- `comparisons.csv`
  `cca2a88a68c6673045344ae8500be4201577f3e0b62be79a37d06419f56b7f26`
- `cohorts.csv`
  `ccf120d4c1f1a5d2a61cd7bf7f376689dc06ed90e1f8165a52574b28bf18d5c1`
- `data_gaps.csv`
  `ec548f7db0ae54d942de1927ecd843d325a37845defdc0fb5416ee7277fe15ab`

The empirical artifact directory is not committed to Git. These hashes preserve
the exact evidence reviewed for this freeze.

## Pre-confirmation findings

These findings are descriptive consequences of the prospectively frozen
measurement system. They are not claims of trading profitability or
independent-trial probabilities.

### Overall outer-target behavior

Event-weighted outer-target rates were approximately:

- exploratory 2010-2019: 0.7351
- calibration 2020-2022: 0.7235
- locked-rule validation 2023-2024: 0.7318

The overall structural rate therefore did not collapse in the locked-rule
validation partition.

### Fibonacci-confluence finding

Across all six symbols, the equal-symbol mean outer-target rate was
approximately:

- confluent: `0.8631`
- non-confluent: `0.7249`

The all-period confluent symbol rates ranged approximately from `0.8551` to
`0.8706`.

The positive confluence association was also observed in the locked-rule
2023-2024 validation partition.

This finding uses only the prospectively frozen single Fibonacci ratio and
one-pip tolerance. No alternative Fibonacci level or tolerance was selected
after inspecting outcomes.

### Retest-then-outer finding

Across all six symbols, the equal-symbol mean `retest_then_outer` rate was
approximately:

- confluent: `0.7416`
- non-confluent: `0.6074`

This is a path-conditioned structural outcome and must not be interpreted as a
trade win rate.

### Retest finding

A successful retest was associated with a lower subsequent outer-target rate
than its comparison cohort.

This negative relationship appeared repeatedly across symbols and temporal
partitions and was not the intuitive direction that might otherwise have been
expected.

It is retained as an empirical finding rather than used to redefine TG2.

## Interpretation limitations

TG4 observations can overlap in time, trend lines, and market episodes.
Consequently the observations must not be treated as independent Bernoulli
trials merely because the sample count is large.

TG4 does not establish:

- transaction-cost-adjusted profitability
- a trade entry rule
- stop placement
- position sizing
- event deduplication suitable for trading
- statistical independence
- an LSTM feature or target
- an optimal Fibonacci ratio, angle scale, or tolerance

The outer-target and retest probabilities describe the frozen TG4 structural
event definitions only.

## 2025 confirmation questions

The following questions are frozen before inspection of 2025 TG4 outcomes.

### C1: Fibonacci confluence

Under the unchanged TG4A configuration, does the held-out 2025 partition
retain a positive outer-target-rate difference between Fibonacci-confluent and
non-confluent observations?

The effect size, component rates, denominators, confidence intervals, and
cross-symbol behavior will be reported.

### C2: Retest then outer

Under the unchanged TG4A configuration, does the held-out 2025 partition
retain a positive `retest_then_outer` rate difference between confluent and
non-confluent observations?

The effect size, component rates, denominators, confidence intervals, and
cross-symbol behavior will be reported.

### C3: Retest association

Under the unchanged TG4A configuration, does the held-out 2025 partition
reproduce the observed negative association between successful retest and
subsequent outer-target attainment?

The effect size, component rates, denominators, confidence intervals, and
cross-symbol behavior will be reported.

## Confirmation interpretation rule

TG4A prospectively defined no numerical empirical pass/fail threshold.
Therefore no numerical threshold is introduced after inspection of the
2010-2024 evidence.

Confirmation will report direction, magnitude, uncertainty, denominators, and
cross-symbol consistency for the frozen questions.

A failure to reproduce a relationship will be reported as such and will not
cause the TG4A configuration or this pre-confirmation record to be rewritten.

## Confirmation lock

Until the 2025 confirmation evidence has been produced and evaluated, the
following must not be changed as part of this confirmation:

- `referenceBarScale`
- Fibonacci ratio
- pip tolerance
- angle bands
- TG1 fractal/trend-line semantics
- TG2 break/retest/pairing/outcome semantics
- TG3 AB/confluence semantics
- outcome horizons
- cohort definitions
- canonical symbol universe
- aggregation or comparison semantics

No additional Fibonacci ratios, tolerances, angle scales, or outcome
definitions may be introduced into this confirmation run.

Any such investigation is a subsequent experiment with a new configuration
and study identity.

## Anti-leakage declaration

At the time this confirmation freeze was written, the 2010-2024
pre-confirmation TG4 results had been inspected.

No TG4 outcome from the reserved 2025 confirmation partition had been
inspected or used to select the confirmation questions above.

The 2025 partition remains reserved for a single unchanged-configuration
confirmation evaluation.

