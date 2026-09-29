# Pocket 2025 Prospective Confirmation Record

## Study identity

Study: `pocket-prospective-confirmation-2025-v1`

Protocol: `phase-pocket-3-prospective-empirical-evaluation-protocol-v1`

Detector: `causal-pocket-detector-phase2-v1`

This record documents the first execution of the frozen Pocket confirmation
protocol against the previously untouched 2025 confirmation period.

The confirmation study is descriptive. It does not establish profitability,
causality, a trading rule, or incremental LSTM predictive value.

## Frozen confirmation boundary

Scoring interval:

- start inclusive: `1735689600` — 2025-01-01 00:00 UTC
- end exclusive: `1767225600` — 2026-01-01 00:00 UTC
- resolution end exclusive: `1767283200` — 2026-01-01 16:00 UTC

Frozen lookbacks:

- L10
- L15 source-default baseline
- L20

Frozen horizons:

- H4
- H16
- H64

Symbols:

- AUDCAD
- AUDUSD
- EURUSD
- GBPUSD
- USDCAD
- USDJPY

The study preserved separate event-weighted and equal-symbol aggregation,
64-bar temporal-thinning sensitivity, UTC-calendar-week block bootstrap
uncertainty, and the predeclared censoring and H64 race semantics.

No lookback winner, composite score, profitability interpretation, trading
rule, feature-selection rule, or post-hoc threshold was introduced.

## Freeze provenance

Freeze commit:

`19d91e9d7c8a6964d5acaa5f55ed97c9b24eeb8f`

Frozen configuration SHA-256:

`efb408fe8498f67320b302e7abc441d0f4030381321eaa1acf90f835bdbfb32b`

## Execution provenance

Execution-enablement commit:

`5bd74694cd2f92ce03c5ae2b81b33c976a447c06`

PocketResearch executable SHA-256:

`efb6106e8ed28eccffaecca47b8bce2fee5688349e3c3a697192816b0302a91f`

The first attempted invocation omitted the required full Git commit and
executable identity. It failed during argument parsing with:

`POCKET_CLI_FULL_GIT_COMMIT_REQUIRED`

No output directory was created by that failed invocation and no 2025 market
data was read.

The subsequent valid execution reported:

`POCKET_CONFIRMATION_EVALUATION_PUBLISHED observations=25829`

The primary artifact subsequently passed:

`POCKET_CONFIRMATION_ARTIFACT_VERIFIED`

The frozen derived report subsequently passed:

`POCKET_CONFIRMATION_DERIVED_REPORT_VERIFIED`

## Primary artifact identity

Artifact schema:

`phase-pocket-4-prospective-confirmation-artifact-v1`

Configuration file SHA-256:

`8a6bd5cc2f978f37b00434336c5f4e412757843c4df4e673ed2aa5f0028c1414`

Observations SHA-256:

`8e0c929ef2ce2ff5d3559bb6e53c12ab6e0cf0c26b6ec14932ad0a02b0e3fd86`

Aggregates SHA-256:

`8e793a71f6e66afd94b96c051673fe0e79a1e8da7d9954008737f750cc422d3d`

Observation count:

`25829`

The archived `primary/` directory is a byte-for-byte copy of the verified
first confirmation artifact.

## Derived artifact identity

Analyzer schema:

`phase-pocket-4-derived-confirmation-report-v1`

Bootstrap seed:

`17272440360112027251`

Bootstrap replicates:

`2000`

Outcomes SHA-256:

`7d827cc0f2f138b0141d406f9abde612cac9d2c352fdd96531bc7d390ca0e45e`

Equal-symbol SHA-256:

`b9ec9b5089774357037720104f389a57beafd90574c13df69ecb7e8316cd5201`

Uncertainty SHA-256:

`1fc8dd78b8ee237a6b6d3536819d1143a75a364456b148cdb10419bc7e801097`

Structural SHA-256:

`38a0564330ec74030de8319f47ff8e4d1e98b6b7bc251ef4fe7e7fbc6a91c66e`

The archived `derived/` directory is a byte-for-byte copy of the verified
first derived confirmation artifact.

## Frozen-question findings

### Q1 — Structural stability

The L15 detector continued to identify substantial bullish and bearish
populations across all six symbols in the untouched 2025 period.

Increasing lookback from L10 through L15 to L20 reduced raw event density and
generally increased event spacing. The 64-bar temporally thinned populations
changed considerably less than the raw populations.

The confirmation evidence is consistent with the preconfirmation finding that
lookback primarily changes detector selectivity and event density.

### Q2 — Touch and close-boundary fill

For L15 equal-symbol aggregation, 2025 touch rates were:

| Direction | H4 | H16 | H64 |
|---|---:|---:|---:|
| Bearish | 0.68319 | 0.82929 | 0.91290 |
| Bullish | 0.68021 | 0.81963 | 0.90278 |

Close-boundary fill rates were:

| Direction | H4 | H16 | H64 |
|---|---:|---:|---:|
| Bearish | 0.35819 | 0.61677 | 0.79140 |
| Bullish | 0.34127 | 0.58716 | 0.76304 |

Across L15 symbol/direction cells, median first touch was generally one bar
among events touched by H4/H16 and one to two bars among events touched by
H64. Median close-boundary fill was approximately two bars at H4, three to
four bars at H16, and five to seven bars at H64.

The frequent and relatively rapid revisit/fill behavior observed before the
confirmation period reproduced in 2025.

### Q3 — Subsequent path behavior

At L15 H64, all twelve symbol/direction cells contained both
`continuation_first` and `revisit_first` observations, together with substantial
`same_bar_intrabar_order_indeterminate` and censored observations.

The frozen protocol did not define a denominator for converting these H64 race
counts into race proportions; therefore this record preserves them as counts
rather than constructing a post-hoc rate.

MFE and MAE remained large and similar in magnitude relative to Pocket width,
while directional close-return medians were much smaller and varied across
symbols and directions.

The confirmation evidence therefore remains consistent with a high-traversal
market state rather than a deterministic continuation or reversal rule.

### Q4 — Lookback sensitivity

L10, L15, and L20 produced very similar equal-symbol touch, close-fill,
MFE/MAE, and directional-return distributions despite materially different raw
event densities.

For H64 touch specifically:

| Lookback | Bearish | Bullish |
|---|---:|---:|
| L10 | 0.91245 | 0.90757 |
| L15 | 0.91290 | 0.90278 |
| L20 | 0.91101 | 0.90203 |

The confirmation study does not identify or select a winning lookback.

### Q5 — Cross-symbol and temporal consistency

For L15 H64, every 2025 symbol/direction cell had touch rates between
approximately 0.882 and 0.927 and close-fill rates between approximately
0.729 and 0.821.

The structural revisit phenomenon therefore did not disappear in any of the
six confirmation symbols.

Directional close-return medians remained heterogeneous across symbols and
directions and should not be generalized into a directional Pocket rule.

The untouched 2025 period independently reproduced the principal structural
behavior observed during preconfirmation research.

## Scientific conclusion

The frozen 2025 confirmation study supports the preconfirmation conclusion
that `causal-pocket-detector-phase2-v1` identifies a reproducible market
structure characterized by frequent, relatively rapid revisit and substantial
subsequent price traversal.

The strongest reproducible information concerns price-path structure and
revisit behavior, not a simple directional-return prediction.

This study does not establish:

- profitability,
- causality,
- a trading strategy,
- an optimal lookback,
- incremental predictive value to the LSTM,
- or justification for adding Pocket Tensor channels.

Any LSTM integration must be treated as a separate prospective experiment with
its causal representation and comparison protocol frozen before evaluating
model outcomes.

## Archived manifest identities

Primary manifest SHA-256:

`52c72fe579e57e730e485f56082fa16402e818279619033cda690e67e9ec9915`

Derived manifest SHA-256:

`92fb01de865e28ea4960231ff7aee9801319dd767207de14cd9aadc2ff477fa6`
