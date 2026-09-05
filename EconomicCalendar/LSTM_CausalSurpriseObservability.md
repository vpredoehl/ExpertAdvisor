# Causal First-Release Surprise Observability

The Phase-5 diagnostic is a read-only view of the Phase-2 causal surprise
calculation. It does not apply the experiment's feature-ablation mask. That
mask is applied later, when Tensor columns are projected into model input, so
an empty-mask control and the canonical two-channel ablation have the same
upstream diagnostic identity when all other scientific fields match.

## Commands

```text
LSTM_Release --causal-surprise-observability=619
LSTM_Release --causal-surprise-observability=620
```

The default scope is `combined`. It evaluates the train and inference
populations independently and sums them; this preserves cold-boundary behavior
and avoids manufacturing rows in any gap between the two ranges. A single
population can be selected with:

```text
--causal-surprise-observability-scope=train
--causal-surprise-observability-scope=infer
```

Each range is half-open. The denominator is the number of candlestick feature
rows in the requested experiment range after excluding the experiment's
stateful full-history warmup prefix. `source_rows` and
`warmup_rows_excluded` make that exclusion explicit. The diagnostic counts
feature rows, not training windows or predictions, and requires neither a
checkpoint nor final-inference evidence.

## Disposition partition

Every denominator row has exactly one terminal disposition from the same
`EconomicEventFeatureEngine` decision used to create Tensor columns 71 and 72:

- `no_relevant_event`
- `provenance_unavailable`
- `ambiguous`
- `not_yet_available`
- `missing_consensus` (the runtime enum is `missingForecast`)
- `incompatible`
- `available`

The first-release actual is usable only when the Phase-1 point-in-time result
is `proven_first_release` and `proven_available_at` is less than or equal to
the completed-bar information cutoff. The exact boundary is therefore
available. Latest/revised/provider-current values and retrieval or ingestion
times are not fallback evidence.

An unavailable row retains the Tensor placeholder `(availability=0,
surprise=0)` but is not a valid zero. Valid available rows are partitioned
using the exact runtime float comparisons `< 0`, `== 0`, and `> 0`.
Distribution minimum, maximum, and mean include available rows only.

Clamp counts are authoritative: the feature engine records whether its own
pre-clamp normalized value crossed the fixed `[-10,+10]` bounds at the same
point where it writes the bounded feature value. No second normalization
implementation is used.

Source summaries are compact counts for available rows by first-release
actual source, selected consensus source, and event family. Each dimension
sums to `surprise_available_count`.

## Identity and pair interpretation

The per-experiment diagnostic identity includes experiment ID and reported
feature-ablation mask. The upstream identity intentionally excludes experiment
ID, the downstream mask, and scheduler metadata. It binds symbol, horizon,
scope/ranges, warmup scope, Donchian mode/lookback, model width/layout, feature
names and Tensor columns, the migration-090 PIT contract, fixed normalization
and clamp contract, repository/selection contract, and diagnostic semantic
version. The coverage identity additionally binds every reported count,
statistic, and source summary.

For the controlled 619/620 shape, upstream parity permits only the intentional
difference between an empty mask and:

```text
causal_first_release_surprise_available,causal_first_release_surprise
```

It fails closed on other mask differences or on any scientific upstream
field mismatch. Interpret the Phase-4 model-result delta together with the
Phase-5 coverage rate: a small delta with broad upstream coverage is evidence
about model utility, while a small delta with sparse availability is
underpowered evidence about the surprise signal.
