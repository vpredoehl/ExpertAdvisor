# LSTM economic-event consensus feature contract

## Source and causal boundary

`LoadEconomicEventsForFeatureRange` performs one ordered range query over
`economic_event` with a left join to `economic_event_selected_consensus`.
Feature code does not select a provider and does not query unselected consensus
observations. Provider identity is retained only for diagnostics.

The persisted consensus rows do not contain a historical `known_at` instant for
each forecast revision. Therefore, the final selected forecast is not projected
into arbitrary earlier bars. For a completed 15-minute bar with information
cutoff `bar_start + 15 minutes`:

- an event exactly at the cutoff supplies consensus, but not actual/surprise;
- an event strictly before the cutoff is released and may supply surprise;
- otherwise consensus/surprise describe the most recent released event;
- a future event after the cutoff supplies nothing.

At a shared timestamp, highest `event_importance` wins; a positive lower
`economic_event_id` is the deterministic tie-breaker. This affects only the
single relevant-event consensus/surprise context. Existing per-family event and
recency features retain their prior behavior.

## Normalization

All inputs use persisted canonical values. Count source scales have already been
applied before this layer.

| Canonical family | Required unit | Static divisor |
| --- | --- | ---: |
| CPI, PPI, PCE, GDP, DURABLE_GOODS, RETAIL_SALES | percent | 10 percentage points |
| FOMC | percent | 10 percentage points |
| EMPLOYMENT, EMPLOYMENT_ANNUAL | count | 1,000,000 |
| JOLTS | count | 10,000,000 |

No dataset distribution, future observation, rolling statistic, or fitted scale
is used.

## Appended components

The previous economic-event block remains columns 49 through 58. Semantic
layout v4 appends columns 59 through 66:

| Column | Name | Encoding |
| ---: | --- | --- |
| 59 | `relevant_event_has_consensus` | 0 or 1 |
| 60 | `relevant_event_consensus_low` | normalized low/scalar endpoint |
| 61 | `relevant_event_consensus_high` | normalized high; scalar duplicates low |
| 62 | `relevant_event_consensus_is_range` | 0 or 1 |
| 63 | `released_event_has_surprise` | 0 or 1 |
| 64 | `released_event_surprise` | normalized `actual - consensus` |
| 65 | `released_event_surprise_abs` | absolute normalized surprise |
| 66 | `released_event_surprise_direction` | -1, 0, or +1 |

Missing consensus encodes all eight fields as zero. A true zero surprise is
distinguished from missing/incompatible surprise by
`released_event_has_surprise = 1`.

Surprise requires scalar forecast and actual values with identical persisted
unit, source scale, and qualifier. Missing or incompatible actual semantics set
the validity bit and all surprise values to zero.

FOMC range consensus preserves both endpoints and sets the range bit. Range
surprise is unavailable because no midpoint or interval-subtraction rule is
part of the persisted semantic contract. Compatible scalar FOMC observations
may produce surprise normally.

## Width and ancestry

- physical Tensor width: 59 -> 67;
- model input width, including four return features: 63 -> 71;
- semantic layout: v3 -> v4, append-only predecessor v3;
- v3 width 63 remains registered and projects exactly columns 0 through 58;
- expansion to width 71 continues through the existing explicit
  `resume_expand_input_width` workflow with zero-initialized appended weights.

Migration 083 only extends the selected view with actual semantics from the
same immutable selected provider observation. It does not mutate consensus
rows or change selection precedence.
