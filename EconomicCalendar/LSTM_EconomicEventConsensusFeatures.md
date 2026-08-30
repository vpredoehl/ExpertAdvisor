# LSTM economic-event consensus and initial-actual feature contract

## Source and causal boundary

`LoadEconomicEventsForFeatureRange` performs one ordered range query over
`economic_event` with left joins to `economic_event_selected_consensus` and
`economic_event_feature_release_actual`. Feature code does not select a
provider, a latest actual, or an unselected consensus observation. Provider and
authoritative source identity are retained only for diagnostics/provenance.

The persisted consensus rows do not contain a historical `known_at` instant for
each forecast revision. Therefore, the final selected forecast is not projected
into arbitrary earlier bars. For a completed 15-minute bar with information
cutoff `bar_start + 15 minutes`:

- an event exactly at the cutoff supplies consensus, but not actual/surprise;
- otherwise consensus describes the most recent released event;
- a future event after the cutoff supplies nothing;
- semantic-layout-v4 surprise remains unavailable at every cutoff;
- semantic-layout-v5 surprise is visible only when the certified initial
  actual has `available_at < information_cutoff`. Equality remains invisible.

## DOL/ETA Weekly Claims treatment

`DOL_ETA/WEEKLY_CLAIMS` is explicitly mapped to the existing employment model
family. It affects only `employment_event` and `employment_recency_decay`; it
does not activate inflation, growth, fed-policy, or consumer-demand features.
The authoritative publication is the U.S. Department of Labor / Employment
and Training Administration's Unemployment Insurance Weekly Claims report,
which contains initial and continued claims in one release artifact. The raw
calendar therefore correctly retains one `WEEKLY_CLAIMS` occurrence rather
than manufacturing separate statistic-level events.

Each imported occurrence uses the release date and 08:30 America/New_York
embargo clock published in its immutable first-party DOL/ETA artifact. Most
timestamps are `exact`. A documented 2011-2012 archive defect used the wrong
EST/EDT abbreviation on 45 otherwise authoritative 08:30 releases; those rows
preserve the published local clock, resolve it with historical
America/New_York rules, and are marked `reconstructed`. This bounded
reconstruction is part of the authoritative calendar audit contract and is
causally suitable for the same strict completed-bar cutoff as exact rows.

Production schema 082 has no selected consensus for Weekly Claims. Therefore,
when it is the relevant event, consensus presence/low/high/range remain zero;
the reserved surprise fields also remain zero.

This compatibility correction did not add or reorder a v4 feature, so its
physical Tensor width remains 67 and model input width remains 71. Although
Weekly Claims changes the values carried by the existing
employment columns relative to the incomplete mapper, v4 had not been deployed
to a production width-63 or width-71 model when the correction was made. The
correction therefore completes the pre-deployment v4 employment-family
contract without changing an already-trained production economic-feature
model. Any non-production width-63 or width-71 artifact trained with the
incomplete mapper must not be assumed semantically interchangeable with newly
generated inputs. Width-53 models project only Tensor columns 0-48 and cannot
consume any economic-event column.

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
| 63 | `released_event_has_surprise` | reserved; always 0 |
| 64 | `released_event_surprise` | reserved; always 0 |
| 65 | `released_event_surprise_abs` | reserved; always 0 |
| 66 | `released_event_surprise_direction` | reserved; always 0 |

Missing consensus encodes all eight v4 fields as zero. The four reserved
surprise positions remain zero permanently, including in v5 tensors.

The persisted OANDA and Myfxbook provider artifacts do not carry an explicit
contract proving that an `actual` is the original value known at the historical
release instant rather than a later revision. Provider identity, event time,
import time, artifact date, observation date, and mere actual-value presence do
not establish that provenance. The runtime therefore never derives surprise
from provider actual fields.

Migration 088 supplies the separate authoritative contract. Its immutable
table stores initial and revised observations, while the feature view exposes
only publication state `initial`, revision sequence 0. Unique event/revision
identity and an immutable trigger prevent a later revision from replacing that
row. Source agency must match the authoritative event; source observation ID,
artifact path/SHA-256, semantic contract, and nonempty JSON provenance are
mandatory. Raw values must reproduce canonical values through their persisted
positive scale.

Semantic layout v5 appends four channels at Tensor columns 67 through 70:

| Column | Name | Encoding |
| ---: | --- | --- |
| 67 | `authoritative_initial_has_surprise` | 1 only for an available compatible scalar pair |
| 68 | `authoritative_initial_surprise` | normalized canonical `actual - forecast` |
| 69 | `authoritative_initial_surprise_abs` | absolute normalized surprise |
| 70 | `authoritative_initial_surprise_direction` | -1, 0, or 1 |

Forecast and actual must both be present scalars with the same canonical unit
and qualifier. Each positive scale is validated as source-to-canonical
provenance; equal scales are not required after canonicalization. Ranges and
semantic mismatches fail closed. Missing forecast/actual remains distinct from
numeric zero, so equal numeric values set `has_surprise=1` with zero magnitude.

FOMC range consensus preserves both endpoints and sets the range bit. No
midpoint or interval-subtraction rule is invented, so range surprise remains
unavailable.

## Width and ancestry

- physical Tensor width: 49 -> 59 -> 67 -> 71;
- model input width, including four return features: 53 -> 63 -> 71 -> 75;
- semantic layout: v3 -> v4 -> v5 with explicit append-only predecessors;
- v3 width 63 remains registered and projects exactly columns 0 through 58;
- v4 width 71 remains registered and projects exactly columns 0 through 66;
- expansion to width 75 continues through the existing explicit
  `resume_expand_input_width` workflow with zero-initialized appended weights.

Semantic-layout-v5 runtime requires migration 088. The historical v4 consensus
contract remains defined by migration 082 and never consumes the new columns.
