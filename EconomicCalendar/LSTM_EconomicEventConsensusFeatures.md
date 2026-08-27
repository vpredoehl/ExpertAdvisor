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
- otherwise consensus describes the most recent released event;
- a future event after the cutoff supplies nothing;
- surprise is unavailable at every cutoff under the current provenance
  contract.

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

This compatibility correction does not add or reorder a feature, so physical
Tensor width remains 67, model input width remains 71, and semantic layout
remains v4. Although Weekly Claims changes the values carried by the existing
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

Missing consensus encodes all eight fields as zero. A true zero surprise is
not represented by the current contract because surprise is disabled.

The persisted OANDA and Myfxbook provider artifacts do not carry an explicit
contract proving that an `actual` is the original value known at the historical
release instant rather than a later revision. Provider identity, event time,
import time, artifact date, observation date, and mere actual-value presence do
not establish that provenance. The runtime therefore does not load provider
actual fields and cannot derive `actual - consensus`. All four surprise
channels are reserved and remain zero until a future persisted, deterministic
first-release/revision provenance contract is implemented.

Current active semantic-layout-v4 model inputs are consensus-only: presence,
normalized low/high endpoints, and the range bit are active, followed by four
reserved zero surprise channels.

FOMC range consensus preserves both endpoints and sets the range bit. No
midpoint or interval-subtraction rule is invented. Like every other family,
FOMC surprise remains unavailable under the current provenance contract.

## Width and ancestry

- physical Tensor width: 59 -> 67;
- model input width, including four return features: 63 -> 71;
- semantic layout: v3 -> v4, append-only predecessor v3;
- v3 width 63 remains registered and projects exactly columns 0 through 58;
- expansion to width 71 continues through the existing explicit
  `resume_expand_input_width` workflow with zero-initialized appended weights.

The runtime requires production schema through migration 082. It queries only
the consensus and provider-diagnostic columns exposed by the migration-082
`economic_event_selected_consensus` view. No migration 083 is required for the
consensus-only Phase-4 deployment.
