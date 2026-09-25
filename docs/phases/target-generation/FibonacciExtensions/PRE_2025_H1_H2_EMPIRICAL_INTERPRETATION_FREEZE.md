# Pre-2025 H1/H2 empirical interpretation freeze

## Status

This document freezes the interpretation of the first causal Fibonacci
extension H1/H2 historical study before any 2025 H1/H2 outcome is accessed.

Historical methodology remains:

- baseline methodology commit:
  `0440faa9a054d7c7c02df75e9162d3d85b33533d`
- frozen evaluation policy:
  `causal-fibonacci-extension-h1-h2-policy-frozen-v2`
- observation schema:
  `causal-fibonacci-extension-observation-v2`
- aggregate schema:
  `causal-fibonacci-extension-aggregate-v2`
- timeframe: completed 15-minute bars
- discovery/evaluation interval:
  `[2010-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`
- actual available historical coverage:
  `2010-01-03T22:00:00Z` through `2024-12-31T21:45:00Z`

This interpretation freeze does not change the prospective policy, endpoint
definitions, horizon, tolerance, invalidation policy, A/B geometry, target
ratios, denominator rules, or censoring rules.

2025 remains an untouched confirmation period.

## First-study population

The frozen pre-2025 study used:

- `audcadrmp`
- `audusdrmp`
- `eurusdrmp`
- `gbpusdrmp`
- `usdcadrmp`
- `usdjpyrmp`

The study processed:

- 2,219,451 usable completed bars
- 568,203 unique structural events
- 9,004 retained material timestamp gaps
- no synthetic bars
- no excluded symbols
- no duplicate event identities

Events overlap structurally and share market paths. They must not be interpreted
as independent trials. The reported Wilson intervals do not model this
dependence.

## Frozen primary endpoint results

| Endpoint | Success | Failure | Resolved | Probability |
|---|---:|---:|---:|---:|
| H1 1.618 | 508,961 | 53,464 | 562,425 | 90.4940% |
| H2 .382 | 434,951 | 127,563 | 562,514 | 77.3227% |
| H2 .500 descriptive | 404,591 | 157,923 | 562,514 | 71.9255% |
| H2 .618 descriptive | 374,327 | 188,186 | 562,513 | 66.5455% |

H2 .382's external expert prior was approximately 80%.

The measured pre-2025 H2 .382 probability minus that prior is approximately:

`-2.6773 percentage points`

The expert prior remains metadata. It was not used as an acceptance threshold
and did not select or tune the frozen policy.

H1 had no numeric expert benchmark.

## Cumulative probability by completed bar

These are descriptive measurements under the unchanged 20-bar endpoints.

| Endpoint | B1 | B2 | B3 | B4 | B8 | B12 | B16 | B20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 1.618 | 73.317% | 78.060% | 80.738% | 82.490% | 86.344% | 88.242% | 89.527% | 90.494% |
| H2 .382 | 45.930% | 52.867% | 57.330% | 60.550% | 68.100% | 72.295% | 75.148% | 77.323% |
| H2 .500 | 38.178% | 45.043% | 49.654% | 53.106% | 61.420% | 66.134% | 69.402% | 71.925% |
| H2 .618 | 31.672% | 38.273% | 42.806% | 46.324% | 55.034% | 60.137% | 63.727% | 66.545% |

The 20-bar policy remains unchanged. These timing measurements are descriptive
results and do not redefine the original hypotheses.

## Timing of successful events

Percentage of eventual successful events completed by each checkpoint:

| Endpoint | B1 | B2 | B3 | B4 | B8 | B12 | B16 | B20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 1.618 | 81.018% | 86.260% | 89.219% | 91.155% | 95.414% | 97.511% | 98.931% | 100.000% |
| H2 .382 | 59.400% | 68.372% | 74.144% | 78.308% | 88.072% | 93.497% | 97.187% | 100.000% |
| H2 .500 | 53.080% | 62.624% | 69.035% | 73.834% | 85.394% | 91.948% | 96.492% | 100.000% |
| H2 .618 | 47.595% | 57.514% | 64.326% | 69.613% | 82.701% | 90.369% | 95.764% | 100.000% |

## Conditional target-hit probability by bar

The following measures:

`P(target first reached on bar N | target not reached before bar N)`

| Bar | H1 1.618 | H2 .382 | H2 .500 | H2 .618 |
|---:|---:|---:|---:|---:|
| 1 | 73.317% | 45.930% | 38.178% | 31.672% |
| 2 | 17.776% | 12.829% | 11.104% | 9.661% |
| 3 | 12.205% | 9.469% | 8.390% | 7.343% |
| 4 | 9.095% | 7.547% | 6.856% | 6.152% |
| 5 | 7.342% | 6.224% | 5.752% | 5.172% |
| 6 | 6.399% | 5.448% | 4.975% | 4.493% |
| 7 | 5.485% | 4.738% | 4.368% | 4.007% |
| 8 | 4.857% | 4.267% | 3.944% | 3.640% |
| 9 | 4.164% | 3.791% | 3.483% | 3.234% |
| 10 | 3.697% | 3.578% | 3.358% | 3.061% |
| 11 | 3.607% | 3.351% | 3.107% | 2.882% |
| 12 | 3.218% | 3.131% | 2.874% | 2.688% |
| 13 | 3.041% | 2.966% | 2.701% | 2.542% |
| 14 | 2.904% | 2.744% | 2.597% | 2.387% |
| 15 | 2.821% | 2.555% | 2.427% | 2.227% |
| 16 | 2.640% | 2.455% | 2.294% | 2.171% |
| 17 | 2.577% | 2.348% | 2.215% | 2.080% |
| 18 | 2.443% | 2.244% | 2.115% | 1.988% |
| 19 | 2.302% | 2.300% | 2.140% | 2.040% |
| 20 | 2.252% | 2.163% | 2.044% | 1.901% |

## Frozen interpretation before confirmation

### H1

The pre-2025 evidence shows a strong association between a causally confirmed
completed close beyond 1.272 and subsequent reach of 1.618 within 20 completed
bars.

The phenomenon is strongly front-loaded.

Of all resolved H1 events:

- 73.317% reached 1.618 on the next completed bar.
- 82.490% had reached it by bar 4.
- 86.344% had reached it by bar 8.
- 90.494% had reached it by bar 20.

Of successful H1 events, 81.018% succeeded on bar 1.

The conditional first-hit probability falls sharply after bar 1 and generally
continues declining as the event ages.

This timing behavior is descriptive evidence. It does not alter the frozen
20-bar H1 endpoint.

### H2

The pre-2025 measured probability of the primary .382 pullback endpoint is
77.323%, compared with the approximately 80% external expert prior.

H2 is also front-loaded, but less extremely than H1.

Of all resolved H2 .382 events:

- 45.930% reached .382 on bar 1.
- 60.550% had reached it by bar 4.
- 68.100% had reached it by bar 8.
- 77.323% had reached it by bar 20.

The deeper descriptive targets maintain the expected ordering throughout the
predefined checkpoints:

`.382 > .500 > .618`

The conditional first-hit probabilities for all three pullback depths generally
decline as the event ages.

This timing behavior is descriptive evidence. It does not alter the frozen
20-bar H2 endpoint or endpoint hierarchy.

## Same-bar/indexing verification

A direct observation-level audit was performed after the timing analysis.

H1:

- successful observations: 508,961
- hit at or before eligibility bar: 0
- bars-to-target distance mismatches: 0
- target timestamp not later than eligibility timestamp: 0

H2 .382:

- successful observations: 434,951
- hit at or before rejection bar: 0
- bars-to-target distance mismatches: 0
- target timestamp not later than rejection timestamp: 0

Therefore the large bar-1 concentrations observed above are not caused by
counting the H1 eligibility bar or H2 rejection bar as the first outcome bar.

## Frozen 2025 confirmation questions

Before any 2025 H1/H2 outcome is accessed, the following confirmation questions
are fixed.

1. Does H1 continue to show a high 1.272-to-1.618 continuation probability
   under the unchanged frozen 20-bar policy?

2. Does H1 retain the strong concentration of successful outcomes in the first
   completed bar after eligibility?

3. Does H1 retain the general conditional first-hit time-decay pattern observed
   pre-2025?

4. Does H2 .382 remain broadly consistent with the pre-2025 measured
   probability of 77.323% under the unchanged frozen policy?

5. Does H2 .382 retain its front-loaded timing and general conditional
   first-hit time-decay pattern?

6. Do the descriptive H2 target probabilities retain the ordering:

   `.382 > .500 > .618`

7. Are the principal findings reasonably consistent across the same predefined
   symbols and UpAB/DownAB directions?

No new numeric acceptance threshold is invented for these questions.

The pre-2025 estimates and timing curves are reference measurements, not
parameters to be optimized against 2025.

## Confirmation boundary

The 2025 confirmation must use the existing frozen policy without changing:

- Fibonacci ratios
- A/B construction
- causal availability
- one-canonical-pip tolerance
- eligibility/rejection definitions
- 20-completed-bar horizon
- no-invalidation policy
- target hierarchy
- censoring rules
- denominator rules

The 2025 analysis must be reported separately from the 2010-2024 discovery
period. It must not be pooled into the pre-2025 estimates before the separate
confirmation result is recorded.

H3 remains closed:

`missing_authoritative_d_extension_contract`

No model feature, Tensor semantic layout, training objective, scheduler
behavior, or production database state is changed or authorized by this
interpretation freeze.
