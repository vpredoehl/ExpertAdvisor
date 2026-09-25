# Causal Fibonacci model-feature design

## 1. Scope and non-goals

This is a design freeze for a future fixed-width, causal model-input
representation of Fibonacci H1/H2 structural state. It is not an
implementation authorization. Sections 9--16 preserve the earlier conditional
proposal and its now-resolved gate; sections 17--23 record the later structural
diagnostic evidence and replace that proposal.

It does not change the frozen H1/H2 methodology, ratios, A/B construction,
TG1/TG3 semantics, one-canonical-pip tolerance, eligibility/rejection rules,
20-completed-bar horizon, no-invalidation policy, target hierarchy, censoring,
or denominator semantics. It does not reopen H3, which remains closed:
`missing_authoritative_d_extension_contract`.

It does not propose using observed H1/H2 success rates, Wilson intervals,
future target hits, eventual success/failure, bars-to-target, censoring, or any
other outcome-derived value as a model feature. The confirmed results establish
that this causal structural family merits investigation; they are not per-bar
observations or model priors.

## 2. Provenance and frozen empirical basis

- Baseline methodology: `0440faa9a054d7c7c02df75e9162d3d85b33533d`
- Pre-2025 interpretation freeze: `8ba1b81101697321e27fe5afc2e4fc4927685ced`
- Frozen confirmation boundary: `4b8577c80474c1e8886fe8c0a464676c5d7554f8`
- Untouched 2025 confirmation artifact: `69317d92bd0dd41880b5104ee90211c403a47a4a`
- Confirmation interpretation: `e9e4a727`
- Policy: `causal-fibonacci-extension-h1-h2-policy-frozen-v2`
- Study: `causal-fibonacci-extension-2025-confirmation-v1`

The authoritative pre-commitment and result records are:

- [`PRE_2025_H1_H2_EMPIRICAL_INTERPRETATION_FREEZE.md`](PRE_2025_H1_H2_EMPIRICAL_INTERPRETATION_FREEZE.md)
- [`2025_H1_H2_CONFIRMATION_INTERPRETATION.md`](2025_H1_H2_CONFIRMATION_INTERPRETATION.md)

The untouched confirmation reproduced the H1 and H2 structural findings and
the `.382 > .500 > .618` ordering. Its timing evidence makes *observed event
age* a reasonable candidate state variable, but does not make the empirical
hazards or aggregate probabilities features.

## 3. Current model-input architecture

The current authoritative layout is discovered from `FeatureLayout.hpp`,
`ModelInputContract.hpp`, `ModelInputExpansion.hpp`, `ModelInputFeatureSemantics.hpp`,
and `Tensor.cpp`:

- semantic layout version: **8**;
- physical Tensor feature width: **76**;
- model-only return suffix: **4** completed-close return features;
- current model input width: **80**;
- the three newest Tensor columns (73--75) are same-completed-bar TG4 pulse
  bits: `tg4_inner_break_any`, `tg4_source_tg3_structurally_eligible`, and
  `tg4_source_tg3_confluent`;
- feature additions are append-only and persisted model layouts/widths are
  checked against the semantic-layout registry;
- expansion initializes new input-gate rows to zero and records semantic and
  lineage provenance.

Tensor rows are created after receipt of one completed canonical bar. Some
existing causal features are predecessor-only and some deliberately include
the just-completed bar; both conventions are explicitly documented per
feature. A Fibonacci feature row must likewise state its completed-bar timing,
rather than silently borrowing one convention.

There is an important configuration distinction. The present production TG4
pulse adapter uses `tg4a-derived-source-utl-up-ab-only-v1`, a 0.618
retracement/confluence configuration with `SourceUTLUpABOnly`. The confirmed
H1/H2 evaluator creates frozen extension/pullback state from the symmetric
UpAB/DownAB A/B structures. The existing three-bit pulse is therefore neither
a sufficient source for, nor evidence of parity with, the proposed H1/H2
feature family.

## 4. Causal Fibonacci state machine relevant to model features

For every TG3 A/B structure:

1. A and B are anchored only after their TG1 fractals are confirmed. The A/B
   structure first exists at `source_availability_bar`, which is B's causal
   confirmation bar. Geometric pivot existence before that bar is not usable.
2. From that point, the frozen 1.272 extension, 1.618 extension, and .382,
   .500, and .618 pullback levels are deterministic functions of the confirmed
   A/B geometry and the frozen tolerance.
3. On each completed bar the historical evaluator preserves this order:
   `ResolvePending`, `ObserveRejections`, `ObserveTouches`, then
   `ObserveBeyond`.
4. A touch is a causal wick/zone observation. H1 eligibility is a strict
   completed close beyond the frozen far 1.272 boundary. H2 eligibility is a
   later completed-close rejection after a previously observed touch.
5. Resolution follows eligibility only on subsequent completed bars. The
   model representation below deliberately does not encode target resolution,
   even when it has become known, so it represents structural state rather
   than an outcome label.

The proposed row timing is **post-completed-bar**: update A/B and H1/H2
structural state using the bar, then emit the feature row used for a decision
made after that bar closes. Consequently an A/B confirmation, touch, strict
beyond, or valid rejection first observed on bar *t* may appear in row *t*.
No target reach on bar *t* is encoded.

## 5. Candidate per-structure inventory and availability

| Candidate | Causally well-defined? | Availability point | Proposed use |
|---|---|---|---|
| `fib_ab_available` | Yes | B confirmation / `source_availability_bar` | Aggregate count, not an unconfirmed pivot bit |
| `fib_ab_direction` | Yes | A/B availability | Direction-separated halves |
| Distance to 1.272 | Yes | A/B availability; recomputed from current completed close | Nearest oriented distance |
| Distance to 1.618 | Yes | A/B availability; recomputed from current completed close | Nearest oriented distance |
| Distance to .382/.500/.618 pullbacks | Yes | A/B availability; recomputed from current completed close | Nearest oriented distances |
| `fib_1272_touched` | Yes | Completed bar whose high/low first reaches the frozen zone | Directional state count |
| `fib_1272_beyond` | Yes | Completed bar whose close is strictly beyond the far boundary | Directional state count and age |
| `fib_1272_rejection_confirmed` | Yes | Later completed rejection bar after a touch | Directional state count and age |
| Bars since A/B availability | Yes | Zero at availability, increments per completed bar | Youngest/oldest aggregate age |
| Bars since beyond/rejection | Yes when event occurred | Zero on event bar, increments per completed bar | Youngest event age |
| H1/H2 success/failure/censoring | Not for this feature family | Future or outcome state | Excluded |
| Target hit/bars-to-target | Not for this feature family | May be known only after eligibility and is an outcome | Excluded |
| Empirical H1/H2 probabilities/hazards | Not a causal bar observation | Derived from aggregate future outcomes | Excluded |

## 6. Leakage and availability audit

| Value on row *t* | Allowed | Required safeguard |
|---|---|---|
| A/B geometry | Only if B was confirmed no later than *t* | Never create state from a future fractal or pre-confirmation pivot |
| Level distance | Yes, from current close and fixed confirmed geometry | Use no future close/high/low; level itself is immutable after availability |
| Touch at *t* | Yes | Process only after rejection logic, matching evaluator order |
| Strict beyond at *t* | Yes | Use completed close and frozen far-edge semantics, not an intrabar forecast |
| Rejection at *t* | Yes | Require a stored earlier touch and preserve `bar > touch_bar` semantics |
| Event age | Yes | Count only completed bars from a stored causal event; zero is valid on the event bar |
| H1/H2 target hit at *t* | Excluded | Do not add resolution, success, failure, or remaining-horizon bits |
| Any future target outcome | Excluded | Feature builder must have no outcome tracker/resolution dependency |

The implementation must preserve the evaluator's `ResolvePending`-before-new-
eligibility ordering. In particular, it must not infer a same-bar H1 or H2
success from a newly eligible state. The feature family avoids the issue by
not encoding target resolution at all.

## 7. Multiple-active-structure problem

The TG3 tracker retains active A/B structures for at most 2,048 bars and has a
512-structure capacity in the frozen historical configuration. The historical
observation CSV is event-oriented, not a per-bar active-set snapshot. It gives
each structure's availability bar but does not preserve per-bar live-set
membership, capacity evictions, or the model-feature retention state.

A local read-only reconstruction from the pre-2025 `observations.csv`, using
only each row's availability bar and a nominal 2,048-bar lifetime, is still
instructive: for each of the six symbols, more than one nominally live
structure occurs on at least 99.995% of reconstructed covered bars, and both
directions nominally overlap on at least 99.993%. The unconstrained envelope
reaches 577--601 structures per symbol. Those values exceed the 512 live-set
capacity and therefore are **not exact active-set statistics**; they prove
that winner selection would discard material simultaneous state, not that the
actual retained maximum exceeded capacity.

The artifacts cannot reliably determine exact multiplicity, same/opposite
direction overlap, or age distributions after bounded retention. They also
cannot establish whether the proposed H1/H2 feature adapter should retain the
same live set as the current UpAB-only production pulse adapter. No K, recency
rule, or structural-strength rule is therefore selected from outcome evidence.

## 8. Representation alternatives

| Family | Causal/fixed-width | Main loss or instability | H1/H2 and age | Testing/ablation |
|---|---|---|---|---|
| One canonical structure | Yes if tie-break is causal | Discards almost all concurrent state; discontinuous winner swaps; recency/closest/strength choice creates arbitrary bias | Good only for winner | Simple, but unjustified |
| Top-K slots | Yes with total causal ordering | K is ungrounded; overflow drops state; slot churn when ranks change; dimensional cost is K times per-structure width | Good within retained slots | Deterministic but K/ordering-sensitive |
| Direction-separated single winner | Fixed and symmetric | Still discards same-direction multiplicity and has winner churn | Good only per winner | Better than one global winner, still lossy |
| Direction-separated set aggregation | Fixed, causal, permutation-invariant | Loses identity and joint cross-level geometry; must define live set exactly | Counts, nearest distances, and state/age summaries preserve key H1/H2 state | Strong deterministic and group-ablation properties |
| Hybrid aggregate plus top-K | Fixed | Retains aggregate robustness but reintroduces K/ranking churn and extra width | Richer | Appropriate only if diagnostics justify K |

## 9. Historical conditional recommended representation

Subject to the implementation gate in section 21, the recommended family is a
**direction-separated set aggregate**. It does not select a winner and has no
K. UpAB and DownAB have mirror-image halves, which preserves directional
symmetry and makes opposing simultaneous structures visible rather than
netting them away.

This is a conditional recommendation, not an implementation-ready final
contract. The live-set definition and the relation to the currently
UpAB-only production adapter must be diagnosed first. If that diagnostic finds
that the proposed H1/H2 state cannot share a deterministic bounded live set
with the desired source configuration, no feature layout should be added.

## 10. Historical conditional 27-column feature schema

If the gate is satisfied, append the following 27 Tensor columns. Each
`{dir}` expands identically for `up` and `down`, for 13 columns per direction.

| Feature | Definition |
|---|---|
| `fib_price_scale_valid` | 1 iff the current causal price scale is finite and positive; otherwise 0 |
| `fib_{dir}_active_count_log` | `log1p(active_{dir}) / log1p(512)` |
| `fib_{dir}_youngest_age_2048` | Minimum active A/B age, `min(age, 2048)/2048`; 0 if no active structure |
| `fib_{dir}_oldest_age_2048` | Maximum active A/B age, `min(age, 2048)/2048`; 0 if no active structure |
| `fib_{dir}_nearest_1272_signed_atr` | Oriented normalized distance of the active structure whose absolute 1.272 distance is smallest |
| `fib_{dir}_nearest_1618_signed_atr` | Same for 1.618 |
| `fib_{dir}_nearest_pullback_0382_signed_atr` | Same for .382 pullback |
| `fib_{dir}_nearest_pullback_0500_signed_atr` | Same for .500 pullback |
| `fib_{dir}_nearest_pullback_0618_signed_atr` | Same for .618 pullback |
| `fib_{dir}_touched_count_log` | `log1p(number of active structures with a stored 1.272 touch) / log1p(512)` |
| `fib_{dir}_beyond_count_log` | Same count for stored strict 1.272 beyond events |
| `fib_{dir}_rejection_count_log` | Same count for stored rejection-confirmed events |
| `fib_{dir}_youngest_beyond_age_20` | Minimum elapsed completed bars since a stored beyond, `min(age,20)/20`; 0 if the count is zero |
| `fib_{dir}_youngest_rejection_age_20` | Minimum elapsed completed bars since a stored rejection, `min(age,20)/20`; 0 if the count is zero |

The count fields are the missing-state indicators for their associated age
fields. `active_count_log == 0` means no active structure in that directional
half; a zero nearest distance is valid only when its active count is positive.
No target-reached, success, failure, expiry, censoring, or eventual-outcome
field is present.

## 11. Normalization, missing state, and direction

For an active structure and level price `L`, define:

```text
direction_sign = +1 for UpAB; -1 for DownAB
denominator    = max(current_completed_bar_ATR14_raw, canonical_symbol_pip_size)
distance       = direction_sign * (L - current_completed_close) / denominator
```

Positive oriented distance means the level lies in the A-to-B extension
direction from the current close. A pullback can therefore be negative after
an UpAB advance (and symmetrically for DownAB), which is intentional. The
nearest structure for each level minimizes absolute oriented distance; exact
ties are broken by the existing stable A/B identity tuple:
`availability_bar, b_bar, direction, a_timestamp, b_timestamp`.

The numerator uses the just-completed close, as existing same-bar Tensor
features do. The denominator reuses the established raw ATR14 convention and
the repository's canonical pip-size safeguard rather than comparing raw EUR
and JPY price differences. If either close, level, ATR, or pip scale is
non-finite, or the denominator is non-positive, emit
`fib_price_scale_valid=0` and all ten nearest-distance fields as zero. Counts
and causal ages remain available. No clipping is proposed yet: existing
ATR-normalized Tensor features are not clipped, and a read-only diagnostic
must measure the proposed distance distribution before any clipping policy is
introduced.

## 12. Deterministic aggregation and event age

The candidate live set must be the exact deterministic set emitted by the
future H1/H2 structural adapter after it has processed completed bar *t*.
It must honor one explicit configuration's age expiration and capacity policy;
it must not iterate a historical artifact or select structures by future
outcome. Counts are sums over that set. Youngest/oldest A/B ages and youngest
event ages use bar-index differences, not elapsed wall-clock time, because the
frozen endpoint is in completed bars.

Age is descriptive current state, not a hard-coded hazard. The `20` cap for
event ages is a representation cap aligned with the frozen H1/H2 horizon; it
does not alter the endpoint. Ages beyond 20 map to 1, and no empirical
probability is attached to any age value.

## 13. Historical projected width and semantic-layout transition

If, and only if, section 21's gate passes:

- Tensor width: `76 + 27 = 103`;
- model input width: `80 + 27 = 107` (including the unchanged four-return
  suffix);
- new semantic layout: **9**, append-only predecessor **8**;
- new registered model input width: **107**;
- new feature family ablation identity: one all-or-nothing
  `causal_fibonacci_h1_h2_state` group, plus optional per-direction groups
  only after the first family-level ablation is established.

The transition must use the existing input-width expansion provenance and
zero-initialization contract. Existing layout-8 models remain layout-8/width
80; no semantic reinterpretation, overwriting, or implicit inference fallback
is permitted.

## 14. Required tests before any model experiment

1. Synthetic TG1/TG3 causal replay proving no A/B feature before B
   confirmation, and exact availability on its confirmation bar.
2. Synthetic UpAB and DownAB mirror tests for all five oriented distances.
3. Exact frozen touch, strict-beyond, and later-rejection boundary tests,
   including the evaluator's rejection/touch/beyond order.
4. Same-bar target-hit test proving no resolution/success feature exists or is
   emitted.
5. Multiple simultaneous same-direction and opposing-direction structure tests
   proving permutation-invariant aggregates and deterministic identity ties.
6. Age-expiration and capacity tests against the selected adapter's explicit
   live-set policy.
7. Cross-symbol ATR/pip normalization tests, zero/invalid scale tests, and
   non-finite fail-closed output tests.
8. Streaming-versus-prefix replay parity through Tensor rows, including no
   future-bar mutation of an earlier feature row.
9. Layout-8 prefix byte parity, layout-9 width/semantic registry validation,
   persisted expansion compatibility, and zero initialization tests.
10. Feature-ablation mask tests for the entire family and for any approved
    direction subgroup.

## 15. Proposed controlled ablation design

No experiment is authorized by this document. After the gate and tests pass,
the first controlled comparison should be paired and identical in all approved
dimensions except the append-only feature family:

- **CONTROL:** current layout-8 inputs, width 80.
- **FIBONACCI:** CONTROL plus the frozen causal Fibonacci H1/H2 aggregate
  family, layout 9, width 107.

The comparison must not encode empirical outcome rates, optimize the feature
schema against 2025, or change target/model/training semantics. Any future
campaign, queueing, or model operation needs separate authorization.

## 16. Historical open diagnostic and implementation gate

**Implementation is not ready.** A small read-only diagnostic is required
before implementation because the available event artifacts cannot recover the
exact bounded active set, and current production TG4 pulses are UpAB-only
while the confirmed H1/H2 state is symmetric.

The diagnostic must replay completed canonical bars without evaluating target
outcomes or touching database state, and report only structural facts for the
selected source configuration:

- per-bar active count distribution and exact maximum;
- same-direction and opposing-direction overlap rates;
- A/B, beyond, and rejection age distributions;
- capacity evictions and age expirations;
- distance-normalization finite/invalid counts and quantiles;
- parity between the proposed adapter's state and the frozen evaluator's
  causal A/B/touch/beyond/rejection events on synthetic fixtures.

Before that diagnostic is specified, it must also answer whether the feature
adapter is a separate frozen symmetric H1/H2 producer or whether a new
production configuration can be made semantically compatible without changing
the existing layout-8 pulse contract. The answer must be versioned, tested,
and append-only. Neither 2025 outcomes nor any post-confirmation performance
comparison may be used to choose the representation.

## 17. Completed structural diagnostic and artifact audit

The historical gate is resolved by the read-only artifact
`causal-fibonacci-model-feature-structural-diagnostic-v1`, recorded at source
baseline `91b87aa4de2695eeeac3629326aa160d53b6da71`. Direct audit of
`manifest.json`, `per_symbol.csv`, `distance_distributions.csv`, and
`summary.md` found the manifest internally consistent: it records frozen
source configuration `tg4a-first-study-preconfirmation-frozen-v1` under
`tg4-analysis-configuration-v1`, read-only/outcome-blind execution, no
warmup replay, requested measurement range
`[2010-01-01T00:00:00Z,2026-01-01T00:00:00Z)`, six symbols
`audcadrmp`, `audusdrmp`, `eurusdrmp`, `gbpusdrmp`, `usdcadrmp`, and
`usdjpyrmp`, and matching per-symbol/combined actual timestamps.

It records `maxABAgeBars=2048`, `maxActiveABStructures=512`, and the explicit
`symmetric_directional_diagnostic_hypothesis`, rather than production's
`SourceUTLUpABOnly` policy. The `combined` CSV row is a sufficient-statistics
merge, not a rendered average: six symbol bar counts
394751/394799/394894/394808/394778/394804 sum to 2,368,834; capacity evictions
497,223, age expirations 106,385, and H1-only/H2-only/both counters
6,059,942/6,099,596/5,834,852 also reconcile additively.

Each symbol, plus combined, has both `geometric` and `event_relevant_20`
distance populations at levels 1.272, 1.618, .382, .500, and .618. Across all
five levels, combined geometric finite/invalid/pip-fallback accounting is
1,189,455,481/0/3,072 and event-relevant accounting is 17,994,390/0/43; each
equals the sum of constituent rows. Metadata specifies exact nearest-rank
count/age quantiles and signed logarithmic distance bins with zero exact and
at-most-0.1-percent representative relative error. Zero invalid counts do not
hide pip fallback use.

This diagnostic establishes structural representation properties, not
profitability, expected return, trading efficacy, or model usefulness. Its
512 saturation describes the actual configured bounded tracker, not the
unbounded theoretical number of structures: capacity evictions show a
constraint, not that 512 is a natural market-state size.

## 18. Audited empirical conclusions

Full geometric state is capacity saturated: combined total p50/p90/p95/p99/max
is 512/512/512/512/512; Up and Down p50 are 254 and 255; multiple and both
directions occur on 99.993499% and 99.992908% of bars. Every symbol has all
five geometric quantiles equal to 512. A/B age p50/p90/p95/p99/max is
981/1768/1869/1980/2048.

Lifetime event state is likewise stale and large: H1 total p50/p90/p95/p99/max
is 444/467/473/481/508, H2 is 449/470/474/482/504, and only 1.146762% of H1
and 1.141716% of H2 state is age 0--20. In contrast, the causal 20-bar union
has total p50/p90/p95/p99/max 6/15/18/27/97, Up/Down p50 3/3, multiple 93.559405%,
and both directions 68.746438%. It is therefore the primary population.

The union contains each structure once if it has a recent H1 or H2 event.
H1-only/H2-only/both-event observed counts are 6,059,942/6,099,596/5,834,852.
For 1.272, geometric absolute p50/p95/p99/max is
12.9063747034/49.1586487738/73.4681465438/309.863665355, while the recent
population is 1.37553205622/4.81730201751/6.80761023897/21.104272671. All
other levels were directly audited. The finite observed recent tails do not
justify clipping, so proposed distances remain un-clipped.

The resolved answers are: use direction-separated permutation-invariant set
aggregation; summarize the narrower recent union rather than full/lifetime
state; replace the historical 27-column proposal; and use a separate symmetric
structural producer while preserving the production TG4 pulse unchanged. The
empirical gate is satisfied, but it did not authorize implementation of the
unchanged 27-column schema.

## 19. Replacement fixed-width schema

The future family has **23 append-only columns**: one shared validity bit and
eleven mirror-image columns for each `up` and `down`. At post-completed bar
*t*, its primary population `U_t` contains each currently retained A/B
structure once when it has H1 beyond or H2 rejection with event age in
`[0,20]`. Age is `t - event_bar`: event bar is age zero, it increases once per
completed bar, age 20 is included, and age 21 is excluded. No unconfirmed
pivot, future bar, target resolution, outcome, profitability, or empirical
success probability enters.

For direction `d`, `U_t,d` is the directional union, `H1_t,d` and `H2_t,d`
are its respective-event subsets, and `B_t,d` their intersection. A both-event
structure counts once in U, but intentionally appears in both H1 and H2
channels and in B; this makes overlap explicit without double-counting the
union. The audited maximum of 97 remains diagnostic evidence only; it is not
a structural bound or a feature-semantic threshold. For exact nonnegative
causal cardinality `n`, define `count_log(n)=log1p(n)`.

| Exact feature name | Population/direction | Definition and normalization | Empty/missing and availability |
|---|---|---|---|
| `fib_recent_price_scale_valid` | shared | 1 iff completed close and ATR/pip denominator are finite and positive; else 0 | 0 when invalid; post *t* |
| `fib_{dir}_recent_union_count_log` | `U_t,d` | `count_log(|U_t,d|)` | 0 empty; post *t* |
| `fib_{dir}_recent_h1_count_log` | `H1_t,d` | `count_log(|H1_t,d|)` | 0 empty; post *t* |
| `fib_{dir}_recent_h2_count_log` | `H2_t,d` | `count_log(|H2_t,d|)` | 0 empty; post *t* |
| `fib_{dir}_recent_h1_h2_both_count_log` | `B_t,d` | `count_log(|B_t,d|)` | 0 empty; post *t* |
| `fib_{dir}_recent_h1_youngest_age_20` | `H1_t,d` | `min(event_age)/20` | 0 empty; H1 may appear at age 0 on *t* |
| `fib_{dir}_recent_h2_youngest_age_20` | `H2_t,d` | `min(event_age)/20` | 0 empty; H2 needs an earlier touch |
| `fib_{dir}_recent_median_1272_signed_atr` | `U_t,d` | exact nearest-rank signed median at 1.272 | 0 empty/invalid scale; post *t* |
| `fib_{dir}_recent_median_1618_signed_atr` | `U_t,d` | same at 1.618 | 0 empty/invalid scale; post *t* |
| `fib_{dir}_recent_median_pullback_0382_signed_atr` | `U_t,d` | same at .382 | 0 empty/invalid scale; post *t* |
| `fib_{dir}_recent_median_pullback_0500_signed_atr` | `U_t,d` | same at .500 | 0 empty/invalid scale; post *t* |
| `fib_{dir}_recent_median_pullback_0618_signed_atr` | `U_t,d` | same at .618 | 0 empty/invalid scale; post *t* |

`{dir}` expands to up/down: 1 + (11 x 2) = 23. Each distance is
`direction_sign*(level-completed_close)/max(completed_bar_wilder_atr14_raw,canonical_symbol_pip_size)`, sign +1 for UpAB and -1 for DownAB. The exact
nearest-rank signed median uses every qualifying directional-union member, is
permutation-invariant, and selects no winner, Top-K, rank, or tie-break. Each
count transform is the raw monotonic `log1p(n)`: zero maps exactly to zero,
there is no clipping or empirical denominator, and future causal populations
larger than the observed maximum retain their count information.

## 20. Future implementation boundary and projection

This is documentation/design freeze only. A separate task must implement and
test a symmetric producer without changing current production TG4 behavior.
Nothing here changes Tensor width 76, model input width 80, layout 8, the
four completed-close-return suffix, persisted semantics, database schema,
scheduler, experiments, or training/inference.

If a future task implements this frozen 23-column schema, the projection is
Tensor width **99** (`76+23`), model input width **103** (`80+23`, retaining
the suffix), and append-only semantic layout **9**. These are projections only.

## 21. Remaining implementation boundary

No design issue blocks a separate implementation task. That task must add
producer-level tests for causality, mirror direction, event ordering/age,
union de-duplication and intentional H1/H2 overlap, scale validity, empty
sets, non-saturating count-log aggregation, exact median aggregation, streaming-prefix parity,
and layout-8 prefix/layout-9 expansion compatibility.

**Ready for separate model-feature implementation task.**
