# Price-Level Phase 4R — model-exposure contract recovery and freeze

## Status, scope, and provenance rule

This is the single implementation contract for the future model-facing raw
Price-Level producer.  It is a design-only Phase 4R artifact: it does not add
a Tensor column, semantic layout, detector accessor, experiment, or ablation
implementation.

The original Phase 4 output is unavailable.  Accordingly, this document uses
the following labels without treating a later resolution as recovered history:

| Label | Meaning in this document |
| --- | --- |
| `RECOVERED` | Explicitly supported by surviving historical Phase 4 material. |
| `PREEXISTING` | A committed Phase 1--3 contract. |
| `FROZEN_V2` | A committed production `causal-price-level/v2` contract. |
| `RESOLVED_4R` | A missing semantic detail resolved here from deterministic, causal, bounded design constraints. |

The ignored `LSTM_PriceLevelStructure_Phase4_ModelExposureDesignGate_Prompt.txt`
survives as intent only: it requires a smallest bounded, causal, scale-robust
raw representation; asks whether raw and/or derived confluence should be
exposed; requires independent future ablation; and explicitly rejects target
or outcome-derived semantics.  It does **not** preserve the former scalar
definitions.  The surviving local Phase 7 integration prompt records the
intended raw 11-channel boundary and its order, but defers the exact meanings
to missing Phase 4 documentation.  It is corroborating project evidence, not
the missing authoritative Phase 4 output.  Therefore it cannot elevate the
raw-only choice or channel order to `RECOVERED` under this document's strict
taxonomy; Phase 4R freezes them as `RESOLVED_4R`, consistently with the
current approved scope.  All scalar encoding, selection, default, and
normalization details are likewise `RESOLVED_4R` unless identified otherwise.

## Evidence and frozen boundary

The committed evidence inspected for this recovery is:

- Phase 1, `docs/features/causal-price-level-engine-phase1.md` and
  `Headers/CausalPriceLevelEngine.hpp`: completed-bar causality, inclusive
  zones, roles, interactions, canonical identity/order, bounded state, and
  deterministic replay (`PREEXISTING`).
- Phase 2, `docs/features/price-level-market-structure-observation-bridge-phase2.md`
  and `Headers/PriceLevelMarketStructureObservationAdapter.hpp`: one-way,
  neutral, non-Tensor observation copying (`PREEXISTING`).
- Phase 3B, `docs/features/price-level-keyed-confluence-phase3b.md` and
  `Headers/PriceLevelKeyedConfluenceDefinitions.hpp`: the separate keyed
  relationship `price-level-reinforced-then-retest-cooccurrence/v1`
  (`PREEXISTING`).
- Phase 5 characterization and bounds documents, especially
  `Phase5B_AgeTimingContractStudy.md` and `Phase5C_BoundsConfirmationStudy.md`:
  the non-outcome-based justification for the production parameters
  (`PREEXISTING`).
- Phase 6, `docs/phases/PriceLevel/Phase6_ProductionCausalPriceLevelV2.md`,
  `Headers/CausalPriceLevelV2Engine.hpp`, and
  `Tests/CausalPriceLevelV2EngineTests.cpp`: the production detector and its
  replay/parity tests (`FROZEN_V2`).
- `Headers/FeatureLayout.hpp`, `Headers/MarketStructureRegistry.hpp`,
  `Headers/CausalFibonacciStructuralFeatures.hpp`,
  `Headers/CausalPocketFeatures.hpp`, and `Headers/FeatureAblation.hpp`:
  append-only layout, finite zero initialization with explicit availability
  bits where scale-dependent values need a gate, and concrete persisted masks
  for future atomic studies (`PREEXISTING`).

`causal-price-level/v2` is consumed exactly as committed:

```text
pivot radius                    3
scale                           median of preceding completed high-low ranges
scale lookback / multiplier     64 / 1.0
scale timing                    pivot_time
maximum age                     512 completed bars
maximum active levels           48
maximum retained pivot evidence 32
completed-bar duration          900 seconds
```

In particular, pivot-time scale excludes the pivot and later bars, startup
uses the predecessor prefix, width is frozen, merge is existing-zone-only and
nearest-anchor then identity, and expiry happens before interactions at
`available_bar + 513`.  Nothing below changes those facts.

## Recovered architectural outcome

The frozen model-facing outcome is **RAW ONLY** (`RESOLVED_4R`): exactly eleven
channels, in the order below, from raw v2 state/events.  No Phase 3B derived
confluence channel is included.

Phase 3B remains independently available as the descriptive, keyed-neutral
relationship `price-level-reinforced-then-retest-cooccurrence/v1`.  It says
that an earlier reinforcement and a later retest share an exact level identity;
it is not a raw detector scalar.  Adding it here would violate RAW ONLY and
would couple a generic relationship definition to the raw group.  A later
study may expose or ablate it independently, without changing this contract.

The group-member canonical semantic names are exactly:

```text
0 available
1 zone_scale_valid
2 zone_gap_signed_clipped
3 zone_relation
4 current_role
5 age_fraction
6 prior_evidence_saturation
7 touch_now
8 cross_direction_now
9 retest_now
10 role_reversal_now
```

The name/order is `RESOLVED_4R`; the complete meaning of each member is frozen
by this document.

## Row timing and selected-level contract

For a model row associated with completed bar `t`, call the v2 engine once on
that bar and use its returned `Update U` only after `AddCompletedBar` has
completed.  Let `d = U.decisionTime`, `c = close(t)`, and `A = U.activeLevels`.
`c` is the current completed bar's finite close, not an intrabar, future,
anchor, high, low, ATR, pip, or target price.  This post-update timing is
`FROZEN_V2` for v2 events and `RESOLVED_4R` for the model projection.

An eligible level is every member `L` of post-update `A`.  There is no role,
originating-pivot-kind, evidence, touch, or width filter.  Thus a zero-width
but valid v2 level remains selectable; scale validity only gates the division
used by channel 2.  For a level with inclusive bounds `[L.lower, L.upper]`,
define the nonnegative close-to-zone distance:

```text
D(c, L) = lower - c   when c < lower
          0           when lower <= c <= upper
          c - upper   when c > upper
```

The deterministic ranking key is, in ascending lexicographic order:

```text
( D(c,L), abs(c - L.anchorPrice), L.anchorPrice, L.identity )
```

The second term distinguishes multiple zones containing the close; the last
two terms reuse the established anchor/identity ordering.  Floating-point
comparisons are direct comparisons of the finite v2 values; no rounded or
fuzzy equality is permitted.  This ranking is `RESOLVED_4R`; v2's stable
anchor and canonical identity are `FROZEN_V2`/`PREEXISTING`.

Selection is exactly:

1. Build `R` from observations in `U.observations` with `kind == retest`,
   `availableAt == d`, and a `levelIdentity` that is still in post-update `A`.
   Each such observation identifies one candidate level.  If `R` is nonempty,
   select its level using the ranking key above.
2. Otherwise, if `A` is nonempty, select its minimum-ranked level.
3. Otherwise there is no selected level.

The event and fallback paths therefore use the identical ranking and identity
ordering.  High- and low-pivot levels receive no extra preference.  A current
retest that belongs to a level evicted later in the same update is not
selectable because it is absent from `A`; fallback then applies.  A level is
selectable on its last active bar (`availableBar + 512`) and can never be
selected at `availableBar + 513`, because v2 expires it before interactions
and before the returned active snapshot.  This is a representation decision
over the immutable v2 update order, not a detector reinterpretation.

## Single unavailable-state contract

Every output is a finite `float`/scalar within the range in the next table.
The default vector is eleven zeroes.  It is emitted only when no level is
selected, with `available = 0`; all other zeros then mean unavailable/default,
not an economic observation.

When `available = 1`, all selected-level fields are meaningful according to
their individual definitions.  In particular:

- `zone_scale_valid = 0` with `available = 1` means the selected frozen
  half-width is zero (the v2-valid nonpositive-denominator case), so channel 2
  is intentionally its zero default.  It does **not** mean no level; channels
  3--10 remain meaningful.  A malformed nonfinite snapshot is rejected before
  any vector is emitted.
- A valid zero gap means that the close is in the inclusive zone, including a
  boundary; channel 3 distinguishes below/inside/above.
- A zero event scalar with `available = 1` means that the selected level has
  no observation of that exact kind on this completed bar.  It is not missing.

The availability bit is deliberately the only no-selected-level sentinel.
This follows the repository's availability-gated finite-vector convention but
is an exact Price-Level encoding decision (`RESOLVED_4R`).

## Compact channel table

`S` below is the selected level, if any.  “Now” means an observation in `U`
for `S.identity` with `availableAt == d`.

| Index | Canonical name | Scalar type / range | Default when no `S` | Primary source | Semantic provenance |
| ---: | --- | --- | --- | --- | --- |
| 0 | `available` | binary `{0,1}` | `0` | post-update `activeLevels` | definition `RESOLVED_4R` |
| 1 | `zone_scale_valid` | binary `{0,1}` | `0` | `S.zoneHalfWidth` | definition `RESOLVED_4R`; width `FROZEN_V2` |
| 2 | `zone_gap_signed_clipped` | bounded continuous `[-1,1]` | `0` | `S.lower`, `S.upper`, `S.zoneHalfWidth`, `close(t)` | `RESOLVED_4R` |
| 3 | `zone_relation` | categorical scalar `{-1,0,1}` | `0` | `S.lower`, `S.upper`, `close(t)` | `RESOLVED_4R` |
| 4 | `current_role` | categorical scalar `{-1,0,1}` | `0` | post-update `S.currentRole` | encoding `RESOLVED_4R`; role `FROZEN_V2` |
| 5 | `age_fraction` | bounded continuous `[0,1]` | `0` | current bar index, `S.availableBar` | encoding `RESOLVED_4R`; lifecycle `FROZEN_V2` |
| 6 | `prior_evidence_saturation` | bounded continuous `[0,1]` | `0` | retained evidence plus same-update establish/reinforce observations | encoding `RESOLVED_4R`; cap/events `FROZEN_V2` |
| 7 | `touch_now` | binary `{0,1}` | `0` | `touch` observation | encoding `RESOLVED_4R`; event `FROZEN_V2` |
| 8 | `cross_direction_now` | categorical scalar `{-1,0,1}` | `0` | `cross_up`/`cross_down` observation | encoding `RESOLVED_4R`; event `FROZEN_V2` |
| 9 | `retest_now` | binary `{0,1}` | `0` | `retest` observation | encoding `RESOLVED_4R`; event `FROZEN_V2` |
| 10 | `role_reversal_now` | binary `{0,1}` | `0` | `role_reversal` observation | encoding `RESOLVED_4R`; event `FROZEN_V2` |

For every row, the member name/index is `RESOLVED_4R`.  The table's
“semantic provenance” is the classification of the scalar definition; its
underlying source state is separately and explicitly classified so that no
Phase 4R representation choice is misdescribed as detector history.

## Exact channel definitions and nonredundancy

### 0. `available`

Emit `1` iff selection produced `S`; otherwise emit `0`.  It is available at
`d`, after the current v2 update.  It requires no selected level to emit its
zero.  It is not redundant: it disambiguates all other default zeroes from a
real inside-zone, neutral-role, age-zero, no-event row.

### 1. `zone_scale_valid`

If `S` exists, emit `1` iff `S.zoneHalfWidth` is finite and strictly greater
than zero; otherwise emit `0`.  Valid v2 input makes nonfinite widths
unreachable.  A producer receiving a finite, otherwise valid snapshot with a
nonpositive width emits this zero; a malformed snapshot with nonfinite
close/bounds/width is rejected before feature emission rather than being used
to manufacture a nonfinite scalar.  A zero half-width is legal v2 state (for
example, a zero predecessor median) but cannot be a denominator.
No v2 level is established without at least one predecessor range; that
startup rule is not itself the validity bit.  This channel is available at
`d`, requires `S`, and has no denominator or clipping.  It is not redundant:
it separates a selectable zero-width zone from a normal-width zone whose gap
happens to be zero.

### 2. `zone_gap_signed_clipped`

This channel requires `S` and `zone_scale_valid = 1`.  With `w =
S.zoneHalfWidth`, emit:

```text
raw_gap = (c - S.lower) / w     when c < S.lower
          0                     when S.lower <= c <= S.upper
          (c - S.upper) / w     when c > S.upper

zone_gap_signed_clipped = max(-1, min(1, raw_gap))
```

Below-zone values are negative and above-zone values positive.  Zero means
inside the inclusive zone, not “at anchor”; both exact zone boundaries are
inside and therefore zero.  The denominator is exactly v2's immutable frozen
half-width, never ATR, pip size, Fibonacci scale, a learned scale, or a
current/future range.  The clipping threshold is one frozen half-width from
the nearest boundary: it is the smallest interpretable unit already supplied
by the zone itself, rather than a conventional ML constant.  If the width is
zero or invalid, emit `0` and rely on channel 1 to distinguish it from a valid
inside-zone zero.  It is available at `d`.  It is not redundant: channel 3
gives only a three-way region; this gives bounded exterior displacement and
sign without leaking raw price.

### 3. `zone_relation`

This channel requires `S` and emits `-1` when `c < S.lower`, `0` when
`S.lower <= c <= S.upper`, and `+1` when `c > S.upper`.  It remains defined
for a zero-width level: equality with the anchor is inside.  It has no
denominator/clipping and is available at `d`.  It is not redundant: it
distinguishes valid in-zone zero gap from below/above direction and makes the
zero convention for channel 2 explicit.

### 4. `current_role`

This channel requires `S` and emits `-1` for
`Role::resistance_like` and `+1` for `Role::support_like`; `0` is only the
unavailable default.  It reads the post-update role, so a same-bar cross uses
v2's already-recorded `roleAfter`.  The labels are detector state labels,
never hindsight support/resistance claims.  It has no denominator/clipping and
is available at `d`.  It is not redundant: location relative to a zone does
not encode the level's pivot/cross-derived state.

### 5. `age_fraction`

This channel requires `S`.  Let `i` be the zero-based current completed-bar
index used by v2 for this update, and `a = S.availableBar`.  Emit:

```text
min(1, max(0, (i - a) / 512))
```

The denominator is the frozen `maxAgeBars = 512`, with floating-point
division.  Age is zero on the availability/confirmation bar.  It is one on
the final selectable bar `a + 512`; the next bar has no selected expired
level and therefore emits the unavailable default.  It has no further
clipping beyond the stated clamp and is available at `d`.  It is not
redundant: evidence amount and current event flags do not state elapsed active
lifetime.

### 6. `prior_evidence_saturation`

This channel requires `S`.  It represents retained pivot evidence that was
already in `S` immediately **before** this current v2 update, not an
unbounded pivot count and not current-bar reinforcement information.

Let `n_post = S.retainedPivotEvidence.size()`.  Let `n_added_now` be the
number of observations in `U.observations` for `S.identity`, available at `d`,
that are either:

- `level_established`; or
- `level_reinforced` with `retainedEvidenceAlreadySaturated == false`.

The producer must verify `0 <= n_added_now <= n_post` and set
`n_prior = n_post - n_added_now`.  Emit `n_prior / 32.0`, clamped to `[0,1]`
only as a defensive assertion of the frozen retained-evidence bound.  A
saturated reinforcement adds zero because v2 did not retain another evidence
ID.  Thus a level at the retained-evidence cap before the update emits `1`,
and a newly established level emits `0` even though it now retains its origin
ID.  This exact subtraction also handles multiple current-update confirming
pivots for the same level.

The channel is available at `d`, has denominator `maxRetainedPivotEvidence =
32`, and has the stated saturation clamp.  It is not redundant: the current
active role, age, zone geometry, and four immediate events do not convey the
bounded prior reinforcement history.  This is deliberately the retained
evidence count rather than `pivotObservationCount`, which is not bounded by
32.

### 7. `touch_now`

This channel requires `S`.  Emit `1` iff `U.observations` contains a `touch`
for `S.identity` available at `d`; otherwise emit `0`.  It uses v2's
edge-triggered inclusive range-overlap event, not a newly invented close test.
It has no denominator/clipping and is available at `d`.  It is not redundant:
being located inside the zone at close does not say that this bar created a
touch transition.

### 8. `cross_direction_now`

This channel requires `S`.  Emit `+1` iff a same-update `cross_up` exists for
`S.identity`; emit `-1` iff a same-update `cross_down` exists; otherwise emit
`0`.  A valid v2 level cannot emit both cross directions in one update.  The
sign is therefore detector direction, not a target, return sign, or model
recommendation.  It has no denominator/clipping and is available at `d`.  It
is not redundant: current role records state after earlier or current crosses;
this scalar records an actual directional transition now.

### 9. `retest_now`

This channel requires `S`.  Emit `1` iff a same-update `retest` exists for
`S.identity`; otherwise emit `0`.  It is available at `d`, has no denominator
or clipping, and may coexist with `touch_now`.  It is not redundant: v2 emits
retest only after its pending-cross state and a later inclusive touch; neither
zone relation nor touch alone encodes that causal sequence.  It also drives
the selection preference, but that preference does not make its observed
value redundant.

### 10. `role_reversal_now`

This channel requires `S`.  Emit `1` iff a same-update `role_reversal` exists
for `S.identity`; otherwise emit `0`.  It is available at `d` and has no
denominator/clipping.  It may coexist with `cross_direction_now`; v2 emits it
only when that cross changes the role.  It is not redundant: a cross can leave
the same role in place, and `current_role` does not reveal whether it changed
this bar.

## Event coexistence and observation rules

Event channels are tied only to the selected active level.  Observations are
matched by exact `levelIdentity`, never by price proximity or an identity hash.
V2 observations are canonically sorted, but selection never depends on their
incidental vector position.  The stated `availableAt == d` test is the exact
current-bar event timing rule.

No one-hot suppression is applied.  In particular, a touch can coexist with a
cross or retest when v2 emits both; role reversal can coexist with its cross;
and a same-bar reinforcement/retest remains raw v2 evidence even though Phase
3B's strictly earlier reinforced-then-retest relationship would not qualify.
The only mutually exclusive event encoding is the signed two-direction cross
scalar, as required by v2's single current close transition for one level.

## Causality, boundedness, and determinism audit

- **No future bars or inference-window lookahead.** The producer uses only
  current completed OHLC close, post-update active state, and observations
  causally available at `d`.  The pivot confirmation delay is represented by
  v2 availability, never backdated.
- **No target or outcomes.** No target, return, profitability, prediction,
  trade, recommendation, or experiment result enters selection or any scalar.
- **No scale contamination or width mutation.** Channel 2 reads the
  per-level frozen v2 half-width.  It does not read current scale, ATR, pip,
  post-pivot ranges, or any mutable/recentred zone.
- **No retrospective role assignment.** Channel 4 and channel 10 read only
  the role stored/emitted by v2 at this update.
- **Deterministic replay.** Given identical completed-bar history and v2
  configuration, v2 produces identical active levels/observations.  Fixed
  post-update selection and exact identity matching produce the same vector.

| Channel | Bounded range | Why finite |
| --- | --- | --- |
| `available`, `zone_scale_valid`, `touch_now`, `retest_now`, `role_reversal_now` | `{0,1}` | Boolean tests only. |
| `zone_gap_signed_clipped` | `[-1,1]` | Strictly positive finite width is required before division; result is clipped. |
| `zone_relation`, `current_role`, `cross_direction_now` | `{-1,0,1}` | Closed categorical mapping. |
| `age_fraction`, `prior_evidence_saturation` | `[0,1]` | Frozen positive denominators and explicit clamps. |

The producer must fail closed on a malformed supposedly-v2 snapshot (nonfinite
close/bounds/width, invalid ordering, evidence accounting invariant failure,
or an impossible duplicate contradictory event) rather than emit an unbounded
or invented value.  Such validation is observational and would not alter v2
semantics.

## Future implementation and atomic ablation contract

The intended future experimental unit is the one atomic group
`price_level_structure` (`RESOLVED_4R`, chosen to follow the registry's
family/group convention).  Its future registry members must preserve the
eleven canonical suffixes and order above.  Its future persisted ablation
identity must expand to the complete concrete eleven-member list; a wildcard
or group alias may be accepted for request convenience only if it resolves and
persists that concrete immutable list, as existing feature-ablation behavior
does.  Phase 7 must not create eleven scientific arms.

The first controlled experiment is exactly:

```text
CONTROL:   same new append-only semantic layout, price_level_structure fully ablated
TREATMENT: same new append-only semantic layout, price_level_structure active
```

All other model semantics, data window/warmup, targets, objective,
configuration, and persisted provenance must be identical.  The group may be
computed in both arms when needed for deterministic construction; ablation
must use the repository's established input-neutralization mechanism.  This is
a future Phase 7 obligation, not authorization to implement it here.

`Update` already publicly exposes the required post-update `activeLevels`,
their anchor/bounds/role/available-bar/retained evidence, and observations
with identity, kind, times, saturation flag, and level payload.  The feature
producer also needs the current completed close and current zero-based bar
index, which are already inputs/coordinates of normal bar processing.  No v2
semantic change or new detector accessor is required.  A future producer may
add a pure observational adapter/helper, but must not alter detector state or
use the Phase 2 bridge/Phase 3B as a substitute source.

## Provenance summary and exclusions

| Decision | Classification |
| --- | --- |
| Minimum bounded/causal raw-exposure intent, independent future ablation, and rejection of target/outcome-derived semantics | `RECOVERED` |
| Raw-only 11-member boundary and member names/order | `RESOLVED_4R` (corroborated but not historically recoverable from the lost Phase 4 output) |
| Completed-bar causality, inclusive zones, canonical identities/orders, event vocabulary, neutral Phase 2 bridge, and separate Phase 3B relationship | `PREEXISTING` |
| v2 parameters, predecessor-prefix pivot-time scale, immutable width, merge/lifecycle/capacity/evidence behavior, post-cross roles, and v2 event timing | `FROZEN_V2` |
| Post-update selected-level policy; close/zone distance; tie ordering; categorical encodings; all defaults; width-validity gate; unit-width clipping; age and prior-evidence formulas; event projection; and `price_level_structure` atomic group spelling | `RESOLVED_4R` |

Explicit exclusions are absolute anchor/width values, pivot/evidence IDs,
timestamps, identity hashes, ATR/pips, Fibonacci quantities, raw targets or
returns, profitability/outcome measures, hindsight labels, and Phase 3B's
derived confluence scalar.  They are unnecessary for the selected bounded raw
representation or violate its causal/scientifically neutral boundary.
