# Price-level keyed descriptive confluence - Phase 3B

## Scope

This is a descriptive-only use of the generic keyed-neutral contract. It
neither changes `causal-price-level/v1` nor projects Price-Level information to
Tensor, Layout 11, model input, workers, database state, or trading actions.
Raw bridged observations remain independently available and unmodified.

## Exact key contract

The v2 Price-Level observation bridge supplies the immutable typed key:

```text
type  = price_level.level_identity.v1
value = exact causal-price-level/v1 Observation::levelIdentity
```

The value is opaque to the generic engine. Equality requires both the type and
the complete value to be identical. Missing or malformed keys fail closed.
The bridge version is v2 because this metadata changes generic observation
identity from `observation-v1` to `observation-v2`; the detector, its source
observation identity, timestamps, provenance copied from the detector, event
roles, neutral polarity, absent confidence, and descriptor schema are not
changed.

## Frozen definition

`PriceLevelBridge::FrozenKeyedConfluenceDefinitions()` currently returns one
definition:

| Field | Frozen value |
| --- | --- |
| ID/version | `price-level-reinforced-then-retest-cooccurrence` / `v1` |
| Left selector | `price_level` / `level_reinforced` |
| Right selector | `price_level` / `retest` |
| Relation | `co_occurrence` |
| Key constraint | exact typed key equality |
| Temporal predicate | `left_available_at_before_right_available_at` |
| Candidate cap | 1 per side inside each qualifying key group |
| Correlation-key cap | 64 qualifying canonical keys per evaluation |

It emits one result only when a reinforced observation for one exact frozen
level became available strictly before a retest observation for that same
level. Equal or reversed availability does not qualify. The result says only
that those two observations co-occur under that exact key and ordering; it
does not state causation, support, resistance, continuation, reversal,
bullishness, bearishness, confidence, strength, success, profitability, or a
trading recommendation.

The strict predicate is required because the definition names a prior
reinforcement. A same-bar reinforcement and retest can both be genuine raw
observations, but are not described as `reinforced then retest`; both raw
records remain available.

## Candidate audit

- `level_established -> touch` and `level_established -> retest` are rejected:
  a same-level later event already presupposes establishment, so the pair adds
  no independent descriptive fact.
- `level_reinforced -> retest` is accepted: retest does not encode whether its
  frozen level had earlier independent pivot reinforcement. The relationship
  exposes that bounded historical fact only.
- `cross_up/down -> retest` is rejected: the detector's retest state is set by
  a preceding cross, but generic key grouping cannot prove the selected cross
  is the immediately preceding/latest one. Pairing would be redundant when
  correct and potentially misleading otherwise.
- `cross_up/down -> role_reversal` is rejected: role reversal is emitted with
  the same cross when its role transition occurs, so a second output merely
  renames source evidence. Strict ordering would also reject their equal
  availability.
- `level_reinforced -> touch` and other broad later-interaction variants are
  rejected: the detector gives no special causal linkage beyond shared level
  membership, and a broad history label would be arbitrary.
- repeated same-level observations are rejected as an open-ended category,
  not a precise detector relationship.

## Causality and bounds

The generic engine first filters `availableAt <= decisionTime`, then groups
family/role candidates by canonical typed key and qualifies both sides before
the 64-key cap. Thus unrelated incomplete/non-neutral keys cannot consume the
cap before a valid same-level group is considered. The strict predicate chooses
one deterministic available pair per retained key, with canonical identity
breaking ties. It creates no Cartesian product. Future records, even malformed
ones, are filtered before validation and cannot change an earlier prefix.

## Domain context

`ResearchSources/Pockets/Pockets.pdf` uses discretionary support/resistance,
break, re-test, and continuation terminology. It does not specify this
detector, its immutable identity, exact typed joining, the selected sequence,
or a neutral machine relationship. It is context only; in particular its
continuation framing was not imported into this definition.
