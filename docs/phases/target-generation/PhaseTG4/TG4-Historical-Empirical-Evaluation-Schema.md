# TG4 machine-readable evidence schema

## Artifact set

Every completed run writes the following files atomically into a previously
unused output directory:

| File | Schema | Purpose |
|---|---|---|
| `observations.csv` | `tg4-inner-break-observation-v1` | One terminal row per causally observed Inner break. |
| `cohorts.csv` | `tg4-aggregate-v1` | Deterministic event counts and outcome denominators/rates by cohort. |
| `comparisons.csv` | `tg4-aggregate-v1` | Comparable 2x2 outcome counts and descriptive rate differences. |
| `angle_distributions.csv` | `tg4-aggregate-v1` | Frozen Inner/Outer angle N, min, max, mean, and sample SD. |
| `equal_symbol_rates.csv` | `tg4-aggregate-v1` | Equal-symbol summaries, distinct from event-weighted pooling. |
| `data_quality.csv` | `tg4-data-quality-v1` | Coverage, duplicate/order, gap, partition, and exclusion audit. |
| `data_gaps.csv` | `tg4-data-gap-v1` | One row per material timestamp gap; no filled bars. |
| `metadata.json` | `tg4-study-metadata-v1` | Baseline, study identity, range, warmup policy, effective configuration, and configuration fingerprint. |
| `report.md` | human-readable | Descriptive report with tiny-cell suppression. |

Temporary `.tmp` files are renamed only after successful close. Existing
final or temporary output names are never overwritten.

## Observation identity and ordering

`event_identity` is an FNV-1a-64 digest over the schema version, symbol,
timeframe, deterministic TG2 candidate identity, break timestamp, and TG2
event sequence. It is an identity checksum, not a security primitive.

Rows are ordered by canonical symbol and then increasing TG2 event sequence.
Candidate identity includes direction; anchor 1, anchor 2, and creation bars;
and their timestamps. The explicit schema version permits later additive or
breaking evolution without silently reinterpreting old evidence.

## Observation column groups

`observations.csv` begins with identity and causal grouping columns:

- `schema_version`, `event_identity`, `symbol`, `timeframe`,
  `temporal_partition`, and `event_sequence`;
- deterministic Inner and optional Outer candidate identities;
- direction and frozen Inner/Outer class and angle fields.

It then contains Inner geometry and break evidence:

- anchor bars, timestamps, prices, and creation facts;
- break bar/timestamp, policy, tolerance, line projection, observed component,
  penetration, directional distance, and OHLC values.

Outcome groups independently record:

- retest policy, tolerance, horizon, state, resolution time/bar, latency,
  projected price, and censor reason;
- pairing policy, eligibility, paired status, reason, and Outer projection;
- Outer-target tolerance/horizon and full outcome;
- retest-conditioned Outer full outcome.

TG3 columns record:

- AB policy, direction, A/B bars, timestamps, prices, confirmations, and AB
  availability;
- configured ratio set and experimental provenance;
- directional/confluence policy and price tolerance;
- confluence state and structural-ineligibility reason;
- matched and nearest ratios, raw and ATR-normalized distance, observation
  ATR, and every level/zone diagnostic.

Empty CSV fields mean not applicable or unavailable. They never mean zero.
`final_record_state` distinguishes resolved, censored, and resolved records
with structural ineligibility.

`metadata.json` serializes the configured TG3 pip count and convention plus
the effective absolute tolerance for every canonical symbol. Observation rows
retain the effective absolute tolerance actually applied. The deterministic
`configuration_fingerprint` covers the reference-bar scale, ratio set,
tolerance convention/count/effective symbol values, and the other effective
configuration fields.

## Aggregate outcome columns

Each of retest, Outer target, and retest-conditioned Outer has explicit:

- structurally eligible;
- structurally ineligible;
- pending;
- censored;
- resolved;
- successes;
- failures;
- denominator;
- rate;
- Wilson 95% lower and upper bounds.

The invariants are:

```text
resolved = successes + failures
denominator = successes + failures
structurally_eligible = pending + censored + resolved
```

A zero denominator produces empty rate and interval fields.

## Causal versus outcome fields

All identity, geometry, class, angle, pairing, AB, level, and confluence fields
are frozen on the completed Inner-break bar. Only the three finite TG2 outcome
groups use later bars. This separation is part of the schema contract, not a
reporting convention.
