# Generic Feature-Ablation Replication Family Summary

`FeatureAblationReplicationFamilySummary` is a pure in-memory C++ core for
summarizing several already-evaluated, matched-seed feature-ablation pairs. It
has no PostgreSQL, CLI, process, scheduler, worker, experiment-resolution, or
experiment-mutation dependency. An eventual adapter may load authoritative
pair evidence and construct `ReplicationMember`; that integration is
intentionally outside this core.

## Family contract

Each member supplies an exact `freshInitializationSeed`, opaque control and
ablation experiment identities, an opaque pair-member identity, an
intervention identity, a scientific-configuration identity, explicit
valid/ready/complete flags, and the authoritative existing
`FeatureAblationPairEvaluation::ComparisonResult`.

The evaluator fails closed when the family is empty; an identity is absent;
the arms are identical; a seed is repeated; an experiment identity, pair-member
identity, or ordered control/ablation identity pair is repeated; intervention identity differs;
scientific configuration identity differs; or a member is invalid or
not ready or incomplete. Identity comparisons are exact byte comparisons. Member output is
ordered by ascending seed, so input permutation does not affect output.

The family core deliberately does not treat experiment IDs as scientific
identity. The opaque identity fields must be supplied by an authoritative
adapter later; this core cannot resolve them.

For each pair metric, availability must be all-or-none across included
members. A metric absent from every member is represented by an absent
descriptive summary. A partially available metric is rejected; it can never
silently change the denominator. Any available delta must be finite.

## Metrics and summaries

The core uses the actual pre-existing pair-delta metric set, retaining a
category on every result:

- Model performance: `leader_score`, `inference_accuracy`, `accept_accuracy`.
- Behavioral/coverage: prediction count; predicted down/neutral/up counts;
  accepted prediction count; accept rate; and down/neutral/up proportions.
- Profitability: actionable count, aggregate terminal-horizon log return, and
  average terminal-horizon log return per actionable prediction.

All values are existing control-minus-ablation deltas. For every available
metric, the descriptive result reports member count, positive/zero/negative
counts (exact comparisons to `0.0`), mean, ordinary sorted median, minimum,
maximum, and population standard deviation:

`sqrt(sum((delta - mean)^2) / N)`.

The sum for the public `double` mean and the squared-deviation accumulation
for the public `double` standard deviation use deterministic `long double`
intermediate arithmetic. The core fails closed if an intermediate or converted
public result is non-finite.

There is no epsilon policy: signs are exact IEEE comparisons, and values must
be finite. A zero-member family is rejected. A one-member metric has standard
deviation zero. The median of an even count is the arithmetic mean of its two
middle sorted values.

For each available metric, the evaluator also produces a leave-one-member-out
entry for each seed. It recomputes the remaining count, mean, median, and sign
counts (and retains the other descriptive fields). A one-member family has one
leave-one-out entry whose remaining summary has `memberCount == 0` and no
numeric optionals.

## Methodological boundary

These are descriptive replication evidence only. Fresh initialization seeds
are not asserted to be statistically independent. The core makes no p-value,
significance, confidence-interval, composite-score, weighted-ranking,
automatic-winner, or arbitrary-success-threshold claim. Metrics are never
collapsed into one score, so disagreement remains visible, including a
favorable mean with an unfavorable median (or the reverse). Leave-one-out is a
sensitivity diagnostic, not statistical validation.

## Isolated test

`Tests/FeatureAblationReplicationFamilySummaryTests.sh` compiles only the
new source and a synthetic in-memory test fixture. It does not link database,
scheduler, worker, or executable code. It verifies permutation invariance,
odd/even medians, sign counts, min/max/mean/population standard deviation,
leave-one-out summaries, identity and readiness failures, metric availability,
and non-finite delta rejection.
