# Controlled replication study specification

The controlled-replication study artifact is a versioned, canonical text file
whose `identity_hash` is recomputed from the semantic specification. It freezes
prospective intent without recording future worker-attempt identifiers or
completed execution provenance. The artifact is therefore suitable for being
archived before outcome inspection; completed producer provenance remains the
authoritative evidence used by the comparison engine.

The supported schema (`version=1`) contains a study identifier and type,
feature-ablation intervention, empty control value, treatment value,
`fresh_initialization_seed` replication dimension, ordered contexts, declared
context dimensions, ordered A/B pair membership and requested seeds, required
configured/provenance field contracts, exclusions, freeze timestamp,
`unweighted_mean_of_context_family_means` aggregation,
`statistical_independence=not_inferred`, and `subjective_winner=NONE`.

The canonical identity uses the repository's `TrainingObjective::DeterministicHash`
(`fnv1a64:<16 lowercase hex>`) over deterministic length-framed semantic
serialization. Rendering and parsing are strict: duplicate IDs, pair reuse,
duplicate seeds, malformed versions, undeclared dimensions, reversed/reused
pairs, and identity-hash mismatches fail closed.

Read-only commands are:

```text
LSTM_Release --validate-controlled-replication-study=/path/to/study.txt
LSTM_Release --compare-controlled-replication-study=/path/to/study.txt
```

Validation does not open PostgreSQL. Comparison opens one repeatable-read
transaction and delegates preserved contexts to the existing
`ExperimentReplicationComparison::EvaluateFamilyComparison` path. Contexts
remain separate families; the cross-context result is an unweighted mean of
qualified family means and never a raw-pair pool. Incomplete exact-final
evidence remains pending/non-inferential, with no p-values, confidence
intervals, winners, recommendations, or efficacy claims.

No production TG4 study is registered by this change. An operator must archive
the exact rendered bytes and their reported identity hash before outcome
inspection if a production study is later authorized.
