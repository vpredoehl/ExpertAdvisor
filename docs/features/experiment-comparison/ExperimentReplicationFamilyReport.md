# Experiment Replication Family Report

`--compare-experiment-replication-families` composes two or more existing
generic replication reports without creating a cross-family aggregate.

```text
LSTM_Release --compare-experiment-replication-families='658:659,660:661,662:663,664:665;668:669,670:671,672:673,674:675'
```

Each semicolon-separated family is evaluated through the existing
`ExperimentPairComparison` and `ExperimentReplicationComparison` rules in one
repeatable-read, read-only transaction. Pair order and arm order are retained;
for a control-first invocation, deltas are ablation minus control.

The report emits each family independently, including every existing pair
summary, exact scientific identity and completed-execution provenance,
missing-evidence state, and the existing unweighted per-family descriptive
metrics. It prints a homogeneous symbol only when every arm in that family has
the same persisted symbol.

There is deliberately no cross-family metric, pooled mean, winner, ranking,
recommendation, significance claim, or independence claim. The final
cross-group record says `cross_family_aggregation=not_performed` and
`raw_pair_pooling=false`; callers compare family summaries side by side.

Completed-result compatibility is unchanged. In particular, a TRAIN or final
INFER producing-executable mismatch within a pair makes that pair and its
family incompatible, suppressing that family's aggregate rather than treating
the difference as a planning-time routing detail. Other family reports remain
available, so a bad or incomplete family is visible without being averaged
away or silently erasing independent family evidence.

The command accepts at least two families, each containing at least two pairs;
experiment IDs must be globally unique across the whole invocation.
