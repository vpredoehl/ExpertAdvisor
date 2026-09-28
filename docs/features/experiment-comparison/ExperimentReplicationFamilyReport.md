# Experiment Replication Family Report

`--compare-experiment-replication-families` composes two or more existing
generic replication reports and, when each context passes its own strict
contract, emits a descriptive cross-context summary without pooling raw
pairs.

```text
LSTM_Release --compare-experiment-replication-families='660:661,662:663,664:665;670:671,672:673,674:675'
```

Each semicolon-separated family is evaluated through the existing
`ExperimentPairComparison` and `ExperimentReplicationComparison` rules in one
repeatable-read, read-only transaction. Pair order and arm order are retained;
for a control-first invocation, deltas are ablation minus control.

The report emits each family independently, including every existing pair
summary, exact configured identity and completed-execution provenance,
missing-evidence state, and the existing unweighted per-family descriptive
metrics. It prints homogeneous symbol and prediction horizon only when every
arm in that family has the same value.

Version 2 adds two deliberately independent evidence records for every pair:

- `configured_scientific_identity` establishes whether the persisted
  experiment inputs form the requested controlled comparison.
- `execution_provenance` establishes whether the completed TRAIN and final
  INFER producer identities are homogeneous.

Neither record weakens the existing strict pair result. In particular,
configured identity does not claim executable behavioral equivalence. Missing
provenance is reported as `undetermined_due_to_missing_evidence`, not as a
match; mismatched provenance remains `incompatible` with its identity field
and both producing identities visible in the pair summary.

The configured-family view emits all configured per-pair observations as
`EXPERIMENT_REPLICATION_CONFIGURED_PAIR_METRIC`. These are descriptive paired
observations only, always accompanied by the execution-provenance state; they
are never combined into a configured-family aggregate.

`EXPERIMENT_REPLICATION_STRICT_COMPLETED_SUBSET` is the only new aggregate. It
contains only completed pairs that satisfy both the existing strict
completed-result contract and homogeneous configured family identity. It
always reports the total configured pair count, strict compatible completed
pair count, and each excluded or caveated seed/pair/reason. Fewer than two
strict compatible completed pairs yields an unavailable aggregate rather than
a zero or a purported multi-seed result. The strict subset does not silently
drop a caveated pair.

When all requested families are strict completed different-seed replications
with the same control-first feature-ablation intervention, the final
cross-context records preserve family order and report each family mean,
positive/zero/negative family-mean counts, and an explicitly unweighted
descriptive mean of those family means. Symbol and prediction horizon are the
declared context dimensions; every other scientific identity and execution
provenance field must match. Any failed family or mismatch suppresses this
summary while retaining the individual family reports. There is no pooled raw
pair estimate, winner, ranking, recommendation, significance claim, or
independence claim; reports continue to emit
`statistical_independence=not_inferred` and `raw_pair_pooling=false`.

Completed-result compatibility is unchanged. In particular, a TRAIN or final
INFER producing-executable mismatch within a pair makes that pair and the
legacy whole-family strict result incompatible. The new report makes the
different conclusion explicit: the configured family remains visible, while
the strict aggregate is limited to a disclosed homogeneous completed subset
when at least two such pairs exist. Other family reports remain available, so
a bad or incomplete family is visible without being averaged away or silently
erasing independent family evidence.

The command accepts at least two families, each containing at least two pairs;
experiment IDs must be globally unique across the whole invocation.
