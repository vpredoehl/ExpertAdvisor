# Consensus feature-ablation pair evaluation

The read-only command is:

```text
--compare-feature-ablation-pair=CONTROL_ID:TREATMENT_ID
```

The argument order is authoritative. The control must have exactly the four
active consensus channels ablated, and the treatment must have an empty
feature-ablation mask. Reversed order is rejected rather than silently
normalized.

## Scientific identity

The only allowed difference is:

```text
control feature_ablation_mask =
  relevant_event_has_consensus,
  relevant_event_consensus_low,
  relevant_event_consensus_high,
  relevant_event_consensus_is_range

treatment feature_ablation_mask = empty
```

The existing `FeatureAblationMask` parser and canonical registry order are the
authority. No group alias or second persisted representation is introduced.
The evaluator emits the canonical feature set and a tagged FNV-1a-64 identity
hash.

The following experiment-level fields must match exactly:

- symbol, prediction horizon, threshold, core/head learning-rate multipliers;
- target epochs, checkpoint interval, train and inference date ranges;
- feature warmup scope, Donchian mode, and Donchian lookback;
- resume model identity and input-width-expansion flag;
- complete training-objective canonical/hash plus objective and loss versions;
- auxiliary-loss mode/coefficient, regression target/normalization, robust-loss
  definition/delta, target clipping, and objective normalization;
- checkpoint inference enablement/minimum epoch/interval;
- checkpoint policy enablement, thresholds, top-N, scope, stop mode, grace
  evaluations, semantic revision, and policy hash;
- git commit/branch/dirty state, build configuration, compiler, database schema,
  scheduler, and binary provenance.

After both workflows complete, these final-model fields must also match:

- model input width, hidden size, layer count, window size;
- model, train-configuration, optimizer, and input-semantic metadata versions;
- semantic input layout version and input-width expansion provenance;
- normalization, class weights, label rule, target metadata;
- optimizer type/update count/buffer counts and persisted learning-rate
  multipliers;
- persisted training symbol and range.

Pending/running pairs can therefore be proven comparable at the experiment
level, but remain `comparable_incomplete` until immutable final-model input
contract provenance exists.

## Evidence selection and output

The evaluator reuses the existing paired-evidence repository only as shared
infrastructure. It does not invoke the training-objective comparison policy.
The exact FINAL inference resolver requires the final model, final scope, no
checkpoint identity, no parent experiment, the persisted symbol/horizon/
threshold/window/label/target/range, completed target epochs, and completed
status. Ambiguous or context-mismatched evidence fails closed.

The matching completed final analysis supplies inference accuracy, accept
accuracy, accept rate, neutral prediction proportion, and leader score. The
authoritative profitability selector is bound to that exact experiment, model,
FINAL inference-result ID, final scope, inference range, and current metric
definition. It never falls back to a checkpoint or a different inference row.

Per arm, output includes the selected identities and:

- inference accuracy, accept accuracy, accept rate, neutral proportion, leader
  score, and prediction count;
- profitability actionable count, aggregate terminal-horizon log-return sum,
  and average terminal-horizon log return per actionable prediction.

For a complete pair, every numeric metric also has a treatment-minus-control
delta. No single metric is interpreted as scientific success or failure.

Dispositions and CLI exits are:

- `comparable_complete`: exit 0;
- `comparable_incomplete`, `missing_final_inference`, or
  `profitability_evidence_unavailable`: exit 4;
- `incompatible_configuration`, `ambiguous_final_inference`, or
  `invalid_ablation_pair`: exit 3;
- argument error: established CLI exit 1;
- PostgreSQL/tool failure: established top-level exit 2.

The service uses one repeatable-read `pqxx::read_transaction` and exposes no
write or persistence operation.

## Event-conditioned efficacy follow-up

Event-conditioned analysis is intentionally not implemented in this phase.
Persisted `inference_eval_result` and `experiment_analysis_result` rows contain
aggregate metrics, and `inference_profitability_observation` contains aggregate
return statistics. They do not retain one row per inference prediction with
its causal bar/release timestamp, predicted class/actionability, label/outcome,
and economic-event consensus-availability state. Consequently, near/away and
consensus-present/missing partitions cannot be reconstructed deterministically
from authoritative persistence.

The smallest follow-up is an immutable prediction-level FINAL inference
evidence table (or content-addressed artifact) keyed by inference-result ID and
ordered prediction timestamp, containing predicted class/actionability,
authoritative label/terminal outcome, and the causal economic-event context
actually available at that timestamp. A later read-only partitioner can join
that evidence to authoritative release timestamps under an explicitly
versioned window policy, deduplicate same-time events deterministically, and
retain the existing `WEEKLY_CLAIMS` semantics.

Reserved surprise columns 63-66 remain disabled and play no role in either the
pair identity or the proposed follow-up.
