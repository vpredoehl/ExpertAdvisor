# Controlled feature-ablation pair evaluation

The generic read-only command is:

```text
--compare-feature-ablation-pair=CONTROL_ID:ABLATION_ID \
--expected-ablation-mask=FEATURE[,FEATURE...]
```

The argument roles are authoritative. The control must have an empty
`feature_ablation_mask`; the ablation arm must have exactly the requested mask.
The existing `FeatureAblationMask` parser is the sole canonicalization and
feature-registry authority. It rejects unknown features and an explicitly empty
expected mask. Ordering, whitespace, and duplicates follow that parser's normal
canonicalization rules. Reversed roles are rejected rather than silently
inverting the result.

For compatibility, omitting `--expected-ablation-mask` retains the historic
consensus-family command contract:

```text
--compare-feature-ablation-pair=LEGACY_ABLATED_ID:LEGACY_ENABLED_ID
```

That mode supplies the four-channel economic-consensus mask and adapts the old
argument order to the generic evaluator. Its numeric interpretation is
unchanged: enabled minus ablated. New feature families must use the explicit
expected mask and `CONTROL_ID:ABLATION_ID` order.

## Scientific identity

The generic evaluator allows only the requested feature-ablation mask to differ.
It compares the following experiment/configuration identity exactly:

- symbol, prediction horizon, train and inference ranges, target epochs,
  checkpoint interval, and threshold;
- base learning rate, core learning-rate multiplier, and head learning-rate
  multiplier;
- configured model-input width and semantic-layout version;
- objective canonical text/hash/version, loss-definition version, auxiliary
  objective configuration, target clipping, and objective normalization;
- feature warmup scope, Donchian mode/lookback, checkpoint-inference
  configuration, checkpoint-policy configuration, and the enabled
  continuation-policy scientific identity;
- deterministic fresh-initialization seed, or compatible resume lineage and
  checkpoint epoch;
- source/build provenance used by the established paired-evidence contract.

After both workflows complete, it also compares the final persisted model's:

- input width, semantic layout, hidden size, layer count, and window size;
- model, training-configuration, optimizer, and input-semantic metadata
  versions;
- class weights, label rule, target semantics/normalization, and training
  symbol/range;
- optimizer type/buffer shape and persisted core, head-weight, and head-bias
  learning-rate multipliers;
- input-width expansion provenance.

Final `optimizerUpdateCount` is deliberately not a compatibility field because
the ablation can change gradient finiteness and therefore successful update
count. Scheduler priority, worker PID, scheduler ownership/fencing state, and
queue/admission timestamps are operational metadata and are not scientific
compatibility fields.

Configured model-input width/layout are available before completion. When a
final model exists, the evaluator cross-checks its persisted width/layout
against the configured identity. Width and layout are not globally hard-coded:
historical compatible identities remain valid, while mixed identities fail.

The first-release surprise pair uses:

```text
control mask = empty
ablation mask = causal_first_release_surprise_available,causal_first_release_surprise
model_input_width = 77
model_input_semantic_layout_version = 6
```

## Evidence selection and output

The exact FINAL inference resolver is authoritative. It binds the completed
final-scope inference row to the experiment's final model and persisted
symbol/horizon/threshold/window/label/target/range/final-epoch context, with no
checkpoint or parent identity. Missing evidence is not ready; ambiguous or
context-mismatched evidence fails closed. No checkpoint, best/latest checkpoint,
or checkpoint-policy selection can substitute for FINAL inference.

The matching completed final analysis supplies inference accuracy, accept
accuracy/rate, leader score, prediction-class counts, accepted count, and class
proportions. Profitability is selected only by exact experiment, final model,
FINAL inference-result ID, final scope, inference range, and metric definition.
It never falls back to a different model, inference row, scope, or metric.
Missing profitability stays unavailable. For zero actionable predictions, an
authoritative aggregate zero is retained while the per-actionable average stays
`NULL`.

Output identifies each arm's experiment, final model, final epoch, exact final
inference result, final analysis, width/layout, and canonical mask. It reports
available classification and profitability metrics for each arm and uses this
explicit sign convention for every delta:

```text
control_minus_ablation = control - ablation
```

A positive delta means the metric was higher with the requested features
present than with them removed.

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
