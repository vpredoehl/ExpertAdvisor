## Maintenance policy

Update this document when application semantics, lifecycle rules, provenance
requirements, or commonly misunderstood relationships change.

Do not update it merely because a column, index, constraint, or data type changes
unless that change affects how the application interprets or queries the data.

After database migrations, regenerate `LSTM_schema.sql`. If the migration changes
application-level meaning, update this document in the same commit.

# LSTM Database Schema Notes

This document supplements `LSTM_schema.sql`.

`LSTM_schema.sql` is authoritative for physical database structure.
This document is authoritative for application-level semantics and conventions.

Do not duplicate column types, indexes, defaults, or constraints here unless
they are necessary to explain application behavior.

## experiment

Primary experiment lifecycle and configuration record.

### Important semantics

- `experiment_id`
  Primary experiment identifier. Use this instead of assuming a generic `id`.

- `prediction_horizon`
  Prediction horizon. Do not use `horizon`; that column does not exist.

- `current_epoch`
  Training progress for the experiment. The `model` table does not contain
  an `epoch` column.

- `last_model_id`
  Most recently associated persisted model/checkpoint.

- `feature_ablation_mask`
  Features disabled for the experiment. Empty value represents the control
  configuration.

- `model_input_width`
  Number of model input features.

- `model_input_semantic_layout_version`
  Identifies the semantic interpretation/order of the model input vector.
  Models with different semantic layouts must not be assumed compatible
  merely because their input widths match.

- `feature_warmup_scope`
  Defines the historical data scope used to warm feature calculations.

- `economic_calendar_snapshot_id`
  Identifies the economic-calendar snapshot used by the experiment.

### Scheduler semantics

- `scheduler_priority`
  Values: `high`, `normal`, `low`.

- `resume_requested`
  Indicates that a paused experiment has explicitly been requested to resume.

- `scheduler_resume_origin`
  Records why the experiment became eligible for resumption, such as
  `operator` or `preemption`.

### Relationships

- `last_model_id` → `model.model_id`
- `parent_experiment_id` → another experiment
- `continuation_source_experiment_id` → experiment used as continuation source
- `continuation_source_model_id` → model used as continuation source

## model

Persisted trained model/checkpoint.

### Important semantics

- `model_id`
  Primary model identifier.

- `experiment_id`
  Experiment that produced the model.

- `parent_model_id`
  Previous/parent model when applicable.

### Important warning

`model` does NOT contain an `epoch` column.

For experiment training progress, use:

    experiment.current_epoch

Do not invent `model.epoch`.

## inference_eval_result

[Document application semantics here.]

## experiment_analysis_result

[Document application semantics here.]

## inference_profitability_observation

[Document application semantics here.]

## Scheduler / worker tables

[Document lifecycle and relationships here.]

## Economic calendar tables

[Document snapshot/provenance semantics here.]

## Common SQL traps

1. Use `experiment.experiment_id`, not `experiment.id`.
2. Use `experiment.prediction_horizon`, not `experiment.horizon`.
3. Do not query `model.epoch`.
4. Do not assume equal `model_input_width` means semantic compatibility.
5. Join `model` to `experiment` when experiment configuration is required.

