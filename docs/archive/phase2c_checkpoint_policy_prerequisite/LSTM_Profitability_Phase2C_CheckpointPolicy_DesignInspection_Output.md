---
title: "LSTM Profitability Phase 2C Checkpoint Policy Design Inspection"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase2C_CheckpointPolicy_DesignInspection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 2C Checkpoint Policy Design Inspection

# Phase 2C inspection report

## 1. Executive conclusion

Recommendation: **B — implement prerequisite checkpoint-policy lifecycle and decision-identity hardening first. Then implement a narrowly bounded profitability-aware checkpoint continue criterion.**

Authoritative checkpoint profitability is substantially closer to lifecycle-safe than initially expected:

- For normal directional checkpoint inference, `inference_eval_result` and `inference_profitability_observation` are written in the same inference-worker transaction and become visible together. [main.cpp:7929](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:7929>)
- Checkpoint analysis occurs only after the scheduler observes that committed inference result. [ExperimentScheduler.cpp:16966](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:16966>)
- Policy evaluation occurs during checkpoint-analysis finalization, after the checkpoint analysis has been linked and marked complete. [ExperimentScheduler.cpp:18707](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:18707>)

The blocker is the checkpoint decision lifecycle, not basic profitability persistence:

- Existing checkpoint rules are **OR-based continue rules**: any configured leader-score, inference-accuracy, or rank rule passing causes `continue`; stopping occurs only when every configured rule fails. [ExperimentScheduler.cpp:11740](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11740>)
- Phase 2B profitability uses an additive **AND eligibility gate** with fail-closed missing evidence. Copying that behavior would silently change checkpoint semantics.
- Checkpoint decisions have no policy revision/hash, analysis identity, inference-result identity, profitability identity, evidence watermark, or ranking-population watermark.
- `PersistCheckpointPolicyDecision()` overwrites the single decision row for a checkpoint evaluation. [ExperimentScheduler.cpp:11818](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11818>)
- Manual reevaluation can overwrite a prior `stop_requested` decision with `continue`, but it does not retract the already-persisted `stop_after_checkpoint_epoch`.
- A stop request directly affects a live training worker, which only checks the request at saved checkpoints. [main.cpp:8074](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:8074>)

The safest initial Phase 2C rule is:

> Exact checkpoint profitability may act as one compound **continue criterion**, alongside the existing OR-based continue criteria. It must not become an additive veto.

Initially, that compound criterion should require:

1. a minimum actionable count; and
2. a minimum average terminal-horizon directional log return per actionable prediction.

Both should be configured together and use AND semantics internally. Aggregate return should be deferred because it is sample-size-dependent and is not capital-normalized portfolio economics.

If profitability is the only remaining criterion that could keep training alive and exact evidence is unavailable, evaluation should be **deferred**, never fail-closed stopped.

One repository-state discrepancy should be resolved before implementation: the worktree is on `lstm-feature-development`, while `AGENTS.md` says the current branch is `phase6`.

---

## 2. Current checkpoint-policy architecture

### Configuration and CLI

Checkpoint policy is configured entirely in `Sources/ExperimentScheduler.cpp` through `SchedulerOptions`:

- `queueCheckpointPolicy`
- `checkpointPolicyMinLeaderScore`
- `checkpointPolicyMinInferAccuracy`
- `checkpointPolicyTopN`
- `checkpointPolicyScope`
- `checkpointPolicyStopMode`
- `checkpointPolicyGraceEvals`
  [ExperimentScheduler.cpp:320](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:320>)

Queue-time options are:

- `--checkpoint-policy`
- `--checkpoint-policy-min-leader-score`
- `--checkpoint-policy-min-infer-accuracy`
- `--checkpoint-policy-top-n`
- `--checkpoint-policy-scope`
- `--checkpoint-policy-stop-mode`
- `--checkpoint-policy-grace-evals`
  [ExperimentScheduler.cpp:3557](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:3557>)

Runtime controls are:

- `--enable-checkpoint-policy=ID`
- `--disable-checkpoint-policy=ID`
- `--set-checkpoint-policy=ID:key=value,...`
- `--evaluate-checkpoint-policy=CHECKPOINT_EVAL_ID`
  [ExperimentScheduler.cpp:23560](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:23560>)

Checkpoint policy requires checkpoint inference at queue time and at enablement. [ExperimentScheduler.cpp:4593](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:4593>)

`CheckpointPolicyConfigurationError()` requires at least one existing continue rule when enabled. [ExperimentScheduler.cpp:1709](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:1709>)

### Persisted configuration

Migration 017 adds these `experiment` columns:

- `checkpoint_policy_enabled`
- `checkpoint_policy_min_leader_score`
- `checkpoint_policy_min_infer_accuracy`
- `checkpoint_policy_top_n`
- `checkpoint_policy_scope`
- `checkpoint_policy_stop_mode`
- `checkpoint_policy_grace_evals`
- `checkpoint_policy_last_decision`
- `checkpoint_policy_last_decision_at`
- `checkpoint_policy_last_checkpoint_eval_id`
- `checkpoint_policy_last_reason`
  [017_checkpoint_policy.sql:1](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/017_checkpoint_policy.sql:1>)

Migration 019 constrains the scopes, stop modes, enabled-rule requirement, and allowed decision values. [019_checkpoint_policy_hardening.sql:35](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/019_checkpoint_policy_hardening.sql:35>)

The live schema was verified read-only and matches these migrations. It contains no checkpoint policy revision, hash, evidence watermark, analysis ID, inference-result ID, or profitability columns.

### Inheritance

Checkpoint policy itself is not inherited.

A repository-wide search shows its configuration columns are written only by queue-time experiment creation and checkpoint-policy control code in `ExperimentScheduler.cpp`.

Continuation child creation can inherit checkpoint **inference enablement and cadence** when the continuation source mode needs child checkpoint evidence, but it does not inherit checkpoint policy. [ExperimentScheduler.cpp:13339](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:13339>)

Thus:

- ordinary new experiment: explicit queue configuration;
- continuation child: may inherit checkpoint inference;
- checkpoint stop policy: defaults disabled unless separately configured.

### Core structs

`CheckpointPolicyConfig` carries configuration plus current lifecycle state:

- enabled/inference-enabled flags;
- thresholds/rank;
- scope/stop mode/grace;
- checkpoint interval and target epochs;
- current epoch and existing stop request;
- experiment status and phase.
  [ExperimentScheduler.cpp:879](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:879>)

`CheckpointPolicyDecision` contains:

- decision/reason;
- rank;
- requested stop epoch;
- passed and failed rule names.
  [ExperimentScheduler.cpp:897](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:897>)

There is no standalone checkpoint-policy service or repository; configuration, SQL, evaluation, persistence, and logging are all embedded in `ExperimentScheduler.cpp`.

---

## 3. Exact lifecycle and call flow

```text
Queue experiment
  SchedulerOptions
  -> ValidateCheckpointPolicyConfig()
  -> InsertExperimentRecord()
  -> experiment.checkpoint_policy_* columns

Training scheduler claims experiment
  -> launches LSTM_Release training worker

Training worker, after every epoch
  -> UpdateSchedulerExperimentProgress()
  -> SavePeriodicCheckpointIfDue()
       -> model row
       -> matrix/model parameters
       -> commit checkpoint model
  -> QueueCheckpointInferenceIfEligible()
       -> experiment_checkpoint_eval(status=pending, phase=infer)
  -> LoadCheckpointStopConfig()
       -> if stop_after_checkpoint_epoch has been reached:
          RecordCheckpointStopReached()
          -> terminalize exact training attempt
          -> set experiment.last_model_id to checkpoint model
          -> transition experiment from running/train to pending/infer or analyze
          -> worker exits

Scheduler poll
  -> EnqueueCheckpointEvalRows()
       fallback discovery of committed periodic checkpoint models
  -> RunCheckpointEvalInferJobs()
       -> ReserveCheckpointWorkerAttempt()
       -> BuildCheckpointEvalInferCommand()
       -> launch LSTM_Release --infer
          --model=<checkpoint_model_id>
          --scheduler-checkpoint-eval-id=<checkpoint_eval_id>

Checkpoint inference worker
  -> ResolveSchedulerInferencePersistenceContext()
       validates checkpoint eval, parent experiment, model, epoch,
       symbol, horizon, threshold, inference range
  -> RunInferenceEvaluation()
       computes accuracy/confusion and profitability
  -> PersistCompletedCheckpointInferenceResult()
  -> PersistInferenceProfitabilityObservation()
  -> commit both in one transaction

Scheduler reaps checkpoint-inference child
  -> FindCompletedCheckpointInferenceResultIdForAttempt()
  -> AdvanceCheckpointEvalToAnalyze()
       experiment_checkpoint_eval(status=pending, phase=analyze)

Scheduler checkpoint-analysis path
  -> ClaimCheckpointAnalysis()
  -> ExecuteCheckpointAnalysisWork() [read-only]
       finds checkpoint inference result
       reads structured inference metrics
       computes leader score
  -> FinalizeCheckpointAnalysis() [write transaction]
       UpsertAnalysisResult(scope=checkpoint)
       links experiment_checkpoint_eval.analysis_id
       marks checkpoint eval completed/done
       EvaluateCheckpointPolicyAfterAnalysis()
          -> LoadCheckpointPolicyConfig() FOR UPDATE
          -> LoadValidatedCheckpointPolicyAnalysis()
          -> CountCompletedCheckpointPolicyEvals()
          -> RankCheckpointPolicyEval()
          -> DecideCheckpointPolicy()
          -> PersistCheckpointPolicyDecision()
          -> on stop_requested:
             ApplyCheckpointPolicyStopRequest()
             sets experiment.stop_after_checkpoint_epoch
       -> commit analysis, decision, and stop request atomically

Training worker continues independently
  -> at its next persisted checkpoint:
     LoadCheckpointStopConfig()
  -> if request is due:
     RecordCheckpointStopReached()
     -> lifecycle transition and worker exit
```

Primary evidence:

- Checkpoint model save: [main.cpp:4924](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:4924>)
- Worker-side evaluation queueing: [main.cpp:3340](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:3340>)
- Scheduler fallback checkpoint discovery: [ExperimentScheduler.cpp:9423](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:9423>)
- Inference command identity: [ExperimentScheduler.cpp:9394](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:9394>)
- Inference and profitability persistence: [main.cpp:6342](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:6342>), [main.cpp:6461](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:6461>)
- Analysis finalization and policy invocation: [ExperimentScheduler.cpp:18671](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:18671>)
- Worker-side stop action: [GlobalExperimentControl.cpp:3630](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:3630>)

---

## 4. Existing decision semantics

### What evidence is evaluated

The policy evaluates exactly the `CheckpointEvalRow` whose analysis has just completed. It does not choose a checkpoint to evaluate.

`LoadValidatedCheckpointPolicyAnalysis()` proves that:

- checkpoint eval is `completed/done`;
- linked analysis is `checkpoint/completed`;
- parent experiment matches;
- checkpoint model and epoch match;
- analysis model, epoch, symbol, and horizon match.
  [ExperimentScheduler.cpp:11572](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11572>)

The policy consumes:

- current checkpoint `leader_score`;
- current checkpoint `infer_accuracy`;
- current checkpoint’s rank among all completed checkpoint analyses in configured scope;
- count of completed checkpoint analyses for the parent experiment.

It does not consume trends or deltas.

### Grace behavior

`CountCompletedCheckpointPolicyEvals()` counts all valid completed checkpoint analyses for the parent, including the current one. [ExperimentScheduler.cpp:11637](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11637>)

The current rule is:

```text
if completedEvalCount < graceEvals:
    continue_grace
```

Therefore `grace_evals=1` does not grant a grace decision to the first evaluation; the first count is already one.

The count includes analyses completed before policy enablement because there is no policy revision or enablement watermark.

### Rank behavior

`RankCheckpointPolicyEval()` ranks the current checkpoint against all completed checkpoint analyses in:

- the same symbol and horizon;
- the same horizon; or
- globally.
  [ExperimentScheduler.cpp:11660](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11660>)

Ordering is:

1. `leader_score DESC NULLS LAST`;
2. `infer_accuracy DESC NULLS LAST`;
3. `checkpoint_epoch DESC`;
4. `checkpoint_eval_id ASC`.

This is ranking, but not checkpoint source selection. Profitability must remain absent from this query and comparator.

### OR-based continue rules

After grace:

- Each configured rule is marked passed or failed.
- If at least one configured rule passes, decision is `continue`.
- Only when every configured rule fails does the policy request a stop.
  [ExperimentScheduler.cpp:11762](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11762>)

This is materially different from continuation policy’s AND gate contract.

### Worker and target effects

`continue` and `continue_grace` do not directly signal or mutate the training worker. They mean “do not issue a stop request.”

`stop_requested` sets `experiment.stop_after_checkpoint_epoch` only if the experiment remains `running/train`. [ExperimentScheduler.cpp:11858](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11858>)

The policy does not change `target_epochs`. It requests an earlier checkpoint stop. When reached, the worker:

- persists the checkpoint as `last_model_id`;
- ends the train attempt;
- advances the same experiment to infer or analyze.
  [GlobalExperimentControl.cpp:3633](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp:3633>)

### Stop modes

`RequestedCheckpointPolicyStopEpoch()` implements:

- `next_checkpoint`: checkpoint after `max(evaluated checkpoint, current_epoch)`;
- `current_checkpoint_if_possible`: current evaluated checkpoint only if persisted `current_epoch <= evaluated epoch`;
- otherwise next checkpoint.
  [ExperimentScheduler.cpp:11705](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11705>)

`mark_pruned_when_not_running` has no distinct implementation. Moreover, `EvaluateCheckpointPolicyAfterAnalysis()` rejects all parents not `running/train`, so this stop mode cannot mark a non-running experiment pruned. [ExperimentScheduler.cpp:11935](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11935>)

That is an existing contract ambiguity and should not be expanded during Phase 2C.

### Persistence, reuse, and staleness

`experiment_checkpoint_decision` stores one mutable row per `checkpoint_eval_id`:

- checkpoint/parent/model/epoch;
- decision/reason;
- leader/inference metrics;
- rank and scope;
- requested stop epoch.
  [017_checkpoint_policy.sql:28](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/017_checkpoint_policy.sql:28>)

Persistence uses `ON CONFLICT (checkpoint_eval_id) DO UPDATE`. Decisions are overwritten, not immutable. [ExperimentScheduler.cpp:11834](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11834>)

Persisted decisions are not reused to skip evaluation and are not read by the worker. The worker reads only `stop_after_checkpoint_epoch`.

There is no:

- semantic policy hash;
- policy revision;
- evidence watermark;
- analysis ID;
- inference-result ID;
- rank-population watermark;
- stale-decision detection.

Manual `--evaluate-checkpoint-policy` reruns the current policy and overwrites the row. [ExperimentScheduler.cpp:12013](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:12013>)

---

## 5. Relationship to scheduler continuation

| Dimension | Checkpoint policy | Scheduler continuation |
|---|---|---|
| Timing | While parent is actively `running/train`, after each asynchronous checkpoint analysis | After source experiment is `completed/done` |
| Evaluated source | Exact checkpoint eval that just completed analysis | Selects final/best/latest source according to `source_mode` |
| Source selection | None | Explicit source selection |
| Existing rule composition | OR-based continue rules | Threshold, profitability, rank, and trend eligibility gates |
| Ranking | Ranks current checkpoint against a scoped population | Ranks selected source/candidate as part of continuation eligibility |
| Trend | None | Optional leader/inference trend across distinct epochs |
| Decision effect | May set future checkpoint stop on live worker | May persist eligibility and queue a new child experiment |
| Target epochs | Never changes target; can truncate current run | Determines child target epochs |
| Persistence | Mutable row per checkpoint eval | Policy revision/hash, evidence watermark, reuse identity |
| Missing profitability in Phase 2B | Fails continuation eligibility closed | Not yet applicable |
| Worker interaction | Direct, through `stop_after_checkpoint_epoch` | No direct effect on source worker |
| Inheritance | No checkpoint-policy inheritance | Explicit continuation-policy inheritance |

Continuation’s semantic identity and evidence reuse are implemented in `ContinuationPolicySemanticCanonicalText()`, `ContinuationPolicySemanticHash()`, and `ContinuationEvidenceWatermark()`. [ContinuationPolicy.cpp:541](</Volumes/Developer SSD/ExpertAdvisor/Sources/ContinuationPolicy.cpp:541>), [ContinuationPolicy.cpp:1013](</Volumes/Developer SSD/ExpertAdvisor/Sources/ContinuationPolicy.cpp:1013>)

Phase 2B attaches profitability only after source selection and conditionally adds the selected profitability identity to the watermark. [ExperimentScheduler.cpp:12821](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:12821>)

### Safe reuse boundary

Phase 2C should reuse:

- `InferenceProfitability::SelectAuthoritativeObservation()`;
- `AuthoritativeObservationSelector`;
- metric-definition canonical text and hash;
- observation/provenance types;
- low-level formatting or metric-comparison helpers, if generalized without continuation semantics.

It should not directly reuse:

- `ContinuationProfitabilityGateEvaluation`;
- `EvaluateContinuationProfitabilityGate()`;
- `ContinuationEvidence`;
- fail-closed missing-evidence semantics;
- continuation policy hashing or decision values as checkpoint semantics.

`EvaluateContinuationProfitabilityGate()` assumes configured requirements are an additive AND gate and treats missing evidence as failed. [ContinuationPolicy.cpp:651](</Volumes/Developer SSD/ExpertAdvisor/Sources/ContinuationPolicy.cpp:651>) That is precisely the dangerous coupling to avoid.

`JoinCheckpointPolicyRules()` is already used by continuation diagnostics; its generic behavior is harmless, but its name reveals existing cross-domain coupling that should not be extended.

---

## 6. Profitability availability and exact provenance

### Actual ordering

For a normal directional checkpoint inference:

1. Checkpoint model is committed.
2. `experiment_checkpoint_eval` is queued.
3. Checkpoint inference computes metrics and profitability.
4. Checkpoint `inference_eval_result` is inserted/upserted.
5. Profitability observation is inserted.
6. Both are committed in the same transaction.
7. Scheduler advances checkpoint eval to analyze.
8. Analysis is persisted.
9. Policy evaluates.

The critical transaction is visible at [main.cpp:7929](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:7929>) through [main.cpp:7968](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:7968>).

Therefore there is no externally visible ordinary race in which a new committed checkpoint inference result exists while its profitability insert is still pending.

### States that can still exist

| State | Can occur? | Reason |
|---|---:|---|
| Profitability exists, analysis absent | Yes | Normal pending-analyze window after inference commit |
| Policy evaluates before normal profitability commit | No | Analysis cannot start until committed inference is found |
| Inference exists, profitability absent | Yes | Historical pre-073 result; non-directional target; malformed/manual data; legacy or exceptional path |
| Multiple checkpoint inference results for one eval | Normally no | Unique partial index on `checkpoint_eval_id` |
| Multiple profitability observations for one inference result | Yes | Immutable observations may differ by source-content/statistics identity; selector reports ambiguity |
| Analysis changes on retry | Possible | Checkpoint analysis is upserted by `checkpoint_eval_id` |
| Rank changes later | Yes | New checkpoint analyses can alter the scoped ranking population |
| Old profitability mistaken for current | Preventable | Only with strict inference-result and checkpoint provenance binding |

Migration 018 guarantees one checkpoint inference row per checkpoint evaluation. [018_checkpoint_inference_result_scope.sql:82](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/018_checkpoint_inference_result_scope.sql:82>)

Migration 073 intentionally does not backfill historical profitability. [073_inference_profitability_observation.sql:1](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/073_inference_profitability_observation.sql:1>)

`PersistInferenceProfitabilityObservation()` is a no-op when inference did not produce profitability statistics, so existence is not universally guaranteed. [main.cpp:6469](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:6469>)

### Required provenance proof

A Phase 2C consumer must first resolve the exact checkpoint inference result with all of:

- `checkpoint_eval_id`;
- parent `experiment_id`;
- `checkpoint_model_id` / `model_id`;
- `checkpoint_epoch`;
- `inference_scope='checkpoint'`;
- `status='completed'`;
- symbol;
- prediction horizon;
- threshold;
- window size;
- label rule;
- target type;
- inference start and end;
- completed epoch.

Then select profitability using:

- exact `inference_eval_result_id`;
- exact parent experiment ID;
- exact checkpoint model ID;
- `Scope::checkpointInference`;
- exact `checkpoint_eval_id`;
- current metric-definition canonical text;
- current metric-definition hash.

Finally, the durable evidence identity should include:

- `analysis_id`;
- `checkpoint_eval_id`;
- checkpoint model and epoch;
- `inference_eval_result_id`;
- `profitability_observation_id`;
- `observation_identity_hash`;
- `metric_definition_hash`;
- `source_content_hash`;
- prediction/actionable counts;
- aggregate and average values;
- availability/unavailable reason.

`SelectAuthoritativeObservation()` already rejects no observation, ambiguity, metric mismatch, and provenance mismatch without recency fallback. [InferenceProfitabilityRepository.cpp:310](</Volumes/Developer SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.cpp:310>)

The database trigger also proves inference-result/model/scope/checkpoint/range provenance at insert time. [073_inference_profitability_observation.sql:92](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/073_inference_profitability_observation.sql:92>)

Phase 2C must never:

- query profitability by model or epoch alone;
- use final-scope profitability;
- choose the most recent observation;
- choose a different checkpoint;
- substitute zero.

---

## 7. Missing-evidence and decision-direction recommendation

### Recommended semantics

Use **B: defer**, with one OR-semantics qualification.

Evaluate legacy continue rules first:

1. If any legacy continue rule passes:
   - decision remains `continue`;
   - profitability may be reported as available, unavailable, or pending;
   - profitability is explicitly marked non-decision-bearing for that outcome.

2. If no legacy rule passes and exact profitability is available:
   - profitability criterion passes → `continue`;
   - profitability criterion fails → `stop_requested`, because all configured continue criteria failed.

3. If no legacy rule passes and profitability is unavailable, ambiguous, stale, mismatched, or lookup-failed:
   - `deferred_profitability`;
   - do not persist a stop request;
   - training continues naturally until a later safe evaluation or superseding checkpoint.

This preserves the current meaning: minimum profitability is a condition that may **KEEP training**, not a new mandatory condition required regardless of existing rule passes.

### Why the alternatives are unsafe

- **A, fail closed:** dangerous for a live worker. Infrastructure lag, legacy evidence, ambiguity, or a metric-version mismatch would become a stop signal.
- **C, ignore:** unsafe when profitability is the only remaining possible continue criterion; ignoring it would allow legacy failures to stop training.
- **D, missing equals zero:** destroys the distinction between absent evidence and an observed zero-actionable result and violates Phase 1 provenance semantics.

### Specific evidence states

- **Unavailable:** defer if decision-bearing.
- **Ambiguous:** defer and require operator/retry resolution; never select newest.
- **Stale or provenance-mismatched:** defer, log exact mismatch.
- **Zero actionable:** valid observation, not missing. With the recommended compound criterion it fails minimum actionable count, and its average remains undefined.
- **Partially populated:** current schema rejects most partial shapes; zero-actionable requires a NULL average. Any consumer mapping or schema inconsistency should become unavailable/deferred, not zero.

Migration 073 enforces the average-presence rule. [073_inference_profitability_observation.sql:67](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/073_inference_profitability_observation.sql:67>)

---

## 8. Concurrency and worker-lifecycle analysis

### Current concurrency model

The scheduler evaluates checkpoint policy in-process during checkpoint-analysis finalization.

`LoadCheckpointPolicyConfig()` locks the parent `experiment` row `FOR UPDATE`. [ExperimentScheduler.cpp:11531](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11531>)

The training worker updates `current_epoch` by updating the same experiment row after each epoch. [main.cpp:3152](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:3152>)

Consequently, policy evaluation and epoch progress serialize at the experiment row:

- if the worker committed its latest epoch first, policy sees it;
- if policy locks first, the worker’s next progress update waits until decision commit.

The worker can continue computing while asynchronous inference and analysis are pending. It only reads a stop request after saving a periodic checkpoint.

### Existing safe property

`RequestedCheckpointPolicyStopEpoch()` bases next-checkpoint mode on `max(evaluated checkpoint, persisted current_epoch)`, preventing a normal late policy evaluation from requesting a checkpoint already passed.

### Remaining race hazards

1. **Stale reevaluation after stop:** a later `continue` overwrite does not clear the stop request.
2. **Delayed old checkpoint:** an old deferred checkpoint could acquire profitability after newer checkpoints have been analyzed.
3. **Policy change while evidence is pending:** no hash/revision ties the eventual decision to the earlier configuration.
4. **Rank population changes:** a stored top-N result has no population watermark.
5. **Analysis retry:** the same checkpoint analysis row can be updated, but the decision does not identify the analysis version.
6. **Multiple immutable observations:** retry with changed source content can create ambiguity for the same inference-result ID.
7. **Current-checkpoint mode:** asynchronous profitability should not attempt to stop at the already-produced checkpoint after a defer.

### Required fail-safe behavior

A future deferred reevaluation must lock the parent and verify:

- parent still `running/train`;
- policy revision/hash unchanged;
- exact checkpoint eval/analysis/inference identities unchanged;
- evidence watermark changed from the deferred decision;
- no newer action-bearing checkpoint evaluation has superseded it;
- no prior terminal stop decision was applied;
- requested stop epoch is strictly ahead of current persisted progress;
- a future safe checkpoint exists below target epochs.

For profitability-bearing decisions, initially require `stop_mode=next_checkpoint`. Do not use `current_checkpoint_if_possible` after deferred evidence, and do not rely on the currently nonfunctional `mark_pruned_when_not_running`.

Once a stop request has been applied, treat that lifecycle action as terminal and irreversible for that checkpoint decision. A later reevaluation may diagnose changed evidence but must not overwrite the historical action.

---

## 9. Policy identity and persistence

### Current state

Checkpoint policy has no semantic hash or revision. `CheckpointPolicyRuleText()` is display text, not identity. [ExperimentScheduler.cpp:7676](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:7676>)

Current evidence identity is limited to:

- checkpoint eval;
- parent;
- checkpoint model;
- checkpoint epoch;
- copied leader/inference values;
- rank and rank scope.

It omits analysis identity, inference identity, profitability identity, and population identity.

### Minimum safe migration

A new migration is required, but was not created.

At minimum it should add:

#### `experiment`

- `checkpoint_policy_min_profit_actionable_count bigint NULL`
- `checkpoint_policy_min_profit_average_log_return double precision NULL`
- `checkpoint_policy_revision bigint NOT NULL DEFAULT 1`
- `checkpoint_policy_last_decision_id bigint NULL`
- optionally `checkpoint_policy_stop_decision_id bigint NULL` to attribute an active policy-owned stop request

Constraints:

- profitability fields are either both NULL or both non-NULL;
- actionable threshold is positive;
- average threshold is finite;
- enabled policy may use the compound profitability criterion as its required continue rule;
- profitability-bearing configuration requires checkpoint inference and `next_checkpoint` stop mode for initial Phase 2C.

#### `experiment_checkpoint_decision`

Add:

- `analysis_id bigint`
- `inference_eval_result_id bigint`
- `profitability_observation_id bigint NULL`
- `policy_revision bigint`
- `policy_hash text`
- `evidence_watermark text`
- `profitability_evidence_state text`
- `profitability_gate_reason text`
- `stop_request_applied boolean`
- optional ranking-population watermark

Decision values should include at least:

- `deferred_profitability`
- `superseded`

The existing unique index on `checkpoint_eval_id` must be replaced with immutable identity such as:

```text
UNIQUE (
    checkpoint_eval_id,
    policy_revision,
    policy_hash,
    evidence_watermark
)
```

Existing legacy rows should remain preserved with nullable identity fields or an explicit `legacy_unversioned` state. They must not be considered reusable action-bearing Phase 2C decisions.

### Semantic identity

The checkpoint policy canonical identity should cover:

- all existing continue rules;
- scope;
- stop mode;
- grace count;
- the profitability compound criterion when configured.

When both profitability controls are NULL, profitability fields should be omitted from canonical text. Although checkpoint policy currently has no hash to preserve, this gives disabled-mode semantic stability analogous to Phase 2B.

### Evidence watermark

The watermark should cover:

- exact checkpoint and analysis identity;
- exact inference-result identity and relevant context;
- exact profitability observation identity or exact unavailable reason;
- the scoped ranking population identity when `top_n` is configured.

Mere observation presence should not enter the watermark when profitability controls are disabled.

---

## 10. Proposed configuration surface

Implement only a compound actionable-count-plus-average criterion:

Queue-time CLI:

- `--checkpoint-policy-min-profitability-actionable-count=N`
- `--checkpoint-policy-min-profitability-average-terminal-horizon-log-return-per-actionable-prediction=VALUE`

Runtime keys:

- `min_profitability_actionable_count`
- `min_profitability_average_terminal_horizon_log_return_per_actionable_prediction`

Persisted columns can use the project’s shorter convention:

- `checkpoint_policy_min_profit_actionable_count`
- `checkpoint_policy_min_profit_average_log_return`

Both must be set together or both cleared with `null`.

Reasoning:

- Average return alone is vulnerable to tiny actionable samples.
- Actionable count alone is not profitability.
- Their combination forms one coherent continue criterion.
- Aggregate return grows with sample count and is not position-sized, cost-adjusted, leverage-adjusted, capital-normalized, or portfolio P&L.
- Three independent profitability continue rules would be especially wrong under the existing OR contract: actionable count by itself could keep an economically poor checkpoint training.

No profitability trend, deterioration, delta, rank, or source-selection setting belongs in Phase 2C.

---

## 11. Backward-compatibility invariants

A future implementation must prove:

1. With both checkpoint profitability fields NULL, `DecideCheckpointPolicy()` produces byte-for-byte equivalent decisions and reasons for all existing inputs.
2. Disabled profitability does not alter policy canonical identity or evidence watermark.
3. Presence of profitability observations alone changes neither decision nor identity.
4. Existing OR semantics among leader, inference, and top-N rules remain unchanged.
5. Profitability is one additional compound continue rule, not an additive veto.
6. Profitability does not modify `RankCheckpointPolicyEval()`.
7. Profitability does not affect checkpoint creation, checkpoint inference cadence, grace count, or target epochs.
8. Profitability never selects another checkpoint or inference result.
9. Phase 2B continuation policy, identity, source selection, ranking, trend, and inheritance remain unchanged.
10. Campaign Manager and recommendation scoring/ranking remain unchanged.
11. Training loss/objective, optimizer, labels, and model selection remain unchanged.
12. Final-scope profitability cannot satisfy checkpoint policy.
13. Missing profitability never becomes numeric zero.
14. No profitability-bearing stop can be issued from a stale or superseded checkpoint decision.
15. Existing historical checkpoint decisions remain readable and are not silently reinterpreted as versioned Phase 2C decisions.

The current Phase 2B static isolation test explicitly prohibits profitability in `DecideCheckpointPolicy()`. [ContinuationProfitabilityPolicyIsolationTests.py:43](</Volumes/Developer SSD/ExpertAdvisor/Tests/ContinuationProfitabilityPolicyIsolationTests.py:43>) Phase 2C will need to replace that specific assertion with narrower guards proving profitability remains outside rank/source/trend/recommendation functions.

---

## 12. Minimum deterministic test matrix

### Pure policy tests

| Case | Expected result |
|---|---|
| Profitability controls unset | Exact legacy decision/reason |
| Observation present but controls unset | Same legacy result and identity |
| Legacy rule passes, profitability unavailable | `continue`; profitability non-decision-bearing |
| All legacy rules fail, exact profitability passes | `continue` due profitability criterion |
| All legacy rules fail, profitability fails count | `stop_requested` |
| All legacy rules fail, profitability fails average | `stop_requested` |
| All legacy rules fail, profitability missing | `deferred_profitability`, no stop |
| Zero-actionable observation | Available evidence; criterion fails count and average undefined |
| Threshold equality | Pass using inclusive `>=` |
| Only one profitability field configured | Configuration rejected |
| Profitability-only policy | Valid if both compound fields configured |
| Disabled profitability hash | Stable relative to legacy canonical configuration |
| Configured profitability change | Policy hash and revision change |
| Evidence observation/hash change | Evidence watermark changes |
| Profitability values change without rank fields | Checkpoint rank unchanged |
| Profitability deterioration across epochs | No trend calculation in Phase 2C |

### Provenance tests

- Exact checkpoint observation accepted.
- Final-scope observation offered for checkpoint policy → unavailable/provenance mismatch.
- Observation from wrong checkpoint eval → rejected.
- Observation from same model but wrong checkpoint epoch → rejected.
- Observation from wrong parent experiment → rejected.
- Observation for different inference-result ID → rejected.
- Metric-definition mismatch → unavailable.
- Multiple exact-metric observations with different source-content identity → ambiguous/deferred.
- Missing observation → deferred.
- Partially populated/corrupt mapping → unavailable/deferred.
- Exact inference result with wrong symbol/horizon/threshold/range → rejected before profitability selection.

### Persistence and identity tests

- Same policy and evidence identity is idempotently reused.
- Changed policy creates a new immutable decision identity.
- Changed analysis/inference/profitability evidence creates a new identity.
- Legacy unversioned decision is preserved but not reused.
- A terminal `stop_requested` decision cannot be overwritten by later `continue`.
- Deferred decision may resolve exactly once under the same policy identity.
- Deferred decision is marked superseded when a newer checkpoint becomes action-bearing.
- `stop_request_applied` and `stop_after_checkpoint_epoch` commit atomically.
- No stop row is applied for deferred decisions.
- Ranking population change alters watermark when top-N is configured.

### Retry tests

- Deterministic retry returns the same inference-result and profitability observation identity.
- Changed source content under the same inference-result ID creates ambiguity and defers.
- Analysis retry changes analysis identity/watermark.
- Evidence arriving after a newer checkpoint cannot stop from the older checkpoint.
- Policy change while deferred prevents reuse under the old policy hash.

### Worker/scheduler lifecycle tests

- Worker advances epochs while inference is pending: no premature stop.
- Policy evaluates with worker already beyond evaluated checkpoint: stop scheduled only at a future checkpoint.
- Worker reaches target before deferred profitability resolves: no late stop.
- Parent leaves `running/train`: delayed evaluation becomes observational/superseded.
- Scheduler authority loss rolls back analysis decision and stop request.
- Current-checkpoint stop mode is rejected or forced non-action-bearing for Phase 2C profitability.
- Policy stop is detected and acted on only by the exact active training attempt.

### Regression tests

- Existing checkpoint leader/inference/top-N/grace behavior.
- Phase 2B continuation profitability tests unchanged.
- Continuation source selection/dedup/rank/trend unchanged.
- Campaign Manager candidate scoring/ranking unchanged.
- Recommendation scoring/ranking unchanged.
- Training objective and checkpoint serialization unchanged.

There is currently no substantive dedicated checkpoint-policy unit suite; the only direct test reference is the Phase 2B isolation assertion. That makes extraction of a small pure checkpoint profitability criterion advisable for deterministic tests.

---

## 13. Files and schema a future implementation would modify

Likely required:

- [Sources/ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp>)
  - `SchedulerOptions`
  - parsing and help
  - `CheckpointPolicyConfig`
  - `CheckpointPolicyDecision`
  - `CheckpointPolicyConfigurationError`
  - queue/control persistence
  - `LoadCheckpointPolicyConfig`
  - exact profitability attachment
  - `DecideCheckpointPolicy`
  - decision persistence/reuse
  - stop-action fencing
  - status output

- [Sources/InferenceProfitabilityRepository.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.hpp>)
- [Sources/InferenceProfitabilityRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.cpp>)
  - preferably no semantic change;
  - reuse exact authoritative selector;
  - only add generic mapping/helper support if necessary.

- New migration after 074
  - configuration columns;
  - policy revision/identity;
  - immutable decision identity;
  - exact evidence foreign keys;
  - deferred/superseded decisions;
  - stop-decision attribution.

- New checkpoint-policy tests
  - pure decision tests;
  - migration/persistence tests;
  - scheduler-worker ordering integration tests.

- [Tests/ContinuationProfitabilityPolicyIsolationTests.py](</Volumes/Developer SSD/ExpertAdvisor/Tests/ContinuationProfitabilityPolicyIsolationTests.py>)
  - narrow the current blanket prohibition while retaining rank/source/trend/campaign isolation.

- [docs/InferenceProfitabilityPersistence.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/InferenceProfitabilityPersistence.rst>)
- [Database/README.md](</Volumes/Developer SSD/ExpertAdvisor/Database/README.md>)
  - document checkpoint-specific semantics and metric limitations.

`LSTM/main.cpp` should not require profitability-computation changes; it already persists checkpoint inference and profitability atomically. Any modification there should be treated as a warning that Phase 2C is expanding beyond its intended consumer-policy boundary.

`ContinuationPolicy.cpp`, Campaign Manager, recommendation scoring, and training code should remain behaviorally unchanged.

---

## 14. Observability requirements

### Checkpoint-policy evaluation logs

Every evaluation should include:

- policy enabled/disabled;
- policy revision/hash;
- checkpoint eval/model/epoch/parent;
- analysis ID;
- inference-result ID and scope;
- profitability observation ID/hash;
- metric-definition and source-content hashes;
- prediction/actionable counts;
- aggregate and average values;
- evidence state: `pending`, `unavailable`, `ambiguous`, `mismatched`, `available`;
- unavailable reason;
- profitability criterion result;
- whether profitability was decision-bearing;
- final decision;
- stop request epoch and whether applied;
- evidence watermark;
- superseding checkpoint/decision, if any.

Distinct markers should include:

- `CHECKPOINT_PROFITABILITY_POLICY_DISABLED`
- `CHECKPOINT_PROFITABILITY_EVIDENCE_PENDING`
- `CHECKPOINT_PROFITABILITY_EVIDENCE_UNAVAILABLE`
- `CHECKPOINT_PROFITABILITY_EVIDENCE_AVAILABLE`
- `CHECKPOINT_PROFITABILITY_CRITERION_PASSED`
- `CHECKPOINT_PROFITABILITY_CRITERION_FAILED`
- `CHECKPOINT_POLICY_DEFERRED_PROFITABILITY`
- `CHECKPOINT_POLICY_STOP_REQUESTED_PROFITABILITY`
- `CHECKPOINT_POLICY_CONTINUE_PROFITABILITY_NON_DECISION_BEARING`

### Scheduler and experiment status

Existing status exposes policy config and last decision but not evidence identity. [ExperimentScheduler.cpp:21900](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:21900>)

Add:

- profitability criterion enabled/disabled;
- both thresholds;
- last profitability evidence state;
- last observation/inference IDs;
- last criterion pass/fail;
- deferred/superseded state;
- last decision’s policy hash and evidence watermark;
- whether a policy-owned stop is pending;
- stop decision ID.

### Policy inspection

Do not use the mutating `--evaluate-checkpoint-policy` command as inspection.

Add a read-only status/inspection path, for example:

```text
--checkpoint-policy-status=CHECKPOINT_EVAL_ID
```

It should display current policy/evidence identity and any durable decision without reevaluating or changing worker state.

---

## 15. Risks and unresolved questions

1. **Branch mismatch:** repository is on `lstm-feature-development`, not the `phase6` branch stated in `AGENTS.md`.
2. **Stop-mode ambiguity:** `mark_pruned_when_not_running` is accepted but not implemented; evaluation rejects non-running parents.
3. **Current-checkpoint race:** `current_checkpoint_if_possible` is poorly aligned with asynchronous profitability.
4. **Mutable historical decisions:** current checkpoint decision rows are overwritten.
5. **Irreversible side effect mismatch:** reevaluation can overwrite decision text without undoing an earlier stop.
6. **No rank-population identity:** top-N decisions can become observationally stale.
7. **No enablement epoch/revision:** grace counting includes checkpoint analyses from before policy enablement.
8. **Non-directional models:** inference can commit without profitability because the optional statistics are absent.
9. **Historical checkpoint inference:** migration 073 deliberately does not backfill.
10. **Observation ambiguity:** immutable retry observations can make exact profitability unavailable.
11. **No dedicated checkpoint-policy test suite:** lifecycle-active behavior currently lacks focused deterministic coverage.
12. **Meaning of “continue”:** it is passive absence of a stop request, not an explicit worker command.
13. **Metric limitations:** even average directional log return remains an inference statistic, not realized portfolio economics.

No production experiment rows or persisted decisions were inspected; only schema metadata was queried read-only.

---

## Final implementation boundary and sequencing

### Sequence 1: checkpoint decision hardening prerequisite

Before profitability becomes decision-bearing:

1. Introduce checkpoint policy revision/hash.
2. Introduce exact evidence watermark and stored analysis/inference identity.
3. Make decision records immutable by semantic/evidence identity.
4. Add deferred/superseded states.
5. Fence stop application to the exact decision and active training attempt.
6. Define terminal behavior after a stop request.
7. Add read-only checkpoint-policy inspection.
8. Resolve or explicitly exclude the two problematic stop modes.

This prerequisite should preserve existing decision outcomes when profitability is unset.

### Sequence 2: narrow Phase 2C

Then implement:

- one compound profitability continue criterion;
- minimum actionable count plus minimum average terminal-horizon directional log return;
- exact checkpoint-only provenance;
- OR composition with existing continue rules;
- defer when profitability is decision-bearing but unavailable;
- `next_checkpoint` only;
- no aggregate threshold;
- no profitability trend;
- no profitability ranking;
- no source selection;
- no continuation/Campaign Manager/recommendation/training changes.

This is a small, defensible Phase 2C only after the decision lifecycle can prove which policy and exact evidence caused a live-worker stop.

## Inspection record

- Files changed: none.
- Database changes: none; schema inspection used an explicit read-only transaction.
- Scheduler/workers launched: none.
- Builds/tests run: none, consistent with inspection-only scope.
- `git status --short`: empty.
- `git diff --stat`: empty.
- Current branch: `lstm-feature-development`.