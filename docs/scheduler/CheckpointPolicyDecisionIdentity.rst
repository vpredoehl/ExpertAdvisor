Checkpoint Policy Decision Identity
===================================

This document describes the checkpoint-policy lifecycle hardening installed by
migration 075.  It is a prerequisite for, not an activation of, checkpoint
profitability policy.  Profitability remains completely non-decision-bearing.

Decision path and transactions
------------------------------

The automatic path remains::

  checkpoint analysis finalized
    -> EvaluateCheckpointPolicyAfterAnalysis()
    -> LoadCheckpointPolicyConfig()
    -> LoadValidatedCheckpointPolicyAnalysis()
    -> CountCompletedCheckpointPolicyEvals()
    -> RankCheckpointPolicyEval()
    -> DecideCheckpointPolicy()
    -> PersistCheckpointPolicyDecision()
    -> ApplyCheckpointPolicyStopRequest()
    -> experiment.stop_after_checkpoint_epoch
    -> the training worker observes the persisted stop epoch
    -> RecordCheckpointStopReached()

Automatic evaluation executes in the scheduler's read-write transaction that
finalizes the exact checkpoint analysis and worker attempt.  Explicit
``--evaluate-checkpoint-policy=CHECKPOINT_EVAL_ID`` uses the same evaluator in a
separate read-write transaction.  The parent experiment row is locked while
policy identity, decision persistence, and any stop action are resolved.  A
stop epoch, its experiment attribution, the decision's applied-action fields,
and supersession transitions commit or roll back together.

``--checkpoint-policy-status=CHECKPOINT_EVAL_ID`` is separate.  It uses a
read-only transaction and never evaluates through the mutating workflow,
inserts a decision, changes a policy revision, changes last-decision fields,
sets a stop epoch, or queues work.

Legacy decision behavior
------------------------

The continue rules retain their original OR semantics.  A passing leader-score
rule, inference-accuracy rule, or top-N rule produces ``continue``.  Only the
failure of every configured rule produces ``stop_requested``.  Grace counting
continues to count all completed valid checkpoint evaluations for the parent,
including evaluations that predate policy enablement; policy revision does not
reset grace.  Rank ordering and scope remain leader score descending, inference
accuracy descending, checkpoint epoch descending, and checkpoint evaluation ID
ascending.

``next_checkpoint``, ``current_checkpoint_if_possible``, and
``mark_pruned_when_not_running`` retain their decision and requested-epoch
calculation.  Stop fencing can make a race-sensitive request observational
rather than applied.  ``mark_pruned_when_not_running`` gains no new pruning
meaning.  A future profitability-aware Phase 2C should initially support only
``next_checkpoint`` unless a later design proves another mode safe.

Semantic policy identity and revision
-------------------------------------

The canonical, delimiter-stable policy identity is::

  enabled
  min_leader_score
  min_infer_accuracy
  top_n
  rank_scope
  stop_mode
  grace_evals
  checkpoint_interval
  target_epochs

Optional numeric values use ``NULL`` or C++ round-trip precision.  The hash is
the repository's deterministic 64-bit FNV-1a convention rendered as 16 lower
case hexadecimal characters.  Current epoch, status, phase, worker identity,
timestamps, last-decision fields, and persisted hash/revision metadata are not
policy identity.  Profitability is not policy identity.

New experiments begin at revision 1.  An effective enable/disable or
decision-bearing control change increments the revision exactly once and stores
the new hash.  An unchanged control request and read-only status do not increment
it.  A legacy NULL hash is materialized without increment during the first
mutating evaluation.  If a decision-bearing configuration was changed outside
the authoritative control workflow, mutating evaluation detects the hash
mismatch, increments the revision, and rematerializes the hash before deciding.

Exact evidence watermark
------------------------

The evidence watermark hashes::

  checkpoint_eval_id
  parent_experiment_id
  checkpoint_model_id
  checkpoint_epoch
  observed persisted current_epoch
  analysis_id
  exact checkpoint inference_eval_result_id
  symbol and prediction horizon
  exact inference from/to dates
  leader_score and inference_accuracy
  rank value, scope, and rank-population watermark when top-N is configured
  completed checkpoint evaluation count and population watermark
  completed/done checkpoint state
  completed analysis state
  completed checkpoint inference state and scope

The exact inference result must be the unique completed checkpoint-scope result
for the checkpoint evaluation and must agree with the parent, checkpoint model,
checkpoint epoch, symbol, horizon, threshold, inference range, and the analysis
inference accuracy.  There is no final-inference fallback, latest-model lookup,
or recency heuristic.  Rank population identity hashes the unchanged sorted
ranking population, including checkpoint evaluation, analysis, available exact
inference result, metric, epoch, and lifecycle identity.  When top-N is not
configured, rank population is explicitly marked non-decision-bearing.

Profitability observations, metrics, definitions, source selection, ranking,
and trends are absent from both policy and evidence identity.

Durable decisions and supersession
----------------------------------

New durable identity is::

  checkpoint_eval_id + policy_revision + policy_hash + evidence_watermark

An identical reevaluation reuses the existing row.  A policy or evidence change
inserts a new row.  Decision meaning is protected by a database immutability
trigger.  The only state-changing lifecycle transitions are
``active -> superseded`` and ``active -> action_applied``.  ``legacy``,
``superseded``, and ``action_applied`` rows are terminal and immutable.

Migration 075 retains old rows as ``legacy`` with NULL policy/evidence identity.
They remain queryable and visible but are never treated as equivalent to a new
versioned identity and cannot become a hardened stop attribution.

Checkpoint authority orders epoch first and checkpoint evaluation ID second.
Older active decisions are superseded when their epoch is lower, when their
epoch ties and their checkpoint evaluation ID is lower, or when the same
checkpoint evaluation is decided under changed policy/evidence.  A stale old
checkpoint is inserted as observational/superseded.  Once a stop is
applied, later reevaluations remain durable observations but are superseded and
cannot replace or cancel the terminal stop decision.  Automatic cancellation is
not provided; manual stop-clear controls remain a separate operator action.
No generic ``deferred`` state is added: this prerequisite has no safe pending
evidence/action condition that should later become executable without a fresh
identity-bearing reevaluation.

Stop attribution and fencing
----------------------------

``experiment.checkpoint_policy_stop_decision_id`` identifies the exact durable
decision that set ``stop_after_checkpoint_epoch``.  The decision records the
applied flag, time, requested epoch, and exact active training worker attempt.

Before applying a stop, the workflow revalidates that:

* the parent is still ``running/train``;
* the checkpoint and exact analysis/inference evidence still match;
* policy revision and hash are current;
* the evidence and ranking/completed populations still hash identically;
* the decision is active and no newer action-bearing checkpoint supersedes it;
* neither a terminal policy stop nor a conflicting manual stop is present;
* the requested epoch is ahead of persisted progress and before target epochs;
* the exact training attempt is active under the current unexpired scheduler
  lease and fencing token.

A failed fence leaves ``stop_after_checkpoint_epoch`` unchanged and marks the
decision superseded with a deterministic reason.  The worker mechanism itself
is unchanged: it continues to consume only ``stop_after_checkpoint_epoch``.

Phase boundary
--------------

This patch adds no checkpoint profitability threshold, CLI option, gate,
ranking, trend, source mode, or stop behavior.  Continuation policy, Campaign
Manager/recommendation scoring, and training objectives are unchanged.  Actual
Profitability Phase 2C activation is a separate future patch.
