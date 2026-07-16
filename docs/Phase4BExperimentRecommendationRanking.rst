Phase 4B Step 2 Advisory Recommendation Ranking
================================================

Status
------

Phase 4B Step 2 is deterministic ranking, comparison, and presentation over
immutable Phase 4B Step 1 evaluation evidence. It remains advisory: it cannot
approve recommendations, create or queue experiments, enter scheduler polling,
or change experiment, review, or evaluation state.

Authority and policy
--------------------

Persisted evaluation runs, results, and components are authoritative. Ranking
joins recommendation rows only for immutable display and scope fields. It does
not reload model metrics, inference, analysis, checkpoints, continuations, or
current experiment state and never reconstructs a Step 1 score.

Policy version 1 defines three buckets:

* ``advisory_ready`` orders final score descending, semantic hash, evaluation
  hash, and evaluation-result ID as a final uniqueness fallback.
* ``blocked`` orders pending, active, then completed duplicates, followed by
  the identity tie-breaks. This is presentation order, not execution priority.
* ``non_actionable`` orders insufficient, stale, unsupported, then invalid
  evidence, followed by the identity tie-breaks.

The policy adds no component or weight. Top positive and penalty component
names come from persisted Step 1 components, ordered by contribution descending
and component name.

The complete source membership is bucketed and deterministically ordered before
the requested limit truncates the combined global order. The limit therefore
does not change which member wins an earlier policy-defined position.

Scopes and bounds
-----------------

Each request selects exactly one scope: evaluation run, recommendation scan,
symbol, horizon, family, symbol plus horizon, or explicit global. Scope
canonical text defines the type and ``NULL`` representation. Input, output,
comparison-component, and list bounds are 1000, 1000, 100, and 1000. Oversized
input fails rather than being truncated by database row arrival.

Evaluation-run scope contains that run's results. Every other scope contains all
matching historical Step 1 evaluation-result rows; it does not silently select
a latest result or collapse multiple evaluations of one recommendation.
The repository selects at most 1,001 evaluation identities before joining
components, so an oversized scope fails at the 1,000-input boundary without
materializing an unbounded component join. Persisted ``component_count`` must
equal the complete component set loaded for every result.

Snapshot identity and persistence
---------------------------------

``experiment_recommendation_ranking_snapshot`` stores versioned canonical
identity, policy, scope, limit, and canonically sorted complete source
membership. Canonical text is authoritative; its tagged hash is an accelerator.
Policy, scope, limit, membership identity, or evaluation-result ID changes
produce a different identity.

``experiment_recommendation_ranking_member`` stores the exact selected Step 1
result, bucket/rank, global ordinal, persisted score/disposition, tie-breaks,
reasons, top component names, and snapshot-time display metadata. It references
rather than copies Step 1 components.

All members persist in one short transaction. A member error rolls back the
entire set. Completion follows an exact member-count check. Identical concurrent
requests converge through canonical uniqueness, a short lock on their shared
ranking-owned snapshot, and exact retry comparison;
mismatches fail. Runtime access is SELECT/INSERT plus column-scoped snapshot
lifecycle/count updates. Members and immutable snapshot fields cannot be
updated or deleted by the runtime role.

The member insert trigger pins the migration-time schema search path and locks
only the ranking-owned snapshot. It rejects new members on terminal snapshots,
while allowing an exact completed-snapshot retry to reach the existing-member
uniqueness check. It also rejects evaluations outside the stored scope or
captured source membership, copied identity/score/disposition/provenance/display
mismatches, and derived reason or top-component mismatches. These database
checks do not redefine the ranking policy; the pure domain policy remains
responsible for ordering and rank assignment.

Comparison
----------

Direct comparison requires matching authoritative evaluation/scoring policy
canonical text and versions, evaluator version, family, and a final score on
both sides. Otherwise an explicit incomparability state replaces any winner.
Component deltas align persisted components by stable name, with missing sides
reported as ``NULL``.

Evaluation comparison has no implicit snapshot and reports score/component
differences only. Ranked-member comparison uses ranks only when both members
belong to the same explicit snapshot; different snapshots are
``incomparable_scope``. No arbitrary latest snapshot is selected.

CLI
---

Examples::

  LSTM_Release --rank-experiment-recommendation-evaluations \
    --recommendation-ranking-evaluation-run-id=17

  LSTM_Release --rank-experiment-recommendation-evaluations \
    --recommendation-ranking-symbol=EURUSD \
    --recommendation-ranking-horizon=12 \
    --recommendation-ranking-limit=100 \
    --recommendation-ranking-dry-run

  LSTM_Release --rank-experiment-recommendation-evaluations \
    --recommendation-ranking-global

  LSTM_Release --list-experiment-recommendation-ranking-snapshots
  LSTM_Release --recommendation-ranking-status=4
  LSTM_Release --list-experiment-recommendation-ranking-members=4 \
    --recommendation-ranking-bucket=advisory_ready
  LSTM_Release --recommendation-ranking-member-status=22
  LSTM_Release --compare-experiment-recommendation-evaluations=41:52
  LSTM_Release --compare-experiment-recommendation-ranking-members=21:22

Dry-run loads Step 1 evidence and emits the same identity, ordering, and
explanations, but inserts and updates nothing. Machine records use
``EXPERIMENT_RECOMMENDATION_RANKING_*``, explicit ``NULL``, and existing percent
escaping. Every ranking/comparison summary reports
``experiment_created=false``, ``experiment_queued=false``, and
``scheduler_modified=false``. Human output separates all three buckets and
states the advisory-only boundary.

Operational safety and deferred work
------------------------------------

Migration 035 is additive, owns only ranking tables, has no source-table
trigger, and has restrictive foreign keys. Ranking takes ``FOR UPDATE`` only on
its own snapshot while verifying member persistence; it does not take such locks
on recommendations, evaluations, experiments, evidence, reviews, or scheduler
state. Write tests use exact disposable schemas.

Profitability, approval, conversion, experiment creation, queueing, autonomous
selection, and scheduler integration remain deferred. Rank is advisory evidence,
not execution authority or a profitability claim.

References
----------

* ``docs/architecture/Volume_I_Foundation.md``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
* ``docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md``
* ``docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md``
* ``docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md``
* ``docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md``
* ``docs/Phase4BExperimentRecommendationEvaluation.rst``
