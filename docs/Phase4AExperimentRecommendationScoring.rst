Phase 4A Step 4: advisory recommendation scoring
================================================

Phase 4A scoring is an explicit deterministic prioritization aid.  It does
not approve recommendations, create experiments, consume scheduler slots, or
predict profitability.  It is never invoked from scheduler polling.

Identity and provenance
-----------------------

Step 1 identities remain unchanged.  The scoring policy has authoritative
canonical text prefixed by ``experiment_recommendation_scoring_policy_v1``;
its tagged FNV-1a hash is diagnostic only.  Migration 028 adds ``source_rank``
because that rank previously existed only in generation memory.  Migrations
029 through 031 enforce a positive rank for every scan-associated Step 3 row,
prevent a persisted rank from being cleared or made nonpositive, and prevent
a persisted Step 3 row from being detached from its originating scan.  They
deliberately do not rewrite legacy rows; legacy rows without a scan or rank
remain valid and are skipped by scoring rather than assigned a guessed value.

Step 2 does not define a combined structural-distance field.  It emits exact
``absolute_delta``, optional ``relative_delta``, and optional ``horizon_delta``
metadata.  Step 3 persists those values unchanged.  Step 4 derives its
scoring-only distance as relative delta when present and absolute delta for
the zero-source fallback.  ``generation_ordinal``, ``structural_rank``, and
Step 4 ``score_rank`` are ordering metadata and are never used as that
distance.

An equal scoring-policy hash with different canonical text is emitted as
``EXPERIMENT_RECOMMENDATION_SCORE_HASH_COLLISION`` and counted on the run.  It
does not merge policies, block scoring, or establish equality.

Formula
-------

All component values are bounded to [0, 1].  Positive components are leader
quality, inference accuracy, evidence strength, neutral balance, structural
proximity, explicit parameter preference, and reciprocal source rank.
Penalties are normalized horizon change and relative mutation.

Leader score is clamped to [0, 1]; inference accuracy must already be in that
range.  Evidence strength is zero at the configured minimum and increases
linearly to one at saturation.  Neutral balance is one at or below the
preferred proportion and declines linearly to zero at the maximum.  A missing
neutral proportion contributes 0.5 only when explicitly allowed.

Structural proximity uses relative delta when available and absolute delta
only for the zero-source fallback::

  proximity = clamp(1 - distance / configured_maximum, 0, 1)

Horizon penalty is zero for non-horizon changes; otherwise it is absolute
horizon delta divided by source horizon, clamped to one.  Relative mutation
uses persisted relative delta, with the same explicit absolute fallback::

  raw_positive = sum(positive_weight * component) / sum(positive_weight)
  raw_penalty = sum(penalty_weight * penalty) / sum(positive_weight)
  raw_total = raw_positive - raw_penalty
  final_score = clamp(raw_total, score_floor, score_ceiling)

Defaults use [0, 1].  Nonfinite or malformed inputs are rejected, never
silently scored as zero.  Parameter preferences default to neutral values and
are explicit operator preferences, not predicted performance.

The policy exposes ``scoring_version``; the nine component/penalty weights;
``minimum_evidence_count`` and ``evidence_saturation_count``; preferred and
maximum neutral proportions; maximum relative and absolute fallback distance;
missing-neutral handling; score floor and ceiling; and one [0, 1] preference
for each supported mutation parameter.  Defaults are 0.25/0.25 for leader and
accuracy, 0.15 for evidence and structural proximity, 0.10 for neutral
balance, 0.05 for parameter preference and source rank, and penalties of 0.05
for horizon changes and 0.10 for relative mutation.

Ranking and terminology
-----------------------

One run orders scores by final score descending, raw positive descending, raw
penalty ascending, source leader score descending, source inference accuracy
descending, evidence count descending, structural distance ascending,
semantic canonical text, recommendation-policy
canonical text, and recommendation ID.  ``score_rank`` is competition rank
for equal final scores, ``tie_group`` identifies equal final scores, and
``ranking_ordinal`` is unique deterministic order.  Source rank, generation
ordinal, structural rank, structural distance, and score rank remain distinct.

Persistence and concurrency
---------------------------

``experiment_recommendation_score_run`` records each explicit invocation,
``experiment_recommendation_score`` stores immutable results, and
``experiment_recommendation_score_component`` stores ordered provenance.
The unique key ``(recommendation_score_run_id, recommendation_id)`` serializes
same-run retries.  A retry is idempotent only when every immutable score field
and the complete ordered component set match after canonical double
round-trip comparison; any mismatch fails with
``recommendation_score_retry_mismatch``.  Separate runs retain separate
history.  Foreign keys use restrictive deletion and no score operation
updates ``experiment`` or ``experiment_recommendation``.

The service creates a run in a short transaction, loads scoreable rows
read-only, scores and ranks in memory, persists each score and its components
atomically, then finalizes the run separately.  The supported scoring
lifecycle is exactly ``proposed`` and ``approved``: the default command scope
is ``proposed``, and ``--recommendation-status-filter=approved`` explicitly
scores approved rows.  ``rejected`` and ``expired`` are not scoreable and are
rejected by the scoring command.  Status filtering never changes a
recommendation's status or review history.

Commands and deferred behavior
------------------------------

Commands are ``--score-experiment-recommendations``,
``--list-experiment-recommendation-scores``,
``--recommendation-score-status=ID``,
``--list-experiment-recommendation-score-runs``,
``--recommendation-score-run-status=ID``, and
``--explain-recommendation-score=ID``.  Explanations are deterministic from
persisted components and include an advisory disclaimer.

Approval, conversion, experiment creation, automatic score scans, and
queueing remain deferred.

Workflow order
--------------

Both of these explicit workflows are valid when the recommendation otherwise
meets scoring and conversion prerequisites::

  recommendation generation
    -> optional/explicit scoring
    -> manual recommendation approval

  recommendation generation
    -> manual recommendation approval
    -> explicit scoring
    -> Phase 4C conversion

Scoring remains advisory in both orders: it does not approve, reject, expire,
create conversion proposals or experiments, queue scheduler work, or launch
workers.
