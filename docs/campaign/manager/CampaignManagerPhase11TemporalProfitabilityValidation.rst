Campaign Manager Phase 11: Temporal Profitability Validation
=============================================================

Phase 11 preserves the Phase 8--10 profitability contracts and keeps the live
profitability ranking weight at exactly zero.  It adds two advisory, read-only
commands.  Neither command changes a recommendation, evaluation, ranking
snapshot, profitability observation, experiment, campaign, scheduler, or
worker.

Historical feasibility audit
----------------------------

::

  LSTM_Release --validate-campaign-profitability-temporal

The command opens one repeatable-read PostgreSQL read transaction and audits
every completed ranking snapshot.  It reconstructs the persisted zero-weight
control only when the snapshot, evaluation run, membership, policy, semantic
identities, member provenance, and frozen exact-FINAL profitability evidence
pass the existing Phase 9/10 contracts.  It then searches for an outcome that:

* was created and completed after the immutable selection cutoff;
* has an inference period starting strictly after that cutoff;
* is a completed FINAL result, never a checkpoint substitute;
* has the exact experiment, model, symbol, and horizon identity;
* has matching result and observation date ranges;
* uses the authoritative profitability metric-definition hash; and
* retains nonempty source-content and observation identity hashes.

Later-arriving evidence with a historical/non-subsequent market window is
reported as future-information leakage, not accepted as an outcome.  Missing,
ambiguous, mismatched, overlapping, or non-reconstructable evidence fails the
cohort closed.

The primary candidate weight is the precommitted ``0.025``.  The command does
not repeat the Phase 10 sweep.  ``0.0225`` and ``0.0275`` are identified only
as non-selective sensitivity values and cannot replace the primary candidate.

Current historical finding
--------------------------

The production database has four persisted ranking snapshots.  None has a
scientifically admissible subsequent profitability outcome.  Snapshots 1, 3,
and 4 predate the current reconstructable ranking-snapshot contract and fail
closed as ``insufficient_ranking_time_provenance``.  Snapshot 5 reconstructs
the authoritative zero-weight control exactly and retains the Phase 10
coverage result (28 valid exact-FINAL members and 51 unavailable members), but
has zero outcomes whose assessment period begins after its August 28, 2026
selection cutoff.

All persisted profitability windows end no later than January 1, 2026.  A
profitability observation associated with snapshot 4 source identity arrived
after its August 15 cutoff, but its inference period is January 1, 2025 through
January 1, 2026.  It is explicitly rejected as temporally contaminated and is
not a holdout outcome.

Consequently Phase 11 renders
``HISTORICAL_HOLDOUT_UNAVAILABLE_FORWARD_VALIDATION_REQUIRED``.  It emits no
historical top-5, top-10, or top-20 outcome comparison, entrant-versus-exit
profitability, aggregate robustness result, or statistical claim because no
admissible cohort exists.

Forward-validation precommit
----------------------------

::

  LSTM_Release \
    --prepare-campaign-profitability-forward-validation=SNAPSHOT_ID,OUTCOME_START,OUTCOME_END

The ISO outcome dates must be strictly after the immutable snapshot decision
date and ``OUTCOME_END`` must be later than ``OUTCOME_START``.  The command
reconstructs the exact zero-weight control, constructs only the precommitted
``0.025`` shadow ranking, and emits deterministic member and top-5/top-10/top-20
records.  Each record binds the recommendation, evaluation result, ranking
member, source experiment/model, symbol/horizon, frozen ranking-time evidence
identity, expected outcome window, and canonical hash.  Subsequent outcome
identity remains ``PENDING``.

No database persistence was added.  The emitted precommit must be committed to
an external immutable/versioned artifact before the outcome window is observed
if a durable scientific timestamp is required.  The command does not launch
inference or create/queue an experiment.  A separately authorized future phase
must produce and validate the exact FINAL outcome artifact.

Coverage policy remains advisory
--------------------------------

Phase 11 does not create an activation threshold.  Evidence for a later human
policy should consider overall and top-5/top-10/top-20 exact-FINAL ranking-time
coverage, subsequent-outcome coverage, missingness concentration by era,
symbol, and horizon, the number of independent temporal cohorts, and whether
selection changes depend on unavailable evidence.  Any later policy should
retain exact-FINAL-only evidence and fail closed on checkpoint substitution.

Production safety
-----------------

Every Phase 11 safety summary reports ``activation=false``,
``live_profitability_weight=0``, ``production_ranking_modified=false``,
``recommendation_modified=false``, ``ranking_snapshot_modified=false``,
``experiment_created=false``, ``experiment_queued=false``,
``scheduler_modified=false``, and ``worker_modified=false``.  There is no
activation command and no Campaign Manager production path consumes weight
``0.025``.
