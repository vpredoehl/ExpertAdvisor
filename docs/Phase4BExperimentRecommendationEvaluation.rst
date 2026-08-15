Phase 4B Step 1: deterministic advisory recommendation evaluation
==================================================================

Boundary
--------

Phase 4B Step 1 evaluates persisted ``proposed`` recommendations without
changing their review status.  It does not approve, reject, expire, convert,
create, or queue anything.  It does not enter scheduler polling or update an
experiment, model, inference result, analysis result, checkpoint record, or
continuation record.  Profitability evidence and claims are outside this step.

Policy and identity
-------------------

``RecommendationEvaluationPolicy`` version 1 is an advisory classification
envelope around the unchanged Phase 4A Step 4
``RecommendationScoringPolicy`` version 1.  Its canonical prefix is
``experiment_recommendation_evaluation_policy_v1`` and it embeds the complete,
length-delimited Step 4 policy canonical text.  Evaluation does not define a
second score formula.

Evidence canonicalization records every persisted field used for
classification or scoring, explicit ``NULL`` markers, and exact experiment
conflicts sorted independently of database row order.  Nonfinite numbers are
represented as rejected evidence markers and classify as invalid; they are
never scored.  Evaluation identity length-delimits the policy, recommendation
semantic/policy identity, and evidence canonical text.  Tagged FNV-1a hashes
are lookup accelerators only; canonical text remains authoritative.  The
recommendation semantic hash is carried from the Step 1 persisted identity and
validated during persistence rather than recomputed as a second identity
implementation.

A run snapshot sorts evaluation identities and records the scan or
recommendation filter and limit.  Identical authoritative inputs therefore
produce one canonical run identity.  A policy or evidence change produces a
different historical run.

Classification
--------------

Version 1 exposes these dispositions:

``advisory_ready``
  The source scan is completed, current completed/final source provenance
  matches the recommendation snapshot, required evidence exists, no exact
  pending/running/completed experiment represents the candidate, and Step 4
  scoring succeeds.

``insufficient_evidence``
  Required model, analysis, sample-count, or policy-required scoring evidence
  is absent.  The final score is ``NULL``, never zero by implication.

``blocked_pending_duplicate`` / ``blocked_active_duplicate`` /
``completed_duplicate``
  An experiment rebuilt through the existing Phase 4A semantic canonicalizer
  exactly matches the recommended configuration and has the named current
  status.  Failed and cancelled experiments are not newly reinterpreted by
  Phase 4B.

``stale_source_evidence``
  The source scan/experiment is no longer completed, or the recommendation's
  source model/analysis provenance no longer matches the current completed
  final evidence.  Staleness is provenance-based and does not read wall-clock
  age.

``unsupported_recommendation_family``
  The changed parameter is outside the four families already represented by
  Phase 4A identity and scoring.

``invalid_persisted_evidence``
  Identity, status, numeric range, or structural metadata is malformed.

Only ``advisory_ready`` has ``eligibility=eligible``, a final score, and score
components.  All other dispositions remain persisted and inspectable with an
explicit block reason and missing-evidence count.

When more than one condition applies, version 1 uses this fail-closed
precedence: invalid persisted evidence, unsupported family, insufficient
evidence, stale provenance, exact experiment duplicate, then advisory-ready.
Within exact experiment conflicts, pending precedes running/paused, which
precedes completed; experiment ID is the deterministic tie-breaker.  This is
an explicit policy order, not lexical status or database row order.

Scoring
-------

The implemented components are exactly the nine Step 4 components:

* leader quality;
* inference accuracy;
* evidence strength;
* neutral balance;
* structural proximity;
* parameter preference;
* reciprocal source rank;
* horizon-change penalty;
* relative-mutation penalty.

Weights, normalization, missing-neutral behavior, bounds, precision, and
explanations come directly from Step 4.  No profitability, expected return,
Sharpe ratio, drawdown, transaction cost, class-stability, continuation, or
checkpoint score was added because Phase 4B Step 1 has no new authoritative
formula for those concepts.

Persistence and concurrency
---------------------------

Migration 034 adds:

* ``experiment_recommendation_evaluation_run``;
* ``experiment_recommendation_evaluation_result``;
* ``experiment_recommendation_evaluation_component``.

Runs retain policy, evaluator/scoring versions, canonical run/snapshot
identity, filters, counters, and lifecycle.  Results retain recommendation and
source provenance, canonical evaluation/evidence identity, eligibility,
disposition, nullable score, reason, explanation, counts, and deterministic
ordinal.  Components retain the ordered Step 4 calculation.  Foreign keys are
restrictive.

The runtime role may select, insert, and perform the narrow completion update
on run rows.  Result and component rows are SELECT/INSERT only; runtime UPDATE
and DELETE are revoked.  Result plus components use one short transaction.
An exact retry compares every persisted result field and the complete ordered
component set.  A mismatch fails with
``recommendation_evaluation_retry_mismatch``.

Run failure uses append-only partial-run evidence.  Each result and its
components are atomic, but a later recommendation failure does not erase
earlier successful results.  The run becomes terminal ``failed`` and retains
its counters and error.  Exact retries may verify and reuse an already
persisted result, but no new result may be added to a failed run; retrying the
same failed service request reports ``recommendation_evaluation_run_failed``
without changing the partial history.  A changed authoritative input or policy
has a different run identity and creates separate history.

Concurrent identical requests take a short transaction advisory lock derived
from the complete canonical identity, use the bounded persisted hash as a
candidate lookup, and then compare the complete canonical text exactly.
The canonical text remains authoritative and is stored unchanged; the hash is
never accepted as semantic equality.  A hash collision can only serialize two
requests briefly, after which distinct canonical identities persist as
distinct rows.  This avoids PostgreSQL B-tree tuple-size limits for broad
evidence snapshots.  Result uniqueness is scoped to run plus recommendation;
the repository applies the same hash-lookup and exact-canonical-conflict check
for result identities.  Ranking snapshots use the same collision-safe bounded
strategy because their membership canonical can embed large evaluation
identities.
Completion is idempotent only for identical counters.  No advisory evaluation
uses ``SELECT FOR UPDATE`` on experiment-owned evidence.  Result persistence
holds a short ``FOR SHARE`` lock only on its evaluation-run row so a lifecycle
completion/failure update cannot race the terminal-status check.

Current final-analysis loading joins the source experiment's ``last_model_id``
and ``analysis_scope='final'``.  Migration 016's partial unique index on
``(experiment_id, model_id)`` for final scope guarantees at most one such row;
checkpoint analyses and final analyses for other models cannot duplicate one
recommendation in the evaluation input.

CLI
---

Evaluation commands are explicit and mutually exclusive with every other
scheduler command::

  LSTM_Release --evaluate-experiment-recommendations \
    --recommendation-scan-id=42

  LSTM_Release --evaluate-experiment-recommendation=81 \
    --recommendation-evaluation-dry-run

  LSTM_Release --list-experiment-recommendation-evaluations \
    --recommendation-scan-id=42 \
    --recommendation-evaluation-disposition=advisory_ready

  LSTM_Release --recommendation-evaluation-status=15
  LSTM_Release --explain-recommendation-evaluation=15
  LSTM_Release --list-experiment-recommendation-evaluation-runs
  LSTM_Release --recommendation-evaluation-run-status=7

``--recommendation-evaluation-policy`` accepts the existing Step 4 scoring
policy key/value syntax; it does not introduce additional score parameters.
``--recommendation-evaluation-limit`` is positive and repository-bounded to
1000.  Dry-run loads and computes evidence but creates no run, result, or
component.

Machine events expose ``recommendation_semantic_hash`` and
``evaluation_identity_hash`` as distinct fields, percent-escape text, and
distinguish ``NULL`` from zero.  Every evaluation event includes
``experiment_created=false``,
``experiment_queued=false``, and ``scheduler_modified=false``.  Human output
uses control-safe readable text, includes positive-component and penalty
summaries when scoring occurred, and repeats the advisory-only disclaimer.

Operational safety and testing
------------------------------

Evidence loading uses read-only transactions.  Writes are confined to the
three recommendation-evaluation tables.  Migration and repository/service
tests create isolated schemas, use disposable identifiers, compare before and
after evidence digests, and remove the exact schema.  The production migration
is not automatically applied by this implementation task.

Deferred work
-------------

Later Phase 4B work may define additional evaluation evidence or operator
workflows through versioned policies and architecture review.  Profitability,
recommendation-to-experiment conversion, approval automation, scheduler
ownership, queueing, and autonomous research remain explicitly deferred.

References
----------

* ``docs/architecture/Volume_I_Foundation.md``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
* ``docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md``
* ``docs/architecture/adr/ADR-0002-deterministic-experiment-identity.md``
* ``docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md``
* ``docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md``
* ``docs/Phase4AExperimentRecommendationScoring.rst``
