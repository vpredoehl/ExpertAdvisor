Phase 5 Step 1 — Atomic Campaign Conversion Execution
======================================================

Purpose and authority
---------------------

Phase 5 begins with one explicit operational convenience over the existing
Phase 4C conversion-execution authority. Given one persisted Phase 4D campaign
materialization, it atomically invokes the ordinary Phase 4C paused conversion
for every exact member. It adds no campaign execution table, mutable campaign
status, alternate proposal eligibility rule, activation authority, scheduler
consumer, or worker behavior.

Membership and eligibility
--------------------------

The sole member set is the ordered
``experiment_recommendation_campaign_materialization_member`` rows belonging to
the requested ``experiment_recommendation_campaign_materialization``. The
existing materialization loader verifies its version, canonical identity,
hashes, member count, ordinals, and exact proposal links. The existing handoff
projection validates each linked proposal and its greatest-ID Phase 4C review,
execution, activation, and experiment evidence.

First execution requires every member to have no execution and a current
ordinary Phase 4C ``approve`` decision. A pending, rejected, missing, malformed,
duplicate, oversized, or inconsistent member fails the whole request. The
command never infers membership or authorization from recommendation IDs,
ranking output, score, policy, symbol, horizon, or proposal attributes.

Transaction, locking, and retry
-------------------------------

A write uses one PostgreSQL transaction. It locks the exact proposal review-
sequence advisory keys in sorted proposal-ID order, reloads all workflow
evidence, builds the complete pure plan, and only then creates paused
experiments and immutable Phase 4C execution rows. The shared transaction-bound
Phase 4C primitive neither starts nor commits a transaction. Any member failure
rolls back every experiment and execution row from the request.

An exact retry in which all members already have valid Phase 4C executions
returns ``already_satisfied`` and inserts nothing. A mixture of executed and
unexecuted members is rejected as a partial-operation conflict; Phase 5 does not
claim or complete a campaign operation that may have been partially performed
through unrelated direct Phase 4C commands. Concurrent identical campaigns,
overlapping campaigns, direct proposal review, and direct Phase 4C execution
serialize on the same proposal-specific lock domain.

The version-1 operation identity binds the complete immutable materialization
identity and contract version. It contains no timestamp, sequence value, or
unordered input. Individual durable execution identity and idempotence remain
the existing Phase 4C contract.

CLI and dry run
---------------

::

   LSTM_Release \
     --execute-recommendation-campaign-materialization=MATERIALIZATION_ID \
     [--dry-run] --yes

The separated form is also accepted. A positive materialization ID is required,
duplicates and combinations with other standalone operations are rejected, and
``--yes`` is mandatory for a write. Parser failures occur before database
dispatch.

``--dry-run`` performs full validation in one repeatable-read, read-only
transaction. It acquires no advisory locks, inserts no row, creates no
experiment, and advances no sequence. Its result is advisory because another
explicit action may change Phase 4C evidence after the snapshot.

Output and safety boundary
--------------------------

``RECOMMENDATION_CAMPAIGN_EXECUTION`` is followed by one
``RECOMMENDATION_CAMPAIGN_EXECUTION_MEMBER`` row per persisted ordinal. Rows
report materialization and operation identities, exact proposal/review/
execution/experiment IDs, counts, deterministic diagnostics, dry-run state, and
explicit scheduler, worker, activation, queueing, and automatic-progression
safety fields. Optional IDs use ``null`` and variable text uses the production
percent encoder.

Each member's ``authorization_review_decision_id`` is the approving review used
to create its immutable execution. Before first execution this is the current
approving review; on an already-satisfied retry it remains the execution's
original authorizing review even if a later ordinary review reversed the
proposal disposition.

Created experiments are ordinary Phase 4C ``paused/train`` conversion output.
The command does not activate them, change them to ``pending``, queue them,
start the scheduler, launch workers, or automatically invoke another command.
Activation remains the separately invoked Phase 4C action for each execution.

Schema
------

No migration or privilege change is required. Campaign scope comes from the
immutable Phase 4D materialization, while each durable outcome is an ordinary
Phase 4C execution row linked to its approving review and paused experiment.

References
----------

* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
* ``docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md``
* ``docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md``
* ``docs/Phase4CExperimentRecommendationConversionExecution.rst``
* ``docs/Phase4DExperimentRecommendationCampaignHandoff.rst``
* ``docs/Phase4DExperimentRecommendationCampaignProposalReview.rst``
