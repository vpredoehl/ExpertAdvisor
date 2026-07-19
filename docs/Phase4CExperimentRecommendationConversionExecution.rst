Phase 4C Step 4 Manual Conversion Execution
============================================

Purpose and boundary
--------------------

Step 4 materializes one explicitly approved immutable conversion proposal as
one new experiment. Approval and conversion remain separate durable audit
events. The experiment is created with ``status=paused`` and ``phase=train``;
it is not queued, started, resumed, scheduled, or assigned to a worker.

Authorization and integrity
---------------------------

The repository takes a transaction-scoped advisory lock for the exact proposal,
reconstructs and validates the persisted proposal, and reads the latest review
decision by greatest generated decision ID in the same transaction.
``pending_review`` and ``rejected`` are rejected. Only ``approved`` permits
first-time conversion.

Review writes take the same proposal-specific advisory lock. This preserves the
proposal table's immutable runtime privileges while giving a concurrent review
and conversion a database-defined serialization order. Advisory-lock hash
collisions can only serialize otherwise unrelated proposals; exact proposal IDs
and constraints remain authoritative. A later review reversal cannot erase a
conversion that already committed.

Transaction and idempotency
---------------------------

Migration 038 adds
``experiment_recommendation_conversion_execution``. One short transaction:

1. serializes access for and revalidates the proposal;
2. returns existing exact execution evidence on retry;
3. verifies the latest decision is ``approve``;
4. inserts one paused experiment from the complete proposed invocation; and
5. inserts the immutable execution row linking proposal, approving decision,
   and experiment.

The proposal and experiment references are independently unique. Concurrent
identical requests converge on one experiment. A configuration collision with
an unrelated existing experiment fails atomically rather than attaching the
proposal to unrelated work. Canonical execution text is authoritative and its
tagged hash is only an accelerator.

CLI
---

The explicit standalone commands are::

  --execute-approved-conversion-proposal=PROPOSAL_ID
  --conversion-proposal-execution-status=PROPOSAL_ID

Creation output identifies the proposal, approving decision, execution audit
row, approval action, and paused experiment. Replay reports the same durable
experiment without creating another. All records state that no experiment was
queued and scheduler state was not modified.

Deferred work
-------------

Starting or resuming the paused experiment, queueing, scheduler capacity,
worker launch, automatic conversion, budgets, and autonomous selection remain
outside Step 4. Existing Phase 3 continuation behavior is unchanged.

References
----------

* ``docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
* ``docs/Phase4CExperimentRecommendationConversionPersistence.rst``
* ``docs/Phase4CExperimentRecommendationConversionProposalReview.rst``
