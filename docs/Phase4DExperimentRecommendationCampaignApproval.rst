Phase 4D Step 3 Explicit Recommendation Campaign Approval
==========================================================

Purpose and boundary
--------------------

Step 3 records one immutable human decision for one exact deterministic
campaign review. The decision is ``approved`` or ``rejected``. It is audit
evidence for possible future campaign execution; it does not create a
proposal, conversion, activation, experiment, scheduler action, or worker.

An operator must explicitly supply the ranking-snapshot ID, the expected
campaign-review identity hash, reviewer identity, reason, and the complete
Step 1 planning policy and scope. Scores, ranks, plan contents, and a zero-
selection review never imply approval.

Stale-evidence protection
-------------------------

The write service opens one PostgreSQL transaction, loads the explicit
completed ranking snapshot and Phase 4C workflow evidence, reconstructs the
Step 1 plan, reconstructs and validates the Step 2 review, and compares the
operator's expected review hash exactly. The approval is inserted in that same
transaction. An identity mismatch, missing snapshot, conflict, validation
failure, or database error rolls the transaction back.

The approval row stores the authoritative snapshot, planning-policy, planning-
scope, plan, and review canonical evidence and hashes; the Step 2 contract
version and complete summary counts; the ordering-verification result; and the
explicit decision, reviewer, and reason. The database timestamp is audit
metadata and is excluded from semantic identity.

Canonical identity and duplicate behavior
-----------------------------------------

Approval contract version 1 length-frames every variable string and binds all
authorization-relevant provenance, summary fields, decision, reviewer, and
reason. The existing recommendation canonical hash is an indexed accelerator;
canonical text is authoritative.

One terminal decision is allowed per exact campaign-review canonical identity.
An identical retry returns the original row as ``existing_identical``. Reusing
the same review with a changed decision, reviewer, reason, or other persisted
payload returns a conflict and never replaces history. A real hash collision
uses a separate collision ordinal. A transaction-scoped advisory lock only
serializes that hash bucket while exact canonical equality decides identity.
Concurrent identical requests therefore converge on one durable row.

A valid zero-selection review may be rejected for audit purposes. It cannot be
approved.

Schema and privileges
---------------------

Migration 040 creates
``experiment_recommendation_campaign_approval``. The ranking-snapshot foreign
key is restrictive. Summary, decision, version, text-size, and zero-selection
constraints fail closed, and every stored identity hash must use the canonical
``fnv1a64:`` plus 16-lowercase-hex form. Canonical columns use the bytewise
``C`` collation.
Exact review-canonical lookup uses a PostgreSQL hash index so the potentially
large authoritative text is not stored in a size-limited btree key; PostgreSQL
still rechecks exact canonical equality.
Runtime role ``pqxx`` receives ``SELECT``, column-limited payload ``INSERT``,
and only sequence ``USAGE``. It cannot insert the generated ID or timestamp,
and receives no ``UPDATE``, ``DELETE``, or ``TRUNCATE`` privilege. No trigger
modifies an experiment or any prior recommendation/conversion evidence.

CLI
---

Approve one exact reconstructed review::

  LSTM_Release --approve-recommendation-campaign \
      --campaign-ranking-snapshot=42 \
      --campaign-review-identity-hash=fnv1a64:... \
      --campaign-reviewer='operator@example' \
      --campaign-review-reason='Reviewed bounded campaign evidence.' \
      [campaign policy options]

Reject uses the same evidence and metadata with
``--reject-recommendation-campaign``. Approval and rejection are mutually
exclusive. All planning policy and scope options must reproduce the reviewed
Step 2 request exactly.

Read-only inspection commands are::

  LSTM_Release --show-recommendation-campaign-approval=ID
  LSTM_Release --list-recommendation-campaign-approvals \
      [--campaign-approval-decision=approved|rejected] \
      [--campaign-approval-limit=N]

Machine output distinguishes recorded, replayed, stale, conflict, invalid,
not-found, and operational-failure outcomes. Write output says precisely that
approval evidence was recorded and always reports campaign execution,
proposal/experiment mutation, activation, scheduler, and worker side effects
as false. Inspection output is marked ``read_only=true``.

Isolated PostgreSQL verification
--------------------------------

The repository test requires an explicit non-production database, refuses the
database name ``LSTM``, owns one per-process schema, reapplies migration 040,
and removes that schema::

  LSTM_TEST_DB_NAME=phase4d_campaign_test \
      /tmp/ExperimentRecommendationCampaignApprovalRepositoryTests

It verifies exact replay, conflict, concurrent convergence, rollback, runtime
immutability, restrictive provenance, unchanged experiment timestamps/counts,
unchanged Phase 4C audit counts, and no sequence activity outside the new
approval sequence.

Deferred work
-------------

Campaign execution, bulk proposal creation/review, activation, profitability-
aware authorization, background scans, and scheduler integration remain
outside Step 3. Campaign approval records explicit human authorization for one
exact deterministic campaign review. It does not create or modify experiments
and does not execute the campaign.

Phase 4D Step 4 may consume an ``approved`` row only to materialize the exact
selected members as Phase 4C conversion proposals. Proposal review, execution,
activation, experiment creation, and scheduling remain separate boundaries.

References
----------

* ``docs/Phase4DExperimentRecommendationCampaignPlanning.rst``
* ``docs/Phase4DExperimentRecommendationCampaignReview.rst``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
