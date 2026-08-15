Phase 4D Step 4 — Approved Campaign Materialization
===================================================

Purpose and boundary
--------------------

Step 4 translates one explicit ``approved`` Phase 4D campaign decision into
the exact ordered set of Phase 4C conversion proposals selected by the
authorized plan. Materialization is not campaign execution: it creates no
experiment, conversion review, conversion execution, or activation and does
not contact the scheduler or a worker.

Authoritative reconstruction
----------------------------

The command accepts an approval primary key plus explicit operator and reason
text. In one PostgreSQL transaction it validates the approval, parses its exact
canonical planning policy and scope, reloads the ranking snapshot and Phase 4C
workflow evidence, and reconstructs the Step 1 plan and Step 2 review. The
reconstructed approval canonical text and hash must equal the persisted Step 3
evidence. Rejected, stale, malformed, or empty approvals fail before any
proposal insert.

Every selected member then passes through the existing Phase 4C conversion
domain contract. That contract still requires effective manual recommendation
approval, completed valid evaluation and score evidence, one supported
mutation, and an exact source-configuration match. Campaign approval is not a
substitute for Phase 4C eligibility.

When the latest approval explicitly references a score, that exact score must
be completed and valid. When it references no score, materialization selects
the latest completed valid score for the same recommendation deterministically
by score-run ID and then score ID. This supports explicit scoring after an
immutable manual approval without rewriting review history.

Identity and persistence
------------------------

Materialization contract version 1 length-frames variable text and binds the
exact campaign approval; ranking snapshot; planning policy and scope; plan and
review identities; approval decision, reviewer, and reason; explicit
materialization operator and reason; ordered selected-member identities; and
every Phase 4C proposal canonical identity and hash. Timestamps are descriptive
and excluded. Canonical text is authoritative; hashes are accelerators only.

Migration ``041_experiment_recommendation_campaign_materialization.sql`` adds
one immutable manifest per approval and ordered member rows. Each member links
the exact ranking member and Phase 4C proposal. Foreign keys are restrictive.
Invoker-rights enforcement triggers require the manifest's copied approval
fields to equal the referenced approved decision, require every member's
snapshot/rank and proposal provenance to match its source rows, and reject a
transaction whose final member count or one-based ordinal range is incomplete.
Runtime ``pqxx`` has only ``SELECT``, column-limited ``INSERT``, and sequence
``USAGE``; it cannot supply generated IDs/timestamps or mutate history.

Atomicity, replay, and concurrency
----------------------------------

Approval validation, reconstruction, proposal lookup/insertion, manifest
insertion, and every member link share one transaction. Any failure rolls back
new proposals and materialization evidence together. Pre-existing exact Phase
4C proposals are reused and linked explicitly.

A transaction-scoped advisory lock is keyed by approval identity and one
manifest is permitted per approval. Exact replay returns the existing manifest;
changed operator, reason, selected membership, proposal provenance, or approval
evidence conflicts. Proposals are persisted in sorted hash/canonical order to
retain deterministic collision-lock ordering, while manifest order remains
campaign rank order.

An exact retry first validates the existing manifest's authoritative canonical
identity against the immutable approval, ordered member links, and normalized
operator request. It does not reconstruct the current plan on replay because
the proposals created by the original materialization are themselves durable
workflow evidence that would intentionally change a newly planned campaign.
First-time materialization always performs the full stale-evidence
reconstruction described above.

CLI
---

::

   LSTM_Release --materialize-recommendation-campaign \
     --campaign-approval-id=123 \
     --campaign-materialized-by='operator@example' \
     --campaign-materialization-reason='Create exact Phase 4C proposal set.'

Read-only inspection::

   LSTM_Release --show-recommendation-campaign-materialization=1
   LSTM_Release --list-recommendation-campaign-materializations \
     --campaign-approval-id=123 --campaign-materialization-limit=100

Output distinguishes a newly recorded manifest from exact replay and reports
ordered member/proposal links plus exact created/reused proposal counts. It
always reports conversion reviews, executions, activations, experiments,
scheduler, and workers as unchanged.

Verification and deferred work
------------------------------

Use only an explicit disposable non-``LSTM`` database::

   createdb phase4d_campaign_test 2>/dev/null || true
   LSTM_TEST_DB_NAME=phase4d_campaign_test \
     /tmp/ExperimentRecommendationCampaignMaterializationRepositoryTests

Phase 4C proposal review, execution to a paused experiment, and explicit
activation remain separate actions. Phase 4D Step 6 provides an explicit atomic
operator convenience for reviewing every exact persisted member through
ordinary Phase 4C review rows. Campaign execution, automatic review or
activation, background scans, scheduler integration, and worker launch remain
outside Step 4.

Phase 4D Step 5 adds a separate read-only handoff projection over the immutable
manifest membership and current Phase 4C lifecycle evidence. It does not alter
the materialization or advance any linked proposal.
