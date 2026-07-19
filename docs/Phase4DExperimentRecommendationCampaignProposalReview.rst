Phase 4D Step 6 — Materialized Campaign Proposal Review
========================================================

Purpose and authority
---------------------

Step 6 is an explicit operator convenience over the existing Phase 4C proposal
review authority. It applies one ``approve`` or ``reject`` decision to every
exact proposal linked by one persisted Step 4 campaign materialization. It does
not introduce a campaign review disposition of its own: every durable result is
an ordinary append-only
``experiment_recommendation_conversion_review_decision`` row, and the greatest
review-decision primary key remains the authoritative current Phase 4C review.

The sole membership authority is the immutable
``experiment_recommendation_campaign_materialization`` manifest and its
one-based ordered
``experiment_recommendation_campaign_materialization_member`` rows. The command
uses the exact stored ``recommendation_conversion_proposal_id`` values in member
ordinal order. It never reconstructs membership from recommendations, rankings,
current policy output, proposal attributes, or Phase 4C list results.

Validation and atomicity
------------------------

The existing Step 4 materialization loader and Step 5 immutable-input validator
together validate the manifest version, canonical/hash pair, selected count,
member hashes, duplicate links, and exact ordinal shape. The Step 5 handoff
derivation then validates every exact linked
proposal and its current Phase 4C review, execution, activation, and experiment
evidence. Missing proposals, malformed review rows, inconsistent provenance,
unsupported evidence, duplicate proposal links, zero members, and campaigns
over the 1000-member safety bound fail closed.

For a write, one PostgreSQL transaction first loads immutable membership, takes
only the existing proposal-review transaction locks in sorted proposal-ID order,
and then reloads and validates all current evidence. No review row is inserted
until every member is eligible. All review inserts and the final result share
that transaction, so a failure rolls back the complete campaign operation.
There is no nested or read-only write transaction, experiment lock, scheduler
lock, polling loop, or background worker.

Idempotence and conflicts
-------------------------

Operation contract version 1 canonically binds the materialization identity,
decision, normalized operator, and normalized reason. Its deterministic Phase
4C request ID contains the materialization primary key, contract version, and
canonical identity hash. Each generated Phase 4C row stores that request ID plus
the same operator and reason, so campaign scope and the exact per-proposal row
mapping are reconstructable without a new Phase 4D table.

An exact retry succeeds as ``already_satisfied`` only when every member's
greatest-ID review is the exact generated operation row. It inserts nothing. An
opposite authoritative decision conflicts. A same-decision row from another
manual request also conflicts because it does not prove this campaign operation.
A mixture of exact already-satisfied members and unreviewed members is rejected
as a partial-operation conflict; it is never silently filled. Existing history
is never updated or deleted.

Dry run and CLI
---------------

::

   LSTM_Release \
     --review-recommendation-campaign-materialization=MATERIALIZATION_ID \
     --campaign-proposal-review-decision=approve|reject \
     --campaign-proposal-review-operator=TEXT \
     --campaign-proposal-review-reason=TEXT \
     [--dry-run] --yes

The distinct ``campaign-proposal-review`` spelling avoids collision with the
existing read-only Step 2 ``--review-recommendation-campaign`` command and the
Step 3 campaign-approval review metadata. The parser requires one positive
materialization ID, a strict decision, non-empty normalized operator and reason,
and ``--yes`` for a write. Duplicate command/metadata forms and combinations
with any other standalone command fail before database dispatch.

``--dry-run`` uses one repeatable-read read-only transaction, performs the full
manifest, membership, proposal, workflow, idempotence, and conflict validation,
and acquires no advisory lock. It inserts no row and advances no sequence.

Machine output and Step 5
-------------------------

One ``RECOMMENDATION_CAMPAIGN_PROPOSAL_REVIEW`` summary precedes member rows in
stored ordinal order. It reports the materialization and operation identities,
decision, percent-encoded operator and reason, counts, result, diagnostics, and
complete safety flags. Each
``RECOMMENDATION_CAMPAIGN_PROPOSAL_REVIEW_MEMBER`` row reports the persisted
member and proposal IDs, previous and resulting greatest-ID review state, insert
status, diagnostics, and contextual safety flags. Optional IDs use ``null``.

Because the durable rows are ordinary Phase 4C reviews, existing Phase 4C show
and history commands observe them directly. The Step 5 campaign handoff
projection observes the whole campaign immediately as
``ready_for_phase4c_execution`` after approval or ``review_rejected`` after
rejection, provided no unrelated workflow inconsistency exists.

Schema and deferred work
------------------------

Step 6 adds no migration, table, sequence, mutable status, or privilege. The
existing Phase 4C request ID, proposal link, decision, operator, reason, and
generated review ID provide the required audit and retry evidence, while the
immutable Step 4 rows provide campaign scope and ordering.

Execution, activation, experiment creation or queueing, experiment status
changes, scheduling, worker launch, polling, repair, subset review, and automatic
progression remain separate and explicitly deferred. The scheduler and workers
do not query or invoke this operation.
