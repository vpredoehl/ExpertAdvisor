Phase 6C Recommendation Campaign Follow-Up Proposal Administrative Review
===========================================================================

Purpose
-------

Phase 6C records one explicit immutable operator decision about one exact
persisted Phase 6B follow-up proposal.  The decision is administrative review
evidence only.  ``approved`` means that an operator approved the persisted
advisory proposal for possible consideration by a later, separately authorized
phase.  ``rejected`` is equally immutable and auditable.

Neither outcome activates or executes follow-up work.  Phase 6C has no
activation adapter, queue, schedule, scheduler signal, worker launch,
experiment creation/modification, recommendation creation, campaign-success
claim, profitability claim, or statistical-validation authority.

Immutable contract and identity
-------------------------------

``RecommendationCampaignFollowUpProposalReview`` is a pure, database-free,
non-assignable value.  Contract version 1 binds:

* the positive persisted Phase 6B proposal ID;
* exact Phase 6A proposal contract version 1;
* exact proposal canonical text and matching tagged hash;
* exactly one ``approved`` or ``rejected`` decision;
* an explicit reviewer identity; and
* non-empty reason text.

The authoritative review canonical grammar begins with
``experiment_recommendation_campaign_follow_up_proposal_review_v1``.  It then
encodes, in fixed order, review contract version; fixed read-only,
administrative-review, and negative authority literals; proposal ID; proposal
version; byte-length-prefixed proposal canonical and hash; decision;
byte-length-prefixed reviewer; and byte-length-prefixed reason.  Decimal
numbers use the classic locale and lengths count bytes.  The canonical text is
authoritative; ``fnv1a64:<16 lowercase hexadecimal digits>`` is only an
accelerator.

Review-event ID and creation timestamp are persistence metadata outside
identity.  Process ID, presentation escaping, scheduler state, and every later
activation or execution state are also excluded.

Validation and failure semantics
--------------------------------

Construction fails closed with stable reasons for unsupported review or
proposal versions, non-positive proposal ID, empty/oversize/NUL proposal
canonical text, proposal canonical/hash mismatch, invalid decision, malformed
reviewer, invalid reason, or oversize review canonical text.  Reviewer identity
is 1--128 ASCII bytes, begins alphanumerically, and otherwise permits only
alphanumerics plus ``._@:/+-``.  Reason text is 1--4096 bytes, valid UTF-8,
contains a byte other than ASCII space, tab, carriage return, or line feed,
contains no NUL or disallowed control byte, and is preserved exactly.

Reload rebuilds the complete typed value from stored payload and compares the
authoritative review canonical and hash.  It then reloads the referenced Phase
6B proposal and compares proposal ID, version, canonical, and hash exactly.
Malformed storage and provenance disagreement fail closed.  The contract and
schema provide no caller-controlled activation, execution, queue, scheduler,
experiment, success, or authorization fields; injected canonical state fails
the full identity comparison.

Schema and repository
---------------------

Migration ``043`` adds one append-only
``experiment_recommendation_campaign_follow_up_proposal_review_event`` table.
It stores the complete proposal binding and review payload, authoritative
review canonical/hash, generated event ID, and generated timestamp.  The
proposal foreign key uses ``ON DELETE RESTRICT`` and an insert trigger verifies
the exact Phase 6B version/canonical/hash provenance.  A unique proposal ID
permits one effective review event per proposal.  Authoritative canonical
columns use PostgreSQL ``C`` collation.

Runtime ``pqxx`` privileges are ``SELECT``, column-scoped payload ``INSERT``,
and review-event sequence ``USAGE`` only.  Runtime cannot provide IDs or
timestamps and has no table-level insert, update, delete, or truncate
privilege.

The repository accepts only the immutable review type.  It validates the
contract, obtains a proposal-ID-scoped transaction advisory lock, verifies the
exact Phase 6B proposal, and inspects the existing event.  The first valid
decision returns ``recorded``.  An exact retry returns
``existing_identical``.  Any different decision, reviewer, reason, or review
identity for that proposal returns the deterministic
``recommendation_campaign_follow_up_proposal_review_conflict`` error.  No event
is updated or superseded.

Read-only presentation
----------------------

Show and list services use explicit ``REPEATABLE READ, READ ONLY``
transactions.  Lists are ordered by descending generated review-event ID and
use a validated 1--1000 limit.  Every event record includes the event and
proposal IDs, proposal hash, review version/hash, decision, reviewer, reason,
timestamp, and these exact safety statements:

Machine records render integer fields with locale-independent decimal digits
and percent-escape delimiter-bearing or non-ASCII payload bytes.

``read_only=true``, ``persisted=true``, ``administrative_review=true``,
``activated=false``, ``execution_authorized=false``,
``follow_up_authorized=false``, ``queued=false``, ``scheduled=false``,
``scheduler_started=false``, ``scheduler_signaled=false``,
``workers_started=false``, ``experiments_created=false``,
``experiments_modified=false``, and ``campaign_success_declared=false``.

No main-program CLI command is added in Phase 6C.  The presentation services
are explicit read-only application seams and never call a scheduler or action
adapter.

Verification
------------

Pure tests cover immutability, both outcomes, canonical grammar, locale
independence, identity sensitivity, malformed input, and fixed negative
authority.  Migration/repository tests cover repeatability, collation, foreign
keys, privileges, generated-column protection, exact round trip, approved and
rejected persistence, sequential and concurrent replay/conflict behavior,
rollback, corruption/provenance rejection, and unchanged Phase 6B, experiment,
and scheduler fixtures.  Presentation tests cover show/not-found, ordered
bounded lists, exact safety output, no sequence advance, no advisory/tuple
locks, and no writes.
