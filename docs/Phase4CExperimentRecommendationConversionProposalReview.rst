Phase 4C Step 3 Manual Conversion Proposal Review
=================================================

Purpose and boundary
--------------------

Phase 4C Step 3 adds durable, explicit operator review for the exact immutable
conversion proposals created by Step 2. A review references the proposal
primary key. It never reconstructs an identity from a recommendation, rank, or
hash. Approval means only that an operator approved that proposal for possible
conversion in a later phase. It creates no experiment, queues no work, and has
no scheduler integration.

Append-only decision history
----------------------------

Migration 037 adds
``experiment_recommendation_conversion_review_decision``. Each row contains a
generated decision ID, the exact proposal ID, ``approve`` or ``reject``, a
caller-visible request ID, optional operator and reason text, and server
timestamps. The proposal foreign key uses restrictive deletion. Runtime access
is ``SELECT`` plus column-level ``INSERT`` for only the decision payload and
sequence usage; the generated ordering ID and timestamps are not caller-
writable, while ``UPDATE``, ``DELETE``, and ``TRUNCATE`` are revoked. Proposal
rows remain immutable.

The history supports deliberate reversals without erasing prior decisions:

* no decision means ``pending_review``;
* otherwise the decision with the greatest generated decision ID determines
  ``approved`` or ``rejected``; and
* timestamps are descriptive and never break ordering ties.

The generated sequence-ID order is the authoritative administrative history
order; it is not transaction commit order. For example, if one transaction
allocates ID 100 and commits after a transaction that allocated ID 101, ID 101
still defines the disposition after both commits. Sequence gaps caused by
rollback or an idempotent conflict have no meaning. This ordering is not
recommendation ranking or scientific preference.

Idempotency and concurrency
---------------------------

Every manual decision command requires a request ID of 1–128 ASCII bytes. It
begins with a letter or digit; later bytes may also be dots, underscores,
colons, or hyphens. The database uniquely
constrains ``(proposal_id, request_id)``. Repeating the same normalized decision,
operator, and reason returns the original decision as
``existing_identical``. Reusing that request ID with a different payload is a
stable conflict. A new request ID represents a deliberate new decision,
including a reversal.

``INSERT ... ON CONFLICT DO NOTHING RETURNING`` makes the database uniqueness
constraint authoritative. Concurrent identical requests converge on one row.
Concurrent different request IDs both remain auditable; the greater generated
decision ID determines the resulting disposition. Reviews of unrelated
proposals are not globally serialized. Review writes and the Step 4 conversion
operation share a transaction-scoped, proposal-specific advisory lock. This
gives a review-versus-conversion race an explicit order without granting UPDATE
access to the immutable proposal row; a lock-hash collision can only add
serialization and never establishes proposal identity.

Validation and repository boundary
----------------------------------

The pure review domain normalizes surrounding ASCII whitespace in optional
operator and reason text, rejects empty values and embedded NUL bytes, and
enforces 200-byte operator and 2000-byte reason limits. The repository owns SQL,
typed row mapping, exact replay comparison, bounded deterministic lists, current
disposition lookup, and proposal-not-found results. It does not repeat Step 1
conversion eligibility, change a proposal, create an experiment, or contact the
scheduler.

CLI
---

The standalone commands are::

  --approve-conversion-proposal=PROPOSAL_ID
  --reject-conversion-proposal=PROPOSAL_ID
  --conversion-proposal-review-request-id=TOKEN
  --conversion-proposal-review-operator=TEXT
  --conversion-proposal-review-reason=TEXT
  --show-conversion-proposal=PROPOSAL_ID
  --list-conversion-proposal-reviews=PROPOSAL_ID
  --list-conversion-proposals-by-review-status=pending_review|approved|rejected
  --conversion-proposal-review-limit=N

Approval and rejection require a request ID. Operator and reason are optional.
Machine records percent-escape untrusted text and always report
``experiment_created=false``, ``experiment_queued=false``, and
``scheduler_modified=false``. Human confirmations repeat that the review is
administrative only.

Deferred work
-------------

Step 4 provides the separate explicit operation that can materialize an
approved proposal as one paused experiment. Queueing, starting or resuming that
experiment, budgets, scheduler capacity, operational execution authorization,
and automation remain deferred. Ranking remains advisory and cannot authorize
review, conversion, or execution.

References
----------

* ``docs/Phase4CExperimentRecommendationConversion.rst``
* ``docs/Phase4CExperimentRecommendationConversionPersistence.rst``
* ``docs/architecture/Volume_I_Foundation.md``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
* ``docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md``
* ``docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md``
