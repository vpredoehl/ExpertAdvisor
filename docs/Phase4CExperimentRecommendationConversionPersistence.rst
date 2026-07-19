Phase 4C Step 2 Durable Manual Conversion Proposal Persistence
==============================================================

Purpose and boundary
--------------------

Phase 4C Step 2 durably records the exact
``ProposedExperimentSpecification`` produced by the pure Step 1 contract. The
record is an immutable, manually prepared proposal—not an experiment, queue
entry, execution authorization, or scheduler instruction. No Step 2 code
creates an experiment, changes recommendation review state, polls the table in
the scheduler, or launches work.

Schema and audit record
-----------------------

Migration 036 adds
``experiment_recommendation_conversion_proposal``. Each row records:

* recommendation and source-experiment IDs with restrictive foreign keys;
* conversion contract version, changed parameter, and exact source/proposed
  canonical values;
* recommendation, evaluation, evaluation-policy, scoring-policy, and explicit
  review-authorization hashes;
* optional ranking-snapshot identity hash as advisory metadata;
* every field of the proposed ``ExperimentInvocationConfiguration``;
* complete canonical source and proposed invocation snapshots;
* authoritative conversion identity canonical text and its accelerator hash;
  and
* a database proposal ID and creation timestamp, neither of which participates
  in scientific identity.

The runtime ``pqxx`` role receives ``SELECT`` and ``INSERT`` plus sequence
usage. It receives no ``UPDATE`` or ``DELETE`` privilege. Foreign keys use
restrictive deletion behavior, so deleting source provenance cannot cascade
through immutable proposal history. No trigger updates recommendations,
experiments, evaluations, reviews, rankings, or scheduler state.

Exact duplicates, collisions, and concurrency
----------------------------------------------

Canonical conversion text is authoritative. The FNV-1a hash remains only an
indexed accelerator. Canonical text uses PostgreSQL's bytewise ``C`` collation,
so exact equality cannot change with the database locale or a nondeterministic
ICU collation. PostgreSQL btree entries are page-size bounded even when the
underlying text value can be TOASTed, so a unique full-canonical key (alone or
paired with the hash) is unsafe at the supported 1 MiB bound. A unique native
or cryptographic digest would instead make digest equality authoritative and
could falsely collapse a theoretical collision. Insertion therefore uses this
collision-safe protocol:

1. acquire a transaction-scoped advisory lock for the accelerator-hash bucket;
2. compare canonical text exactly;
3. return the existing row when all identity-bearing persisted values agree;
4. otherwise allocate the next collision ordinal within that hash bucket; and
5. insert under a unique ``(hash, collision_ordinal)`` constraint.

Two concurrent identical repository calls therefore converge on one row.
Different canonical texts sharing a hash remain separate rows and return the
stable ``created_with_hash_collision`` outcome. A canonical match with
inconsistent persisted identity fields is reported as corruption rather than
silently accepted. Ranking-only metadata changes do not alter conversion
identity or create another logical proposal; the first recorded advisory
ranking provenance remains the snapshot attached to that proposal.

The advisory lock is a transaction lock: rollback, connection loss, or
statement failure releases it with the transaction. A lock collision between
unrelated accelerator hashes can only serialize those transactions; it cannot
change identity or merge rows. Each repository insertion takes one bucket lock,
so repository calls have no multi-lock ordering cycle and do not lock source
experiment, recommendation, evaluation, ranking, or scheduler rows.

Alternatives considered
-----------------------

The persistence design deliberately does not use a unique full-text B-tree, a
``(hash, canonical_text)`` B-tree, or digest-only uniqueness. The first two are
not valid for the supported long canonical values; the third cannot preserve
exact collision semantics. A separate collision-bucket table or a stored
procedure would move the same serialization protocol into more schema objects
without simplifying identity. The repository-owned hash bucket and ordinal is
therefore the smallest design that supports long authoritative text, exact
canonical duplicates, and distinct same-hash identities. The unique
``(hash, collision_ordinal)`` B-tree also supplies the bucket lookup used for
collision allocation, avoiding a redundant second hash-prefix B-tree. Runtime
code must use this repository entry point; raw ``INSERT`` is a storage
privilege, not an alternative identity API.

Repository boundary
-------------------

``ExperimentRecommendationConversionRepository`` accepts only a completed
Step 1 proposal. It validates structural persistence invariants, faithfully
maps all values, and supports:

* transactional insert with ``created``, ``existing_identical``, or
  ``created_with_hash_collision`` outcome;
* lookup by proposal ID or exact canonical/hash identity; and
* bounded, proposal-ID-ordered listing by recommendation or source experiment.

It does not load evidence and rerun eligibility, infer authorization from rank,
select a recommendation, or duplicate the Step 1 mutation rules. There is no
Step 2 service or CLI because persistence orchestration beyond an already-built
proposal is not needed for that increment. Phase 4C Step 3 adds only proposal
inspection and manual review commands; it does not prepare, regenerate, mutate,
or execute a proposal.

Migration and operational safety
--------------------------------

The migration is additive and repeatable. Migrations are the project’s
source-controlled schema bootstrap path; the tracked production backup is not
regenerated or modified. Repository tests require an explicit non-``LSTM``
``LSTM_TEST_DB_NAME``, create one exact disposable schema, apply migration 036
twice, use owner-only schema cleanup, and verify the parent experiment fixture
remains unchanged.

Deferred work
-------------

Manual CLI proposal preparation, proposal-to-experiment creation, operator
execution commands, budgets, queueing, scheduler capacity, and any automatic
workflow remain deferred. Step 3 provides inspection and append-only manual
review only. Ranking, proposal persistence, and review remain advisory and
cannot authorize execution. Profitability is not inferred.

References
----------

* ``docs/Phase4CExperimentRecommendationConversion.rst``
* ``docs/Phase4CExperimentRecommendationConversionProposalReview.rst``
* ``docs/architecture/Volume_I_Foundation.md``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
* ``docs/architecture/adr/ADR-0001-postgresql-source-of-truth.md``
* ``docs/architecture/adr/ADR-0003-advisory-recommendation-evaluation.md``
* ``docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md``
