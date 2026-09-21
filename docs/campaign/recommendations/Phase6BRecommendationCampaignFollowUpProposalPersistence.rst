Phase 6B Recommendation Campaign Follow-Up Proposal Persistence and Preview
===========================================================================

Purpose
-------

Phase 6B durably records an already-built Phase 6A
``RecommendationCampaignFollowUpProposal`` and provides a read-only operator
preview of that persisted value.  It does not build or reinterpret a proposal,
record an operator decision, approve follow-up, activate or execute anything,
create an experiment, or interact with the scheduler.

Schema
------

Migration ``042`` adds an append-only proposal manifest and an ordered member
table.  The manifest stores the exact proposal version/canonical/hash triple,
the exact assessment, policy, and policy-decision triples, complete campaign
and materialization identities, the copied favorable/eligible/non-authorizing
summary, reason, and member count.  The member table stores every ordered
Phase 6A member identity, including nullable expected-experiment ID.

Canonical columns use the bytewise PostgreSQL ``C`` collation.  The proposal
canonical value retains the Phase 6A 1 MiB limit.  Tagged hashes have format
checks but remain accelerators.  Exact canonical equality is authoritative.
The pair ``(proposal_identity_hash, collision_ordinal)`` is unique, permitting
same-hash/different-canonical values while the repository prevents duplicate
exact canonicals under one hash-scoped transaction advisory lock.

Restrictive foreign keys and invoker-rights insert triggers verify the exact
immutable Phase 4D campaign approval, materialization, and ordered
materialization-member provenance copied into the proposal.  A deferred
constraint trigger requires a complete contiguous member set at commit.
Runtime ``pqxx`` access is limited to ``SELECT``, column-scoped payload
``INSERT``, and sequence ``USAGE``.  It has no ``UPDATE``, ``DELETE``,
``TRUNCATE``, generated-ID, or generated-timestamp capability.

Repository and immutable reload
-------------------------------

``PersistRecommendationCampaignFollowUpProposal`` accepts only the immutable
Phase 6A type.  It locks the tagged-hash bucket, compares exact canonical text,
returns ``existing_identical`` for an exact retry, or inserts one manifest and
all members in the caller's transaction.  Collision ordinals are repository
metadata and never enter proposal identity.

Find-by-ID and find-by-identity return
``PersistedRecommendationCampaignFollowUpProposal``.  Its contained proposal
is the original non-assignable Phase 6A value, not a mutable persistence DTO.
A private friendship grants only the Phase 6B hydration builder access to the
Phase 6A constructors.  Hydration reruns supported-version, canonical/hash,
campaign, materialization, count, ordered-member, duplicate-member,
eligibility, favorable-interpretation, false-authorization, summary, and full
proposal-canonical/payload validation.  Malformed or incomplete persisted data
therefore fails closed instead of yielding a partially trusted object.

Read-only preview
-----------------

The preview service loads one persisted proposal in an explicit
``REPEATABLE READ, READ ONLY`` transaction.  It renders the complete
authoritative proposal canonical text, all source identity hashes, summary,
ordered members, persistence timestamp, and explicit negative authority
fields.  Every preview record states that approval, activation, execution,
authorization, queueing, scheduling, workers, and experiment mutation are
false.  Preview does not advance a sequence or acquire an advisory/row lock.

Authority boundary
------------------

Persistence means only that the advisory Phase 6A value is durably available
for inspection.  Neither the row's existence nor
``eligible_for_operator_review`` is an operator decision or follow-up
authorization.  Phase 6B adds no approval/rejection table or API, activation,
execution, scheduler integration, queue, experiment creation, or follow-up
authorization.  Those remain out of scope.

Verification
------------

Focused tests cover migration repeatability and privileges, exact round-trip
and immutable reload, canonical/hash preservation, exact duplicate replay,
malformed duplicate detection, malformed hash/member rejection, full preview,
read-only command behavior, and absence of later-phase schema objects.  Phase
6A and Phase 4D/5 regressions protect the upstream authority boundaries.
