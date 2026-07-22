Phase 6D Recommendation Campaign Follow-Up Proposal Governance Ratification
===========================================================================

Purpose and distinct authority
------------------------------

Phase 6C and Phase 6D answer different governance questions.

Phase 6C asks whether one exact persisted Phase 6B follow-up proposal is
acceptable on its merits for possible advancement.  An identified reviewer
records one immutable ``approved`` or ``rejected`` administrative review.

Phase 6D asks whether, given that exact approved review and its complete
authoritative provenance, an authorized governance actor ratifies advancement
of the reviewed proposal into the next separately controlled phase.  The only
positive Phase 6D decision is ``ratified``.  Ratification is immutable
governance evidence; it does not grant any capability in Phase 6E.

Eligibility and separation of duties
------------------------------------

Only one exact persisted, fully reloadable Phase 6C ``approved`` review is
eligible.  A rejected, missing, malformed, stale, unsupported, or
identity-inconsistent review fails closed.  The ratifier identity must differ
byte-for-byte from the Phase 6C reviewer identity.  Self-ratification is
rejected by pure construction, service and repository validation, a database
check constraint, and the provenance trigger.

The ratifier acts under the fixed version-1 role
``follow_up_governance_ratifier``.  Callers cannot select or inject a different
role.  Phase 6D never rebuilds the Phase 6A proposal or recomputes Phase 5
assessment or policy truth; validated Phase 6C reload transitively validates
the exact persisted Phase 6B proposal and embedded Phase 6A identity.

Immutable domain contract
-------------------------

``RecommendationCampaignFollowUpProposalRatification`` is a pure,
database-free, non-assignable contract-version-1 value.  It binds:

* the positive generated Phase 6C review-event ID;
* exact Phase 6C review version, canonical text, tagged hash, ``approved``
  decision, and reviewer identity;
* exact reviewed Phase 6B proposal ID and Phase 6A version, canonical text,
  and tagged hash;
* the fixed ``follow_up_governance_ratifier`` authority role;
* the Phase 6D ``ratified`` decision;
* a distinct ratifier identity; and
* a non-empty ratification basis.

Reviewer and ratifier identities use the stable 1--128 byte governance-actor
syntax: ASCII alphanumeric first, followed only by ASCII alphanumeric or
``._@:/+-``.  Ratification basis is 1--4096 bytes of valid UTF-8, preserves
its exact bytes, requires non-whitespace content, and rejects NUL, DEL, and
disallowed control bytes.

Canonical identity
------------------

The canonical grammar begins with
``experiment_recommendation_campaign_follow_up_proposal_ratification_v1``.
It encodes the exact review and proposal provenance, reviewer, approved-review
eligibility, fixed authority role, ``ratified`` decision, ratifier, basis,
``separation_of_duties_required=true``, and the advancement-only governance
meaning.  It also fixes ``phase_6e_capability_granted=false`` and all negative
operational authority fields.

Delimiter-capable values use byte-length framing.  Decimal integers and byte
lengths use the classic locale.  Canonical text is authoritative and bounded
to 2,113,536 bytes; the tagged FNV-1a hash is only an accelerator.  Generated
ratification-event ID and timestamp remain persistence metadata outside
identity.

Persistence, replay, and concurrency
------------------------------------

Migration ``044`` adds the append-only
``experiment_recommendation_campaign_follow_up_ratification_event``
table.  Restrictive foreign keys bind the exact review and proposal.  Fixed
checks enforce the approved upstream decision, ratified Phase 6D decision,
authority role, and separation of duties.  An invoker-rights trigger with a
pinned safe search path verifies the complete Phase 6C/6B provenance,
reviewer identity, eligibility, and distinct ratifier.  At most one
ratification exists per exact review and proposal.

The repository validates the pure value, takes a transaction-scoped advisory
lock over the review-event conflict domain, reloads the exact Phase 6C/6B
chain, and inserts atomically.  The first valid write returns ``recorded``;
exact replay returns ``existing_identical``.  A changed ratifier, basis, role,
upstream identity, or canonical payload conflicts.  Equivalent concurrent
writes converge; conflicting writes produce one winner and one deterministic
conflict.  Canonical equality remains authoritative over hash equality.

Lookup by ratification-event, review-event, or proposal ID and bounded list
operations are read-only.  Hydration rebuilds the immutable value and
revalidates the upstream chain.  Reads do not advance sequences or acquire
advisory or tuple locks.

Service boundary
----------------

``RatifyRecommendationCampaignFollowUpProposal`` accepts only the exact
review-event ID, expected review identity hash, ratifier identity, and
ratification basis.  It validates request syntax, starts one transaction,
loads the exact review, verifies expected identity and approved eligibility,
enforces separation of duties, constructs the pure value once, persists it,
and returns the authoritative result after commit.  The authority role and
decision are fixed by the contract, not supplied by the request.

No main-program CLI is added in Phase 6D.

Privileges and explicit non-authorities
---------------------------------------

Runtime role ``pqxx`` receives only table ``SELECT``, payload-column
``INSERT``, and sequence ``USAGE``.  It cannot insert generated fields or
update, delete, or truncate ratifications.  ``PUBLIC`` and runtime receive no
trigger-function execution.  Privilege verification treats NULL ACLs through
PostgreSQL defaults, including lowercase sequence object type ``s``.

Ratification does not activate or execute anything; authorize follow-up or
campaign execution; queue or schedule work; signal or start the scheduler;
launch workers; create or modify experiments, models, recommendations, or
continuations; declare campaign success or profitability; or claim
statistical validation.  Phase 6E remains a separate authority because any
later capability requires its own accepted contract, transaction, safety
rules, and explicit consumer.

Verification and operational safety
-----------------------------------

Focused tests cover the fixed golden identity hash, locale independence,
valid UTF-8, DEL and hostile-control rejection, approved-only eligibility,
fixed role, self-ratification refusal, distinct-actor success, exact replay,
changed-payload conflict, hash collisions, concurrency, rollback, malformed
storage, append-only least privilege, NULL-ACL fallback, safe trigger context,
read-only queries, clean install, upgrade from 043, repeatability, and
unchanged Phase 6A/6B/6C, experiment, and scheduler evidence.
