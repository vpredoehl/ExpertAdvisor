Phase 5 Recommendation Campaign Outcome Assessment
==================================================

Purpose and boundary
--------------------

Step 5a defines a pure, immutable, point-in-time domain assessment of the
scientific outcomes associated with an exact recommendation-campaign
materialization.  It is read-only, non-persistent, non-authoritative, and does
not declare campaign success or authorize follow-up work.

Lifecycle, consistency, and comparison
--------------------------------------

Lifecycle is explicit (``not_terminal``, ``succeeded``, ``failed``,
``cancelled``, or ``unknown``).  Consistency is independent
(``consistent`` or ``inconsistent``), and inconsistent downstream evidence is
retained with deterministic diagnostic codes.  Callers can also carry a known
inconsistent lifecycle or workflow observation without changing or discarding
the exact materialized-member identity.  Scientific comparisons are derived
only for consistent succeeded members.  Member outcomes distinguish not-ready,
comparable success, context-changed success, metric-gap success, terminal
failure, terminal cancellation, and inconsistency.

Multi-metric evidence
---------------------

Source and result evidence contain metric collections.  Collections reject a
duplicate metric identity and are canonicalized by metric identity, so input
order does not affect output or assessment identity.  A comparison is derived
for every metric identity in the source/result union.  Member outcome counts
remain separate from metric-comparison and delta counts.

Provenance and deterministic identity
-------------------------------------

Each materialized member may identify its expected downstream experiment.
Succeeded-result evidence is checked against that exact experiment.  Terminal
members require an expected experiment, and expected experiments, observed
result experiments, result models, and typed result identities may not be
reused across members.  Unexpected result evidence is retained and diagnosed
rather than discarded.  Provenance defects remain inconsistent members when
top-level materialization membership is trustworthy; invalid or ambiguous
member identity fails the assessment.  Campaign and materialization inputs
must agree on both the persisted campaign-approval ID and its canonical
identity hash.

Canonical text and hashing bind lifecycle, consistency, diagnostics, canonical
source and result metrics, comparisons, exact provenance, and aggregate member
and metric summaries.  Authoritative campaign and materialization canonical
texts are length-framed into the assessment identity alongside their compact
hashes.  Canonical number formatting is locale-independent.  ``observed_at``
is intentionally excluded from evidence identity.

Non-goals
---------

Step 5a does not provide a repository, database access, persistence, service,
CLI, scheduler, worker, filesystem behavior, campaign-success policy, or
follow-up authorization.
