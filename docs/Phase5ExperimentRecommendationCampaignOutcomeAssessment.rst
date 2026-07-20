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

Step 5b authoritative integration
---------------------------------

Step 5b supplies the read-only integration without changing the Step 5a domain
boundary.  One PostgreSQL ``REPEATABLE READ``, ``READ ONLY`` transaction loads
the exact materialization through the Phase 4D production loader, loads its
exact campaign approval, and obtains lifecycle and workflow consistency from
the existing Phase 5 campaign-status projection.  It does not reconstruct a
second lifecycle classifier.

The Step 5a campaign identity is the persisted approval canonical text and
hash.  Its materialization input receives the materialization row's persisted
``approval_identity_hash`` unchanged, so Step 5a verifies the approval-to-
materialization binding.  Exact materialization member ordinal, member ID,
ranking-member ID, recommendation ID, source-experiment ID, proposal ID, and
the execution-linked expected experiment ID flow from the validated status
snapshot.  A status member reported as inconsistent is passed to Step 5a as
``inputConsistency=inconsistent``; Step 5b neither repairs nor normalizes the
upstream evidence.

Scientific evidence
-------------------

Source evidence follows the materialized recommendation's persisted source
experiment, model, and analysis identities.  Source final-inference selection
uses the immutable source invocation canonical text captured by the exact
persisted Phase 4C proposal; it never substitutes the source experiment's
current configuration.  Result evidence follows the exact conversion-
execution experiment, its current linked model, and its final inference and
final analysis rows.  The comparison context is taken from the selected
persisted final inference identity: canonical symbol, prediction horizon,
threshold, persisted window size, the versioned
``inference_eval_result_label_v1`` representation of the
``label_rule_id``/``target_type`` pair, and canonical ``YYYY-MM-DD`` inference
range.

The version-1 Step 5b mapping carries ``inference_accuracy`` and
``leader_score`` with ``NumericDelta`` support.  The source values are the
immutable values captured on the recommendation; result values are the final
analysis values.  Missing analysis or metric evidence remains absent or null,
and changed contexts remain non-comparable.  No metric is recomputed and no
campaign-success threshold is applied.

Other persisted analysis fields were deliberately not mapped in this version:
the immutable recommendation does not retain a directly corresponding value
with the same scientific meaning.  In particular, Step 5b does not reconstruct
predicted-neutral proportions or evidence totals from result counts.  Such
values therefore cannot be mistaken for a valid numeric delta.

CLI and output
--------------

::

   LSTM_Release \
     --recommendation-campaign-outcome-assessment=MATERIALIZATION_ID

The separated option form is also accepted.  The command is intrinsically
read-only, so ``--yes`` and ``--dry-run`` are rejected.  Output begins with one
``RECOMMENDATION_CAMPAIGN_OUTCOME_ASSESSMENT`` record and then one
``RECOMMENDATION_CAMPAIGN_OUTCOME_ASSESSMENT_MEMBER`` record per exact
materialized ordinal.  Records expose contract and identity hashes, the exact
Step 5a aggregate classification and counts, lifecycle, separate input and
final consistency, diagnostics, provenance, contexts, metrics, comparisons,
explicit evidence/context-presence booleans and ``null`` values,
percent-encoded variable text, the materialization's persisted campaign-
approval hash, and explicit no-write/no-scheduler/no-success/no-follow-up
safety fields.

No schema migration or privilege change is required.  The assessment remains
point-in-time and non-persistent; Step 5b inserts, updates, deletes, activates,
queues, retries, and launches nothing and authorizes no follow-up.
