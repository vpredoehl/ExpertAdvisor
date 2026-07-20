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

Step 5b authoritative-evidence integration
------------------------------------------

Step 5b supplies the read-only integration without changing the Step 5a domain
boundary.  One PostgreSQL ``REPEATABLE READ``, ``READ ONLY`` transaction loads
authoritative persisted evidence: the exact materialization through the Phase
4D production loader, its exact campaign approval, and the evidence underlying
lifecycle and workflow consistency from the existing Phase 5 campaign-status
projection.  Step 5b builds a non-authoritative, point-in-time assessment from
that evidence.  It does not reconstruct a second lifecycle classifier.

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

Step 5c pure outcome policy
----------------------------

Step 5c adds only a database-free policy and interpretation contract over one
immutable Step 5a assessment.  It does not reopen source or result evidence,
recompute a metric or delta, or change any Step 5a lifecycle, consistency,
member-outcome, comparison, diagnostic, or aggregate classification.  Evidence
classification and policy judgment remain separate fields.

Policy contract version 1 has an explicit canonical identity.  Its default
minimum comparable-member evidence coverage is one ``succeeded_comparable``
member, and its required metric rules are ``inference_accuracy`` and
``leader_score``, each with the explicit ``higher_is_favorable`` direction.
Required rules are sorted by metric identity and duplicates are rejected.  A
caller may construct another version-1 policy with a different positive
minimum or deterministic set of metric-direction rules; changing the policy
changes both policy and decision identity.  The canonical policy binds the
contract version, coverage minimum, sorted metric identities and directions,
fixed all-member consistency, terminal, and successful comparability
requirements, favorable-interpretation requirement for operator-review
eligibility, and fixed non-authorizing semantics.

``minimumComparableMemberCount`` is only an advisory evidence-coverage
threshold.  It is not statistical significance, confidence, causal evidence,
profitability evidence, a guarantee of repeatability, or campaign success.  No
materiality threshold or statistical, causal, profitability, or weighted
campaign score is inferred.

For every exact member and required metric, the policy output preserves the
Step 5a comparison classification and delta as optional evidence, then records
a separate ``favorable``, ``neutral``, ``unfavorable``, or ``not_evaluable``
judgment.  It never subtracts result and source values again.  Positive and
negative meaning is determined only by the versioned metric-direction rule;
an exact zero remains neutral.

Evidence sufficiency is separate from campaign interpretation.  Inconsistent,
not-ready, cancelled, context-changed, metric-gap, missing-source,
missing-result, unavailable, unsupported, missing-required-metric, and
minimum-comparable-member conditions remain explicit conservative reasons.
A terminal failure is adverse lifecycle evidence rather than a scientific
metric gap.  Thus an all-failed campaign is interpreted ``unfavorable`` even
though it lacks the minimum comparable-member evidence coverage; a failed
member mixed with complete favorable evidence is ``mixed``.  Cancellation,
inconsistency, unfinished lifecycle, or non-comparable successful evidence
makes the campaign interpretation ``inconclusive``.  Complete comparable
metric judgments yield ``favorable``, ``neutral``, ``unfavorable``, or
``mixed`` without declaring the campaign successful.

Follow-up eligibility is only ``eligible_for_operator_review`` when every
exact member is a comparable successful outcome, the minimum evidence coverage
is met, all required metrics are comparable, and the campaign interpretation
is ``favorable``.  Neutral, unfavorable, mixed, and inconclusive campaigns are
``not_eligible``; sufficient evidence alone is not enough.  Failure,
cancellation, incompleteness, inconsistency, context change, or a metric gap
also leaves it ``not_eligible``.  This is an advisory screening result, not a
recommendation, approval, or authorization: the immutable result always has
``follow_up_authorized=false`` and ``follow_up_authorizing=false``.

Decision contract version 1 binds the complete policy canonical text and hash,
the exact Step 5a assessment canonical text and hash, complete campaign
approval and materialization identities, ordered member identities including
the expected experiment, assessment lifecycle/consistency/outcome/diagnostics,
evidence classifications, metric judgments, interpretations, reasons, counts,
and explicit non-authorization result.  The complete upstream assessment
canonical already binds Step 5a provenance; the direct fields are deliberately
included as well because the decision carries and exposes those identities and
assessment classifications.  ``observed_at`` remains display metadata and is
excluded through the upstream assessment identity.  Unsupported policy or
assessment contracts, invalid enums, duplicate policy metrics, duplicate or
ambiguous top-level member identities, malformed identity/hash pairs, and
invalid counts fail closed.

Step 5c adds no repository, service, formatter, CLI, migration, privilege,
table, sequence, lock, write, scheduler call, worker behavior, experiment or
campaign mutation, persistence, operator-review event, follow-up generation,
approval, queueing, activation, retry, continuation, or automatic action.  The
existing Step 5b CLI continues to report only the non-authoritative assessment
built from authoritative persisted evidence and does not silently apply outcome
policy.
