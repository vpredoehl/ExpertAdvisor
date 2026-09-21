Campaign Operations Phase 5
===========================

Campaign Operations Phase 5 is architectural Phase G: Operational Completion
and Audit Views. Migration ``054`` is additive to the accepted Phase F
baseline. It records one immutable administrative completion decision only
when every accepted prerequisite is proven from authoritative rows at one
serializable evidence point. It does not enable Phase H production dispatch,
change experiment lifecycle, interpret scientific results, refund committed
units, or add scheduler or worker authority.

Complete if settled
-------------------

::

   LSTM_Release --campaign-operations-complete-if-settled CAMPAIGN_ID \
     --campaign-operations-operation-key KEY \
     --campaign-operations-actor ACTOR \
     --campaign-operations-reason REASON --yes

The service locks both authorization domains in stable order, then budget,
campaign, reservations, and requests in the accepted global order. PostgreSQL
``SERIALIZABLE`` retains the required
repeatable snapshot and adds write-skew detection for concurrent cancellation
and reconciliation child rows. Serialization failures, deadlocks, and a
concurrent unique winner are retried as whole transactions. A connection loss
or lost response reconnects, looks up the campaign, and compares the complete
stored canonical before any further append. Proven absence permits a bounded
whole-transaction retry; an outcome that cannot be proved fails closed.

The operation returns ``blocked`` with deterministic blocker codes while any
grant obligation, held or inconsistent reservation, nonterminal request,
lease, incomplete or ambiguous attempt, incomplete binding/control ownership,
nonterminal bound lifecycle, unsettled cancellation, blocking reconciliation
observation, pause gate, or budget inconsistency remains. Missing or ambiguous
evidence fails closed.

One successful operation appends the completion event and its causal audit
reference in the same transaction. Repeating the same campaign, operation
key, actor, and reason returns the original event. A changed replay returns
the deterministic non-mutating ``conflicting_replay`` disposition.
There is no force-complete, update, delete, reopen, supersede, or override
operation.

Classification
--------------

Contradictory or unsettled evidence always blocks before classification.
Settled evidence uses the accepted precedence:

* unbound permanent request failure: ``operational_request_failed``;
* failed plus any other terminal scope: ``mixed_terminal_outcomes``;
* all bound work failed: ``downstream_failure``;
* completed plus cancelled scope: ``terminal_partial_completion``;
* all scope cancelled or never dispatched: ``all_scope_cancelled``;
* all exact members bound and completed: ``all_downstream_completed``.

These map only to the disjoint administrative terminal states. Operational
completion never rewrites a lifecycle result and never means profitability,
statistical quality, recommendation acceptance, or scientific success.

Status and historical truth
---------------------------

::

   LSTM_Release --campaign-operations-completion-status CAMPAIGN_ID

Status is read-only. It reports current operational classification, exact
blockers, the recorded event ID/time/hash and evidence summaries,
cancellation and reconciliation disposition, and current lifecycle evidence
as a separate dimension. Scientific outcome is labeled
``NOT_AUTHORITATIVE_NOT_EVALUATED``. A later lifecycle retry or requeue sets
the post-completion-change display and may raise the current reconciliation
classification, but it cannot mutate or reopen historical completion.

Persistence and privileges
--------------------------

``campaign_operations_completion_event`` has one unique row per campaign,
terminal/classification and accounting checks, and complete canonical evidence
for every authorization head, budget head/arithmetic, ordered reservation
event, request, dispatch attempt/outcome, binding/control owner, cancellation
settlement, reconciliation observation/resolution, and point-in-time member
lifecycle fact,
and a deferred same-transaction audit requirement. Database triggers reject
updates, deletes, and truncation even through the owning migration role.
Completion gates cover every Phase B--F authoritative insert plus guarded
reservation/request updates, including provenance, settlement, outcome, and
reconciliation paths, so no later Campaign Operations fact can reopen a
completed campaign. Owning-service lookup still returns an already committed
identical historical fact without attempting a write.

``campaign_operations_completion_writer`` is a NOLOGIN capability with
column-scoped insert, exact evidence reads, and required lock-function
execution only. It has no completion update/delete, lifecycle mutation,
scheduler, worker, cancellation-settlement, reconciliation-resolution, or
broad experiment privilege. Readers receive only status/evidence reads.
Migration ``054`` grants no capability to ``pqxx`` or another login; runtime
assignment remains a separate reviewed deployment action.
