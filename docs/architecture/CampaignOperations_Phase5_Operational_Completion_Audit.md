# Campaign Operations Phase 5 — Operational Completion and Audit

Status: Implemented architectural Phase G
Migration: `054_campaign_operations_completion_and_audit.sql`
Authority: accepted specification §22 and §31.7; ADR-0014, ADR-0015, ADR-0017

## Contract

Phase G adds one explicit `complete if settled` workflow and a subordinate
read-only status model. The decision point is a serializable transaction with
the accepted authorization → budget → campaign/completion → reservations →
requests lock order. Both dispatch and adoption/control authorization domains
are acquired in stable order before the budget lock. `SERIALIZABLE` preserves repeatable-read evidence and
adds PostgreSQL SSI detection for cancellation/reconciliation child-row write
skew. No scheduler, lifecycle, recommendation-governance, or scientific
decision is made.

Completion fails closed until every §22 prerequisite is proven. The database
blocker function names the exact pause, authorization, budget, reservation,
request, lease/attempt, binding/control-owner, lifecycle, cancellation, and
reconciliation defect. Contradiction and reconciliation-required evidence
precede terminal classifications.

After blockers are empty, the classification function applies the fixed
precedence: operational request failure, mixed terminal outcomes, downstream
failure, partial completion, all scope cancelled, then all downstream
completed. The event binds campaign identity, budget head and equations, all
authorization heads, budget arithmetic, ordered reservation events, requests,
dispatch attempts and outcomes, bindings and control owners, cancellation
requests and settlements, reconciliation observations and resolutions,
point-in-time lifecycle facts, counts, actor, capability, reason, operation
key, contract version, canonical identity, and hash.

## Persistence and replay

`campaign_operations_completion_event` is unique by campaign. Its insert
trigger reloads classification and every evidence canonical while all locks
remain held. Member, budget, terminal/classification, identity-shape, and
foreign-key checks reject malformed rows. A deferred trigger requires the
matching immutable audit reference before commit.

Exact replay returns the original event without mutation. A changed logical
replay returns `conflicting_replay` without mutation. Database triggers reject
update/delete of completion and
audit history, and no override-shaped column or command exists. A unique
campaign constraint plus serializable whole-transaction retry makes concurrent
attempts converge on one truth. SQL states `40001` and `40P01` retry the whole
transaction boundedly. Connection loss uses a new connection and full-canonical
campaign lookup before another append; ambiguous outcomes fail closed.

Every Phase B–F authoritative mutation table has a completion gate. Guarded
reservation and request updates are gated separately. Exact service replay is
resolved before a write, while changed replay and all genuinely new authority,
settlement, dispatch, cancellation, or reconciliation evidence are rejected.

## Read model

`campaign_operations_completion_status_v1` and the repository projection are
derived from authoritative rows. They expose recorded completion separately
from current lifecycle evidence, cancellation, reconciliation, and exact
blockers. Later lifecycle changes are displayed as
`post_completion_lifecycle_changed`; historical evidence and classification
remain unchanged. Scientific outcome is not evaluated by this component.

## Security and compatibility

The completion writer is a dedicated NOLOGIN role. It receives only exact
reads, lock-function execution, sequence usage, and column-scoped inserts.
Completion rows are not updatable or deletable by runtime roles. The
security-definer evidence functions have pinned schema search paths; their
owner receives only the lifecycle columns needed for read-only evidence.
`pqxx` receives no role membership. Migration 054 is replay-idempotent and
does not rewrite Phase 1–4, recommendation, lifecycle, scheduler, worker,
budget, cancellation, or reconciliation history.

Phase H production enablement remains separate and is not implemented here.
