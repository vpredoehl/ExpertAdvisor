# Campaign Operations Phase 3 — Durable Dispatch

Status: Implemented for isolated integration verification only
Architectural phase: Phase E
Last revised: 2026-07-25

## Scope

Campaign Operations Phase 3 is exactly architectural Phase E, Durable
Dispatch and Atomic Lifecycle Handoff. It starts with the Phase 2 `ready`
request and `held` reservation. It does not implement production enablement,
cancellation/reconciliation (Phase F), completion (Phase G), archival (Phase
H), scheduler claiming, worker launch, or process supervision.

## Transactions and lock order

Acquisition is one short transaction. It locks both potentially relevant
authorization semantic domains in deterministic order, then budget, campaign,
reservation, and request. It revalidates the exact active dispatch grant,
active budget head, held reservation, request version, absent binding, lease
availability, and disabled production flag. A compare-and-set moves
`ready` to `dispatching`, increments the version once, stores only a digest of
the opaque lease token, uses PostgreSQL time for expiry, and appends immutable
attempt and audit evidence before commit.

Handoff is a separate caller-owned transaction. It reacquires and retains the
same levels 1–5, validates the Phase 4D materialization, acquires the
established sorted Phase 5 proposal/review/execution domain, and only then
classifies downstream evidence. This closes direct Phase 5 overlap before
evidence can authorize creation or adoption. It calls
`LaunchRecommendationCampaignInTransaction`; it does not duplicate lifecycle
SQL or open a nested transaction. The launch reacquires the same transaction
locks and continues through the established activation and experiment domains.

The transaction inserts every ordered member binding and permanent V1 control
owner, commits the full held reservation, moves `dispatching` to `bound`,
clears the lease, and appends outcome/audit evidence. Deferred constraints
reject a bound request unless cardinality, membership, control ownership,
settlement, and outcome evidence are complete at commit.

## Evidence classification and adoption

No Phase 5 evidence follows the create-and-bind path. Partial, paused-only,
progressed, or causally mismatched evidence fails closed with a stable
reconciliation-required outcome and is not repaired. A complete existing
Campaign Operations binding is reloaded and validated before any Phase 5
invocation.

Exact complete existing `pending/train` work is adoption, not replay.
Adoption requires an independently active
`adopt_existing_pending_and_control` authorization locked before the budget
domain, exact complete Phase 4D-to-Phase 5 provenance, no progressed member,
and no control-owner collision. The binding and owner disposition records
adoption distinctly.

## Replay and uncertain results

Acquisition and handoff retries begin with authoritative lookup. A fully consistent
binding/request/reservation/owner/attempt/outcome/Phase 4D/Phase 5 chain is
returned without invoking Phase 5 again. SQLSTATE `40001` and `40P01` retry
the complete affected transaction from the service boundary with a bound of
three attempts and jitter.
Retry exhaustion is the stable
`transient_database_retry_exhausted`/`dispatch_retry_exhausted_<SQLSTATE>`
failure; arbitrary SQL errors propagate and are not treated as retryable.
An unknown commit result is recovered only from a complete authoritative
binding. Proven absence permits only the same whole-operation retry;
partial, stale, contradictory, or ambiguous evidence fails closed. Phase 3
itself does not repair projections. Campaign Operations Phase 4 implements
the separate Phase F cancellation, reservation release, lease-expiry
observation, and bounded recovery authority.

## Privilege and execution safety

Migration 048 creates separate NOLOGIN dispatcher and transactional Phase 5
capabilities. Neither is granted to `pqxx` or to any login principal. PUBLIC,
`pqxx`, and unrelated roles receive no new Phase 3 writes. The handoff role
has only the reads, column-scoped established Phase 5 writes, Phase 3 inserts,
and guarded transition functions needed by the atomic transaction. It has no
scheduler table, claim, attempt, process, or worker capability.

The only executable adapter accepts one exact request and requires both an
exact connected database name beginning with
`expertadvisor_campaign_operations_phase3_test_` and the literal
`I_UNDERSTAND_PHASE3_TEST_ONLY` acknowledgement. It does not poll, launch,
signal, or change scheduler configuration. Verification fault hooks can be
supplied only through this isolated adapter; there is no production CLI,
configuration, or database retry-control surface. `production_dispatch_enabled`
remains constrained to false. ADR-0016 remains an unsatisfied production gate.
