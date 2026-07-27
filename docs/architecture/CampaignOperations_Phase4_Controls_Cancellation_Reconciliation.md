# Campaign Operations Phase 4 — Controls, Cancellation, and Reconciliation

Status: Implemented
Architectural phase: Phase F
Last revised: 2026-07-25

## Scope

Phase 4 implements exactly architectural Phase F. It owns append-only campaign
gates, cancellation intent and settlement, lifecycle-delegated cancellation,
deterministic reconciliation observations, and the narrowly safe recovery of
expired Phase E leases. It does not add scheduler signaling, worker/process
control, running-experiment stop authority, production dispatch, completion,
archival, reporting, analytics, budgeting changes, forecasting, optimization,
or autonomous behavior.

## Controls and dispatch ownership

The control stream is a fork-resistant version chain. The first event is
`pause`; transitions then alternate between `resume` and `pause`. Exact
replays return the current identical event and changed or stale replays fail.
The database predicate for future actions is evaluated during request
acceptance and Phase E dispatch selection, lease acquisition, and handoff.
Pause never steals or invalidates a committed dispatch lease.

## Cancellation transactions

Cancellation intent is an immutable operation keyed by campaign, complete
materialization scope, and caller-supplied operation key. Intent and settlement
are separate durable facts. A request target, or a campaign-only target when no
request exists, has one permanent cancellation owner. Distinct operation keys
are rejected under the complete cancellation lock domain rather than creating
independently unresolved obligations.

Unbound settlement uses the global budget → campaign → reservation → request
lock order. With no active lease or downstream evidence it atomically releases
held units, cancels the request, and records transition, settlement, and audit.
An active Phase E lease leaves the request and reservation unchanged while
durable intent waits for expiry. `waiting_for_lease_expiry` is returned only
while PostgreSQL proves the lease is active; an expired lease whose evidence
still prevents settlement returns `reconciliation_required`.

Bound settlement never refunds committed units. After the intent transaction
releases Campaign Operations locks, a lifecycle-owned capability locks each
controlled experiment and records accepted, already-terminal, or
running-not-supported evidence in its own transaction. The cancellation
coordinator then settles only from a complete owner/evidence set. This
preserves the lifecycle boundary and makes termination between calls
replayable. Both unbound and bound settlement append resolutions for every
unresolved observation with the exact corresponding cancellation evidence in
the settlement transaction.

## Reconciliation

Reconciliation detects before it repairs. The observer writes immutable,
canonical observations and exact cursor evidence in request-ID order with a
maximum batch size of 1000. A cursor row supplies the durable batch identity;
every observation has a non-null foreign key to exactly one cursor. Cursor and
membership commit in one transaction before requested recovery. Deferred
database triggers validate the exact member count, run key, request bounds,
ordering endpoint, and empty-batch shape at commit. Replay first
resolves `(run key, prior target)` to the cursor identity and loads only that
identity's members, never a run-key/range approximation or current candidates.
Empty batches are equally durable. The observer has no resolution privilege.

Batch persistence acquires all campaign locks in ascending ID order before all
request locks in ascending ID order. This is the applicable subset of the
global authorization → budget → campaign → reservation → request order and
cannot invert cancellation or recovery. The reconciler receives `EXECUTE` only
on the campaign and request lock helpers needed for this persistence path.

Recovery is owned by a separate capability and accepts only an observation
whose exact expected request version still matches. PostgreSQL time must prove
the lease expired, and both the service and guarded database transition require
no binding, downstream execution, current dispatch-attempt outcome, or expired
reservation. Recovery returns the request to `ready`, retains the held
reservation, appends a recovery outcome and audit, and records a resolution.
If overlapping cursors recorded equivalent request/version/evidence, later
observations prove and reuse that same accepted outcome, append their own
resolution, and do not repeat the request transition.
Canonical expiration timestamps use a fixed UTC representation with
microsecond precision and are independent of PostgreSQL session time zone,
connection settings, and process locale.
Expired held reservations on ready, unbound requests are observed and
delegated to the reservation service; Phase F does not settle them. All other
observations remain fail-closed for their named owning service or an operator.

## Durability, replay, and privileges

Migration 049 makes control and reconciliation history append-only, validates
canonical/hash identities, pins security-definer search paths, and uses
deferred completeness constraints to reject partial externally visible
transitions at commit. Identical control, cancellation, lifecycle, observation,
resolution, and cursor retries reuse authoritative evidence; changed identities
conflict.

Separate NOLOGIN roles own campaign controls, cancellation coordination,
reconciliation observation, recovery, lifecycle cancellation execution, and
lifecycle cancellation writes. The migration grants none of these capabilities
to `pqxx` or any login role. Control status uses the existing
`campaign_operations_reader` boundary and a read-only repeatable-read
transaction. Cancellation receives only the causal-audit and guarded-function
privileges needed to close exact cancellation observations.
Neither recovery nor cancellation can insert a resolution directly. Their
separate security-definer functions validate an exact authoritative settlement
or dispatch outcome and its audit, derive capability ownership, and persist a
typed causal foreign key.

Cancellation retries keep transaction-derived IDs, replay state, candidates,
and settlement results local to one attempt and publish them only after commit.
Pending unbound replay checks for settlement both before and after taking the
complete cancellation lock domain; the post-lock check makes concurrent exact
replays converge without weakening evidence-conflict detection.
Cursor persistence and each individual recovery use bounded three-attempt
whole-transaction retries with jitter. Each attempt reconstructs its state on
a fresh connection. Broken or in-doubt commit outcomes perform exact durable
cursor or resolution lookup before retrying or surfacing failure.
