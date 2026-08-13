# ADR-0012: Campaign budget authority and reservation accounting

Status: Accepted
Date: 2026-07-24
Deciders: Project architecture
Affected volumes: Volume X §§3–7; Volume XII §§5–7
Supersedes: None
Superseded by: None

## 1. Context and problem statement

Campaign Operations needs a deterministic limit before accepting work.
Scheduler capacity, estimated runtime, monetary cost, and profitability are not
authoritative units for the existing Phase 4D/5 workflow. Competing budget
grant, snapshot, and adjustment representations would make availability and
reservation settlement ambiguous.

## 2. Decision

V1 uses one append-only, monotonically versioned budget ledger as the only
budget authority and one exact materialization-member dispatch unit as its
only unit.

For the unique latest ledger head:

- `G` is its resulting authorized total.
- `A` is cumulative units ever reserved.
- `C` is cumulative committed units.
- `L` is cumulative released or expired units.
- `H = A - C - L` is currently held.
- `M = G - C - H = G - A + L` is unallocated.
- `V = M` only for an active ledger head; otherwise `V = 0`.

The invariants are:

```text
A = C + L + H
G = C + H + M
C + H <= G
G, A, C, L, H, M, V >= 0
```

Ledger transitions are:

- `grant`: version 1, positive total, active.
- `amend`: nonzero signed delta from an active head; the result cannot be less
  than `C + H`.
- `revoke`: sets the result exactly to `C + H`, marks the head revoked, and
  makes `V = 0` without releasing or uncommitting anything.
- `supersede`: the only transition permitted after a revoked head; it names
  that head, supplies a new total at least `C + H`, and returns the ledger to
  active.

There is no authoritative budget snapshot or separate adjustment table.
Balances are derived under lock from the ledger and reservation evidence.

A V1 reservation:

- binds one campaign, accepting authorization, exact active budget head,
  system logical operation, complete materialization, member-unit amount, and
  optional semantic expiry;
- has immutable payload and guarded states `held`, `committed`, `released`,
  `expired`, or `reconciliation_required`;
- must equal the exact authoritative materialization member count;
- funds exactly one accepted request;
- commits only with a complete downstream binding set;
- releases or expires only after deterministic proof that no handoff/binding
  committed; and
- never refunds after commitment, even if downstream work later fails or is
  cancelled.

Budget is a limit, not operational authorization or scheduler capacity.

## 3. Rationale and decision drivers

- Phase 4D already supplies an exact immutable member count.
- Integer member units are deterministic and independent of hardware.
- One ledger avoids competing balance authorities.
- Reservations prevent concurrent oversubscription and make uncertain handoffs
  fail closed.

## 4. Consequences

### 4.1 Positive consequences

- Accounting can be recomputed exactly after restart.
- Concurrent reservations cannot exceed the current active budget.
- Revocation and later reauthorization are explicit and auditable.
- Failures cannot silently refund committed work or double-consume replayed
  work.

### 4.2 Negative consequences and trade-offs

- V1 cannot express CPU, accelerator, cloud-cost, or monetary budgets.
- Append-only arithmetic requires under-lock aggregate validation.
- Ambiguous handoff evidence keeps units held until reconciliation.

### 4.3 Risks and mitigations

- Oversubscription: serialize every ledger mutation, acquisition, commitment,
  release, expiry, and completion on one budget domain.
- Torn accounting: pair every guarded reservation transition with its immutable
  event in the same transaction.
- Premature release: require exact no-binding/no-downstream-commit evidence;
  process or PID absence is insufficient.

## 5. Compatibility and migration

Migration 045 contains no budget or reservation tables. A future additive
migration may add the ledger, reservation projection, reservation events, and
narrow roles without modifying production experiments or Phase 4–6 rows.

Existing Phase 1 campaign and authorization identities are the required
foreign-key provenance. No legacy campaign receives an implicit budget.

## 6. Implementation implications

- Ledger mutation takes the level-2 budget lock and then the level-3 campaign
  lock; active authorization is not required merely to administer a budget.
- Reservation/request acceptance takes authorization → budget → campaign and
  inserts the held reservation, acquisition event, ready request, and audit
  references atomically.
- Every held-reservation settlement takes authorization first only when its
  exact cause requires it, then always budget → campaign → reservation →
  request. No path may acquire budget after campaign.
- PostgreSQL transaction time owns expiry decisions.
- Exact replay does not increment `A`; near-exhaustion losers return the stable
  insufficient-unit outcome unless they prove an identical request.

## 7. Verification and operational evidence

- Pure arithmetic and exhaustive ledger/reservation transition tests.
- Golden identity tests for ledger entries, reservations, and transition
  events.
- Migration checks for natural uniqueness, append-only history, guarded
  states, constraints, indexes, ACLs, and safe rerun.
- Independent-connection races for successor ledger entries, acquisition near
  exhaustion, commitment/release, cancellation/release, and completion.
- Corruption tests for every equation and state/event mismatch.
- Negative tests proving budget does not authorize dispatch or reserve
  scheduler capacity.

## 8. Alternatives considered

### 8.1 CPU-time or monetary budget

Rejected for V1 because no accepted deterministic measurement or reservation
contract exists.

### 8.2 Mutable available-balance row

Rejected as a competing authority that cannot independently reconstruct
history or prove replay.

### 8.3 Release committed units after failure or cancellation

Rejected because the budget accounts for accepted downstream handoff, not
scientific success or worker outcome.

## 9. Relationships to other ADRs

- ADR-0010 defines the exact materialization scope.
- ADR-0011 remains the only permission to accept or dispatch work.
- ADR-0013 owns the request and binding that hold or commit a reservation.
- ADR-0015 owns release, expiry, and uncertainty handling.
- ADR-0017 owns budget and reservation privileges.

## 10. References

- [Accepted Campaign Operations specification §§14–15, 18–20](../../../ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md)
- [Volume X](../Volume_X_Research_Automation.md)
- [Volume XII](../Volume_XII_Database.md)

## 11. Revision history

| Date | Change |
|---|---|
| 2026-07-24 | Accepted the single member-unit budget ledger and deterministic held-reservation accounting. |
