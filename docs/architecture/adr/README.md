# Architecture Decision Records

## 1. Purpose

Architecture Decision Records preserve the context, decision, alternatives,
and consequences of material architectural choices. ADRs explain why the
architecture has its current shape; domain volumes describe the resulting
ongoing contract.

## 2. Numbering and filenames

- Numbers are four digits, monotonic, and never reused: `ADR-0001`.
- Filename format is `ADR-NNNN-short-kebab-title.md`.
- A reserved or rejected number remains consumed.
- Renaming the title does not change the number.
- The next ADR number is one greater than the highest number present here.

## 3. Status values

| Status | Meaning |
|---|---|
| `Proposed` | Under review; not implementation authority. |
| `Accepted` | Approved and governing from its decision date. |
| `Rejected` | Considered and declined; retained as rationale. |
| `Deprecated` | Still historical, but no longer recommended for new work. |
| `Superseded by ADR-NNNN` | Replaced by a later accepted decision. |

Accepted ADRs are immutable decision records. Editorial corrections may fix
links or spelling without changing meaning and must be noted in revision
history. A changed decision receives a new ADR that supersedes the old one.

## 4. Changes requiring an ADR

Volume I §16.1 is authoritative. Typical triggers include identity semantics,
durable ownership, schema strategy, transaction/concurrency rules, scheduler
scope, lifecycle transitions, recommendation conversion, security/privileges,
and breaking public-interface changes.

## 5. Required sections

Every ADR MUST contain:

1. Title and metadata: number, status, date, deciders, affected volumes.
2. Context and problem statement.
3. Decision.
4. Rationale and decision drivers.
5. Consequences, both positive and negative.
6. Compatibility and migration.
7. Verification and operational evidence.
8. Alternatives considered.
9. References.
10. Revision history.

The [ADR template](ADR_TEMPLATE.md) is normative for new records.

## 6. Proposal workflow

1. Copy `ADR_TEMPLATE.md` to the next numbered filename.
2. Set status `Proposed` and identify affected volume sections.
3. Describe the problem independently from the preferred solution.
4. Record alternatives, compatibility, migration, rollback, concurrency,
   permissions, operational safety, and tests.
5. Review against Volume I and all affected accepted ADRs.
6. Record the decision status without deleting rejected alternatives.
7. Update affected volumes and their revision histories when accepted.
8. Implement only after acceptance unless Volume I §16.4 applies.

## 7. Index

| ADR | Status | Decision |
|---|---|---|
| [ADR-0001](ADR-0001-postgresql-source-of-truth.md) | Accepted | PostgreSQL is the durable source of truth. |
| [ADR-0002](ADR-0002-deterministic-experiment-identity.md) | Accepted | Experiment semantic identity uses deterministic canonical content. |
| [ADR-0003](ADR-0003-advisory-recommendation-evaluation.md) | Accepted | Recommendation generation, scoring, and review remain advisory and non-executing. |
| [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md) | Accepted | Scheduler work requires explicit ownership and durable lifecycle claims. |
| [ADR-0005](ADR-0005-manual-recommendation-conversion.md) | Accepted | Explicit approved-proposal conversion creates one paused experiment. |
| [ADR-0006](ADR-0006-phase-6a-follow-up-proposal.md) | Accepted | Phase 6A follow-up proposals are pure, identity-bound, and non-authorizing. |
| [ADR-0007](ADR-0007-phase-6b-follow-up-proposal-persistence.md) | Accepted | Phase 6B persists exact advisory proposals for read-only preview without action authority. |
| [ADR-0008](ADR-0008-phase-6c-follow-up-proposal-administrative-review.md) | Accepted | Phase 6C records one immutable administrative proposal review without action authority. |
| [ADR-0009](ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md) | Accepted | Phase 6D separately ratifies governance advancement after an approved review, with mandatory separation of duties and no operational authority. |
| [ADR-0010](ADR-0010-campaign-operations-ownership-and-scope.md) | Accepted | Campaign Operations owns bounded orchestration over one exact Phase 4D materialization. |
| [ADR-0011](ADR-0011-campaign-operational-authorization.md) | Accepted | One explicit append-only grant chain is the only Campaign Operations permission to act. |
| [ADR-0012](ADR-0012-campaign-budget-and-reservations.md) | Accepted | One member-unit ledger and deterministic reservations own campaign budget accounting. |
| [ADR-0013](ADR-0013-operational-request-and-handoff.md) | Accepted | A durable request/outbox precedes one atomic accepted Phase 5 lifecycle handoff. |
| [ADR-0014](ADR-0014-campaign-lifecycle-controls-completion-and-archival.md) | Accepted | Derived campaign lifecycle, campaign-only controls, completion, and non-destructive archival remain separate from experiment execution. |
| [ADR-0015](ADR-0015-cancellation-reconciliation-and-recovery.md) | Accepted | Cancellation intent/settlement, detection-only reconciliation, and bounded recovery preserve owning-service authority. |
| [ADR-0016](ADR-0016-scheduler-atomic-claim-hardening.md) | Accepted | Scheduler-owned atomic claim hardening gates production campaign dispatch. |
| [ADR-0017](ADR-0017-campaign-privileges-and-audit.md) | Accepted | Separate least-privilege capabilities, invoker-rights Phase 5 invocation, and append-only audit prevent authority overlap. |
| [ADR-0018](ADR-0018-scheduler-generation-52-exact-attempt-authority.md) | Accepted | Generation-52 binds scheduler mutation, signaling, finalization, analysis, and cutover to one exact-attempt authority. |

## 8. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-07-15 | Established ADR numbering, statuses, required sections, workflow, and initial index. |
| 1.1.0 | 2026-07-19 | Added ADR-0005 for explicit manual conversion to a paused experiment. |
| 1.2.0 | 2026-07-20 | Added ADR-0006 for the pure non-authorizing Phase 6A follow-up proposal. |
| 1.3.0 | 2026-07-20 | Added ADR-0007 for exact append-only Phase 6B persistence and read-only preview. |
| 1.4.0 | 2026-07-21 | Added ADR-0008 for exact append-only Phase 6C administrative review without action authority. |
| 1.5.0 | 2026-07-22 | Proposed ADR-0009 for exact append-only Phase 6D governance ratification with mandatory separation of duties and no Phase 6E authority. |
| 1.6.0 | 2026-07-24 | Accepted ADR-0009 and ADR-0010 through ADR-0017 to close Phase 6D and Campaign Operations architectural authority. |
| 1.7.0 | 2026-07-29 | Added ADR-0018 for generation-52 exact-attempt scheduler authority and technical cutover. |
