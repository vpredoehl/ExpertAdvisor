---
title: "Campaign Operations Phase 3 Architecture Readiness Review"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase3_Architecture_Readiness_CEE_Review_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 3 Architecture Readiness Review

## Executive summary

The current implementation provides a stable architectural foundation for the bounded Campaign Operations Phase 3 increment: durable dispatch, lease/attempt evidence, and atomic Phase 5 lifecycle handoff.

No architectural prerequisite blocker was found. PostgreSQL remains authoritative, the required transaction composition seam exists, Phase 2 requests are durable and safely undispatchable, and scheduler/worker authority remains isolated.

This verdict authorizes beginning Phase 3 implementation only. Production dispatch remains prohibited until ADR-0016 scheduler claim hardening is implemented and independently verified.

## Architecture assessment

### Layering and durable authority

The effective path remains:

```text
CLI → Service → Repository → PostgreSQL
```

- CLI code in `ExperimentScheduler.cpp` invokes Campaign Operations services; it does not perform Campaign Operations SQL.
- Services own and commit transactions for budget administration and request acceptance: [CampaignOperationsService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsService.cpp:147>).
- Repositories accept caller-owned transactions and do not commit independently.
- PostgreSQL enforces identities, authorization provenance, budget accounting, request/reservation atomicity, ACLs, and initial undispatchable state.
- Phase 2 requests already contain state/version and lease fields, while `production_dispatch_enabled` is database-constrained to false: [047_campaign_operations_budget_request_acceptance.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/047_campaign_operations_budget_request_acceptance.sql:363>).

One maintainability concern is that `CampaignOperationsRepository.cpp` already combines several conceptual repositories. Phase 3 should use dedicated dispatch/binding repository and service files rather than extending that unit indefinitely. This is organization within the accepted architecture, not a redesign.

### Transaction boundaries

The necessary composition boundary already exists:

- Phase 2 acceptance acquires authorization → budget → campaign and atomically creates the held reservation, ready request, acquisition event, and audit.
- Phase 5 exposes `LaunchRecommendationCampaignInTransaction`, explicitly starting and committing no transaction: [ExperimentRecommendationCampaignLaunchRepository.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignLaunchRepository.hpp:36>).
- ADR-0013 requires Phase 3 to hold authorization → budget → campaign → reservation → request through Phase 5 mutation, binding insertion, reservation commitment, and request binding: [ADR-0013](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0013-operational-request-and-handoff.md:39>).

Therefore Phase 3 can compose the accepted Phase 5 workflow without duplicating lifecycle or experiment SQL.

### Concurrency model

The current foundation establishes the first three global lock levels:

- Authorization semantic lock
- Budget semantic lock
- Campaign row lock

These are exposed transactionally in [CampaignOperationsRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.cpp:1387>). Existing Phase 5 repositories supply ordered proposal, activation, and experiment lock domains for levels 6–8.

Phase 3 must add:

- Explicit reservation and request row-lock helpers for levels 4–5.
- Request state/version lease CAS.
- Stable request-ID candidate selection.
- Separate immutable attempt acquisition and outcome evidence.
- Both authorization-domain locks when pending-work adoption is possible, ordered deterministically.
- Whole-transaction retry for `40001`/`40P01`.
- Canonical result lookup before retrying an uncertain handoff result.

No process-local mutex is required for correctness.

### Replay model

The foundation provides:

- System-derived logical-operation identity.
- One request per campaign/action/version.
- Full canonical hydration and comparison.
- Stable existing-identical versus changed-payload conflict behavior.
- Durable request and reservation evidence after restart.

Phase 3 must preserve the critical rule that a complete existing binding is reloaded before Phase 5 can be called again. A generic SQL retry wrapper is insufficient for commit uncertainty; handoff requires a dispatch-specific lookup-first recovery path.

### Coupling review

- **Scheduler:** No Campaign Operations service or repository calls scheduler selection, capacity, claim, attempt, process, or worker APIs. CLI co-location in `ExperimentScheduler.cpp` is structural, not an authority transfer.
- **Experiment lifecycle:** Coupling is limited to the accepted transaction-bound Phase 4C/5 workflow.
- **Workers:** No Campaign Operations worker dependency or process-control path exists.
- **Dispatch:** No dispatch tables, role grants, or consumer are currently enabled.
- **Experiment creation:** Campaign Operations currently performs none. Phase 3 must invoke Phase 5 rather than write experiment rows itself.
- **Budget settlement:** Current schema anticipates committed/released/expired states, but only acquisition is implemented. Phase 3 should add only held→committed settlement; release/expiry remain later-phase responsibilities.
- **Future phases:** Control, cancellation, reconciliation, completion, scheduler priority, partial dispatch, and scientific interpretation remain absent. Existing enums and reserved states do not grant authority.

### Scheduler limitation

The scheduler still performs an unlocked pending read and an unconditional running update: [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:5191>) and [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11924>).

That is a real architectural gap against ADR-0016, but it is explicitly a production-enablement gate rather than a gate to beginning Phase 3. ADR-0016 requires atomic claim/attempt implementation and independent verification before enablement: [ADR-0016](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0016-scheduler-atomic-claim-hardening.md:57>).

## Readiness assessment

| Area | Assessment |
|---|---|
| Accepted authority | Ready |
| Durable request/outbox | Ready |
| Authorization and budget evidence | Ready |
| Transaction composition | Ready |
| PostgreSQL authority | Ready |
| Lock-order foundation | Ready; levels 4–5 are Phase 3 work |
| Replay foundation | Ready; handoff lookup-first recovery is Phase 3 work |
| Migration strategy | Ready for additive migration 048 |
| ACL model | Ready; dispatcher/Phase 5 role analysis belongs in Phase 3 |
| Test infrastructure | Ready |
| Production dispatch | Not ready; intentionally gated by ADR-0016 |

## Blockers

There are no blockers to beginning Phase 3.

There is one hard blocker to production enablement: scheduler atomic claim/attempt hardening has not been implemented or independently verified.

Phase 3 acceptance must therefore retain all of these gates:

- `production_dispatch_enabled=false` for existing and newly accepted requests.
- No dispatcher or Phase 5 transactional capability granted to `pqxx`.
- No automatic production polling.
- Explicit single-request execution restricted to a clearly disposable/isolated test database.
- No scheduler launch, signaling, or worker execution in Phase 3 tests.

## Recommended Phase 3 sequencing

1. Freeze Phase 3 pure contracts and golden identities for leases, attempts, outcomes, bindings, control owners, and reservation commitment.

2. Add migration 048 with attempt/outcome, binding, and control-owner tables; guarded request/reservation transitions; exact ACLs; and deferred complete-binding enforcement. Preserve every Phase 2 row byte-for-byte and keep production disabled.

3. Implement focused dispatch/binding repositories, including levels 4–5 locks, candidate selection, lease CAS, canonical reload, complete binding persistence, and commitment transitions.

4. Implement the dispatch service as transaction owner. It should revalidate levels 1–5, classify downstream evidence before mutation, invoke `LaunchRecommendationCampaignInTransaction`, insert all bindings/control owners, commit the reservation and request, and record outcome/audit atomically.

5. Handle downstream evidence explicitly:

   - No Phase 5 evidence: normal create-and-bind.
   - Complete existing `pending/train`: require the separate adoption authorization and free control-owner key.
   - Partial, paused-only, progressed, or causally ambiguous evidence: fail closed; do not silently complete or adopt it.

6. Add isolated integration and concurrency tests for every Phase 3 acceptance criterion, including duplicate dispatch, revocation, budget mutation, direct Phase 5 overlap, adoption conflict, rollback points, lost response, restart, and scheduler/worker isolation.

7. Add the bounded test-only CLI last, after persistence and service tests prove the gate. Keep production polling and privilege enablement out of Phase 3.

## Improvements that can wait

- Extracting the existing Phase 1 campaign/authorization repositories into smaller files.
- Moving the Campaign Operations CLI parser out of the large scheduler translation unit.
- Adding production operator commands for campaign creation and authorization.
- Aligning older high-level lifecycle documentation that still describes recommendation conversion as future work.
- Scheduler claim hardening itself may run as a separate track, provided Phase 3 remains disabled.

## Verification record

- Files changed: none.
- Behavioral change: none.
- Review used `rg`, `sed`, numbered source inspection, Git history/status/diff checks, and Xcode project validation.
- `plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj`: passed.
- `git diff --check`: passed.
- Build and executable tests were not run because this was a read-only architecture review and a build would write DerivedData.
- `git status --short`: clean.
- `git diff --stat`: empty.
- Reviewed branch/commit: `campaign-operations` at `3a761d1`.

The accepted Phase 3 transaction seam, durable outbox, database authority, concurrency foundation, and isolation boundaries are sufficient to proceed; only production enablement remains gated.

READY TO BEGIN PHASE 3