---
title: "Campaign Operations Phase H Targeted Architecture Correction"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_TargetedArchitectureCorrection_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase H Targeted Architecture Correction

## 1. Decision and scope

Phase H architecture is now accepted and implementation-authorizing for H1–H3. This pass changed documentation only: no migration 055, C++, CLI, scheduler, worker, database role, or live operational behavior was implemented.

## 2. Files created and updated

Created:

- [ADR-0019](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md:1>)
- [Normative Phase H architecture](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:1>)

Updated:

- [Accepted Campaign Operations architecture](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md:1>)
- [Volume X](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_X_Research_Automation.md:1>)
- [Volume XI](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_XI_Scheduler.md:1>)
- [Volume XII](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_XII_Database.md:1>)
- [ADR index](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/README.md:68>)

## 3. ADR-0019 authority summary

ADR-0019 accepts:

- durable global production admission and immutable enable/disable evidence;
- default-off deployment and disable-first rollback;
- Campaign Operations ownership of the Manager;
- exact caller-keyed single-request dispatch;
- bounded sequential run-once processing;
- exact generation-52 scheduler and executable-build evidence;
- immutable per-request admission and Attempt V2;
- one shared Phase E engine;
- canary-first rollout;
- scheduler, lifecycle, worker, process, capacity, and scientific-policy isolation;
- explicit exclusion of continuous operation until H4.

It precisely amends the former “scheduler-owned only” Phase H roadmap and configuration/privilege-only enablement wording without superseding ADR-0010 through ADR-0018.

## 4. Corrected Phase H architecture summary

The normative document freezes component ownership, persistence inventory, canonical grammar, replay, recovery, repository/service boundaries, command boundary, lock order, privileges, readiness, rollout, and acceptance tests.

H1–H3 are independently safe while no enable event and no production LOGIN membership exist. H4 cannot be inferred from multi-manager database correctness.

## 5. Frozen identity table

| Identity | Frozen canonical authority |
|---|---|
| Scheduler evidence | Exactly generation 52, complete cutover, migration/protocol contracts, and exact cutover timestamp/actor/executable/process evidence |
| Manager build | Service contract, clean Git commit, Release configuration, compiler contract, executable SHA-256, version 1 |
| Enable event | Caller key, predecessor ID/canonical, expected/resulting versions, scheduler canonical, verification reference, actor, enabler capability, service/build contracts, reason, version |
| Disable event | Caller key, exact predecessor ID/canonical/version, resulting version, actor, disabler capability, reason, contract version |
| First admission | Exact request ID/canonical/version, first dispatch key, enable ID/canonical, requesting actor, original `session_user`, build, dispatcher capability, version |
| Attempt V2 | Request/admission/enable canonicals, operation key, ordinal, expected/resulting versions, lease digest/expiry, actor, original principal, build, capability, version |
| Recovery | Reuses the exact stored Attempt V2; recovering principal/process/build/time are diagnostic only |

PID, hostname, session UUID, invocation time, batch UUID, and process instance identity are excluded from logical identities.

## 6. Frozen operation-key table

- Enable and disable use caller-supplied durable keys.
- Exact single-request dispatch requires a caller-supplied key.
- Attempt acquisition and handoff retain the same per-request key.
- Run-once derives `mgr-v1:<fnv1a64(source-canonical)>:<expected-version>` from the complete immutable request canonical and expected version; the complete source canonical is persisted and compared.
- Batch selection is deliberately non-durable.
- Unexpired recovery reuses the original key, attempt, digest, and expiry.
- Phase F recovery advances request version; the next acquisition uses a new key and ordinal.
- Random process, host, session, or batch identity is prohibited.
- Uncertain-commit lookup uses the enable/disable key, `(request_id, operation_key)`, or exact attempt/binding identity—never mutable-state inference.

## 7. Exact scheduler-generation contract

Phase H V1 accepts only:

- generation `52`;
- `cutover_state=complete`;
- byte-identical versioned protocol canonical;
- exact independent verification reference;
- exact approved service/build canonical.

Generation 53 or any canonical mismatch makes the current enable head ineffective. Acquisition and unbound handoff fail; old leases expire and use Phase F recovery. Re-enable requires disable, a new contract version, new independent verification, and a new enable event.

Access is through pinned snapshot/`FOR SHARE` security-definer functions. Campaign Operations receives no raw scheduler-table access.

## 8. Boolean/admission committed-state equations

```text
production_dispatch_enabled
  == EXISTS(exact immutable request production admission)

first admission
  => exactly one matching first Attempt V2
  => exactly one matching production audit

Attempt V1
  => production_dispatch_enabled = false
  => no production admission
  => no production enablement/admission fields

Attempt V2
  => production_dispatch_enabled = true
  => exact admission evidence
  => exact enablement evidence
```

The architecture requires guarded false→true only, permanent rejection of true→false, deferred bidirectional constraints, owner-DML guards, atomic rollback proof, no backfill, and no authorization from the Boolean alone. This mirrors the accepted `completion_boundary_closed` witness correction.

## 9. Disable/re-enable state machine

The normative state table fixes:

- enable/acquisition and disable/acquisition serialization;
- handoff-first versus disable-first outcomes;
- exact binding replay after disable;
- E2 never authorizing an E1 lease;
- protocol-upgrade failure for acquired/unbound work;
- old-lease expiry and Phase F recovery;
- new version/key/ordinal after recovery;
- pause, cancellation, completion, authorization-revocation, and reservation-expiry races.

Disable never deletes evidence, cancels experiments, reverses reservations, signals workers, or mutates scheduler state.

## 10. Completion V1 nested-canonical proof

Completion V1 remains valid; Completion V2 is not introduced.

The direct trace is:

1. [Migration 054](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/054_campaign_operations_completion_and_audit.sql:304>) embeds each complete stored `attempt_identity_canonical` byte-for-byte in `requests_v1`.
2. [CompletionRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionRepository.cpp:254>) loads that exact PostgreSQL evidence text.
3. [Completion.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletion.cpp:123>) frames the complete request evidence into `campaign_operations_completion_v1`.

Because Attempt V2 contains complete admission and enable canonicals, Completion V1 transitively binds the complete production chain.

## 11. H1–H4 boundaries

- H1: authority, migration 055 persistence, canonicals, V1/V2 compatibility, roles, guards, readiness; no handoff or Manager batch.
- H2: common Phase E engine, enable/disable, exact canary dispatch, caller key, in-doubt recovery and concurrency tests; no batch.
- H3: optimistic bounded sequential run-once, deterministic keys, request-local continuation, global-failure stop; no daemon.
- H4: separately accepted supervision, cadence, restart/autostart, shutdown, health, logging, deployment, churn, and rollback.

## 12. Volume and ADR-index updates

- Volume X now records Manager ownership, exact admission-to-Phase-E flow, run-once-first behavior, disable-first rollback, and H4 exclusion.
- Volume XI records generation-52 implementation/review status, narrow exact evidence access, no scheduler polling of Campaign Operations, and no approval inheritance.
- Volume XII records migration 055’s frozen tables, functions, views, constraints, roles, V1/V2 equations, Completion V1 proof, ACLs, and no LOGIN grants.
- ADR-0019 is indexed as Accepted.
- The accepted roadmap and traceability matrix now assign production admission and bounded Manager ownership correctly.

## 13. Lock order

```text
0a exact scheduler protocol evidence, shared when mutating
0b global production-enable domain
1  authorization domains
2  budget
3  campaign/completion
4  reservations ascending
5  requests ascending
6+ existing Phase 5/lifecycle domains
```

Enable, disable, readiness, selection, acquisition, handoff, Phase F recovery, cancellation, completion, and scheduler-readiness paths are individually documented. Candidate selection is read-only; request-first `FOR UPDATE SKIP LOCKED` is prohibited.

## 14. In-doubt recovery contract

Enable, disable, acquisition, and handoff must:

1. discard the uncertain connection;
2. open a fresh connection;
3. locate the exact durable operation;
4. validate the complete canonical chain;
5. return identical only for complete byte-exact evidence;
6. retry only after proven absence and unchanged authority;
7. fail closed on partial or contradictory evidence.

Actual `pqxx::in_doubt_error` injection is required before commit, after commit/before response, and during lookup. Arbitrary `sql_error` is never replay success.

## 15. Privilege and deployment contract

Separate NOLOGIN capabilities are frozen for enabler, disabler, dispatcher, production Phase 5 transactional access, reader, and scheduler-evidence owner/reader.

The Manager lacks enabler, disabler, test roles, and scheduler mutation privileges. `pqxx` receives no new membership. Migration 055 grants no LOGIN membership. Emergency rollback is disable, stop Manager, revoke roles, and retain all evidence.

## 16. Readiness and rollout contract

Readiness is read-only and repeatable-read and reports migration/checksum, canonical versions, exact scheduler evidence, verification reference, enable head/effectiveness, service/build contracts, session principals, direct/inherited roles, admitted/unadmitted counts, current/old-event leases, expiry/recovery eligibility, reconciliation counts, and Completion V1 proof status.

Rollout is fixed as disabled migration/deployment → safe-window scheduler regressions → Manager-login audit → readiness → enable → one exact canary → evidence inspection → one-request run-once → bounded reviewed increases → separate H4 consideration.

## 17. Verification performed

- Inspected proposal/review outputs and transcripts, accepted architecture, ADR-0010–0018, migrations 047–054, dispatch/control/completion C++ paths, scheduler protocol, tests, volumes, index, and repository state.
- Inspected every changed document.
- Internal relative-link check: passed.
- ADR index completeness: passed.
- Canonical fixed-literal byte lengths: passed (`61` and `50`).
- Phase H tracked diff check and untracked-document whitespace check: passed.
- No documentation checker was present in the repository.
- No build, runtime test, database mutation, role change, scheduler action, or process action was performed.

Full `git diff --check` returns 2 only because the pre-existing/live CRLF content in `screenlog.0` reports trailing whitespace. The Phase H files pass independently.

## 18. Residual risks

Runtime implementation, migration 055, golden vectors, fault injection, concurrency tests, ACL tests, and safe-window scheduler/global-control process regressions remain future H1–H3 work. These are implementation obligations, not unresolved architecture choices.

`screenlog.0` continued growing during this pass and was not modified by this work.

## 19. Exact final repository state

Branch: `campaign-operations`
HEAD: `283f6a5ca5cd39c26e0f3599eff246b0b12d32fd`
Staged changes: none.

```text
 M ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md
 M docs/architecture/Volume_XII_Database.md
 M docs/architecture/Volume_XI_Scheduler.md
 M docs/architecture/Volume_X_Research_Automation.md
 M docs/architecture/adr/README.md
 M screenlog.0
?? CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md
?? CampaignOperations_PhaseH_FocusedArchitectureVerification_Output.md
?? docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md
?? docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md
```

Exact `git diff --stat` snapshot:

```text
 ...mpaignOperations_Revised_Architecture_Output.md |  159 +-
 docs/architecture/Volume_XII_Database.md           |   59 +-
 docs/architecture/Volume_XI_Scheduler.md           |   32 +-
 docs/architecture/Volume_X_Research_Automation.md  |   48 +-
 docs/architecture/adr/README.md                    |    2 +
 screenlog.0                                        | 1675 ++++++++++++++++++++
 6 files changed, 1896 insertions(+), 79 deletions(-)
```

The two new authority documents contain 1,060 additional untracked lines and therefore do not appear in `git diff --stat`. The two Phase H proposal/review outputs and `screenlog.0` were pre-existing inputs/unrelated worktree state.

PHASE_H_TARGETED_ARCHITECTURE_CORRECTION_COMPLETE