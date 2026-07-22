## 1. Executive Summary

The revised architecture is directionally sound, repository-consistent in its principal boundaries, and substantially complete. It correctly preserves Phase 4D membership authority, reuses the Phase 5 transaction-bound workflow, isolates scheduler ownership, separates operational completion from scientific interpretation, and treats PostgreSQL/canonical text as authoritative.

It is not safe to accept unchanged. Several localized ambiguities could permit duplicate campaigns or requests, overlapping budget authority, unclear control ownership, mutable “immutable” events, and a revocation-versus-handoff race.

These issues do not require subsystem redesign. They require targeted corrections before acceptance as the Phase A baseline.

## 2. Recommended Disposition

**Accept with required targeted corrections.**

Acceptance should become effective only after the conditions in section 22 are incorporated and independently checked. A full architectural redesign or another broad review is unnecessary unless a correction changes the fixed direction.

## 3. Blocking Defects

1. **Logical campaign and request uniqueness is incomplete.**  
   V1 permits one operational campaign per materialization *and origin mode*, allowing the same executable materialization to acquire multiple coordination envelopes. Request identity includes actor/reason, but no separate durable idempotency key or unique campaign/action/scope constraint prevents another actor, reason, or grant from creating a second full-materialization request. See [architecture:284–290, 320–328](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:284>) and [architecture:350–357](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:350>).

2. **Campaign origin remains simultaneously normative and open.**  
   The architecture uses “ratification-origin campaign,” permits an origin mode in identity, and then leaves the category and ratification requirement unresolved. Phase 6D defines no new executable materialization, so ratification cannot safely act as an executable origin. See [architecture:242–251](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:242>) and [architecture:1213–1226](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:1213>).

3. **The budget write model has duplicated authority.**  
   `campaign_budget_grant / budget_version` and `campaign_budget_adjustment` both describe append-only changes to the effective total, while the text allows an adjustment to be either a version or a separate event. Two plausible implementations could disagree about which rows determine `G`. See [architecture:286–287](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:286>) and [architecture:451–466](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:451>).

4. **Several purportedly append-only facts have no coherent persistence lifecycle.**  
   The campaign is immutable and administrative state is event-derived, but no conceptual campaign-control event represents pause/resume. A dispatch attempt is immutable yet contains an outcome that becomes known only after selection. A cancellation event is immutable yet contains a “later settlement reference.” See [architecture:227–232](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:227>), [architecture:288–295](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:288>), and [architecture:537–545](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:537>).

5. **Adoption of existing pending work lacks an exact authority edge.**  
   The architecture permits `reused_pending_and_bound` and gives one binding active control ownership, but later leaves adoption and ownership open. Binding pre-existing work as a controller could grant cancellation authority over work Campaign Operations did not create. See [architecture:548–559](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:548>) and [architecture:1217–1223](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:1217>).

6. **Handoff lock acquisition is internally inconsistent.**  
   The global order begins with authorization, budget, campaign, reservation, and request. The handoff table and diagram instead begin with request/reservation. An implementation following the latter could recheck authorization without holding its serialization domain through commit, allowing revocation to race with downstream mutation. See [architecture:616–629](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:616>), [architecture:641–665](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:641>), and [architecture:899–914](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Revised_Architecture_Output.txt:899>).

## 4. Required Targeted Corrections

1. Freeze V1 campaign origin as the exact Phase 4D materialization origin. Treat Phase 6D ratification as optional governance provenance, never as executable origin or scope. Future follow-up origin remains unsupported until a separate workflow creates a new exact materialization.

2. Enforce at most one V1 operational campaign for an exact materialization, independent of provenance labels.

3. Define one durable logical-operation/idempotency key and one uniqueness domain for the complete V1 campaign/action/materialization scope. Changed payload under that key must conflict; actor, reason, or a later grant must not create a second request for the same V1 operation.

4. Select one budget authority: preferably an append-only, monotonically versioned ledger whose entries contain delta, prior version, and resulting total. Do not retain an independent snapshot authority and an adjustment authority.

5. Add or identify append-only evidence for campaign pause/resume and other administrative controls. Do not derive these states from a mutable projection alone.

6. Declare dispatch attempts to be audit history, not current authority. Request state/version and its lease must be authoritative for selection. Freeze whether attempt acquisition and terminal outcome are separate immutable events or one final immutable record.

7. Record cancellation settlement separately from the immutable cancellation request, or otherwise make its mutability explicit and guarded.

8. Require explicit authorization to adopt existing exact `pending/train` Phase 5 evidence, and transactionally reject adoption when another controller exists. Freeze whether control ownership is permanent for V1 or has a separate append-only transfer/release contract.

9. Complete the identity catalog with dispatch-attempt, campaign-control, reservation-settlement, cancellation-settlement, and other replay-significant transition identities. Resolve the conflict between completion/reconciliation snapshot evidence and the general exclusion of mutable lifecycle facts.

10. State that handoff acquires and holds levels 1–5 in global order before entering existing Phase 4C/5 levels 6–9, through the binding/reservation/request commit.

11. Define exact precedence for overlapping completion classifications, especially completed-plus-failed and failed-plus-cancelled combinations.

12. Replace ambiguous reconciliation wording with: **“Reconciliation detects and records; owning services perform every repair transition.”**

13. State explicitly that Campaign Operations Phases A–D require no scheduler changes; Phase E may exist only disabled or in isolated test environments; production dispatch requires scheduler hardening to be accepted, implemented, and independently verified.

14. Replace the blanket “all ADRs before implementation” wording with explicit per-phase ADR gates, while retaining ADR-0010 as the first Campaign Operations authority.

## 5. Non-Blocking Clarifications

- Clarify that the existing Phase 5 “launch” name means lifecycle handoff to `pending/train`, not worker launch.
- State whether a successfully authorized adoption of existing pending work consumes member-dispatch units; current rules imply that it does.
- Freeze nullable/no-expiry canonical syntax and prohibit inferred local-time defaults if authorization expiry remains optional.
- State that audit fields are required “where applicable”; unrelated actions cannot possess every listed causal ID.
- Identify the narrowly privileged writer for an optional current-state projection.
- Retention and archival periods may remain deferred; they do not block Phase A.
- The stale branch/completion text in `AGENTS.md` is documentation drift, not Campaign Operations authority.

## 6. Phase 6 Closure Assessment

The document correctly distinguishes three facts:

- Phase 6D implementation exists: migration 044, domain/repository/service code, project membership, and documentation are present. The migration fixes non-operational semantics at [migration 044:1–4, 136–144](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql:1>), and the service owns one transaction through commit at [ratification service:48–85](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationService.cpp:48>).
- ADR-0009 remains formally `Proposed`, and its decision remains titled “Proposed decision.” See [ADR-0009:1–24](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md:1>).
- Proposed ADRs are not implementation authority under the repository’s ADR rules. See [ADR index rules:18–30](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/README.md:18>) and [Volume I:529–562](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_I_Foundation.md:529>).

Volume VIII and Volume XII also still describe Phase 6D as proposed despite documenting its implementation: [Volume VIII:1–3](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_VIII_Recommendation_Engine.md:1>) and [Volume XII:1–3](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_XII_Database.md:1>).

No reviewed statement improperly treats ADR-0009 as authority. The proposed documentation-only closure is correct and required before Campaign Operations implementation.

## 7. Campaign Origin and Scope Assessment

The initial executable scope is repository-consistent: Phase 4D persists one immutable ordered materialization, and its members are complete, unique, and provenance-bound. See [Phase 4D materialization:7–48](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase4DExperimentRecommendationCampaignMaterialization.rst:7>) and [migration 041:96–142, 228–299](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/041_experiment_recommendation_campaign_materialization.sql:96>).

A targeted correction is required. V1 needs an explicit versioned origin domain with only the materialization-origin category executable. “Ratification-origin” is currently unsafe terminology because Phase 6D supplies governance evidence but no new materialization. Follow-up proposals remain non-authorizing and do not create scope: [Phase 6A:118–129](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase6ARecommendationCampaignFollowUpProposal.rst:118>).

## 8. Operational Authorization Assessment

The main authorization boundary is sound:

- Ratification, budget, and scheduler capacity are not authorization.
- The grant binds exact campaign, scope, action, actor, role, reason, and validity.
- Revocation blocks new reservation/request/dispatch actions but does not erase committed bindings.
- Already-bound work is routed through lifecycle-owned cancellation.
- Ratification-origin separation from reviewer/ratifier is stated.

Required remaining edges are:

- Define exactly how an active grant is selected when grants, revocations, supersessions, and validity intervals overlap.
- Require the handoff transaction to serialize against grant revocation through commit.
- Require distinct authority for adopting and controlling existing pending work.
- State whether capability separation also requires different human principals. The only currently fixed principal inequality is reviewer/ratifier separation; other role separation must not be inferred silently.

## 9. Data Model Assessment

| Entity | Assessment |
|---|---|
| `operational_campaign` | Necessary, but V1 uniqueness must be per exact materialization, not materialization-plus-origin-mode. |
| `operational_authorization_event` | Necessary and correctly append-only; effective-chain rules need precision. |
| Budget grant/version/adjustment | Necessary capability, but the three labels overlap as competing write authorities. |
| `campaign_reservation` | Necessary; guarded state is acceptable if every transition has replayable append-only evidence. |
| `campaign_operational_request` | Necessary durable outbox; logical-scope uniqueness is missing. |
| `campaign_dispatch_attempt` | Useful only as audit/diagnostic history; it must not determine current dispatch authority. |
| `campaign_request_binding` | Necessary and correctly stable after experiment progress; complete cardinality and control ownership need freezing. |
| `campaign_cancellation_event` | Necessary request evidence; later settlement cannot mutate an immutable event. |
| `campaign_reconciliation_observation` | Necessary append-only detection evidence; it must not repair directly. |
| `campaign_completion_event` | Necessary terminal operational decision; classification precedence needs precision. |
| Audit projection | Correctly derived and not a second source of truth. |
| Current-state projection | Correctly rebuildable and non-authoritative; writer privileges need clarification. |

The architecture also needs a persisted campaign-control event or another explicit authoritative representation for pause/resume.

## 10. Identity and Replay Assessment

Canonical-text authority, hash collision handling, bytewise comparison, UTF-8 framing, locale independence, collection ordering, and generated-timestamp exclusion are well specified and align with [Volume I:148–190](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_I_Foundation.md:148>).

Defects requiring correction:

- `campaign_dispatch_attempt` claims a canonical identity but has neither a canonical prefix nor identity-input definition.
- The text refers to an idempotency key without modeling it.
- Mutable transition requests and settlement events lack versioned replay identities.
- Campaign origin is an identity input without a frozen domain.
- Reconciliation and completion identities deliberately include observed lifecycle evidence, while the general rule appears to exclude mutable lifecycle state. Per-entity rules must explicitly permit point-in-time evidence for observations/completion while excluding later lifecycle state from campaign/request/binding identity.
- Semantic expiry instants must specify null/default behavior and fixed precision.

Bindings are correctly stable after downstream progress, and lost-response replay correctly avoids reinvoking Phase 5.

## 11. Lifecycle Assessment

The architecture correctly separates campaign, member, reservation, request, experiment, scheduler-attempt, and scientific-result dimensions.

The following targeted gaps remain:

- Revocation/expiry is defined only as `authorized → awaiting` before binding, not for budgeted, ready, leased, or active projections.
- `recorded` request state has no defined transition or observable purpose.
- “Resume to prior eligible state” requires a deterministic derived target, not remembered mutable state.
- `inconsistent` and `reconciliation_required` need explicit unresolved/resolved evidence semantics. Historical observations cannot permanently dominate merely because they exist.
- Routine reconciliation must never clear `inconsistent`; only an owning service’s accepted correction/supersession followed by append-only resolution evidence may do so.
- The architecture must state what an immutable campaign completion means if the ordinary lifecycle later retries or requeues the bound experiment. The repository currently permits failed retry and inference/analysis requeue at [scheduler controls:5050–5075](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:5050>).

## 12. Budget and Reservation Assessment

The algebra is sound:

- `H = A − C − L`
- `M = G − C − H = G − A + L`
- `C + H ≤ G`

With serialized budget locking and guarded reservation transitions, overcommitment is preventable. The document correctly keeps units held after uncertain handoff, forbids refund after binding, and separates budget from scheduler capacity.

The authoritative representation must nevertheless be singular. An append-only versioned ledger is the coherent choice; any stored resulting total is validated ledger evidence, not a second snapshot authority.

Expiration is race-safe only if expiry and dispatch lock the same request/reservation and the handoff holds the lease/row lock. The document states this, but required/null expiry semantics remain unresolved.

## 13. Request, Dispatch, and Binding Assessment

The durable request/outbox is justified and compatible with existing authority. Phase 5 already:

- operates on one exact materialization;
- runs execution and activation in one caller-owned transaction;
- uses established proposal, activation, and experiment lock domains;
- rejects partial direct overlap;
- treats progressed activated experiments as non-idempotent.

See [Phase 5 launch:7–18, 39–78](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase5ExperimentRecommendationCampaignLaunch.rst:7>) and [transaction-bound interface:36–43](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignLaunchRepository.hpp:36>).

Bindings can therefore be committed atomically with Phase 5 results and safely replayed after lost responses. Unbound progressed experiments correctly fail closed.

Required corrections are logical-request uniqueness, explicit adoption authority, complete binding cardinality, exact per-member execution/activation dispositions, and the audit-only status of dispatch attempts.

## 14. Scheduler Interaction Assessment

Scheduler isolation is correct. Campaign Operations creates or activates ordinary lifecycle work but does not poll, claim, allocate capacity, launch, or supervise.

The scheduler gap is repository-confirmed:

- Pending candidates are selected by an unlocked read at [scheduler:4822–4839](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:4822>).
- `MarkExperimentRunning` updates only by experiment ID, without an expected pending status/phase predicate or durable attempt identity, at [scheduler:11532–11558](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:11532>).
- This falls short of ADR-0004’s atomic-claim requirement at [ADR-0004:21–33](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md:21>).

The architecture correctly rejects singleton operation as a complete solution. It should add the exact Phase A–E/production-enablement statement identified in section 4.

## 15. Transaction and Locking Assessment

Transaction boundaries are otherwise coherent:

- No transaction spans worker launch, execution, scheduler polling, operator interaction, or reconciliation loops.
- Phase 5’s actual order is proposal IDs, activation execution IDs, then experiment IDs: [launch repository:122–159](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentRecommendationCampaignLaunchRepository.cpp:122>).
- Direct Phase 4C/5 paths can start at the existing suffix without acquiring Campaign Operations locks.
- Separate downstream cancellation after releasing Campaign Operations locks avoids reverse acquisition.

The blocking ambiguity is the handoff’s levels 1–5 acquisition. Authorization, budget, campaign, reservation, and request must be acquired in the declared order and held through Phase 5 mutation, binding, reservation commitment, and request transition. Otherwise authorization revalidation is only a stale read.

## 16. Cancellation Assessment

Cancellation is correctly distinct from pause, cannot refund committed units, does not erase running work, and delegates experiment transitions to lifecycle authority.

Repository behavior supports the stated constraints: ordinary cancellation accepts only `pending` or `paused` and rejects `running` at [scheduler controls:5041–5048](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:5041>). That command locks the experiment row, but the current scheduler’s unconditional running update still creates the documented race.

Required corrections are:

- separate immutable cancellation request from settlement evidence;
- freeze control ownership for adopted experiments;
- serialize dispatch/cancellation on expected request versions;
- define deterministic cancellation/completion disposition when terminal evidence wins.

## 17. Completion Assessment

The operational/scientific separation is strong and repository-consistent. Phase 5 scientific assessment is explicitly read-only, non-authoritative, and non-success-declaring at [Phase 5 outcome assessment:4–23](</Volumes/Developer SSD/ExpertAdvisor/docs/Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst:4>).

Completion properly requires settled reservations, requests, bindings, cancellations, lifecycle evidence, and consistent budget arithmetic.

Two corrections are required:

1. Define disjoint precedence for mixed terminal classifications. “Completed and failed” and “any terminal failed member” overlap.
2. Define the effect of later lifecycle retry/requeue. The completion event may remain an immutable decision about settled obligations at its recorded evidence point, but current status must not misleadingly imply that the downstream experiment remains terminal.

`terminal_completed` for completed-plus-cancelled members is coherent only because it means operational settlement, not universal downstream success; that meaning should remain explicit.

## 18. Reconciliation Assessment

Selection is bounded, cursor-ordered, restart-safe, and limited to relevant states. Ambiguous evidence correctly blocks release, redispatch, invented binding, and completion.

Ownership is mostly correct, but the definition still says reconciliation may “detect or repair,” and the permitted-repairs list can be read as direct mutation authority. The baseline should state exactly:

> Reconciliation detects and records; owning services perform every repair transition.

Observations and repair outcomes must remain separately attributable.

## 19. Privilege and Audit Assessment

The role decomposition is adequate and keeps scheduler policy-blind. Append-only history is protected, mutable request/reservation state is transition-scoped, and audit is not a competing truth source.

Actor, capability, reason, expected/prior/resulting state, causal identities, diagnostics, database time, and replay disposition are sufficient.

Required privilege clarifications:

- A dispatcher may obtain only the narrow Phase 5 transactional capability required for exact handoff.
- Adoption of pre-existing experiments requires explicit control authority.
- Reconciliation observation capability must not carry repair mutation privileges.
- An optional projection writer requires a narrow role and its projection cannot be consulted for mutation authorization.
- Audit fields should be mandatory where applicable rather than populated with invented IDs.

## 20. ADR and Implementation-Phase Assessment

The ADR decomposition is generally complete, minimally scoped, and correctly places ADR-0010 ownership/Volume X relationship first. Volume X is currently reserved and non-executable: [Volume X:1–28](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_X_Research_Automation.md:1>).

No ADR must be combined or split. ADR-0014 and ADR-0015 should be cross-reviewed because cancellation settlement affects completion, but they may remain separate.

The gating language needs correction:

- Phase A is documentation/ADR-only.
- Phases A–D may proceed without scheduler changes once their own governing ADRs are accepted.
- Phase E requires its handoff/security ADRs and must remain disabled/test-only.
- ADR-0016 acceptance, implementation, and independent verification gate production dispatch—not pure domain or non-scheduler persistence work.
- No implementation phase may silently rely on a merely proposed ADR governing that phase.

## 21. Terminology Findings

Terms already used consistently: operational authorization, activation, scheduler admission, scheduler claim, cancellation versus pause, operational completion, and scientific outcome.

Terms requiring freeze:

- **Campaign origin:** materialization-origin versus ratification-origin/follow-up-origin.
- **Grant:** always distinguish operational authorization grant from budget grant.
- **Budget version/adjustment/ledger entry:** select one authoritative vocabulary.
- **Operational attempt:** explicitly audit-only Campaign Operations dispatch attempt.
- **Dispatch:** request-to-lifecycle handoff, never scheduler claim.
- **Binding versus control ownership:** immutable causality is not automatically active control.
- **Phase 5 launch:** lifecycle handoff, not worker launch.
- **Terminal completed:** operationally settled, not necessarily all downstream experiments successful.
- **Reconciliation repair:** performed by owning services, never by the observer.

## 22. Exact Conditions for Baseline Acceptance

The architecture may become the accepted baseline when all of the following are true:

1. ADR-0009 is formally accepted by the proper decider and Phase 6 documentation is aligned.
2. ADR-0010 is accepted first and explicitly bounds Campaign Operations against Volume X.
3. V1 origin is frozen to one exact Phase 4D materialization; ratification is provenance only.
4. One operational campaign and one logical V1 operation/request are enforceable per exact materialization scope.
5. A single append-only budget representation is selected.
6. Campaign control, dispatch attempt/outcome, cancellation settlement, and reconciliation resolution evidence have coherent append-only/mutable boundaries.
7. Existing-work adoption and downstream control ownership have explicit authorization and cardinality rules.
8. All replay-significant identities, null/expiry semantics, and lifecycle-snapshot exceptions are versioned.
9. Handoff lock order and revocation serialization are stated consistently.
10. Completion classification precedence and post-completion lifecycle behavior are frozen.
11. Reconciliation is explicitly detection/recording only.
12. Per-phase ADR gates replace the contradictory blanket gating language.
13. Phase E is explicitly disabled/test-only until scheduler claim hardening is accepted, implemented, and independently verified.
14. Canonical golden vectors and exhaustive state-transition matrices are required before their respective implementation increments.

## 23. Overall Readiness Assessment

The revised architecture is close to baseline quality and does not contradict the repository’s accepted Phase 4–6 authority. Its central subsystem boundary and implementation direction should be retained.

It is ready to support Phase A documentation and ADR work after the targeted corrections above are incorporated. It is not ready for migration, runtime implementation, production dispatch, scheduler integration, or operational enablement.

Final classification: **accept with required targeted corrections**.

## 24. Review Verification

- Files changed: none.
- Tests/builds run: none; this was a read-only architectural evaluation.
- Database accessed: none.
- Production scheduler, training, inference, or analysis processes accessed: none.
- Current branch: `campaign-operations`.
- HEAD: `d78771fe836f18efa2b1119176394221366f8141`.
- `git status --short`: clean; no output.
- `git diff --stat`: no differences; no output.
- `git diff --check`: passed, exit status 0; no output.