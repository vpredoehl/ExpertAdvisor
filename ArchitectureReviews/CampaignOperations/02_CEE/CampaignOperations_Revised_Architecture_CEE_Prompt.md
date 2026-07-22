Using xhigh reasoning, produce a complete revised Campaign Operations architecture for the ExpertAdvisor/LSTM repository.

GOAL

Replace the incomplete CampaignOperations_Architecture_Output.txt transcript with a clean, complete, repository-grounded architecture document suitable for independent Critical Engineering Evaluation and implementation planning.

This is an architecture-generation task only.

Do not implement code.
Do not modify production behavior.
Do not create migrations.
Do not design CLI commands.
Do not begin scheduler integration.
Do not begin Campaign Operations implementation.
Do not redesign completed Phases 4 through 6.
Treat all accepted authority chains as fixed.

IMPORTANT OUTPUT REQUIREMENT

The final response must contain only the completed architecture deliverable and a brief final verification appendix.

Do not include:
- repository-wide file listings;
- tool transcripts;
- shell command output;
- internal progress notes;
- exploratory search results;
- incomplete drafts;
- implementation patches;
- migration SQL;
- CLI designs.

Use targeted repository inspection only. Do not perform an uncontrolled repository-wide listing. Read only the specific architecture documents, ADRs, domain contracts, repositories, services, migrations, tests, scheduler implementation sections, and lifecycle implementation sections necessary to support the design.

Before exiting, verify that the final response includes every required section listed below. If any required section is missing, complete it before returning the final response.

ARCHITECTURAL CONTEXT

The repository currently provides the completed authority chain:

Planning
→ Review
→ Approval
→ Materialization
→ Execution
→ Follow-up Review
→ Governance Ratification

Phase 6D governance ratification exists in implementation and migration 044, but ADR-0009 and affected architecture volumes may still contain inconsistent Proposed/Accepted status language.

First determine, from repository evidence, whether Phase 6D has been accepted and implemented. If so, explicitly identify the documentation status corrections required to align ADR-0009, its ADR index entry, and affected architecture volumes. Do not edit those files in this task.

The next candidate architectural concern is long-lived Campaign Operations.

Campaign Operations must remain separate from recommendation governance, experiment lifecycle authority, scheduler capacity and worker ownership, worker computation, and scientific outcome interpretation.

Campaign Operations must not treat Phase 6D ratification alone as operational execution authorization.

CEE FINDINGS THAT MUST BE RESOLVED

1. FORMAL PHASE 6 CLOSURE
- Determine whether Phase 6D is operationally and architecturally complete.
- Identify exact documentation status inconsistencies involving ADR-0009 and affected volumes.
- State that no Campaign Operations implementation may rely on ambiguous Proposed/Accepted status.
- Do not invent a Phase 6E merely to continue numbering.
- Explain whether Campaign Operations is the next bounded concern and whether it belongs within, refines, or supersedes parts of Volume X Research Automation.

2. OPERATIONAL AUTHORIZATION
Define the exact authority that permits Campaign Operations to act. Ratification alone is insufficient.
Specify the authoritative upstream evidence; the explicit operational authorization entity or event; the actor or role that grants it; separation from budget availability; separation from scheduler admission; replay and conflict semantics; revocation, cancellation, expiration, or supersession rules, if any; and what remains impossible without this authorization.

3. OWNERSHIP BOUNDARIES
Clearly distinguish ownership for Recommendation Governance, Campaign Operations, Experiment Lifecycle, Scheduler, Workers, and PostgreSQL.
Also identify what belongs nowhere, including automatic experiment creation from scores or ratification; scheduler interpretation of recommendation policy; Campaign Operations direct worker launch; Campaign Operations rewriting experiment outcomes; budget availability treated as authority; process presence treated as completion; and duplicated campaign membership reconstructed from current queries.

4. TERMINOLOGY
Define precise, non-overlapping meanings for recommendation campaign; campaign materialization; follow-up proposal; governance ratification; operational campaign; operational authorization; budget grant; reservation; operational request; dispatch; scheduler admission; scheduler claim; operational attempt; cancellation; reconciliation; operational completion; and scientific outcome.

5. SUBSYSTEM DECOMPOSITION
Design pure domain contracts; authorization service; budget service; reservation service; operational-request service; dispatch/handoff service; campaign-control service; reconciliation service; completion service; audit projection; repositories; read-only status projection; scheduler-facing boundary; and operator-facing boundary.
State which components are authoritative and which are derived projections.

6. DATA MODEL
Define conceptual entities and relationships without writing migrations:
operational_campaign; operational_authorization_event; campaign_budget_grant or budget_version; campaign_budget_adjustment, if needed; campaign_reservation; campaign_operational_request; campaign_request_binding or dispatch result; campaign_cancellation_event; campaign_reconciliation_observation; campaign_completion_event; append-only audit events; optional current-state projection.
For each entity define owner, purpose, immutable versus mutable fields, authoritative identity, canonical text, hash role, foreign-key provenance, uniqueness, replay semantics, conflict domain, lifecycle relationship, and retention expectations.
Canonical text must remain authoritative. Hashes may only accelerate lookup or locking.

7. IDENTITY AND DETERMINISTIC REPLAY
Define authoritative canonical identities for operational campaign; operational authorization; budget grant/version; reservation; operational request; cancellation request; dispatch binding; reconciliation observation; and completion decision.
Specify versioning, bytewise canonicalization, locale independence, ordering, timestamp treatment, collision handling, exact retry, changed-payload conflict, lost-response recovery, and request-to-downstream-result binding after experiments progress beyond pending state.

8. LIFECYCLE STATE MACHINE
Provide a complete Campaign Operations state machine separating administrative campaign control state; member operational state; reservation state; request state; downstream experiment state; scheduler attempt state; and scientific result state.
Include drafted; awaiting operational authorization; authorized; budgeted; reserving; ready; dispatching; active; paused; cancellation_requested; cancelling; terminal_completed; terminal_cancelled; terminal_failed; inconsistent; and reconciliation_required.
For every transition define authority, preconditions, transaction owner, resulting durable evidence, idempotent retry, and invalid-transition behavior.

9. BUDGETING
Choose and justify an initial deterministic budget unit using existing authoritative information.
Do not introduce generalized speculative CPU-time or cost estimation unless explicitly deferred.
Define who grants and amends budget; append-only/versioned structure; budget invariant; available, reserved, committed, released, exhausted amounts; reservation timing; consumption timing; release rules; failure handling; cancellation handling; uncertain downstream handling; expiration or reclamation; overrun prevention; concurrency control; separation from scheduler capacity; and separation from operational authorization.
Include an explicit invariant relating granted, reserved, committed, released, and remaining.

10. RESERVATIONS
Define reservation identity, scope, request relationship, acquisition transaction, conflict scope, expiration, settlement, release, cancellation behavior, uncertain-result handling, leakage prevention, and restart recovery.
At most one accepted operational request may consume a reservation unless explicitly versioned otherwise.

11. OPERATIONAL REQUESTS AND HANDOFF
An operational request is Campaign Operations-owned durable intent. It is not a scheduler claim, worker attempt, experiment outcome, or scientific success.
Define exact upstream authorization binding, campaign/member scope, intended downstream action, deterministic identity, status, acceptance, dispatch, downstream binding, replay, rejection, cancellation, and uncertain commit handling.
Choose and justify either:
A. reservation, request, and downstream lifecycle creation commit in one transaction; or
B. reservation and request commit first, followed by a durable outbox/dispatch/compensating protocol.
If existing workflows prevent one transaction, design the durable handoff explicitly.

12. SCHEDULER INTERACTION
Preserve ADR-0004.
The scheduler must not poll recommendation-governance tables, interpret Campaign Operations policy, grant operational authorization, allocate campaign budget, reconstruct campaign membership, or become the campaign state machine.
Campaign Operations must not assign scheduler capacity, select runnable phases, claim workers, launch workers, supervise processes, or mark experiment completion.
Any work handed to the scheduler must first become valid ordinary experiment lifecycle state through an accepted workflow.
Address the current scheduler implementation gap: it polls ordinary pending experiments and its running transition may not yet implement ADR-0004's compare-and-set attempt contract.
State whether the first Campaign Operations increment relies only on supported single-scheduler behavior or requires a separate prior scheduler-hardening ADR and increment.
Do not silently redesign the scheduler inside Campaign Operations.

13. TRANSACTION BOUNDARIES
Define explicit transactions for authorization recording, campaign creation, budget grant/amendment, reservation acquisition, request creation, dispatch selection, downstream handoff, request binding, cancellation, reconciliation observation, reservation settlement, and operational completion.
No transaction may remain open across process launch, worker execution, long-running polling, or external commands.
For each transaction specify rows/entities read, rows/entities written, lock scope, validation, commit result, retry result, and uncertain-outcome recovery.

14. GLOBAL LOCK ORDER
Define one global lock order across authorization, budget, operational campaign, reservation, operational request, proposal review/execution domains, activation domains, experiment rows, and scheduler attempt rows if involved.
Explain deadlock avoidance for overlapping campaigns, direct Phase 4C/5 operations, cancellations, reconciliation, and scheduler claims.

15. CANCELLATION
Define cancellation before authorization, after authorization, before reservation, after reservation, after request creation, after dispatch, after experiment creation, while pending, after scheduler claim, while running, concurrent with completion, and after terminal completion.
Cancellation must not delete immutable history, rewrite completed experiment outcomes, assume a running process can be erased, or release budget before downstream state is authoritatively settled.
Define operator authority, audit evidence, retry, conflict, and terminal semantics.

16. COMPLETION
Operational completion must be distinct from scientific success.
Define completion for all-success, mixed success/failure, cancellation, terminal partial completion, unresolved reservations, inconsistent evidence, and reconciliation-required state.
Specify who records completion, required evidence, append-only semantics, budget settlement, request settlement, member summary, replay, conflict, and blocking conditions.

17. RECONCILIATION AND RESTART RECOVERY
Design bounded, restart-safe reconciliation.
Define authoritative inputs, bounded batch selection, reconciliation identity, observations versus mutations, repair authority, non-creative behavior, retry, uncertain request detection, reservation leakage detection, cancellation settlement, terminal detection, operator escalation, and inconsistent state handling.
Reconciliation must not create duplicate requests, recreate downstream work because a process is absent, release reservations based only on missing PID, mark completion without authoritative lifecycle evidence, or scan without bounds.

18. PRIVILEGE MODEL
Define least-privilege capability boundaries for operational authorization, budget administration, reservation/request dispatch, cancellation, reconciliation, read-only status, and scheduler lifecycle access.
Prefer SELECT, payload-column INSERT, sequence USAGE, narrowly scoped UPDATE or security-definer functions only where mutable state is required, no broad UPDATE/DELETE/TRUNCATE, append-only audit, revoked trigger-function execution, pinned safe search paths, and explicit privilege tests.
The scheduler should not receive recommendation-governance policy access unless strictly necessary.

19. AUDITABILITY
Every authoritative action should record actor, role, reason, exact expected identity, prior state, resulting state, causal request, reservation, downstream experiment or execution IDs, outcome, timestamp metadata, canonical identity, and replay disposition where applicable.
Audit rows must not become a competing mutable current-state table.

20. CONCURRENCY ANALYSIS
Analyze duplicate campaign creation, budget grant races, reservation races, duplicate request races, cancellation versus dispatch, cancellation versus scheduler claim, cancellation versus completion, reconciliation versus dispatch, reconciliation versus cancellation, direct Phase 4C/5 operations overlapping Campaign Operations, multiple Campaign Operations processes, multiple scheduler processes, hash collisions, lost responses, rollback, and partial external progress.
For each state conflict domain, winner, loser, retry result, and preserved invariant.

21. IMPLEMENTATION PHASING
Recommend the smallest safe sequence. The first increment must be architecture/ADR work, not a migration or scheduler integration.
At minimum evaluate:
Phase A: formal Phase 6 closure and terminology; Campaign Operations ownership ADR; no runtime behavior.
Phase B: pure domain identities and state machine; no database.
Phase C: append-only operational authorization and campaign manifest persistence; read-only inspection only.
Phase D: budget and reservation ledger; no scheduler integration.
Phase E: operational request and durable dispatch/handoff; no direct worker launch.
Phase F: cancellation and reconciliation.
Phase G: operational completion.
Phase H: optional scheduler hardening or new work-class integration only under separate accepted ADR.

22. TEST STRATEGY
Define tests for pure identity/state contracts, canonical golden vectors, migration repeatability, ACL boundaries, corruption rejection, exact and conflicting replay, independent-connection concurrency, budget races, reservation leakage, uncertain commit, cancellation races, completion races, bounded reconciliation, restart recovery, scheduler isolation, unchanged Phase 4–6 evidence, and production-process safety.

23. ADR RECOMMENDATIONS
List ADRs required before implementation, including ownership/Volume X relationship, operational authorization, budget/reservation semantics, request/handoff, lifecycle/completion, cancellation/reconciliation, scheduler interaction or claim hardening, and privilege model.
State which ADR must be accepted first.

24. RISKS AND OPEN QUESTIONS
Identify blocking, implementation, migration, and operational risks; deferred questions; and decisions that must not be made implicitly during coding.

REQUIRED FINAL DELIVERABLE SECTIONS

The final response must contain all of these sections, in this exact order:

1. Executive Summary
2. Repository-Grounded Context
3. Formal Phase 6 Closure
4. Architectural Rationale
5. Terminology
6. Ownership Boundaries
7. Campaign Operations Responsibilities
8. Prohibited Responsibilities
9. Subsystem Decomposition
10. Operational Authorization
11. Data Model
12. Identity and Deterministic Replay
13. Lifecycle State Machine
14. Budget Model
15. Reservation Model
16. Operational Request and Handoff Model
17. Scheduler Interaction Contract
18. Transaction Boundaries
19. Global Lock Order
20. Concurrency Analysis
21. Cancellation Semantics
22. Completion Semantics
23. Reconciliation and Restart Recovery
24. Security and Privilege Model
25. Audit Requirements
26. Component Interaction Diagrams
27. Repository Interface Responsibilities
28. Service Layer Responsibilities
29. Test Strategy
30. ADR Recommendations
31. Recommended Implementation Phases
32. Risks
33. Open Architectural Questions
34. Acceptance Preconditions
35. Overall Readiness Assessment
36. Final Deliverable Verification

FINAL DELIVERABLE VERIFICATION

In section 36, include a checklist confirming:
- all 36 required sections are present;
- no code was implemented;
- no files were modified;
- no migrations were created;
- no CLI commands were designed;
- no scheduler integration was implemented;
- the architecture addresses every CEE finding;
- repository inspection was targeted rather than an uncontrolled repository-wide listing;
- the final response is a complete architecture document rather than a tool transcript;
- the recommended first implementation increment is architecture/ADR-only.

Also report:
- files changed;
- tests/builds run;
- database or production processes accessed;
- current branch and HEAD;
- git status --short;
- git diff --stat;
- git diff --check.

Do not return until every required section is complete.
