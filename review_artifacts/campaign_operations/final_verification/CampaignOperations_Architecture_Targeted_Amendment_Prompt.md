
# Campaign Operations Architecture Targeted Amendment

Using xhigh reasoning, amend the current Campaign Operations architecture document so that it fully resolves the findings from the focused CEE review and final verification.
Goal
Update the existing Campaign Operations architecture document into a complete, internally consistent, implementation-ready architecture specification.
This is an architecture-document amendment task, not another review.
You must directly edit the authoritative architecture document rather than merely describing recommended changes.
Authoritative Target
The intended authoritative target is:

ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md

First confirm that this exact file exists.

If it does not exist, search the repository for the actual Campaign Operations revised architecture document by filename and content. Identify the document that contains the full Campaign Operations architecture reviewed by the focused CEE and final verification.

Do not amend a review output, prompt, transcript, obsolete copy, or unrelated architecture volume.

If exactly one authoritative target can be established from repository evidence, amend that file in place and report its actual path.

If no authoritative target can be established unambiguously, do not create a replacement document and do not edit any file. Report the candidate files and the unresolved ambiguity.
Required Review Evidence
Use the following completed reviews as mandatory correction evidence:
ArchitectureReviews/CampaignOperations/03_Focused_CEE/CampaignOperations_Revised_Architecture_Focused_CEE_Output.md
If the Markdown version does not exist, use the corresponding existing text artifact:
CampaignOperations_Revised_Architecture_Focused_CEE_Output.txt
Also inspect the completed final-verification output:
CampaignOperations_Final_Verification_CEE_Output.md
If that Markdown file does not exist, locate and use the corresponding existing text artifact:
CampaignOperations_Final_Verification_CEE_Output.txt
Search the repository for these artifacts when their exact locations differ. Do not treat a missing suggested path as proof that the review does not exist.
Repository Context
Inspect the relevant repository documentation and implementation evidence needed to keep the amended architecture aligned with the actual system, including:
* architectural volumes and their authority rules;
* ADRs and the ADR index;
* Recommendation Governance Phases 4, 5, and 6;
* Phase 4C conversion-execution authority;
* Phase 4D campaign materialization;
* Phase 5 assessment and operational handoff behavior;
* Phase 6A–6D proposal, review, approval, and ratification contracts;
* experiment lifecycle ownership;
* scheduler ownership and claim behavior;
* PostgreSQL transaction, locking, privilege, and migration conventions;
* existing Campaign Operations implementation or planning artifacts;
* AGENTS.md only where relevant to architectural status or repository conventions.
Treat committed repository evidence as authoritative for the current implementation state, but do not allow implementation accidents to silently redefine architecture.
Required Outcome
The amended architecture must resolve every valid finding from the focused CEE and final verification.
It must clearly and precisely define:
1. architectural authority and acceptance status;
2. Campaign Operations ownership;
3. upstream provenance and identity;
4. operational authorization;
5. immutable executable scope;
6. budget grants and accounting;
7. reservations;
8. durable operational requests;
9. dispatch and accepted handoff;
10. downstream bindings;
11. scheduler separation;
12. pause and resume behavior;
13. cancellation semantics;
14. reconciliation;
15. restart and crash recovery;
16. operational completion;
17. audit evidence;
18. PostgreSQL constraints and transaction boundaries;
19. concurrency and lock ordering;
20. authorization and least-privilege boundaries;
21. idempotency and retry behavior;
22. versioning and canonical identity;
23. failure classification;
24. migration sequencing;
25. implementation phases and acceptance criteria.
Mandatory Architectural Boundaries
Preserve and make explicit all of the following boundaries.
Recommendation Governance versus Campaign Operations
Recommendation Governance owns:
* planning;
* recommendation and proposal policy;
* review and approval;
* immutable Phase 4D materialization;
* Phase 4C proposal authority;
* Phase 5 assessment evidence;
* Phase 6 proposal, review, approval, and governance ratification.
Campaign Operations owns:
* operational-campaign identity;
* operational grants;
* budgets;
* reservations;
* durable requests;
* dispatch attempts;
* downstream bindings;
* operational controls;
* reconciliation;
* operational completion;
* operational audit evidence.
Campaign Operations must not recompute or reinterpret Recommendation Governance decisions.
Governance ratification is not operational authority
Phase 6D ratification must never:
* authorize execution;
* create experiments;
* create operational requests;
* reserve budget;
* dispatch work;
* alter scheduler eligibility;
* imply an operational roadmap stage.
Where ratification is required by an origin policy, it is only one upstream prerequisite.
A separate explicit Campaign Operations authorization grant is required.
Do not invent a Phase 6E merely to continue phase numbering.
Budget is not authorization
Budget availability must never substitute for operational authorization.
The architecture must distinguish:
* operational grant;
* budget grant;
* reservation;
* request acceptance;
* dispatch;
* lifecycle handoff;
* scheduler admission;
* scheduler claim;
* worker execution;
* lifecycle completion;
* operational completion;
* scientific interpretation.
Campaign membership
One operational campaign must bind to one exact immutable executable materialization scope.
Campaign membership must come only from the authoritative persisted Phase 4D materialization and ordered materialization members.
It must not be reconstructed from:
* current recommendation rows;
* current rankings;
* proposal queries;
* mutable filters;
* present-day eligibility;
* Phase 6 follow-up proposals;
* governance ratification alone.
Accepted downstream handoff
Campaign Operations must invoke an already accepted lifecycle or Phase 5/Phase 4C authority.
It must not:
* insert experiments directly;
* bypass proposal execution validation;
* perform direct lifecycle SQL;
* create scheduler attempts;
* mark experiments runnable outside the accepted lifecycle workflow.
Scheduler isolation
The scheduler owns:
* ordinary lifecycle eligibility polling;
* capacity;
* atomic scheduler claims;
* scheduler attempts;
* worker launch;
* supervision;
* scheduler recovery;
* completion transitions assigned to it.
Campaign Operations must not:
* assign scheduler capacity;
* choose runnable phases;
* claim scheduler attempts;
* launch workers;
* supervise processes;
* use process presence as durable truth.
Scientific versus operational completion
Operational completion must not mean:
* profitable;
* scientifically successful;
* statistically valid;
* accepted by recommendation policy;
* suitable for production.
Operational completion concerns only whether Campaign Operations obligations and downstream lifecycle bindings have been durably settled.
Required Corrections from the Verification Evidence
At minimum, resolve the verification findings concerning:
* incomplete or ambiguous architectural authority;
* Proposed versus Accepted status inconsistencies;
* missing exact ownership boundaries;
* incomplete authority separation;
* ambiguous grant, budget, reservation, request, dispatch, and binding semantics;
* insufficient transaction and lock definitions;
* missing concurrency behavior;
* incomplete idempotency rules;
* incomplete crash-window and restart recovery rules;
* unsafe or creative reconciliation behavior;
* unclear cancellation ownership;
* unclear completion rules;
* weak audit causality;
* insufficient PostgreSQL invariants;
* incomplete least-privilege design;
* migration and rollout ambiguity;
* implementation sequencing not tied to acceptance criteria.
Do not assume this list replaces the review outputs. Read and address every concrete finding in those artifacts.
Required Data and Identity Design
Define stable, canonical identities for all durable Campaign Operations concepts, including as applicable:
* operational campaign;
* operational authorization grant;
* budget grant or amendment;
* reservation;
* operational request;
* operational dispatch attempt;
* request-to-downstream binding;
* pause or resume event;
* cancellation request;
* reconciliation observation;
* operational completion decision.
For each identity, specify:
* canonical input fields;
* contract version;
* canonical serialization;
* digest or hash usage;
* natural uniqueness;
* database uniqueness;
* immutable versus mutable fields;
* provenance references;
* retry and replay behavior.
Do not rely on timestamps, generated IDs, row order, or actor-provided prose as the sole semantic identity.
Required Budget and Reservation Semantics
Use integer member-dispatch units unless the repository already contains an accepted stronger contract.
Define:
* grant creation;
* amendments;
* revocation or supersession;
* reservation acquisition;
* reservation commitment;
* reservation release;
* reservation expiration;
* over-reservation prevention;
* concurrent reservation behavior;
* exact accounting equations;
* terminal request settlement;
* retry behavior;
* reconciliation behavior.
The architecture must prevent:
* double spending;
* negative available balance;
* one reservation funding multiple requests;
* one request consuming multiple incompatible reservations;
* release after commitment;
* commitment after release;
* silent resurrection of expired or revoked authority.
Required Transaction and Concurrency Design
For every mutating workflow, define:
* transaction boundary;
* authoritative rows read;
* rows locked;
* lock order;
* validations performed under lock;
* append-only events inserted;
* current-state projections updated;
* uniqueness relied upon;
* commit point;
* retry behavior after serialization, deadlock, or uniqueness conflict;
* crash behavior before and after commit.
Cover at least:
* operational-campaign creation;
* authorization grant;
* budget grant or amendment;
* reservation acquisition;
* request creation;
* dispatch selection;
* downstream handoff;
* binding creation;
* pause;
* resume;
* cancellation request;
* reservation release or settlement;
* reconciliation;
* operational completion.
Use deterministic lock ordering.
Do not use broad process-local mutexes as the source of durable correctness.
Required Dispatch Semantics
Define a durable sequence such as:
accepted request
→ reservation held
→ dispatch attempt recorded
→ accepted downstream workflow invoked
→ immutable downstream binding recorded
→ reservation committed or deterministically released
Address the external-call or multi-transaction crash window explicitly.
The architecture must explain how retries determine whether:
* no downstream artifact was created;
* the downstream artifact was created but the binding was not recorded;
* the binding already exists;
* the request was already satisfied;
* the request permanently failed;
* reconciliation may safely recover the binding.
Recovery must use deterministic identity and authoritative downstream evidence. It must not invent an experiment or infer success from process presence.
Required Reconciliation Rules
Reconciliation must be bounded, non-creative, and evidence-driven.
It may:
* observe authoritative state;
* restore a missing projection;
* record an observation;
* retry an accepted idempotent operation;
* attach an existing downstream artifact when deterministic identity proves the relationship;
* invoke a transition owned by the responsible service.
It must not:
* create missing governance evidence;
* invent authorization;
* invent budget;
* invent a reservation;
* fabricate a downstream experiment;
* rewrite lifecycle truth;
* reinterpret scientific results;
* bypass the owning service.
Specify reconciliation reason codes, evidence fields, bounded batch behavior, and restart safety.
Required Cancellation Model
Distinguish:
* stopping future Campaign Operations dispatch;
* cancelling an unbound request;
* cancelling a bound but not yet running lifecycle artifact;
* requesting cancellation of running lifecycle work;
* scheduler or worker termination;
* immutable terminal lifecycle evidence.
Campaign Operations may record and coordinate cancellation only through accepted lifecycle authority.
It must not directly signal worker processes merely because a campaign is cancelled.
Required Operational Completion Model
Define explicit completion prerequisites.
A campaign must not be operationally complete while any of the following remain unsettled:
* active operational authority requiring closure;
* outstanding reservations;
* accepted but unbound requests;
* dispatch attempts with ambiguous outcomes;
* bound downstream work without terminal lifecycle evidence;
* unresolved cancellation obligations;
* unresolved reconciliation observations that block closure;
* inconsistent budget settlement.
Completion must be append-only and idempotent.
Specify whether completion can be administratively overridden. Prefer no override unless an already accepted architecture authority exists.
PostgreSQL Requirements
Provide an implementation-oriented persistence design.
For each authoritative table, define:
* purpose;
* primary key;
* foreign keys;
* natural unique constraints;
* check constraints;
* immutable columns;
* append-only enforcement;
* indexes;
* runtime write role;
* read roles;
* ownership;
* revocation strategy;
* projection relationship.
Prefer:
* restrictive foreign keys;
* explicit unique constraints;
* integer accounting;
* append-only event tables;
* immutable provenance;
* database-enforced invariants;
* least-privilege runtime roles.
Do not rely only on application validation for correctness-critical invariants.
Documentation Authority and Acceptance
Correct stale architectural status language where the amendment legitimately owns that content.
Where ADR-0009 or related records still say Proposed despite an accepted implementation state, do not silently declare acceptance without repository authority.
Instead:
1. determine whether acceptance has already been explicitly established elsewhere;
2. align the architecture document with that evidence;
3. identify documentation-only acceptance corrections that must occur;
4. do not treat a merely proposed ADR as implementation authority;
5. do not invent approval evidence.
The amended document must clearly distinguish:
* operational implementation status;
* architectural acceptance status;
* remaining documentation-alignment work.

File-Change Scope

The primary required edit is the authoritative Campaign Operations architecture document.

Do not modify ADR-0009, the ADR index, Volumes VIII or XII, AGENTS.md, Phase 6 implementation documents, production code, migrations, tests, or project configuration during this task unless this prompt explicitly names those files as amendment targets.

Where related documents contain stale or inconsistent status language, record the exact required documentation-alignment actions in the amended architecture and in the final completion report. Do not silently edit those separate authority records as part of this task.
Implementation Sequencing
Provide bounded implementation increments.
Each increment must state:
* scope;
* new contracts;
* persistence changes;
* service changes;
* CLI or operator behavior;
* tests;
* migration effects;
* explicit exclusions;
* acceptance criteria;
* rollback or recovery considerations.
The sequence must preserve existing accepted behavior at every step.
Do not combine all Campaign Operations behavior into one implementation increment.
Testing Requirements
Define required tests for:
* canonical identity;
* repository constraints;
* transaction rollback;
* concurrent reservation attempts;
* duplicate requests;
* duplicate dispatch;
* crash-window recovery;
* binding reconstruction from deterministic evidence;
* pause and resume;
* cancellation races;
* reconciliation idempotency;
* completion races;
* privilege enforcement;
* migration upgrade;
* restart recovery;
* scheduler isolation;
* lifecycle ownership;
* audit completeness.
Tests must prove both positive behavior and prohibited behavior.
Editing Requirements
Make the smallest coherent set of changes necessary to produce a complete architecture.
You may substantially rewrite weak sections where incremental edits would preserve ambiguity.
Preserve valid material from the current architecture.
Remove or replace:
* contradictory statements;
* stale prospective language;
* unresolved placeholders;
* ambiguous authority language;
* duplicate definitions;
* implementation claims unsupported by repository evidence;
* roadmap speculation presented as accepted architecture.
Maintain a professional architecture-document structure with:
* title and status;
* authority statement;
* executive decision;
* scope;
* terminology;
* ownership;
* domain model;
* persistence model;
* transactions and concurrency;
* workflows;
* failure and recovery;
* authorization;
* audit;
* migration;
* testing;
* implementation sequencing;
* acceptance criteria;
* unresolved decisions, only where genuinely unresolved.
Prohibitions
Do not:
* implement production code;
* add migrations;
* modify tests;
* change scheduler behavior;
* queue experiments;
* activate campaigns;
* create a Phase 6E;
* redesign Recommendation Governance;
* redesign the scheduler;
* introduce autonomous recommendation selection;
* add adaptive budgeting;
* add forecasting;
* add profitability interpretation;
* add scientific policy;
* expand into the broader reserved Volume X automation roadmap;
* mark unresolved authority as accepted without evidence;
* merely write another review report.
Verification Before Completion
Before finishing:
1. compare the amended document against every finding in the focused CEE output;
2. compare it against every finding in the final-verification output;
3. search for contradictory definitions;
4. search for Proposed, proposed decision, Phase 6E, automatic, direct insert, scheduler, ratification, authorization, budget, reservation, reconciliation, and completion;
5. confirm that no positive evidence is treated as interchangeable with another;
6. confirm that all mutation workflows define durable authority, transaction boundaries, idempotency, and recovery;
7. confirm that Campaign Operations never bypasses the accepted experiment lifecycle;
8. confirm that scheduler isolation remains absolute;
9. confirm that the architecture does not claim scientific success from operational completion;
10. inspect the final diff for accidental unrelated edits.
Required Final Response
Return a concise completion report containing:
1. the architecture file amended;
2. the review evidence used;
3. a summary of the substantive corrections;
4. a finding-by-finding resolution table with:
    * finding;
    * resolution;
    * amended section;
    * status: resolved, partially resolved, or not resolved;
5. any genuinely unresolved architectural decisions;
6. files changed;
7. validation performed;
8. confirmation that no production implementation was changed.
Do not paste the entire amended architecture into the terminal response. The authoritative result must be the edited Markdown file.

