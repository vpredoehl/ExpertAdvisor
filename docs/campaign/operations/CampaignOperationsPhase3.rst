Campaign Operations Phase 3
===========================

Campaign Operations Phase 3 is architectural Phase E, Durable Dispatch and
Atomic Lifecycle Handoff. It consumes the Phase 2 ``ready`` request and
``held`` reservation. It does not enable production dispatch or implement
architectural Phases F, G, or H.

Implementation
--------------

``CampaignOperationsDispatchService`` owns two PostgreSQL transactions. The
short acquisition transaction locks authorization, budget, campaign,
reservation, and request in the accepted order; performs a request-version
compare-and-set to ``dispatching``; stores only a lease-token digest with a
database-owned expiry; and appends immutable attempt and audit evidence.

The handoff transaction reacquires and retains those locks, validates the
exact Phase 4D materialization, acquires the sorted Phase 5
proposal/review/execution domain, and then classifies existing Phase 4C/5
evidence. This prevents direct Phase 5 overlap from changing the evidence
between classification and invocation.
``LaunchRecommendationCampaignInTransaction`` reacquires those transaction
locks and continues through the activation/experiment domains. The handoff
atomically inserts the complete ordered
per-member binding and permanent V1 control-owner sets, commits the full held
reservation, moves the request to ``bound``, clears the lease, and appends
outcome/audit evidence. Deferred PostgreSQL constraints reject incomplete
bound state at commit.

Replay and recovery
-------------------

Every acquisition or handoff retry first reloads and validates authoritative
lease, binding, or outcome
evidence. A complete binding is returned without invoking Phase 5 again.
Durable ``dispatching`` lease/attempt evidence permits restart of an
uncommitted handoff. SQLSTATE ``40001`` and ``40P01`` retry the complete
affected transaction at the outer service boundary, at most three times with
jitter. Exhaustion returns
the stable ``transient_database_retry_exhausted`` failure classification with
``dispatch_retry_exhausted_<SQLSTATE>``; other SQLSTATE values are not
reclassified as replay or retryable conflicts. Partial, paused-only,
progressed, or ambiguous evidence is recorded fail-closed and is not repaired
by Phase 3 itself. Campaign Operations Phase 4 implements the separate Phase F
control, cancellation, and bounded reconciliation authority.

Existing pending work is adopted only when all materialized members already
have exact ``pending/train`` Phase 5 evidence and a separately active
``adopt_existing_pending_and_control`` grant exists. Adoption reuses no
ordinary-dispatch inference and records distinct binding/control
dispositions.

Safety and privileges
---------------------

Migration 048 creates separate NOLOGIN dispatcher and transactional Phase 5
capabilities. They are not granted to ``pqxx`` or a login principal. The
transactional role receives only the established Phase 5 columns and Phase 3
evidence/transition capabilities; it receives no scheduler or worker access.

The only execution adapter dispatches one explicit request and requires both
an exact connected database name beginning with
``expertadvisor_campaign_operations_phase3_test_`` and the literal
``I_UNDERSTAND_PHASE3_TEST_ONLY`` acknowledgement. It never polls, starts a
scheduler, launches/signals a worker, or enables production.
Deterministic fault hooks are accepted only by this isolated adapter and have
no production polling, configuration, CLI, or database control surface.
``production_dispatch_enabled`` remains constrained false. ADR-0016 therefore
remains an unsatisfied production gate.
