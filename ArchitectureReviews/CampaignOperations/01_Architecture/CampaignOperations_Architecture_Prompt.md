Using High reasoning, perform a complete architectural design for the new Campaign Operations subsystem.

GOAL

Design the architecture that follows the completed Phase 6 governance work.

Do not implement code.

Do not modify existing production behavior.

Do not redesign completed Phases 4–6.

Treat all completed authority chains as fixed architecture.

BACKGROUND

The repository currently provides:

Planning
→ Review
→ Approval
→ Materialization
→ Execution
→ Follow-up Review
→ Governance

The governance chain is complete.

The next architectural concern is long-lived campaign operations.

Campaign Operations must remain a separate subsystem and must not transfer policy authority into the scheduler.

OBJECTIVES

Design the complete Campaign Operations architecture, including:

• subsystem responsibilities
• ownership boundaries
• lifecycle state machine
• operational entities
• database model
• repository interfaces
• service layer responsibilities
• scheduler interaction contract
• operator interaction
• reconciliation
• restart recovery
• audit requirements
• budgeting concepts
• reservation concepts
• execution requests
• completion semantics
• cancellation semantics
• deterministic replay
• failure recovery
• concurrency model
• transactional boundaries
• privilege model

Clearly identify:

• what belongs inside Campaign Operations
• what remains inside the scheduler
• what remains inside recommendation governance
• what belongs nowhere

REQUIREMENTS

Do not implement code.

Do not create migrations.

Do not design CLI commands.

Do not invent APIs unless required to explain architecture.

Do not redesign existing repository components.

Do not introduce automatic experiment creation.

Do not introduce autonomous decision making.

Do not change completed authority boundaries.

Produce:

1. Executive summary.
2. Architectural rationale.
3. Complete subsystem decomposition.
4. State machine.
5. Component interaction diagrams (textual).
6. Data ownership.
7. Transaction boundaries.
8. Failure recovery model.
9. Concurrency analysis.
10. Security / privilege analysis.
11. ADR recommendations.
12. Recommended implementation phases.
13. Risks.
14. Open architectural questions.

The deliverable should be an architecture document suitable for implementation planning.
