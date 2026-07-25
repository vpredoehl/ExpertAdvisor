---
title: "Campaign Operations Final Verification CEE Prompt"
document_type: "architecture review prompt"
status: "completed"
reasoning_effort: "xhigh"
---

# Campaign Operations Final Verification CEE Prompt

Using xhigh reasoning, perform a focused verification review of the amended Campaign Operations architecture.

INPUT

CampaignOperations_Revised_Architecture_Output.txt

GOAL

Verify that every required targeted correction identified by the previous focused CEE has been correctly incorporated without introducing architectural regressions.

This is a verification review, not a redesign.

Do not redesign the subsystem.
Do not broaden scope.
Do not introduce new features.
Do not implement code.
Do not modify repository files.

Review only whether the architecture now satisfies the previously identified required targeted corrections.

Evaluate:

- campaign uniqueness
- campaign origin
- authorization
- budget authority
- request idempotency
- dispatch attempts
- bindings
- reconciliation ownership
- completion semantics
- cancellation semantics
- identity and replay
- scheduler isolation
- transaction ordering
- lock ordering
- implementation sequencing
- terminology consistency

Classify each correction as:

- Resolved
- Partially Resolved
- Not Resolved

OUTPUT

Produce exactly these sections:

1. Executive Summary
2. Overall Disposition
3. Resolved Corrections
4. Remaining Required Corrections
5. New Issues Introduced
6. Readiness for Phase A Baseline
7. Review Verification

Do not redesign the architecture.
If no new architectural defects are found beyond unresolved targeted corrections, explicitly state that no further broad architectural review is recommended.
