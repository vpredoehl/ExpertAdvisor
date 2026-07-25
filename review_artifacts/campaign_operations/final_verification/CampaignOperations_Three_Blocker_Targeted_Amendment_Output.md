---
title: "Campaign Operations Three-Blocker Targeted Amendment"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Three_Blocker_Targeted_Amendment_Prompt.txt"
reasoning_effort: "xhigh"
model: "default"
---

# Campaign Operations Three-Blocker Targeted Amendment

## 1. Executive summary

Amended the candidate architecture to version `1.1-candidate`, correcting only the three specified defects. It is ready for a focused independent verification and remains non-authoritative pending existing ADR gates.

## 2. Corrections made

1. Cancellation locking:
   - Unbound cancellation with reservation release now uses budget → campaign → reservation → request.
   - Authorization precedes budget only when that workflow requires it.
   - No workflow may acquire budget after campaign.
   - Reservation settlement is serialized against all specified budget and commitment operations.
   - Bound-work cancellation releases all Campaign Operations locks before calling lifecycle authority.
   - Immutable cancellation request/settlement separation and existing failure behavior remain unchanged.

2. Authorization supersession:
   - Persisted kinds are exactly `granted`, `revoked`, and `expiry_observed`.
   - `supersede` is a service transition that inserts one successor `granted` event.
   - The successor names the prior head and increments the chain once.
   - No `superseded` event row exists.
   - Identity, effective-head evaluation, replay, conflicts, repositories, tests, and ADR guidance now use this model consistently.

3. Initial lifecycle authority:
   - Campaign creation transitions directly from none to `awaiting_operational_authorization`.
   - The immutable campaign row and creation/audit evidence establish the initial state.
   - Restart derives that state from the campaign row.
   - `drafted` is permitted only as an optional pre-commit in-memory concept.
   - No readiness table, generalized creation event, or mutable status column was introduced.

## 3. Sections changed

Document metadata and §§5, 7, 9–15, 18–24, 26–31, 34, 36, and the revision history.

## 4. Scope confirmation

No unrelated architecture was redesigned. No production code, migrations, tests, ADRs, CLI commands, project configuration, database state, scheduler behavior, workers, or experiment rows were changed.

## 5. Validation

Full-document contradiction searches found:

- No durable `none → drafted` or `drafted → awaiting_operational_authorization`.
- No persisted authorization `superseded` kind or two-event supersession.
- No held-reservation release transaction acquiring campaign before budget.
- Remaining `drafted`/`superseded` matches are explicit prohibitions, historical descriptions, budget terminology, or test assertions.

Commands:

- `git diff --check` — passed, exit 0.
- `git diff --name-only` — only the target document.
- Builds/executable tests — not run, as this was documentation-only.

## 6. Files changed

[CampaignOperations_Revised_Architecture_Output.md](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md:1>)

`git status --short`:

```text
 M ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md
?? ArchitectureReviews/CampaignOperations/03_Focused_CEE/CampaignOperations_Revised_Architecture_Focused_CEE_Output.md
?? ArchitectureReviews/Phase4/
?? ArchitectureReviews/Phase5/
?? ArchitectureReviews/Phase6/
?? CampaignOperations_Architecture_Targeted_Amendment_Output.md
?? CampaignOperations_Architecture_Targeted_Amendment_Prompt.md
?? CampaignOperations_Final_Independent_Verification_Output.md
?? CampaignOperations_Final_Verification_CEE_Output.md
?? CampaignOperations_Final_Verification_CEE_Prompt.md
```

The unrelated untracked paths were not touched.

`git diff --stat`:

```text
 ...mpaignOperations_Revised_Architecture_Output.md | 1498 ++++++++++++++++----
 1 file changed, 1193 insertions(+), 305 deletions(-)
```

This stat includes the target document’s full pre-existing worktree delta from `HEAD`, not solely this narrow amendment.

## 7. Remaining unresolved decisions

None within the three targeted findings. Existing external governance and scheduler gates—including ADR-0009, ADR-0010 onward, and scheduler hardening—remain unchanged.