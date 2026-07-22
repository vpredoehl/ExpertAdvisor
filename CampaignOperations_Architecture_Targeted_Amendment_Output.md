---
title: "Campaign Operations Architecture Targeted Amendment"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Architecture_Targeted_Amendment_Prompt.md"
reasoning_effort: "xhigh"
model: "default"
---

# Campaign Operations Architecture Targeted Amendment

Completed the targeted architecture amendment in:

[CampaignOperations_Revised_Architecture_Output.md](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md>)

The document correctly remains a candidate baseline—not accepted implementation authority—because ADR-0009 is still Proposed and ADR-0010 is not yet accepted.

### Review evidence used

- [Focused CEE output](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/03_Focused_CEE/CampaignOperations_Revised_Architecture_Focused_CEE_Output.md>)
- [Final verification output](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_Final_Verification_CEE_Output.md>)
- Repository authority, ADR, Phase 4C/4D/5/6, lifecycle, scheduler, migration, privilege, and implementation evidence.

### Substantive corrections

- Froze V1 origin and membership to one exact Phase 4D materialization.
- Added campaign and logical-request uniqueness.
- Replaced competing budget representations with one versioned ledger.
- Fully specified identities, tables, constraints, privileges, indexes, and replay.
- Defined every mutation’s locks, transaction, commit, retry, and crash behavior.
- Separated dispatch acquisition/outcome, cancellation request/settlement, and reconciliation observation/resolution.
- Added explicit adoption authority and permanent V1 control ownership.
- Defined cancellation, restart recovery, failure classes, and disjoint completion precedence.
- Made reconciliation detection-only and scheduler isolation absolute.
- Added bounded implementation increments with per-phase ADR gates and acceptance criteria.

### Finding resolution

| Finding | Resolution | Amended section | Status |
|---|---|---|---|
| Campaign origin | Phase 4D-only V1 origin; ratification is provenance/prerequisite | §§1, 5, 10, 12, 35 | Resolved |
| Campaign uniqueness | Unique materialization binding, independent of provenance | §§11–12 | Resolved |
| Request idempotency | System-derived logical-operation key and database uniqueness | §§11–12, 16 | Resolved |
| Budget authority | One append-only grant/amend/revoke/supersede ledger | §§11, 14 | Resolved |
| Campaign-control evidence | Versioned append-only pause/resume events | §§11–13 | Resolved |
| Dispatch attempts | Separate immutable acquisition and outcome; audit-only | §§11–12, 16, 18 | Resolved |
| Cancellation settlement | Separate immutable request and settlement | §§11–12, 18, 21 | Resolved |
| Adoption/control ownership | Explicit adoption grant, unit consumption, permanent unique owner | §§10–12, 16, 21 | Resolved |
| Identity and replay | Complete identity catalog, expiry syntax, lifecycle-snapshot exception | §12 | Resolved |
| Lock ordering | Levels 1–5 held through atomic handoff commit | §§18–20, 26 | Resolved |
| Completion semantics | Disjoint precedence, no override, post-completion lifecycle behavior | §§13, 22 | Resolved |
| Reconciliation ownership | Detection-only with reason codes and owning-service resolution | §§18, 23 | Resolved |
| Scheduler enablement | A–D independent; E test-only until independently verified hardening | §§1, 17, 29, 31 | Resolved |
| Implementation sequencing | Per-increment ADR gates, tests, migration effects, rollback, acceptance | §§30–31 | Resolved |

### Genuinely unresolved decisions

V1 domain semantics are closed. External decisions still requiring their owning ADRs are:

- Institutional capability assignment and any additional human separation.
- Exact narrow PostgreSQL mechanism for invoking Phase 5.
- Scheduler atomic-claim implementation.
- Lifecycle authority for stopping running work.
- Retention and archival periods.

### Validation

- Confirmed the exact authoritative target existed before editing.
- Compared the amendment against every focused-CEE and final-verification finding.
- Performed all requested terminology/contradiction searches.
- Verified all 36 numbered sections, 14 balanced code fences, table structure, and absence of placeholders.
- `git diff --check`: passed.
- No build or executable tests run; this was documentation-only.
- No database, scheduler, worker, or production experiment state accessed.

### Files changed

Only the authoritative architecture Markdown was edited. No production code, migrations, tests, project configuration, ADRs, volumes, or scheduler behavior changed.

`git status --short`:

```text
 M ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md
?? ArchitectureReviews/CampaignOperations/03_Focused_CEE/CampaignOperations_Revised_Architecture_Focused_CEE_Output.md
?? ArchitectureReviews/Phase4/
?? ArchitectureReviews/Phase5/
?? ArchitectureReviews/Phase6/
?? CampaignOperations_Architecture_Targeted_Amendment_Prompt.md
?? CampaignOperations_Final_Verification_CEE_Output.md
?? CampaignOperations_Final_Verification_CEE_Prompt.md
```

The untracked review artifacts already existed and were not modified.

`git diff --stat`:

```text
ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md | 1304 +++++++++++++++-----
1 file changed, 1025 insertions(+), 279 deletions(-)
```