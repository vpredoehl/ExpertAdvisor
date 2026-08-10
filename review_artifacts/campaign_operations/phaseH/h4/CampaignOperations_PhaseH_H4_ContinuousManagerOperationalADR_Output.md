---
title: "Campaign Operations Phase H H4 Continuous Manager Operational ADR"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H4_ContinuousManagerOperationalADR_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Campaign Operations Phase H H4 Continuous Manager Operational ADR

Drafted the proposed H4 operational ADR and report.

- [ADR-0020](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0020-campaign-manager-continuous-operation.md>) selects an external-supervisor model that repeatedly invokes bounded H3 run-once; no in-process daemon or new CLI.
- [ADR index](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/README.md>) now lists it as `Proposed`.
- [H4 output report](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_H4_ContinuousManagerOperationalADR_Output.md>) contains the required baseline, traceability, validations, and gate token.

Validation: `git diff --check`, `git diff --cached --check`, and untracked-file whitespace checks passed. No build/tests run; active scheduler and training processes were left untouched. No runtime, schema, database, ACL, migration, backup, commit, or push occurred.

`git status --short`:

```text
 M docs/architecture/adr/README.md
?? CampaignOperations_PhaseH_H4_ContinuousManagerOperationalADR_Output.md
?? CampaignOperations_PhaseH_H4_Kickoff_Output.md
?? docs/architecture/adr/ADR-0020-campaign-manager-continuous-operation.md
```

`git diff --stat`: `docs/architecture/adr/README.md | 2 ++` (new ADR/report are intentionally untracked).

Remaining risk: ADR-0020 is not accepted and must undergo independent review before any H4 implementation.