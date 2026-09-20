---
title: "Phase 22M EA::LSTM Per-Instance Geometry Audit"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase22M_LSTMPerInstanceGeometryAudit_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# Phase 22M EA::LSTM Per-Instance Geometry Audit

Audit complete: **GO WITH PREREQUISITES**.

Created the sole unstaged artifact:

[LSTM_Phase22M_LSTMPerInstanceGeometryAudit_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM_Phase22M_LSTMPerInstanceGeometryAudit_Output.md)

Key conclusion: internalize immutable per-instance hidden size; treat `n_out` as derived `H`, not independent configuration. Label, optimizer, evaluation, logging, DB, scheduler, and Metal-runtime policy remain outside geometry.

Validation: `git diff --check` passed. No build or production-connected execution was run.

Files changed: only the report.

`git status --short`:

```text
?? LSTM_Phase22M_LSTMPerInstanceGeometryAudit_Output.md
```

`git diff --stat`: empty (the report is untracked).