---
title: "Scheduler Restart Multi-Instance Worker Ownership Focused Independent CEE Review"
document_type: "architecture review"
status: "final"
generated_from: "Scheduler_Restart_MultiInstance_WorkerOwnership_Focused_Independent_CEE_Review_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Scheduler Restart Multi-Instance Worker Ownership Focused Independent CEE Review

Completed the independent CEE review. The full report is here:

[Focused Independent CEE Review](</Volumes/Developer SSD/ExpertAdvisor/Scheduler_Restart_MultiInstance_WorkerOwnership_Focused_Independent_CEE_Review_Output.md>)

Verdict:

- Ready to commit: **No**
- Ready for production: **No**
- Fully covered scenarios: **17/40**
- Partially covered: **17/40**
- Missing: **6/40**
- Manual-only: **0/40**

Primary blockers:

- Continuation evaluation/queueing can mutate after lease loss.
- Signaling and child reaping are not consistently bound to the exact durable worker attempt.
- Direct CLI worker/analysis paths bypass ownership and capacity controls.
- Migration 051 does not prevent unsafe old/new scheduler coexistence.
- Lock ordering contains reachable cycles.
- Checkpoint analysis holds broad locks in an unbounded transaction and lacks a durable analyze attempt.
- Legacy attempts without a PID can consume capacity indefinitely.

Validation independently executed:

- Release build: passed.
- Canonical-path tests: passed.
- Migration/repository tests: passed.
- Scheduler ownership process integration: passed.
- Global pause/resume/cancel integration: passed.
- Relevant ASAN/UBSAN tests: passed.
- Shell syntax and `git diff --check`: passed.
- No disposable test processes remained.
- Production scheduler and workers were not modified.

No implementation correction was made; the only file added by this review is the report. The identified defects require coordinated targeted fixes and regression tests rather than a safe isolated patch.

Final `git status --short` includes the existing implementation changes, the new report, and an externally created `watch_20260729-142309` artifact. Tracked diff remains:

```text
9 files changed, 3033 insertions(+), 294 deletions(-)
```

The report and other new implementation files are untracked and therefore are not included in that `git diff --stat`.