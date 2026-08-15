---
title: "Return Feature Scheduler Control Full Worker Identity Reset Correction"
document_type: "architecture review"
status: "final"
generated_from: "ReturnFeature_SchedulerControl_FullWorkerIdentityReset_Correction_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# Return Feature Scheduler Control Full Worker Identity Reset Correction

Implemented the complete lifecycle worker-identity reset for all three scheduler-control actions.

Reset fields now include:

- PID, PGID, process-start identity
- executable and command line
- `active_scheduler_worker_attempt_id`
- `worker_control_state`
- `worker_global_pause_request_id`

Added and passed isolated integration coverage for retry, both requeues, dry-run, historical-attempt preservation, unrelated rows, and fresh replacement attempts.

Release build passed. The existing ownership migration test remains blocked by its unrelated fixture error (`model.name` missing).

Detailed report: [ReturnFeature_SchedulerControl_FullWorkerIdentityReset_Correction_Output.md](</Volumes/Developer SSD/ExpertAdvisor/ReturnFeature_SchedulerControl_FullWorkerIdentityReset_Correction_Output.md>)

Production processes and database state were left undisturbed.