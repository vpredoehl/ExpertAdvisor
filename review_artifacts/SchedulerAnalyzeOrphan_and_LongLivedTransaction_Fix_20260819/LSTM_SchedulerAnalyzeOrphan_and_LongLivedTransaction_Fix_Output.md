---
title: "LSTM Scheduler Analyze Orphan and Long-Lived Transaction Fix"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_SchedulerAnalyzeOrphan_and_LongLivedTransaction_Fix_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Scheduler Analyze Orphan and Long-Lived Transaction Fix

Implemented both reliability fixes without touching recommendation-evaluation code or production rows.

### Changes

- [Sources/ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:16169)
  - Root cause: orphan recovery only accepted an experiment still in `running/analyze`. Final analysis persistence had already advanced it to `completed/done`, so the exact bound attempt was skipped.
  - Recovery now recognizes that completed/done crash window while preserving exact attempt ownership.
  - It reuses `HasCompletedAnalysisResultForAttempt`.
  - Completed durable evidence terminalizes the attempt as:
    `completed / process_missing_result_recovered / exact_process_identity_absent`.
  - Without durable evidence, the attempt follows the existing failed/no-result path.

- [LSTM/main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp:7288)
  - Root cause: one `pqxx::work` transaction covered startup configuration, model/tensor setup, and potentially training.
  - Resume, configuration, Forex metadata, and Forex tensor reads now use short, read-only transactions.
  - The runtime model transaction commits before the epoch loop at [main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp:7801).
  - Per-symbol Forex transactions preserve multi-symbol behavior.

- [SchedulerAnalyzeOrphanRecoveryTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/SchedulerAnalyzeOrphanRecoveryTests.sh:1)
  - Adds disposable-database regression coverage.

### Verification

Passed:

```text
SchedulerAnalyzeOrphanRecoveryTests passed
SchedulerTrainOrphanCheckpointRecoveryTests passed
```

The new test proves:

- Completed final analysis recovers the stale attempt as completed.
- Missing durable analysis does not falsely complete.
- Active analyze reservations drop from 2 to 0.
- A pending checkpoint analysis is subsequently claimed and completed normally.

Also passed:

```text
bash -n Tests/SchedulerAnalyzeOrphanRecoveryTests.sh
git diff --check
```

A full Debug build of the `LSTM Release` target succeeded:

```text
** BUILD SUCCEEDED **
```

It emitted 93 existing-style warnings, primarily libpqxx deprecated API warnings.

### Release build

The requested isolated Release build did **not** succeed. It exited 65 before source compilation because [GenerateBuildProvenance.py](/Volumes/Developer%20SSD/ExpertAdvisor/Scripts/GenerateBuildProvenance.py:37) rejects any Release build from a dirty worktree:

```text
Release provenance requires a clean source tree
```

There is no documented override. Since committing was explicitly disallowed, bypassing that provenance control would be inappropriate. The temporary Release DerivedData directory was removed.

After these changes are committed, rerun exactly:

```bash
rm -rf /tmp/ExpertAdvisor_LSTM_Reliability_Fix_DerivedData
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath /tmp/ExpertAdvisor_LSTM_Reliability_Fix_DerivedData \
  build
```

### Remaining runtime verification

I did not launch a training worker because production scheduler/training processes were active. During a controlled patched training run, prefix the normal launch with:

```bash
PGAPPNAME=ea_training_txn_verify
```

Then monitor during the epoch loop:

```bash
watch -n 2 "psql -X -d LSTM -c \"
SELECT pid,datname,state,xact_start,backend_xid,backend_xmin,
       wait_event_type,wait_event,left(query,160)
FROM pg_stat_activity
WHERE application_name='ea_training_txn_verify'
ORDER BY datname,pid;\""
```

During model computation, idle connections should have `xact_start`, `backend_xid`, and `backend_xmin` all null—not `idle in transaction`. Brief active transactions during checkpoint persistence are expected.

### Repository state

Current branch: `lstm-feature-development`

```text
 M LSTM/main.cpp
 M Sources/ExperimentScheduler.cpp
?? Tests/SchedulerAnalyzeOrphanRecoveryTests.sh
```

`git diff --stat`:

```text
 LSTM/main.cpp                   | 109 +++++++++++++++++++++++++---------------
 Sources/ExperimentScheduler.cpp |  29 ++++++++---
 2 files changed, 90 insertions(+), 48 deletions(-)
```

The untracked regression script contains 187 lines and is not included by `git diff --stat`. No commit or push was performed.