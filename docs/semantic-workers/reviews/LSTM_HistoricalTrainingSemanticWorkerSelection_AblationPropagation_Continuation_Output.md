---
title: "LSTM Historical Training Semantic Worker Selection and Ablation Propagation Continuation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_HistoricalTrainingSemanticWorkerSelection_AblationPropagation_Continuation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Historical Training Semantic Worker Selection and Ablation Propagation Continuation

Implemented and verified the TRAIN feature-ablation propagation fix while preserving the existing historical semantic-worker routing work.

### Behavior

Root cause: `ExperimentRow` loaded and canonicalized `feature_ablation_mask`, but `BuildTrainCommand()` never copied it into the child argv.

The scheduler now initializes TRAIN argv with the selected executable and persisted ablation identity through [TrainingWorkerCommand.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/SchedulerCore/TrainingWorkerCommand.hpp:9). [BuildTrainCommand()](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/SchedulerCore/ProductionSchedulerDaemon.cpp:3485) invokes this before the fresh/resume branch.

- Empty mask: no `--ablate-features` option.
- Nonempty mask: exact `--ablate-features=<canonical-mask>`.
- Applies to fresh and resumed training.
- No reordering or reconstruction occurs during command construction.
- Existing checkpoint/resume compatibility validation remains unchanged.
- INFER and ANALYZE behavior was not broadened.

The combined disposable regression constructs historical layout-8/80 and current layout-9/103 artifacts, selects the historical TRAIN worker, propagates the canonical ablation mask, and proves the newer executable is absent from argv.

### Verification

All passed:

```text
bash Tests/SchedulerTrainingWorkerRoutingTests.sh
bash Tests/SemanticWorkerRegistryTests.sh
bash Tests/SchedulerDaemonConfigurationTests.sh
bash Tests/FinalExperimentDispatchServiceTests.sh
bash Tests/SchedulerSemanticAdmissionTests.sh
bash Tests/WorkerAttemptLifecycleServiceTests.sh
bash Tests/SchedulerAnalyzeWorkerRoutingTests.sh
bash Tests/LaunchArgumentsFreshInitializationSeedTests.sh
```

Coverage includes:

- Fresh control and ablation TRAIN commands.
- Resumed control and ablation TRAIN commands.
- Canonical mask preservation and rejection of alternate text in argv.
- Historical TRAIN selection plus ablation propagation in one regression.
- Worker-attempt provenance.
- Independent INFER selection.
- Unchanged ANALYZE routing.

Debug build:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Debug" \
  -configuration Debug -derivedDataPath DerivedData/ExpertAdvisor build
```

Result: exit 0. Existing warnings remained for LLVM23 metadata, libpqxx deprecations, and libomp deployment-version mismatch; none originated from this change.

`git diff --check`: passed. Untracked files were also checked with `git diff --no-index --check`; no whitespace diagnostics.

No PostgreSQL-backed end-to-end dispatch was run because the available integration fixtures clone the live database schema. The synthetic registry/unit coverage required no database access.

### Files

Continuation-specific changes:

- [ProductionSchedulerDaemon.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/SchedulerCore/ProductionSchedulerDaemon.cpp:3485)
- [TrainingWorkerCommand.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/SchedulerCore/TrainingWorkerCommand.hpp:1)
- [SemanticWorkerRegistryTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/SemanticWorkerRegistryTests.cpp:236)
- [SchedulerTrainingWorkerRoutingTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/SchedulerTrainingWorkerRoutingTests.sh:1)
- [semantic-layout-inference-worker-routing.md](/Volumes/Developer%20SSD/ExpertAdvisor/docs/semantic-workers/semantic-layout-inference-worker-routing.md:96)

Existing routing WIP files were preserved.

`git status --short`:

```text
 M Sources/SchedulerCore/ProductionSchedulerDaemon.cpp
 M Sources/SchedulerCore/SemanticWorkerRegistry.cpp
 M Tests/SemanticWorkerRegistryTests.cpp
 M Tests/SemanticWorkerRegistryTests.sh
 M Tests/WorkerAttemptLifecycleServiceTests.cpp
 M docs/semantic-workers/semantic-layout-inference-worker-routing.md
?? LSTM_HistoricalTrainingSemanticWorkerSelection_Regression_Output.md
?? Sources/SchedulerCore/TrainingWorkerCommand.hpp
?? Sources/SchedulerCore/TrainingWorkerSelection.hpp
?? Tests/SchedulerTrainingWorkerRoutingTests.sh
```

`git diff --stat`:

```text
 .../SchedulerCore/ProductionSchedulerDaemon.cpp    | 144 ++++++++--
 Sources/SchedulerCore/SemanticWorkerRegistry.cpp   |  37 +++
 Tests/SemanticWorkerRegistryTests.cpp              | 306 ++++++++++++++++++++-
 Tests/SemanticWorkerRegistryTests.sh               |   3 +-
 Tests/WorkerAttemptLifecycleServiceTests.cpp       |  19 +-
 .../semantic-layout-inference-worker-routing.md    |  52 +++-
 6 files changed, 521 insertions(+), 40 deletions(-)
```

The stat excludes the four untracked files.

No live database, scheduler, worker, experiment 668/669, replication materializer, production Release build, artifact publication, commit, or push action was performed. The production scheduler and workers were left stopped.