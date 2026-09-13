---
title: "LSTM Phase 20H Scheduler Experiment Transition Decomposition"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase20H_SchedulerExperimentTransitionDecomposition_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Phase 20H Scheduler Experiment Transition Decomposition

Implemented Phase 20H as one extraction: operator-driven experiment cancel/retry/requeue orchestration now lives in `ExperimentTransitionService`.

Key results:

- `ExperimentScheduler.cpp`: 28,407 → 27,984 lines, a 423-line reduction.
- PostgreSQL locking and transition updates moved behind `PostgresSchedulerRepository`.
- Existing retry checkpoint promotion, training requeue predicates, worker-attempt protections, dry-run behavior, logging, and forced inference rerun semantics are preserved.
- The accepted `bf7366e` semantic-admission/model-identity behavior remains untouched.
- Work remains unstaged.

The full report is [LSTM_Phase20H_SchedulerExperimentTransitionDecomposition_Output.md](</Volumes/Developer SSD/ExpertAdvisor/LSTM_Phase20H_SchedulerExperimentTransitionDecomposition_Output.md>).

Validation passed:

- `ExperimentTransitionServiceTests.sh`
- `PostgresSchedulerRepositoryTests.sh`
- `SchedulerInternalSeamTests.sh`
- `SchedulerOrchestrationServiceTests.sh`
- `WorkerAttemptLifecycleServiceTests.sh`
- `SchedulerSemanticAdmissionTests.sh`
- Xcode project lint
- Focused `SchedulerCore` Release build
- `git diff --check`

The focused build retained existing `-Ofast` and legacy `exec_params` warnings; the newly added pqxx code uses the 7.10 API.

Not run:

- Full `LSTM Release` build and repeat-build measurement.
- Binary-level training-requeue and retry-checkpoint-promotion integrations.

Standalone scheduler extraction is not yet justified. Phase 20I should extract the remaining checkpoint-policy/checkpoint-evaluation orchestration.

Intended Phase 20H file set:

```text
ExpertAdvisor.xcodeproj/project.pbxproj
Sources/SchedulerCore/ExperimentScheduler.cpp
Sources/SchedulerCore/ExperimentTransitionService.cpp
Sources/SchedulerCore/ExperimentTransitionService.hpp
Sources/SchedulerCore/PostgresSchedulerRepository.cpp
Sources/SchedulerCore/PostgresSchedulerRepository.hpp
Sources/SchedulerCore/SchedulerRepository.hpp
Tests/ExperimentTransitionServiceTests.cpp
Tests/ExperimentTransitionServiceTests.sh
Tests/PostgresSchedulerRepositoryTests.cpp
Tests/SchedulerInternalSeamTests.cpp
Tests/SchedulerOrchestrationServiceTests.cpp
Tests/WorkerAttemptLifecycleServiceTests.cpp
LSTM_Phase20H_SchedulerExperimentTransitionDecomposition_Output.md
```

Current `git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/SchedulerCore/ExperimentScheduler.cpp
 M Sources/SchedulerCore/PostgresSchedulerRepository.cpp
 M Sources/SchedulerCore/PostgresSchedulerRepository.hpp
 M Sources/SchedulerCore/SchedulerRepository.hpp
 M Tests/PostgresSchedulerRepositoryTests.cpp
 M Tests/SchedulerInternalSeamTests.cpp
 M Tests/SchedulerOrchestrationServiceTests.cpp
 M Tests/WorkerAttemptLifecycleServiceTests.cpp
?? LSTM_Phase20H_SchedulerExperimentTransitionDecomposition_Output.md
?? Sources/SchedulerCore/ExperimentTransitionService.cpp
?? Sources/SchedulerCore/ExperimentTransitionService.hpp
?? Tests/ExperimentTransitionServiceTests.cpp
?? Tests/ExperimentTransitionServiceTests.sh
```

Current `git diff --stat` excludes untracked new files:

```text
9 files changed, 455 insertions(+), 526 deletions(-)
```

Terminal handoff:

```bash
git diff --check

git add -- \
  ExpertAdvisor.xcodeproj/project.pbxproj \
  Sources/SchedulerCore/ExperimentScheduler.cpp \
  Sources/SchedulerCore/ExperimentTransitionService.cpp \
  Sources/SchedulerCore/ExperimentTransitionService.hpp \
  Sources/SchedulerCore/PostgresSchedulerRepository.cpp \
  Sources/SchedulerCore/PostgresSchedulerRepository.hpp \
  Sources/SchedulerCore/SchedulerRepository.hpp \
  Tests/ExperimentTransitionServiceTests.cpp \
  Tests/ExperimentTransitionServiceTests.sh \
  Tests/PostgresSchedulerRepositoryTests.cpp \
  Tests/SchedulerInternalSeamTests.cpp \
  Tests/SchedulerOrchestrationServiceTests.cpp \
  Tests/WorkerAttemptLifecycleServiceTests.cpp \
  LSTM_Phase20H_SchedulerExperimentTransitionDecomposition_Output.md

git diff --cached --name-status
git diff --cached --stat
git status --short

git commit -m "Extract scheduler experiment transition service"

set -o pipefail
phase20h_build_1="/tmp/lstm-phase20h-build-1.log"
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath "$PWD/DerivedData/Release" \
  build 2>&1 | tee "$phase20h_build_1"

Tests/SchedulerTrainingRequeueIntegrationTests.sh \
  "$PWD/DerivedData/Release/Build/Products/Release/LSTM_Release"

Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh \
  "$PWD/DerivedData/Release/Build/Products/Release/LSTM_Release"

phase20h_build_2="/tmp/lstm-phase20h-build-2.log"
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath "$PWD/DerivedData/Release" \
  build 2>&1 | tee "$phase20h_build_2"

for phase20h_log in "$phase20h_build_1" "$phase20h_build_2"; do
  printf '%s\n' "$phase20h_log"
  awk '
    /^CompileC / { compile_c++ }
    /^Libtool /  { libtool++ }
    /^Ld /       { ld++ }
    END {
      printf "CompileC=%d\nLibtool=%d\nLd=%d\n",
             compile_c+0, libtool+0, ld+0
    }
  ' "$phase20h_log"
done

printf '%s\n' "Repeated-build recompiled sources:"
rg '^CompileC ' "$phase20h_build_2" |
  sed -E 's|^CompileC .* (/Volumes/Developer\\ SSD/ExpertAdvisor/.*) normal (arm64|x86_64) .*$|\1|' |
  sed 's/\\ / /g' |
  sort -u

git status --short
```