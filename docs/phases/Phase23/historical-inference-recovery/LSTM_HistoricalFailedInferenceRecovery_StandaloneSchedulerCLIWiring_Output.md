---
title: "LSTM Historical Failed Inference Recovery Standalone Scheduler CLI Wiring"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_HistoricalFailedInferenceRecovery_StandaloneSchedulerCLIWiring_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Historical Failed Inference Recovery Standalone Scheduler CLI Wiring

Implemented standalone historical failed-inference recovery CLI wiring without linking `ExperimentScheduler.o`.

Files changed:

- [GlobalExperimentControl.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.hpp>)
- [GlobalExperimentControl.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp>)
- [SchedulerMain.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/SchedulerMain.cpp>)
- [HistoricalFailedInferenceRecoveryIntegrationTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/HistoricalFailedInferenceRecoveryIntegrationTests.sh>)
- [StandaloneSchedulerExecutableTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/StandaloneSchedulerExecutableTests.sh>)

Behavioral changes:

- Added standalone recognition/parsing for both `--recover-failed-inference VALUE` and `--recover-failed-inference=VALUE`.
- Enforces one positive experiment ID, exactly one of `--dry-run`/`--yes`, and rejects duplicates and unknown arguments.
- Delegates to the existing `RunHistoricalFailedInferenceRecoveryCommand`.
- Dispatches directly from `SchedulerMain.cpp`; legacy `ExperimentScheduler` compatibility remains unchanged.
- Integration tests now default to `lstm-scheduler`.
- Standalone architecture checks remain intact and now cover recovery parsing.

Validation:

- `bash -n Tests/StandaloneSchedulerExecutableTests.sh Tests/HistoricalFailedInferenceRecoveryIntegrationTests.sh` — passed.
- Focused Debug `lstm-scheduler` build — passed.
- `EA_BUILD_CONFIGURATION=Debug Tests/StandaloneSchedulerExecutableTests.sh` — passed; no `ExperimentScheduler.o` extraction or forbidden symbols.
- `Tests/ProductionSchedulerCompositionLinkHarness.sh` — passed.
- `Tests/HistoricalFailedInferenceRecoveryIntegrationTests.sh .../Debug/lstm-scheduler` — passed against its disposable database.
- `git diff --check` — passed.

The standard Release build was attempted but correctly stopped at `GenerateBuildProvenance.py` because the requested uncommitted changes make the source tree dirty. That enforcement was not weakened or bypassed. The changed production composition also compiled under the harness’s `-Werror` settings.

No production recovery was executed. Experiments 650 and 651 were untouched. No commit was created.

`git status --short`:

```text
 M LSTM/SchedulerMain.cpp
 M Sources/GlobalExperimentControl.cpp
 M Sources/GlobalExperimentControl.hpp
 M Tests/HistoricalFailedInferenceRecoveryIntegrationTests.sh
 M Tests/StandaloneSchedulerExecutableTests.sh
```

`git diff --stat`:

```text
 LSTM/SchedulerMain.cpp                             |  12 +++
 Sources/GlobalExperimentControl.cpp                | 115 +++++++++++++++++++++
 Sources/GlobalExperimentControl.hpp                |   9 ++
 ...ricalFailedInferenceRecoveryIntegrationTests.sh |   6 +-
 Tests/StandaloneSchedulerExecutableTests.sh        |  49 ++++++++-
 5 files changed, 187 insertions(+), 4 deletions(-)
```