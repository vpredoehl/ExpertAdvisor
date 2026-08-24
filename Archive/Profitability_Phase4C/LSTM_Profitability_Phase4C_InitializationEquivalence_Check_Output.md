---
title: "LSTM Profitability Phase 4C Initialization Equivalence Check"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase4C_InitializationEquivalence_Check_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 4C Initialization Equivalence Check

## Verdict: IDENTICAL DETERMINISTIC INITIALIZATION

The proposed USD/CAD H6 control/treatment pair is scientifically clean with respect to fresh-model random initialization.

All trainable parameters shared by both objectives are byte-for-byte identical. The treatment has one additional active auxiliary regression head, initialized deterministically to fixed constants. Thus the complete active parameter sets differ intentionally by objective, but there is no random-initialization confound.

## Initialization path

Scheduler path:

`RunSchedulerOnce` → `RunTrainJobs` → `LoadPendingExperiments` → `BuildTrainCommand` → `LaunchReservedChildProcess` → `fork` → `execv(LSTM_Release)`.

Relevant evidence:

- The fresh command carries experiment/objective identity but no model ID: [ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:9473)
- Each worker is a newly executed process image: [ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:9249)
- The worker resolves the persisted objective: [main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp:7507)
- It constructs `EA::LSTM`, then applies `SetTrainingObjective`: [main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp:7901)
- Fresh training does not load the latest model because `load_latest=false`: [BuildConfig.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/BuildConfig.hpp:18)
- The first optimizer update occurs later through `CalculateBatch`: [main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp:8172)

## RNG and parameter initialization

The only production RNG affecting model initialization is:

```cpp
thread_local std::mt19937 rng{42};
std::uniform_real_distribution<float> dist(low, high);
```

See [LSTM.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:756).

It is a fixed literal seed, not configuration-, experiment-, model-, PID-, clock-, or random-device-derived. Because the scheduler uses `execv`, every fresh worker begins with a new RNG at seed 42. Sequential constructions in one process would consume an advancing stream and differ, which is why the regression test uses separate processes.

| Parameter family | Initialization |
|---|---|
| Fused LSTM/core `param` | 117×256, fixed stream, uniform `[-0.01, 0.01]` |
| Core `bias` | Zero, then forget-gate columns set to `1.5` |
| Direction-head weight | 64×3, continuation of fixed stream, uniform `[-0.05, 0.05]` |
| Direction-head bias | Exact zero |
| Treatment auxiliary weight | 64×1, exact `0.01f` |
| Treatment auxiliary bias | 1×1, exact `0.0f` |
| Recurrent hidden/cell state | Exact zero; not trainable |
| Optimizer/completed-epoch counters | Exact zero; plain SGD has no random/momentum tensors |

The core/direction initialization is implemented at [LSTM.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:3059). Objective selection occurs afterward. Only the auxiliary objective initializes and activates the scalar head: [LSTM.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp:3195).

Under legacy classification, those scalar-head matrices exist as storage but are not explicitly initialized, used for loss, or updated. They are therefore not part of the control arm’s active trainable parameter set.

## Inputs that can affect initialization

| Input | Effect |
|---|---|
| Experiment ID | None |
| Model ID | None for fresh training; none is passed or loaded |
| PID | None |
| Wall clock | None |
| Queue order | None; each worker `execv`s a fresh image |
| Objective identity | No effect on shared parameters; deterministically activates/fills only the treatment auxiliary head |
| Architecture/input width | Affects shapes and draw count; identical in the proposed pair |
| Different binary/standard library | Could change the distribution mapping across builds; use the same binary for both arms |

## Empirical verification

Added [FreshModelInitializationTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/FreshModelInitializationTests.cpp:61) and [FreshModelInitializationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/FreshModelInitializationTests.sh:38).

The test:

- Compiles the current production `LSTM.cpp`.
- Constructs USD/CAD models with the production default 53-input/64-hidden architecture.
- Uses the production order: constructor, then objective selection.
- Launches four distinct fresh processes in both objective orders.
- Compares all 30,403 shared trainable floats, recurrent state, and counters by exact binary `cmp`.
- Verifies the 65 treatment-only auxiliary parameters exactly, without tolerance.
- Performs no forward pass or optimizer update.

Results:

- `./Tests/FreshModelInitializationTests.sh` — passed.
- `./Tests/TrainingObjectiveTests.sh` — passed.
- `bash -n Tests/FreshModelInitializationTests.sh` — passed.
- `git diff --check` — passed.
- Separate whitespace checks for both untracked test files — passed.

The database-mutating scheduler integration test and canonical Release build were not run, per constraints.

## Recommendation

Proceed with the proposed Phase 4C USD/CAD H6 pair. Use the same executable/build and identical architecture/configuration for both arms. The pair shares identical random initialization; the deterministic auxiliary head is an intentional part of the treatment objective.

## Files changed by this verification

Only two test files were added; no production file was modified:

- [FreshModelInitializationTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/FreshModelInitializationTests.cpp)
- [FreshModelInitializationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/FreshModelInitializationTests.sh)

All other dirty files were pre-existing Phase 4C work.

`git status --short`:

```text
 M Headers/TrainingObjective.hpp
 M LSTM/main.cpp
 M Sources/ExperimentScheduler.cpp
 M Tests/TrainingObjectiveTests.cpp
 M docs/architecture/LSTMTrainingObjectiveProvenance.md
?? LSTM_Profitability_Phase4C_PairedObjectiveExperiment_Design_Output.md
?? Tests/FreshModelInitializationTests.cpp
?? Tests/FreshModelInitializationTests.sh
?? Tests/TrainingObjectiveSchedulerIntegrationTests.sh
```

`git diff --stat`—which excludes untracked test files:

```text
 Headers/TrainingObjective.hpp                      | 11 +++++
 LSTM/main.cpp                                      | 55 +++++++++++++++++++++-
 Sources/ExperimentScheduler.cpp                    | 41 +++++++++-------
 Tests/TrainingObjectiveTests.cpp                   |  7 +++
 .../LSTMTrainingObjectiveProvenance.md             | 17 ++++++-
 5 files changed, 112 insertions(+), 19 deletions(-)
```

The two task-only untracked tests contain 217 lines total. No files were staged or committed.