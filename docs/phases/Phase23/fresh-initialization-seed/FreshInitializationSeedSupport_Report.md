# Fresh initialization seed support

## Architecture traced

`Database/migrations` defines `experiment`; `SchedulerCore/ExperimentScheduler.cpp` parses queue requests and duplicate identity; `ProductionSchedulerDaemon.cpp` persists/reloads rows and builds workers; `LSTM/main.cpp` parses training arguments; `LSTM/LSTM.cpp` initializes fresh parameters. Pair evidence is loaded by `FeatureAblationPairEvaluationRepository.cpp`, then consumed by the corrected replication continuation service.

## Change

Migration 094 adds non-null `experiment.fresh_initialization_seed`, defaulting historical and omitted fresh requests to 42. Queue accepts `--fresh-initialization-seed N`, validates positive uint32 values, prints it, and includes it in duplicate identity. Fresh worker commands carry the persisted row value; resumed workers do not receive a seed. The training CLI rejects a seed with `--resume-model-id`.

LSTM construction receives the seed and creates a constructor-local `mt19937`; only fresh LSTM parameter initialization reads that engine. This preserves default-42 behavior while making results independent of prior model construction. Pair evidence loads the persisted seed for fresh arms and leaves resumed arms lineage-governed.

## Validation

- `git diff --check` — passed.
- `bash -n Tests/FreshModelInitializationTests.sh` — passed.
- `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Debug" -configuration Debug -derivedDataPath DerivedData/ExpertAdvisor build` — passed (pre-existing warnings emitted).
- `LSTM_TEST_PRODUCTS_DIR=... ./Tests/FreshModelInitializationTests.sh` — not run to completion: its existing standalone compile command omits the installed libpqxx include directory and fails at `pqxx/pqxx` not found.

No production experiment was queued or modified. No commit was created. The changes remain in the working tree.

## Follow-up risk

No disposable-DB integration test was run. The standalone test harness needs its libpqxx include/link configuration repaired before its strengthened seed assertions can execute.
