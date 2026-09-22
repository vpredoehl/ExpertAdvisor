---
title: "LSTM Explicit Fresh Initialization Seed Support"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_ExplicitFreshInitializationSeedSupport_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# LSTM Explicit Fresh Initialization Seed Support

Implemented and left uncommitted.

Key changes:
- Added migration `094_fresh_initialization_seed.sql`: non-null experiment seed, default `42`.
- Added `--fresh-initialization-seed N` queue/training handling with positive uint32 validation.
- Included seed in fresh-experiment duplicate identity and queue output.
- Propagated persisted seed to fresh training workers; resume rejects a seed override.
- Replaced mutable thread-local seed-42 stream with constructor-local deterministic RNG boundary.
- Pair-evaluation evidence now loads persisted fresh seed.
- Strengthened fresh-model test inputs for explicit seed variation.
- Added report: [FreshInitializationSeedSupport_Report.md](/Volumes/Developer%20SSD/ExpertAdvisor/FreshInitializationSeedSupport_Report.md)

Validation:
- `git diff --check` passed.
- `bash -n Tests/FreshModelInitializationTests.sh` passed.
- Debug build passed:
  `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Debug" -configuration Debug -derivedDataPath DerivedData/ExpertAdvisor build`
- Fresh initialization script did not complete because its existing standalone compile invocation cannot locate `pqxx/pqxx`.

No production experiments were queued or modified. No commit was created.