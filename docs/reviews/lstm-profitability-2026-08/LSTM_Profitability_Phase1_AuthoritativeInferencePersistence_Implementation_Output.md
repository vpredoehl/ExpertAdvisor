---
title: "LSTM Profitability Phase 1 Authoritative Inference Persistence Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase1_AuthoritativeInferencePersistence_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 1 Authoritative Inference Persistence Implementation

Implemented Profitability Phase 1 as an additive, immutable persistence foundation. No profitability consumer policy was activated.

### Implementation

Key files:

- Shared calculation: [InferenceProfitability.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/InferenceProfitability.hpp:15>) and [InferenceProfitability.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/InferenceProfitability.cpp:75>)
- Repository/read interface: [InferenceProfitabilityRepository.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.hpp:14>) and [InferenceProfitabilityRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.cpp:150>)
- Inference integration: [main.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp:6462>)
- Migration: [073_inference_profitability_observation.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/073_inference_profitability_observation.sql:1>)
- Tests: [InferenceProfitabilityTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/InferenceProfitabilityTests.cpp:1>) and [InferenceProfitabilityRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/InferenceProfitabilityRepositoryTests.cpp:1>)
- Documentation: [InferenceProfitabilityPersistence.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/InferenceProfitabilityPersistence.rst:1>)
- Xcode project and [Database README](</Volumes/Developer SSD/ExpertAdvisor/Database/README.md:39>)

### Persisted schema

`inference_profitability_observation` contains:

- observation ID
- nullable `experiment_id` only for genuinely unlinked legacy FINAL models
- model and inference-evaluation IDs
- explicit final/checkpoint scope
- checkpoint evaluation ID
- inference range
- prediction and actionable counts
- winning and losing actionable counts
- positive, negative, and aggregate terminal-horizon log-return sums
- nullable average per actionable prediction
- canonical metric definition and hash
- ordered source-content hash
- canonical observation identity and hash
- creation timestamp

Constraints enforce scope shape, count relationships, finite returns, average presence, foreign keys, and checkpoint provenance. Triggers validate completed inference provenance and reject all updates/deletes.

No historical backfill occurs.

### Exact metric semantics

- Prediction: each evaluable classification window.
- Predicted Down requires Down probability to be strictly greater than both alternatives.
- Predicted Up requires Up probability to be strictly greater than both alternatives.
- Ties or neither strict maximum produce Neutral.
- Actionable: predicted Up/Down with finite, positive decision and terminal closes.
- Up return: `ln(terminal_close / decision_close)`.
- Down return: its negative.
- Neutral and invalid-price directional predictions remain predictions but are not actionable.
- Win: directional return `> 0`; loss: `< 0`; exact zero is neither.
- Horizon: configured terminal prediction horizon.
- No clipping, spread, commission, slippage, sizing, leverage, allocation, or overlap-capital model.
- Zero actionable predictions persist valid zero sums and a `NULL` average.

The same accumulator is called by the common final/checkpoint evaluator.

### Identity and idempotency

Source identity hashes the ordered ordinal, predicted class, and exact float-bit representations of decision and terminal prices.

Observation canonical identity includes provenance, scope, checkpoint identity, dates, full metric definition, source identity, and canonical statistics. It is uniquely constrained.

- Identical rerun: returns the existing observation ID.
- Changed source content: inserts a distinct immutable observation.
- Changed metric definition: inserts a distinct immutable observation.
- Existing observations are never rewritten.

Read helpers support retrieval by observation ID or exact inference result, scope, checkpoint identity, metric identity, and source identity.

### Behavior boundary

Final infer-all, scheduler-final inference, and scheduler-checkpoint inference now persist observations in the same transaction as successful inference-evaluation persistence. Failed evaluation does not create an observation.

No changes were made to:

- recommendation or campaign eligibility/scoring/ranking
- `minimumProfitability`
- scheduler continuation decisions
- checkpoint stop/continue policy
- training features or objective

### Validation

Passed:

- `./Tests/InferenceProfitabilityTests.sh`
- `./Tests/InferenceProfitabilityRepositoryTests.sh`
- `ContinuationPolicyInheritanceTests`
- `ExperimentRecommendationScoringTests`
- `git diff --check`
- Debug compilation of the complete LSTM Release target:
  `xcodebuild ... -configuration Debug ... ENABLE_USER_SCRIPT_SANDBOXING=NO build`

The exact required Release command was run, but the existing provenance build phase intentionally rejected the dirty review worktree:

```text
GenerateBuildProvenance.py: Release provenance requires a clean source tree
** BUILD FAILED **
```

A final Release build therefore remains necessary after the reviewed changes are committed or otherwise placed in a clean worktree. Added source files emitted no new warnings in the successful Debug build.

### Remaining Phase 2 work

Future phases must deliberately select exact scope/metric/source identities before consuming observations. No continuation, checkpoint, recommendation, or campaign policy integration is present.

### `git status --short`

```text
 M Database/README.md
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M LSTM/main.cpp
A  lstm_watch.sql
?? Database/migrations/073_inference_profitability_observation.sql
?? LSTM_CampaignManager_Phase1_CandidateSelection_Inspection_Output.md
?? LSTM_CampaignManager_Phase2_ScoringRanking_Inspection_Output.md
?? LSTM_CampaignManager_Phase3_ProfitabilityIntegration_Design_Inspection_Output.md
?? Sources/InferenceProfitability.cpp
?? Sources/InferenceProfitability.hpp
?? Sources/InferenceProfitabilityRepository.cpp
?? Sources/InferenceProfitabilityRepository.hpp
?? Tests/InferenceProfitabilityRepositoryTests.cpp
?? Tests/InferenceProfitabilityRepositoryTests.sh
?? Tests/InferenceProfitabilityTests.cpp
?? Tests/InferenceProfitabilityTests.sh
?? docs/InferenceProfitabilityPersistence.rst
```

The pre-existing inspection outputs and `lstm_watch.sql` were not modified as part of this implementation.

### `git diff --stat`

```text
 Database/README.md                      |   5 +
 ExpertAdvisor.xcodeproj/project.pbxproj |  16 ++
 LSTM/main.cpp                           | 400 ++++++++++++++++++--------------
 3 files changed, 243 insertions(+), 178 deletions(-)
```

New untracked implementation files are not included in Git’s unstaged diff-stat output.