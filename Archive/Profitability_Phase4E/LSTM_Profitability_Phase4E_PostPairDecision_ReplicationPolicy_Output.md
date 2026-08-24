---
title: "LSTM Profitability Phase 4E Post-Pair Decision and Replication Policy"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase4E_PostPairDecision_ReplicationPolicy_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 4E Post-Pair Decision and Replication Policy

Implemented Phase 4E as a pure, deterministic policy with no database, scheduler, CLI, or experiment-launch integration.

The complete design and audit is in [LSTM_Profitability_Phase4E_PostPairDecision_ReplicationPolicy_Output.md](</Volumes/Developer SSD/ExpertAdvisor/LSTM_Profitability_Phase4E_PostPairDecision_ReplicationPolicy_Output.md>).

Key behavior:

- One `PROMISING` → `REPLICATE_BROADER`.
- One `MIXED` → reason-targeted `REPLICATE`.
- One `NOT_PROMISING` → one diverse confirmation.
- Two diverse `NOT_PROMISING` → `STOP_OBJECTIVE`.
- `INVALID_COMPARISON` → `REPAIR_COMPARABILITY`.
- `INCOMPLETE` → `WAIT_FOR_VALID_RESULT`.
- Coefficient exploration requires four independent valid units, at least three `PROMISING`, ≥75% promising, ≤25% mixed, zero negatives, three symbols, and two horizons.
- Six valid units without eligibility stops screening.
- Exact deterministic reruns count once because current initialization uses seed 42.
- Materiality policies now have canonical text and deterministic identity; mismatches fail closed.

Files changed:

- [PostPairReplicationPolicy.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/PostPairReplicationPolicy.hpp>)
- [PostPairReplicationPolicy.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/PostPairReplicationPolicy.cpp>)
- [PairedTrainingObjectiveEvaluation.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/PairedTrainingObjectiveEvaluation.hpp>)
- [PairedTrainingObjectiveEvaluation.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/PairedTrainingObjectiveEvaluation.cpp>)
- [PostPairReplicationPolicyTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/PostPairReplicationPolicyTests.cpp>)
- [PostPairReplicationPolicyTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/PostPairReplicationPolicyTests.sh>)
- [project.pbxproj](</Volumes/Developer SSD/ExpertAdvisor/ExpertAdvisor.xcodeproj/project.pbxproj>)
- Required Phase 4E report.

Validation:

- Phase 4E policy tests: PASS
- Phase 4D evaluator tests: PASS
- TrainingObjective tests: PASS
- Shell syntax: PASS
- Xcode project lint: PASS
- `git diff --check`, including explicit untracked-file checks: PASS
- Release build used only `DerivedData/Development`; stopped with exit 65 at the expected clean-worktree provenance gate. The gate was not bypassed.

Remaining risk: a complete Release build requires these changes to be committed into a clean worktree.

`git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/PairedTrainingObjectiveEvaluation.cpp
 M Sources/PairedTrainingObjectiveEvaluation.hpp
?? LSTM_Profitability_Phase4E_PostPairDecision_ReplicationPolicy_Output.md
?? Sources/PostPairReplicationPolicy.cpp
?? Sources/PostPairReplicationPolicy.hpp
?? Tests/PostPairReplicationPolicyTests.cpp
?? Tests/PostPairReplicationPolicyTests.sh
```

`git diff --stat`:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj       |  8 +++
 Sources/PairedTrainingObjectiveEvaluation.cpp | 73 +++++++++++++++++++++++++++
 Sources/PairedTrainingObjectiveEvaluation.hpp |  4 ++
 3 files changed, 85 insertions(+)
```

Untracked files are excluded from that statistic.

```text
LIVE_STATE_NOT_CHECKED_PRODUCTION_DATABASE_ACCESS_PROHIBITED
LIVE_DATABASE_ACCESSED=false
LSTM_DATABASE_ACCESSED=false
FOREX_DATABASE_ACCESSED=false
PRODUCTION_SCHEDULER_TOUCHED=false
PRODUCTION_DERIVEDDATA_TOUCHED=false
DISPOSABLE_DATABASE_USED=none
```

Experiments 599/600 were not inspected. After their eventual Phase 4D result, apply the disposition-specific policy above; under no outcome should that single pair authorize coefficient tuning.