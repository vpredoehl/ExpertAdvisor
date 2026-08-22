---
title: "LSTM Profitability Phase 2B Scheduler Continuation Policy Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase2B_SchedulerContinuationPolicy_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 2B Scheduler Continuation Policy Implementation

Implemented Profitability Phase 2B as an explicit, default-disabled scheduler continuation gate. No experiments, scheduler state, production rows, or workers were modified. No commit was created.

## Implementation

Added three optional continuation-policy fields:

- `min_profitability_actionable_count`
- `min_profitability_aggregate_terminal_horizon_log_return_sum`
- `min_profitability_average_terminal_horizon_log_return_per_actionable_prediction`

All configured conditions use AND semantics. They are additive to the existing exclusion, evidence, leader-score, inference-accuracy, top-N, trend, source, and lineage requirements.

The metrics remain directional log-return inference statistics—not portfolio P&L, realized profit, account return, or transaction-cost-adjusted return. Aggregate return is sample-size sensitive; average return reduces that dependency but is not normalized across symbols or horizons. The independent actionable-count gate provides explicit sample-size control without inventing a composite score.

Policy flow is now:

```text
source experiment
→ configured source mode
→ exact final / best-checkpoint / latest-checkpoint source
→ historical evidence and existing gates
→ optional contemporaneous profitability gate
→ existing eligibility and queue handling
```

Profitability does not change source selection, ranking, candidate ordering, leader/inference trends, or evidence count.

## Compatibility and evidence semantics

When all three fields are unset:

- The gate reports `profitability_policy_disabled` and passes.
- Profitability remains excluded from the policy canonical text and semantic hash.
- Profitability remains excluded from the legacy evidence watermark.
- Existing source selection, evidence count, trends, rankings, decisions, and queue behavior are unchanged.
- Merely having profitability observations in the database has no effect.

When any field is configured:

- Exact authoritative evidence for the already-selected source is required.
- Missing, ambiguous, provenance-mismatched, or lookup-error evidence fails closed as `profitability_evidence_unavailable`.
- Missing evidence is never interpreted as zero.
- Final and checkpoint observations cannot substitute for each other.
- A zero-actionable observation remains available evidence: aggregate zero is defined, average is undefined, and a positive actionable-count minimum fails.

Configured profitability fields participate in policy identity and existing child-policy inheritance. Selected profitability evidence enters the decision watermark only when the policy is configured.

Diagnostics include policy state, configured thresholds, source scope, observation identity, counts, aggregate/average values, per-condition results, and the combined result. Rejection reasons include:

- `profitability_evidence_unavailable`
- `profitability_actionable_count_below_minimum`
- `profitability_aggregate_terminal_horizon_log_return_sum_below_minimum`
- `profitability_average_terminal_horizon_log_return_per_actionable_prediction_below_minimum`
- `profitability_average_terminal_horizon_log_return_per_actionable_prediction_undefined`

The decision value is `rejected_profitability`.

## Persistence and CLI

Added [migration 074](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/074_continuation_profitability_policy.sql), using nullable, default-disabled columns:

- `continuation_policy_min_profit_actionable_count`
- `continuation_policy_min_profit_aggregate_log_return_sum`
- `continuation_policy_min_profit_average_log_return`

The migration preserves old rows, validates positive counts and finite numeric thresholds, and extends continuation decision constraints. It was validated twice in a disposable schema for idempotency but was not applied operationally.

CLI parsing supports setting or clearing each field with `null`. Help text documents source selection, missing evidence, zero-actionable behavior, metric limitations, and the absence of profitability ranking/trend.

Read-only `--continuation-status` and automatic/manual continuation use the same authoritative evaluator.

## Files changed

- [ContinuationPolicy.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ContinuationPolicy.hpp)
- [ContinuationPolicy.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ContinuationPolicy.cpp)
- [ContinuationPolicyInheritance.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ContinuationPolicyInheritance.cpp)
- [ContinuationPolicyPersistence.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ContinuationPolicyPersistence.cpp)
- [ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp)
- [074_continuation_profitability_policy.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/074_continuation_profitability_policy.sql)
- [ContinuationProfitabilityPolicyTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/ContinuationProfitabilityPolicyTests.cpp)
- [ContinuationProfitabilityPolicyTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/ContinuationProfitabilityPolicyTests.sh)
- [ContinuationProfitabilityPolicyMigrationTests.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/ContinuationProfitabilityPolicyMigrationTests.sql)
- [ContinuationProfitabilityPolicyMigrationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/ContinuationProfitabilityPolicyMigrationTests.sh)
- [ContinuationPolicyInheritanceTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/ContinuationPolicyInheritanceTests.cpp)
- [ContinuationProfitabilityPolicyIsolationTests.py](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/ContinuationProfitabilityPolicyIsolationTests.py)
- [InferenceProfitabilityPersistence.rst](/Volumes/Developer%20SSD/ExpertAdvisor/docs/InferenceProfitabilityPersistence.rst)
- [Database README](/Volumes/Developer%20SSD/ExpertAdvisor/Database/README.md)

## Validation

Passed:

- `./Tests/ContinuationProfitabilityPolicyTests.sh`
- `./Tests/ContinuationProfitabilityPolicyMigrationTests.sh`
- `ContinuationPolicyInheritanceTests`
- `python3 Tests/ContinuationProfitabilityPolicyIsolationTests.py`
- `./Tests/InferenceProfitabilityTests.sh`
- `./Tests/InferenceProfitabilityRepositoryTests.sh`
- `ExperimentRecommendationScoringTests`
- Continuation persistence test compilation
- `git diff --check`

The new tests cover disabled equivalence, observations with disabled policy, every implemented threshold pass/fail, missing and ambiguous evidence, zero-actionable behavior, final/best/latest source semantics, no source reselection, unchanged watermark/trend, policy identity, inheritance, parsing, clearing, and invalid/non-finite values.

Debug build passed:

```bash
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Debug \
  -derivedDataPath DerivedData/ExpertAdvisor-Phase2B-Debug \
  ENABLE_USER_SCRIPT_SANDBOXING=NO \
  build
```

The override was needed because Xcode’s user-script sandbox denied creation of the generated provenance temporary file. Existing libpqxx deprecation warnings remain; no new compiler errors were introduced.

The canonical Release build was not run because the repository is intentionally dirty and its provenance checks require an appropriate clean state.

Scheduler continuation integration tests and the operational persistence test were not run: an automatic scheduler and multiple training workers are active, and operational migration 074 has not been applied. Migration 073 remains the latest operational migration.

Checkpoint-policy isolation is proven by the static boundary test and by no changes to checkpoint decision functions. Recommendation/Campaign Manager isolation is proven by the static boundary test, passing recommendation-scoring tests, and no changes to recommendation-scoring files.

Phase 2C remains responsible for any profitability-aware checkpoint stop/continue policy.

## Worktree

`git status --short`:

```text
 M Database/README.md
 M Sources/ContinuationPolicy.cpp
 M Sources/ContinuationPolicy.hpp
 M Sources/ContinuationPolicyInheritance.cpp
 M Sources/ContinuationPolicyPersistence.cpp
 M Sources/ExperimentScheduler.cpp
 M Tests/ContinuationPolicyInheritanceTests.cpp
 M Tests/ContinuationProfitabilityPolicyIsolationTests.py
 M docs/InferenceProfitabilityPersistence.rst
?? Database/migrations/074_continuation_profitability_policy.sql
?? Tests/ContinuationProfitabilityPolicyMigrationTests.sh
?? Tests/ContinuationProfitabilityPolicyMigrationTests.sql
?? Tests/ContinuationProfitabilityPolicyTests.cpp
?? Tests/ContinuationProfitabilityPolicyTests.sh
```

`git diff --stat` for tracked files:

```text
 Database/README.md                                 |   6 +
 Sources/ContinuationPolicy.cpp                     | 277 +++++++++++++++++++++
 Sources/ContinuationPolicy.hpp                     |  38 ++-
 Sources/ContinuationPolicyInheritance.cpp          |   5 +
 Sources/ContinuationPolicyPersistence.cpp          |  17 ++
 Sources/ExperimentScheduler.cpp                    | 145 ++++++++++-
 Tests/ContinuationPolicyInheritanceTests.cpp       |   5 +
 ...ontinuationProfitabilityPolicyIsolationTests.py |  28 ++-
 docs/InferenceProfitabilityPersistence.rst         |  96 +++++--
 9 files changed, 592 insertions(+), 25 deletions(-)
```

The five new untracked migration/test files are not included in Git’s unstaged `diff --stat`.