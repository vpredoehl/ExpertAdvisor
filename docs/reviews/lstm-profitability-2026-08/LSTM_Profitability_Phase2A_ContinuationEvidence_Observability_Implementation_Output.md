---
title: "LSTM Profitability Phase 2A Continuation Evidence Observability Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase2A_ContinuationEvidence_Observability_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 2A Continuation Evidence Observability Implementation

Implemented Profitability Phase 2A as diagnostic-only evidence plumbing.

### Implementation

Files changed:

- [ContinuationPolicy.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ContinuationPolicy.hpp:91) and [ContinuationPolicy.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ContinuationPolicy.cpp:538)
- [ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:12113)
- [InferenceProfitabilityRepository.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.hpp:75) and [InferenceProfitabilityRepository.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/InferenceProfitabilityRepository.cpp:310)
- [InferenceProfitabilityRepositoryTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/InferenceProfitabilityRepositoryTests.cpp:117)
- [ContinuationPolicyInheritanceTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/ContinuationPolicyInheritanceTests.cpp:523)
- [ContinuationProfitabilityPolicyIsolationTests.py](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/ContinuationProfitabilityPolicyIsolationTests.py:1)
- [InferenceProfitabilityPersistence.rst](/Volumes/Developer%20SSD/ExpertAdvisor/docs/InferenceProfitabilityPersistence.rst:65)

No migration was required. Migration 073 already provides the authoritative schema; no profitability data was duplicated into continuation tables.

The optional continuation evidence contains observation ID/hash, inference-result ID, final/checkpoint scope, checkpoint ID, metric/source hashes, prediction/actionable/win/loss counts, positive/negative/aggregate directional log-return sums, and nullable per-actionable average.

Selection rules:

- Final: exact source experiment, model, inference result, `final` scope, null checkpoint ID, and exact current canonical metric definition/hash.
- Checkpoint: the same exact provenance plus `checkpoint` scope and the exact selected checkpoint ID.
- Multiple matching source observations are ambiguous; no arbitrary latest row is selected.
- Final and checkpoint evidence never substitute for each other.

Unavailable evidence remains optional and reports stable reasons including `no_profitability_observation`, `ambiguous_profitability_observation`, `metric_definition_mismatch`, `provenance_mismatch`, and `profitability_schema_unavailable`. Zero actionable predictions remain available with zero aggregate and a null average.

Diagnostics were added to continuation evaluation logs, automatic-continuation logs, and the now explicitly read-only `--continuation-status` path.

### Behavior-neutral proof

Tests verify that changing profitability evidence does not change:

- best/latest/final source selection;
- epoch deduplication, evidence count, or ordering;
- evidence watermark;
- trend metric, value, or result.

Static policy-isolation guards verify profitability is absent from continuation eligibility, ranking, policy evaluation, automatic queue ordering, checkpoint policy, recommendation scoring, and recommendation ranking. No Campaign Manager, checkpoint-policy, training, or recommendation implementation files were changed.

### Validation

Passed:

- `./Tests/InferenceProfitabilityTests.sh`
- `./Tests/InferenceProfitabilityRepositoryTests.sh`
- `ContinuationPolicyInheritanceTests`
- `ExperimentRecommendationScoringTests`
- `python3 Tests/ContinuationProfitabilityPolicyIsolationTests.py`
- `git diff --check`
- Complete Debug compilation/link of the `LSTM Release` scheme

The exact canonical Release command was run but failed before target compilation:

```text
GenerateBuildProvenance.py: Release provenance requires a clean source tree
** BUILD FAILED **
```

Therefore Release success is not claimed. The Debug build emitted the repository’s existing warning backlog; no new warning was identified in the added profitability code.

Scheduler integration suites were not run because an active production scheduler and training workers were detected. No experiment or scheduler state was modified.

The inspected local database has not yet applied migration 073. Until that existing migration is applied operationally, continuation diagnostics report `profitability_schema_unavailable` while preserving existing continuation behavior.

### Reserved for later phases

Phase 2B policy activation, Phase 2C checkpoint-policy use, Campaign Manager Phase 3A exposure, profitability thresholds/scoring/ranking, and training use remain unimplemented.

`git status --short`:

```text
 M Sources/ContinuationPolicy.cpp
 M Sources/ContinuationPolicy.hpp
 M Sources/ExperimentScheduler.cpp
 M Sources/InferenceProfitabilityRepository.cpp
 M Sources/InferenceProfitabilityRepository.hpp
 M Tests/ContinuationPolicyInheritanceTests.cpp
 M Tests/InferenceProfitabilityRepositoryTests.cpp
 M docs/InferenceProfitabilityPersistence.rst
?? Tests/ContinuationProfitabilityPolicyIsolationTests.py
```

`git diff --stat`:

```text
 Sources/ContinuationPolicy.cpp                  | 287 +++++++++++++++++
 Sources/ContinuationPolicy.hpp                  |  60 ++++
 Sources/ExperimentScheduler.cpp                 | 399 +++++++++++-------------
 Sources/InferenceProfitabilityRepository.cpp    | 104 ++++++
 Sources/InferenceProfitabilityRepository.hpp    |  38 +++
 Tests/ContinuationPolicyInheritanceTests.cpp    | 167 ++++++++++
 Tests/InferenceProfitabilityRepositoryTests.cpp | 127 +++++++-
 docs/InferenceProfitabilityPersistence.rst      |  64 +++-
 8 files changed, 1026 insertions(+), 220 deletions(-)
```

The untracked policy-isolation test is not included in Git’s unstaged diff stat.