---
title: "LSTM Profitability Phase 2C Prerequisite Checkpoint Policy Decision Hardening Targeted Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase2C_Prerequisite_CheckpointPolicyDecisionHardening_TargetedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 2C Prerequisite Checkpoint Policy Decision Hardening Targeted Correction

Implemented the two targeted corrections. The worktree remains uncommitted, and migration 075 was not applied to production.

1. Files changed by this correction

- [Database/migrations/075_checkpoint_policy_decision_identity.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/075_checkpoint_policy_decision_identity.sql:180>)
- [Sources/ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp:12040>)
- [Tests/CheckpointPolicyMigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/CheckpointPolicyMigrationTests.sql:63>)
- [Tests/CheckpointPolicyIsolationTests.py](</Volumes/Developer SSD/ExpertAdvisor/Tests/CheckpointPolicyIsolationTests.py:89>)
- [docs/CheckpointPolicyDecisionIdentity.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/CheckpointPolicyDecisionIdentity.rst:120>)

2. Migration 075 operational state

Read-only query result:

```text
075_not_applied
```

Migration 075 was therefore corrected directly. No migration 076 was created.

3. Database lifecycle rule

The trigger now permits only:

```text
active -> active
active -> superseded
active -> action_applied
```

Subject to the existing semantic-immutability and shape constraints.

Any material update to a `legacy`, `superseded`, or `action_applied` row raises SQLSTATE `55000`. Supersession and stop-attribution metadata may be populated atomically during their valid transition, but become immutable afterward.

4. Checkpoint authority ordering

For the same parent experiment, newer authority is:

```text
higher checkpoint_epoch
or
equal checkpoint_epoch and higher checkpoint_eval_id
```

`checkpoint_decision_id DESC` remains only a deterministic final status-query tie-breaker. Same-evaluation semantic replacement is handled explicitly.

5. Supersession behavior

- Lower epoch: superseded.
- Same epoch/lower `checkpoint_eval_id`: now superseded.
- Same `checkpoint_eval_id` with changed policy/evidence identity: previous active identity superseded with `policy_or_evidence_changed`.
- Identical semantic reevaluation: existing row reused; no insertion or supersession.
- `action_applied`, `legacy`, and other-parent rows are not superseded.

6–12. Lifecycle and safety proof

The migration test now deterministically proves:

- `active → superseded` succeeds.
- `superseded → active/action_applied` fails.
- Superseded terminal metadata cannot be rewritten.
- `active → action_applied` succeeds.
- `action_applied → active/superseded` fails.
- Applied attribution metadata cannot be rewritten.
- `legacy → active` fails.
- Semantic fields remain immutable.
- Epoch 20 is superseded by epoch 40.
- Equal-epoch eval 201 is superseded by eval 202, leaving exactly one active decision.
- Changed same-evaluation semantic identity supersedes the old identity.
- Identical reevaluation leaves exactly two semantic histories—one superseded and one active—with no duplicate.
- A newer equal-epoch evaluation fences an older stop candidate.
- Later observations leave the original applied stop decision and experiment attribution intact.
- Other-parent active decisions remain untouched.

13. Legacy OR semantics

`./Tests/CheckpointPolicyHardeningTests.sh` passed, covering leader-score OR inference-accuracy OR top-N behavior, grace, and stop-mode calculations.

14. Profitability isolation

`python3 Tests/CheckpointPolicyIsolationTests.py` passed. Profitability remains absent from checkpoint decision evaluation, policy hash, evidence watermark, ranking, stop calculation, and stop application.

15. Continuation policy

Continuation source files were untouched.

Passed:

```text
ContinuationPolicyInheritanceTests
ContinuationProfitabilityPolicyTests
ContinuationProfitabilityPolicyIsolationTests
```

16. Campaign Manager/recommendation scoring

Recommendation code was untouched. `ExperimentRecommendationScoringTests` compiled with `-IHeaders -ISources` and passed.

17. Commands and results

Passed:

```text
./Tests/CheckpointPolicyHardeningTests.sh
./Tests/CheckpointPolicyMigrationTests.sh
python3 Tests/CheckpointPolicyIsolationTests.py

ContinuationPolicyInheritanceTests
./Tests/ContinuationProfitabilityPolicyTests.sh
python3 Tests/ContinuationProfitabilityPolicyIsolationTests.py

./Tests/InferenceProfitabilityTests.sh
./Tests/InferenceProfitabilityRepositoryTests.sh
ExperimentRecommendationScoringTests
```

Migration 075 was applied twice successfully in its disposable-schema test.

The unrelated migration-074 harness still fails with:

```text
ERROR: zero actionable minimum was accepted
```

This is the previously documented database-wide constraint-name collision. Migration 074 was not changed.

18. `git diff --check`

Passed with no output.

19. Debug build

Passed:

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Debug \
  -derivedDataPath DerivedData/ExpertAdvisor \
  ENABLE_USER_SCRIPT_SANDBOXING=NO build
```

Result:

```text
** BUILD SUCCEEDED **
```

The build still emits existing libpqxx `exec_params` deprecation warnings and an existing Campaign Operations unreachable-code warning.

20. Release build

Canonical Release was attempted without bypassing provenance:

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor build
```

It stopped before compilation as expected:

```text
GenerateBuildProvenance.py: Release provenance requires a clean source tree
** BUILD FAILED **
```

Release remains pending a committed, clean worktree.

21. Remaining issues

- Existing migration-074 disposable-schema constraint-name collision.
- Existing compiler warnings noted above.
- Canonical Release cannot complete until the worktree is committed and clean.
- Migration 075 remains intentionally unapplied.

`git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/ExperimentScheduler.cpp
?? Database/migrations/075_checkpoint_policy_decision_identity.sql
?? LSTM_Phase2C_Prerequisite_CheckpointPolicyDecisionHardening_Verification.tar.gz
?? LSTM_Phase2C_Prerequisite_CheckpointPolicyDecisionHardening_Verification/
?? LSTM_Profitability_Phase2C_CheckpointPolicy_DesignInspection_Output.md
?? LSTM_Profitability_Phase2C_Prerequisite_CheckpointPolicyDecisionHardening_Output.md
?? LSTM_Profitability_Phase2C_Prerequisite_Independent_Verification_Findings.md
?? Sources/CheckpointPolicy.cpp
?? Sources/CheckpointPolicy.hpp
?? Tests/CheckpointPolicyHardeningTests.cpp
?? Tests/CheckpointPolicyHardeningTests.sh
?? Tests/CheckpointPolicyIsolationTests.py
?? Tests/CheckpointPolicyMigrationTests.sh
?? Tests/CheckpointPolicyMigrationTests.sql
?? docs/CheckpointPolicyDecisionIdentity.rst
?? package_phase2c_prerequisite_verification.sh
```

`git diff --stat`—which excludes untracked files:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj |    8 +
 Sources/ExperimentScheduler.cpp         | 1053 +++++++++++++++++++++++++------
 2 files changed, 861 insertions(+), 200 deletions(-)
```