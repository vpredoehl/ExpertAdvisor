---
title: "LSTM Profitability Phase 2C Prerequisite Checkpoint Policy Decision Hardening"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase2C_Prerequisite_CheckpointPolicyDecisionHardening_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 2C Prerequisite Checkpoint Policy Decision Hardening

Implemented the checkpoint-policy lifecycle hardening prerequisite without activating profitability.

1. Files changed

- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp>)
- [CheckpointPolicy.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CheckpointPolicy.hpp>)
- [CheckpointPolicy.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CheckpointPolicy.cpp>)
- [project.pbxproj](</Volumes/Developer SSD/ExpertAdvisor/ExpertAdvisor.xcodeproj/project.pbxproj>)
- [075_checkpoint_policy_decision_identity.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/075_checkpoint_policy_decision_identity.sql>)
- [CheckpointPolicyHardeningTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CheckpointPolicyHardeningTests.cpp>)
- [CheckpointPolicyHardeningTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/CheckpointPolicyHardeningTests.sh>)
- [CheckpointPolicyIsolationTests.py](</Volumes/Developer SSD/ExpertAdvisor/Tests/CheckpointPolicyIsolationTests.py>)
- [CheckpointPolicyMigrationTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/CheckpointPolicyMigrationTests.sh>)
- [CheckpointPolicyMigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/CheckpointPolicyMigrationTests.sql>)
- [CheckpointPolicyDecisionIdentity.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/CheckpointPolicyDecisionIdentity.rst>)

The pre-existing untracked design-inspection document was not modified.

2. Migration

Migration 075: `075_checkpoint_policy_decision_identity.sql`.

It is additive, preserves historical rows, removes only the old one-row-per-evaluation unique index, and adds an immutability trigger.

3. Branch/instruction resolution

Actual branch: `lstm-feature-development`, six commits ahead of `github/lstm-feature-development`.

No local `phase6` branch exists. The `AGENTS.md` branch reference is stale, but its build, safety, architecture, and review instructions remain authoritative. No branch switch was performed.

4. Semantic policy identity

Canonical fields:

`enabled|min_leader_score|min_infer_accuracy|top_n|rank_scope|stop_mode|grace_evals|checkpoint_interval|target_epochs`

The hash is deterministic 64-bit FNV-1a, matching the continuation hashing convention without coupling checkpoint semantics to continuation policy.

5. Policy revisions

- New/legacy experiments default safely to revision 1.
- Effective enable/disable and decision-bearing configuration changes increment revision.
- Identical mutations and read-only inspection do not.
- Legacy NULL hashes are materialized without increment.
- Out-of-band semantic drift is detected during mutating evaluation, increments revision, and rematerializes the hash.

6. Evidence watermark

Includes exact checkpoint evaluation, parent, model, epoch, persisted current progress, analysis ID, exact checkpoint inference-result ID, symbol, horizon, inference range, leader score, inference accuracy, completed-evaluation count/population, lifecycle statuses, and decision-bearing rank/population identity.

Profitability is excluded.

7. Immutable decision identity

`checkpoint_eval_id + policy_revision + policy_hash + evidence_watermark`

Identical reevaluation reuses the row. Policy/evidence changes append a new row. Database enforcement prevents rewriting decision meaning.

8. Historical decisions

Existing rows become explicit `legacy` rows with NULL policy/evidence identity. They remain queryable and visible but cannot be reused as hardened authoritative stop decisions.

9. Stop attribution

`experiment.checkpoint_policy_stop_decision_id` identifies the exact decision that set `stop_after_checkpoint_epoch`.

The decision records:

- `stop_request_applied`
- applied timestamp
- requested stop epoch
- exact active training worker-attempt ID

These changes commit atomically in the enclosing transaction.

10. Stop fencing

Stop application verifies parent lifecycle, exact evidence, current revision/hash, decision authority, no newer action-bearing checkpoint, no existing terminal/manual stop, future/reachable stop epoch, exact active training attempt, and current scheduler lease/fencing token. Failed fences supersede the decision without setting the stop field.

11. Reevaluation

- Same policy/evidence: idempotent reuse.
- Changed policy/evidence: new durable identity.
- Applied stops remain terminal.
- Later continue observations cannot cancel or overwrite an applied stop.
- Older checkpoints cannot act after a newer authoritative decision.

12. Supersession

Added `active`, `superseded`, `action_applied`, and `legacy` lifecycle states with deterministic supersession reasons. A generic deferred state was intentionally omitted because no safe pending action exists without fresh reevaluation.

13. Rank population

Top-N identity hashes the unchanged sorted population using evaluation, analysis, available inference-result identity, metrics, epoch, and lifecycle state. Rank ordering and scope are unchanged. When top-N is absent, ranking is explicitly non-decision-bearing.

14. Read-only status

Added:

```text
--checkpoint-policy-status=CHECKPOINT_EVAL_ID
```

It uses a read-only transaction and reports configuration, revision/hash, exact analysis/inference identity, evidence watermark, durable decisions, authoritative decision, stop attribution, and supersession. It performs no evaluation, insertion, scheduler mutation, or queuing.

15. Tests added

New deterministic policy, isolation, and migration suites cover OR/grace/stop-mode equivalence, hash/watermark identity, append-only persistence, legacy preservation, immutability, stop attribution, shared evaluator use, and profitability isolation.

16. Test results

Passed:

- Checkpoint-policy hardening tests
- Checkpoint-policy migration tests, including repeat migration
- Checkpoint-policy isolation tests
- Continuation policy inheritance tests
- Continuation profitability policy unit tests
- Continuation profitability isolation tests
- Inference profitability repository tests
- Recommendation scoring tests
- `git diff --check`

The existing migration-074 harness failed because its constraint-name checks are database-wide and collide with already-installed production constraint names. Migration 074 and continuation semantics were not modified to conceal that pre-existing isolation defect.

17. Debug build

Passed:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" \
  -configuration Debug -derivedDataPath DerivedData/ExpertAdvisor \
  ENABLE_USER_SCRIPT_SANDBOXING=NO build
```

The override was required because the existing provenance script’s atomic temporary file is rejected by Xcode’s script sandbox.

18. Release build

The canonical Release invocation was run but stopped before compilation:

```text
GenerateBuildProvenance.py: Release provenance requires a clean source tree
** BUILD FAILED **
```

I did not bypass repository provenance or commit automatically. No `xcodebuild clean` was run.

19. Legacy behavior equivalence

Pure deterministic fixtures prove unchanged:

- OR-based continue behavior
- rule ordering and reason strings
- grace behavior
- next-checkpoint requested epoch
- current-checkpoint-if-possible requested epoch
- mark-pruned-when-not-running behavior
- ranking order

20. Profitability isolation

Static and runtime tests prove profitability is absent from checkpoint decision evaluation, semantic identity, evidence identity, rank, requested-stop calculation, and stop application.

21. Continuation isolation

No continuation implementation or migration was changed. Continuation inheritance, profitability policy, and isolation tests pass.

22. Campaign Manager isolation

No Campaign Manager, recommendation, or scoring source was changed. Recommendation scoring tests pass.

23. Deferred to actual Phase 2C

Deferred intentionally:

- All checkpoint profitability thresholds and CLI options
- Profitability ranking, trends, or source selection
- Profitability-aware stop/continue behavior
- Expanded stop-mode support beyond recommending `next_checkpoint`
- Automatic stop cancellation

No workers, scheduler state, experiments, or production policy configuration were changed.

Final `git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/ExperimentScheduler.cpp
?? Database/migrations/075_checkpoint_policy_decision_identity.sql
?? LSTM_Profitability_Phase2C_CheckpointPolicy_DesignInspection_Output.md
?? Sources/CheckpointPolicy.cpp
?? Sources/CheckpointPolicy.hpp
?? Tests/CheckpointPolicyHardeningTests.cpp
?? Tests/CheckpointPolicyHardeningTests.sh
?? Tests/CheckpointPolicyIsolationTests.py
?? Tests/CheckpointPolicyMigrationTests.sh
?? Tests/CheckpointPolicyMigrationTests.sql
?? docs/CheckpointPolicyDecisionIdentity.rst
```

`git diff --stat`:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj |    8 +
 Sources/ExperimentScheduler.cpp         | 1051 +++++++++++++++++++++++++------
 2 files changed, 859 insertions(+), 200 deletions(-)
```

Untracked new files are not included in `git diff --stat` until staged.