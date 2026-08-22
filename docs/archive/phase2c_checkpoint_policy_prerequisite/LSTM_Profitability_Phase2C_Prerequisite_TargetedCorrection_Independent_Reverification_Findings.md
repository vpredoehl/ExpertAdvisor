# Phase 2C Prerequisite Targeted-Correction — Independent Reverification

**Target:** Checkpoint-policy lifecycle / decision-identity hardening prerequisite
**Review scope:** Post-correction reverification of the two prior FAIL findings plus regression boundaries
**Result:** **PASS — ready to stage/commit**

## Executive conclusion

The targeted correction successfully resolves both defects identified in the previous independent verification.

The corrected implementation now provides:

- terminal superseded/action-applied/legacy lifecycle semantics at the database boundary;
- consistent checkpoint authority ordering by epoch, then checkpoint_eval_id;
- persistence supersession consistent with stop fencing;
- serialized evaluation for the same parent experiment;
- preserved append-only semantic decision history;
- preserved stop attribution and worker fencing;
- preserved legacy checkpoint-policy OR semantics;
- preserved profitability isolation.

No new correctness defect was found in the corrected scope.

The prerequisite is therefore ready for the next operational sequence:

`stage/archive -> commit -> clean Release build -> operational migration 075 -> read-only live verification`

Actual profitability-aware Phase 2C should remain deferred until those operational steps pass.

---

## Archive integrity

The supplied correction-review archive was extracted successfully.

`SHA256SUMS.txt` verifies successfully for the packaged review files.

The archive records:

- branch: `lstm-feature-development`;
- implementation still uncommitted;
- migration 075 still present as new/uncommitted work.

The repository `AGENTS.md` snapshot still describes `phase6` as the current branch. This remains a repository-instruction hygiene discrepancy, but it is not introduced by the targeted correction and does not invalidate the reviewed implementation.

---

# Prior Finding 1 — Superseded decision terminality

## Previous defect

The prior migration trigger allowed a `superseded` decision to be changed back into `active` if supersession metadata was cleared.

That violated the intended monotonic lifecycle model.

## Corrected trigger

Migration 075 now explicitly constrains active lifecycle transitions and makes terminal rows immutable.

The relevant semantics are:

- `active -> active`
- `active -> superseded`
- `active -> action_applied`

while:

- `legacy`
- `superseded`
- `action_applied`

are terminal.

The trigger now rejects any material UPDATE to a row whose old status is one of those terminal states.

This is stronger than merely rejecting the status change: it also prevents later mutation of terminal lifecycle metadata.

The existing semantic-identity fields remain independently protected by the immutable-meaning guard.

## Database-shape interaction

The migration CHECK constraints remain consistent with these lifecycle rules:

- `superseded` requires supersession timestamp/reason;
- non-superseded rows require supersession metadata to remain NULL;
- applied stop attribution requires `identity_status='action_applied'` and a valid stop-request shape.

Therefore there is no shape-valid path that can revive a terminal decision.

## Test coverage

The corrected migration tests cover:

- active -> superseded succeeds;
- superseded -> active fails;
- superseded -> action_applied fails;
- superseded terminal metadata mutation fails;
- active -> action_applied succeeds;
- action_applied -> active fails;
- action_applied -> superseded fails;
- action-applied attribution metadata mutation fails;
- legacy -> active fails;
- immutable semantic meaning remains protected.

### Finding 1 disposition

**PASS — corrected.**

---

# Prior Finding 2 — Equal-epoch older decision supersession

## Previous defect

The earlier persistence path superseded:

- lower checkpoint epochs; and
- changed identities for the same checkpoint evaluation;

but did not supersede a different checkpoint evaluation at the same epoch when its `checkpoint_eval_id` was lower.

That disagreed with the ordering already used by the stop fence.

## Corrected persistence ordering

`PersistCheckpointPolicyDecision()` now supersedes prior active decisions when:

- `checkpoint_epoch < new_epoch`; or
- `checkpoint_epoch == new_epoch AND checkpoint_eval_id < new_eval_id`; or
- `checkpoint_eval_id == new_eval_id` for changed semantic identity.

This establishes the intended ordering:

1. higher checkpoint epoch is newer;
2. at equal epoch, higher checkpoint_eval_id is newer.

Same-evaluation changed-policy/evidence history is handled explicitly.

## Stop fence ordering

`ApplyCheckpointPolicyStopRequest()` uses the same authority key for detecting a newer checkpoint decision:

- higher checkpoint epoch; or
- equal epoch and higher checkpoint_eval_id.

Thus a stale same-epoch/lower-eval stop candidate cannot act.

## Status authority

Read-only status orders nonterminal checkpoint authority by:

- checkpoint epoch descending;
- checkpoint_eval_id descending;
- checkpoint_decision_id descending as a deterministic final tie-breaker.

It intentionally gives `action_applied` rows precedence over ordinary active observations.

This is not an ordering inconsistency. An `action_applied` row represents a terminal lifecycle side effect already committed to the experiment and cannot be replaced/cancelled by later observational evaluations. That terminal precedence is consistent with the stop-attribution contract.

## Test coverage

The corrected migration tests explicitly cover:

- epoch 20 superseded by epoch 40;
- same epoch eval 201 superseded by eval 202;
- exactly one active decision remains for that parent;
- same checkpoint_eval_id with changed policy/evidence supersedes the old semantic identity;
- identical semantic reevaluation creates no duplicate;
- newer same-epoch authority fences the older stop candidate.

### Finding 2 disposition

**PASS — corrected.**

---

# Concurrency and transaction verification

## Parent experiment serialization

`LoadCheckpointPolicyConfig()` selects the parent `experiment` row with:

`FOR UPDATE`

The automatic and manual mutating evaluation paths use the same evaluator and remain inside a single PostgreSQL work transaction.

That row lock serializes checkpoint-policy evaluations for the same parent experiment.

As a consequence:

- two checkpoint evaluations for the same parent cannot independently race through policy revision materialization and persistence;
- the later transaction sees decisions/supersession committed by the earlier one;
- equal-epoch authority ordering is applied deterministically.

This is an important safety property and closes the most concerning race around the targeted correction.

## Stop application

Before applying a stop, the code revalidates:

- exact checkpoint evidence;
- current policy revision/hash;
- decision active state;
- evidence watermark;
- absence of a previously applied terminal policy stop;
- absence of another stop request;
- authoritative active worker attempt;
- active scheduler lease/fencing token;
- future/reachable stop epoch;
- absence of a newer active/action-applied checkpoint authority.

The final experiment UPDATE repeats the material predicates atomically.

The subsequent decision-attribution UPDATE occurs in the same transaction; failure throws and prevents the experiment stop from committing without its decision attribution.

**Finding: PASS.**

---

# Append-only and idempotency regression

The semantic uniqueness identity remains:

`checkpoint_eval_id + policy_revision + policy_hash + evidence_watermark`

Persistence uses `ON CONFLICT ... DO NOTHING`, not a semantic `DO UPDATE`.

Identical reevaluation therefore reuses the existing row.

Changed policy/evidence creates a new row and supersedes the previous active identity.

Terminal historical decisions are not rewritten.

**Finding: PASS.**

---

# Stop attribution regression

An applied checkpoint stop remains bound through:

`experiment.checkpoint_policy_stop_decision_id`

The attributed decision remains `action_applied` and terminal.

Later evaluations after a terminal stop are persisted as superseded observations and cannot replace/cancel the applied stop attribution.

Manual stop-clear behavior remains a separate operator workflow.

**Finding: PASS.**

---

# Legacy decision and policy behavior regression

The pure checkpoint-policy logic remains unchanged in meaning:

- leader-score pass OR
- inference-accuracy pass OR
- top-N pass

=> continue.

Only when all configured continue criteria fail is a stop requested.

Grace and existing stop-mode calculations remain covered by the hardening tests.

Historical pre-075 rows remain `legacy` and cannot be promoted into hardened active decisions.

**Finding: PASS.**

---

# Profitability isolation

Profitability remains completely non-decision-bearing in this prerequisite.

The packaged isolation test verifies profitability is absent from:

- checkpoint decision evaluation;
- pure policy decision logic;
- checkpoint policy canonical/hash identity;
- checkpoint evidence identity/watermark;
- rank/rank-population calculation;
- requested stop-epoch calculation;
- stop application/fencing.

The standalone checkpoint semantic/evidence module contains no profitability references.

Training remains a consumer of `stop_after_checkpoint_epoch` only and is not coupled to the checkpoint decision table.

The targeted correction did not modify continuation-profitability semantics or recommendation/Campaign Manager scoring.

**Finding: PASS.**

---

# Read-only status regression

`--checkpoint-policy-status` remains implemented as a read-only transaction.

The packaged isolation guard verifies it does not call:

- mutating checkpoint evaluation;
- decision persistence;
- stop application;
- read/write transaction conversion.

It exposes current policy revision/hash and decision/evidence identity without mutation.

**Finding: PASS.**

---

# Migration 075 assessment

Migration 075 remains the correct place for the correction because the supplied workflow states it has not yet been operationally applied.

The correction tightens the migration itself rather than creating a follow-on migration.

The migration remains replay-oriented according to the supplied disposable-schema test, which applies migration 075 twice.

Historical rows remain preserved as legacy.

The semantic unique index supports append-only hardened identity.

No profitability decision field or constraint is added.

### Operational caution

Migration 075 is not operationally non-blocking. It alters populated tables, replaces constraints/indexes/trigger behavior, and adds foreign keys/indexes.

Apply it during a quiet database window after the clean Release build and after checking live PostgreSQL transactions/locks.

**Finding: PASS with normal operational-lock caution.**

---

# Test/build evidence

The targeted-correction report records successful execution of:

- `CheckpointPolicyHardeningTests`
- `CheckpointPolicyMigrationTests`
- `CheckpointPolicyIsolationTests`
- `ContinuationPolicyInheritanceTests`
- `ContinuationProfitabilityPolicyTests`
- `ContinuationProfitabilityPolicyIsolationTests`
- `InferenceProfitabilityTests`
- `InferenceProfitabilityRepositoryTests`
- `ExperimentRecommendationScoringTests`
- `git diff --check`
- Debug `LSTM Release` build

Migration 075 was reportedly applied twice successfully in the disposable-schema migration harness.

The independent package's `git_diff_check.txt` is empty.

The canonical Release build remains pending because the worktree is intentionally dirty and Release provenance requires a clean committed tree.

The known migration-074 disposable-schema constraint-name collision remains unrelated to this targeted correction.

---

# Final verification matrix

```text
Prior FAIL: superseded decision terminality:          PASS
Prior FAIL: equal-epoch older supersession:           PASS

Legacy checkpoint OR semantics:                       PASS
Grace semantics:                                      PASS
Stop-mode compatibility:                              PASS
Policy revision/hash identity:                        PASS
Exact analysis/inference binding:                     PASS
Evidence watermark:                                   PASS
Rank-population identity:                             PASS
Append-only semantic history:                         PASS
Idempotent reevaluation:                              PASS
Changed policy/evidence supersession:                 PASS
Parent-row concurrency serialization:                 PASS
Old-checkpoint stop fencing:                          PASS
Equal-epoch stop fencing:                             PASS
Scheduler lease/fencing checks:                       PASS
Stop request + attribution atomicity:                 PASS
Applied-stop terminality:                             PASS
Historical legacy preservation:                       PASS
Read-only policy status:                              PASS
Profitability non-decision-bearing:                   PASS
Continuation profitability isolation:                 PASS
Campaign/recommendation isolation:                    PASS
Debug build evidence:                                 PASS
git diff --check:                                     PASS

Canonical clean Release build:                        PENDING
Migration 075 operational application:                PENDING
Live read-only checkpoint-policy verification:        PENDING
```

# Disposition

## PASS — ready to stage/commit

No further targeted code correction is required from this independent reverification.

The correct next sequence is:

1. archive/stage the prerequisite implementation and review artifacts;
2. commit;
3. confirm clean worktree;
4. run canonical Release build;
5. check PostgreSQL activity/locks;
6. apply migration 075;
7. perform live read-only `--checkpoint-policy-status` verification;
8. only after those pass, proceed to the narrow profitability-aware Phase 2C implementation.

Actual Phase 2C profitability activation should not begin before the clean build, operational migration, and read-only live verification complete.
