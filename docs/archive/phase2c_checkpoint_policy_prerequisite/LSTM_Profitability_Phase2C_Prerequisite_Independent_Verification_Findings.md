# Phase 2C Prerequisite — Independent Verification Findings

**Target:** Checkpoint-policy lifecycle / decision-identity hardening prerequisite
**Phase boundary:** Profitability must remain completely non-decision-bearing
**Result:** **TARGETED CORRECTION REQUIRED BEFORE COMMIT**

## Executive conclusion

The prerequisite implementation is architecturally strong and substantially satisfies the intended hardening design:

- legacy checkpoint-policy OR semantics are preserved;
- semantic policy revision/hash is implemented;
- exact analysis + checkpoint inference provenance is bound into the evidence identity;
- decisions are append-only by semantic identity;
- stop actions are attributed to an exact durable decision;
- stop application is fenced against stale policy/evidence, old checkpoints, worker-attempt drift, scheduler authority drift, existing stop state, and unreachable stop epochs;
- read-only `--checkpoint-policy-status` exists;
- profitability remains completely non-decision-bearing.

The independent review did **not** find a path by which profitability influences checkpoint evaluation, rank, semantic identity, evidence identity, requested stop epoch, stop application, continuation policy, Campaign Manager scoring, or training.

However, two lifecycle-state defects remain in the new hardening layer. They are small and targeted, but they matter because the purpose of this prerequisite is to make durable checkpoint decisions monotonic and authoritative before profitability can influence a live worker.

Therefore the implementation should **not be committed yet**. Apply a targeted correction, rerun the focused migration/lifecycle tests, then reverify.

---

## Verified architecture

### 1. Legacy decision behavior is preserved

`CheckpointPolicy.cpp` extracts the existing policy into a deterministic pure function.

The continue rules remain OR-based:

- leader-score passes; OR
- inference-accuracy passes; OR
- top-N passes

=> `continue`.

Only when all configured continue rules fail does the evaluator produce `stop_requested`.

Grace behavior and requested-stop-epoch behavior are preserved for:

- `next_checkpoint`;
- `current_checkpoint_if_possible`;
- `mark_pruned_when_not_running`.

The new pure tests explicitly cover these legacy semantics.

**Finding: PASS.**

### 2. Semantic policy identity is appropriate

The canonical policy identity includes:

- enabled;
- min leader score;
- min inference accuracy;
- top-N;
- rank scope;
- stop mode;
- grace evaluations;
- checkpoint interval;
- target epochs.

It intentionally excludes observational/runtime fields including:

- current epoch;
- persisted hash;
- policy revision itself;
- worker attempt;
- status/phase.

Including checkpoint interval and target epochs is justified because they can change the requested stop epoch.

The deterministic 64-bit FNV-1a hash is stable and independent of continuation-policy semantics.

**Finding: PASS.**

### 3. Policy revision behavior is coherent

Checkpoint-policy control mutations compare canonical semantic configuration before incrementing `checkpoint_policy_revision`.

Mutating evaluation also detects out-of-band semantic drift:

- NULL legacy hash is materialized without incrementing revision;
- a non-NULL hash that disagrees with current semantic configuration increments the revision and rematerializes the hash.

Read-only status computes the current hash but does not materialize or increment it.

**Finding: PASS.**

### 4. Exact checkpoint inference provenance is now bound

`LoadValidatedCheckpointPolicyAnalysis()` requires exactly one matching completed checkpoint inference result and verifies consistency across:

- checkpoint_eval_id;
- parent experiment;
- checkpoint model;
- checkpoint epoch;
- analysis linkage;
- symbol;
- prediction horizon;
- checkpoint inference scope;
- completed inference status;
- threshold;
- inference range;
- analysis/inference accuracy.

Final inference is not used as a fallback.

Ambiguous/multiple completed checkpoint inference rows are rejected as `checkpoint_inference_result_not_exactly_one`.

**Finding: PASS.**

### 5. Evidence watermark is materially stronger

The evidence watermark contains:

- checkpoint evaluation;
- parent experiment;
- checkpoint model and epoch;
- observed current epoch;
- exact analysis ID;
- exact checkpoint inference-result ID;
- symbol and horizon;
- inference range;
- leader score;
- inference accuracy;
- rank when decision-bearing;
- rank scope;
- rank-population watermark;
- completed-evaluation count/population watermark;
- lifecycle status fields.

Profitability is absent.

Including observed current epoch is sensible because current progress can alter the requested safe stop checkpoint.

**Finding: PASS.**

### 6. Append-only semantic decision persistence works

The new semantic uniqueness key is:

`checkpoint_eval_id + policy_revision + policy_hash + evidence_watermark`

Identical reevaluation uses `ON CONFLICT ... DO NOTHING` and then reuses the exact existing row.

The persistence path does not use `DO UPDATE` to rewrite decision meaning.

Different policy/evidence identities produce distinct rows.

Legacy historical rows remain explicit `legacy` rows with NULL hardened identity fields.

**Finding: PASS, subject to lifecycle-transition defect #1 below.**

### 7. Stop application is strongly fenced

The stop action revalidates exact evidence immediately before action.

It then checks, within the same transaction and while the parent row is locked:

- parent remains `running/train`;
- policy revision/hash remain current;
- decision remains active;
- decision evidence watermark remains exact;
- no terminal policy stop has already been applied;
- no conflicting stop request already exists;
- an active training worker attempt exists;
- the worker attempt belongs to the current scheduler authority;
- worker lifecycle is compatible with active training;
- scheduler lease is active and unexpired;
- worker fencing token matches the active scheduler lease;
- requested stop epoch is still in the future;
- requested stop epoch is before target;
- no newer active/action-applied checkpoint decision exists.

The final UPDATE repeats the material fence predicates atomically.

If attribution UPDATE fails, the transaction throws; because the experiment stop write and attribution write share the enclosing transaction, the stop cannot commit without attribution.

**Finding: PASS.**

### 8. Applied stop attribution is durable

`experiment.checkpoint_policy_stop_decision_id` identifies the exact decision that caused `stop_after_checkpoint_epoch`.

The durable decision records:

- `stop_request_applied`;
- application timestamp;
- worker-attempt ID;
- requested stop epoch.

The migration trigger prevents an `action_applied` decision from leaving that state and prevents `stop_request_applied` from reverting true -> false.

**Finding: PASS.**

### 9. Read-only status really is read-only by construction

`RunCheckpointPolicyStatusCommand()` begins a read-only transaction and does not invoke:

- mutating checkpoint evaluation;
- decision persistence;
- stop application;
- scheduler mutation.

It reports current policy identity, exact evidence, evidence watermark, durable decisions, authoritative decision, last decision, and stop attribution.

**Finding: PASS.**

### 10. Profitability isolation is preserved

The extracted checkpoint semantic/evidence module contains no profitability references.

The static isolation suite explicitly rejects profitability references from:

- decision evaluation;
- rank;
- rank-population loading;
- stop application;
- policy canonical identity;
- evidence canonical identity;
- stop-epoch calculation;
- pure decision function;
- recommendation scoring.

Training remains a consumer of `stop_after_checkpoint_epoch` and does not know the checkpoint decision table.

**Finding: PASS.**

---

# Targeted correction required

## Defect 1 — `superseded` is not terminal at the database boundary

### Current behavior

Migration 075's immutability trigger prevents:

- mutation of decision meaning;
- `stop_request_applied=true` from reverting to false;
- `action_applied` from transitioning to another state.

But it does **not** prohibit:

`superseded -> active`

A superseded row can be revived if an UPDATE also clears:

- `superseded_at`;
- `superseded_reason`;
- `superseded_by_decision_id`.

That UPDATE satisfies the current CHECK constraints, and the immutability trigger does not reject the lifecycle transition.

The application code currently only writes:

- `active -> superseded`;
- `active -> action_applied`.

So the normal application path does not intentionally revive decisions.

However, the database schema is an authoritative workflow contract, and this prerequisite specifically exists to make stale/superseded checkpoint decisions durable and non-authoritative. Allowing a superseded decision to become active again violates that lifecycle contract and weakens the claimed immutability guarantee.

### Why it matters

A stale decision is explicitly being retired because it must no longer be able to act on a live worker.

Once superseded, its authority should be monotonic:

`active -> superseded` is terminal.

It should never be possible for a later bug, maintenance query, or future code path to revive the row.

### Required correction

Strengthen `expertadvisor_guard_checkpoint_decision_immutable()` so that:

- `legacy` cannot transition to another state;
- `superseded` cannot transition to another state;
- `action_applied` cannot transition to another state;
- `active` may transition only to:
  - `active` for allowed non-semantic lifecycle metadata updates, if needed;
  - `superseded`;
  - `action_applied`.

At minimum add a guard equivalent in semantics to:

`IF OLD.identity_status = 'superseded' AND NEW.identity_status <> 'superseded' THEN RAISE ...`

and add a deterministic migration test proving a superseded row cannot be revived.

**Severity:** material prerequisite defect, targeted fix.

---

## Defect 2 — newer same-epoch checkpoint can leave older decision marked `active`

### Current behavior

When a newly inserted decision becomes active, `PersistCheckpointPolicyDecision()` supersedes prior active rows when:

- their checkpoint epoch is lower; OR
- they have the same `checkpoint_eval_id` (policy/evidence changed for that evaluation).

It does **not** supersede a different checkpoint evaluation with:

- the same checkpoint epoch; and
- a lower checkpoint_eval_id.

Yet elsewhere the implementation explicitly defines newer authority at equal epoch using the higher `checkpoint_eval_id`.

Both:

- authoritative status ordering; and
- stop fencing

treat the higher `checkpoint_eval_id` as newer.

As a result, after a newer same-epoch evaluation is inserted, the older row can remain:

`identity_status='active'`

even though it is no longer authoritative.

The atomic stop fence prevents that older row from actually applying a stop because it checks for a newer active/action-applied row. So this is **not currently a worker-stop safety bypass**.

It is nevertheless inconsistent with the lifecycle model:

- `active` is supposed to represent a currently authoritative/non-superseded decision;
- status can show multiple active decisions for the same parent;
- the older one is logically stale but not marked superseded.

### Required correction

When a new active decision becomes authoritative, supersede prior active rows that are older under the same total ordering used by authority/fencing:

- lower checkpoint epoch; OR
- same checkpoint epoch and lower checkpoint_eval_id; OR
- same checkpoint evaluation with a different policy/evidence identity.

The supersession predicate should match the ordering used by:

- the newer-decision fence;
- the authoritative decision query.

Add a deterministic test with two different checkpoint_eval_ids at the same checkpoint epoch and prove that only the newer one remains active.

**Severity:** lifecycle/observability correctness defect; targeted fix.

---

# Additional observations

## Migration 075 locking characteristics

Migration 075 is additive in data shape, but it is not operationally "non-blocking."

It:

- alters populated tables;
- drops an existing unique index;
- creates new indexes;
- adds and validates CHECK/FK constraints;
- creates/replaces a trigger function and trigger.

Operational application should therefore be performed during a quiet database window, with active transactions/locks checked first.

This is not an implementation defect.

## Release build

The implementation report states that Debug passed but the canonical Release build stopped because the worktree was intentionally dirty and build provenance requires a clean tree.

That is expected before commit and does not count as a source failure.

After targeted correction and independent reverification, the correct sequence remains:

1. stage/archive;
2. commit;
3. verify clean worktree;
4. canonical Release build;
5. apply migration 075;
6. perform read-only `--checkpoint-policy-status` operational verification.

## Migration-074 test harness failure

The supplied implementation report attributes the migration-074 harness failure to database-wide constraint-name lookup colliding with constraints already installed in the production database.

The Phase 2C prerequisite patch does not modify migration 074 or continuation semantics.

Nothing in the packaged Phase 2C diff contradicts that diagnosis.

This independent archive does not include a runnable production PostgreSQL instance, so the diagnosis cannot be reproduced here against the user's live database. It should be treated as an unrelated existing test-isolation problem unless live verification shows otherwise.

---

# Recommended disposition

Do **not** start actual profitability-aware Phase 2C yet.

Apply one targeted checkpoint-decision lifecycle correction covering both findings:

1. make `superseded` terminal in the database immutability trigger;
2. supersede older same-epoch/lower-eval active decisions according to the same total ordering used by authoritative selection and stop fencing;
3. add migration tests for terminal supersession;
4. add deterministic persistence/lifecycle coverage for equal-epoch newer checkpoint authority;
5. rerun:
   - checkpoint hardening tests;
   - checkpoint migration tests;
   - checkpoint isolation tests;
   - continuation profitability/isolation regressions;
   - recommendation scoring regression;
   - `git diff --check`;
   - Debug build.

Then package the corrected diff for a short independent reverification.

---

# Verification status

```text
Legacy checkpoint OR semantics:                 PASS
Grace semantics:                                PASS
Stop-mode calculation compatibility:            PASS
Semantic policy hash:                           PASS
Policy revisioning:                             PASS
Exact analysis/inference binding:                PASS
Evidence watermark:                             PASS
Rank-population identity:                       PASS
Append-only semantic decision identity:          PASS
Stop-action fencing:                            PASS
Stop-action attribution:                        PASS
Atomic stop + attribution transaction:           PASS
Applied-stop terminality:                       PASS
Read-only policy status:                        PASS
Profitability non-decision-bearing:              PASS
Continuation isolation:                         PASS
Campaign/recommendation isolation:               PASS

Superseded decision terminality:                FAIL
Equal-epoch older decision supersession:         FAIL

Independent prerequisite result:                TARGETED CORRECTION REQUIRED
Ready to commit:                                NO
Ready for actual Phase 2C profitability:         NO
```

## Bottom line

The implementation is close. The hard concurrency and stop-fencing architecture is sound in the reviewed paths. The remaining issues are concentrated in lifecycle-state monotonicity rather than the core evaluator or worker fence.

Fix those two state-transition defects before committing. Once corrected and reverified, this prerequisite should be in good shape to support the narrow profitability-aware Phase 2C increment.
