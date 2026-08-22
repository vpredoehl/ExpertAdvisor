# Campaign Manager Phase 3A — Independent Diff Review Findings

**Target:** FINAL profitability provenance integration, Migration 076
**Requested disposition:** PASS / TARGETED CORRECTION REQUIRED / REDESIGN REQUIRED
**Disposition:** **TARGETED CORRECTION REQUIRED**

## Review scope and source limitation

The packaged `.tar.gz` shown in the Terminal transcript was not present in the active runtime filesystem for this review. I therefore reviewed the actual Phase 3A source/SQL/test diffs embedded in the uploaded implementation transcript, together with the implementation report and earlier Campaign Manager design inspection. This is sufficient to evaluate the implementation logic and migration content, but it is not a checksum-level verification of the separately packaged archive.

## Executive conclusion

Phase 3A is architecturally sound and stayed within the intended non-decision-bearing boundary:

- authoritative FINAL profitability is loaded through the existing exact final-inference resolver;
- checkpoint profitability is excluded;
- recommendation profitability is snapshotted immutably;
- evaluation provenance carries the frozen observational profitability evidence;
- profitability weight and contribution are exactly zero;
- score, ranking, tie-breaking, eligibility, campaign-plan identity, and campaign selection remain unchanged;
- Phase 3B and Phase 3C were not accidentally implemented.

However, Migration 076 has one material database-boundary provenance weakness.

The migration trigger validating `source_final_inference_eval_result_id` proves only that the referenced inference result:

- has the referenced ID;
- uses `NEW.source_model_id`;
- is `completed`;
- has `inference_scope='final'`;
- has no checkpoint evaluation;
- has no parent experiment.

It does **not** independently prove that this final inference result belongs to the recommendation's `source_experiment_id` / exact source context.

When a profitability observation is present, the second half of the trigger closes most of that gap because the observation is checked against `NEW.source_experiment_id`, model, inference result, scope, range, and snapshotted metrics. But for a Phase-3A row representing **unavailable profitability** with a non-NULL `source_final_inference_eval_result_id`, the database can accept an inference-result reference that is final/completed for the same model but not proven to be the exact FINAL result for the recommendation's source experiment/context.

The C++ source loader is stricter and uses `ResolveExactFinalInferenceResult(experiment_id, last_model_id)` before `SelectAuthoritativeObservation(...)`, so the normal application path does not appear to create this bad state. The defect is therefore a **schema/provenance integrity gap**, not a runtime selection bug.

This should be corrected before commit because Phase 3A's stated purpose is immutable, authoritative provenance.

---

# 1. Migration 076 correctness

## What passes

Migration 076 is additive and leaves historical rows honest:

- legacy/pre-3A rows use `final_profitability_provenance_version IS NULL`;
- Phase-3A unavailable rows have version 1 plus an explicit unavailable reason;
- available rows require a FINAL profitability observation and provenance payload;
- zero actionable observations permit aggregate zero and require average return to remain NULL;
- NaN/Infinity are rejected;
- hash formats are constrained;
- recommendation profitability snapshot fields are protected by an immutability trigger;
- indexes are added for profitability-observation lookup;
- evaluation rows remain append-only to the runtime role through the existing migration-034 permission contract.

The migration's shape checks correctly keep legacy, unavailable, and available evidence distinct.

## Defect: incomplete database validation of the FINAL inference-result reference

The trigger `validate_recommendation_final_profitability_provenance()` checks the referenced `inference_eval_result` using predicates equivalent to:

```sql
result.id = NEW.source_final_inference_eval_result_id
AND result.model_id = NEW.source_model_id
AND result.status = 'completed'
AND result.inference_scope = 'final'
AND result.checkpoint_eval_id IS NULL
AND result.parent_experiment_id IS NULL
```

This is necessary but not sufficient for the exact Phase-3A provenance claim.

It does not bind that inference result to `NEW.source_experiment_id`, nor does this first validation branch prove the exact experiment inference context/range/configuration that `ResolveExactFinalInferenceResult()` proves in application code.

### Why this matters

Migration 076 explicitly allows a Phase-3A unavailable row with:

- `final_profitability_provenance_version = 1`;
- `source_final_profitability_observation_id IS NULL`;
- an explicit unavailable reason;
- `source_final_profitability_inference_scope = 'final'`.

That shape does not require the profitability observation branch to run.

If `source_final_inference_eval_result_id` is populated on such a row, only the weaker first inference-result validation protects it.

The database should not be able to persist immutable provenance stating that a recommendation observed final inference result X unless X is proven to be the exact final inference result for the recommendation source.

### Required correction

Strengthen Migration 076's FINAL inference validation to use repository-native exact source provenance.

At minimum, prove the result is associated with the recommendation source experiment through the authoritative model/experiment linkage. If the schema supports the same context fields used by `ResolveExactFinalInferenceResult()`, validate the complete exact source context or delegate to an equivalent invariant.

The corrected trigger should prevent:

- a final result from another source experiment;
- a final result for a model not belonging to the recommendation source experiment;
- a final result from another inference interval/context from being frozen as the recommendation's exact final result.

Do not solve this with “latest” or `MAX(id)` logic.

Add a migration test that deliberately constructs two plausible final inference rows / source contexts and proves the wrong one is rejected at the **database boundary**, including the unavailable-profitability case.

**Finding:** FAIL — targeted correction required.

---

# 2. Exact FINAL profitability source loading

The application path is stronger than the migration trigger.

The implementation report and transcript show use of:

- `ResolveExactFinalInferenceResult(experiment_id, last_model_id)`;
- `SelectAuthoritativeObservation(...)`.

The resolver requires exact final context and the observation selector binds the profitability observation to the exact inference-result ID, source experiment, model, final scope, checkpoint absence, and current metric identity.

No `MAX(id)`, “latest final for model”, checkpoint fallback, or backtest fallback was introduced.

**Finding:** PASS.

---

# 3. Checkpoint-profitability exclusion

The implementation explicitly requires FINAL scope and `checkpoint_eval_id IS NULL`.

Migration 076's observation trigger likewise requires the referenced profitability observation to have:

- `inference_scope='final'`;
- `checkpoint_eval_id IS NULL`.

The migration tests include a checkpoint-observation rejection case.

**Finding:** PASS.

---

# 4. Immutable recommendation snapshot

The recommendation row freezes:

- provenance version;
- exact FINAL inference result ID;
- exact profitability observation ID;
- unavailable reason;
- scope/range;
- actionable count;
- aggregate return;
- average return;
- metric-definition hash;
- source-content hash;
- observation-identity hash.

The migration installs `reject_recommendation_profitability_snapshot_mutation()` and tests direct snapshot mutation rejection.

A later profitability observation cannot reinterpret the already-frozen recommendation.

**Finding:** PASS.

---

# 5. Evaluation provenance

Evaluation results copy the frozen recommendation profitability evidence and add:

- `profitability_evidence_canonical`;
- `profitability_evidence_hash`.

Repository retry equality checks include the profitability snapshot and observational canonical/hash, so an idempotent retry cannot silently reinterpret the evaluation.

Existing migration-034 permissions revoke UPDATE/DELETE on evaluation results from the runtime role, preserving append-only evaluation history.

**Finding:** PASS.

---

# 6. Unavailable versus zero-actionable semantics

The migration distinguishes:

### Legacy

`final_profitability_provenance_version IS NULL` and all Phase-3A fields NULL.

### Unavailable

Version 1, explicit unavailable reason, no profitability observation/value payload.

### Available zero actionable

- valid profitability observation ID;
- `actionable_count = 0`;
- aggregate return may be zero;
- average return must be NULL.

This preserves the required difference between “no evidence” and a valid observation with zero actionable predictions.

**Finding:** PASS.

---

# 7. Exact-zero scoring proof

The implementation defines constants equivalent to:

```cpp
kPhase3AProfitabilityScoringWeight = 0.0;
kPhase3AProfitabilityScoreContribution = 0.0;
```

and protects both with compile-time `static_assert(... == 0.0)` checks.

Profitability was not introduced as a normal score component.

That is stronger than a merely small configurable weight.

**Finding:** PASS.

---

# 8. Score / ranking / tie / eligibility invariance

The Phase 3A evidence is deliberately excluded from:

- candidate eligibility;
- scoring-policy canonical/hash;
- existing score components;
- final score calculation;
- ranking keys;
- ranking identity;
- tie-breaking;
- recommendation source selection.

The dedicated tests compare decision-bearing evidence/identity before and after changing profitability and cover positive, negative, missing, zero-actionable, ranking-reversal, and tie cases.

**Finding:** PASS.

---

# 9. Campaign plan / action invariance

The implementation follows the earlier Phase 3 design boundary:

- campaign planning may print observed FINAL profitability;
- the existing decision-bearing `profitabilityMetric` / `minimumProfitability` path remains inactive;
- profitability does not enter campaign-plan identity;
- selection/decision reasons remain unchanged;
- Campaign Manager dispatch remains downstream of persisted approved/materialized workflow.

No Phase-3A profitability activation was introduced.

**Finding:** PASS.

---

# 10. Observational identity/hash boundary

The new `profitability_evidence_canonical/hash` is observational.

It is intentionally excluded from:

- recommendation decision identity;
- scoring-policy identity;
- evaluation decision-bearing evidence/identity;
- ranking identity and tie-breaks;
- campaign-plan identity.

At the same time, evaluation persistence retains enough frozen provenance to reconstruct the profitability evidence observed at evaluation time.

This is the correct Phase 3A boundary.

**Finding:** PASS.

---

# 11. Concurrency / persistence safety

The normal workflow persists recommendation source evidence as one recommendation snapshot and evaluation evidence as immutable evaluation results.

No live re-join is used to reinterpret historical profitability during campaign planning; the design follows the ranking-member -> exact evaluation-result -> frozen profitability provenance path.

The primary remaining concern is the migration validation gap described above, not a transaction race in the reviewed application path.

**Finding:** PASS with the schema-integrity correction required above.

---

# 12. Phase 3B boundary

The implementation does not repair existing scoring-policy/ranking homogeneity or stale cross-policy comparison behavior.

That remains appropriately deferred to Phase 3B.

**Finding:** PASS.

---

# 13. Phase 3C boundary

The reviewed Phase 3A implementation adds none of:

- profitability threshold;
- profitability eligibility gate;
- nonzero profitability weight;
- profitability normalization;
- percentile/z-score/winsorization;
- profitability ranking;
- profitability tie-break;
- profitability source preference;
- profitability-based campaign rejection/selection.

**Finding:** PASS.

---

# 14. Test adequacy

The test suite is broad and behavior-oriented rather than field-presence-only.

Reported passing coverage includes:

- Phase 3A profitability domain tests;
- exact source repository tests;
- recommendation repository tests;
- migration 076 tests;
- candidate generator;
- scoring;
- evaluation;
- ranking;
- campaign planning;
- campaign approval;
- inference profitability;
- continuation profitability/isolation;
- checkpoint hardening/isolation/migration.

The targeted missing test is:

> database-level rejection of a wrong-but-final inference result for a recommendation source in the unavailable-profitability case.

Add that as part of the correction.

The known migration-074 test harness failure remains unrelated to Phase 3A.

---

# 15. Build/diff status

Reported:

- `git diff --check`: PASS;
- Debug build with provenance script and sandboxing disabled: PASS;
- canonical Release build: pending clean commit because Release provenance rejects dirty worktrees.

No Release failure attributable to Phase 3A source code was reported.

---

# Final disposition

```text
Migration 076 shape / legacy handling:            PASS
Exact FINAL C++ source binding:                    PASS
Checkpoint-profitability exclusion:               PASS
Immutable recommendation snapshot:                PASS
Evaluation provenance:                            PASS
Unavailable-vs-zero semantics:                    PASS
Profitability weight exact zero:                  PASS
Score invariance:                                 PASS
Ranking invariance:                               PASS
Tie invariance:                                   PASS
Eligibility invariance:                           PASS
Campaign plan/action invariance:                  PASS
Observational identity/hash boundary:             PASS
Phase 3B boundary:                                PASS
Phase 3C boundary:                                PASS
Test breadth:                                     PASS

DB exact FINAL inference-result/source binding:   FAIL

Overall: TARGETED CORRECTION REQUIRED
Ready to stage/commit:                            NO
Ready for Phase 3B:                               NO
```

## Recommended next step

Make one narrow correction to Migration 076 and its migration tests:

1. bind `source_final_inference_eval_result_id` at the database boundary to the exact recommendation source experiment/context, not merely the source model + final scope;
2. add an adversarial unavailable-profitability fixture proving a wrong final inference result cannot be persisted;
3. rerun migration 076 tests and the Phase 3A repository/invariance suites;
4. package the corrected diff for short independent reverification.

Do not broaden the correction into Phase 3B or Phase 3C.
