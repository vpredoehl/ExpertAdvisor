# Campaign Manager Phase 3A — Migration 076 Targeted-Correction Independent Reverification

**Target:** Migration 076 FINAL-inference/source binding correction
**Archive:** `LSTM_CampaignManager_Phase3A_Migration076_FinalInferenceBinding_Reverification.tar.gz`
**Disposition:** **PASS — Phase 3A ready to stage/commit**

## Executive conclusion

The targeted correction fully resolves the prior database-boundary provenance defect.

Migration 076 now validates `source_final_inference_eval_result_id` against the same materially relevant source context used by the repository's authoritative `ResolveExactFinalInferenceResult()` logic. The corrected database contract binds the referenced FINAL inference result to:

- the recommendation's source experiment;
- the recommendation's source model;
- the experiment's `last_model_id`;
- the model's immutable `experiment_id` lineage;
- symbol;
- prediction horizon;
- threshold;
- window size;
- label rule;
- target type;
- completed epochs;
- exact configured inference date range;
- completed FINAL scope;
- non-checkpoint / non-parented inference shape;
- and uniqueness of the exact matching result.

The critical unavailable-profitability case is now protected independently of profitability-observation presence. A recommendation may validly record the exact FINAL inference result while profitability is unavailable and `source_final_profitability_observation_id` is NULL, but a wrong-experiment, wrong-range, or checkpoint inference result is rejected directly by migration-076 validation.

No production C++ changed in the targeted correction, and profitability remains exactly non-decision-bearing.

The corrected Phase 3A implementation is ready for:

`archive/stage -> commit -> clean Release build -> operational migration 076 -> live read-only observability verification -> Phase 3B`

Phase 3C should remain deferred.

---

## 1. Archive integrity

The uploaded archive was extracted successfully.

All 31 entries listed in `SHA256SUMS.txt` verify successfully against the packaged files.

The archive includes:

- corrected migration 076;
- corrected adversarial SQL migration test;
- pre-correction migration/test snapshots;
- focused before/after diffs;
- authoritative C++ FINAL-inference resolver;
- recommendation source loader;
- model-lineage migrations;
- prior review/correction reports;
- Git state and diff metadata.

**Finding: PASS.**

---

## 2. Prior defect status

### Previous defect

The pre-correction trigger validated `source_final_inference_eval_result_id` only as:

- matching result ID;
- matching source model;
- completed;
- `inference_scope='final'`;
- no checkpoint evaluation;
- no parent experiment.

That was insufficient to prove that the result was the exact FINAL inference result for the recommendation source experiment/context.

This was especially problematic when profitability was unavailable, because no profitability observation existed to provide stricter source provenance.

### Corrected behavior

The targeted diff replaces that weak result check with an exact-result CTE structure modeled directly on `ResolveExactFinalInferenceResult()`.

**Prior defect status: FIXED.**

---

## 3. Exact FINAL source binding

Migration 076 now builds a `source_context` from the recommendation source:

- `experiment.experiment_id = NEW.source_experiment_id`;
- `experiment.last_model_id = NEW.source_model_id`;
- `model.model_id = experiment.last_model_id`;
- `model.experiment_id = experiment.experiment_id`.

This is the correct bidirectional lineage contract for the frozen recommendation source.

It also requires complete `train_config_meta` and derives the exact model inference context from the persisted model metadata.

The referenced `inference_eval_result` must then match that context exactly.

**Exact FINAL source binding: PASS.**

---

## 4. Model lineage

The correction legitimately relies on `model.experiment_id`.

Migration 012 establishes:

- `model.experiment_id`;
- an FK from model to experiment.

Migration 070 prevents a model whose `experiment_id` is already set from being redirected or cleared.

Therefore a model used to prove recommendation provenance cannot later be silently rebound to a different experiment.

The corrected migration additionally requires:

`model.experiment_id = experiment.experiment_id`

and:

`experiment.last_model_id = NEW.source_model_id`.

**Model-lineage proof: PASS.**

---

## 5. Exact inference range/context binding

The SQL materially mirrors `ResolveExactFinalInferenceResult()`.

It derives from `train_config_meta`:

- prediction horizon;
- threshold;
- window size;
- label-rule ID;
- completed epochs.

It derives target type from `target_meta`, defaulting to 1 in the same manner as the C++ resolver.

It derives the expected inference range from:

- `experiment.infer_start::date::text`;
- `experiment.infer_end::date::text`.

The inference result must match:

- model;
- symbol;
- prediction horizon;
- threshold;
- window size;
- label rule;
- target type;
- from date;
- to date;
- completed epochs;
- completed status;
- FINAL scope;
- `checkpoint_eval_id IS NULL`;
- `parent_experiment_id IS NULL`.

The SQL also requires exactly one matching exact result and that its ID equals the snapshotted `source_final_inference_eval_result_id`.

This is materially equivalent to the C++ resolver's exact-result selection contract.

**Exact range/context binding: PASS.**

---

## 6. Unavailable-profitability binding

This was the critical prior failure mode.

The corrected migration validates the FINAL inference-result reference before and independently of the profitability-observation branch.

The test creates a valid unavailable recommendation with:

- provenance version 1;
- correct `source_final_inference_eval_result_id`;
- `source_final_profitability_observation_id = NULL`;
- explicit unavailable reason;
- FINAL scope.

That row succeeds.

The adversarial unavailable cases then deliberately supply invalid FINAL provenance without an observation:

- checkpoint inference result;
- valid FINAL result belonging to another experiment/model context;
- correct experiment/model but wrong FINAL inference range.

All are expected to fail with SQLSTATE `23514`.

Thus missing profitability does not weaken FINAL-source provenance.

**Unavailable-profitability binding: PASS.**

---

## 7. Wrong-experiment FINAL rejection

The test fixture creates:

- experiment A with model 10;
- experiment B with model 20;
- a structurally valid completed FINAL result 200 for experiment B/model 20.

It then attempts to freeze result 200 into a recommendation whose source experiment is A.

The row is rejected by migration-076 provenance validation.

This is a direct database-boundary test and does not rely on the C++ source loader.

**Wrong-experiment rejection: PASS.**

---

## 8. Wrong-range FINAL rejection

The fixture creates result 102 as:

- completed;
- FINAL;
- same experiment/model as the valid source;
- otherwise structurally plausible;
- but with a different inference date range.

The unavailable-profitability attempt using result 102 is rejected.

A second adversarial case attaches a profitability observation that is internally consistent with that wrong-range result; it is still rejected because the result itself is not the exact FINAL source context.

This proves a profitability observation cannot legitimize a wrong FINAL result.

**Wrong-range/context rejection: PASS.**

---

## 9. Checkpoint inference rejection

The test directly attempts to freeze a checkpoint inference result into the FINAL inference provenance field while no profitability observation is attached.

It is rejected.

The corrected SQL independently requires:

- `inference_scope='final'`;
- `checkpoint_eval_id IS NULL`;
- `parent_experiment_id IS NULL`.

**Checkpoint rejection: PASS.**

---

## 10. Correct unavailable case

A Phase-3A unavailable recommendation with:

- correct source experiment/model;
- correct exact FINAL inference result;
- no profitability observation;
- explicit unavailable reason;

is accepted.

Therefore the correction did not accidentally make profitability availability mandatory.

**Correct unavailable case: PASS.**

---

## 11. Correct available case

The existing available positive-FINAL profitability snapshot remains accepted.

The correction did not break the normal exact observation path.

**Correct available case: PASS.**

---

## 12. Zero-actionable semantics

The migration fixture preserves a valid FINAL profitability observation with:

- `actionable_count = 0`;
- aggregate return = 0;
- average return = NULL.

That row remains valid and distinguishable from unavailable profitability.

**Zero-actionable semantics: PASS.**

---

## 13. Migration replay/idempotency

The packaged migration test applies migration 076 twice in the disposable test schema.

The correction report records both passes.

The corrected SQL continues to use the repository's replay-safe migration conventions.

**Migration replay: PASS.**

---

## 14. Snapshot immutability

The existing migration test still verifies that a frozen recommendation profitability snapshot cannot be modified later.

The targeted correction did not weaken the immutability trigger.

A later profitability observation also cannot reinterpret the historical frozen recommendation/evaluation provenance.

**Snapshot immutability: PASS.**

---

## 15. Profitability exact-zero invariant

The packaged `ExperimentRecommendation.hpp` still defines:

`kPhase3AProfitabilityScoringWeight = 0.0`

and:

`kPhase3AProfitabilityScoreContribution = 0.0`.

No C++ production file changed during the targeted correction.

Therefore the SQL provenance correction cannot alter the score formula.

**Profitability exact-zero invariant: PASS.**

---

## 16. Ranking / selection invariance

The targeted correction changes only:

- migration 076;
- the migration SQL fixture.

It does not change:

- candidate discovery;
- eligibility;
- scoring;
- ranking;
- tie-breaking;
- recommendation selection;
- campaign-plan identity;
- Campaign Manager action.

The correction report records the Phase 3A and existing recommendation/ranking/campaign regression suites as passing.

**Ranking/selection invariance: PASS.**

---

## 17. Phase 3B boundary

No ranking-policy homogeneity or cross-policy staleness remediation was introduced.

Those concerns remain correctly deferred to Phase 3B.

**Phase 3B boundary: PASS.**

---

## 18. Phase 3C boundary

The correction introduces none of:

- profitability thresholds;
- gates;
- nonzero scoring;
- normalization;
- profitability ranking;
- profitability tie-breaking;
- profitability-based selection.

**Phase 3C boundary: PASS.**

---

## 19. Test adequacy

The adversarial migration test directly exercises the database boundary rather than relying on application source loading.

It covers the defect that mattered most:

> unavailable profitability + wrong-but-FINAL inference result.

It also adds stronger wrong-range coverage, including the case where a profitability observation is internally consistent with the wrong result.

The reported broader regression runs include:

- migration 076 test twice;
- Phase 3A source repository tests;
- Phase 3A recommendation repository tests;
- Phase 3A profitability tests;
- inference profitability repository tests;
- continuation profitability isolation;
- checkpoint policy isolation;
- existing recommendation;
- candidate generation;
- scoring;
- evaluation;
- ranking;
- campaign planning.

One manually assembled campaign-planning link command initially omitted an existing dependency; the corrected complete command passed. This is not evidence of a product defect.

**Test adequacy: PASS.**

---

## 20. One implementation nuance

The corrected SQL duplicates the exact-final resolver logic rather than invoking a shared database function. That means future changes to the C++ resolver's provenance contract must also update migration-trigger logic or a later schema function.

This is not a defect in the present correction: the current SQL and C++ resolver materially match.

It is worth documenting as a maintenance invariant:

> authoritative FINAL inference context logic must stay synchronized between application source selection and database provenance validation.

This does not block Phase 3A.

---

## Final verification matrix

```text
Prior DB provenance defect:                       FIXED / PASS
Exact FINAL source binding:                       PASS
Exact range/context binding:                      PASS
Unavailable-profitability binding:                PASS
Wrong-experiment rejection:                       PASS
Wrong-range rejection:                            PASS
Checkpoint rejection:                             PASS
Correct unavailable case:                         PASS
Correct available case:                           PASS
Zero-actionable semantics:                        PASS
Migration replay:                                 PASS
Snapshot immutability:                            PASS
Model-lineage immutability:                       PASS
Profitability exact-zero invariant:               PASS
Ranking/selection invariance:                     PASS
Phase 3B boundary:                                PASS
Phase 3C boundary:                                PASS
Archive checksums:                                PASS

Overall disposition:                              PASS
Ready to stage/commit:                            YES
```

## Recommended next sequence

No additional targeted code correction is required from this reverification.

Proceed with:

1. archive the Phase 3A implementation/review/correction artifacts;
2. clean temporary review-package files;
3. stage the complete Phase 3A implementation;
4. run `git diff --cached --check`;
5. commit;
6. verify a clean worktree;
7. run the canonical clean Release build;
8. inspect live PostgreSQL activity/locks;
9. operationally apply migration 076;
10. perform live read-only recommendation/Campaign Manager observability verification;
11. proceed to Campaign Manager Phase 3B.

Do not activate Phase 3C profitability scoring until Phase 3B is complete and real profitability distributions have been inspected.
