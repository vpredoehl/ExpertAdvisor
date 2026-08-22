# LSTM Profitability Phase 2A — Independent Verification Findings

**Review target:** `LSTM_Profitability_Phase2A_Verification_Package.tar.gz`
**Review scope:** packaged Phase 2A source/test/documentation snapshot
**Verdict:** **PASS WITH ONE TARGETED CORRECTION RECOMMENDED BEFORE PHASE 2B**

## Executive summary

The packaged Phase 2A implementation is architecturally consistent with the intended behavior-neutral contract. Profitability is carried as optional diagnostic evidence alongside continuation evidence, is kept out of continuation policy identity, source selection, deduplication, evidence watermarks, ranking, thresholds, trend calculations, eligibility, automatic queue ordering, checkpoint policy, and recommendation scoring/ranking. Missing profitability remains distinct from a valid zero-actionable observation, and final/checkpoint scope separation is explicit.

One material provenance weakness was found in the **final-model diagnostic attachment path**: the continuation evidence query obtains the final `inference_eval_result` with a lateral query constrained only by `model_id`, `inference_scope='final'`, and `status='completed'`, then takes the latest row by completion time/ID. It does **not** constrain the row to the source experiment's expected symbol, horizon, threshold, or inference date range even though those values exist in `ContinuationSourceExperiment`. If more than one completed final inference row exists for the same model, Phase 2A can attach an authoritative profitability observation for the wrong final inference run relative to the continuation source context.

This does not currently change continuation decisions because profitability is diagnostic-only, so it is not a Phase 2A policy-isolation failure. However, the lookup should be corrected **before Phase 2B makes profitability decision-bearing**, and preferably as a targeted Phase 2A correction now.

## Findings

### F1 — Final-model inference-result selection is not fully bound to continuation source context

**Severity:** Medium now; High if carried into Phase 2B
**Status:** Correction recommended

In `Sources/ExperimentScheduler.cpp`, `LoadContinuationEvidence()` selects final inference evidence using logic equivalent to:

```sql
SELECT ier.id, ier.accept_model
FROM inference_eval_result ier
WHERE ier.model_id = a.model_id
  AND ier.inference_scope = 'final'
  AND ier.status = 'completed'
ORDER BY ier.completed_at DESC, ier.id DESC
LIMIT 1
```

The source continuation configuration already carries `symbol`, `predictionHorizon`, `cNextThreshold`, `inferStart`, and `inferEnd`, but these are not used to bind this lateral final-inference lookup.

The subsequent `SelectAuthoritativeObservation()` call is strict about the selected inference-result ID, model, experiment, scope, checkpoint identity, and metric identity. Therefore the profitability observation is authoritative **for that inference row**, but the code has not proven that the chosen inference row is the one corresponding to the continuation source experiment's intended final inference context.

**Risk scenario:** a model has more than one completed final inference row due to re-inference over a different date range or context. The latest row wins, and Phase 2A diagnostics can report profitability from that row while the continuation analysis/source configuration refers to another inference context.

**Recommended correction:** bind the final lateral `inference_eval_result` query to the source experiment's exact inference identity, using the repository's established exact predicates where applicable, including at least:

- model ID;
- symbol;
- prediction horizon;
- threshold;
- `from_date` / `to_date` matching the source inference range;
- `inference_scope='final'`;
- `checkpoint_eval_id IS NULL`;
- completed status.

If exact binding yields zero or more than one candidate, profitability should remain unavailable with an explicit diagnostic rather than selecting the newest row.

### F2 — Behavior-neutral separation is well preserved

**Severity:** None / positive finding
**Status:** Pass

The new `ContinuationProfitabilityEvidence` is optional and carried on `ContinuationEvidence`. Core continuation functions remain profitability-free, including the source comparator, epoch deduplication, evidence watermark, source selection, ranking, trend evaluation, automatic queue ordering, checkpoint decision path, and recommendation scoring/ranking guards.

The implementation also refreshes diagnostic profitability after reloading a selected evidence point without putting profitability into the persisted decision identity or evidence watermark. This is the correct separation for Phase 2A.

### F3 — Missing evidence and zero-actionable evidence are correctly distinct

**Severity:** None / positive finding
**Status:** Pass

Unavailable profitability is represented by absence plus a reason such as `no_profitability_observation`, `ambiguous_profitability_observation`, `metric_definition_mismatch`, `provenance_mismatch`, `no_completed_inference_result`, or `profitability_schema_unavailable`.

A valid observation with `actionable_count=0` remains available, with zero aggregate/gross sums and a null per-actionable average. This avoids the dangerous equivalence of "missing" and numeric zero.

### F4 — Final/checkpoint observation scope separation is strong

**Severity:** None / positive finding
**Status:** Pass

`AuthoritativeObservationSelector` enforces final-vs-checkpoint shape. Checkpoint selection requires an experiment and checkpoint identity; final selection requires no checkpoint identity. Exact provenance and exact current metric definition are checked, and multiple matching immutable observations under the same provenance/metric identity are classified as ambiguous rather than resolved by recency.

This is suitable groundwork for later Phase 2B/2C use once F1 is corrected.

### F5 — `--continuation-status` is explicitly read-only

**Severity:** None / positive finding
**Status:** Pass

The packaged implementation uses `SetTransactionReadOnly(w)` for `RunContinuationStatusCommand()`. The policy-isolation test also guards this property and separately confirms the actual policy-control command remains write-capable.

This is a good operational hardening improvement for a diagnostic command.

### F6 — Static policy-isolation test is useful but not sufficient as sole proof

**Severity:** Low
**Status:** Acceptable with existing dynamic tests

`ContinuationProfitabilityPolicyIsolationTests.py` provides a useful regression tripwire by asserting that named policy/ranking/trend functions do not contain the word `profitability`. It should be treated as a supplement, not a semantic proof: it is textual, function-name based, and can miss indirection or future helper calls.

The accompanying dynamic continuation tests that compare selection/dedup/trend behavior with changed profitability evidence are therefore important and should remain.

### F7 — Verification package is not a complete line-level delta package

**Severity:** Low / verification limitation
**Status:** Packaging improvement recommended

The archive contains the relevant post-change files and implementation report, but it does **not** contain the proposed `git diff --binary HEAD` or another known pre-Phase-2A baseline. Therefore this review can validate the packaged code's internal architecture and identify defects, but cannot independently prove every line changed by Phase 2A versus the prior committed state.

For future independent reviews, include a patch/diff file plus `git status --short` and the base commit hash.

The archive also does not include `Sources/ExperimentRecommendationScoring.cpp`, although the packaged Python policy-isolation test references it. Thus that static test cannot be rerun standalone from only this archive; it requires the full repository.

### F8 — Minor documentation duplication

**Severity:** Trivial
**Status:** Optional cleanup

`docs/InferenceProfitabilityPersistence.rst` repeats the phrase/line introducing `ambiguous_profitability_observation` in the Phase 2A unavailable-reasons paragraph. This has no behavioral impact.

## Verification conclusion

I would accept the Phase 2A architecture and behavior-neutral boundary. The implementation successfully exposes immutable profitability evidence without making it decision-bearing.

Before beginning Phase 2B, make one targeted correction: **bind the final `inference_eval_result` selected for profitability diagnostics to the exact continuation source inference context instead of selecting the latest completed final inference row for the model.** Add a regression test with two completed final inference rows for the same model but different inference ranges/contexts and prove that only the exact source-context row is attached.

After that correction, apply migration 073 operationally, perform the clean Release build, and exercise read-only `--continuation-status` against real final and checkpoint evidence. Those operational steps would close the remaining verification gap before profitability is allowed to influence continuation policy.

## Recommended disposition

**Phase 2A:** Accept with targeted provenance correction.
**Phase 2B:** Do not activate yet; correct F1 and complete operational verification first.
