# LSTM Profitability Phase 2B — Independent Diff Verification Findings

**Verification target:** Profitability Phase 2B Scheduler Continuation Policy Implementation
**Baseline:** `ad1112b0d3a5ff635ed1a750b881e33159720377` (`Expose profitability as continuation evidence`)
**Result:** **PASS — no targeted code correction required before commit**
**Scope:** Independent review of the packaged tracked diff plus direct review of the packaged new migration/test files.

## Executive conclusion

The Phase 2B implementation satisfies the intended architecture: profitability is an explicit, default-disabled **scheduler continuation gate** applied to the exact continuation source selected by the existing source-mode machinery.

The review did not find a path by which profitability changes:

- source selection,
- ranking or tie-breaking,
- leader-score or inference-accuracy trend logic,
- evidence count,
- checkpoint stop/continue policy,
- recommendation/Campaign Manager scoring,
- or training.

When profitability policy is disabled, the new fields are omitted from the continuation policy canonical identity, and profitability remains excluded from the legacy continuation evidence watermark. This preserves the pre-Phase-2B semantic policy hash and decision behavior.

When one or more profitability gates are enabled, the configured thresholds enter semantic policy identity and the exact selected profitability observation enters the decision watermark. This is the correct transition from Phase 2A diagnostic evidence to Phase 2B decision-bearing evidence.

## Verified policy surface

The implementation adds three optional policy primitives:

- `min_profitability_actionable_count`
- `min_profitability_aggregate_terminal_horizon_log_return_sum`
- `min_profitability_average_terminal_horizon_log_return_per_actionable_prediction`

Configured requirements use AND semantics.

No composite profitability score, ranking weight, cross-symbol normalization, cross-horizon normalization, transaction-cost model, or profitability trend logic was introduced.

The underlying metrics remain terminal-horizon directional log-return inference statistics and are not represented as portfolio P&L.

## Default-disabled compatibility

`ContinuationProfitabilityPolicyConfigured()` returns false when all three fields are unset.

In that state:

- `EvaluateContinuationProfitabilityGate()` passes with `profitability_policy_disabled`;
- profitability fields are not appended to `ContinuationPolicySemanticCanonicalText()`;
- the previous semantic policy hash is therefore preserved;
- `ContinuationEvidenceWatermark()` remains unchanged and profitability-free;
- profitability does not enter source selection, ranking, trend, deduplication, or evidence count.

The targeted unit tests explicitly compare the legacy canonical text/hash before and after profitability observations are present with the policy disabled.

**Finding: PASS.**

## Exact source semantics

The continuation evaluator first loads evidence and selects the source using the existing `SelectContinuationSourceEvidence()` logic.

Only after selection does it call `EvaluateContinuationProfitabilityGate(config, evaluation.selected)`.

The profitability gate therefore evaluates:

- the exact final source for `final_model`;
- the exact selected best checkpoint for `best_checkpoint`;
- the exact selected latest checkpoint for `latest_checkpoint`.

The source-selection functions themselves do not reference profitability.

The existing Phase 2A exact-final-inference resolver remains in the evidence loader, and checkpoint evidence retains exact checkpoint provenance.

**Finding: PASS.**

## Missing, ambiguous, and zero-actionable evidence

When profitability is disabled, missing profitability evidence has no effect.

When profitability is configured and the selected source has no authoritative profitability observation, the gate fails closed with `profitability_evidence_unavailable`.

Unavailable evidence is not converted to zero.

A valid zero-actionable observation remains distinct from missing evidence:

- aggregate return remains a defined zero;
- average return remains undefined;
- a positive actionable-count requirement fails;
- an average-return requirement fails as undefined.

**Finding: PASS.**

## Evaluator ordering

The authoritative `EvaluateContinuationPolicy()` flow is:

1. load policy;
2. perform existing enablement/configuration/source readiness checks;
3. load/deduplicate existing continuation evidence;
4. select the continuation source using existing logic;
5. attach/evaluate profitability for that selected source;
6. conditionally extend the decision watermark when profitability is configured;
7. perform existing resumability/ranking setup;
8. process persisted/reused decision identity;
9. apply existing evidence/trend/leader/inference gates;
10. apply profitability gate;
11. apply top-N and trend rejection handling;
12. persist the resulting decision as before.

A failed profitability gate cannot fall through to an eligible decision.

If multiple old and new gates fail simultaneously, the evaluator may report `rejected_profitability` before a later rank/trend rejection. This is a rejection-reason precedence choice for profitability-enabled policies, not an eligibility bypass.

**Finding: PASS.**

## Ranking and trend isolation

Independent inspection confirmed that profitability is absent from:

- `BetterBestContinuationSource`
- `SelectContinuationSourceEvidence`
- `PreferContinuationEvidenceAtSameEpoch`
- `DeduplicateContinuationEvidence`
- `ContinuationEvidenceWatermark`
- `EvaluateContinuationTrend`
- `BetterContinuationRankCandidate`
- `RankContinuationSource`
- `BetterContinuationAutoQueueCandidate`
- `DecideCheckpointPolicy`

Thus profitability can filter an otherwise selected continuation candidate but does not re-order surviving candidates or alter the existing trend metric.

**Finding: PASS.**

## Policy identity and decision watermark

When profitability is configured, the exact configured gate values are appended to `ContinuationPolicySemanticCanonicalText()`.

Therefore different decision-bearing profitability thresholds produce different semantic policy hashes.

When profitability is not configured, those fields are omitted rather than serialized as NULL. This preserves the pre-Phase-2B canonical text/hash.

When configured, `ContinuationProfitabilityEvidenceIdentity()` hashes availability/provenance and, for available evidence, includes:

- observation ID and immutable observation identity hash;
- exact inference result;
- scope;
- checkpoint identity;
- metric-definition hash;
- source-content hash;
- prediction count;
- actionable count;
- aggregate return;
- average return.

That profitability identity is then folded into the continuation decision watermark.

If evidence was previously unavailable and later becomes available, the watermark changes and the authoritative evaluator will not silently reuse the old decision identity.

**Finding: PASS.**

## Automatic preflight safety

The compact automatic-continuation preflight predates decision-bearing profitability.

The implementation explicitly forces full authoritative evaluation whenever profitability policy is configured, using `profitability_requires_full_evaluation`.

This avoids incorrectly treating a previously queued/satisfied continuation as current using a preflight identity that does not itself carry profitability primitives.

**Finding: PASS.**

## Inheritance and persistence

Configured profitability policy values are:

- loaded from `experiment`;
- parsed and validated by the existing continuation-policy update path;
- persisted by the policy-control command;
- copied into inherited continuation child policies;
- included in the source policy hash used for inheritance provenance.

Unset profitability fields remain unset through inheritance.

**Finding: PASS.**

## Migration 074

The packaged migration adds nullable/default-disabled columns:

- `continuation_policy_min_profit_actionable_count`
- `continuation_policy_min_profit_aggregate_log_return_sum`
- `continuation_policy_min_profit_average_log_return`

It enforces:

- positive actionable-count thresholds;
- finite aggregate/average thresholds;
- `rejected_profitability` as an allowed experiment policy decision;
- `rejected_profitability` as an allowed durable continuation-decision value.

The migration test verifies defaults, invalid values, valid round-trip persistence, and the new decision value.

### Operational migration caution

Migration 074 is **not completely non-blocking**. It alters the populated `experiment` table and drops/re-adds CHECK constraints on both `experiment` and `experiment_continuation_decision`. Re-adding CHECK constraints can validate existing rows and requires DDL locks.

This is not a code defect, but operational application should be treated similarly to the earlier migration caution: apply it during a quiet database window or after verifying active transactions/locks.

The new Phase 2B binary also intentionally treats the continuation schema as unavailable until the three migration-074 columns exist. Therefore deployment order should be:

1. commit/build;
2. apply migration 074;
3. use the new binary for continuation evaluation/status.

**Finding: PASS with deployment-order caution.**

## Profitability-only policy behavior

The existing continuation configuration contract still requires at least one pre-existing threshold/ranking/trend criterion (`min_leader_score`, `min_infer_accuracy`, `top_n`, or trend mode). Migration 074 does not weaken the older database enablement constraint either.

Therefore profitability is genuinely **additive**: it cannot become the sole continuation-selection criterion.

This is consistent with the Phase 2B requirement that profitability not replace or weaken existing continuation requirements. It should be kept documented because an operator cannot enable a policy consisting only of profitability gates.

**Finding: intentional behavior, not a defect.**

## Checkpoint-policy and Campaign Manager boundaries

The tracked diff contains no decision-bearing profitability integration in checkpoint stop/continue logic.

The recommendation-scoring source file was not included in the verification bundle, so the packaged static isolation script cannot be independently re-run end-to-end from the archive alone. However:

- the exact tracked diff does not modify recommendation/Campaign Manager scoring files;
- the implementation report records a passing recommendation-scoring regression;
- no Campaign Manager source appears among the changed files.

**Finding: PASS, with package-level test reproducibility limitation.**

## Verification-package limitation

The exact tracked `git diff --binary HEAD` is present and correctly pins baseline `ad1112b`.

However, Git's normal diff does not contain untracked files. The new migration and Phase 2B test files were therefore not present in the diff itself. They were included separately in the tarball and were reviewed directly.

For a future verification package, staging new files before generating `git diff --cached --binary` would produce a single authoritative patch containing both tracked modifications and new files.

This packaging limitation does not indicate an implementation defect.

## Test evidence reviewed

The implementation report records passing:

- `ContinuationProfitabilityPolicyTests`
- `ContinuationProfitabilityPolicyMigrationTests`
- `ContinuationPolicyInheritanceTests`
- `ContinuationProfitabilityPolicyIsolationTests.py`
- `InferenceProfitabilityTests`
- `InferenceProfitabilityRepositoryTests`
- `ExperimentRecommendationScoringTests`
- continuation persistence compilation
- `git diff --check`
- complete Debug LSTM Release target build

Live scheduler integration and operational persistence tests were intentionally not run because active scheduler/workers were present and migration 074 was not operationally applied.

The clean canonical Release build remains a post-commit verification step because repository provenance requires a clean tree.

## Final assessment

### Code/design

**PASS**

No targeted correction is required before staging and committing Phase 2B.

### Still required before declaring Phase 2B operationally complete

1. Archive/stage/commit the Phase 2B implementation and review artifacts.
2. Run the canonical Release build from the clean committed worktree.
3. Apply migration 074 during an appropriate DB window.
4. Run read-only `--continuation-status` verification using the new schema.
5. Preferably exercise at least one explicit profitability gate in a safe controlled/read-only evaluation context before moving to Phase 2C.

### Recommended status

```text
Phase 2B architecture/design:              PASS
Default-disabled compatibility:            PASS
Source/provenance semantics:               PASS
Policy identity:                           PASS
Decision watermark:                        PASS
Ranking/trend isolation:                    PASS
Inheritance/persistence:                    PASS
Migration design:                           PASS
Targeted/unit validation reported:          PASS
Independent diff review:                    PASS
Targeted correction required:               NO
Clean Release build:                        PENDING
Migration 074 operationally applied:        PENDING
Live/read-only operational verification:    PENDING
```
