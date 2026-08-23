# LSTM Campaign Manager Phase 3C Profitability Distribution
## Independent Verification Findings

**Review artifact:** `LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_IndependentReview.tar.gz`
**Reviewed:** 2026-08-23
**Disposition:** **PASS WITH TARGETED CORRECTIONS**

## Executive assessment

The Phase 3C implementation successfully preserves the central safety invariant: profitability is advisory and non-decision-bearing. The new distribution layer is isolated from production scoring and ranking inputs, and the package shows compile-time constants fixing `profitability_weight` and `profitability_score_contribution` at zero. The implementation also fails closed on mixed profitability population identities and handles zero-actionable observations without fabricating a profitability value.

Two corrections are recommended before treating Phase 3C as independently verified and complete:

1. **Calendar-date validation is incomplete.** `IsoDate()` checks only YYYY-MM-DD shape, month 1-12, and day 1-31. It therefore accepts impossible dates such as `2025-02-31`, `2025-04-31`, and non-leap `2025-02-29`. This conflicts with the stated fail-closed behavior for malformed inference windows.
2. **The independent-review archive is not self-contained for its advertised tests.** `Tests/ProfitabilityDistributionTests.sh` requires `Sources/ExperimentRecommendation.cpp` and `Sources/InferenceProfitability.cpp` (and associated headers), but those files are absent from the archive. Consequently the packaged test suite cannot be independently compiled and executed from the supplied artifact alone.

Neither finding indicates that profitability can currently affect production decisions. The first is a validation defect in the advisory analysis layer; the second is a verification/package-completeness defect.

## Verified strengths

### 1. Profitability remains non-decision-bearing

`Sources/ProfitabilityDistribution.hpp` declares:

- `kPhase3CProfitabilityWeight = 0.0`
- `kPhase3CProfitabilityScoreContribution = 0.0`

Both are guarded by `static_assert`, and the normalization result exposes them as compile-time constants rather than configurable policy fields. The profitability distribution types are not present in the packaged production scoring input or ranking API. Searches of the packaged scoring/ranking sources show profitability provenance in evaluation, but no profitability score component in `ExperimentRecommendationScoring` or ranking logic.

The Phase 3C test source also contains before/after regression checks asserting that running profitability analysis does not alter recommendation scores, score ranking, Phase 3B bucket rank, or global ordinal.

**Finding:** PASS.

### 2. Population homogeneity is fail-closed

`ProfitabilityPopulationIdentity` binds the population to:

- metric definition canonical/hash;
- inference scope;
- exact inference start/end;
- symbol;
- prediction horizon;
- model input width;
- feature-class semantic identity;
- inference-evaluation semantic identity;
- scoring semantic identity;
- evaluation semantic identity.

`AnalyzeProfitabilityDistribution()` validates every observation before analysis and rejects the entire request when one member differs from the established population identity. It does not silently discard incompatible observations.

This is conservative and consistent with Phase 3B's semantic-homogeneity philosophy.

**Finding:** PASS, with a later-policy design question about whether scoring/evaluation semantic identities are scientifically necessary grouping dimensions for profitability distributions.

### 3. Membership and order determinism

Input observations are sorted by observation identity canonical text and observation ID before membership canonicalization. Duplicate observation IDs and duplicate observation identity canonicals are rejected. Distribution values are independently sorted before quantiles and statistics are computed.

Canonical policy text sorts requested quantiles, and numeric formatting delegates to the repository's canonical recommendation-double formatting. The supplied test source includes input-order and locale-invariance checks.

**Finding:** PASS by static inspection.

### 4. Zero-actionable handling is correct

A zero-actionable observation must have zero win/loss counts, zero gross/aggregate return, and no average return. It remains in population membership and total observation counts, but does not enter the profitability-value vector. Its normalized result has no raw profitability metric, percentile, or bounded candidate and receives zero support reliability.

This avoids conflating "no actionable predictions" with "zero profitability."

**Finding:** PASS.

### 5. Primary metric and normalization are reasonable advisory choices

The primary variable is average terminal-horizon log return per actionable prediction, avoiding the direct exposure-count scaling of aggregate return. The sign-preserving empirical midrank mapping places negative values below 0.5, zero at 0.5, and positive values above 0.5 while bounding outlier influence.

Support reliability is maintained separately as `actionable_count / (actionable_count + half_saturation)` and is not multiplied into a production recommendation score.

These are appropriate Phase 3C analytical candidates, provided the current defaults are not automatically promoted into future production policy.

**Finding:** PASS for advisory use.

## Required correction 1: real calendar-date validation

### Evidence

`Sources/ProfitabilityDistribution.cpp` implements `IsoDate()` by checking:

- string length 10;
- hyphens at positions 4 and 7;
- digits elsewhere;
- month between 1 and 12;
- day between 1 and 31.

It does not validate month-specific day counts or leap years.

Examples currently accepted by `IsoDate()` include:

- `2025-02-31`
- `2025-04-31`
- `2025-02-29`

A self-consistent population using one of these impossible dates can therefore pass the inference-window validation so long as the start string sorts before the end string.

### Severity

**Medium.** It does not affect production ranking today, but it violates the Phase 3C fail-closed provenance contract and could allow a malformed analytical population identity to be canonicalized and hashed.

### Recommended correction

Replace the current date-shape helper with deterministic Gregorian calendar validation. The implementation should validate year/month/day, month lengths, and leap-year rules without using local timezone or wall-clock state.

Add tests covering at least:

- valid `2024-02-29`;
- invalid `2025-02-29`;
- invalid `2025-02-30`;
- invalid `2025-04-31`;
- valid year boundary;
- start date equal to end date rejected;
- start date after end date rejected.

## Required correction 2: make the verification archive self-contained

### Evidence

The packaged `Tests/ProfitabilityDistributionTests.sh` compiles:

- `Sources/ExperimentRecommendation.cpp`
- `Sources/ExperimentRecommendationScoring.cpp`
- `Sources/ExperimentRecommendationEvaluation.cpp`
- `Sources/ExperimentRecommendationRanking.cpp`
- `Sources/InferenceProfitability.cpp`
- `Sources/ProfitabilityDistribution.cpp`
- `Tests/ProfitabilityDistributionTests.cpp`

The supplied review archive does **not** contain:

- `Sources/ExperimentRecommendation.cpp`
- `Sources/InferenceProfitability.cpp`
- the complete header set needed to build those files.

Executing the supplied test script from the extracted archive fails immediately with missing-file errors for `ExperimentRecommendation.cpp` and `InferenceProfitability.cpp`.

### Severity

**Medium for independent verification; no runtime product impact.** The implementation output reports successful tests in the original worktree, but those claims cannot be independently reproduced solely from this package.

### Recommended correction

Repackage Phase 3C with every source/header dependency required by `ProfitabilityDistributionTests.sh`, or provide a deliberately self-contained test harness whose dependencies are all included.

After repackaging, rerun:

1. `Tests/ProfitabilityDistributionTests.sh`
2. the relevant Phase 3A profitability regression tests;
3. ASan/UBSan Phase 3C tests if available;
4. `git diff --check` in the source worktree.

## Design observations that are not blockers

### Exact cohort identity may be over-constrained

Including Phase 3B scoring and evaluation semantic identities in the statistical profitability cohort is maximally safe, but those semantics answer whether recommendation evaluations can be ranked together, not necessarily whether two raw FINAL profitability measurements are scientifically comparable. Once the backfill is complete, measure cohort cardinality while progressively adding/removing these identity dimensions before deciding whether they belong in a future distribution-policy version.

### Minimum population size and support half-saturation are not production policy

The header defaults `minimumAnalyzablePopulationSize` to 5 and support half-saturation to 100 actionables. The tests commonly use a minimum of 3 for coverage. These values are suitable implementation defaults for an advisory mechanism but have not been empirically justified as production thresholds.

### Insufficient populations still expose candidate values

For a nonzero-actionable member in an insufficient population, the code computes and exposes an empirical percentile and bounded candidate but marks the state `insufficient_population`. This is acceptable for inspection because contribution remains zero; a future decision-bearing consumer must fail closed on state and must not use the candidate merely because the optional value exists.

### Observation identity binding is trusted at the projection boundary

The layer verifies that `observationIdentityHash` matches `observationIdentityCanonical`, but it does not independently parse that canonical identity and prove it binds every projected field. Likewise `sourceContentHash` is format-validated because no source-content canonical is carried. This is acceptable only if the adapter constructing `ProfitabilityObservation` is itself authoritative and validated. A future database/repository adapter should preserve that trust boundary explicitly.

## Independent test status

**Static review:** completed.
**Packaged Phase 3C test execution:** **BLOCKED by incomplete archive dependencies.**
**Production safety invariant:** verified by source inspection.
**Live database mutation performed by this review:** none.
**Scheduler/process changes performed by this review:** none.

## Final disposition

**PASS WITH TARGETED CORRECTIONS.**

The implementation's core architecture is sound and the central Phase 3C invariant is preserved: profitability cannot change current Campaign Manager scoring or ranking. Correct the Gregorian date validator and provide a self-contained independent-review package, then rerun the Phase 3C tests. After those corrections pass, this implementation is suitable to stage/commit as the non-decision-bearing profitability distribution layer.

Profitability should remain at weight/contribution zero until the backfill is complete and an empirical post-backfill study resolves cohort size, cohort identity, support stability, and normalization-policy questions.
