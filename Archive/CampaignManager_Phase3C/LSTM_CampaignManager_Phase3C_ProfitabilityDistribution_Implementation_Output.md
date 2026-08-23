---
title: "LSTM Campaign Manager Phase 3C Profitability Distribution Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Campaign Manager Phase 3C Profitability Distribution Implementation

Implemented Phase 3C as a deterministic, advisory-only profitability distribution layer. It cannot affect production scoring, ranking, eligibility, continuation, planning, queueing, or scheduler behavior.

### Changed files

- [ProfitabilityDistribution.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityDistribution.hpp:12>) — advisory data structures, policy, summaries, normalization results, and compile-time zero contribution constants.
- [ProfitabilityDistribution.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityDistribution.cpp:226>) — fail-closed validation, cohort identity, statistics, normalization, support reliability, and canonical output.
- [ProfitabilityDistributionTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ProfitabilityDistributionTests.cpp:201>) — deterministic coverage of all requested scenarios, including unchanged scoring and Phase 3B ranking.
- [ProfitabilityDistributionTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/ProfitabilityDistributionTests.sh:1>) — isolated `/tmp` test compilation.
- [CampaignManagerPhase3CProfitabilityDistribution.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/CampaignManagerPhase3CProfitabilityDistribution.rst:1>) — design, formulas, edge cases, safety boundary, and deferred work.
- [project.pbxproj](</Volumes/Developer SSD/ExpertAdvisor/ExpertAdvisor.xcodeproj/project.pbxproj:16>) — adds the new source to Debug and Release targets.

The pre-existing untracked Phase 3C inspection report was not modified.

### Behavioral change

The layer now:

- Validates authoritative profitability provenance, hashes, counts, numeric finiteness, aggregate/average consistency, and zero-actionable shape.
- Rejects an entire mixed or malformed requested population—nothing incompatible is silently dropped.
- Produces count, support, sign counts, min/max, mean, median, configurable quantiles, population standard deviation, and MAD.
- Produces deterministic membership and population canonical identities/hashes.
- Normalizes with a sign-preserving empirical midrank percentile:
  - negative values map below `0.5`;
  - zero maps to `0.5`;
  - positive values map above `0.5`.
- Reports actionable-count reliability separately as `n / (n + half_saturation)`.
- Explicitly represents invalid, insufficient, zero-actionable, and no-analyzable states.
- Emits `profitability_weight=0` and `profitability_score_contribution=0` in all canonical advisory output.

### Verification

Passed:

- `Tests/ProfitabilityDistributionTests.sh`
- `Tests/ExperimentRecommendationPhase3AProfitabilityTests.sh`
- `Tests/InferenceProfitabilityTests.sh`
- Phase 3C tests under AddressSanitizer and UndefinedBehaviorSanitizer.
- Isolated Debug build:
  `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Debug" -configuration Debug -derivedDataPath DerivedData/Phase3C-Verification-Debug build`

The isolated Debug build succeeded. Its only reported warning was an existing missing `LLVM22.xctoolchain/Info.plist` environment warning.

The isolated Release build was attempted at `DerivedData/Phase3C-Verification`, but the existing build-provenance script correctly refused a Release build from a dirty worktree. That authority check was not bypassed.

Scoring/ranking evidence:

- Tests compare recommendation scores and `RankRecommendationScores` before and after Phase 3C analysis.
- Tests also compare Phase 3B `RankRecommendationEvaluationEvidence` bucket ranks and global ordinals before and after analysis.
- Results are identical.
- Existing scoring policy canonical text and score components contain no profitability component.
- No existing production source includes the Phase 3C header.
- The only production-target integration is compilation of the standalone advisory implementation.

### Operational safety

- Database migration added: **No**
- Live database mutated: **No**
- Production database queried: **No**
- Scheduler stopped or restarted: **No**
- Experiments requeued, paused, cancelled, or altered: **No**
- Canonical executable replaced: **No**
- Migrations 073–077 modified: **No**
- Inference persistence paths modified: **No**

The scheduler retained its original PID and continued naturally into another backfill inference worker. The canonical executable remains timestamped `2026-08-22 20:22:47 -0500`; builds were written only to isolated paths.

### Phase 3C readiness assessment

A. Exact population identity: metric canonical/hash, inference scope, exact start/end window, exact symbol, prediction horizon, model input width, feature-class identity, inference-evaluation semantic identity, and Phase 3B scoring/evaluation semantic identities. Observation/source identities prove membership but are not grouping dimensions.

B. Primary variable: average terminal-horizon directional log return per actionable prediction. It removes first-order actionable-count scaling, unlike aggregate return, while remaining explicitly non-portfolio P&L.

C. Normalization grouping: exact symbol+horizon+window plus the semantic dimensions above. Global normalization is not currently justified. No unauthoritative symbol-family grouping is invented.

D. Actionable support: interpret separately as reliability, never profitability. The implemented half-saturation curve is descriptive and does not claim that overlapping windows are independent samples.

E. Recommended future candidate: sign-preserving empirical midrank percentile. It is deterministic, bounded, tie-stable, and outlier-resistant while preserving negative/zero/positive meaning.

F. Negative profitability: raw value remains visible and the bounded candidate is always below `0.5`.

G. All-equal/tiny populations: all-equal dispersion is zero; equal negative/zero/positive values map to `0.25/0.5/0.75`. Single and undersized populations remain explicitly `insufficient_population`, even though advisory transforms are exposed.

H. Can Phase 3C change ranking today? **No.** There is no scoring/ranking type connection, weight and contribution are compile-time zero, and before/after scoring and ranking regression tests pass.

I. Did this interfere with the backfill? **No.**

J. Readiness: the advisory statistical/provenance bridge is ready. Decision-bearing profitability remains blocked on backfill completion, cohort coverage, stability and effective-sample analysis, cost/overlap treatment, empirical transform comparison, versioned policy/schema design, regression evidence, and explicit approval.

### Repository status

`git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
?? LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_Inspection_Output.md
?? Sources/ProfitabilityDistribution.cpp
?? Sources/ProfitabilityDistribution.hpp
?? Tests/ProfitabilityDistributionTests.cpp
?? Tests/ProfitabilityDistributionTests.sh
?? docs/CampaignManagerPhase3CProfitabilityDistribution.rst
```

`git diff --stat`:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj | 8 ++++++++
 1 file changed, 8 insertions(+)
```

The stat excludes the new untracked files listed above.