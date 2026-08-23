---
title: "LSTM Campaign Manager Phase 3C Targeted Corrections"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignManager_Phase3C_TargetedCorrections_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Campaign Manager Phase 3C Targeted Corrections

Implemented both targeted corrections. No production workflow or decision policy changed.

### Changes

- [ProfitabilityDistribution.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityDistribution.cpp:35>)
  - Enforces exactly `YYYY-MM-DD` using ASCII digit parsing.
  - Rejects year `0000`, months outside `01–12`, day `00`, and month-overflow days.
  - Implements Gregorian leap years: divisible by 4, except 100, except 400.
  - Uses no locale, timezone, clock, or date-library state.
- [ProfitabilityDistributionTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ProfitabilityDistributionTests.cpp:224>)
  - Added all requested date cases, plus year zero, invalid end date, equal window, and decreasing window.
- [CampaignManagerPhase3CProfitabilityDistribution.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/CampaignManagerPhase3CProfitabilityDistribution.rst:46>)
  - Documents strict real-Gregorian validation.
- [package_phase3c_profitability_distribution_review.sh](</Volumes/Developer SSD/ExpertAdvisor/package_phase3c_profitability_distribution_review.sh:1>)
  - Includes the compiler-derived repository-local dependency closure.
  - Fails closed for absent inputs.
  - Adds payload checksums and normalized metadata for reproducible output.
  - Automatically verifies checksums and runs the test after clean extraction.
- Regenerated [independent-review archive](</Volumes/Developer SSD/ExpertAdvisor/LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_IndependentReview.tar.gz>).

Tests cover:

- `2025-02-28` accepted
- `2025-02-29` rejected
- `2024-02-29` accepted
- `2100-02-29` rejected
- `2000-02-29` accepted
- April 31, months 00/13, day 00, year 0000 rejected
- Valid ordinary dates accepted
- Missing/wrong separators, incorrect widths, suffixes, and non-digits rejected
- Invalid end dates and non-increasing windows rejected

### Verification

- `Tests/ProfitabilityDistributionTests.sh` — PASS
- `Tests/ExperimentRecommendationPhase3AProfitabilityTests.sh` — PASS
- `Tests/InferenceProfitabilityTests.sh` — PASS
- `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Debug" -configuration Debug -derivedDataPath DerivedData/Phase3C-Verification-Debug build` — PASS
- Isolated Release build — correctly refused by `GenerateBuildProvenance.py` because the worktree is dirty. The guard was not bypassed.
- `git diff --check` — PASS
- Packaging script executed twice — PASS; identical SHA-256 both times.
- Scheduler/process status was checked first. Active training/inference workers were not interrupted.
- Canonical Release executable remained unchanged:
  `5d4974da55ee4de70bfa09d28903295e727925882d0fad3bdc5ed543cdf04218`

Clean-extraction command performed by the package script:

```bash
tar -xzf LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_IndependentReview.tar.gz -C "${verification_dir}"
(
  cd "${verification_dir}"
  shasum -a 256 -c SHA256SUMS.txt
  env -i PATH="${PATH}" TMPDIR="${TMPDIR:-/tmp}" \
    bash Tests/ProfitabilityDistributionTests.sh
)
```

Result: every packaged checksum reported `OK`; test exited successfully. Reproducibility check produced the same archive hash twice.

### Archive

Name:

```text
LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_IndependentReview.tar.gz
```

SHA-256:

```text
8fdaf4b3b2c152daf49a390632a59bb485614af20d86c179398735b23fb6ffab
```

Exact contents:

```text
ExpertAdvisor.xcodeproj/project.pbxproj
Headers/CanonicalSymbol.hpp
Headers/Donchian20Mode.hpp
Headers/DonchianLookback.hpp
Headers/FeatureWarmupScope.hpp
Sources/ExperimentRecommendation.cpp
Sources/ExperimentRecommendation.hpp
Sources/ExperimentRecommendationScoring.cpp
Sources/ExperimentRecommendationScoring.hpp
Sources/ExperimentRecommendationEvaluation.cpp
Sources/ExperimentRecommendationEvaluation.hpp
Sources/ExperimentRecommendationRanking.cpp
Sources/ExperimentRecommendationRanking.hpp
Sources/InferenceProfitability.cpp
Sources/InferenceProfitability.hpp
Sources/ProfitabilityDistribution.cpp
Sources/ProfitabilityDistribution.hpp
Tests/ProfitabilityDistributionTests.cpp
Tests/ProfitabilityDistributionTests.sh
docs/CampaignManagerPhase3CProfitabilityDistribution.rst
LSTM_CampaignManager_Phase3C_Independent_Verification_Findings.md
LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_Implementation_Output.md
LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_Inspection_Output.md
package_phase3c_profitability_distribution_review.sh
Database/migrations/073_inference_profitability_observation.sql
Database/migrations/074_continuation_profitability_policy.sql
Database/migrations/076_campaign_manager_final_profitability_provenance.sql
Database/migrations/077_campaign_manager_ranking_semantic_homogeneity.sql
SHA256SUMS.txt
```

### Safety confirmation

Source inspection found Phase 3C distribution references only in its implementation, tests, and Xcode build registration. No Phase 3C distribution value is consumed by scoring, ranking, eligibility, planning, scheduler policy, or experiment control. Both compile-time contribution constants remain zero.

No database state, migration, scheduler, worker, inference/backfill path, experiment, thresholds, normalization semantics, or Campaign Manager decision behavior was changed.

### Git state

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
?? LSTM_CampaignManager_Phase3C_Independent_Verification_Findings.md
?? LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_Implementation_Output.md
?? LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_IndependentReview.tar.gz
?? LSTM_CampaignManager_Phase3C_ProfitabilityDistribution_Inspection_Output.md
?? Sources/ProfitabilityDistribution.cpp
?? Sources/ProfitabilityDistribution.hpp
?? Tests/ProfitabilityDistributionTests.cpp
?? Tests/ProfitabilityDistributionTests.sh
?? docs/CampaignManagerPhase3CProfitabilityDistribution.rst
?? package_phase3c_profitability_distribution_review.sh
```

```text
 ExpertAdvisor.xcodeproj/project.pbxproj | 8 ++++++++
 1 file changed, 8 insertions(+)
```

The status contains the same scoped Phase 3C files present at the start; no unrelated file appeared. Because most Phase 3C files remain untracked, normal `git diff --stat` only reports the already-present project-file change. No commit was created.

Remaining concern: a Release build cannot be completed until the worktree is clean, by design. The isolated Debug build and all focused tests passed.