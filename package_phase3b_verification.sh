#!/usr/bin/env bash
set -euo pipefail

PACKAGE="LSTM_CampaignManager_Phase3B_ScoringPolicy_RankingHomogeneity_Verification"
STAGE="/tmp/${PACKAGE}"
ARCHIVE="${PACKAGE}.tar.gz"

rm -rf "$STAGE"
rm -f "$ARCHIVE"
mkdir -p "$STAGE/repository"

FILES=(
  # Production: scoring
  "Sources/ExperimentRecommendationScoring.hpp"
  "Sources/ExperimentRecommendationScoring.cpp"

  # Production: evaluation
  "Sources/ExperimentRecommendationEvaluation.hpp"
  "Sources/ExperimentRecommendationEvaluation.cpp"
  "Sources/ExperimentRecommendationEvaluationRepository.hpp"
  "Sources/ExperimentRecommendationEvaluationRepository.cpp"
  "Sources/ExperimentRecommendationEvaluationService.cpp"

  # Production: ranking
  "Sources/ExperimentRecommendationRanking.hpp"
  "Sources/ExperimentRecommendationRanking.cpp"
  "Sources/ExperimentRecommendationRankingRepository.hpp"
  "Sources/ExperimentRecommendationRankingRepository.cpp"
  "Sources/ExperimentRecommendationRankingService.cpp"

  # Production: downstream campaign-planning trust boundary
  "Sources/ExperimentRecommendationCampaignPlanningRepository.cpp"

  # Migration
  "Database/migrations/077_campaign_manager_ranking_semantic_homogeneity.sql"

  # Modified tests
  "Tests/ExperimentRecommendationCampaignPlanningRepositoryTests.cpp"
  "Tests/ExperimentRecommendationEvaluationRepositoryTests.cpp"
  "Tests/ExperimentRecommendationRankingRepositoryTests.cpp"
  "Tests/ExperimentRecommendationRankingTests.cpp"

  # New migration test
  "Tests/CampaignManagerPhase3BRankingSemanticMigrationTests.sql"

  # Phase 3B documentation
  "docs/CampaignManagerPhase3BRankingSemanticHomogeneity.rst"
)

echo "Verifying expected files exist..."

for file in "${FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: expected Phase 3B file is missing: $file" >&2
        exit 1
    fi
done

echo "Copying Phase 3B files..."

for file in "${FILES[@]}"; do
    mkdir -p "$STAGE/repository/$(dirname "$file")"
    cp -p "$file" "$STAGE/repository/$file"
done

# ----------------------------------------------------------------------
# Review metadata
# ----------------------------------------------------------------------

git rev-parse HEAD > "$STAGE/BASE_COMMIT.txt"
git branch --show-current > "$STAGE/BRANCH.txt"
git status --short > "$STAGE/GIT_STATUS.txt"
git diff --stat > "$STAGE/GIT_DIFF_STAT.txt"
git diff --check > "$STAGE/GIT_DIFF_CHECK.txt"

# Full tracked Phase 3B diff.
git diff -- \
  Sources/ExperimentRecommendationScoring.hpp \
  Sources/ExperimentRecommendationScoring.cpp \
  Sources/ExperimentRecommendationEvaluation.hpp \
  Sources/ExperimentRecommendationEvaluation.cpp \
  Sources/ExperimentRecommendationEvaluationRepository.hpp \
  Sources/ExperimentRecommendationEvaluationRepository.cpp \
  Sources/ExperimentRecommendationEvaluationService.cpp \
  Sources/ExperimentRecommendationRanking.hpp \
  Sources/ExperimentRecommendationRanking.cpp \
  Sources/ExperimentRecommendationRankingRepository.hpp \
  Sources/ExperimentRecommendationRankingRepository.cpp \
  Sources/ExperimentRecommendationRankingService.cpp \
  Sources/ExperimentRecommendationCampaignPlanningRepository.cpp \
  Tests/ExperimentRecommendationCampaignPlanningRepositoryTests.cpp \
  Tests/ExperimentRecommendationEvaluationRepositoryTests.cpp \
  Tests/ExperimentRecommendationRankingRepositoryTests.cpp \
  Tests/ExperimentRecommendationRankingTests.cpp \
  > "$STAGE/PHASE3B_TRACKED.diff"

# Explicit manifest.
{
    echo "Phase 3B Scoring Policy Ranking Homogeneity Verification Package"
    echo
    echo "Base commit:"
    git rev-parse HEAD
    echo
    echo "Branch:"
    git branch --show-current
    echo
    echo "Packaged files:"
    printf '%s\n' "${FILES[@]}"
    echo
    echo "Intentionally excluded:"
    echo "LSTM_CampaignManager_Phase3B_ScoringPolicy_RankingHomogeneity_Inspection_Output.md"
    echo "  Reason: pre-existing inspection artifact; implementation report states it was untouched."
} > "$STAGE/MANIFEST.txt"

# File hashes for exact-content verification.
(
    cd "$STAGE/repository"
    find . -type f -print0 |
      sort -z |
      xargs -0 shasum -a 256
) > "$STAGE/SHA256SUMS.txt"

# ----------------------------------------------------------------------
# Sanity check current worktree against expected Phase 3B implementation
# ----------------------------------------------------------------------

{
    echo "=== Current git status ==="
    git status --short
    echo
    echo "=== Expected tracked Phase 3B changes ==="
    printf 'M  %s\n' \
      "Sources/ExperimentRecommendationCampaignPlanningRepository.cpp" \
      "Sources/ExperimentRecommendationEvaluation.cpp" \
      "Sources/ExperimentRecommendationEvaluation.hpp" \
      "Sources/ExperimentRecommendationEvaluationRepository.cpp" \
      "Sources/ExperimentRecommendationEvaluationRepository.hpp" \
      "Sources/ExperimentRecommendationEvaluationService.cpp" \
      "Sources/ExperimentRecommendationRanking.cpp" \
      "Sources/ExperimentRecommendationRanking.hpp" \
      "Sources/ExperimentRecommendationRankingRepository.cpp" \
      "Sources/ExperimentRecommendationRankingRepository.hpp" \
      "Sources/ExperimentRecommendationRankingService.cpp" \
      "Sources/ExperimentRecommendationScoring.cpp" \
      "Sources/ExperimentRecommendationScoring.hpp" \
      "Tests/ExperimentRecommendationCampaignPlanningRepositoryTests.cpp" \
      "Tests/ExperimentRecommendationEvaluationRepositoryTests.cpp" \
      "Tests/ExperimentRecommendationRankingRepositoryTests.cpp" \
      "Tests/ExperimentRecommendationRankingTests.cpp"
    echo
    echo "=== Expected new Phase 3B files ==="
    echo "Database/migrations/077_campaign_manager_ranking_semantic_homogeneity.sql"
    echo "Tests/CampaignManagerPhase3BRankingSemanticMigrationTests.sql"
    echo "docs/CampaignManagerPhase3BRankingSemanticHomogeneity.rst"
} > "$STAGE/WORKTREE_EXPECTATIONS.txt"

tar -C /tmp -czf "$ARCHIVE" "$PACKAGE"

echo
echo "Created:"
ls -lh "$ARCHIVE"

echo
echo "Archive contents:"
tar -tzf "$ARCHIVE"

echo
echo "Archive SHA-256:"
shasum -a 256 "$ARCHIVE"
