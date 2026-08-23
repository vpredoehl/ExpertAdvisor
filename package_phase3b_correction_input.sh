#!/usr/bin/env bash
set -euo pipefail

PACKAGE="LSTM_CampaignManager_Phase3B_SQLValidator_Correction_Input"
STAGE="/tmp/$PACKAGE"
ARCHIVE="${PACKAGE}.tar.gz"

rm -rf "$STAGE"
rm -f "$ARCHIVE"

mkdir -p "$STAGE/repository"

FILES=(
  "Database/migrations/077_campaign_manager_ranking_semantic_homogeneity.sql"
  "Tests/CampaignManagerPhase3BRankingSemanticMigrationTests.sql"
  "Sources/ExperimentRecommendationScoring.hpp"
  "Sources/ExperimentRecommendationScoring.cpp"
  "Sources/ExperimentRecommendationEvaluation.hpp"
  "Sources/ExperimentRecommendationEvaluation.cpp"
  "Sources/ExperimentRecommendationEvaluationRepository.hpp"
  "Sources/ExperimentRecommendationEvaluationRepository.cpp"
)

echo "Verifying required files..."

for file in "${FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: missing required file: $file" >&2
        exit 1
    fi
done

echo "Copying correction-input files..."

for file in "${FILES[@]}"; do
    mkdir -p "$STAGE/repository/$(dirname "$file")"
    cp -p "$file" "$STAGE/repository/$file"
done

# ------------------------------------------------------------
# Repository state
# ------------------------------------------------------------

git rev-parse HEAD > "$STAGE/BASE_COMMIT.txt"
git branch --show-current > "$STAGE/BRANCH.txt"
git status --short > "$STAGE/GIT_STATUS.txt"
git diff --check > "$STAGE/GIT_DIFF_CHECK.txt"

# Include the current Phase 3B diff for these files so the
# correction can be distinguished from the pre-Phase-3B base.
git diff -- \
  Database/migrations/077_campaign_manager_ranking_semantic_homogeneity.sql \
  Tests/CampaignManagerPhase3BRankingSemanticMigrationTests.sql \
  Sources/ExperimentRecommendationScoring.hpp \
  Sources/ExperimentRecommendationScoring.cpp \
  Sources/ExperimentRecommendationEvaluation.hpp \
  Sources/ExperimentRecommendationEvaluation.cpp \
  Sources/ExperimentRecommendationEvaluationRepository.hpp \
  Sources/ExperimentRecommendationEvaluationRepository.cpp \
  > "$STAGE/CURRENT_RELEVANT_DIFF.diff"

# ------------------------------------------------------------
# Include earlier migrations that may establish subordinate
# canonical/hash invariants.  These are reference-only.
# ------------------------------------------------------------

mkdir -p "$STAGE/reference_migrations"

for file in \
    Database/migrations/034*.sql \
    Database/migrations/076_campaign_manager_final_profitability_provenance.sql
do
    if [[ -f "$file" ]]; then
        cp -p "$file" "$STAGE/reference_migrations/"
    fi
done

# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

{
    echo "Phase 3B SQL Validator Narrow Correction Input Package"
    echo
    echo "Purpose:"
    echo "1. Make migration 077 SQL scoring-policy provenance validation"
    echo "   enforce the same accepted semantic/canonical domain as C++."
    echo "2. Add adversarial migration tests for correctly hashed but"
    echo "   semantically invalid or noncanonical scoring policies."
    echo "3. Verify subordinate evaluation-result canonical/hash"
    echo "   coherence at the database boundary."
    echo "4. Keep profitability completely non-decision-bearing."
    echo "5. Make no ranking/scoring behavior changes."
    echo
    echo "Primary files:"
    printf '%s\n' "${FILES[@]}"
    echo
    echo "Reference migrations:"
    echo "Database/migrations/034*.sql if present"
    echo "Database/migrations/076_campaign_manager_final_profitability_provenance.sql"
    echo
    echo "Expected correction scope:"
    echo "Database/migrations/077_campaign_manager_ranking_semantic_homogeneity.sql"
    echo "Tests/CampaignManagerPhase3BRankingSemanticMigrationTests.sql"
    echo
    echo "C++ files are supplied as authoritative semantic references;"
    echo "they should not require modification unless verification proves"
    echo "a genuine application/database contract mismatch."
} > "$STAGE/MANIFEST.txt"

# ------------------------------------------------------------
# Hash everything supplied
# ------------------------------------------------------------

(
    cd "$STAGE"
    find repository reference_migrations \
      -type f \
      -print0 |
      sort -z |
      xargs -0 shasum -a 256
) > "$STAGE/SHA256SUMS.txt"

# ------------------------------------------------------------
# Build archive
# ------------------------------------------------------------

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
