#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

// Pure Phase 4A Step 4 scoring policy. Canonical text is authoritative;
// the tagged 64-bit hash is only a lookup/diagnostic accelerator.
struct RecommendationScoringPolicy
{
    int scoringVersion = 1;
    double leaderScoreWeight = 0.25;
    double inferenceAccuracyWeight = 0.25;
    double evidenceStrengthWeight = 0.15;
    double neutralBalanceWeight = 0.10;
    double structuralDistanceWeight = 0.15;
    double parameterPreferenceWeight = 0.05;
    double sourceRankWeight = 0.05;
    double horizonChangePenaltyWeight = 0.05;
    double relativeMutationPenaltyWeight = 0.10;
    long long minimumEvidenceCount = 1;
    long long evidenceSaturationCount = 5000;
    double preferredNeutralProportion = 1.0 / 3.0;
    double maximumNeutralProportion = 0.80;
    double maximumRelativeMutation = 0.50;
    double maximumAbsoluteStructuralDistance = 1.0;
    bool allowMissingNeutralProportion = true;
    double scoreFloor = 0.0;
    double scoreCeiling = 1.0;
    double coreLrPreference = 1.0;
    double headLrPreference = 1.0;
    double labelThresholdPreference = 1.0;
    double predictionHorizonPreference = 1.0;
};

RecommendationScoringPolicy ParseRecommendationScoringPolicy(
    const std::string& text);
std::optional<std::string> ValidateRecommendationScoringPolicy(
    const RecommendationScoringPolicy& policy);
std::string RecommendationScoringPolicyCanonicalText(
    const RecommendationScoringPolicy& policy);
std::string RecommendationScoringPolicyHash(
    const RecommendationScoringPolicy& policy);

// Phase 3B semantic identity.  The policy canonical remains authoritative;
// this identity additionally freezes the compiled scoring algorithm contract
// whose behavior is not represented by configurable policy values alone.
struct RecommendationScoringSemanticIdentity
{
    std::string canonical;
    std::string hash;
    int version = 0;

    bool operator==(const RecommendationScoringSemanticIdentity&) const = default;
};

std::optional<std::string> ValidateRecommendationScoringPolicyProvenance(
    const std::string& canonical,
    const std::string& hash,
    int scoringVersion);
RecommendationScoringSemanticIdentity RecommendationScoringSemanticIdentityForPolicy(
    const RecommendationScoringPolicy& policy);
RecommendationScoringSemanticIdentity
RecommendationScoringSemanticIdentityFromPolicyProvenance(
    const std::string& scoringPolicyCanonical,
    const std::string& scoringPolicyHash,
    int scoringVersion);
std::optional<std::string> ValidateRecommendationScoringSemanticIdentity(
    const RecommendationScoringSemanticIdentity& identity,
    const std::string& scoringPolicyCanonical,
    const std::string& scoringPolicyHash,
    int scoringVersion);

struct RecommendationScoringInput
{
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    int sourcePredictionHorizon = 0;
    int sourceRankWithinGroup = 0;
    double sourceLeaderScore = 0.0;
    double sourceInferenceAccuracy = 0.0;
    std::optional<double> sourcePredictedNeutralProportion;
    long long sourceEvidenceCount = 0;
    std::string changedParameter;
    std::string sourceValueCanonical;
    std::string proposedValueCanonical;
    double absoluteDelta = 0.0;
    std::optional<double> relativeDelta;
    std::optional<int> horizonDelta;
    int generationOrdinal = 0;
    int structuralRank = 0;
    std::string semanticCanonicalText;
    std::string invocationCanonicalText;
    std::string recommendationPolicyCanonicalText;
    std::string duplicateType;
    std::string recommendationStatus;
};

struct RecommendationScoreComponent
{
    std::string componentName;
    std::string reasonCode;
    std::string inputCanonical;
    double normalizedValue = 0.0;
    double weight = 0.0;
    double weightedContribution = 0.0;
    bool penalty = false;
    std::string explanation;

    bool operator==(const RecommendationScoreComponent&) const = default;
};

struct RecommendationScoreResult
{
    bool valid = false;
    std::string reasonCode;
    std::string explanationSummary;
    std::string scoringPolicyCanonical;
    std::string scoringPolicyHash;
    int scoringVersion = 0;
    double rawPositiveScore = 0.0;
    double rawPenaltyScore = 0.0;
    double rawTotalScore = 0.0;
    double finalScore = 0.0;
    double structuralDistance = 0.0;
    std::vector<RecommendationScoreComponent> components;
};

RecommendationScoreResult ScoreExperimentRecommendation(
    const RecommendationScoringPolicy& policy,
    const RecommendationScoringInput& input);

// Step 2 persists absoluteDelta and, when the source value is nonzero,
// relativeDelta. It does not define a separate combined distance field.
// Step 4 derives this scoring-only distance from those exact metadata.
double DeriveRecommendationScoringDistance(
    const RecommendationScoringInput& input);

struct RankedRecommendationScore
{
    RecommendationScoringInput input;
    RecommendationScoreResult score;
    int scoreRank = 0;
    int tieGroup = 0;
    int rankingOrdinal = 0;
};

std::vector<RankedRecommendationScore> RankRecommendationScores(
    std::vector<RankedRecommendationScore> scores);

} // namespace EA::ExperimentRecommendation
