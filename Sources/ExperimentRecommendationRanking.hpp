#pragma once

#include "ExperimentRecommendationEvaluation.hpp"

#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int kMaximumRecommendationRankingInputs = 1000;
inline constexpr int kMaximumRecommendationRankingMembers = 1000;
inline constexpr int kMaximumRecommendationRankingListLimit = 1000;
inline constexpr int kMaximumRecommendationComparisonComponents = 100;

struct RecommendationRankingPolicy
{
    int rankingVersion = 1;
};

std::optional<std::string> ValidateRecommendationRankingPolicy(
    const RecommendationRankingPolicy& policy);
std::string RecommendationRankingPolicyCanonicalText(
    const RecommendationRankingPolicy& policy);
std::string RecommendationRankingCanonicalHash(const std::string& canonical);

enum class RecommendationRankingScopeType
{
    evaluationRun,
    recommendationScan,
    symbol,
    horizon,
    family,
    symbolHorizon,
    global
};

struct RecommendationRankingScope
{
    RecommendationRankingScopeType type =
        RecommendationRankingScopeType::evaluationRun;
    std::optional<long long> evaluationRunId;
    std::optional<long long> recommendationScanId;
    std::optional<std::string> symbol;
    std::optional<int> horizon;
    std::optional<std::string> family;
};

std::string RecommendationRankingScopeTypeText(
    RecommendationRankingScopeType value);
std::optional<std::string> ValidateRecommendationRankingScope(
    const RecommendationRankingScope& scope);
std::string RecommendationRankingScopeCanonicalText(
    const RecommendationRankingScope& scope);
std::string RecommendationRankingScopeValueText(
    const RecommendationRankingScope& scope);

enum class RecommendationRankingBucket
{
    advisoryReady,
    blocked,
    nonActionable
};

std::string RecommendationRankingBucketText(RecommendationRankingBucket value);
std::optional<RecommendationRankingBucket> ParseRecommendationRankingBucket(
    const std::string& value);

struct RecommendationRankingEvaluation
{
    long long evaluationResultId = -1;
    long long evaluationRunId = -1;
    long long recommendationId = -1;
    long long recommendationScanId = -1;
    long long sourceExperimentId = -1;
    std::optional<long long> sourceModelId;
    std::optional<long long> sourceAnalysisId;
    std::string symbol;
    int horizon = 0;
    std::string family;
    std::string sourceValueCanonical;
    std::string proposedValueCanonical;
    std::string recommendationSemanticHash;
    std::string evaluationIdentityCanonical;
    std::string evaluationIdentityHash;
    std::string evaluationPolicyCanonical;
    std::string evaluationPolicyHash;
    int evaluationVersion = 0;
    int evaluatorVersion = 0;
    std::string scoringPolicyCanonical;
    std::string scoringPolicyHash;
    int scoringVersion = 0;
    RecommendationScoringSemanticIdentity scoringSemanticIdentity;
    RecommendationEvaluationSemanticIdentity evaluationSemanticIdentity;
    RecommendationEligibility eligibility = RecommendationEligibility::ineligible;
    RecommendationEvaluationDisposition disposition =
        RecommendationEvaluationDisposition::invalidPersistedEvidence;
    std::string reasonCode;
    std::string explanation;
    std::optional<double> finalScore;
    int componentCount = 0;
    int missingEvidenceCount = 0;
    std::vector<RecommendationScoreComponent> components;
};

enum class RecommendationRankingPopulationSemanticState
{
    verifiedHomogeneous,
    empty,
    legacyHeterogeneous,
    legacyUnverified
};

std::string RecommendationRankingPopulationSemanticStateText(
    RecommendationRankingPopulationSemanticState value);

struct RecommendationRankingPopulationSemanticValidation
{
    RecommendationRankingPopulationSemanticState state =
        RecommendationRankingPopulationSemanticState::legacyUnverified;
    std::optional<RecommendationScoringSemanticIdentity> scoringIdentity;
    std::optional<RecommendationEvaluationSemanticIdentity> evaluationIdentity;
    int distinctScoringIdentityCount = 0;
    int distinctEvaluationIdentityCount = 0;
    std::string reason;
    std::vector<long long> evaluationResultIds;
    std::vector<long long> evaluationRunIds;
    std::vector<std::string> scoringSemanticHashes;
    std::vector<std::string> evaluationSemanticHashes;

    bool acceptableForNewSnapshot() const
    {
        return state ==
                   RecommendationRankingPopulationSemanticState::verifiedHomogeneous ||
               state == RecommendationRankingPopulationSemanticState::empty;
    }
};

RecommendationRankingPopulationSemanticValidation
ValidateRecommendationRankingPopulationSemantics(
    const std::vector<RecommendationRankingEvaluation>& evaluations);

struct RecommendationRankingMember
{
    RecommendationRankingEvaluation evaluation;
    RecommendationRankingBucket bucket =
        RecommendationRankingBucket::nonActionable;
    int bucketRank = 0;
    int globalOrdinal = 0;
    std::string tieBreakPrimary;
    std::string tieBreakSemanticHash;
    std::string tieBreakEvaluationHash;
    std::string inclusionReason;
    std::optional<std::string> blockReason;
    std::optional<std::string> topPositiveComponent;
    std::optional<std::string> topPenaltyComponent;
};

RecommendationRankingBucket RecommendationRankingBucketForDisposition(
    RecommendationEvaluationDisposition disposition);
std::vector<RecommendationRankingMember> RankRecommendationEvaluationEvidence(
    const RecommendationRankingPolicy& policy,
    const std::vector<RecommendationRankingEvaluation>& evaluations,
    int outputLimit);
std::string RecommendationRankingMembershipCanonicalText(
    const std::vector<RecommendationRankingEvaluation>& evaluations);
std::string RecommendationRankingSnapshotIdentityCanonicalText(
    const RecommendationRankingPolicy& policy,
    const RecommendationRankingScope& scope,
    int outputLimit,
    const std::vector<RecommendationRankingEvaluation>& evaluations);
std::string RecommendationRankingSnapshotIdentityCanonicalTextFromMembership(
    const RecommendationRankingPolicy& policy,
    const RecommendationRankingScope& scope,
    int outputLimit,
    const std::string& membershipCanonical,
    const RecommendationRankingPopulationSemanticValidation& semantics);

enum class RecommendationComparisonState
{
    comparable,
    incomparablePolicyVersion,
    incomparableEvaluatorVersion,
    incomparableScoringSemantics,
    incomparableEvaluationSemantics,
    invalidScoringProvenance,
    invalidEvaluationProvenance,
    incomparableMissingScore,
    incomparableScope,
    incomparableFamily,
    identicalScoreTie,
    leftRankedHigher,
    rightRankedHigher
};

std::string RecommendationComparisonStateText(
    RecommendationComparisonState value);

struct RecommendationComponentDifference
{
    std::string componentName;
    std::optional<double> leftContribution;
    std::optional<double> rightContribution;
    std::optional<double> contributionDelta;
    bool penalty = false;
};

struct RecommendationComparisonResult
{
    long long leftEvaluationResultId = -1;
    long long rightEvaluationResultId = -1;
    std::string leftRecommendationSemanticHash;
    std::string rightRecommendationSemanticHash;
    std::string leftEvaluationPolicyHash;
    std::string rightEvaluationPolicyHash;
    std::string leftScoringPolicyHash;
    std::string rightScoringPolicyHash;
    int leftScoringVersion = 0;
    int rightScoringVersion = 0;
    int leftEvaluatorVersion = 0;
    int rightEvaluatorVersion = 0;
    RecommendationComparisonState state =
        RecommendationComparisonState::incomparableMissingScore;
    std::optional<double> leftScore;
    std::optional<double> rightScore;
    std::optional<double> scoreDelta;
    std::optional<int> leftRank;
    std::optional<int> rightRank;
    std::optional<int> rankDelta;
    std::vector<RecommendationComponentDifference> componentDifferences;
    std::optional<std::string> largestPositiveDifference;
    std::optional<std::string> largestPenaltyDifference;
    std::string explanation;
};

RecommendationComparisonResult CompareRecommendationEvaluations(
    const RecommendationRankingEvaluation& left,
    const RecommendationRankingEvaluation& right,
    std::optional<int> leftRank = std::nullopt,
    std::optional<int> rightRank = std::nullopt,
    bool sameScope = true);

} // namespace EA::ExperimentRecommendation
