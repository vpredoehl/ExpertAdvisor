#pragma once

#include "ExperimentRecommendation.hpp"
#include "InferenceProfitabilityRepository.hpp"
#include "ProfitabilityDistribution.hpp"

#include <optional>
#include <map>
#include <string>
#include <vector>

namespace EA::ProfitabilityVerification
{

inline constexpr int kEvidenceContractVersion = 1;
inline constexpr int kShadowRankingPolicyVersion = 1;
inline constexpr double kLiveProfitabilityRankingWeight = 0.0;
inline constexpr double kLiveProfitabilityScoreContribution = 0.0;
inline constexpr int kWeightedShadowRankingPolicyVersion = 1;
inline constexpr double kMaximumPhase9ProfitabilityShadowWeight = 0.05;
static_assert(kLiveProfitabilityRankingWeight == 0.0);
static_assert(kLiveProfitabilityScoreContribution == 0.0);

enum class EvidenceState
{
    valid,
    unavailable,
    incomplete,
    ambiguous,
    invalidProvenance,
    invalidMetricDefinition,
    invalidValues
};

std::string EvidenceStateText(EvidenceState state);
bool EvidenceStateIsInvalid(EvidenceState state);

struct ExpectedFinalEvidence
{
    long long experimentId = -1;
    long long modelId = -1;
    long long inferenceEvalResultId = -1;
    std::string inferenceStart;
    std::string inferenceEnd;
};

struct EvidenceResult
{
    int contractVersion = kEvidenceContractVersion;
    long long experimentId = -1;
    std::optional<long long> finalModelId;
    std::optional<long long> finalInferenceEvalResultId;
    std::optional<InferenceProfitability::Observation> observation;
    EvidenceState state = EvidenceState::incomplete;
    std::string reason = "not_evaluated";
    std::string evidenceIdentityCanonical;
    std::string evidenceIdentityHash;
};

EvidenceResult ValidateExactFinalObservation(
    const ExpectedFinalEvidence& expected,
    const std::optional<InferenceProfitability::Observation>& observation);

EvidenceResult EnforceFrozenCampaignEvidence(
    const std::optional<long long>& candidateSourceModelId,
    const std::optional<ExperimentRecommendation::RecommendationSource::
        FinalProfitabilityEvidence>& frozen,
    EvidenceResult current);

EvidenceResult ValidateFrozenCampaignEvidence(
    long long sourceExperimentId,
    const std::optional<long long>& sourceModelId,
    const ExperimentRecommendation::RecommendationSource::
        FinalProfitabilityEvidence& frozen,
    const std::optional<InferenceProfitability::Observation>& observation);

std::vector<long long> ParseDeclaredExperimentIds(const std::string& value);

int ExitCode(const std::vector<EvidenceResult>& results);

enum class ProfitabilitySign
{
    positive,
    zero,
    negative,
    zeroActionable,
    unavailable,
    invalid
};

std::string ProfitabilitySignText(ProfitabilitySign sign);

struct ShadowCandidate
{
    long long rankingMemberId = -1;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    long long recommendationEvaluationResultId = -1;
    long long recommendationEvaluationRunId = -1;
    std::optional<long long> sourceModelId;
    std::string symbol;
    int horizon = 0;
    int currentRank = 0;
    std::optional<double> currentScore;
    double leaderScore = 0.0;
    double inferenceAccuracy = 0.0;
    std::optional<double> predictedNeutralProportion;
    EvidenceResult profitability;

    int profitabilityShadowRank = 0;
    int rankDelta = 0;
    ProfitabilitySign profitabilitySign = ProfitabilitySign::unavailable;
    std::string canonical;
    std::string hash;
};

struct ShadowRanking
{
    int policyVersion = kShadowRankingPolicyVersion;
    std::string policyCanonical;
    std::string policyHash;
    std::vector<ShadowCandidate> candidates;
    std::string canonical;
    std::string hash;
};

std::string ShadowRankingPolicyCanonicalText();
std::string ShadowRankingPolicyHash();
ShadowRanking BuildShadowRanking(std::vector<ShadowCandidate> candidates);

std::vector<double> ParseProfitabilityShadowWeights(const std::string& value);

struct WeightedShadowCandidate
{
    ShadowCandidate source;
    std::string normalizationState;
    std::string normalizationReason;
    std::optional<double> empiricalMidrankPercentile;
    std::optional<double> boundedCandidateMetric;
    std::optional<double> supportReliability;
    std::optional<double> normalizedProfitabilityValue;
    std::optional<double> profitabilityContribution;
    double shadowFinalScore = 0.0;
    int shadowRank = 0;
    int rankDelta = 0;
    std::string canonical;
    std::string hash;
};

struct WeightedShadowRanking
{
    int policyVersion = kWeightedShadowRankingPolicyVersion;
    long long controlSnapshotId = -1;
    long long sourceEvaluationRunId = -1;
    double shadowWeight = 0.0;
    std::string policyCanonical;
    std::string policyHash;
    ExperimentRecommendation::ProfitabilityShadowNormalizationAnalysis
        normalization;
    std::vector<WeightedShadowCandidate> candidates;
    std::string canonical;
    std::string hash;
};

struct CampaignProfitabilityShadowSource
{
    long long controlSnapshotId = -1;
    long long sourceEvaluationRunId = -1;
    std::string controlSnapshotIdentityCanonical;
    std::string controlSnapshotIdentityHash;
    std::string controlRankingPolicyCanonical;
    std::string controlRankingPolicyHash;
    int persistedMemberCount = 0;
    std::vector<ShadowCandidate> candidates;
};

enum class CoverageRecoveryClass
{
    available,
    recoverableHistoricalAbsence,
    contextMismatch,
    noExactFinalInference,
    legacyIncompleteProvenance,
    invalidIncompleteProvenance,
    otherUnavailable
};

std::string CoverageRecoveryClassText(CoverageRecoveryClass value);

struct CampaignProfitabilityCoverageMember
{
    long long rankingMemberId = -1;
    long long recommendationId = -1;
    long long recommendationEvaluationResultId = -1;
    long long sourceExperimentId = -1;
    std::optional<long long> sourceModelId;
    std::string symbol;
    int horizon = 0;
    int controlRank = 0;
    ExperimentRecommendation::RecommendationSource::FinalProfitabilityEvidence
        frozenEvidence;
    EvidenceResult validatedEvidence;
    CoverageRecoveryClass recoveryClass =
        CoverageRecoveryClass::invalidIncompleteProvenance;
    bool exactFinalInferenceResultExists = false;
    bool anyFinalInferenceResultExists = false;
    bool exactFinalProfitabilityObservationExists = false;
    bool anyInferenceProfitabilityObservationExists = false;
    // Snapshot-5 recommendation/evaluation evidence is immutable.  This is
    // deliberately false even when an exact FINAL result could be replayed by
    // a separately authorized future lifecycle operation.
    bool frozenSnapshotBackfillPermitted = false;
    std::string reconstructionAssessment;
    std::string canonical;
    std::string hash;
};

struct CampaignProfitabilityCoverageAudit
{
    long long controlSnapshotId = -1;
    long long sourceEvaluationRunId = -1;
    std::vector<CampaignProfitabilityCoverageMember> members;
    std::map<std::string, std::size_t> reasonCounts;
    std::map<std::string, std::size_t> recoveryClassCounts;
    std::string canonical;
    std::string hash;
};

struct CalibrationMovementStatistics
{
    int count = 0;
    int movedUp = 0;
    int unchanged = 0;
    int movedDown = 0;
    double meanRankDelta = 0.0;
    double meanAbsoluteRankMovement = 0.0;
    double medianAbsoluteRankMovement = 0.0;
    double p90AbsoluteRankMovement = 0.0;
    int maximumUpwardMovement = 0;
    int maximumDownwardMovement = 0;
};

struct CalibrationTopN
{
    int n = 0;
    int retained = 0;
    int entered = 0;
    int exited = 0;
    std::vector<long long> entrantRecommendationIds;
    std::vector<long long> exitingRecommendationIds;
    std::vector<long long> memberRecommendationIds;
    int positiveCount = 0;
    int negativeCount = 0;
    int zeroCount = 0;
    int unavailableCount = 0;
    int validEvidenceCoverageCount = 0;
    double validEvidenceCoveragePercentage = 0.0;
};

struct ProfitabilityCalibrationWeight
{
    double weight = 0.0;
    int totalMembers = 0;
    int validProfitabilityMembers = 0;
    int unavailableMembers = 0;
    int positiveProfitabilityMembers = 0;
    int negativeProfitabilityMembers = 0;
    int zeroProfitabilityMembers = 0;
    CalibrationMovementStatistics totalMovement;
    CalibrationMovementStatistics positiveMovement;
    CalibrationMovementStatistics negativeMovement;
    CalibrationMovementStatistics unavailableMovement;
    std::vector<CalibrationTopN> topN;
    std::string rankingHash;
    std::string canonical;
    std::string hash;
};

struct ProfitabilityCalibrationPairwise
{
    double leftWeight = 0.0;
    double rightWeight = 0.0;
    int ordinalChanges = 0;
    double meanAbsoluteOrdinalDifference = 0.0;
    int largestOrdinalDifference = 0;
    std::vector<long long> largestOrdinalDifferenceRecommendationIds;
    std::map<int, int> topNOverlap;
    std::map<int, std::vector<long long>> topNDifferenceRecommendationIds;
    std::string canonical;
    std::string hash;
};

struct ProfitabilityCalibrationStabilityRegion
{
    int topN = 0;
    double firstWeight = 0.0;
    double lastWeight = 0.0;
    std::vector<long long> memberRecommendationIds;
};

struct ProfitabilityCalibrationResponseCurve
{
    std::optional<double> firstBestTop5Weight;
    std::optional<double> firstTop10AtLeastNineWeight;
    std::optional<double> firstBestTop10Weight;
    std::optional<double> firstTop20ImprovementWeight;
    std::optional<double> minimumEffectiveWeight;
    std::optional<double> minimumEffectiveRegionEnd;
    bool provisional0025InsideMinimumEffectiveRegion = false;
    std::vector<double> membershipDiscontinuityWeights;
    std::vector<ProfitabilityCalibrationStabilityRegion> stabilityRegions;
    std::string assessment;
    std::string canonical;
    std::string hash;
};

struct ProfitabilityCalibrationReport
{
    long long controlSnapshotId = -1;
    long long sourceEvaluationRunId = -1;
    std::vector<ProfitabilityCalibrationWeight> weights;
    std::vector<ProfitabilityCalibrationPairwise> anchorPairwise;
    ProfitabilityCalibrationResponseCurve responseCurve;
    std::string canonical;
    std::string hash;
};

std::vector<double> Phase10ProfitabilityCalibrationWeights();
ProfitabilityCalibrationReport BuildProfitabilityCalibrationReport(
    const std::vector<WeightedShadowRanking>& rankings);

WeightedShadowRanking BuildWeightedShadowRanking(
    std::vector<ShadowCandidate> candidates,
    long long controlSnapshotId,
    long long sourceEvaluationRunId,
    const std::string& controlSnapshotIdentityHash,
    double shadowWeight,
    const ExperimentRecommendation::ProfitabilityShadowNormalizationPolicy&
        normalizationPolicy = {});

enum class ReadinessAction
{
    readyForShadowValidation,
    blockedEvidenceContract,
    blockedSoftwareReadiness,
    needsPolicyDecision,
    eligibleForActivationReview
};

std::string ReadinessActionText(ReadinessAction action);

struct ReadinessGate
{
    bool profitabilitySoftwareReady = false;
    bool campaignProfitabilityContractReady = false;
    bool profitabilityShadowRankingReady = false;
    bool activationPerformed = false;
    ReadinessAction action = ReadinessAction::blockedSoftwareReadiness;
};

ReadinessGate EvaluateReadinessGate(bool softwareReady,
                                    bool integrationLoaded,
                                    const ShadowRanking& shadow);

} // namespace EA::ProfitabilityVerification
