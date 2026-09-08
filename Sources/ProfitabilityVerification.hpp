#pragma once

#include "ExperimentRecommendation.hpp"
#include "InferenceProfitabilityRepository.hpp"
#include "ProfitabilityDistribution.hpp"

#include <optional>
#include <cstdint>
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
inline constexpr double kPhase11PrecommittedProfitabilityWeight = 0.025;
inline constexpr const char* kPhase12ValidationCohortIdentityHash =
    "fnv1a64:fe7aee4a1aed8a5e";
inline constexpr const char* kPhase12ArtifactSha256 =
    "8d2176ef26513f6b69d690fa5b550a870aab7190221bace24a18be9200a89bbc";
inline constexpr const char* kPhase12ArtifactPath =
    "docs/archive/phase11/forward-validation/"
    "LSTM_ProfitabilityForwardValidation_Snapshot5_20260831_20260930.txt";
inline constexpr const char* kPhase12PreparationArtifactPath =
    "docs/archive/phase12/prospective-outcome/"
    "LSTM_CampaignManager_ProfitabilityFrozenModel_"
    "ProspectiveOutcomeJobs_Phase12.txt";
inline constexpr const char* kPhase12PreparationArtifactSha256 =
    "441a1957ac2ffd53a0693e2332d2b9f7af1c1f9977ff688f9d8e043199f581e1";
inline constexpr const char* kPhase12PreparationIdentityHash =
    "fnv1a64:efeec6ea26199cc7";
inline constexpr long long kPhase12RankingSnapshotId = 5;
inline constexpr long long kPhase12SourceEvaluationRunId = 6;
inline constexpr const char* kPhase12OutcomeStart = "2026-08-31";
inline constexpr const char* kPhase12OutcomeEnd = "2026-09-30";
inline constexpr const char* kPhase12ControlRankingHash =
    "fnv1a64:33527191afa4caec";
inline constexpr const char* kPhase12CandidateRankingHash =
    "fnv1a64:e4478d9578b1e8c9";
inline constexpr int kPhase12MemberCount = 79;
static_assert(kLiveProfitabilityRankingWeight == 0.0);
static_assert(kLiveProfitabilityScoreContribution == 0.0);
static_assert(kPhase11PrecommittedProfitabilityWeight == 0.025);

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

enum class TemporalCohortClassification
{
    admissibleTemporalHoldout,
    insufficientRankingTimeProvenance,
    insufficientSubsequentOutcome,
    overlappingInputAndOutcomePeriod,
    futureInformationLeakage,
    contextOrIdentityMismatch,
    otherFailClosed
};

std::string TemporalCohortClassificationText(
    TemporalCohortClassification value);

struct CampaignProfitabilityTemporalCohort
{
    long long rankingSnapshotId = -1;
    long long sourceEvaluationRunId = -1;
    std::string asOfTimestamp;
    int totalCandidateCount = 0;
    int validRankingTimeProfitabilityEvidenceCount = 0;
    int unavailableRankingTimeEvidenceCount = 0;
    int legitimateSubsequentOutcomeCount = 0;
    int pointInTimeProvenanceViolationCount = 0;
    int overlappingInputAndOutcomeCount = 0;
    int futureInformationLeakageCount = 0;
    int contextOrIdentityMismatchCount = 0;
    std::map<std::string, std::size_t> rankingTimeUnavailableReasonCounts;
    std::optional<std::string> rankingInputStart;
    std::optional<std::string> rankingInputEnd;
    std::optional<std::string> outcomeStart;
    std::optional<std::string> outcomeEnd;
    bool exactControlReconstruction = false;
    bool rankingPopulationReconstructable = false;
    TemporalCohortClassification classification =
        TemporalCohortClassification::otherFailClosed;
    std::string reason;
    std::string canonical;
    std::string hash;
};

struct CampaignProfitabilityTemporalFeasibilityAudit
{
    std::vector<CampaignProfitabilityTemporalCohort> cohorts;
    std::map<std::string, std::size_t> classificationCounts;
    std::string canonical;
    std::string hash;
};

struct CampaignProfitabilityForwardValidationMember
{
    ShadowCandidate source;
    int controlRank = 0;
    int candidateRank = 0;
    int rankDelta = 0;
    bool evidenceAvailableAtSelectionTime = false;
    std::string rankingTimeProfitabilityObservationIdentityHash;
    std::string canonical;
    std::string hash;
};

struct CampaignProfitabilityForwardValidationTopN
{
    int n = 0;
    std::vector<long long> controlRecommendationIds;
    std::vector<long long> candidateRecommendationIds;
    std::vector<long long> retainedRecommendationIds;
    std::vector<long long> candidateOnlyEntrants;
    std::vector<long long> controlOnlyExits;
    std::string canonical;
    std::string hash;
};

struct CampaignProfitabilityForwardValidationPrecommit
{
    int protocolVersion = 1;
    long long rankingSnapshotId = -1;
    long long sourceEvaluationRunId = -1;
    std::string decisionTimestamp;
    std::string expectedOutcomeStart;
    std::string expectedOutcomeEnd;
    double controlWeight = 0.0;
    double candidateWeight = kPhase11PrecommittedProfitabilityWeight;
    std::string controlRankingHash;
    std::string candidateRankingHash;
    std::vector<CampaignProfitabilityForwardValidationMember> members;
    std::vector<CampaignProfitabilityForwardValidationTopN> topN;
    std::string canonical;
    std::string hash;
};

CampaignProfitabilityForwardValidationPrecommit
BuildCampaignProfitabilityForwardValidationPrecommit(
    const CampaignProfitabilityShadowSource& source,
    const std::string& decisionTimestamp,
    const std::string& expectedOutcomeStart,
    const std::string& expectedOutcomeEnd);

enum class OutcomeJobReadiness
{
    waitingForOutcomeData,
    partiallyAvailable,
    readyToExecute,
    incompatibleSource
};

std::string OutcomeJobReadinessText(OutcomeJobReadiness value);

struct CampaignProfitabilityOutcomeJob
{
    std::string validationCohortIdentityHash;
    long long rankingSnapshotId = -1;
    long long sourceEvaluationRunId = -1;
    long long sourceExperimentId = -1;
    long long sourceModelId = -1;
    std::string symbol;
    int horizon = 0;
    std::string originalTrainStart;
    std::string originalTrainEnd;
    std::string originalInferenceStart;
    std::string originalInferenceEnd;
    std::string outcomeStart;
    std::string outcomeEnd;
    double threshold = 0.0;
    int labelRuleId = 0;
    int targetType = 0;
    int windowSize = 0;
    int inputWidth = 0;
    std::string featureSemanticCanonical;
    std::string featureSemanticHash;
    std::string modelArtifactContentHash;
    long long modelParameterRowCount = 0;
    std::string modelLineageCanonical;
    std::string modelLineageHash;
    std::vector<long long> recommendationIds;
    std::map<int, std::string> topNRoleByRecommendation;
    std::string topNParticipation;
    bool modelExists = false;
    bool exactModelExperimentLink = false;
    bool exactFinalSourceModel = false;
    bool exactOriginalFinalInference = false;
    bool checkpointSubstitution = false;
    bool compatible = false;
    std::string compatibilityState;
    OutcomeJobReadiness readiness =
        OutcomeJobReadiness::waitingForOutcomeData;
    std::string metricDefinitionCanonical;
    std::string metricDefinitionHash;
    std::string canonical;
    std::string hash;
};

struct CampaignProfitabilityOutcomePreparation
{
    std::string artifactPath;
    std::string artifactSha256;
    bool artifactIdentityVerified = false;
    std::string currentDate;
    std::vector<CampaignProfitabilityOutcomeJob> jobs;
    std::vector<CampaignProfitabilityForwardValidationTopN> topN;
    std::string canonical;
    std::string hash;
};

struct CampaignProfitabilityOutcomePersistRequest
{
    CampaignProfitabilityOutcomeJob job;
    double inferenceAccuracy = 0.0;
    std::uint64_t predictionCount = 0;
    std::uint64_t actionableCount = 0;
    std::uint64_t winningActionableCount = 0;
    std::uint64_t losingActionableCount = 0;
    double grossPositiveReturn = 0.0;
    double grossNegativeReturn = 0.0;
    double aggregateReturn = 0.0;
    std::optional<double> averageReturn;
    std::string sourceContentHash;
};

struct CampaignProfitabilityOutcomePersistResult
{
    long long resultId = -1;
    bool created = false;
    std::string outcomeIdentityCanonical;
    std::string outcomeIdentityHash;
};

// Phase 13 compares recommendation selection slots. A repeated recommendation
// backed by one source model therefore retains its repeated selection weight,
// while sourceModelId remains the unit of independent outcome evidence.
struct CampaignProfitabilityProspectiveOutcome
{
    long long resultId = -1;
    std::string validationCohortIdentityHash;
    long long rankingSnapshotId = -1;
    long long sourceEvaluationRunId = -1;
    long long sourceExperimentId = -1;
    long long sourceModelId = -1;
    std::string outcomeStart;
    std::string outcomeEnd;
    std::string jobIdentityHash;
    std::string featureSemanticHash;
    std::string modelLineageHash;
    std::string modelArtifactContentHash;
    std::string metricDefinitionCanonical;
    std::string metricDefinitionHash;
    std::string sourceContentHash;
    std::uint64_t predictionCount = 0;
    std::uint64_t actionableCount = 0;
    std::uint64_t winningActionableCount = 0;
    std::uint64_t losingActionableCount = 0;
    double grossPositiveReturn = 0.0;
    double grossNegativeReturn = 0.0;
    double aggregateReturn = 0.0;
    std::optional<double> averageReturn;
    std::string outcomeIdentityCanonical;
    std::string outcomeIdentityHash;
};

enum class ProspectiveComparisonReadiness
{
    pendingOutcomes,
    incompleteChangedSelectionCoverage,
    comparisonComplete,
    incompatibleSource,
    artifactIdentityMismatch,
    metricIdentityMismatch,
    cohortIdentityMismatch,
    outcomeWindowIdentityMismatch
};

std::string ProspectiveComparisonReadinessText(
    ProspectiveComparisonReadiness value);

struct CampaignProfitabilitySourceCoverage
{
    std::vector<long long> requiredSourceModelIds;
    std::vector<long long> coveredSourceModelIds;
    std::vector<long long> missingSourceModelIds;
    std::vector<long long> incompatibleSourceModelIds;
    int requiredCount = 0;
    int coveredCount = 0;
    double percentage = 0.0;
};

struct CampaignProfitabilityContribution
{
    int recommendationCount = 0;
    int uniqueSourceModelCount = 0;
    std::uint64_t predictionCount = 0;
    std::uint64_t actionableCount = 0;
    double aggregateReturn = 0.0;
    std::optional<double> averageReturnPerActionablePrediction;
};

struct CampaignProfitabilityProspectiveTopNComparison
{
    int n = 0;
    std::vector<long long> controlRecommendationIds;
    std::vector<long long> candidateRecommendationIds;
    std::vector<long long> retainedRecommendationIds;
    std::vector<long long> entrantRecommendationIds;
    std::vector<long long> exitRecommendationIds;
    std::map<long long, std::vector<long long>>
        recommendationIdsBySourceModel;
    std::vector<long long> controlSourceModelIds;
    std::vector<long long> candidateSourceModelIds;
    std::vector<long long> retainedSourceModelIds;
    std::vector<long long> entrantSourceModelIds;
    std::vector<long long> exitSourceModelIds;
    CampaignProfitabilitySourceCoverage changedSelectionCoverage;
    std::optional<CampaignProfitabilityContribution> entrantContribution;
    std::optional<CampaignProfitabilityContribution> exitContribution;
    std::optional<double> candidateMinusControlIncrementalProfitability;
    ProspectiveComparisonReadiness readiness =
        ProspectiveComparisonReadiness::pendingOutcomes;
    std::vector<std::string> blockingReasons;
    bool final = false;
    std::string canonical;
    std::string hash;
};

struct CampaignProfitabilityProspectiveComparisonRequest
{
    std::string validationCohortIdentityHash;
    std::string phase11ArtifactSha256;
    std::string phase12PreparationArtifactSha256;
    std::string phase12PreparationIdentityHash;
    std::string metricDefinitionCanonical;
    std::string metricDefinitionHash;
    std::string outcomeStart;
    std::string outcomeEnd;
    std::string currentDate;
    CampaignProfitabilityOutcomePreparation preparation;
    std::vector<CampaignProfitabilityProspectiveOutcome> outcomes;
};

struct CampaignProfitabilityProspectiveComparison
{
    int protocolVersion = 1;
    std::string validationCohortIdentityHash;
    std::string phase11ArtifactSha256;
    std::string phase12PreparationArtifactSha256;
    std::string phase12PreparationIdentityHash;
    std::string metricDefinitionCanonical;
    std::string metricDefinitionHash;
    std::string outcomeStart;
    std::string outcomeEnd;
    std::string currentDate;
    CampaignProfitabilitySourceCoverage fullFrozenCohortCoverage;
    std::vector<CampaignProfitabilityProspectiveTopNComparison> topN;
    ProspectiveComparisonReadiness readiness =
        ProspectiveComparisonReadiness::pendingOutcomes;
    std::vector<std::string> blockingReasons;
    bool final = false;
    std::string canonical;
    std::string hash;
};

CampaignProfitabilityProspectiveComparison
BuildCampaignProfitabilityProspectiveComparison(
    CampaignProfitabilityProspectiveComparisonRequest request);

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
