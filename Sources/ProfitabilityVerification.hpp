#pragma once

#include "ExperimentRecommendation.hpp"
#include "InferenceProfitabilityRepository.hpp"
#include "ProfitabilityDistribution.hpp"

#include <optional>
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
