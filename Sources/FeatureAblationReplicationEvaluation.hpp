#pragma once

#include "FeatureAblationPairEvaluation.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace EA::FeatureAblationReplicationEvaluation
{

namespace Pair = FeatureAblationPairEvaluation;

inline constexpr int kReplicationEvaluationVersion = 1;
inline constexpr int kReplicationPolicyVersion = 1;
inline constexpr int kMinimumValidReplications = 3;
inline constexpr std::string_view kEconomicEventSoftwareReadinessVersion =
    "economic_event_features_productionization_readiness_v1";

enum class MemberEvidenceState
{
    Complete,
    Incomplete,
    Invalid,
    MissingEvidence,
    ProfitabilityUnavailable
};

enum class ReplicationDecision
{
    InsufficientEvidence,
    Promising,
    Mixed,
    NotPromising
};

enum class ProductionizationAction
{
    AwaitReplication,
    EligibleForActivationReview,
    DoNotEnable,
    BlockedSoftwareReadiness
};

struct ReplicationPolicy
{
    int version = kReplicationPolicyVersion;
    int minimumValidReplications = kMinimumValidReplications;
};

struct SoftwareReadinessAudit
{
    int version = 1;
    bool causalReleaseBoundary = true;
    bool consensusProvenanceRetained = true;
    bool missingConsensusExplicit = true;
    bool scalarRangeSemanticsValidated = true;
    bool normalizationFailClosed = true;
    bool finiteFeatureValuesEnforced = true;
    bool inputSemanticIdentityFailClosed = true;
    bool deterministicFeaturePathTested = true;
    bool availabilityDiagnosticsPresent = true;
    bool readOnlyEvaluation = true;
    bool activationMutationPathAbsent = true;
};

struct MemberEvaluation
{
    std::size_t ordinal = 0;
    long long controlExperimentId = 0;
    long long treatmentExperimentId = 0;
    std::string symbol;
    std::optional<int> predictionHorizon;
    MemberEvidenceState evidenceState = MemberEvidenceState::MissingEvidence;
    Pair::Disposition pairDisposition = Pair::Disposition::ComparableIncomplete;
    std::string pairEvaluationIdentityHash;
    std::string ablationIdentityHash;
    bool scientificallyValidComplete = false;
    std::vector<std::string> incompleteReasons;
    std::vector<std::string> invalidReasons;
    Pair::ComparisonResult comparison;
};

struct MetricSummary
{
    std::size_t valueCount = 0;
    std::size_t positiveCount = 0;
    std::size_t negativeCount = 0;
    std::size_t zeroCount = 0;
    std::optional<double> sum;
    std::optional<double> mean;
    std::optional<double> median;
};

struct PopulationSummary
{
    std::size_t declaredPairCount = 0;
    std::size_t completeComparablePairCount = 0;
    std::size_t incompletePairCount = 0;
    std::size_t invalidPairCount = 0;
    std::size_t missingEvidencePairCount = 0;
    std::size_t profitabilityUnavailablePairCount = 0;
    std::size_t profitabilityAvailablePairCount = 0;
    std::size_t profitabilityPositivePairCount = 0;
    std::size_t profitabilityNegativePairCount = 0;
    std::size_t profitabilityZeroPairCount = 0;
    std::size_t distinctSymbolCount = 0;
    std::size_t distinctHorizonCount = 0;
};

struct ReplicationEvaluation
{
    std::vector<MemberEvaluation> members;
    PopulationSummary population;
    MetricSummary aggregateProfitability;
    MetricSummary averageProfitability;
    MetricSummary inferenceAccuracy;
    MetricSummary leaderScore;
    MetricSummary neutralProportion;
    MetricSummary actionableCount;
    ReplicationDecision decision = ReplicationDecision::InsufficientEvidence;
    ProductionizationAction action =
        ProductionizationAction::AwaitReplication;
    bool evidenceIntegrityFailure = false;
    bool softwareReady = false;
    std::vector<std::string> integrityReasons;
    std::string membershipIdentityCanonical;
    std::string membershipIdentityHash;
    std::string policyCanonical;
    std::string policyHash;
    std::string softwareReadinessCanonical;
    std::string softwareReadinessHash;
    std::string evaluationIdentityCanonical;
    std::string evaluationIdentityHash;
};

std::vector<std::pair<long long, long long>> ParseExperimentIdPairs(
    std::string_view text);

MemberEvaluation MakeMemberEvaluation(
    std::size_t ordinal,
    const Pair::ArmEvidence& control,
    const Pair::ArmEvidence& treatment,
    const Pair::ComparisonResult& comparison);

std::string PolicyCanonicalText(const ReplicationPolicy& policy);
std::string PolicyHash(const ReplicationPolicy& policy);
std::string SoftwareReadinessCanonicalText(
    const SoftwareReadinessAudit& audit);
bool SoftwareReady(const SoftwareReadinessAudit& audit);

ReplicationEvaluation Evaluate(
    std::vector<MemberEvaluation> members,
    const ReplicationPolicy& policy = {},
    const SoftwareReadinessAudit& softwareAudit = {});

std::string MemberEvidenceStateText(MemberEvidenceState value);
std::string ReplicationDecisionText(ReplicationDecision value);
std::string ProductionizationActionText(ProductionizationAction value);
int ExitCode(const ReplicationEvaluation& result);

} // namespace EA::FeatureAblationReplicationEvaluation
