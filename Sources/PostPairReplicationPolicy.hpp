#pragma once

#include "PairedTrainingObjectiveEvaluation.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace EA::PostPairReplicationPolicy
{

namespace Pair = PairedTrainingObjectiveEvaluation;

enum class NextAction
{
    WaitForValidResult,
    RepairComparability,
    StopObjective,
    Replicate,
    ReplicateBroader,
    EligibleForCoefficientExploration
};

// Every threshold is caller-visible. ConservativeFirstAuxiliaryScreening()
// supplies the Phase 4E recommendation, but Evaluate() never hides or infers
// a scientific threshold.
struct Policy
{
    std::size_t minimumValidReplications = 4;
    std::size_t minimumPromisingReplications = 3;
    double minimumPromisingFraction = 0.75;
    double maximumMixedFraction = 0.25;
    std::size_t maximumNotPromisingForEligibility = 0;
    std::size_t minimumDistinctSymbols = 3;
    std::size_t minimumDistinctHorizons = 2;
    std::size_t confirmatoryNotPromisingReplications = 2;
    std::size_t maximumValidReplicationsBeforeStop = 6;
    std::string initialSymbol;
    int initialHorizon = 0;
    bool requireReplicationOutsideInitialUnit = true;
};

Policy ConservativeFirstAuxiliaryScreening(
    std::string initialSymbol,
    int initialHorizon);
std::string PolicyCanonicalText(const Policy& policy);
std::string PolicyIdentity(const Policy& policy);

// This is the Phase 4E input contract. It is deliberately a summary of an
// already-evaluated Phase 4D result; Phase 4E neither reloads nor reselects
// experiment evidence. replicationUnitCanonical must describe the complete
// scientific configuration and input evidence while excluding experiment IDs
// and objective identity; its deterministic hash is replicationUnitIdentity.
// With the current seed-42 initialization semantics, exact deterministic
// reruns must use the same replication-unit canonical text and identity. A
// future stochastic policy must include the initialization seed/lineage.
struct PairSummary
{
    std::string pairIdentity;
    std::string replicationUnitCanonical;
    std::string replicationUnitIdentity;
    std::string symbol;
    int horizon = 0;
    Pair::ObjectiveProvenance controlObjective;
    Pair::ObjectiveProvenance treatmentObjective;
    Pair::ProfitabilityPrimaryMetric primaryProfitabilityMetric =
        Pair::ProfitabilityPrimaryMetric::
            AggregateTerminalHorizonLogReturnSum;
    std::string materialityPolicyCanonical;
    std::string materialityPolicyIdentity;
    Pair::MetricDelta primaryProfitability;
    Pair::MetricDelta aggregateProfitability;
    Pair::MetricDelta averageProfitability;
    Pair::MetricDelta inferenceAccuracy;
    Pair::MetricDelta acceptAccuracy;
    Pair::MetricDelta acceptRate;
    Pair::MetricDelta predictedNeutralProportion;
    Pair::MetricDelta leaderScore;
    Pair::Disposition disposition = Pair::Disposition::Incomplete;
    std::vector<std::string> invalidReasons;
    std::vector<std::string> incompleteReasons;
    std::vector<std::string> interpretationReasons;
    bool operator==(const PairSummary&) const = default;
};

PairSummary Summarize(
    std::string pairIdentity,
    std::string replicationUnitCanonical,
    const Pair::ArmEvidence& control,
    const Pair::ArmEvidence& treatment,
    const Pair::MaterialityPolicy& materialityPolicy,
    const Pair::ComparisonResult& comparison);

struct Evaluation
{
    Pair::ObjectiveProvenance objectiveUnderReview;
    std::size_t validPairCount = 0;
    std::size_t promisingCount = 0;
    std::size_t mixedCount = 0;
    std::size_t notPromisingCount = 0;
    std::size_t invalidCount = 0;
    std::size_t incompleteCount = 0;
    std::size_t distinctSymbolCount = 0;
    std::size_t distinctHorizonCount = 0;
    std::string materialityPolicyIdentity;
    std::string replicationPolicyIdentity;
    bool replicationDiversitySatisfied = false;
    NextAction nextAction = NextAction::WaitForValidResult;
    std::vector<std::string> reasons;
    // Unique pair identities, sorted lexicographically. Pair-level provenance
    // is retained even when a deterministic replication unit counts once.
    std::vector<PairSummary> pairEvidence;
};

Evaluation Evaluate(
    const Pair::ObjectiveProvenance& objectiveUnderReview,
    const Policy& policy,
    std::vector<PairSummary> comparisons);

std::string NextActionText(NextAction value);
std::string RenderMachineReadable(const Evaluation& evaluation);

} // namespace EA::PostPairReplicationPolicy
