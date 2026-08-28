#include "../Sources/ProfitabilityVerification.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Verification = EA::ProfitabilityVerification;
namespace Profitability = EA::InferenceProfitability;
namespace Recommendation = EA::ExperimentRecommendation;

namespace
{

Verification::ExpectedFinalEvidence Expected()
{
    return {101, 201, 301, "2025-01-01", "2025-12-31"};
}

void BindObservationIdentity(Profitability::Observation& value)
{
    Profitability::ObservationRequest request;
    request.provenance = value.provenance;
    request.statistics = value.statistics;
    request.sourceContentHash = value.sourceContentHash;
    request.metricDefinitionCanonical = value.metricDefinitionCanonical;
    value.observationIdentityCanonical =
        Profitability::BuildObservationIdentityCanonical(request);
    value.observationIdentityHash = Profitability::DeterministicHash(
        value.observationIdentityCanonical);
}

Profitability::Observation Observation(double aggregate = 0.40,
                                       std::uint64_t actionable = 20)
{
    Profitability::Observation value;
    value.observationId = 401;
    value.provenance.experimentId = 101;
    value.provenance.modelId = 201;
    value.provenance.inferenceEvalResultId = 301;
    value.provenance.scope = Profitability::Scope::finalInference;
    value.provenance.inferenceStart = "2025-01-01";
    value.provenance.inferenceEnd = "2025-12-31";
    value.statistics.predictionCount = 25;
    value.statistics.actionableCount = actionable;
    if (aggregate > 0.0)
    {
        value.statistics.winningActionableCount = actionable;
        value.statistics.grossPositiveTerminalHorizonLogReturnSum = aggregate;
    }
    else if (aggregate < 0.0)
    {
        value.statistics.losingActionableCount = actionable;
        value.statistics.grossNegativeTerminalHorizonLogReturnSum = aggregate;
    }
    value.statistics.aggregateTerminalHorizonLogReturnSum = aggregate;
    if (actionable != 0)
        value.averageTerminalHorizonLogReturnPerActionablePrediction =
            aggregate / static_cast<double>(actionable);
    value.metricDefinitionCanonical =
        Profitability::kMetricDefinitionCanonical;
    value.metricDefinitionHash = Profitability::MetricDefinitionHash();
    value.sourceContentHash = Profitability::DeterministicHash("source");
    BindObservationIdentity(value);
    value.createdAt = "2026-08-28 00:00:00+00";
    return value;
}

Verification::EvidenceResult Result(double aggregate,
                                    std::uint64_t actionable = 20)
{
    return Verification::ValidateExactFinalObservation(
        Expected(), Observation(aggregate, actionable));
}

Verification::ShadowCandidate Candidate(long long memberId,
                                        int currentRank,
                                        double accuracy,
                                        Verification::EvidenceResult result)
{
    Verification::ShadowCandidate candidate;
    candidate.rankingMemberId = memberId;
    candidate.recommendationId = 1000 + memberId;
    candidate.sourceExperimentId = result.experimentId;
    candidate.sourceModelId = result.finalModelId;
    candidate.currentRank = currentRank;
    candidate.currentScore = accuracy;
    candidate.leaderScore = accuracy;
    candidate.inferenceAccuracy = accuracy;
    candidate.predictedNeutralProportion = 0.1;
    candidate.profitability = std::move(result);
    return candidate;
}

Recommendation::RecommendationSource::FinalProfitabilityEvidence Frozen(
    const Verification::EvidenceResult& result)
{
    assert(result.observation);
    const auto& observation = *result.observation;
    Recommendation::RecommendationSource::FinalProfitabilityEvidence frozen;
    frozen.finalInferenceEvalResultId = result.finalInferenceEvalResultId;
    frozen.profitabilityObservationId = observation.observationId;
    frozen.inferenceScope = "final";
    frozen.inferenceStart = observation.provenance.inferenceStart;
    frozen.inferenceEnd = observation.provenance.inferenceEnd;
    frozen.actionablePredictionCount = static_cast<long long>(
        observation.statistics.actionableCount);
    frozen.aggregateTerminalHorizonLogReturnSum =
        observation.statistics.aggregateTerminalHorizonLogReturnSum;
    frozen.averageTerminalHorizonLogReturnPerActionablePrediction =
        observation.averageTerminalHorizonLogReturnPerActionablePrediction;
    frozen.metricDefinitionHash = observation.metricDefinitionHash;
    frozen.sourceContentHash = observation.sourceContentHash;
    frozen.observationIdentityHash = observation.observationIdentityHash;
    return frozen;
}

template <typename Function>
void ThrowsInvalidArgument(Function&& function)
{
    bool threw = false;
    try
    {
        function();
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main()
{
    const auto valid = Result(0.40);
    assert(valid.state == Verification::EvidenceState::valid);
    assert(Verification::ExitCode({valid}) == 0);

    const auto unavailable = Verification::ValidateExactFinalObservation(
        Expected(), std::nullopt);
    assert(unavailable.state == Verification::EvidenceState::unavailable);
    assert(!unavailable.observation);
    assert(Verification::ExitCode({unavailable}) == 4);

    auto checkpoint = Observation();
    checkpoint.provenance.scope = Profitability::Scope::checkpointInference;
    checkpoint.provenance.checkpointEvalId = 501;
    assert(Verification::ValidateExactFinalObservation(Expected(), checkpoint)
               .state == Verification::EvidenceState::invalidProvenance);

    auto wrongExperiment = Observation();
    wrongExperiment.provenance.experimentId = 999;
    assert(Verification::ValidateExactFinalObservation(
               Expected(), wrongExperiment).state ==
           Verification::EvidenceState::invalidProvenance);
    auto wrongModel = Observation();
    wrongModel.provenance.modelId = 999;
    assert(Verification::ValidateExactFinalObservation(Expected(), wrongModel)
               .state == Verification::EvidenceState::invalidProvenance);
    auto wrongResult = Observation();
    wrongResult.provenance.inferenceEvalResultId = 999;
    assert(Verification::ValidateExactFinalObservation(Expected(), wrongResult)
               .state == Verification::EvidenceState::invalidProvenance);
    auto wrongRange = Observation();
    wrongRange.provenance.inferenceEnd = "2026-01-01";
    assert(Verification::ValidateExactFinalObservation(Expected(), wrongRange)
               .state == Verification::EvidenceState::invalidProvenance);

    auto wrongMetric = Observation();
    wrongMetric.metricDefinitionCanonical = "wrong";
    wrongMetric.metricDefinitionHash = Profitability::DeterministicHash("wrong");
    assert(Verification::ValidateExactFinalObservation(Expected(), wrongMetric)
               .state == Verification::EvidenceState::invalidMetricDefinition);

    auto nonfiniteAggregate = Observation();
    nonfiniteAggregate.statistics.aggregateTerminalHorizonLogReturnSum =
        std::numeric_limits<double>::infinity();
    assert(Verification::ValidateExactFinalObservation(
               Expected(), nonfiniteAggregate).state ==
           Verification::EvidenceState::invalidValues);
    auto nonfiniteAverage = Observation();
    nonfiniteAverage.averageTerminalHorizonLogReturnPerActionablePrediction =
        std::numeric_limits<double>::quiet_NaN();
    assert(Verification::ValidateExactFinalObservation(
               Expected(), nonfiniteAverage).state ==
           Verification::EvidenceState::invalidValues);
    auto invalidCounts = Observation();
    invalidCounts.statistics.actionableCount = 26;
    assert(Verification::ValidateExactFinalObservation(
               Expected(), invalidCounts).state ==
           Verification::EvidenceState::invalidValues);

    auto accumulatedReturns = Observation();
    accumulatedReturns.statistics.predictionCount = 18334;
    accumulatedReturns.statistics.actionableCount = 12011;
    accumulatedReturns.statistics.winningActionableCount = 6000;
    accumulatedReturns.statistics.losingActionableCount = 6011;
    accumulatedReturns.statistics.grossPositiveTerminalHorizonLogReturnSum =
        6.62603827126973;
    accumulatedReturns.statistics.grossNegativeTerminalHorizonLogReturnSum =
        -5.882329511385737;
    accumulatedReturns.statistics.aggregateTerminalHorizonLogReturnSum =
        0.7437087598839783;
    accumulatedReturns.averageTerminalHorizonLogReturnPerActionablePrediction =
        accumulatedReturns.statistics.aggregateTerminalHorizonLogReturnSum /
        static_cast<double>(accumulatedReturns.statistics.actionableCount);
    BindObservationIdentity(accumulatedReturns);
    assert(Verification::ValidateExactFinalObservation(
               Expected(), accumulatedReturns).state ==
           Verification::EvidenceState::valid);

    accumulatedReturns.statistics.aggregateTerminalHorizonLogReturnSum += 1e-6;
    accumulatedReturns.averageTerminalHorizonLogReturnPerActionablePrediction =
        accumulatedReturns.statistics.aggregateTerminalHorizonLogReturnSum /
        static_cast<double>(accumulatedReturns.statistics.actionableCount);
    BindObservationIdentity(accumulatedReturns);
    assert(Verification::ValidateExactFinalObservation(
               Expected(), accumulatedReturns).state ==
           Verification::EvidenceState::invalidValues);

    auto zeroActionable = Observation(0.0, 0);
    assert(Verification::ValidateExactFinalObservation(
               Expected(), zeroActionable).state ==
           Verification::EvidenceState::valid);
    zeroActionable.averageTerminalHorizonLogReturnPerActionablePrediction = 0.0;
    assert(Verification::ValidateExactFinalObservation(
               Expected(), zeroActionable).state ==
           Verification::EvidenceState::invalidValues);

    const auto positive = Result(0.40);
    const auto negative = Result(-0.40);
    assert(positive.state == Verification::EvidenceState::valid);
    assert(negative.state == Verification::EvidenceState::valid);
    assert(unavailable.state != negative.state);

    const auto validAgain = Result(0.40);
    assert(valid.evidenceIdentityCanonical ==
           validAgain.evidenceIdentityCanonical);
    assert(valid.evidenceIdentityHash == validAgain.evidenceIdentityHash);
    auto invalidIdentity = Observation();
    invalidIdentity.observationIdentityCanonical += "tampered";
    invalidIdentity.observationIdentityHash = Profitability::DeterministicHash(
        invalidIdentity.observationIdentityCanonical);
    assert(Verification::ValidateExactFinalObservation(
               Expected(), invalidIdentity).state ==
           Verification::EvidenceState::invalidProvenance);

    assert((Verification::ParseDeclaredExperimentIds("9,3,7") ==
            std::vector<long long>{9, 3, 7}));
    ThrowsInvalidArgument([] {
        (void)Verification::ParseDeclaredExperimentIds("9,3,9");
    });
    ThrowsInvalidArgument([] {
        (void)Verification::ParseDeclaredExperimentIds("9,,3");
    });

    std::vector<Verification::ShadowCandidate> candidates;
    candidates.push_back(Candidate(11, 1, 0.99, negative));
    candidates.push_back(Candidate(12, 2, 0.50, positive));
    auto tie = Result(0.40);
    tie.observation->observationId = 402;
    candidates.push_back(Candidate(13, 3, 0.60, tie));
    const auto shadow = Verification::BuildShadowRanking(candidates);
    const auto shadowAgain = Verification::BuildShadowRanking(candidates);
    assert(shadow.canonical == shadowAgain.canonical);
    assert(shadow.hash == shadowAgain.hash);
    assert(shadow.candidates[0].rankingMemberId == 12);
    assert(shadow.candidates[1].rankingMemberId == 13);
    assert(shadow.candidates[2].rankingMemberId == 11);
    assert(shadow.candidates[0].profitabilitySign ==
           Verification::ProfitabilitySign::positive);
    assert(shadow.candidates[2].profitabilitySign ==
           Verification::ProfitabilitySign::negative);
    assert(shadow.candidates[2].inferenceAccuracy == 0.99);
    for (const auto& candidate : shadow.candidates)
    {
        if (candidate.rankingMemberId == 11) assert(candidate.currentRank == 1);
        if (candidate.rankingMemberId == 12) assert(candidate.currentRank == 2);
        if (candidate.rankingMemberId == 13) assert(candidate.currentRank == 3);
    }

    auto frozen = Frozen(valid);
    auto enforced = Verification::EnforceFrozenCampaignEvidence(
        valid.finalModelId, frozen, valid);
    assert(enforced.state == Verification::EvidenceState::valid);
    frozen.observationIdentityHash = Profitability::DeterministicHash("later");
    enforced = Verification::EnforceFrozenCampaignEvidence(
        valid.finalModelId, frozen, valid);
    assert(enforced.state == Verification::EvidenceState::invalidProvenance);
    assert(enforced.reason ==
           "candidate_frozen_profitability_evidence_mismatch");

    assert(Verification::kLiveProfitabilityRankingWeight == 0.0);
    assert(Verification::kLiveProfitabilityScoreContribution == 0.0);
    const auto gate = Verification::EvaluateReadinessGate(true, true, shadow);
    assert(gate.profitabilitySoftwareReady);
    assert(gate.campaignProfitabilityContractReady);
    assert(gate.profitabilityShadowRankingReady);
    assert(!gate.activationPerformed);
    assert(gate.action == Verification::ReadinessAction::
        readyForShadowValidation);

    const auto invalidGate = Verification::EvaluateReadinessGate(
        true, true, Verification::BuildShadowRanking({
            Candidate(20, 1, 0.9, enforced)}));
    assert(!invalidGate.campaignProfitabilityContractReady);
    assert(!invalidGate.profitabilityShadowRankingReady);
    assert(!invalidGate.activationPerformed);
    assert(invalidGate.action ==
           Verification::ReadinessAction::blockedEvidenceContract);
}
