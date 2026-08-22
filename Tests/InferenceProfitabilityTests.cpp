#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "../Sources/InferenceProfitability.hpp"
#include "../Sources/InferenceProfitabilityRepository.hpp"

using namespace EA::InferenceProfitability;

namespace
{

bool NearlyEqual(double lhs, double rhs, double tolerance = 1e-12)
{
    return std::fabs(lhs - rhs) <= tolerance;
}

std::string ErrorFrom(const auto& operation)
{
    try
    {
        operation();
    }
    catch (const std::exception& error)
    {
        return error.what();
    }
    return {};
}

ObservationRequest ValidFinalRequest(const Statistics& statistics,
                                     const std::string& sourceHash)
{
    ObservationRequest request;
    request.provenance.experimentId = 10;
    request.provenance.modelId = 20;
    request.provenance.inferenceEvalResultId = 30;
    request.provenance.scope = Scope::finalInference;
    request.provenance.inferenceStart = "2025-01-01";
    request.provenance.inferenceEnd = "2026-01-01";
    request.statistics = statistics;
    request.sourceContentHash = sourceHash;
    return request;
}

} // namespace

int main()
{
    Accumulator zeroActionable;
    zeroActionable.Observe(kNeutralClass, 100.0f, 110.0f);
    zeroActionable.Observe(
        kUpClass, std::numeric_limits<float>::quiet_NaN(), 110.0f);
    const Statistics zero = zeroActionable.statistics();
    assert(zero.predictionCount == 2);
    assert(zero.actionableCount == 0);
    assert(zero.winningActionableCount == 0);
    assert(zero.losingActionableCount == 0);
    assert(zero.aggregateTerminalHorizonLogReturnSum == 0.0);
    assert(!zero.AverageTerminalHorizonLogReturnPerActionablePrediction());

    Accumulator positive;
    positive.Observe(kUpClass, 100.0f, 110.0f);
    positive.Observe(kDownClass, 100.0f, 90.0f);
    const Statistics gains = positive.statistics();
    const double upGain = std::log(1.1);
    const double downGain = -std::log(0.9);
    assert(gains.predictionCount == 2);
    assert(gains.actionableCount == 2);
    assert(gains.winningActionableCount == 2);
    assert(gains.losingActionableCount == 0);
    assert(NearlyEqual(
        gains.grossPositiveTerminalHorizonLogReturnSum,
        upGain + downGain));
    assert(NearlyEqual(
        gains.aggregateTerminalHorizonLogReturnSum,
        upGain + downGain));

    Accumulator negative;
    negative.Observe(kUpClass, 100.0f, 90.0f);
    negative.Observe(kDownClass, 100.0f, 110.0f);
    const Statistics losses = negative.statistics();
    assert(losses.winningActionableCount == 0);
    assert(losses.losingActionableCount == 2);
    assert(losses.grossNegativeTerminalHorizonLogReturnSum < 0.0);
    assert(losses.aggregateTerminalHorizonLogReturnSum < 0.0);

    Accumulator mixed;
    mixed.Observe(kNeutralClass, 100.0f, 150.0f);
    mixed.Observe(kUpClass, 100.0f, 120.0f);
    mixed.Observe(kDownClass, 100.0f, 120.0f);
    mixed.Observe(kUpClass, 100.0f, 100.0f);
    const Statistics mixedStats = mixed.statistics();
    assert(mixedStats.predictionCount == 4);
    assert(mixedStats.actionableCount == 3);
    assert(mixedStats.winningActionableCount == 1);
    assert(mixedStats.losingActionableCount == 1);
    assert(NearlyEqual(mixedStats.aggregateTerminalHorizonLogReturnSum, 0.0));
    assert(mixedStats.AverageTerminalHorizonLogReturnPerActionablePrediction());
    assert(NearlyEqual(
        *mixedStats.AverageTerminalHorizonLogReturnPerActionablePrediction(),
        0.0));

    Accumulator identical;
    identical.Observe(kNeutralClass, 100.0f, 150.0f);
    identical.Observe(kUpClass, 100.0f, 120.0f);
    identical.Observe(kDownClass, 100.0f, 120.0f);
    identical.Observe(kUpClass, 100.0f, 100.0f);
    assert(identical.SourceContentHash() == mixed.SourceContentHash());
    Accumulator changedSource;
    changedSource.Observe(kNeutralClass, 100.0f, 151.0f);
    changedSource.Observe(kUpClass, 100.0f, 120.0f);
    changedSource.Observe(kDownClass, 100.0f, 120.0f);
    changedSource.Observe(kUpClass, 100.0f, 100.0f);
    assert(changedSource.SourceContentHash() != mixed.SourceContentHash());

    const ObservationRequest finalRequest =
        ValidFinalRequest(mixedStats, mixed.SourceContentHash());
    const std::string finalIdentity =
        BuildObservationIdentityCanonical(finalRequest);
    assert(finalIdentity.find("inference_scope=5:final") != std::string::npos);
    assert(finalIdentity.find("checkpoint_eval_id=4:NULL") !=
           std::string::npos);

    ObservationRequest checkpointRequest = finalRequest;
    checkpointRequest.provenance.scope = Scope::checkpointInference;
    checkpointRequest.provenance.checkpointEvalId = 40;
    assert(BuildObservationIdentityCanonical(checkpointRequest) != finalIdentity);

    ObservationRequest invalidFinal = finalRequest;
    invalidFinal.provenance.checkpointEvalId = 40;
    assert(ErrorFrom([&] { BuildObservationIdentityCanonical(invalidFinal); }) ==
           "final_profitability_has_checkpoint_identity");
    ObservationRequest invalidCheckpoint = finalRequest;
    invalidCheckpoint.provenance.scope = Scope::checkpointInference;
    assert(ErrorFrom([&] {
        BuildObservationIdentityCanonical(invalidCheckpoint);
    }) == "checkpoint_profitability_missing_checkpoint_identity");

    ObservationRequest changedMetric = finalRequest;
    changedMetric.metricDefinitionCanonical += ";test_semantic_revision=2";
    assert(BuildObservationIdentityCanonical(changedMetric) != finalIdentity);
    ObservationRequest changedEvidence = finalRequest;
    changedEvidence.sourceContentHash = changedSource.SourceContentHash();
    assert(BuildObservationIdentityCanonical(changedEvidence) != finalIdentity);

    assert(MetricDefinitionHash().starts_with("fnv1a64:"));
    assert(std::string{kMetricDefinitionCanonical}.find(
               "transaction_costs=not_modeled") != std::string::npos);
    return 0;
}
