#include <algorithm>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Headers/PricePoint.hpp"
#include "../Headers/Tensor.hpp"
#include "../Sources/InferenceProfitability.hpp"
#include "../Sources/StrategyEvaluationAdapters/TensorMarketPathAdapter.hpp"

namespace Adapter = EA::StrategyEvaluationAdapters;
namespace Profitability = EA::InferenceProfitability;
namespace Strategy = EA::StrategyEvaluation;

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled()
{
    return false;
}

namespace
{

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

Feature Bar(std::int64_t timestamp,
            float open,
            float high,
            float low,
            float close)
{
    Feature feature;
    feature.time = PriceTP{std::chrono::seconds{timestamp}};
    feature.open = open;
    feature.high = high;
    feature.low = low;
    feature.close = close;
    feature.tickVolume = 1.0f;
    return feature;
}

Adapter::TensorMarketPathAdapterContext Context()
{
    Adapter::TensorMarketPathAdapterContext context;
    context.modelId = 4242;
    context.inferenceScientificIdentityCanonical =
        "inference_scientific_identity_v1;model=4242;window=3;horizon=2;";
    context.inferenceScientificIdentityHash = Profitability::DeterministicHash(
        context.inferenceScientificIdentityCanonical);
    context.evaluationStart = "2025-01-01";
    context.evaluationEnd = "2025-01-02";
    context.inferenceWindowSize = 3;
    context.predictionHorizon = 2;
    context.metricDefinitionCanonical =
        Strategy::kFixedStopMetricDefinitionCanonical;
    context.metricDefinitionHash = Strategy::FixedStopMetricDefinitionHash();
    return context;
}

} // namespace

int main()
{
    Tensor tensor{"eurusdrmp"};
    tensor.Add(Bar(1000, 99.0f, 101.0f, 98.0f, 100.0f));
    tensor.Add(Bar(1900, 100.0f, 102.0f, 99.0f, 101.0f));
    tensor.Add(Bar(2800, 101.0f, 103.0f, 100.0f, 102.0f));
    tensor.Add(Bar(3700, 102.0f, 104.0f, 95.0f, 103.0f));
    tensor.Add(Bar(4600, 103.0f, 105.0f, 94.0f, 104.0f));
    tensor.Add(Bar(5500, 104.0f, 106.0f, 93.0f, 105.0f));

    const std::vector<Adapter::TensorInferenceDecision> decisions = {
        {0, {{0.1f, 0.2f, 0.7f}}},
        {1, {{0.7f, 0.2f, 0.1f}}}};
    const auto first = Adapter::AdaptTensorMarketPath(
        tensor, Context(), decisions);
    const auto repeated = Adapter::AdaptTensorMarketPath(
        tensor, Context(), decisions);
    assert(first.Canonical() == repeated.Canonical());
    assert(first.Hash() == repeated.Hash());
    assert(first.Provenance().symbol == "eurusdrmp");
    assert(first.Provenance().priceDomain == "ask");
    assert(first.Provenance().barIntervalSeconds == 900);
    assert(first.Provenance().adapterFamily ==
           Adapter::kTensorMarketPathAdapterFamily);

    const auto& observations = first.Observations();
    assert(observations.size() == 2);
    assert(observations[0].inferenceWindowStartRow == 0);
    assert(observations[0].decisionRow == 2);
    assert(observations[0].terminalRow == 4);
    assert(observations[0].decisionTimestampUnixSeconds == 2800);
    assert(observations[0].terminalTimestampUnixSeconds == 4600);
    assert(observations[0].decisionClose == 102.0f);
    assert(observations[0].terminalClose == 104.0f);
    assert(observations[0].predictedClass == Profitability::kUpClass);
    assert(observations[0].marketPath.size() == 2);
    assert(observations[0].marketPath[0].sourceRow == 3);
    assert(observations[0].marketPath[0].timestampUnixSeconds == 3700);
    assert(observations[0].marketPath[0].open == 102.0f);
    assert(observations[0].marketPath[0].high == 104.0f);
    assert(observations[0].marketPath[0].low == 95.0f);
    assert(observations[0].marketPath[0].close == 103.0f);
    assert(observations[0].marketPath[1].sourceRow == 4);
    assert(observations[1].decisionRow == 3);
    assert(observations[1].terminalRow == 5);
    assert(observations[1].predictedClass == Profitability::kDownClass);

    auto reversed = decisions;
    std::reverse(reversed.begin(), reversed.end());
    assert(ErrorFrom([&] {
        Adapter::AdaptTensorMarketPath(tensor, Context(), reversed);
    }) == "unordered_market_path_observations");

    auto outOfBounds = decisions;
    outOfBounds[1].windowStartRow = 2;
    assert(ErrorFrom([&] {
        Adapter::AdaptTensorMarketPath(tensor, Context(), outOfBounds);
    }) == "tensor_market_path_window_out_of_bounds");

    auto unsupported = Context();
    unsupported.schemaVersion = 2;
    assert(ErrorFrom([&] {
        Adapter::AdaptTensorMarketPath(tensor, unsupported, decisions);
    }) == "unsupported_tensor_market_path_adapter_context_version");

    return 0;
}
