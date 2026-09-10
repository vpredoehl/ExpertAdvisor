#include "../Headers/PricePoint.hpp"
#include "../Headers/Tensor.hpp"
#include "../Sources/StrategyEvaluationAdapters/TensorPhase19StateAdapter.hpp"

#include <cassert>
#include <chrono>

namespace Adapter = EA::StrategyEvaluationAdapters;

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled()
{
    return false;
}

namespace
{

Feature Bar(std::int64_t timestamp, float close)
{
    Feature value;
    value.time = PriceTP{std::chrono::seconds{timestamp}};
    value.open = close - 0.01f;
    value.high = close + 0.02f;
    value.low = close - 0.02f;
    value.close = close;
    value.tickVolume = 100.0f;
    return value;
}

Adapter::TensorMarketPathAdapterContext Context()
{
    Adapter::TensorMarketPathAdapterContext context;
    context.inferenceWindowSize = 3;
    context.predictionHorizon = 2;
    return context;
}

} // namespace

int main()
{
    Tensor left{"audchfrmp"};
    Tensor right{"audchfrmp"};
    for (int index = 0; index < 3; ++index)
    {
        left.Add(Bar(1000 + index * 900, 1.0f + index * 0.01f));
        right.Add(Bar(1000 + index * 900, 1.0f + index * 0.01f));
    }
    // These bars occur strictly after the decision row and differ radically.
    left.Add(Bar(3700, 1.03f));
    left.Add(Bar(4600, 1.04f));
    right.Add(Bar(3700, 5.0f));
    right.Add(Bar(4600, 0.1f));

    const std::vector<Adapter::TensorInferenceDecision> decisions{{
        0, {{0.1f, 0.1f, 0.8f}}}};
    const auto leftState = Adapter::AdaptTensorPhase19EntryStates(
        left, Context(), decisions);
    const auto rightState = Adapter::AdaptTensorPhase19EntryStates(
        right, Context(), decisions);
    assert(leftState == rightState);
    assert(leftState.size() == 1);
    assert(leftState[0].observationOrdinal == 0);
    for (double value : leftState[0].values)
        assert(std::isfinite(value));
    return 0;
}
