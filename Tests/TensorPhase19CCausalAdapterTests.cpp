#include "../Headers/ModelInputContract.hpp"
#include "../Headers/PricePoint.hpp"
#include "../Headers/Tensor.hpp"
#include "../Sources/StrategyEvaluationAdapters/TensorPhase19CCausalAdapter.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Adapter = EA::StrategyEvaluationAdapters;
namespace Strategy = EA::StrategyEvaluation;

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled()
{
    return false;
}

namespace
{

Feature Bar(std::int64_t timestamp, float close)
{
    return {close - 0.01f, close, close + 0.02f, close - 0.02f,
            PriceTP{std::chrono::seconds{timestamp}}, 100.0f};
}

Adapter::TensorPhase19CCausalAdapterContext Context()
{
    Adapter::TensorPhase19CCausalAdapterContext context;
    context.inferenceWindowSize = 3;
    context.modelInputWidth = EA::kLegacyModelInputWidth;
    context.semanticLayoutVersion = 1;
    return context;
}

std::string Error(const std::function<void()>& operation)
{
    try { operation(); }
    catch (const std::exception& error) { return error.what(); }
    return {};
}

} // namespace

int main()
{
    Tensor left{"audchfrmp"};
    Tensor right{"audchfrmp"};
    for (int index = 0; index < 3; ++index)
    {
        const auto bar = Bar(1000 + index * 900, 1.0f + index * 0.01f);
        left.Add(bar);
        right.Add(bar);
    }
    left.Add(Bar(3700, 1.03f));
    left.Add(Bar(4600, 1.04f));
    right.Add(Bar(3700, 5.0f));
    right.Add(Bar(4600, 0.1f));

    const std::vector<Adapter::TensorInferenceDecision> decisions{{
        0, {{0.1f, 0.1f, 0.8f}}}};
    const auto leftResult = Adapter::AdaptTensorPhase19CCausalEntryStates(
        left, Context(), decisions);
    const auto rightResult = Adapter::AdaptTensorPhase19CCausalEntryStates(
        right, Context(), decisions);
    assert(leftResult.predictors == rightResult.predictors);
    assert(leftResult.entryStates.size() == 1);
    assert(rightResult.entryStates.size() == 1);
    assert(leftResult.entryStates[0].predictorValues ==
           rightResult.entryStates[0].predictorValues);
    const auto& state = leftResult.entryStates[0];
    assert(state.entrySourceRow == 2);
    assert(state.entryTimestampUnixSeconds == 2800);
    assert(state.predictedClass == 2);
    assert(state.direction == "long");
    assert(state.predictorValues.size() == EA::kLegacyModelInputWidth);
    assert(leftResult.predictors.size() == EA::kLegacyModelInputWidth);
    assert(leftResult.predictors.front().modelInputColumn == 0);
    assert(leftResult.predictors.back().name ==
           "lookback_log_return_16_scaled");
    for (double value : state.predictorValues) assert(std::isfinite(value));

    Strategy::Phase19CCausalEntryState request = state;
    request.predictorValues.clear();
    const auto joined = Adapter::AdaptTensorPhase19CCausalJoinRequests(
        left, Context(), {request});
    assert(joined.entryStates.size() == 1);
    assert(joined.entryStates[0].predictorValues == state.predictorValues);

    request.entryTimestampUnixSeconds += 1;
    assert(Error([&] {
        Adapter::AdaptTensorPhase19CCausalJoinRequests(
            left, Context(), {request});
    }) == "phase19c_entry_timestamp_join_mismatch");

    request = state;
    request.predictorValues.clear();
    request.entrySourceRow = left.RowCount();
    assert(Error([&] {
        Adapter::AdaptTensorPhase19CCausalJoinRequests(
            left, Context(), {request});
    }) == "phase19c_entry_source_row_out_of_bounds");
}
