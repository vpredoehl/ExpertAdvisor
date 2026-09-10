#include "TensorPhase19StateAdapter.hpp"

#include <cstddef>
#include <stdexcept>

namespace EA::StrategyEvaluationAdapters
{

std::vector<StrategyEvaluation::Phase19EntryState>
AdaptTensorPhase19EntryStates(
    const Tensor& tensor,
    const TensorMarketPathAdapterContext& context,
    const std::vector<TensorInferenceDecision>& decisions)
{
    if (context.schemaVersion !=
        kTensorMarketPathAdapterContextSchemaVersion)
        throw std::invalid_argument(
            "unsupported_tensor_market_path_adapter_context_version");
    if (context.inferenceWindowSize == 0)
        throw std::invalid_argument(
            "invalid_tensor_market_path_window_or_horizon");

    std::vector<StrategyEvaluation::Phase19EntryState> states;
    states.reserve(decisions.size());
    const auto& identities =
        StrategyEvaluation::Phase19FeatureIdentities();
    for (std::size_t ordinal = 0; ordinal < decisions.size(); ++ordinal)
    {
        const auto& decision = decisions[ordinal];
        if (decision.windowStartRow >= tensor.RowCount() ||
            context.inferenceWindowSize - 1 >
                tensor.RowCount() - 1 - decision.windowStartRow)
            throw std::invalid_argument(
                "tensor_phase19_decision_row_out_of_bounds");
        const std::size_t decisionRow = decision.windowStartRow +
            context.inferenceWindowSize - 1;
        const auto row = MetaNN::LowerAccess(
            *(tensor.begin() + static_cast<std::ptrdiff_t>(decisionRow)));
        StrategyEvaluation::Phase19EntryState state;
        state.observationOrdinal = ordinal;
        for (std::size_t index = 0; index < identities.size(); ++index)
            state.values[index] = row.RawMemory()[identities[index].tensorColumn];
        states.push_back(state);
    }
    return states;
}

} // namespace EA::StrategyEvaluationAdapters
