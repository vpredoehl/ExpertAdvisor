#pragma once

#include "TensorMarketPathAdapter.hpp"
#include "../StrategyEvaluationCore/Phase19StateInteractionAnalysis.hpp"

namespace EA::StrategyEvaluationAdapters
{

// Extracts the predeclared Phase 19 state from the same completed decision
// Tensor row used by inference. It never reads a subsequent market-path row.
std::vector<StrategyEvaluation::Phase19EntryState>
AdaptTensorPhase19EntryStates(
    const Tensor& tensor,
    const TensorMarketPathAdapterContext& context,
    const std::vector<TensorInferenceDecision>& decisions);

} // namespace EA::StrategyEvaluationAdapters
