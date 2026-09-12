#pragma once

#include "TensorMarketPathAdapter.hpp"
#include "../StrategyEvaluationCore/Phase19CCausalPathPredictability.hpp"
#include "../../Headers/FeatureAblation.hpp"

#include <cstddef>
#include <vector>

namespace EA::StrategyEvaluationAdapters
{

struct TensorPhase19CCausalAdapterContext
{
    std::size_t inferenceWindowSize = 0;
    std::size_t modelInputWidth = 0;
    int semanticLayoutVersion = 0;
    EA::FeatureAblationMask featureAblationMask;
};

struct TensorPhase19CCausalAdapterResult
{
    std::vector<StrategyEvaluation::Phase19CPredictorIdentity> predictors;
    std::vector<StrategyEvaluation::Phase19CCausalEntryState> entryStates;
    std::string featureAblationIdentity;
};

TensorPhase19CCausalAdapterResult AdaptTensorPhase19CCausalEntryStates(
    const Tensor& tensor,
    const TensorPhase19CCausalAdapterContext& context,
    const std::vector<TensorInferenceDecision>& decisions);

TensorPhase19CCausalAdapterResult AdaptTensorPhase19CCausalJoinRequests(
    const Tensor& tensor,
    const TensorPhase19CCausalAdapterContext& context,
    std::vector<StrategyEvaluation::Phase19CCausalEntryState> requests);

} // namespace EA::StrategyEvaluationAdapters
