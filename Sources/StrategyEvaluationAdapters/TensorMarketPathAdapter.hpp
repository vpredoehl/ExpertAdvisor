#pragma once

#include "../StrategyEvaluationCore/StrategyEvaluation.hpp"
#include "../../Headers/Tensor.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace EA::StrategyEvaluationAdapters
{

inline constexpr int kTensorMarketPathAdapterContextSchemaVersion = 1;
inline constexpr const char* kTensorMarketPathAdapterFamily =
    "authoritative_tensor_raw_ohlc";
inline constexpr int kTensorMarketPathAdapterVersion = 1;

struct TensorMarketPathAdapterContext
{
    int schemaVersion = kTensorMarketPathAdapterContextSchemaVersion;
    long long modelId = 0;
    std::string inferenceScientificIdentityCanonical;
    std::string inferenceScientificIdentityHash;
    std::string evaluationStart;
    std::string evaluationEnd;
    std::size_t inferenceWindowSize = 0;
    std::size_t predictionHorizon = 0;
    std::string metricDefinitionCanonical;
    std::string metricDefinitionHash;
};

// Each row is the start of one window actually evaluated by inference. Model
// probabilities are decision-time data; predicted class is derived with the
// same strict-max/tie-neutral rule used by production inference.
struct TensorInferenceDecision
{
    std::size_t windowStartRow = 0;
    StrategyEvaluation::PredictionProbabilities probabilities;
};

StrategyEvaluation::AuthoritativeMarketPath AdaptTensorMarketPath(
    const Tensor& tensor,
    const TensorMarketPathAdapterContext& context,
    const std::vector<TensorInferenceDecision>& decisions);

} // namespace EA::StrategyEvaluationAdapters
