#include "InferenceEvaluationFacts.hpp"

#include "../Headers/LSTM.hpp"
#include "../Headers/TargetLabel.hpp"
#include "../Headers/Tensor.hpp"
#include "StrategyEvaluationCore/StrategyEvaluation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace EA::InferenceEvaluationFacts
{
namespace
{
void EvaluateClassificationBatch(
    EA::LSTM& model,
    const ::Tensor& tensor,
    const Window& batch,
    const Request& request,
    InferenceProfitability::Accumulator& profitability,
    EvaluationFacts& facts)
{
    for (auto it = batch.begin();
         it + static_cast<std::ptrdiff_t>(request.windowSize - 1 +
                                           request.predictionHorizon) <
             batch.end();
         ++it)
    {
        const auto window = Window{
            it, it + static_cast<std::ptrdiff_t>(request.windowSize)};
        const auto probabilities =
            model.PredictNextDirectionProbs(window, /*resetState=*/true);
        const int predicted = StrategyEvaluation::PredictedClassForProbabilities(
            StrategyEvaluation::PredictionProbabilities{probabilities});
        const auto label = BuildLookaheadClassInfo(
            tensor, it, request.windowSize, request.predictionHorizon,
            request.thresholdLogret);
        const int actual = label.assignedClass;

        ++facts.confusion[static_cast<std::size_t>(actual)]
                          [static_cast<std::size_t>(predicted)];
        ++facts.windowCount;
        if (predicted == actual)
            ++facts.correctDirection;
        profitability.Observe(predicted, label.closeT, label.targetClose);

        if (request.captureStrategyDecisions)
        {
            facts.strategyDecisions.push_back({
                static_cast<std::size_t>(it - tensor.begin()), probabilities});
        }
    }
}

void EvaluateRegressionBatch(EA::LSTM& model,
                             const ::Tensor& tensor,
                             const Window& batch,
                             const Request& request,
                             EvaluationFacts& facts)
{
    const auto predictedLogReturns =
        model.RollingPredictNextLogReturn(batch, /*resetAtStart=*/true);
    std::vector<float> predictedRelative;
    predictedRelative.reserve(predictedLogReturns.size());
    for (const float value : predictedLogReturns)
        predictedRelative.push_back(std::exp(value) - 1.0f);

    std::vector<float> actualLogReturns;
    std::vector<float> actualRelative;
    actualLogReturns.reserve(predictedRelative.size());
    actualRelative.reserve(predictedRelative.size());
    for (auto it = batch.begin();
         it + static_cast<std::ptrdiff_t>(request.windowSize - 1 +
                                           request.predictionHorizon) <
             batch.end();
         ++it)
    {
        const auto last = it + static_cast<std::ptrdiff_t>(request.windowSize - 1);
        const auto target = last +
            static_cast<std::ptrdiff_t>(request.predictionHorizon);
        const float close = tensor.RawCloseAtIterator(last);
        const float terminalClose = tensor.RawCloseAtIterator(target);
        const float actual =
            (std::isfinite(close) && std::isfinite(terminalClose) &&
             close > 0.0f && terminalClose > 0.0f)
                ? std::log(terminalClose / close)
                : 0.0f;
        actualLogReturns.push_back(actual);
        actualRelative.push_back(std::exp(actual) - 1.0f);
    }

    if (predictedRelative.size() != actualRelative.size())
        throw std::runtime_error("inference_regression_prediction_alignment_mismatch");
    if (predictedRelative.empty())
        return;

    constexpr float kActedLogThreshold = 2e-4f;
    constexpr float kPredictedMoveThreshold = 1e-4f;
    constexpr float kActualMoveThreshold = 1e-4f;
    for (std::size_t index = 0; index < predictedRelative.size(); ++index)
    {
        const float predictedLogReturn = predictedLogReturns[index];
        const float actualLogReturn = actualLogReturns[index];
        if (std::fabs(predictedLogReturn) >= kActedLogThreshold)
        {
            ++facts.actedLog;
            if ((predictedLogReturn >= 0.0f) == (actualLogReturn >= 0.0f))
                ++facts.correctLog;
        }

        facts.absoluteRelativeMoveErrorSum += std::abs(
            static_cast<double>(predictedRelative[index]) -
            static_cast<double>(actualRelative[index]));
        if (std::fabs(predictedRelative[index]) < kPredictedMoveThreshold ||
            std::fabs(actualRelative[index]) < kActualMoveThreshold)
            continue;
        ++facts.actedDirection;
        if ((predictedRelative[index] >= 0.0f) ==
            (actualRelative[index] >= 0.0f))
            ++facts.correctDirection;
    }
    facts.windowCount += predictedRelative.size();
}
} // namespace

EvaluationFacts Evaluate(EA::LSTM& model,
                         const ::Tensor& tensor,
                         const Request& request)
{
    if (request.windowSize == 0 || request.predictionHorizon == 0)
        throw std::invalid_argument("inference_evaluation_invalid_window_or_horizon");

    EvaluationFacts facts;
    InferenceProfitability::Accumulator profitability;
    tensor.ForEachBatchFrom(request.logicalOutputStartIndex,
                            [&](const auto batch)
    {
        if (model.targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
            EvaluateClassificationBatch(model, tensor, batch, request,
                                        profitability, facts);
        else
            EvaluateRegressionBatch(model, tensor, batch, request, facts);
    });

    if (model.targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
    {
        if (profitability.statistics().predictionCount != facts.windowCount)
            throw std::runtime_error("inference_profitability_prediction_count_mismatch");
        facts.accuracy = facts.windowCount
            ? static_cast<double>(facts.correctDirection) /
                  static_cast<double>(facts.windowCount)
            : 0.0;
        facts.actedLog = facts.windowCount;
        facts.correctLog = facts.correctDirection;
        facts.actedDirection = facts.windowCount;
        facts.acceptance = ComputeAcceptanceSummary(facts.confusion);
        facts.profitability = profitability.statistics();
        facts.profitabilitySourceContentHash = profitability.SourceContentHash();
    }
    else
    {
        facts.accuracy = facts.actedLog
            ? static_cast<double>(facts.correctLog) /
                  static_cast<double>(facts.actedLog)
            : 0.0;
    }
    return facts;
}
} // namespace EA::InferenceEvaluationFacts
