#pragma once

#include "InferenceProfitability.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

class Tensor;

namespace EA
{
class LSTM;

namespace InferenceEvaluationFacts
{

inline constexpr std::size_t kClassCount = 3;
using ConfusionMatrix =
    std::array<std::array<std::size_t, kClassCount>, kClassCount>;

// All values that formerly came from mutable evaluation globals are explicit.
// The tensor and model are borrowed only for this call; EvaluationFacts owns
// every result it exposes after the call returns.
struct Request
{
    std::size_t windowSize = 0;
    std::size_t predictionHorizon = 0;
    float thresholdLogret = 0.0f;
    std::size_t logicalOutputStartIndex = 0;
    bool captureStrategyDecisions = false;
};

struct AcceptanceSummary
{
    bool acceptModel = false;
    std::string rejectReason = "none";
    std::array<double, kClassCount> predFrac {0.0, 0.0, 0.0};
    std::array<double, kClassCount> actualFrac {0.0, 0.0, 0.0};
    std::array<double, kClassCount> precision {0.0, 0.0, 0.0};
    std::array<double, kClassCount> recall {0.0, 0.0, 0.0};
};

struct StrategyDecision
{
    std::size_t windowStartRow = 0;
    std::array<float, kClassCount> probabilities {0.0f, 0.0f, 0.0f};
};

struct EvaluationFacts
{
    double accuracy = 0.0;
    std::size_t correctLog = 0;
    std::size_t actedLog = 0;
    std::size_t windowCount = 0;
    double absoluteRelativeMoveErrorSum = 0.0;
    std::size_t correctDirection = 0;
    std::size_t actedDirection = 0;
    ConfusionMatrix confusion {};
    AcceptanceSummary acceptance;
    InferenceProfitability::Statistics profitability;
    std::string profitabilitySourceContentHash;
    std::vector<StrategyDecision> strategyDecisions;
};

inline AcceptanceSummary ComputeAcceptanceSummary(const ConfusionMatrix& confusion)
{
    constexpr double acceptNeutralMax = 0.60;
    constexpr double acceptMinDown = 0.15;
    constexpr double acceptMinUp = 0.15;
    AcceptanceSummary summary;
    std::array<std::size_t, kClassCount> actualCounts {0, 0, 0};
    std::array<std::size_t, kClassCount> predictedCounts {0, 0, 0};
    std::size_t total = 0;
    for (std::size_t actual = 0; actual < kClassCount; ++actual)
        for (std::size_t predicted = 0; predicted < kClassCount; ++predicted)
        {
            actualCounts[actual] += confusion[actual][predicted];
            predictedCounts[predicted] += confusion[actual][predicted];
            total += confusion[actual][predicted];
        }
    for (std::size_t klass = 0; klass < kClassCount; ++klass)
    {
        summary.predFrac[klass] = total ?
            static_cast<double>(predictedCounts[klass]) / total : 0.0;
        summary.actualFrac[klass] = total ?
            static_cast<double>(actualCounts[klass]) / total : 0.0;
        summary.precision[klass] = predictedCounts[klass] ?
            static_cast<double>(confusion[klass][klass]) /
                predictedCounts[klass] : 0.0;
        summary.recall[klass] = actualCounts[klass] ?
            static_cast<double>(confusion[klass][klass]) /
                actualCounts[klass] : 0.0;
    }
    const auto appendReason = [&](const char* reason)
    {
        if (summary.rejectReason == "none")
            summary.rejectReason = reason;
        else
            summary.rejectReason += std::string{";"} + reason;
    };
    if (summary.predFrac[1] > acceptNeutralMax)
        appendReason("pred_neutral_gt_0.60");
    if (summary.predFrac[0] < acceptMinDown)
        appendReason("pred_down_lt_0.15");
    if (summary.predFrac[2] < acceptMinUp)
        appendReason("pred_up_lt_0.15");
    summary.acceptModel = summary.rejectReason == "none";
    return summary;
}

// This numerical evaluation boundary has no database, transaction, CLI,
// scheduler, artifact, or logging dependency.
EvaluationFacts Evaluate(EA::LSTM& model,
                         const ::Tensor& tensor,
                         const Request& request);

} // namespace InferenceEvaluationFacts
} // namespace EA
