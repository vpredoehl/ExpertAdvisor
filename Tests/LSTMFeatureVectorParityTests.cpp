#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "ReturnFeatureHistory.hpp"

namespace
{

constexpr std::size_t kReturnFeatureCount = EA::kMultiHorizonReturnLookbacks.size();
constexpr std::size_t kTensorFeatureCount = 32;
constexpr std::size_t kModelInputWidth = kTensorFeatureCount + kReturnFeatureCount;
constexpr std::size_t kWindowRows = 64;

using FeatureWindow = std::vector<float>;

struct Ohlc
{
    float open;
    float high;
    float low;
    float close;
};

FeatureWindow BuildTrainingInputWindow(const std::vector<std::array<float, kTensorFeatureCount>>& sourceRows,
                                       const std::vector<float>& rawCloses,
                                       std::size_t enclosingBatchGlobalStart,
                                       std::size_t windowStartWithinBatch)
{
    FeatureWindow assembled(kWindowRows * kModelInputWidth);
    for (std::size_t row = 0; row < kWindowRows; ++row)
    {
        const std::size_t batchLocalRow = windowStartWithinBatch + row;
        const std::size_t globalPosition = enclosingBatchGlobalStart + batchLocalRow;
        const std::size_t appended = EA::AssembleLstmModelInputRowAtGlobalPosition(
            assembled.data() + row * kModelInputWidth,
            sourceRows.at(globalPosition).data(),
            kTensorFeatureCount,
            globalPosition,
            1000.0f,
            [&rawCloses](std::size_t position) { return rawCloses.at(position); });
        assert(appended == kReturnFeatureCount);
    }
    return assembled;
}

FeatureWindow BuildInferenceInputWindow(const std::vector<std::array<float, kTensorFeatureCount>>& sourceRows,
                                        const std::vector<float>& rawCloses,
                                        std::size_t inferenceWindowGlobalStart)
{
    FeatureWindow assembled(kWindowRows * kModelInputWidth);
    for (std::size_t row = 0; row < kWindowRows; ++row)
    {
        const std::size_t globalPosition = inferenceWindowGlobalStart + row;
        const std::size_t appended = EA::AssembleLstmModelInputRowAtGlobalPosition(
            assembled.data() + row * kModelInputWidth,
            sourceRows.at(globalPosition).data(),
            kTensorFeatureCount,
            globalPosition,
            1000.0f,
            [&rawCloses](std::size_t position) { return rawCloses.at(position); });
        assert(appended == kReturnFeatureCount);
    }
    return assembled;
}

void AssertByteIdentical(const FeatureWindow& lhs, const FeatureWindow& rhs)
{
    assert(lhs.size() == rhs.size());
    assert(std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0);
}

float ReturnAt(const FeatureWindow& window, std::size_t row, std::size_t horizonIndex)
{
    return window.at(row * kModelInputWidth + kTensorFeatureCount + horizonIndex);
}

} // namespace

int main()
{
    static_assert(kTensorFeatureCount == 32);
    static_assert(kReturnFeatureCount == 4);
    static_assert(kModelInputWidth == 36);

    constexpr std::size_t sourceRowCount = 256;
    std::vector<Ohlc> marketRows(sourceRowCount);
    std::vector<std::array<float, kTensorFeatureCount>> sourceRows(sourceRowCount);
    std::vector<float> rawCloses(sourceRowCount);
    for (std::size_t row = 0; row < sourceRowCount; ++row)
    {
        // Keep every deterministic value inside the existing [-10, 10]
        // input clamp so byte equality here is equality of the exact values
        // presented to the LSTM.
        const float close = std::pow(1.0001f, static_cast<float>(row));
        marketRows[row] = Ohlc{close * 0.9998f, close * 1.0004f,
                               close * 0.9996f, close};
        rawCloses[row] = marketRows[row].close;
        for (std::size_t col = 0; col < kTensorFeatureCount; ++col)
            sourceRows[row][col] =
                (marketRows[row].open + marketRows[row].high +
                 marketRows[row].low + marketRows[row].close) *
                static_cast<float>((col % 8) + 1) / 100.0f;
    }

    // These are the coordinate translations used by CalculateBatch
    // (enclosing-batch-local) and PredictNext* (window-local). The compared
    // assembly function is the production one invoked by both paths.
    for (const std::size_t start : {0u, 1u, 4u, 8u, 16u, 80u})
    {
        const auto training = BuildTrainingInputWindow(sourceRows, rawCloses, 0, start);
        const auto inference = BuildInferenceInputWindow(sourceRows, rawCloses, start);
        AssertByteIdentical(training, inference);
    }

    const auto atStart0 = BuildInferenceInputWindow(sourceRows, rawCloses, 0);
    for (std::size_t horizon = 0; horizon < kReturnFeatureCount; ++horizon)
        assert(ReturnAt(atStart0, 0, horizon) == 0.0f);

    // At a window boundary, only unavailable true Tensor history is zero.
    // Available pre-window history must not be reset by the inference slice.
    const auto atStart1 = BuildInferenceInputWindow(sourceRows, rawCloses, 1);
    assert(ReturnAt(atStart1, 0, 0) != 0.0f);
    assert(ReturnAt(atStart1, 0, 1) == 0.0f);
    assert(ReturnAt(atStart1, 0, 2) == 0.0f);
    assert(ReturnAt(atStart1, 0, 3) == 0.0f);

    const auto atStart4 = BuildInferenceInputWindow(sourceRows, rawCloses, 4);
    assert(ReturnAt(atStart4, 0, 0) != 0.0f);
    assert(ReturnAt(atStart4, 0, 1) != 0.0f);
    assert(ReturnAt(atStart4, 0, 2) == 0.0f);
    assert(ReturnAt(atStart4, 0, 3) == 0.0f);

    const auto atStart8 = BuildInferenceInputWindow(sourceRows, rawCloses, 8);
    assert(ReturnAt(atStart8, 0, 0) != 0.0f);
    assert(ReturnAt(atStart8, 0, 1) != 0.0f);
    assert(ReturnAt(atStart8, 0, 2) != 0.0f);
    assert(ReturnAt(atStart8, 0, 3) == 0.0f);

    const auto atStart16 = BuildInferenceInputWindow(sourceRows, rawCloses, 16);
    for (std::size_t horizon = 0; horizon < kReturnFeatureCount; ++horizon)
        assert(ReturnAt(atStart16, 0, horizon) != 0.0f);

    // Future changes are not read by a causal return feature.
    const auto beforeFutureMutation = BuildInferenceInputWindow(sourceRows, rawCloses, 80);
    auto futureMutatedCloses = rawCloses;
    futureMutatedCloses[144] = 9999.0f;
    const auto afterFutureMutation = BuildInferenceInputWindow(sourceRows, futureMutatedCloses, 80);
    AssertByteIdentical(beforeFutureMutation, afterFutureMutation);

    return 0;
}
