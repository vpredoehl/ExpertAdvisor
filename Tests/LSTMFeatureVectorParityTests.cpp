#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "../Headers/ModelInputContract.hpp"
#include "../Headers/ReturnFeatureHistory.hpp"

namespace
{

constexpr float kFeatureScale = 1000.0f;

bool IsZero(float value)
{
    return value == 0.0f;
}

std::vector<float> BuildModelInputRow(const std::vector<std::vector<float>>& sourceRows,
                                      const std::vector<float>& rawCloses,
                                      std::size_t currentGlobalPosition,
                                      std::size_t modelInputWidth,
                                      const EA::FeatureAblationMask& ablationMask = {})
{
    const auto inputContract = EA::ResolveModelInputContract(
        modelInputWidth,
        sourceRows.at(currentGlobalPosition).size());
    std::vector<float> result(modelInputWidth, -1.0f);
    EA::CopyTensorFeaturesForModelInput(
        result.data(), sourceRows.at(currentGlobalPosition).data(),
        inputContract, ablationMask);
    const std::size_t appended = EA::AppendMultiHorizonReturnFeaturesAtGlobalPosition(
        currentGlobalPosition,
        result.data(),
        inputContract.tensorFeatureCount,
        kFeatureScale,
        [&rawCloses](std::size_t globalPosition)
        {
            return rawCloses.at(globalPosition);
        });
    assert(appended == EA::kModelReturnFeatureCount);
    return result;
}

std::vector<float> BuildTrainingStyleModelInputRow(
    const std::vector<std::vector<float>>& sourceRows,
    const std::vector<float>& rawCloses,
    std::size_t enclosingBatchGlobalStart,
    std::size_t batchLocalRow,
    std::size_t modelInputWidth,
    const EA::FeatureAblationMask& ablationMask = {})
{
    return BuildModelInputRow(sourceRows,
                              rawCloses,
                              enclosingBatchGlobalStart + batchLocalRow,
                              modelInputWidth, ablationMask);
}

std::vector<float> BuildInferenceStyleModelInputRow(
    const std::vector<std::vector<float>>& sourceRows,
    const std::vector<float>& rawCloses,
    std::size_t inferenceWindowGlobalStart,
    std::size_t windowLocalRow,
    std::size_t modelInputWidth,
    const EA::FeatureAblationMask& ablationMask = {})
{
    return BuildModelInputRow(sourceRows,
                              rawCloses,
                              inferenceWindowGlobalStart + windowLocalRow,
                              modelInputWidth, ablationMask);
}

void AssertByteIdentical(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    assert(lhs.size() == rhs.size());
    assert(std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0);
}

} // namespace

int main()
{
    static_assert(EA::kMultiHorizonReturnLookbacks.size() == EA::kModelReturnFeatureCount);
    static_assert(EA::kLegacyModelInputWidth == 36);
    static_assert(EA::kDonchianModelInputWidth == 38);
    static_assert(EA::kSessionPhaseModelInputWidth == 40);
    static_assert(EA::kRelativeTickVolumeModelInputWidth == 41);
    static_assert(EA::kCausalReturnSurpriseModelInputWidth == 42);
    static_assert(EA::kCausalVolatilityRegimeModelInputWidth == 43);
    static_assert(EA::kPreEconomicEventModelInputWidth == 53);
    static_assert(EA::kEconomicEventModelInputWidth == 63);
    static_assert(EA::kCurrentModelInputWidth == 75);

    constexpr std::size_t sourceRowCount = 96;
    std::vector<std::vector<float>> sourceRows(sourceRowCount,
                                                std::vector<float>(feature_size));
    std::vector<float> rawCloses(sourceRowCount);
    for (std::size_t row = 0; row < sourceRowCount; ++row)
    {
        rawCloses[row] = std::pow(1.001f, static_cast<float>(row));
        for (std::size_t col = 0; col < feature_size; ++col)
            sourceRows[row][col] = static_cast<float>(row * 100 + col);
    }

    // Training prebuilds against a batch-local row; inference begins a local
    // window at the same observation. Both must resolve that row to one
    // Tensor-global history coordinate.
    constexpr std::size_t kPreviouslyFailingGlobalStart = 16;
    const auto trainingAtOffset16 = BuildTrainingStyleModelInputRow(
        sourceRows, rawCloses, kPreviouslyFailingGlobalStart, 0,
        EA::kCurrentModelInputWidth);
    const auto inferenceAtOffset16 = BuildInferenceStyleModelInputRow(
        sourceRows, rawCloses, kPreviouslyFailingGlobalStart, 0,
        EA::kCurrentModelInputWidth);
    AssertByteIdentical(trainingAtOffset16, inferenceAtOffset16);
    for (std::size_t col = feature_size; col < EA::kCurrentModelInputWidth; ++col)
        assert(!IsZero(trainingAtOffset16[col]));

    const auto consensusAblation = EA::FeatureAblationMask::Parse(
        std::string{EA::kEconomicEventConsensusAblationMaskText});
    const auto ablatedTraining = BuildTrainingStyleModelInputRow(
        sourceRows, rawCloses, kPreviouslyFailingGlobalStart, 0,
        EA::kCurrentModelInputWidth, consensusAblation);
    const auto ablatedInference = BuildInferenceStyleModelInputRow(
        sourceRows, rawCloses, kPreviouslyFailingGlobalStart, 0,
        EA::kCurrentModelInputWidth, consensusAblation);
    AssertByteIdentical(ablatedTraining, ablatedInference);
    assert(ablatedTraining.size() == EA::kCurrentModelInputWidth);
    for (std::size_t col = 0; col < relevantEventHasConsensusCol; ++col)
        assert(ablatedTraining[col] == trainingAtOffset16[col]);
    for (std::size_t col = relevantEventHasConsensusCol;
         col <= relevantEventConsensusIsRangeCol; ++col)
        assert(IsZero(ablatedTraining[col]));
    for (std::size_t col = releasedEventHasSurpriseCol;
         col < EA::kCurrentModelInputWidth; ++col)
        assert(ablatedTraining[col] == trainingAtOffset16[col]);

    // Cover the availability boundaries before and at 1, 4, 8, and 16 bars,
    // plus a window beginning well after the maximum lookback.
    for (const std::size_t globalPosition : {0u, 1u, 3u, 4u, 7u, 8u, 15u, 16u, 48u})
    {
        const auto training = BuildTrainingStyleModelInputRow(
            sourceRows, rawCloses, globalPosition, 0, EA::kCurrentModelInputWidth);
        const auto inference = BuildInferenceStyleModelInputRow(
            sourceRows, rawCloses, globalPosition, 0, EA::kCurrentModelInputWidth);
        AssertByteIdentical(training, inference);

        for (std::size_t horizonIndex = 0;
             horizonIndex < EA::kMultiHorizonReturnLookbacks.size();
             ++horizonIndex)
        {
            const bool historyAvailable =
                globalPosition >= EA::kMultiHorizonReturnLookbacks[horizonIndex];
            const float value = training[feature_size + horizonIndex];
            assert(historyAvailable ? !IsZero(value) : IsZero(value));
        }
    }

    // The fixed appended order is return-1, return-4, return-8, return-16.
    // With strictly increasing geometric source prices, those four values
    // increase in exactly that order at a position with complete history.
    assert(trainingAtOffset16[feature_size + 0] < trainingAtOffset16[feature_size + 1]);
    assert(trainingAtOffset16[feature_size + 1] < trainingAtOffset16[feature_size + 2]);
    assert(trainingAtOffset16[feature_size + 2] < trainingAtOffset16[feature_size + 3]);

    // Changes after the current global position cannot affect its features.
    const auto beforeFutureMutation = BuildInferenceStyleModelInputRow(
        sourceRows, rawCloses, 32, 0, EA::kCurrentModelInputWidth);
    std::vector<float> futureMutatedCloses = rawCloses;
    futureMutatedCloses[33] = 9999.0f;
    futureMutatedCloses[80] = 0.0001f;
    const auto afterFutureMutation = BuildInferenceStyleModelInputRow(
        sourceRows, futureMutatedCloses, 32, 0, EA::kCurrentModelInputWidth);
    AssertByteIdentical(beforeFutureMutation, afterFutureMutation);

    // The return helper remains unclamped; production applies the existing
    // model-input clamp after feature construction.
    std::vector<float> largeMoveCloses(sourceRowCount);
    for (std::size_t row = 0; row < sourceRowCount; ++row)
        largeMoveCloses[row] = std::pow(1.1f, static_cast<float>(row));
    const auto largeMoveInput = BuildInferenceStyleModelInputRow(
        sourceRows, largeMoveCloses, 32, 0, EA::kCurrentModelInputWidth);
    assert(largeMoveInput[feature_size] > 10.0f);
    assert(std::clamp(largeMoveInput[feature_size], -10.0f, 10.0f) == 10.0f);

    // Both persisted model-input contracts retain their unchanged prefix
    // projection and append the same four return columns.
    const auto legacyTraining = BuildTrainingStyleModelInputRow(
        sourceRows, rawCloses, 48, 0, EA::kLegacyModelInputWidth);
    const auto legacyInference = BuildInferenceStyleModelInputRow(
        sourceRows, rawCloses, 48, 0, EA::kLegacyModelInputWidth);
    AssertByteIdentical(legacyTraining, legacyInference);
    assert(legacyTraining.size() == EA::kLegacyModelInputWidth);
    assert(trainingAtOffset16.size() == EA::kCurrentModelInputWidth);

    return 0;
}
