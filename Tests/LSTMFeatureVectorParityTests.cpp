#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "../Headers/ModelInputContract.hpp"
#include "../Headers/ModelInputExpansion.hpp"
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
    static_assert(EA::kCurrentModelInputWidth == 103);
    static_assert(EA::kModelInputSemanticLayoutVersion == 9);
    static_assert(feature_size == 99);
    static_assert(tg4InnerBreakAnyCol == 73);
    static_assert(tg4SourceTg3StructurallyEligibleCol == 74);
    static_assert(tg4SourceTg3ConfluentCol == 75);

    constexpr std::size_t sourceRowCount = 96;
    std::vector<std::vector<float>> sourceRows(sourceRowCount,
                                                std::vector<float>(feature_size));
    std::vector<float> rawCloses(sourceRowCount);
    for (std::size_t row = 0; row < sourceRowCount; ++row)
    {
        rawCloses[row] = std::pow(1.001f, static_cast<float>(row));
        for (std::size_t col = 0; col < feature_size; ++col)
            sourceRows[row][col] = static_cast<float>(row * 100 + col);
        const std::array<std::array<float, 3>, 4> legalTG4States{{
            {{0.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f}},
            {{1.0f, 1.0f, 0.0f}}, {{1.0f, 1.0f, 1.0f}}}};
        const auto& tg4 = legalTG4States[row % legalTG4States.size()];
        sourceRows[row][tg4InnerBreakAnyCol] = tg4[0];
        sourceRows[row][tg4SourceTg3StructurallyEligibleCol] = tg4[1];
        sourceRows[row][tg4SourceTg3ConfluentCol] = tg4[2];
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

    // Layout 7 is retained as a historical semantic identity. Its 73-column
    // Tensor prefix is byte-identical to layout 8's prefix; only layout 8
    // includes the three TG4 columns before its relocated return suffix.
    const auto layout7AtOffset16 = BuildModelInputRow(
        sourceRows, rawCloses, kPreviouslyFailingGlobalStart,
        EA::kCausalEconomicEventSurpriseModelInputWidth);
    assert(layout7AtOffset16.size() == 77);
    for (std::size_t col = 0;
         col < causal_economic_event_surprise_feature_size; ++col)
        assert(layout7AtOffset16[col] == trainingAtOffset16[col]);
    assert(trainingAtOffset16[tg4InnerBreakAnyCol] == 0.0f);
    assert(trainingAtOffset16[tg4SourceTg3StructurallyEligibleCol] == 0.0f);
    assert(trainingAtOffset16[tg4SourceTg3ConfluentCol] == 0.0f);
    for (std::size_t returnIndex = 0;
         returnIndex < EA::kModelReturnFeatureCount; ++returnIndex)
    {
        assert(layout7AtOffset16[
                   causal_economic_event_surprise_feature_size + returnIndex] ==
               trainingAtOffset16[feature_size + returnIndex]);
    }

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

    // Layout-9 Fibonacci ablation is a pure projected-input treatment. Use
    // distinct nonzero sentinels across the whole 103-value model input so a
    // zero-valued producer output cannot hide an incorrectly omitted column.
    static_assert(fibRecentPriceScaleValidCol == 76);
    static_assert(fibDownRecentMedianPullback0618Col == 98);
    static_assert(feature_size == 99);
    static_assert(EA::kModelReturnFeatureCount == 4);
    static_assert(EA::kCurrentModelInputWidth == 103);
    const std::array<std::string_view, 23> fibonacciNames{{
        "fib_recent_price_scale_valid",
        "fib_up_recent_union_count_log",
        "fib_up_recent_h1_count_log",
        "fib_up_recent_h2_count_log",
        "fib_up_recent_h1_h2_both_count_log",
        "fib_up_recent_h1_youngest_age_20",
        "fib_up_recent_h2_youngest_age_20",
        "fib_up_recent_median_1272_signed_atr",
        "fib_up_recent_median_1618_signed_atr",
        "fib_up_recent_median_pullback_0382_signed_atr",
        "fib_up_recent_median_pullback_0500_signed_atr",
        "fib_up_recent_median_pullback_0618_signed_atr",
        "fib_down_recent_union_count_log",
        "fib_down_recent_h1_count_log",
        "fib_down_recent_h2_count_log",
        "fib_down_recent_h1_h2_both_count_log",
        "fib_down_recent_h1_youngest_age_20",
        "fib_down_recent_h2_youngest_age_20",
        "fib_down_recent_median_1272_signed_atr",
        "fib_down_recent_median_1618_signed_atr",
        "fib_down_recent_median_pullback_0382_signed_atr",
        "fib_down_recent_median_pullback_0500_signed_atr",
        "fib_down_recent_median_pullback_0618_signed_atr",
    }};
    std::vector<float> ablationSource(feature_size);
    for (std::size_t column = 0; column < ablationSource.size(); ++column)
        ablationSource[column] = static_cast<float>(1000 + column);
    const auto layout9Contract = EA::ResolveModelInputContract(
        EA::kCurrentModelInputWidth, ablationSource.size());
    const auto completeFibonacciMask = EA::FeatureAblationMask::Parse(
        std::string{EA::kCausalFibonacciStructuralAblationMaskText});
    assert(completeFibonacciMask.tensorColumns().size() == fibonacciNames.size());
    assert((completeFibonacciMask.tensorColumns() ==
            std::vector<std::size_t>{
                fibRecentPriceScaleValidCol, fibUpRecentUnionCountLogCol,
                fibUpRecentH1CountLogCol, fibUpRecentH2CountLogCol,
                fibUpRecentH1H2BothCountLogCol, fibUpRecentH1YoungestAge20Col,
                fibUpRecentH2YoungestAge20Col, fibUpRecentMedian1272Col,
                fibUpRecentMedian1618Col, fibUpRecentMedianPullback0382Col,
                fibUpRecentMedianPullback0500Col, fibUpRecentMedianPullback0618Col,
                fibDownRecentUnionCountLogCol, fibDownRecentH1CountLogCol,
                fibDownRecentH2CountLogCol, fibDownRecentH1H2BothCountLogCol,
                fibDownRecentH1YoungestAge20Col, fibDownRecentH2YoungestAge20Col,
                fibDownRecentMedian1272Col, fibDownRecentMedian1618Col,
                fibDownRecentMedianPullback0382Col,
                fibDownRecentMedianPullback0500Col,
                fibDownRecentMedianPullback0618Col}));
    std::string reverseFibonacciNames;
    for (auto it = fibonacciNames.rbegin(); it != fibonacciNames.rend(); ++it) {
        if (!reverseFibonacciNames.empty()) reverseFibonacciNames += ',';
        reverseFibonacciNames += *it;
        const auto individual = EA::FeatureAblationMask::Parse(std::string{*it});
        assert(individual.tensorColumns().size() == 1);
    }
    assert(EA::FeatureAblationMask::Parse(reverseFibonacciNames).CanonicalText() ==
           EA::kCausalFibonacciStructuralAblationMaskText);
    assert(completeFibonacciMask.CanonicalText() ==
           EA::kCausalFibonacciStructuralAblationMaskText);

    const auto BuildSentinelModelInput = [&](const EA::FeatureAblationMask& mask) {
        std::vector<float> values(EA::kCurrentModelInputWidth);
        EA::CopyTensorFeaturesForModelInput(values.data(), ablationSource.data(),
                                            layout9Contract, mask);
        for (std::size_t column = feature_size;
             column < EA::kCurrentModelInputWidth; ++column)
            values[column] = static_cast<float>(1000 + column);
        return values;
    };
    const auto sentinelControl = BuildSentinelModelInput({});
    const auto sentinelEmptyMask = BuildSentinelModelInput(
        EA::FeatureAblationMask::Parse(""));
    AssertByteIdentical(sentinelControl, sentinelEmptyMask);
    const auto sentinelAblation = BuildSentinelModelInput(completeFibonacciMask);
    std::size_t changed = 0;
    for (std::size_t column = 0; column < sentinelControl.size(); ++column) {
        const bool fibonacciColumn = column >= fibRecentPriceScaleValidCol &&
            column <= fibDownRecentMedianPullback0618Col;
        assert(sentinelAblation[column] ==
               (fibonacciColumn ? 0.0f : sentinelControl[column]));
        changed += sentinelAblation[column] != sentinelControl[column] ? 1 : 0;
    }
    assert(changed == fibonacciNames.size());
    for (std::size_t column = 0; column < fibRecentPriceScaleValidCol; ++column)
        assert(sentinelAblation[column] == sentinelControl[column]);
    for (std::size_t column = feature_size;
         column < EA::kCurrentModelInputWidth; ++column)
        assert(sentinelAblation[column] == sentinelControl[column]);
    bool unknownFibonacciNameRejected = false;
    try { (void)EA::FeatureAblationMask::Parse("fib_unknown_feature"); }
    catch (const std::invalid_argument&) { unknownFibonacciNameRejected = true; }
    assert(unknownFibonacciNameRejected);

    return 0;
}
