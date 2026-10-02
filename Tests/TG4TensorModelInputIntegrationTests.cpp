#include "FeatureLayout.hpp"
#include "ModelInputContract.hpp"
#include "ModelInputExpansion.hpp"
#include "ModelInputFeatureSemantics.hpp"
#include "PricePoint.hpp"
#include "ReturnFeatureHistory.hpp"
#include "Tensor.hpp"
#include "TG4ProductionStreamingPulseAdapter.hpp"
#include "FixedConfluenceTensorAdapter.hpp"
#include "CausalFibonacciStructuralFeatures.hpp"
#include "CausalPocketFeatures.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <string_view>
#include <vector>

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled()
{
    return false;
}

namespace
{

Feature Bar(std::size_t index)
{
    const float base = 1.0800f + static_cast<float>(index) * 0.00011f;
    const float wiggle = (index % 5 == 0) ? -0.00007f :
        ((index % 7 == 0) ? 0.00009f : 0.0f);
    Feature bar{};
    bar.time = PriceTP{std::chrono::seconds{
        1'741'150'800 + static_cast<long long>(index) * 900}};
    bar.open = base;
    bar.close = base + wiggle;
    bar.high = std::max(bar.open, bar.close) + 0.00015f;
    bar.low = std::min(bar.open, bar.close) - 0.00015f;
    bar.tickVolume = 100.0f + static_cast<float>(index % 17);
    return bar;
}

std::vector<float> MaterializeTrainingStyle(const Tensor& tensor,
                                            std::size_t row)
{
    const auto contract = EA::ResolveModelInputContract(
        EA::kCurrentModelInputWidth,
        static_cast<std::size_t>((*tensor.begin()).Shape()[1]));
    const auto iterator = tensor.begin() + static_cast<std::ptrdiff_t>(row);
    const auto physical = MetaNN::LowerAccess(*iterator);
    std::vector<float> input(EA::kCurrentModelInputWidth, -1.0f);
    EA::CopyTensorFeaturesForModelInput(input.data(), physical.RawMemory(),
                                        contract);
    const std::size_t appended =
        EA::AppendMultiHorizonReturnFeaturesAtGlobalPosition(
            row, input.data(), contract.tensorFeatureCount, 1000.0f,
            [&tensor](std::size_t position) {
                return tensor.RawCloseAtIterator(
                    tensor.begin() + static_cast<std::ptrdiff_t>(position));
            });
    assert(appended == EA::kModelReturnFeatureCount);
    return input;
}

std::vector<float> MaterializeInferenceStyle(const Tensor& tensor,
                                             std::size_t windowStart,
                                             std::size_t localRow)
{
    // Keep this intentionally separate from the training-style path above:
    // inference resolves its global row from a score-window-local position.
    const std::size_t globalRow = windowStart + localRow;
    const auto contract = EA::ResolveModelInputContract(
        EA::kCurrentModelInputWidth,
        static_cast<std::size_t>((*tensor.begin()).Shape()[1]));
    const auto iterator = tensor.begin() +
        static_cast<std::ptrdiff_t>(globalRow);
    const auto physical = MetaNN::LowerAccess(*iterator);
    std::vector<float> input(EA::kCurrentModelInputWidth, -2.0f);
    EA::CopyTensorFeaturesForModelInput(input.data(), physical.RawMemory(),
                                        contract);
    const std::size_t appended =
        EA::AppendMultiHorizonReturnFeaturesAtGlobalPosition(
            globalRow, input.data(), contract.tensorFeatureCount, 1000.0f,
            [&tensor](std::size_t position) {
                return tensor.RawCloseAtIterator(
                    tensor.begin() + static_cast<std::ptrdiff_t>(position));
            });
    assert(appended == EA::kModelReturnFeatureCount);
    return input;
}

void AssertExact(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    assert(lhs.size() == rhs.size());
    assert(std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0);
}

void TestLayoutAndCanonicalPulseAlignment()
{
    static_assert(EA::kModelInputSemanticLayoutVersion == 11);
    static_assert(EA::kCurrentModelInputWidth == 116);
    static_assert(feature_size == 112);
    static_assert(tg4InnerBreakAnyCol == 73);
    static_assert(tg4SourceTg3StructurallyEligibleCol == 74);
    static_assert(tg4SourceTg3ConfluentCol == 75);

    std::vector<Feature> bars;
    bars.reserve(192);
    Tensor tensor{"eurusdrmp"};
    EA::CausalFibonacciFeatures::Producer fibReplayProducer{"eurusdrmp"};
    EA::CausalPocketFeatures::Producer pocketReplayProducer{"eurusdrmp"};
    EA::MarketStructure::Production::TG4ProductionConfluenceBridge confluenceBridge{
        "eurusdrmp"};
    EA::MarketStructure::TensorProjection::FixedConfluenceTensorAdapter
        confluenceAdapter;
    std::vector<std::array<float, EA::CausalFibonacciFeatures::kFeatureCount>> fibReplay;
    std::vector<std::array<float, EA::CausalPocketFeatures::kFeatureCount>>
        pocketReplay;
    for (std::size_t index = 0; index < 192; ++index)
    {
        bars.push_back(Bar(index));
        const EA::TG1A::Candle completedBar{
            bars.back().time.time_since_epoch().count(),
            static_cast<double>(bars.back().open),
            static_cast<double>(bars.back().high),
            static_cast<double>(bars.back().low),
            static_cast<double>(bars.back().close)};
        fibReplay.push_back(fibReplayProducer.AddCompletedBar(completedBar));
        pocketReplay.push_back(
            pocketReplayProducer.AddCompletedBar(completedBar));
        tensor.Add(bars.back());
    }
    const auto replay = EA::TG4Pulse::ReplayCanonicalCompletedBars(
        "eurusdrmp", bars);
    assert(replay.size() == tensor.RowCount());

    for (std::size_t row = 0; row < tensor.RowCount(); ++row)
    {
        const auto iterator = tensor.begin() + static_cast<std::ptrdiff_t>(row);
        const auto physical = MetaNN::LowerAccess(*iterator);
        const float* values = physical.RawMemory();
        assert(tensor.RawTimeAtIterator(iterator) == replay[row].barStart);
        assert(values[tg4InnerBreakAnyCol] ==
               static_cast<float>(replay[row].bits[0]));
        assert(values[tg4SourceTg3StructurallyEligibleCol] ==
               static_cast<float>(replay[row].bits[1]));
        assert(values[tg4SourceTg3ConfluentCol] ==
               static_cast<float>(replay[row].bits[2]));
        assert(values[tg4SourceTg3ConfluentCol] <=
               values[tg4SourceTg3StructurallyEligibleCol]);
        assert(values[tg4SourceTg3StructurallyEligibleCol] <=
               values[tg4InnerBreakAnyCol]);
        for (std::size_t column = 0; column < fibReplay[row].size(); ++column)
            assert(values[fibRecentPriceScaleValidCol + column] ==
                   fibReplay[row][column]);
        for (std::size_t column = 0; column < pocketReplay[row].size(); ++column)
            assert(values[pocketRecentPriceScaleValidCol + column] ==
                   pocketReplay[row][column]);
        const auto description = confluenceBridge.Describe(
            replay[row], std::chrono::sys_seconds{
                replay[row].barStart.time_since_epoch() + std::chrono::seconds{900}});
        const auto expectedConfluence = confluenceAdapter.Adapt(
            description, std::chrono::sys_seconds{
                replay[row].barStart.time_since_epoch() + std::chrono::seconds{900}});
        assert(values[confluenceTg4StructuralFibonacciRetracementSupportAvailableCol] ==
               expectedConfluence[0]);
        assert(values[confluenceTg4StructuralFibonacciRetracementContradictionAvailableCol] ==
               expectedConfluence[1]);
    }

    const auto semantics = EA::ModelInputFeatureSemantics(116);
    assert(semantics.at(73).name == "tg4_inner_break_any");
    assert(semantics.at(74).name == "tg4_source_tg3_structurally_eligible");
    assert(semantics.at(75).name == "tg4_source_tg3_confluent");
    assert(semantics.at(73).categorical && semantics.at(74).categorical &&
           semantics.at(75).categorical);
    assert(semantics.at(76).name == "fib_recent_price_scale_valid");
    assert(semantics.at(98).name == "fib_down_recent_median_pullback_0618_signed_atr");
    constexpr std::array<std::string_view, EA::CausalPocketFeatures::kFeatureCount>
        pocketNames{{
            "pocket_recent_price_scale_valid",
            "pocket_bull_recent_count_log",
            "pocket_bull_youngest_age20",
            "pocket_bull_median_touch_distance",
            "pocket_bull_median_close_distance",
            "pocket_bull_median_width",
            "pocket_bear_recent_count_log",
            "pocket_bear_youngest_age20",
            "pocket_bear_median_touch_distance",
            "pocket_bear_median_close_distance",
            "pocket_bear_median_width"}};
    for (std::size_t column = 0; column < pocketNames.size(); ++column)
        assert(semantics.at(pocketRecentPriceScaleValidCol + column).name ==
               pocketNames[column]);
    assert(semantics.at(110).name ==
           "confluence_tg4_structural_fibonacci_retracement_support_available");
    assert(semantics.at(111).name ==
           "confluence_tg4_structural_fibonacci_retracement_contradiction_available");
    assert(semantics.at(112).name == "lookback_log_return_1_scaled");
    assert(semantics.at(115).name == "lookback_log_return_16_scaled");

    // Full-history construction occurs before the scored window. Both model
    // consumers therefore use the exact stateful Tensor row at its boundary.
    constexpr std::size_t scoreWindowStart = 96;
    const auto training = MaterializeTrainingStyle(tensor, scoreWindowStart);
    const auto inference = MaterializeInferenceStyle(
        tensor, scoreWindowStart - 16, 16);
    AssertExact(training, inference);
    for (std::size_t column = tg4InnerBreakAnyCol;
         column <= tg4SourceTg3ConfluentCol; ++column)
        assert(training[column] == 0.0f || training[column] == 1.0f);
    for (std::size_t column = 0; column < pocketReplay[scoreWindowStart].size();
         ++column)
        assert(training[pocketRecentPriceScaleValidCol + column] ==
               pocketReplay[scoreWindowStart][column]);

    // State is causal: extending a stream cannot mutate already-emitted rows.
    std::vector<std::array<float, EA::CausalFibonacciFeatures::kFeatureCount>>
        prefix = fibReplay;
    std::vector<std::array<float, EA::CausalPocketFeatures::kFeatureCount>>
        pocketPrefix = pocketReplay;
    for (std::size_t index = 192; index < 224; ++index)
        tensor.Add(Bar(index));
    for (std::size_t row = 0; row < prefix.size(); ++row) {
        const auto physical = MetaNN::LowerAccess(
            *(tensor.begin() + static_cast<std::ptrdiff_t>(row)));
        for (std::size_t column = 0; column < prefix[row].size(); ++column)
            assert(physical.RawMemory()[fibRecentPriceScaleValidCol + column] ==
                   prefix[row][column]);
        for (std::size_t column = 0; column < pocketPrefix[row].size(); ++column)
            assert(physical.RawMemory()[pocketRecentPriceScaleValidCol + column] ==
                   pocketPrefix[row][column]);
    }

    // Layout 9 projects only the immutable Tensor prefix through column 98,
    // then appends its own return suffix at rows 99..102. Pocket columns are
    // absent from the persisted width-103 model identity.
    const auto layout9 = EA::ResolveModelInputContract(103, feature_size);
    assert(layout9.tensorFeatureCount == causal_fibonacci_structural_feature_size);
    std::vector<float> historicalLayout9(103, -1.0f);
    const auto boundary = tensor.begin() +
        static_cast<std::ptrdiff_t>(scoreWindowStart);
    const auto boundaryPhysical = MetaNN::LowerAccess(*boundary);
    EA::CopyTensorFeaturesForModelInput(historicalLayout9.data(),
                                        boundaryPhysical.RawMemory(), layout9);
    for (std::size_t column = 0; column < layout9.tensorFeatureCount; ++column)
        assert(historicalLayout9[column] == training[column]);
    for (std::size_t column = layout9.tensorFeatureCount;
         column < historicalLayout9.size(); ++column)
        assert(historicalLayout9[column] == -1.0f);
    const std::size_t layout9Appended =
        EA::AppendMultiHorizonReturnFeaturesAtGlobalPosition(
            scoreWindowStart, historicalLayout9.data(), layout9.tensorFeatureCount,
            1000.0f, [&tensor](std::size_t position) {
                return tensor.RawCloseAtIterator(
                    tensor.begin() + static_cast<std::ptrdiff_t>(position));
            });
    assert(layout9Appended == EA::kModelReturnFeatureCount);
    for (std::size_t column = 0; column < EA::kModelReturnFeatureCount; ++column)
        assert(historicalLayout9[layout9.tensorFeatureCount + column] ==
               training[EA::kCurrentModelInputWidth -
                        EA::kModelReturnFeatureCount + column]);

    const auto layout9Semantics = EA::ModelInputFeatureSemantics(103);
    assert(layout9Semantics.at(98).name ==
           "fib_down_recent_median_pullback_0618_signed_atr");
    assert(layout9Semantics.at(99).name == "lookback_log_return_1_scaled");
    assert(layout9Semantics.at(102).name == "lookback_log_return_16_scaled");

    // Distinct Pocket and return sentinels prove no Pocket value can alias
    // into Layout 9's independently materialized historical return suffix.
    std::array<float, feature_size> layout9SentinelPhysical{};
    for (std::size_t column = 0; column < layout9SentinelPhysical.size(); ++column)
        layout9SentinelPhysical[column] = 100.0f + static_cast<float>(column);
    constexpr std::array<float, EA::kModelReturnFeatureCount> layout9Returns{{
        -10.0f, -20.0f, -30.0f, -40.0f}};
    std::array<float, EA::kCausalFibonacciStructuralModelInputWidth>
        layout9SentinelInput{};
    layout9SentinelInput.fill(-1.0f);
    EA::CopyTensorFeaturesForModelInput(layout9SentinelInput.data(),
                                        layout9SentinelPhysical.data(), layout9);
    std::copy(layout9Returns.begin(), layout9Returns.end(),
              layout9SentinelInput.begin() +
                  static_cast<std::ptrdiff_t>(layout9.tensorFeatureCount));
    for (std::size_t column = 0; column < layout9.tensorFeatureCount; ++column)
        assert(layout9SentinelInput[column] == layout9SentinelPhysical[column]);
    for (std::size_t column = 0; column < layout9Returns.size(); ++column)
    {
        assert(layout9SentinelInput[layout9.tensorFeatureCount + column] ==
               layout9Returns[column]);
        assert(layout9SentinelInput[layout9.tensorFeatureCount + column] !=
               layout9SentinelPhysical[pocketRecentPriceScaleValidCol + column]);
    }

    // The retained layout-7 identity projects only the unchanged prefix and
    // retains its return suffix semantics at its historical positions.
    const auto layout7 = EA::ResolveModelInputContract(77, feature_size);
    std::vector<float> historical(77, -1.0f);
    EA::CopyTensorFeaturesForModelInput(historical.data(),
                                        boundaryPhysical.RawMemory(), layout7);
    for (std::size_t column = 0;
         column < causal_economic_event_surprise_feature_size; ++column)
        assert(historical[column] == training[column]);

    // Distinct sentinels prove that a current physical layout-10 row cannot
    // alias TG4 positions into layout-7's historical return suffix.
    std::array<float, feature_size> sentinelPhysical{};
    for (std::size_t column = 0;
         column < causal_economic_event_surprise_feature_size; ++column)
        sentinelPhysical[column] = 100.0f + static_cast<float>(column);
    sentinelPhysical[tg4InnerBreakAnyCol] = 901.0f;
    sentinelPhysical[tg4SourceTg3StructurallyEligibleCol] = 902.0f;
    sentinelPhysical[tg4SourceTg3ConfluentCol] = 903.0f;
    std::array<float, 77> historicalSentinel{};
    historicalSentinel.fill(-1.0f);
    EA::CopyTensorFeaturesForModelInput(historicalSentinel.data(),
                                        sentinelPhysical.data(), layout7);
    constexpr std::array<float, EA::kModelReturnFeatureCount> returns{{
        -10.0f, -20.0f, -30.0f, -40.0f}};
    std::copy(returns.begin(), returns.end(),
              historicalSentinel.begin() +
                  static_cast<std::ptrdiff_t>(
                      causal_economic_event_surprise_feature_size));
    for (std::size_t column = 0;
         column < causal_economic_event_surprise_feature_size; ++column)
        assert(historicalSentinel[column] == sentinelPhysical[column]);
    for (std::size_t index = 0; index < returns.size(); ++index)
    {
        assert(historicalSentinel[
                   causal_economic_event_surprise_feature_size + index] ==
               returns[index]);
        assert(historicalSentinel[
                   causal_economic_event_surprise_feature_size + index] !=
               sentinelPhysical[tg4InnerBreakAnyCol +
                                std::min(index, std::size_t{2})]);
    }
}

void TestPulseStateContract()
{
    using State = EA::TG3::ConfluenceState;
    assert((EA::TG4Pulse::ProductionStreamingAdapter::AggregateSameBar({}) ==
            std::array<std::uint8_t, 3>{{0, 0, 0}}));
    assert((EA::TG4Pulse::ProductionStreamingAdapter::AggregateSameBar(
                {State::StructurallyIneligible}) ==
            std::array<std::uint8_t, 3>{{1, 0, 0}}));
    assert((EA::TG4Pulse::ProductionStreamingAdapter::AggregateSameBar(
                {State::NoConfluence}) ==
            std::array<std::uint8_t, 3>{{1, 1, 0}}));
    assert((EA::TG4Pulse::ProductionStreamingAdapter::AggregateSameBar(
                {State::Confluence}) ==
            std::array<std::uint8_t, 3>{{1, 1, 1}}));
}

} // namespace

int main()
{
    TestLayoutAndCanonicalPulseAlignment();
    TestPulseStateContract();
    return 0;
}
