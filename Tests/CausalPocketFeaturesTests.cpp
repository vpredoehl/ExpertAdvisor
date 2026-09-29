#include "CausalPocketFeatures.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace
{
using EA::CausalPocketFeatures::Aggregate;
using EA::CausalPocketFeatures::BearMedianCloseDistance;
using EA::CausalPocketFeatures::BearMedianTouchDistance;
using EA::CausalPocketFeatures::BearMedianWidth;
using EA::CausalPocketFeatures::BearRecentCountLog;
using EA::CausalPocketFeatures::BearYoungestAge20;
using EA::CausalPocketFeatures::BullMedianCloseDistance;
using EA::CausalPocketFeatures::BullMedianTouchDistance;
using EA::CausalPocketFeatures::BullMedianWidth;
using EA::CausalPocketFeatures::BullRecentCountLog;
using EA::CausalPocketFeatures::BullYoungestAge20;
using EA::CausalPocketFeatures::Producer;
using EA::CausalPocketFeatures::RecentPriceScaleValid;
using EA::Pocket::PocketDirection;
using EA::Pocket::PocketObservation;

EA::TG1A::Candle Candle(std::int64_t timestamp, double open, double high,
                        double low, double close)
{
    return {timestamp, open, high, low, close};
}

PocketObservation Observation(PocketDirection direction, std::size_t event,
                              std::size_t cutoff, double lower, double upper)
{
    return {direction, {lower, upper}, event, static_cast<std::int64_t>(event),
            cutoff, static_cast<std::int64_t>(cutoff), cutoff,
            static_cast<std::int64_t>(cutoff), "15m_completed"};
}

template <typename Function>
bool Throws(Function&& function)
{
    try { function(); }
    catch (const std::exception&) { return true; }
    return false;
}

std::vector<EA::TG1A::Candle> BullishPocketBars()
{
    std::vector<EA::TG1A::Candle> result;
    std::int64_t timestamp = 0;
    for (std::size_t index = 0; index < 15; ++index)
        result.push_back(Candle(timestamp++, 5.0, 10.0, 0.0, 5.0));
    result.push_back(Candle(timestamp++, 10.0, 12.0, 9.0, 11.0));
    result.push_back(Candle(timestamp++, 12.0, 13.0, 11.5, 12.0));
    return result;
}

void TestConfirmationAgeAndStreamingDeterminism()
{
    const std::vector<EA::TG1A::Candle> bars = BullishPocketBars();
    Producer left{"eurusdrmp"};
    Producer right{"eurusdrmp"};
    std::vector<std::array<float, EA::CausalPocketFeatures::kFeatureCount>> prefix;
    for (std::size_t index = 0; index < bars.size(); ++index)
    {
        const auto a = left.AddCompletedBar(bars[index]);
        const auto b = right.AddCompletedBar(bars[index]);
        assert(a == b);
        prefix.push_back(a);
        if (index < 16) assert(a[BullRecentCountLog] == 0.0F);
    }
    const auto confirmation = prefix.back();
    assert(std::fabs(confirmation[BullRecentCountLog] - std::log1p(1.0)) < 1e-6F);
    assert(confirmation[BullYoungestAge20] == 0.0F); // cutoff age zero is valid.

    for (std::size_t index = 0; index < 20; ++index)
        (void)left.AddCompletedBar(Candle(17 + static_cast<std::int64_t>(index),
                                          5.0, 10.0, 0.0, 5.0));
    const auto age20 = prefix.back();
    // The last streamed neutral bar is bar 36: confirmation bar 16 is age 20.
    const auto age20Replay = [&] {
        Producer replay{"eurusdrmp"};
        for (const auto& bar : bars) (void)replay.AddCompletedBar(bar);
        std::array<float, EA::CausalPocketFeatures::kFeatureCount> value{};
        for (std::size_t index = 0; index < 20; ++index)
            value = replay.AddCompletedBar(Candle(17 + static_cast<std::int64_t>(index),
                                                  5.0, 10.0, 0.0, 5.0));
        return value;
    }();
    (void)age20;
    assert(age20Replay[BullRecentCountLog] > 0.0F);
    const auto age21 = left.AddCompletedBar(Candle(37, 5.0, 10.0, 0.0, 5.0));
    assert(age21[BullRecentCountLog] == 0.0F);
}

void TestAggregatePopulationMedianOrientationAndScale()
{
    const std::vector<PocketObservation> observations{
        Observation(PocketDirection::Bullish, 8, 10, 90.0, 110.0),
        Observation(PocketDirection::Bullish, 9, 12, 95.0, 105.0),
        Observation(PocketDirection::Bearish, 10, 12, 90.0, 110.0),
    };
    const auto values = Aggregate(observations, 12, 100.0, 10.0, 1.0);
    assert(values[RecentPriceScaleValid] == 1.0F);
    assert(std::fabs(values[BullRecentCountLog] - std::log1p(2.0)) < 1e-6F);
    assert(values[BullYoungestAge20] == 0.0F);
    assert(std::fabs(values[BullMedianTouchDistance] - 0.5F) < 1e-6F);
    assert(std::fabs(values[BullMedianCloseDistance] + 1.0F) < 1e-6F);
    assert(std::fabs(values[BullMedianWidth] - 1.0F) < 1e-6F);
    assert(std::fabs(values[BearRecentCountLog] - std::log1p(1.0)) < 1e-6F);
    assert(values[BearYoungestAge20] == 0.0F);
    assert(std::fabs(values[BearMedianTouchDistance] - 1.0F) < 1e-6F);
    assert(std::fabs(values[BearMedianCloseDistance] + 1.0F) < 1e-6F);
    assert(std::fabs(values[BearMedianWidth] - 2.0F) < 1e-6F);

    const auto pipFallback = Aggregate({Observation(PocketDirection::Bullish,
        9, 10, 10.0, 11.0)}, 10, 10.0, 0.05, 0.1);
    assert(pipFallback[RecentPriceScaleValid] == 1.0F);
    assert(std::fabs(pipFallback[BullMedianTouchDistance] - 10.0F) < 1e-6F);

    const auto invalid = Aggregate({Observation(PocketDirection::Bullish,
        9, 10, 10.0, 11.0)}, 10,
        std::numeric_limits<double>::quiet_NaN(), 2.0, 0.1);
    assert(invalid[RecentPriceScaleValid] == 0.0F);
    assert(invalid[BullRecentCountLog] > 0.0F && invalid[BullYoungestAge20] == 0.0F);
    assert(invalid[BullMedianTouchDistance] == 0.0F &&
           invalid[BullMedianCloseDistance] == 0.0F && invalid[BullMedianWidth] == 0.0F);
}

void TestDuplicateAndMalformedInputRejection()
{
    const PocketObservation observation =
        Observation(PocketDirection::Bullish, 9, 10, 10.0, 11.0);
    assert(Throws([&] { (void)Aggregate({observation, observation}, 10, 10.0, 2.0, .1); }));
    assert(Throws([&] { (void)Aggregate({Observation(PocketDirection::Bullish,
        9, 10, 11.0, 10.0)}, 10, 10.0, 2.0, .1); }));

    Producer producer{"eurusdrmp"};
    (void)producer.AddCompletedBar(Candle(1, 1.0, 2.0, 0.0, 1.0));
    assert(Throws([&] { (void)producer.AddCompletedBar(Candle(1, 1.0, 2.0, 0.0, 1.0)); }));
    Producer invalid{"eurusdrmp"};
    assert(Throws([&] { (void)invalid.AddCompletedBar(Candle(1,
        std::numeric_limits<double>::quiet_NaN(), 2.0, 0.0, 1.0)); }));
}

void TestCompletedBarAtrOrdering()
{
    const std::vector<EA::TG1A::Candle> bars = BullishPocketBars();
    Producer producer{"eurusdrmp"};
    EA::CausalFibonacciStructuralFeatureConfiguration::Configuration configuration;
    EA::TG1A::CausalFractalTrendLineGeometry geometry(
        configuration.geometry(), {"eurusdrmp", std::string{configuration.timeframe()}});
    std::array<float, EA::CausalPocketFeatures::kFeatureCount> actual{};
    for (const auto& bar : bars)
    {
        (void)geometry.AddCompletedBar(bar);
        actual = producer.AddCompletedBar(bar);
    }
    const double atr = *geometry.CurrentAtr();
    assert(atr > configuration.CanonicalPipSize("eurusdrmp"));
    // The emitted bullish Pocket touches 11.5; its current close is 12.0.
    assert(std::fabs(actual[BullMedianTouchDistance] -
                     static_cast<float>((11.5 - 12.0) / atr)) < 1e-6F);
}

} // namespace

int main()
{
    TestConfirmationAgeAndStreamingDeterminism();
    TestAggregatePopulationMedianOrientationAndScale();
    TestDuplicateAndMalformedInputRejection();
    TestCompletedBarAtrOrdering();
}
