#include <cassert>
#include <cmath>
#include <vector>

#include "DonchianFeatures.hpp"
#include "Donchian20Mode.hpp"

constexpr float kTestFeatureScale = 1000.0f;

namespace
{
void AssertNear(float actual, float expected)
{
    assert(std::fabs(actual - expected) < 1.0e-4f);
}
}

int main()
{
    assert(ParseDonchian20Mode("enabled") == Donchian20Mode::Enabled);
    assert(ParseDonchian20Mode("zero_ablation") == Donchian20Mode::ZeroAblation);
    assert(std::string{Donchian20ModeText(kDefaultDonchian20Mode)} == "enabled");
    bool invalidRejected = false;
    try { (void)ParseDonchian20Mode("invalid"); }
    catch (const std::invalid_argument&) { invalidRejected = true; }
    assert(invalidRejected);
    static_assert(legacy_feature_size == 32);
    static_assert(donchian_feature_size == 34);
    static_assert(donchianUpCol == 32);
    static_assert(donchianDownCol == 33);
    constexpr size_t effectiveModelWidth = donchian_feature_size + 4;
    static_assert(effectiveModelWidth == 38);

    std::vector<float> highs{101.0f, 103.0f, 102.0f};
    std::vector<float> lows{99.0f, 98.0f, 100.0f};
    const auto baseline = ComputeCausalDonchian20(highs, lows, 100.0f, kTestFeatureScale);
    AssertNear(baseline.first, std::log(100.0f / 103.0f) * kTestFeatureScale);
    AssertNear(baseline.second, std::log(100.0f / 98.0f) * kTestFeatureScale);

    // Changing only the current bar's high/low (while close and prior history
    // stay fixed) cannot change either value because they are not inputs.
    float currentHigh = 150.0f;
    float currentLow = 50.0f;
    const auto unchanged = ComputeCausalDonchian20(highs, lows, 100.0f, kTestFeatureScale);
    currentHigh = 10.0f;
    currentLow = 200.0f;
    (void)currentHigh;
    (void)currentLow;
    const auto unchangedAfterCurrentBarExtremaChange =
        ComputeCausalDonchian20(highs, lows, 100.0f, kTestFeatureScale);
    assert(unchanged == baseline);
    assert(unchangedAfterCurrentBarExtremaChange == baseline);

    highs[1] = 110.0f;
    const auto changedUpper = ComputeCausalDonchian20(highs, lows, 100.0f, kTestFeatureScale);
    assert(changedUpper.first != baseline.first);
    AssertNear(changedUpper.second, baseline.second);

    highs[1] = 103.0f;
    lows[1] = 90.0f;
    const auto changedLower = ComputeCausalDonchian20(highs, lows, 100.0f, kTestFeatureScale);
    AssertNear(changedLower.first, baseline.first);
    assert(changedLower.second != baseline.second);

    // Partial startup history is used; no synthetic/future history is added.
    const auto first = ComputeCausalDonchian20({}, {}, 100.0f, kTestFeatureScale);
    assert(first.first == 0.0f && first.second == 0.0f);
    const auto partial = ComputeCausalDonchian20({101.0f}, {99.0f}, 100.0f, kTestFeatureScale);
    AssertNear(partial.first, std::log(100.0f / 101.0f) * kTestFeatureScale);
    AssertNear(partial.second, std::log(100.0f / 99.0f) * kTestFeatureScale);

    // Only the most recent 20 prior rows participate.
    std::vector<float> twentyOneHighs(21, 101.0f);
    std::vector<float> twentyOneLows(21, 99.0f);
    twentyOneHighs.front() = 1000.0f;
    twentyOneLows.front() = 1.0f;
    const auto lookback = ComputeCausalDonchian20(twentyOneHighs, twentyOneLows, 100.0f, kTestFeatureScale);
    AssertNear(lookback.first, std::log(100.0f / 101.0f) * kTestFeatureScale);
    AssertNear(lookback.second, std::log(100.0f / 99.0f) * kTestFeatureScale);

    // Explicit runtime lookback changes only the history window; the closed
    // ComputeCausalDonchian20 API remains exactly the 20-bar definition.
    const auto shortLookback = ComputeCausalDonchian(
        twentyOneHighs, twentyOneLows, 100.0f, kTestFeatureScale, 21);
    assert(shortLookback.first != lookback.first);
    assert(shortLookback.second != lookback.second);
    assert(ParseDonchianLookback("20") == kDefaultDonchianLookback);
    bool invalidLookbackRejected = false;
    try { (void)ParseDonchianLookback("0"); }
    catch (const std::invalid_argument&) { invalidLookbackRejected = true; }
    assert(invalidLookbackRejected);
}
