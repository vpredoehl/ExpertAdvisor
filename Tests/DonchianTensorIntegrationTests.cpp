#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>

#include "DonchianFeatures.hpp"
#include "FeatureLayout.hpp"
#include "PricePoint.hpp"
#include "Tensor.hpp"

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled()
{
    return false;
}

namespace
{
void AssertNear(float actual, float expected)
{
    assert(std::fabs(actual - expected) < 1.0e-4f);
}

Feature MakeFeature(std::size_t index,
                    float open,
                    float close,
                    float high,
                    float low)
{
    Feature f {};
    f.open = open;
    f.close = close;
    f.high = high;
    f.low = low;
    f.time = PriceTP {
        std::chrono::seconds {
            static_cast<long long>(index * 15 * 60)
        }
    };
    return f;
}

std::pair<float, float> LastDonchianValues(const Tensor& tensor)
{
    assert(tensor.begin() != tensor.end());

    auto it = tensor.end();
    --it;

    auto low = MetaNN::LowerAccess(*it);
    const float* p = low.RawMemory();

    return {
        p[donchianUpCol],
        p[donchianDownCol]
    };
}

void PopulateIdenticalPriorHistory(Tensor& tensor)
{
    // 20 prior rows with deterministic extrema:
    //
    // prior max high = 105
    // prior min low  = 95
    //
    // Close values remain positive and realistic enough for all of
    // Tensor::Add's existing derived-feature calculations.
    for (std::size_t i = 0; i < donchian_lookback; ++i)
    {
        const float close = 100.0f + static_cast<float>(i) * 0.01f;
        const float high =
            (i == 7) ? 105.0f : close + 1.0f;
        const float low =
            (i == 13) ? 95.0f : close - 1.0f;

        tensor.Add(
            MakeFeature(
                i,
                close - 0.10f,
                close,
                high,
                low));
    }
}

void TestCurrentBarHighLowCannotAffectDonchian()
{
    Tensor normalCurrentBar("donchian_tensor_integration_normal");
    Tensor extremeCurrentBar("donchian_tensor_integration_extreme");

    PopulateIdenticalPriorHistory(normalCurrentBar);
    PopulateIdenticalPriorHistory(extremeCurrentBar);

    constexpr float currentOpen = 99.75f;
    constexpr float currentClose = 100.0f;

    // Same prior history, same current open/close/time.
    //
    // Only current high/low differ, and they differ drastically enough that
    // accidental inclusion in the current Donchian extrema would be obvious.
    normalCurrentBar.Add(
        MakeFeature(
            donchian_lookback,
            currentOpen,
            currentClose,
            101.0f,
            99.0f));

    extremeCurrentBar.Add(
        MakeFeature(
            donchian_lookback,
            currentOpen,
            currentClose,
            1000.0f,
            1.0f));

    const auto normal = LastDonchianValues(normalCurrentBar);
    const auto extreme = LastDonchianValues(extremeCurrentBar);

    // Integration-level causality proof:
    //
    // Tensor::Add must produce identical Donchian columns despite radically
    // different CURRENT high/low values.
    AssertNear(normal.first, extreme.first);
    AssertNear(normal.second, extreme.second);

    // Also prove Tensor::Add produced exactly the expected prior-window
    // formulas, not merely equal-but-wrong values.
    AssertNear(
        normal.first,
        std::log(currentClose / 105.0f) * kFeatureScale);

    AssertNear(
        normal.second,
        std::log(currentClose / 95.0f) * kFeatureScale);
}

void TestZeroAblationPreservesWidthAndPositions()
{
    Tensor tensor("donchian_tensor_integration_zero_ablation",
                  Donchian20Mode::ZeroAblation);
    PopulateIdenticalPriorHistory(tensor);
    tensor.Add(MakeFeature(donchian_lookback, 99.75f, 100.0f, 101.0f, 99.0f));

    auto it = tensor.end();
    --it;
    auto low = MetaNN::LowerAccess(*it);
    const float* p = low.RawMemory();
    assert(tensor.GetDonchian20Mode() == Donchian20Mode::ZeroAblation);
    assert(feature_size == 42);
    assert(p[donchianUpCol] == 0.0f);
    assert(p[donchianDownCol] == 0.0f);
    assert(p[0] != 0.0f);
}
}

int main()
{
    static_assert(feature_size == 42);
    static_assert(donchianUpCol == 32);
    static_assert(donchianDownCol == 33);
    static_assert(sessionPhaseSinCol == 34);
    static_assert(sessionPhaseCosCol == 35);
    static_assert(relativeTickVolumeCol == 36);
    static_assert(causalReturnSurpriseCol == 37);
    static_assert(donchian_lookback == 20);

    TestCurrentBarHighLowCannotAffectDonchian();
    TestZeroAblationPreservesWidthAndPositions();

    return 0;
}
