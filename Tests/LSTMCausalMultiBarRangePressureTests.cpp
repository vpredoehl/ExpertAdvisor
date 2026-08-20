#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>

#include <MetaNN/data/facilities/lower_access.h>

#include "CausalMultiBarRangePressureFeatures.hpp"
#include "FeatureLayout.hpp"
#include "ModelInputContract.hpp"
#include "PricePoint.hpp"
#include "Tensor.hpp"

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled() { return false; }

namespace
{
void AssertNear(float actual, float expected)
{
    assert(std::fabs(actual - expected) < 1.0e-6f);
}

Feature BarAt(std::size_t index, float high, float low, float close)
{
    Feature bar{close, close, high, low,
                PriceTP{std::chrono::seconds{static_cast<long long>(index * 900)}}};
    bar.tickVolume = 100.0f;
    return bar;
}

float RowValue(const Tensor& tensor, std::size_t row, std::size_t column)
{
    return MetaNN::LowerAccess(*(tensor.begin() + static_cast<std::ptrdiff_t>(row)))
        .RawMemory()[column];
}

void RetainBaseline(CausalMultiBarRangePressure& feature, std::size_t count = 15)
{
    for (std::size_t index = 0; index < count; ++index)
        (void)feature.AddCompletedBar(20.0f, 10.0f, 15.0f);
}

void TestDefinitionAndNumericalSafety()
{
    CausalMultiBarRangePressure atHigh;
    RetainBaseline(atHigh);
    AssertNear(atHigh.AddCompletedBar(30.0f, 10.0f, 30.0f), 1.0f);

    CausalMultiBarRangePressure atLow;
    RetainBaseline(atLow);
    AssertNear(atLow.AddCompletedBar(20.0f, 0.0f, 0.0f), -1.0f);

    CausalMultiBarRangePressure atMidpoint;
    RetainBaseline(atMidpoint);
    AssertNear(atMidpoint.AddCompletedBar(30.0f, 10.0f, 20.0f), 0.0f);

    CausalMultiBarRangePressure flat;
    for (std::size_t index = 0; index < CausalMultiBarRangePressure::kBarWindow;
         ++index)
        assert(flat.AddCompletedBar(5.0f, 5.0f, 5.0f) == 0.0f);

    CausalMultiBarRangePressure tiny;
    const float next = std::nextafter(1.0f, 2.0f);
    assert(tiny.AddCompletedBar(next, 1.0f, next) == 0.0f);

    CausalMultiBarRangePressure extreme;
    const float max = std::numeric_limits<float>::max();
    const float extremeValue = extreme.AddCompletedBar(max, 0.0f, max);
    assert(std::isfinite(extremeValue));
    AssertNear(extremeValue, 1.0f);

    CausalMultiBarRangePressure nonfinite;
    assert(nonfinite.AddCompletedBar(20.0f, 10.0f,
                                     std::numeric_limits<float>::quiet_NaN()) == 0.0f);
}

void TestCausalWindowWarmupAndEviction()
{
    CausalMultiBarRangePressure feature;
    // The repository's causal startup policy uses the available prefix. The
    // first bar is therefore current-only, then the window grows to 16 bars.
    assert(feature.RetainedBarCount() == 0);
    AssertNear(feature.AddCompletedBar(20.0f, 10.0f, 15.0f), 0.0f);
    assert(feature.RetainedBarCount() == 1);

    CausalMultiBarRangePressure eviction;
    (void)eviction.AddCompletedBar(100.0f, 0.0f, 50.0f);
    for (std::size_t index = 1; index < 15; ++index)
        (void)eviction.AddCompletedBar(10.0f, 0.0f, 5.0f);
    // At t=15, the original high remains in the 16-bar inclusive window.
    AssertNear(eviction.AddCompletedBar(10.0f, 0.0f, 5.0f), -0.9f);
    assert(eviction.RetainedBarCount() == CausalMultiBarRangePressure::kBarWindow);
    // At t=16, t=0 falls out and only bars t=1..16 are consulted.
    AssertNear(eviction.AddCompletedBar(10.0f, 0.0f, 10.0f), 1.0f);

    CausalMultiBarRangePressure oldHigh;
    CausalMultiBarRangePressure differentOldHigh;
    (void)oldHigh.AddCompletedBar(100.0f, 0.0f, 50.0f);
    (void)differentOldHigh.AddCompletedBar(1000.0f, 0.0f, 500.0f);
    for (std::size_t index = 1; index < 16; ++index)
    {
        (void)oldHigh.AddCompletedBar(10.0f, 0.0f, 5.0f);
        (void)differentOldHigh.AddCompletedBar(10.0f, 0.0f, 5.0f);
    }
    AssertNear(oldHigh.AddCompletedBar(10.0f, 0.0f, 10.0f),
               differentOldHigh.AddCompletedBar(10.0f, 0.0f, 10.0f));
}

void TestTensorIntegrationCausalityAndAblation()
{
    static_assert(causalMultiBarRangePressureCol == 45);
    static_assert(feature_size == 48);
    static_assert(EA::kCausalDirectionalAdverseExcursionModelInputWidth == 49);
    static_assert(EA::kCausalMultiBarRangePressureModelInputWidth == 50);
    static_assert(EA::kCurrentModelInputWidth == 52);

    Tensor tensor{"causal-multi-bar-range-pressure"};
    for (std::size_t row = 0; row < 15; ++row)
        tensor.Add(BarAt(row, 20.0f, 10.0f, 15.0f));
    tensor.Add(BarAt(15, 30.0f, 10.0f, 30.0f));
    AssertNear(RowValue(tensor, 0, causalMultiBarRangePressureCol), 0.0f);
    AssertNear(RowValue(tensor, 15, causalMultiBarRangePressureCol), 1.0f);

    Tensor left{"range-pressure-causality-left"};
    Tensor right{"range-pressure-causality-right"};
    for (std::size_t row = 0; row <= 16; ++row)
    {
        left.Add(BarAt(row, 20.0f, 10.0f, 15.0f));
        right.Add(BarAt(row, 20.0f, 10.0f, 15.0f));
    }
    const float valueAtT = RowValue(left, 16, causalMultiBarRangePressureCol);
    left.Add(BarAt(17, 1000000.0f, 0.001f, 1000000.0f));
    right.Add(BarAt(17, 11.0f, 10.0f, 10.0f));
    AssertNear(valueAtT, RowValue(left, 16, causalMultiBarRangePressureCol));
    AssertNear(valueAtT, RowValue(right, 16, causalMultiBarRangePressureCol));

    std::array<float, feature_size> source{};
    for (std::size_t column = 0; column < source.size(); ++column)
        source[column] = static_cast<float>(100 + column);
    const auto historical = EA::ResolveModelInputContract(49, source.size());
    assert(historical.tensorFeatureCount == causalMultiBarRangePressureCol);
    std::array<float, EA::kCausalDirectionalAdverseExcursionModelInputWidth>
        historicalInput{};
    historicalInput.fill(-1.0f);
    EA::CopyTensorFeaturesForModelInput(historicalInput.data(), source.data(), historical);
    assert(historicalInput[causalDirectionalAdverseExcursionCol] ==
           source[causalDirectionalAdverseExcursionCol]);

    const auto current = EA::ResolveModelInputContract(
        EA::kCurrentModelInputWidth, source.size());
    std::array<float, EA::kCurrentModelInputWidth> ablatedInput{};
    const auto mask = EA::FeatureAblationMask::Parse("multi_bar_range_pressure");
    assert(mask.CanonicalText() == "multi_bar_range_pressure");
    EA::CopyTensorFeaturesForModelInput(ablatedInput.data(), source.data(), current, mask);
    for (std::size_t column = 0; column < feature_size; ++column)
        assert(ablatedInput[column] ==
               (column == causalMultiBarRangePressureCol ? 0.0f : source[column]));
}
} // namespace

int main()
{
    TestDefinitionAndNumericalSafety();
    TestCausalWindowWarmupAndEviction();
    TestTensorIntegrationCausalityAndAblation();
    return 0;
}
