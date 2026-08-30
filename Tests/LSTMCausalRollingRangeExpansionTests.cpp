#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>

#include <MetaNN/data/facilities/lower_access.h>

#include "CausalRollingRangeExpansionFeatures.hpp"
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

Feature BarAt(std::size_t index, float high, float low)
{
    const float close = (high + low) * 0.5f;
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

void TestDefinitionStartupAndNumericalSafety()
{
    static_assert(CausalRollingRangeExpansion::kShortLookback == 8);
    static_assert(CausalRollingRangeExpansion::kLongLookback == 32);

    CausalRollingRangeExpansion identical;
    for (std::size_t index = 0; index < 8; ++index)
        AssertNear(identical.AddCompletedBar(10.0f, 0.0f), 1.0f);

    CausalRollingRangeExpansion half;
    for (std::size_t index = 0; index < 24; ++index)
        (void)half.AddCompletedBar(20.0f, 0.0f);
    for (std::size_t index = 0; index < 8; ++index)
        (void)half.AddCompletedBar(15.0f, 5.0f);
    AssertNear(half.AddCompletedBar(15.0f, 5.0f), 0.0f);

    CausalRollingRangeExpansion compressed;
    (void)compressed.AddCompletedBar(100.0f, 0.0f);
    for (std::size_t index = 1; index < 24; ++index)
        (void)compressed.AddCompletedBar(50.0f, 50.0f);
    float compressedValue = 0.0f;
    for (std::size_t index = 24; index < 32; ++index)
        compressedValue = compressed.AddCompletedBar(51.0f, 49.0f);
    AssertNear(compressedValue, -0.96f);

    CausalRollingRangeExpansion startup;
    assert(startup.RetainedBarCount() == 0);
    assert(startup.AddCompletedBar(10.0f, 10.0f) == 0.0f);
    assert(startup.RetainedBarCount() == 1);
    AssertNear(startup.AddCompletedBar(20.0f, 10.0f), 1.0f);

    CausalRollingRangeExpansion flat;
    for (std::size_t index = 0; index < 32; ++index)
        assert(flat.AddCompletedBar(5.0f, 5.0f) == 0.0f);

    CausalRollingRangeExpansion tiny;
    const float next = std::nextafter(1.0f, 2.0f);
    assert(tiny.AddCompletedBar(next, 1.0f) == 0.0f);

    CausalRollingRangeExpansion extreme;
    const float extremeValue = extreme.AddCompletedBar(
        std::numeric_limits<float>::max(), 0.0f);
    assert(std::isfinite(extremeValue));
    AssertNear(extremeValue, 1.0f);

    CausalRollingRangeExpansion nonfinite;
    assert(nonfinite.AddCompletedBar(std::numeric_limits<float>::quiet_NaN(),
                                     0.0f) == 0.0f);
    assert(nonfinite.AddCompletedBar(20.0f, 10.0f) == 0.0f);
}

void TestCurrentBarInclusionAndExactEviction()
{
    CausalRollingRangeExpansion shortCurrent;
    for (std::size_t index = 0; index < 7; ++index)
        (void)shortCurrent.AddCompletedBar(20.0f, 10.0f);
    // The eighth, current completed bar expands the inclusive short envelope.
    AssertNear(shortCurrent.AddCompletedBar(30.0f, 10.0f), 1.0f);

    CausalRollingRangeExpansion longCurrent;
    for (std::size_t index = 0; index < 8; ++index)
        (void)longCurrent.AddCompletedBar(20.0f, 10.0f);
    // The current bar is also part of the inclusive long envelope.
    AssertNear(longCurrent.AddCompletedBar(100.0f, 0.0f), 1.0f);

    CausalRollingRangeExpansion shortEviction;
    (void)shortEviction.AddCompletedBar(100.0f, 0.0f);
    for (std::size_t index = 1; index < 8; ++index)
        (void)shortEviction.AddCompletedBar(20.0f, 10.0f);
    AssertNear(shortEviction.AddCompletedBar(20.0f, 10.0f), -0.8f);

    CausalRollingRangeExpansion longEviction;
    (void)longEviction.AddCompletedBar(100.0f, 0.0f);
    for (std::size_t index = 1; index < 32; ++index)
        (void)longEviction.AddCompletedBar(20.0f, 10.0f);
    AssertNear(longEviction.AddCompletedBar(20.0f, 10.0f), 1.0f);
    assert(longEviction.RetainedBarCount() == CausalRollingRangeExpansion::kLongLookback);
}

void TestTensorIntegrationCausalityAblationAndHistoricalWidth()
{
    static_assert(causalRollingRangeExpansionCol == 46);
    static_assert(return_autocorrelation_feature_size == 49);
    static_assert(feature_size == 71);
    static_assert(EA::kCausalMultiBarRangePressureModelInputWidth == 50);
    static_assert(EA::kCausalRollingRangeExpansionModelInputWidth == 51);
    static_assert(EA::kPreEconomicEventModelInputWidth == 53);
    static_assert(EA::kCurrentModelInputWidth == 75);

    Tensor tensor{"causal-rolling-range-expansion"};
    for (std::size_t row = 0; row < 8; ++row)
        tensor.Add(BarAt(row, 20.0f, 10.0f));
    tensor.Add(BarAt(8, 30.0f, 10.0f));
    AssertNear(RowValue(tensor, 0, causalRollingRangeExpansionCol), 1.0f);
    AssertNear(RowValue(tensor, 7, causalRollingRangeExpansionCol), 1.0f);
    AssertNear(RowValue(tensor, 8, causalRollingRangeExpansionCol), 1.0f);

    Tensor left{"rolling-range-causality-left"};
    Tensor right{"rolling-range-causality-right"};
    for (std::size_t row = 0; row <= 32; ++row)
    {
        left.Add(BarAt(row, 20.0f, 10.0f));
        right.Add(BarAt(row, 20.0f, 10.0f));
    }
    const float valueAtT = RowValue(left, 32, causalRollingRangeExpansionCol);
    left.Add(BarAt(33, 1000000.0f, 0.001f));
    right.Add(BarAt(33, 11.0f, 10.0f));
    AssertNear(valueAtT, RowValue(left, 32, causalRollingRangeExpansionCol));
    AssertNear(valueAtT, RowValue(right, 32, causalRollingRangeExpansionCol));

    std::array<float, feature_size> source{};
    for (std::size_t column = 0; column < source.size(); ++column)
        source[column] = static_cast<float>(100 + column);

    const auto historical = EA::ResolveModelInputContract(50, source.size());
    assert(historical.tensorFeatureCount == causalRollingRangeExpansionCol);
    std::array<float, EA::kCausalMultiBarRangePressureModelInputWidth> historicalInput{};
    historicalInput.fill(-1.0f);
    EA::CopyTensorFeaturesForModelInput(historicalInput.data(), source.data(), historical);
    assert(historicalInput[causalMultiBarRangePressureCol] ==
           source[causalMultiBarRangePressureCol]);

    const auto current = EA::ResolveModelInputContract(51, source.size());
    std::array<float, EA::kCurrentModelInputWidth> ablatedInput{};
    const auto mask = EA::FeatureAblationMask::Parse("rolling_range_expansion");
    assert(mask.CanonicalText() == "rolling_range_expansion");
    EA::CopyTensorFeaturesForModelInput(ablatedInput.data(), source.data(), current, mask);
    for (std::size_t column = 0;
         column < causal_rolling_range_expansion_feature_size; ++column)
        assert(ablatedInput[column] ==
               (column == causalRollingRangeExpansionCol ? 0.0f : source[column]));
}
} // namespace

int main()
{
    TestDefinitionStartupAndNumericalSafety();
    TestCurrentBarInclusionAndExactEviction();
    TestTensorIntegrationCausalityAblationAndHistoricalWidth();
    return 0;
}
