#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>

#include <MetaNN/data/facilities/lower_access.h>

#include "CausalDirectionalRangeFeatures.hpp"
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

void Retain(CausalDirectionalRangeFeatures& feature,
            float open, float high, float low, float close)
{
    feature.RetainCompletedBar(open, high, low, close);
}

void TestDefinitionDegeneratesAndFiniteness()
{
    CausalDirectionalRangeFeatures feature;
    assert(feature.PriorDirectionalBodyRange() == 0.0f); // no prior bar

    Retain(feature, 1.0f, 1.4f, 0.8f, 1.3f);
    AssertNear(feature.PriorDirectionalBodyRange(), 0.5f);
    Retain(feature, 1.3f, 1.4f, 0.8f, 1.0f);
    AssertNear(feature.PriorDirectionalBodyRange(), -0.5f);
    Retain(feature, 1.1f, 1.4f, 0.8f, 1.1f);
    assert(feature.PriorDirectionalBodyRange() == 0.0f); // doji
    Retain(feature, 1.0f, 1.0f, 1.0f, 1.0f);
    assert(feature.PriorDirectionalBodyRange() == 0.0f); // zero range
    Retain(feature, 1.0f, 0.8f, 1.4f, 1.3f);
    assert(feature.PriorDirectionalBodyRange() == 0.0f); // negative range

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    for (const auto values : std::array<std::array<float, 4>, 5>{{
             {{nan, 1.4f, 0.8f, 1.3f}},
             {{1.0f, nan, 0.8f, 1.3f}},
             {{1.0f, 1.4f, nan, 1.3f}},
             {{1.0f, 1.4f, 0.8f, nan}},
             {{1.0f, infinity, 0.8f, 1.3f}}}})
    {
        Retain(feature, values[0], values[1], values[2], values[3]);
        assert(feature.PriorDirectionalBodyRange() == 0.0f);
        assert(std::isfinite(feature.PriorDirectionalBodyRange()));
    }

    // Malformed open/close values are intentionally not clipped when the
    // finite, positive high-low denominator remains mathematically valid.
    Retain(feature, 0.0f, 1.0f, 0.0f, 2.0f);
    AssertNear(feature.PriorDirectionalBodyRange(), 2.0f);
}

void TestOneBarCausalityAndAdvancement()
{
    CausalDirectionalRangeFeatures left;
    CausalDirectionalRangeFeatures right;
    Retain(left, 1.0f, 1.4f, 0.8f, 1.3f);
    Retain(right, 1.0f, 1.4f, 0.8f, 1.3f);

    // Current bars are not retained until after this value is fixed.
    AssertNear(left.PriorDirectionalBodyRange(), right.PriorDirectionalBodyRange());
    Retain(left, 10.0f, 1000.0f, -1000.0f, 900.0f);
    Retain(right, -10.0f, 20.0f, -20.0f, -5.0f);
    AssertNear(left.PriorDirectionalBodyRange(), 0.445f);
    AssertNear(right.PriorDirectionalBodyRange(), 0.125f);

    // Retention overwrites rather than accumulates: only one prior bar is used.
    Retain(left, 2.0f, 3.0f, 1.0f, 1.0f);
    AssertNear(left.PriorDirectionalBodyRange(), -0.5f);
}

Feature BarAt(std::size_t index, float open, float high, float low, float close)
{
    Feature bar{open, close, high, low,
                PriceTP{std::chrono::seconds{static_cast<long long>(index * 900)}}};
    bar.tickVolume = 100.0f;
    return bar;
}

void TestTensorPlacementAndModelProjection()
{
    static_assert(causalDirectionalRangeCol == 39);
    static_assert(return_autocorrelation_feature_size == 49);
    static_assert(feature_size == 71);
    static_assert(EA::kCausalVolatilityRegimeModelInputWidth == 43);
    static_assert(EA::kCausalDirectionalRangeModelInputWidth == 44);
    static_assert(EA::kPreEconomicEventModelInputWidth == 53);
    static_assert(EA::kCurrentModelInputWidth == 75);

    Tensor tensor{"causal-directional-range"};
    tensor.Add(BarAt(0, 1.0f, 1.4f, 0.8f, 1.3f));
    tensor.Add(BarAt(1, 100.0f, 1000.0f, 0.01f, 900.0f));
    const auto first = MetaNN::LowerAccess(*tensor.begin());
    const auto second = MetaNN::LowerAccess(*(tensor.begin() + 1));
    assert(first.RawMemory()[causalDirectionalRangeCol] == 0.0f);
    AssertNear(second.RawMemory()[causalDirectionalRangeCol], 0.5f);
    for (std::size_t column = 0; column < causalDirectionalRangeCol; ++column)
        assert(std::isfinite(second.RawMemory()[column]));

    std::array<float, feature_size> source{};
    for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<float>(i);
    const auto historical = EA::ResolveModelInputContract(43, source.size());
    std::array<float, 45> historicalOutput{};
    historicalOutput.fill(-1.0f);
    EA::CopyTensorFeaturesForModelInput(
        historicalOutput.data(), source.data(), historical);
    assert(historicalOutput[causalVolatilityRegimeCol] == 38.0f);
    assert(historicalOutput[causalDirectionalRangeCol] == -1.0f);

    const auto current = EA::ResolveModelInputContract(44, source.size());
    std::array<float, 45> currentOutput{};
    EA::CopyTensorFeaturesForModelInput(currentOutput.data(), source.data(), current);
    assert(currentOutput[causalDirectionalRangeCol] == 39.0f);
}
} // namespace

int main()
{
    TestDefinitionDegeneratesAndFiniteness();
    TestOneBarCausalityAndAdvancement();
    TestTensorPlacementAndModelProjection();
    return 0;
}
