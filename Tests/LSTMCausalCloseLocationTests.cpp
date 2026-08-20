#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>

#include <MetaNN/data/facilities/lower_access.h>

#include "CausalCloseLocationFeatures.hpp"
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

void Retain(CausalCloseLocationFeatures& feature, float high, float low, float close)
{
    feature.RetainCompletedBar(high, low, close);
}

void TestDefinitionDegeneratesAndFiniteness()
{
    CausalCloseLocationFeatures feature;
    assert(feature.PriorCloseLocation() == 0.0f);

    Retain(feature, 1.4f, 0.8f, 1.4f);
    AssertNear(feature.PriorCloseLocation(), 1.0f);
    Retain(feature, 1.4f, 0.8f, 0.8f);
    AssertNear(feature.PriorCloseLocation(), -1.0f);
    Retain(feature, 1.4f, 0.8f, 1.1f);
    AssertNear(feature.PriorCloseLocation(), 0.0f);
    Retain(feature, 1.4f, 0.8f, 1.25f);
    AssertNear(feature.PriorCloseLocation(), 0.5f);
    Retain(feature, 1.4f, 0.8f, 0.95f);
    AssertNear(feature.PriorCloseLocation(), -0.5f);
    Retain(feature, 1.0f, 1.0f, 1.0f);
    assert(feature.PriorCloseLocation() == 0.0f);
    Retain(feature, 0.8f, 1.4f, 1.0f);
    assert(feature.PriorCloseLocation() == 0.0f);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    for (const auto values : std::array<std::array<float, 3>, 4>{{
             {{nan, 0.8f, 1.1f}},
             {{1.4f, nan, 1.1f}},
             {{1.4f, 0.8f, nan}},
             {{infinity, 0.8f, 1.1f}}}})
    {
        Retain(feature, values[0], values[1], values[2]);
        assert(feature.PriorCloseLocation() == 0.0f);
        assert(std::isfinite(feature.PriorCloseLocation()));
    }

    // Finite malformed closes use the deterministic formula; they are not clipped.
    Retain(feature, 1.4f, 0.8f, 1.7f);
    AssertNear(feature.PriorCloseLocation(), 2.0f);
    Retain(feature, 1.4f, 0.8f, 0.5f);
    AssertNear(feature.PriorCloseLocation(), -2.0f);
}

void TestOneBarCausalityAndAdvancement()
{
    CausalCloseLocationFeatures left;
    CausalCloseLocationFeatures right;
    Retain(left, 1.4f, 0.8f, 1.25f);
    Retain(right, 1.4f, 0.8f, 1.25f);

    // Row-t output is fixed before either current bar becomes retained.
    AssertNear(left.PriorCloseLocation(), right.PriorCloseLocation());
    Retain(left, 1000.0f, -1000.0f, 900.0f);
    Retain(right, 20.0f, -20.0f, -5.0f);
    AssertNear(left.PriorCloseLocation(), 0.9f);
    AssertNear(right.PriorCloseLocation(), -0.25f);

    // Retention replaces, rather than accumulates, completed-bar state.
    Retain(left, 3.0f, 1.0f, 1.0f);
    AssertNear(left.PriorCloseLocation(), -1.0f);
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
    static_assert(causalCloseLocationCol == 40);
    static_assert(feature_size == 48);
    static_assert(EA::kCausalDirectionalRangeModelInputWidth == 44);
    static_assert(EA::kCurrentModelInputWidth == 52);

    Tensor tensor{"causal-close-location"};
    tensor.Add(BarAt(0, 1.0f, 1.4f, 0.8f, 1.25f));
    tensor.Add(BarAt(1, 100.0f, 1000.0f, 0.01f, 900.0f));
    const auto first = MetaNN::LowerAccess(*tensor.begin());
    const auto second = MetaNN::LowerAccess(*(tensor.begin() + 1));
    assert(first.RawMemory()[causalCloseLocationCol] == 0.0f);
    // The old 0..39 prefix remains the pre-increment first-row baseline.
    for (std::size_t column = 0; column < causalCloseLocationCol; ++column)
    {
        const float expected = column == sessionPhaseCosCol ? 1.0f : 0.0f;
        AssertNear(first.RawMemory()[column], expected);
    }
    AssertNear(second.RawMemory()[causalCloseLocationCol], 0.5f);
    // Existing prefix identity remains: the v11 directional-range feature is unchanged.
    AssertNear(second.RawMemory()[causalDirectionalRangeCol], 0.41666666f);
    for (std::size_t column = 0; column < causalCloseLocationCol; ++column)
        assert(std::isfinite(second.RawMemory()[column]));

    std::array<float, feature_size> source{};
    for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<float>(i);
    const auto historical = EA::ResolveModelInputContract(44, source.size());
    std::array<float, 45> historicalOutput{};
    historicalOutput.fill(-1.0f);
    EA::CopyTensorFeaturesForModelInput(
        historicalOutput.data(), source.data(), historical);
    assert(historicalOutput[causalDirectionalRangeCol] == 39.0f);
    assert(historicalOutput[causalCloseLocationCol] == -1.0f);

    const auto current = EA::ResolveModelInputContract(45, source.size());
    std::array<float, 45> currentOutput{};
    EA::CopyTensorFeaturesForModelInput(currentOutput.data(), source.data(), current);
    assert(currentOutput[causalCloseLocationCol] == 40.0f);
    assert(currentOutput[causalDirectionalPersistenceCol] == 0.0f);
}
} // namespace

int main()
{
    TestDefinitionDegeneratesAndFiniteness();
    TestOneBarCausalityAndAdvancement();
    TestTensorPlacementAndModelProjection();
    return 0;
}
