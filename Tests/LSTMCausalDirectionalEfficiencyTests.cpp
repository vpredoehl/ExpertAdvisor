#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>

#include <MetaNN/data/facilities/lower_access.h>

#include "CausalDirectionalPersistenceFeatures.hpp"
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

void Retain(CausalDirectionalPersistenceFeatures& feature,
            const std::initializer_list<float>& closes)
{
    for (const float close : closes) feature.RetainCompletedClose(close);
}

void TestDefinitionWarmupAndValidity()
{
    CausalDirectionalPersistenceFeatures feature;
    assert(feature.PriorDirectionalEfficiency() == 0.0f);
    for (std::size_t count = 1; count < 8; ++count)
    {
        feature.RetainCompletedClose(static_cast<float>(count));
        assert(feature.PriorDirectionalEfficiency() == 0.0f);
    }

    feature.RetainCompletedClose(8.0f);
    AssertNear(feature.PriorDirectionalEfficiency(), 1.0f);

    CausalDirectionalPersistenceFeatures identical;
    Retain(identical, {1, 1, 1, 1, 1, 1, 1, 1});
    assert(identical.PriorDirectionalEfficiency() == 0.0f);

    CausalDirectionalPersistenceFeatures decreasing;
    Retain(decreasing, {8, 7, 6, 5, 4, 3, 2, 1});
    AssertNear(decreasing.PriorDirectionalEfficiency(), -1.0f);

    CausalDirectionalPersistenceFeatures alternating;
    Retain(alternating, {1, 2, 1, 2, 1, 2, 1, 1});
    assert(alternating.PriorDirectionalEfficiency() == 0.0f);

    CausalDirectionalPersistenceFeatures positiveHalf;
    Retain(positiveHalf, {1, 2, 1, 2, 3, 3, 3, 3});
    AssertNear(positiveHalf.PriorDirectionalEfficiency(), 0.5f);

    CausalDirectionalPersistenceFeatures negativeHalf;
    Retain(negativeHalf, {3, 2, 3, 2, 1, 1, 1, 1});
    AssertNear(negativeHalf.PriorDirectionalEfficiency(), -0.5f);

    CausalDirectionalPersistenceFeatures flatThenMove;
    Retain(flatThenMove, {1, 1, 1, 1, 1, 1, 1, 2});
    AssertNear(flatThenMove.PriorDirectionalEfficiency(), 1.0f);

    CausalDirectionalPersistenceFeatures nonfinite;
    Retain(nonfinite, {1, 2, 3, std::numeric_limits<float>::quiet_NaN(),
                       5, 6, 7, 8});
    assert(nonfinite.PriorDirectionalEfficiency() == 0.0f);
    assert(std::isfinite(nonfinite.PriorDirectionalEfficiency()));
}

void TestCausalityEvictionAndBounds()
{
    CausalDirectionalPersistenceFeatures left;
    CausalDirectionalPersistenceFeatures right;
    Retain(left, {1, 2, 3, 4, 5, 6, 7, 8});
    Retain(right, {1, 2, 3, 4, 5, 6, 7, 8});
    AssertNear(left.PriorDirectionalEfficiency(), right.PriorDirectionalEfficiency());

    // Different current bars do not alter the already-fixed row-t value.
    left.RetainCompletedClose(9.0f);
    right.RetainCompletedClose(1.0f);
    AssertNear(left.PriorDirectionalEfficiency(), 1.0f);
    AssertNear(right.PriorDirectionalEfficiency(), -1.0f / 13.0f);

    // The ninth close evicts the oldest close exactly: [2..9] is monotonic.
    CausalDirectionalPersistenceFeatures evicted;
    Retain(evicted, {100, 1, 2, 3, 4, 5, 6, 7});
    assert(std::fabs(evicted.PriorDirectionalEfficiency()) < 1.0f);
    evicted.RetainCompletedClose(8.0f);
    AssertNear(evicted.PriorDirectionalEfficiency(), 1.0f);

    // Only the retained eight closes matter after repeated advancement.
    evicted.RetainCompletedClose(9.0f);
    AssertNear(evicted.PriorDirectionalEfficiency(), 1.0f);
    assert(std::fabs(evicted.PriorDirectionalEfficiency()) <= 1.0f + 1.0e-6f);
}

Feature BarAt(std::size_t index, float close)
{
    Feature bar{close - 0.1f, close, close + 0.2f, close - 0.2f,
                PriceTP{std::chrono::seconds{static_cast<long long>(index * 900)}}};
    bar.tickVolume = 100.0f;
    return bar;
}

float RowValue(const Tensor& tensor, std::size_t index, std::size_t column)
{
    return MetaNN::LowerAccess(*(tensor.begin() + static_cast<std::ptrdiff_t>(index)))
        .RawMemory()[column];
}

void TestTensorIntegrationAndProjection()
{
    static_assert(causalDirectionalPersistenceCol == 41);
    static_assert(feature_size == 49);
    static_assert(EA::kCausalCloseLocationModelInputWidth == 45);
    static_assert(EA::kCurrentModelInputWidth == 53);

    Tensor tensor{"causal-directional-efficiency"};
    for (std::size_t index = 0; index < 8; ++index)
        tensor.Add(BarAt(index, static_cast<float>(index + 1)));
    for (std::size_t index = 0; index < 8; ++index)
        assert(RowValue(tensor, index, causalDirectionalPersistenceCol) == 0.0f);
    tensor.Add(BarAt(8, 5000.0f));
    AssertNear(RowValue(tensor, 8, causalDirectionalPersistenceCol), 1.0f);

    // The 0..40 prefix remains untouched by this append-only feature.
    for (std::size_t column = 0; column < causalDirectionalPersistenceCol; ++column)
        assert(std::isfinite(RowValue(tensor, 8, column)));

    std::array<float, feature_size> source{};
    for (std::size_t index = 0; index < source.size(); ++index)
        source[index] = static_cast<float>(index);
    const auto v12 = EA::ResolveModelInputContract(45, source.size());
    std::array<float, 46> v12Output{};
    v12Output.fill(-1.0f);
    EA::CopyTensorFeaturesForModelInput(v12Output.data(), source.data(), v12);
    assert(v12Output[causalCloseLocationCol] == 40.0f);
    assert(v12Output[causalDirectionalPersistenceCol] == -1.0f);

    const auto v13 = EA::ResolveModelInputContract(46, source.size());
    std::array<float, 46> v13Output{};
    EA::CopyTensorFeaturesForModelInput(v13Output.data(), source.data(), v13);
    assert(v13Output[causalDirectionalPersistenceCol] == 41.0f);
    assert(v13Output[causalReturnSignPersistenceCol] == 0.0f);
}
} // namespace

int main()
{
    TestDefinitionWarmupAndValidity();
    TestCausalityEvictionAndBounds();
    TestTensorIntegrationAndProjection();
    return 0;
}
