#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>

#include <MetaNN/data/facilities/lower_access.h>

#include "CausalReturnSignPersistenceFeatures.hpp"
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

void Retain(CausalReturnSignPersistenceFeatures& feature,
            const std::initializer_list<float>& closes)
{
    for (const float close : closes) feature.RetainCompletedClose(close);
}

void TestDefinitionAndState()
{
    CausalReturnSignPersistenceFeatures feature;
    assert(feature.PriorReturnSignPersistence() == 0.0f);
    for (std::size_t count = 1; count < 8; ++count)
    {
        feature.RetainCompletedClose(static_cast<float>(count));
        assert(feature.PriorReturnSignPersistence() == 0.0f);
    }
    feature.RetainCompletedClose(8.0f);
    AssertNear(feature.PriorReturnSignPersistence(), 1.0f);

    CausalReturnSignPersistenceFeatures decreasing;
    Retain(decreasing, {8, 7, 6, 5, 4, 3, 2, 1});
    AssertNear(decreasing.PriorReturnSignPersistence(), 1.0f);

    CausalReturnSignPersistenceFeatures alternating;
    Retain(alternating, {1, 2, 1, 2, 1, 2, 1, 2});
    AssertNear(alternating.PriorReturnSignPersistence(), -1.0f);

    CausalReturnSignPersistenceFeatures positiveMixed;
    Retain(positiveMixed, {0, 1, 2, 1, 2, 3, 4, 5});
    AssertNear(positiveMixed.PriorReturnSignPersistence(), 1.0f / 3.0f);

    CausalReturnSignPersistenceFeatures negativeMixed;
    Retain(negativeMixed, {0, 1, 0, 1, 0, 1, 2, 1});
    AssertNear(negativeMixed.PriorReturnSignPersistence(), -2.0f / 3.0f);

    CausalReturnSignPersistenceFeatures zero;
    Retain(zero, {1, 2, 2, 1, 1, 2, 2, 1});
    AssertNear(zero.PriorReturnSignPersistence(), 0.0f);

    CausalReturnSignPersistenceFeatures identical;
    Retain(identical, {1, 1, 1, 1, 1, 1, 1, 1});
    assert(identical.PriorReturnSignPersistence() == 0.0f);

    CausalReturnSignPersistenceFeatures nonfinite;
    Retain(nonfinite, {1, 2, 3, std::numeric_limits<float>::quiet_NaN(),
                       5, 6, 7, 8});
    assert(nonfinite.PriorReturnSignPersistence() == 0.0f);

    CausalReturnSignPersistenceFeatures evicted;
    Retain(evicted, {100, 1, 2, 3, 4, 5, 6, 7});
    evicted.RetainCompletedClose(8.0f);
    AssertNear(evicted.PriorReturnSignPersistence(), 1.0f);
    evicted.RetainCompletedClose(9.0f);
    AssertNear(evicted.PriorReturnSignPersistence(), 1.0f);

    CausalReturnSignPersistenceFeatures olderLeft;
    CausalReturnSignPersistenceFeatures olderRight;
    Retain(olderLeft, {-100, 1, 2, 3, 4, 5, 6, 7, 8});
    Retain(olderRight, {100, 1, 2, 3, 4, 5, 6, 7, 8});
    AssertNear(olderLeft.PriorReturnSignPersistence(),
               olderRight.PriorReturnSignPersistence());

    for (const float value : {feature.PriorReturnSignPersistence(),
                              decreasing.PriorReturnSignPersistence(),
                              alternating.PriorReturnSignPersistence(),
                              positiveMixed.PriorReturnSignPersistence(),
                              negativeMixed.PriorReturnSignPersistence()})
        assert(std::isfinite(value) && value >= -1.0f && value <= 1.0f);
}

Feature BarAt(std::size_t index, float close)
{
    Feature bar{close - 0.1f, close, close + 0.2f, close - 0.2f,
                PriceTP{std::chrono::seconds{static_cast<long long>(index * 900)}}};
    bar.tickVolume = 100.0f;
    return bar;
}

float RowValue(const Tensor& tensor, std::size_t row, std::size_t column)
{
    return MetaNN::LowerAccess(*(tensor.begin() + static_cast<std::ptrdiff_t>(row)))
        .RawMemory()[column];
}

void TestTensorIntegrationAndContracts()
{
    static_assert(causalReturnSignPersistenceCol == 42);
    static_assert(return_autocorrelation_feature_size == 49);
    static_assert(feature_size == 67);
    static_assert(EA::kCausalDirectionalPersistenceModelInputWidth == 46);
    static_assert(EA::kPreEconomicEventModelInputWidth == 53);
    static_assert(EA::kCurrentModelInputWidth == 71);

    Tensor tensor{"causal-return-sign-persistence"};
    for (std::size_t index = 0; index < 8; ++index)
        tensor.Add(BarAt(index, static_cast<float>(index + 1)));
    for (std::size_t index = 0; index < 8; ++index)
        assert(RowValue(tensor, index, causalReturnSignPersistenceCol) == 0.0f);
    tensor.Add(BarAt(8, 5000.0f));
    AssertNear(RowValue(tensor, 8, causalReturnSignPersistenceCol), 1.0f);

    Tensor left{"causal-current-exclusion-left"};
    Tensor right{"causal-current-exclusion-right"};
    for (std::size_t index = 0; index < 8; ++index)
    {
        left.Add(BarAt(index, static_cast<float>(index + 1)));
        right.Add(BarAt(index, static_cast<float>(index + 1)));
    }
    left.Add(BarAt(8, 5000.0f));
    right.Add(BarAt(8, -5000.0f));
    AssertNear(RowValue(left, 8, causalReturnSignPersistenceCol),
               RowValue(right, 8, causalReturnSignPersistenceCol));
    for (std::size_t column = 0; column < causalReturnSignPersistenceCol; ++column)
        assert(std::isfinite(RowValue(tensor, 8, column)));

    std::array<float, feature_size> source{};
    for (std::size_t index = 0; index < source.size(); ++index)
        source[index] = static_cast<float>(index);
    const auto v13 = EA::ResolveModelInputContract(46, source.size());
    std::array<float, 46> v13Output{};
    v13Output.fill(-1.0f);
    EA::CopyTensorFeaturesForModelInput(v13Output.data(), source.data(), v13);
    assert(v13.tensorFeatureCount == 42);
    assert(v13Output[causalDirectionalPersistenceCol] == 41.0f);
    assert(v13Output[causalReturnSignPersistenceCol] == -1.0f);

    const auto v14 = EA::ResolveModelInputContract(47, source.size());
    std::array<float, 47> v14Output{};
    EA::CopyTensorFeaturesForModelInput(v14Output.data(), source.data(), v14);
    assert(v14.tensorFeatureCount == causalReturnDirectionImbalanceCol);
    assert(v14Output[causalReturnSignPersistenceCol] == 42.0f);
}
} // namespace

int main()
{
    TestDefinitionAndState();
    TestTensorIntegrationAndContracts();
    return 0;
}
