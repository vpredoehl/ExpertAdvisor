#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>

#include <MetaNN/data/facilities/lower_access.h>

#include "CausalReturnDirectionImbalanceFeatures.hpp"
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

void Retain(CausalReturnDirectionImbalanceFeatures& feature,
            const std::initializer_list<float>& closes)
{
    for (const float close : closes) feature.RetainCompletedClose(close);
}

void TestDefinitionAndCausality()
{
    CausalReturnDirectionImbalanceFeatures feature;
    assert(feature.PriorReturnDirectionImbalance() == 0.0f);
    for (std::size_t count = 1; count < 8; ++count)
    {
        feature.RetainCompletedClose(static_cast<float>(count));
        assert(feature.PriorReturnDirectionImbalance() == 0.0f);
    }
    feature.RetainCompletedClose(8.0f);
    AssertNear(feature.PriorReturnDirectionImbalance(), 1.0f);

    CausalReturnDirectionImbalanceFeatures decreasing;
    Retain(decreasing, {8, 7, 6, 5, 4, 3, 2, 1});
    AssertNear(decreasing.PriorReturnDirectionImbalance(), -1.0f);

    CausalReturnDirectionImbalanceFeatures positiveMixed;
    Retain(positiveMixed, {0, 1, 2, 3, 4, 5, 4, 3});
    AssertNear(positiveMixed.PriorReturnDirectionImbalance(), 3.0f / 7.0f);

    CausalReturnDirectionImbalanceFeatures negativeMixed;
    Retain(negativeMixed, {0, 1, 2, 1, 0, -1, -2, -3});
    AssertNear(negativeMixed.PriorReturnDirectionImbalance(), -3.0f / 7.0f);

    CausalReturnDirectionImbalanceFeatures zeroMoves;
    Retain(zeroMoves, {0, 1, 1, 2, 2, 1, 1, 0});
    assert(zeroMoves.PriorReturnDirectionImbalance() == 0.0f);

    CausalReturnDirectionImbalanceFeatures identical;
    Retain(identical, {1, 1, 1, 1, 1, 1, 1, 1});
    assert(identical.PriorReturnDirectionImbalance() == 0.0f);

    CausalReturnDirectionImbalanceFeatures nonfinite;
    Retain(nonfinite, {1, 2, 3, std::numeric_limits<float>::quiet_NaN(),
                       5, 6, 7, 8});
    assert(nonfinite.PriorReturnDirectionImbalance() == 0.0f);

    CausalReturnDirectionImbalanceFeatures left;
    CausalReturnDirectionImbalanceFeatures right;
    Retain(left, {1, 2, 3, 4, 5, 6, 7, 8});
    Retain(right, {1, 2, 3, 4, 5, 6, 7, 8});
    AssertNear(left.PriorReturnDirectionImbalance(), right.PriorReturnDirectionImbalance());
    left.RetainCompletedClose(9.0f);
    right.RetainCompletedClose(-9.0f);
    AssertNear(left.PriorReturnDirectionImbalance(), 1.0f);
    AssertNear(right.PriorReturnDirectionImbalance(), 5.0f / 7.0f);

    CausalReturnDirectionImbalanceFeatures olderLeft;
    CausalReturnDirectionImbalanceFeatures olderRight;
    Retain(olderLeft, {-100, 1, 2, 3, 4, 5, 6, 7, 8});
    Retain(olderRight, {100, 1, 2, 3, 4, 5, 6, 7, 8});
    AssertNear(olderLeft.PriorReturnDirectionImbalance(),
               olderRight.PriorReturnDirectionImbalance());

    for (const float value : {feature.PriorReturnDirectionImbalance(),
                              decreasing.PriorReturnDirectionImbalance(),
                              positiveMixed.PriorReturnDirectionImbalance(),
                              negativeMixed.PriorReturnDirectionImbalance(),
                              nonfinite.PriorReturnDirectionImbalance()})
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
    static_assert(causalReturnDirectionImbalanceCol == 43);
    static_assert(feature_size == 49);
    static_assert(EA::kCausalReturnSignPersistenceModelInputWidth == 47);
    static_assert(EA::kCausalReturnDirectionImbalanceModelInputWidth == 48);
    static_assert(EA::kCurrentModelInputWidth == 53);

    Tensor tensor{"causal-return-direction-imbalance"};
    for (std::size_t index = 0; index < 8; ++index)
        tensor.Add(BarAt(index, static_cast<float>(index + 1)));
    for (std::size_t index = 0; index < 8; ++index)
        assert(RowValue(tensor, index, causalReturnDirectionImbalanceCol) == 0.0f);
    tensor.Add(BarAt(8, -5000.0f));
    AssertNear(RowValue(tensor, 8, causalReturnDirectionImbalanceCol), 1.0f);

    Tensor left{"causal-current-exclusion-left"};
    Tensor right{"causal-current-exclusion-right"};
    for (std::size_t index = 0; index < 8; ++index)
    {
        left.Add(BarAt(index, static_cast<float>(index + 1)));
        right.Add(BarAt(index, static_cast<float>(index + 1)));
    }
    left.Add(BarAt(8, 5000.0f));
    right.Add(BarAt(8, -5000.0f));
    AssertNear(RowValue(left, 8, causalReturnDirectionImbalanceCol),
               RowValue(right, 8, causalReturnDirectionImbalanceCol));

    std::array<float, feature_size> source{};
    for (std::size_t index = 0; index < source.size(); ++index)
        source[index] = static_cast<float>(index);
    const auto v14 = EA::ResolveModelInputContract(47, source.size());
    std::array<float, 47> v14Output{};
    v14Output.fill(-1.0f);
    EA::CopyTensorFeaturesForModelInput(v14Output.data(), source.data(), v14);
    assert(v14.tensorFeatureCount == 43);
    assert(v14Output[causalReturnSignPersistenceCol] == 42.0f);
    assert(v14Output[causalReturnDirectionImbalanceCol] == -1.0f);

    const auto v15 = EA::ResolveModelInputContract(48, source.size());
    std::array<float, 49> v15Output{};
    EA::CopyTensorFeaturesForModelInput(v15Output.data(), source.data(), v15);
    assert(v15.tensorFeatureCount == causalDirectionalAdverseExcursionCol);
    assert(v15Output[causalReturnDirectionImbalanceCol] == 43.0f);
    assert(v15Output[causalDirectionalAdverseExcursionCol] == 0.0f);
}
} // namespace

int main()
{
    TestDefinitionAndCausality();
    TestTensorIntegrationAndContracts();
    return 0;
}
