#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>

#include <MetaNN/data/facilities/lower_access.h>

#include "CausalDirectionalAdverseExcursionFeatures.hpp"
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

void Retain(CausalDirectionalAdverseExcursionFeatures& feature,
            const std::initializer_list<float>& closes)
{
    for (const float close : closes) feature.RetainCompletedClose(close);
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

void TestDefinitionWarmupAndInvalidData()
{
    CausalDirectionalAdverseExcursionFeatures feature;
    assert(feature.PriorDirectionalAdverseExcursion() == 0.0f);
    for (std::size_t count = 1; count < 8; ++count)
    {
        feature.RetainCompletedClose(static_cast<float>(count));
        assert(feature.PriorDirectionalAdverseExcursion() == 0.0f);
    }
    feature.RetainCompletedClose(8.0f);
    assert(feature.PriorDirectionalAdverseExcursion() == 0.0f);

    CausalDirectionalAdverseExcursionFeatures decreasing;
    Retain(decreasing, {8, 7, 6, 5, 4, 3, 2, 1});
    assert(decreasing.PriorDirectionalAdverseExcursion() == 0.0f);

    CausalDirectionalAdverseExcursionFeatures positiveInitial;
    Retain(positiveInitial, {10, 8, 9, 11, 12, 13, 14, 15});
    AssertNear(positiveInitial.PriorDirectionalAdverseExcursion(), 2.0f / 9.0f);

    CausalDirectionalAdverseExcursionFeatures negativeInitial;
    Retain(negativeInitial, {10, 12, 11, 9, 8, 7, 6, 5});
    AssertNear(negativeInitial.PriorDirectionalAdverseExcursion(), 2.0f / 9.0f);

    CausalDirectionalAdverseExcursionFeatures positiveLater;
    Retain(positiveLater, {10, 11, 9, 8, 12, 13, 14, 15});
    // path = 1+2+1+4+1+1+1 = 11; the later minimum at 8 is adverse 2.
    AssertNear(positiveLater.PriorDirectionalAdverseExcursion(), 2.0f / 11.0f);

    CausalDirectionalAdverseExcursionFeatures negativeLater;
    Retain(negativeLater, {10, 9, 11, 12, 8, 7, 6, 5});
    AssertNear(negativeLater.PriorDirectionalAdverseExcursion(), 2.0f / 11.0f);

    CausalDirectionalAdverseExcursionFeatures netZero;
    Retain(netZero, {10, 4, 12, 3, 11, 2, 13, 10});
    assert(netZero.PriorDirectionalAdverseExcursion() == 0.0f);

    CausalDirectionalAdverseExcursionFeatures identical;
    Retain(identical, {1, 1, 1, 1, 1, 1, 1, 1});
    assert(identical.PriorDirectionalAdverseExcursion() == 0.0f);

    CausalDirectionalAdverseExcursionFeatures nonfinite;
    Retain(nonfinite, {1, 2, 3, std::numeric_limits<float>::quiet_NaN(),
                       5, 6, 7, 8});
    assert(nonfinite.PriorDirectionalAdverseExcursion() == 0.0f);

    for (const float value : {positiveInitial.PriorDirectionalAdverseExcursion(),
                              negativeInitial.PriorDirectionalAdverseExcursion(),
                              positiveLater.PriorDirectionalAdverseExcursion(),
                              negativeLater.PriorDirectionalAdverseExcursion(),
                              nonfinite.PriorDirectionalAdverseExcursion()})
    {
        assert(std::isfinite(value));
        assert(value >= 0.0f && value <= 1.0f);
    }
}

void TestCausalityAdvancementAndEviction()
{
    CausalDirectionalAdverseExcursionFeatures left;
    CausalDirectionalAdverseExcursionFeatures right;
    Retain(left, {10, 8, 9, 11, 12, 13, 14, 15});
    Retain(right, {10, 8, 9, 11, 12, 13, 14, 15});
    const float before = left.PriorDirectionalAdverseExcursion();
    AssertNear(before, right.PriorDirectionalAdverseExcursion());

    // The current close cannot alter the value already fixed for row t.
    left.RetainCompletedClose(10000.0f);
    right.RetainCompletedClose(-10000.0f);
    assert(left.PriorDirectionalAdverseExcursion() != right.PriorDirectionalAdverseExcursion());

    CausalDirectionalAdverseExcursionFeatures evicted;
    Retain(evicted, {100, 10, 8, 9, 11, 12, 13, 14});
    evicted.RetainCompletedClose(15.0f);
    AssertNear(evicted.PriorDirectionalAdverseExcursion(), 2.0f / 9.0f);

    CausalDirectionalAdverseExcursionFeatures oldHistoryA;
    CausalDirectionalAdverseExcursionFeatures oldHistoryB;
    Retain(oldHistoryA, {999, 10, 8, 9, 11, 12, 13, 14});
    Retain(oldHistoryB, {-999, 10, 8, 9, 11, 12, 13, 14});
    oldHistoryA.RetainCompletedClose(15.0f);
    oldHistoryB.RetainCompletedClose(15.0f);
    AssertNear(oldHistoryA.PriorDirectionalAdverseExcursion(),
               oldHistoryB.PriorDirectionalAdverseExcursion());
}

void TestTensorIntegrationAndContracts()
{
    static_assert(causalDirectionalAdverseExcursionCol == 44);
    static_assert(return_autocorrelation_feature_size == 49);
    static_assert(feature_size == 59);
    static_assert(EA::kCausalReturnDirectionImbalanceModelInputWidth == 48);
    static_assert(EA::kCausalDirectionalAdverseExcursionModelInputWidth == 49);
    static_assert(EA::kPreEconomicEventModelInputWidth == 53);
    static_assert(EA::kCurrentModelInputWidth == 63);

    Tensor tensor{"causal-directional-adverse-excursion"};
    const std::array<float, 9> closes{10, 8, 9, 11, 12, 13, 14, 15, 9999};
    for (std::size_t row = 0; row < closes.size(); ++row)
        tensor.Add(BarAt(row, closes[row]));
    for (std::size_t row = 0; row < 8; ++row)
        assert(RowValue(tensor, row, causalDirectionalAdverseExcursionCol) == 0.0f);
    AssertNear(RowValue(tensor, 8, causalDirectionalAdverseExcursionCol), 2.0f / 9.0f);
    for (std::size_t column = 0; column < causalDirectionalAdverseExcursionCol; ++column)
        assert(std::isfinite(RowValue(tensor, 8, column)));

    Tensor left{"causality-left"};
    Tensor right{"causality-right"};
    for (std::size_t row = 0; row < 8; ++row)
    {
        left.Add(BarAt(row, closes[row]));
        right.Add(BarAt(row, closes[row]));
    }
    left.Add(BarAt(8, 10000.0f));
    right.Add(BarAt(8, -10000.0f));
    AssertNear(RowValue(left, 8, causalDirectionalAdverseExcursionCol),
               RowValue(right, 8, causalDirectionalAdverseExcursionCol));

    std::array<float, feature_size> source{};
    for (std::size_t index = 0; index < source.size(); ++index)
        source[index] = static_cast<float>(index);
    const auto historical = EA::ResolveModelInputContract(48, source.size());
    std::array<float, 49> historicalOutput{};
    historicalOutput.fill(-1.0f);
    EA::CopyTensorFeaturesForModelInput(historicalOutput.data(), source.data(), historical);
    assert(historical.tensorFeatureCount == 44);
    assert(historicalOutput[causalReturnDirectionImbalanceCol] == 43.0f);
    assert(historicalOutput[causalDirectionalAdverseExcursionCol] == -1.0f);
    const auto current = EA::ResolveModelInputContract(49, source.size());
    std::array<float, 49> currentOutput{};
    EA::CopyTensorFeaturesForModelInput(currentOutput.data(), source.data(), current);
    assert(current.tensorFeatureCount == 45);
    assert(currentOutput[causalDirectionalAdverseExcursionCol] == 44.0f);
}
} // namespace

int main()
{
    TestDefinitionWarmupAndInvalidData();
    TestCausalityAdvancementAndEviction();
    TestTensorIntegrationAndContracts();
    return 0;
}
