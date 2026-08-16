#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

#include <MetaNN/data/facilities/lower_access.h>

#include "CausalVolatilityRegimeFeatures.hpp"
#include "FeatureLayout.hpp"
#include "ModelInputContract.hpp"
#include "PricePoint.hpp"
#include "Tensor.hpp"

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled() { return false; }

namespace
{
void AssertNear(float actual, double expected)
{
    assert(std::fabs(static_cast<double>(actual) - expected) < 1.0e-4);
}

float Advance(float close, double logReturn)
{
    return static_cast<float>(static_cast<double>(close) * std::exp(logReturn));
}

void AddReturns(CausalVolatilityRegime8x32& regime, float& close,
                const std::initializer_list<double>& returns)
{
    for (const double value : returns)
    {
        close = Advance(close, value);
        (void)regime.AddCompletedClose(close);
    }
}

void TestCausalDefinition()
{
    static_assert(causalVolatilityRegimeShortLookback == 8);
    static_assert(causalVolatilityRegimeLongLookback == 32);
    CausalVolatilityRegime8x32 regime;
    float close = 100.0f;
    assert(regime.AddCompletedClose(close) == 0.0f); // first row/no history
    AddReturns(regime, close, {0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2});
    assert(regime.PriorReturnCount() == 8);
    close = Advance(close, 0.7);
    assert(regime.AddCompletedClose(close) == 0.0f); // <= 8: equal windows

    CausalVolatilityRegime8x32 exact;
    close = 100.0f;
    assert(exact.AddCompletedClose(close) == 0.0f);
    for (std::size_t i = 0; i < 24; ++i)
    {
        close = Advance(close, 0.1);
        (void)exact.AddCompletedClose(close);
    }
    for (std::size_t i = 0; i < 8; ++i)
    {
        close = Advance(close, 0.2);
        (void)exact.AddCompletedClose(close);
    }
    close = Advance(close, -0.9); // must not participate in this value
    AssertNear(exact.AddCompletedClose(close),
               0.2 / std::sqrt((24.0 * 0.1 * 0.1 + 8.0 * 0.2 * 0.2) / 32.0) - 1.0);

    CausalVolatilityRegime8x32 expanded;
    CausalVolatilityRegime8x32 compressed;
    float expandedClose = 100.0f;
    float compressedClose = 100.0f;
    expanded.AddCompletedClose(expandedClose);
    compressed.AddCompletedClose(compressedClose);
    for (std::size_t i = 0; i < 24; ++i)
    {
        expandedClose = Advance(expandedClose, 0.01);
        compressedClose = Advance(compressedClose, 0.2);
        expanded.AddCompletedClose(expandedClose);
        compressed.AddCompletedClose(compressedClose);
    }
    for (std::size_t i = 0; i < 8; ++i)
    {
        expandedClose = Advance(expandedClose, 0.2);
        compressedClose = Advance(compressedClose, 0.01);
        expanded.AddCompletedClose(expandedClose);
        compressed.AddCompletedClose(compressedClose);
    }
    expandedClose = Advance(expandedClose, 0.01);
    compressedClose = Advance(compressedClose, 0.01);
    assert(expanded.AddCompletedClose(expandedClose) > 0.0f);
    assert(compressed.AddCompletedClose(compressedClose) < 0.0f);
}

void TestStateBoundsCausalityAndFiniteness()
{
    CausalVolatilityRegime8x32 zero;
    assert(zero.AddCompletedClose(100.0f) == 0.0f);
    for (std::size_t i = 0; i < 40; ++i)
        assert(zero.AddCompletedClose(100.0f) == 0.0f);

    CausalVolatilityRegime8x32 left;
    CausalVolatilityRegime8x32 right;
    float a = 100.0f;
    float b = 100.0f;
    left.AddCompletedClose(a);
    right.AddCompletedClose(b);
    for (std::size_t i = 0; i < 12; ++i)
    {
        a = Advance(a, 0.05);
        b = Advance(b, 0.05);
        left.AddCompletedClose(a);
        right.AddCompletedClose(b);
    }
    AssertNear(left.AddCompletedClose(Advance(a, 0.8)),
               right.AddCompletedClose(Advance(b, -0.8)));

    CausalVolatilityRegime8x32 evicted;
    float close = 100.0f;
    evicted.AddCompletedClose(close);
    close = Advance(close, 0.5);
    evicted.AddCompletedClose(close);
    for (std::size_t i = 0; i < 32; ++i)
    {
        close = Advance(close, 0.01);
        (void)evicted.AddCompletedClose(close);
        assert(evicted.PriorReturnCount() <= 32);
    }
    AssertNear(evicted.AddCompletedClose(Advance(close, 0.01)), 0.0);

    CausalVolatilityRegime8x32 finite;
    for (const float value : {std::numeric_limits<float>::quiet_NaN(), 100.0f,
                              0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                              100.0f, 101.0f})
        assert(std::isfinite(finite.AddCompletedClose(value)));
}

Feature BarAt(std::size_t index, float close)
{
    Feature bar{close - 0.1f, close, close + 0.2f, close - 0.2f,
                PriceTP{std::chrono::seconds{static_cast<long long>(index * 900)}}};
    bar.tickVolume = 100.0f;
    return bar;
}

void TestTensorPlacementAndProjection()
{
    static_assert(causalVolatilityRegimeCol == 38);
    static_assert(feature_size == 45);
    static_assert(EA::kCausalVolatilityRegimeModelInputWidth == 43);
    Tensor tensor{"causal-volatility-regime"};
    float close = 100.0f;
    tensor.Add(BarAt(0, close));
    close = Advance(close, 0.1);
    tensor.Add(BarAt(1, close));
    const auto first = MetaNN::LowerAccess(*tensor.begin());
    assert(first.RawMemory()[causalVolatilityRegimeCol] == 0.0f);

    std::array<float, feature_size> source{};
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<float>(i);
    const auto historical = EA::ResolveModelInputContract(42, source.size());
    std::array<float, 43> historicalOutput{};
    historicalOutput.fill(-1.0f);
    EA::CopyTensorFeaturesForModelInput(historicalOutput.data(), source.data(), historical);
    assert(historicalOutput[causalReturnSurpriseCol] == 37.0f);
    assert(historicalOutput[causalVolatilityRegimeCol] == -1.0f);
    const auto current = EA::ResolveModelInputContract(43, source.size());
    std::array<float, 43> currentOutput{};
    EA::CopyTensorFeaturesForModelInput(currentOutput.data(), source.data(), current);
    assert(currentOutput[causalVolatilityRegimeCol] == 38.0f);
}
} // namespace

int main()
{
    TestCausalDefinition();
    TestStateBoundsCausalityAndFiniteness();
    TestTensorPlacementAndProjection();
    return 0;
}
