#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

#include <MetaNN/data/facilities/lower_access.h>

#include "CausalReturnSurpriseFeatures.hpp"
#include "FeatureLayout.hpp"
#include "ModelInputContract.hpp"
#include "PricePoint.hpp"
#include "ReturnFeatureHistory.hpp"
#include "Tensor.hpp"

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled()
{
    return false;
}

namespace
{
constexpr std::size_t kModelWidth = EA::kCurrentModelInputWidth;

void AssertNear(float actual, double expected)
{
    assert(std::fabs(static_cast<double>(actual) - expected) < 1.0e-4);
}

float Advance(float close, double logReturn)
{
    return static_cast<float>(static_cast<double>(close) * std::exp(logReturn));
}

void TestFeatureContract()
{
    static_assert(causalReturnSurpriseLookback == 32);
    CausalReturnSurprise32 surprise;
    float close = 100.0f;
    assert(surprise.AddCompletedClose(close) == 0.0f); // bootstrap
    close = Advance(close, 0.1);
    assert(surprise.AddCompletedClose(close) == 0.0f); // no predecessor return
    close = Advance(close, -0.2);
    AssertNear(surprise.AddCompletedClose(close), -2.0);
    close = Advance(close, 0.3);
    AssertNear(surprise.AddCompletedClose(close),
               0.3 / std::sqrt((0.1 * 0.1 + 0.2 * 0.2) / 2.0));

    // Nonzero predecessor mean proves this is RMS magnitude, not standard
    // deviation: sqrt(mean(0.1^2, 0.2^2)), without mean subtraction.
    CausalReturnSurprise32 rms;
    close = 100.0f;
    assert(rms.AddCompletedClose(close) == 0.0f);
    close = Advance(close, 0.1);
    assert(rms.AddCompletedClose(close) == 0.0f);
    close = Advance(close, 0.2);
    AssertNear(rms.AddCompletedClose(close), 0.2 / 0.1);
    close = Advance(close, 0.1);
    const double rmsDenominator = std::sqrt((0.1 * 0.1 + 0.2 * 0.2) / 2.0);
    AssertNear(rms.AddCompletedClose(close), 0.1 / rmsDenominator);
    assert(std::fabs(0.1 / rmsDenominator - 2.0) > 1.0);

    CausalReturnSurprise32 positiveClamp;
    close = 100.0f;
    positiveClamp.AddCompletedClose(close);
    close = Advance(close, 0.01);
    positiveClamp.AddCompletedClose(close);
    close = Advance(close, 0.2);
    assert(positiveClamp.AddCompletedClose(close) == 10.0f);

    CausalReturnSurprise32 negativeClamp;
    close = 100.0f;
    negativeClamp.AddCompletedClose(close);
    close = Advance(close, 0.01);
    negativeClamp.AddCompletedClose(close);
    close = Advance(close, -0.2);
    assert(negativeClamp.AddCompletedClose(close) == -10.0f);

    CausalReturnSurprise32 zeroDenominator;
    assert(zeroDenominator.AddCompletedClose(100.0f) == 0.0f);
    assert(zeroDenominator.AddCompletedClose(100.0f) == 0.0f);
    assert(zeroDenominator.AddCompletedClose(110.0f) == 0.0f);
}

void TestValidityAndRollingState()
{
    CausalReturnSurprise32 invalid;
    for (const float close : {
             std::numeric_limits<float>::quiet_NaN(), 100.0f, 0.0f,
             -1.0f, std::numeric_limits<float>::infinity(), 100.0f})
    {
        const float result = invalid.AddCompletedClose(close);
        assert(std::isfinite(result));
        assert(result == 0.0f);
    }
    assert(invalid.PriorReturnCount() == 0);

    assert(invalid.AddCompletedClose(100.0f) == 0.0f);
    assert(invalid.AddCompletedClose(101.0f) == 0.0f);
    assert(invalid.AddCompletedClose(102.0f) > 0.0f);
    assert(invalid.AddCompletedClose(
               std::numeric_limits<float>::quiet_NaN()) == 0.0f);

    CausalReturnSurprise32 rolling;
    float close = 100.0f;
    rolling.AddCompletedClose(close);
    close = Advance(close, 0.1);
    rolling.AddCompletedClose(close);
    for (std::size_t i = 0; i < 31; ++i)
    {
        close = Advance(close, 0.01);
        (void)rolling.AddCompletedClose(close);
    }
    assert(rolling.PriorReturnCount() == causalReturnSurpriseLookback);
    close = Advance(close, 0.01);
    AssertNear(rolling.AddCompletedClose(close),
               0.01 / std::sqrt((0.1 * 0.1 + 31.0 * 0.01 * 0.01) / 32.0));
    assert(rolling.PriorReturnCount() == causalReturnSurpriseLookback);
    close = Advance(close, 0.01);
    AssertNear(rolling.AddCompletedClose(close), 1.0);
    assert(rolling.PriorReturnCount() == causalReturnSurpriseLookback);
}

void TestCausalityAndFutureDependency()
{
    CausalReturnSurprise32 positive;
    CausalReturnSurprise32 negative;
    float baseClose = 100.0f;
    positive.AddCompletedClose(baseClose);
    negative.AddCompletedClose(baseClose);
    baseClose = Advance(baseClose, 0.1);
    positive.AddCompletedClose(baseClose);
    negative.AddCompletedClose(baseClose);
    baseClose = Advance(baseClose, -0.05);
    positive.AddCompletedClose(baseClose);
    negative.AddCompletedClose(baseClose);
    const float positiveCurrent = Advance(baseClose, 0.2);
    const float negativeCurrent = Advance(baseClose, -0.2);
    const float positiveFeature = positive.AddCompletedClose(positiveCurrent);
    const float negativeFeature = negative.AddCompletedClose(negativeCurrent);
    AssertNear(positiveFeature / 0.2,
               negativeFeature / -0.2);

    CausalReturnSurprise32 futureA;
    CausalReturnSurprise32 futureB;
    float a = 100.0f;
    float b = 100.0f;
    futureA.AddCompletedClose(a);
    futureB.AddCompletedClose(b);
    a = Advance(a, 0.1);
    b = Advance(b, 0.1);
    futureA.AddCompletedClose(a);
    futureB.AddCompletedClose(b);
    a = Advance(a, 0.2);
    b = Advance(b, 0.4);
    futureA.AddCompletedClose(a);
    futureB.AddCompletedClose(b);
    a = Advance(a, 0.1);
    b = Advance(b, 0.1);
    const float nextA = futureA.AddCompletedClose(a);
    const float nextB = futureB.AddCompletedClose(b);
    AssertNear(nextA, 0.1 / std::sqrt((0.1 * 0.1 + 0.2 * 0.2) / 2.0));
    AssertNear(nextB, 0.1 / std::sqrt((0.1 * 0.1 + 0.4 * 0.4) / 2.0));
    assert(nextA != nextB);
}

Feature BarAt(std::size_t index, float close)
{
    Feature bar{close - 0.1f, close, close + 0.2f, close - 0.2f,
                PriceTP{std::chrono::seconds{
                    static_cast<long long>(index * 15 * 60)}}};
    bar.tickVolume = 100.0f;
    return bar;
}

std::array<float, feature_size> BaseRowAt(const Tensor& tensor, std::size_t index)
{
    const auto row = *(tensor.begin() + static_cast<std::ptrdiff_t>(index));
    const auto access = MetaNN::LowerAccess(row);
    std::array<float, feature_size> result{};
    std::memcpy(result.data(), access.RawMemory(), result.size() * sizeof(float));
    return result;
}

std::array<float, kModelWidth> BuildModelRow(const Tensor& tensor,
                                              std::size_t globalPosition)
{
    const auto base = BaseRowAt(tensor, globalPosition);
    std::array<float, kModelWidth> result{};
    const auto contract = EA::ResolveModelInputContract(kModelWidth, base.size());
    EA::CopyTensorFeaturesForModelInput(result.data(), base.data(), contract);
    assert(EA::AppendMultiHorizonReturnFeaturesAtGlobalPosition(
               globalPosition, result.data(), contract.tensorFeatureCount,
               kFeatureScale, [&tensor](std::size_t position)
               {
                   return tensor.RawCloseAtIterator(
                       tensor.begin() + static_cast<std::ptrdiff_t>(position));
               }) == EA::kModelReturnFeatureCount);
    return result;
}

void TestTensorPlacementAndParity()
{
    static_assert(relativeTickVolumeCol == 36);
    static_assert(causalReturnSurpriseCol == 37);
    static_assert(causalVolatilityRegimeCol == 38);
    static_assert(return_autocorrelation_feature_size == 49);
    static_assert(feature_size == 71);
    static_assert(EA::kPreEconomicEventModelInputWidth == 53);
    static_assert(EA::kCurrentModelInputWidth == 75);

    Tensor tensor{"causal-return-surprise"};
    float close = 100.0f;
    tensor.Add(BarAt(0, close));
    close = Advance(close, 0.01);
    tensor.Add(BarAt(1, close));
    close = Advance(close, 0.02);
    tensor.Add(BarAt(2, close));
    const auto first = BaseRowAt(tensor, 0);
    const auto third = BaseRowAt(tensor, 2);
    assert(first[causalReturnSurpriseCol] == 0.0f);
    AssertNear(third[causalReturnSurpriseCol], 2.0);

    const auto training = BuildModelRow(tensor, 2);
    const auto inference = BuildModelRow(tensor, 2);
    assert(std::memcmp(training.data(), inference.data(),
                       training.size() * sizeof(float)) == 0);
    assert(training[causalReturnSurpriseCol] == third[causalReturnSurpriseCol]);
}
} // namespace

int main()
{
    TestFeatureContract();
    TestValidityAndRollingState();
    TestCausalityAndFutureDependency();
    TestTensorPlacementAndParity();
    return 0;
}
