#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include <MetaNN/data/facilities/lower_access.h>

#include "DonchianFeatures.hpp"
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

void AssertNear(float actual, float expected)
{
    assert(std::fabs(actual - expected) < 1.0e-4f);
}

void AssertBytesEqual(const void* lhs, const void* rhs, std::size_t bytes)
{
    assert(std::memcmp(lhs, rhs, bytes) == 0);
}

Feature MakeBar(std::size_t index, float close, float high, float low)
{
    return Feature{close - 0.05f, close, high, low,
                   PriceTP{} + std::chrono::seconds(static_cast<long long>(index * 900))};
}

std::vector<Feature> MakeBars(std::size_t count)
{
    std::vector<Feature> bars;
    bars.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        const float close = 100.0f + static_cast<float>(i) * 0.05f;
        bars.push_back(MakeBar(i, close,
                               close + 0.3f + static_cast<float>(i % 7) * 0.1f,
                               close - 0.25f - static_cast<float>(i % 5) * 0.1f));
    }
    return bars;
}

Tensor Build(const std::vector<Feature>& bars, Donchian20Mode mode)
{
    Tensor tensor{"donchian20-test", mode};
    for (const Feature& bar : bars)
        tensor.Add(bar);
    return tensor;
}

std::array<float, kModelWidth> AssembleProductionRow(const Tensor& tensor,
                                                      std::size_t globalPosition)
{
    std::array<float, kModelWidth> row {};
    const auto source = *(tensor.begin() + static_cast<std::ptrdiff_t>(globalPosition));
    auto low = MetaNN::LowerAccess(source);
    const std::size_t appended = EA::AssembleLstmModelInputRowAtGlobalPosition(
        row.data(), low.RawMemory(), feature_size, globalPosition, kFeatureScale,
        [&tensor](std::size_t position)
        {
            return tensor.RawCloseAtIterator(
                tensor.begin() + static_cast<std::ptrdiff_t>(position));
        });
    assert(appended == EA::kMultiHorizonReturnLookbacks.size());
    for (float& value : row)
    {
        assert(std::isfinite(value));
        value = std::clamp(value, -10.0f, 10.0f);
    }
    return row;
}

void TestFormulaCausalityAndRollingWindow()
{
    std::vector<float> highs(20, 101.0f);
    std::vector<float> lows(20, 99.0f);
    highs[3] = 105.0f;
    lows[14] = 95.0f;
    const auto value = ComputeCausalDonchian20(highs, lows, 100.0f, kFeatureScale);
    AssertNear(value.first, std::log(100.0f / 105.0f) * kFeatureScale);
    AssertNear(value.second, std::log(100.0f / 95.0f) * kFeatureScale);

    // Current-bar high/low are intentionally absent from the calculation.
    const auto unchanged = ComputeCausalDonchian20(highs, lows, 100.0f, kFeatureScale);
    AssertBytesEqual(&value, &unchanged, sizeof(value));

    // The 21st prior bar leaves the window exactly when the newest 20 remain.
    highs.insert(highs.begin(), 1000.0f);
    lows.insert(lows.begin(), 1.0f);
    const auto expired = ComputeCausalDonchian20(highs, lows, 100.0f, kFeatureScale);
    AssertNear(expired.first, value.first);
    AssertNear(expired.second, value.second);

    const auto startup = ComputeCausalDonchian20({101.0f}, {99.0f}, 100.0f, kFeatureScale);
    AssertNear(startup.first, std::log(100.0f / 101.0f) * kFeatureScale);
    AssertNear(startup.second, std::log(100.0f / 99.0f) * kFeatureScale);
}

void TestTensorModesParityAndNoFutureLeakage()
{
    const auto bars = MakeBars(96);
    const Tensor enabled = Build(bars, Donchian20Mode::Enabled);
    const Tensor ablated = Build(bars, Donchian20Mode::ZeroAblation);
    constexpr std::size_t t = 48;
    const auto enabledRow = AssembleProductionRow(enabled, t);
    const auto ablatedRow = AssembleProductionRow(ablated, t);
    assert(enabledRow[donchianUpCol] != 0.0f);
    assert(enabledRow[donchianDownCol] != 0.0f);
    for (std::size_t col = 0; col < kModelWidth; ++col)
    {
        if (col == donchianUpCol || col == donchianDownCol)
            assert(ablatedRow[col] == 0.0f);
        else
            assert(enabledRow[col] == ablatedRow[col]);
    }

    // The distinct current high/low must not influence t's two channels.
    auto currentMutated = bars;
    currentMutated[t].high = 10000.0f;
    currentMutated[t].low = 0.001f;
    const Tensor changedCurrent = Build(currentMutated, Donchian20Mode::Enabled);
    const auto changedCurrentRow = AssembleProductionRow(changedCurrent, t);
    assert(enabledRow[donchianUpCol] == changedCurrentRow[donchianUpCol]);
    assert(enabledRow[donchianDownCol] == changedCurrentRow[donchianDownCol]);

    // Future observations cannot alter the feature vector at t.
    auto futureMutated = bars;
    futureMutated[t + 1].high = 10000.0f;
    futureMutated[t + 1].low = 0.001f;
    const Tensor changedFuture = Build(futureMutated, Donchian20Mode::Enabled);
    const auto changedFutureRow = AssembleProductionRow(changedFuture, t);
    AssertBytesEqual(enabledRow.data(), changedFutureRow.data(),
                     enabledRow.size() * sizeof(float));

}

void TestCompatibilityAndClamp()
{
    static_assert(feature_size == 34);
    static_assert(EA::kLegacyModelInputWidth == 36);
    static_assert(EA::kCurrentModelInputWidth == 38);
    std::array<float, feature_size> tensorRow {};
    for (std::size_t i = 0; i < tensorRow.size(); ++i)
        tensorRow[i] = static_cast<float>(i);
    const auto legacy = EA::ResolveModelInputContract(36, tensorRow.size());
    const auto current = EA::ResolveModelInputContract(38, tensorRow.size());
    std::array<float, 36> legacyInput {};
    std::array<float, 38> currentInput {};
    EA::CopyTensorFeaturesForModelInput(legacyInput.data(), tensorRow.data(), legacy);
    EA::CopyTensorFeaturesForModelInput(currentInput.data(), tensorRow.data(), current);
    for (std::size_t i = 0; i < legacy_feature_size; ++i)
        assert(legacyInput[i] == tensorRow[i]);
    assert(currentInput[donchianUpCol] == tensorRow[donchianUpCol]);
    assert(currentInput[donchianDownCol] == tensorRow[donchianDownCol]);

    const auto extreme = ComputeCausalDonchian20({10000.0f}, {0.01f}, 100.0f, kFeatureScale);
    assert(std::isfinite(extreme.first) && std::isfinite(extreme.second));
    assert(std::clamp(extreme.first, -10.0f, 10.0f) == -10.0f);
    assert(std::clamp(extreme.second, -10.0f, 10.0f) == 10.0f);
}
} // namespace

int main()
{
    TestFormulaCausalityAndRollingWindow();
    TestTensorModesParityAndNoFutureLeakage();
    TestCompatibilityAndClamp();
    return 0;
}
