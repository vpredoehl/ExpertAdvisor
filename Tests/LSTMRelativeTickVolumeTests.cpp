#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#include <MetaNN/data/facilities/lower_access.h>

#include "FeatureLayout.hpp"
#include "ModelInputContract.hpp"
#include "PricePoint.hpp"
#include "RelativeTickVolumeFeatures.hpp"
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
    assert(std::fabs(actual - expected) < 1.0e-5f);
}

Feature BarAt(std::size_t index, float volume, float close = 100.0f)
{
    Feature bar{close - 0.1f, close, close + 0.2f, close - 0.2f,
                PriceTP{std::chrono::seconds{
                    static_cast<long long>(index * 15 * 60)}}};
    bar.tickVolume = volume;
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

std::array<float, kModelWidth> BuildTrainingRow(const Tensor& tensor,
                                                 std::size_t globalPosition)
{
    const auto base = BaseRowAt(tensor, globalPosition);
    std::array<float, kModelWidth> result{};
    const auto contract = EA::ResolveModelInputContract(kModelWidth, base.size());
    EA::CopyTensorFeaturesForModelInput(result.data(), base.data(), contract);
    const std::size_t appended = EA::AppendMultiHorizonReturnFeaturesAtGlobalPosition(
        globalPosition, result.data(), contract.tensorFeatureCount, kFeatureScale,
        [&tensor](std::size_t position)
        {
            return tensor.RawCloseAtIterator(
                tensor.begin() + static_cast<std::ptrdiff_t>(position));
        });
    assert(appended == EA::kModelReturnFeatureCount);
    return result;
}

std::array<float, kModelWidth> BuildInferenceRow(const Tensor& tensor,
                                                  std::size_t globalPosition)
{
    const auto row = *(tensor.begin() + static_cast<std::ptrdiff_t>(globalPosition));
    const auto access = MetaNN::LowerAccess(row);
    std::array<float, kModelWidth> result{};
    const auto contract = EA::ResolveModelInputContract(kModelWidth, feature_size);
    EA::CopyTensorFeaturesForModelInput(result.data(), access.RawMemory(), contract);
    const std::size_t appended = EA::AppendMultiHorizonReturnFeaturesAtGlobalPosition(
        globalPosition, result.data(), contract.tensorFeatureCount, kFeatureScale,
        [&tensor](std::size_t position)
        {
            return tensor.RawCloseAtIterator(
                tensor.begin() + static_cast<std::ptrdiff_t>(position));
        });
    assert(appended == EA::kModelReturnFeatureCount);
    return result;
}

void TestFormulaWarmupAndDegeneratePolicy()
{
    static_assert(relativeTickVolumeLookback == 32);
    CausalRelativeTickVolume32 relative;
    assert(relative.AddCompletedBar(100.0f) == 0.0f); // no completed predecessor
    AssertNear(relative.AddCompletedBar(200.0f), std::log(201.0f / 101.0f));
    AssertNear(relative.AddCompletedBar(50.0f),
               std::log(51.0f / (((100.0f + 200.0f) / 2.0f) + 1.0f)));

    CausalRelativeTickVolume32 exactLookback;
    for (std::size_t i = 0; i < relativeTickVolumeLookback; ++i)
        assert(exactLookback.AddCompletedBar(100.0f) == 0.0f);
    AssertNear(exactLookback.AddCompletedBar(200.0f), std::log(201.0f / 101.0f));
    AssertNear(exactLookback.AddCompletedBar(100.0f),
               std::log(101.0f / (((31.0f * 100.0f + 200.0f) / 32.0f) + 1.0f)));

    CausalRelativeTickVolume32 eviction;
    eviction.AddCompletedBar(10000.0f);
    for (std::size_t i = 0; i < relativeTickVolumeLookback; ++i)
        eviction.AddCompletedBar(100.0f);
    assert(eviction.PriorBarCount() == relativeTickVolumeLookback);
    assert(eviction.AddCompletedBar(100.0f) == 0.0f); // 10000 was evicted.

    CausalRelativeTickVolume32 degenerate;
    assert(degenerate.AddCompletedBar(0.0f) == 0.0f);
    assert(degenerate.AddCompletedBar(0.0f) == 0.0f);
    AssertNear(degenerate.AddCompletedBar(9.0f), std::log(10.0f));
    const float nonfinite = degenerate.AddCompletedBar(
        std::numeric_limits<float>::quiet_NaN());
    assert(std::isfinite(nonfinite));
    assert(std::isfinite(CanonicalTickVolume(
        std::numeric_limits<float>::infinity())));
    assert(CanonicalTickVolume(-1.0f) == 0.0f);
}

void TestTensorPlacementCausalityAndParity()
{
    static_assert(legacy_feature_size == 32);
    static_assert(donchian_feature_size == 34);
    static_assert(session_phase_feature_size == 36);
    static_assert(relativeTickVolumeCol == 36);
    static_assert(relative_tick_volume_feature_size == 37);
    static_assert(causalReturnSurpriseCol == 37);
    static_assert(feature_size == 44);
    static_assert(EA::kRelativeTickVolumeModelInputWidth == 41);
    static_assert(EA::kCurrentModelInputWidth == 48);

    Tensor baseline{"relative-tick-volume"};
    Tensor changedCurrentVolume{"relative-tick-volume"};
    for (std::size_t i = 0; i < relativeTickVolumeLookback; ++i)
    {
        baseline.Add(BarAt(i, 100.0f, 100.0f + static_cast<float>(i) * 0.01f));
        changedCurrentVolume.Add(
            BarAt(i, 100.0f, 100.0f + static_cast<float>(i) * 0.01f));
    }
    baseline.Add(BarAt(32, 200.0f, 100.32f));
    changedCurrentVolume.Add(BarAt(32, 50.0f, 100.32f));
    const auto baselineRow = BaseRowAt(baseline, 32);
    const auto changedCurrentRow = BaseRowAt(changedCurrentVolume, 32);
    AssertNear(baselineRow[relativeTickVolumeCol], std::log(201.0f / 101.0f));
    AssertNear(changedCurrentRow[relativeTickVolumeCol], std::log(51.0f / 101.0f));
    assert(std::memcmp(baselineRow.data(), changedCurrentRow.data(),
                       relativeTickVolumeCol * sizeof(float)) == 0);

    std::vector<Feature> bars;
    for (std::size_t i = 0; i < 64; ++i)
        bars.push_back(BarAt(i, 100.0f + static_cast<float>(i),
                             100.0f + static_cast<float>(i) * 0.01f));
    Tensor original{"relative-tick-volume-causality"};
    for (const Feature& bar : bars) original.Add(bar);
    std::vector<Feature> futureChanged = bars;
    futureChanged.at(33).tickVolume = 1000000.0f;
    Tensor changedFuture{"relative-tick-volume-causality"};
    for (const Feature& bar : futureChanged) changedFuture.Add(bar);
    const auto originalRow = BaseRowAt(original, 32);
    const auto futureChangedRow = BaseRowAt(changedFuture, 32);
    assert(std::memcmp(originalRow.data(), futureChangedRow.data(),
                       originalRow.size() * sizeof(float)) == 0);

    const auto training = BuildTrainingRow(original, 32);
    const auto inference = BuildInferenceRow(original, 32);
    assert(std::memcmp(training.data(), inference.data(),
                       training.size() * sizeof(float)) == 0);
    assert(training[relativeTickVolumeCol] == originalRow[relativeTickVolumeCol]);
}
} // namespace

int main()
{
    TestFormulaWarmupAndDegeneratePolicy();
    TestTensorPlacementCausalityAndParity();
    return 0;
}
