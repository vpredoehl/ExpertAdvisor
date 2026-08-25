#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <MetaNN/data/facilities/lower_access.h>

#include "ModelInputContract.hpp"
#include "ReturnFeatureHistory.hpp"
#include "SessionPhaseFeatures.hpp"
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

Feature BarAt(PriceTP timestamp, float close = 100.0f)
{
    return Feature{close - 0.1f, close, close + 0.2f, close - 0.2f, timestamp};
}

std::array<float, feature_size> BaseRowAt(const Tensor& tensor, std::size_t index)
{
    const auto row = *(tensor.begin() + static_cast<std::ptrdiff_t>(index));
    const auto access = MetaNN::LowerAccess(row);
    std::array<float, feature_size> result {};
    std::memcpy(result.data(), access.RawMemory(), result.size() * sizeof(float));
    return result;
}

std::array<float, kModelWidth> BuildTrainingRow(const Tensor& tensor,
                                                 std::size_t globalPosition)
{
    const auto base = BaseRowAt(tensor, globalPosition);
    std::array<float, kModelWidth> result {};
    const auto contract = EA::ResolveModelInputContract(
        kModelWidth, base.size());
    EA::CopyTensorFeaturesForModelInput(result.data(), base.data(), contract);
    const std::size_t appended = EA::AppendMultiHorizonReturnFeaturesAtGlobalPosition(
        globalPosition, result.data(), contract.tensorFeatureCount, kFeatureScale,
        [&tensor](std::size_t position)
        {
            return tensor.RawCloseAtIterator(
                tensor.begin() + static_cast<std::ptrdiff_t>(position));
        });
    assert(appended == EA::kMultiHorizonReturnLookbacks.size());
    return result;
}

std::array<float, kModelWidth> BuildInferenceRow(const Tensor& tensor,
                                                  std::size_t globalPosition)
{
    // This mirrors the production inference window coordinate path; both
    // paths delegate ordered base/return assembly to the shared production API.
    const auto source = *(tensor.begin() + static_cast<std::ptrdiff_t>(globalPosition));
    const auto access = MetaNN::LowerAccess(source);
    std::array<float, kModelWidth> result {};
    const auto contract = EA::ResolveModelInputContract(
        kModelWidth, feature_size);
    EA::CopyTensorFeaturesForModelInput(
        result.data(), access.RawMemory(), contract);
    const std::size_t appended = EA::AppendMultiHorizonReturnFeaturesAtGlobalPosition(
        globalPosition, result.data(), contract.tensorFeatureCount, kFeatureScale,
        [&tensor](std::size_t position)
        {
            return tensor.RawCloseAtIterator(
                tensor.begin() + static_cast<std::ptrdiff_t>(position));
        });
    assert(appended == EA::kMultiHorizonReturnLookbacks.size());
    return result;
}

void TestUtcPhaseFormulaAndPeriodicity()
{
    static_assert(return_autocorrelation_feature_size == 49);
    static_assert(feature_size == 59);
    static_assert(sessionPhaseSinCol == 34);
    static_assert(sessionPhaseCosCol == 35);
    static_assert(relativeTickVolumeCol == 36);
    static_assert(causalReturnSurpriseCol == 37);
    static_assert(EA::kPreEconomicEventModelInputWidth == 53);
    static_assert(EA::kCurrentModelInputWidth == 63);

    const PriceTP midnight{};
    const std::array<std::pair<long long, std::pair<float, float>>, 4> quarters{{
        {0, {0.0f, 1.0f}},
        {6 * 60 * 60, {1.0f, 0.0f}},
        {12 * 60 * 60, {0.0f, -1.0f}},
        {18 * 60 * 60, {-1.0f, 0.0f}},
    }};
    for (const auto& [seconds, expected] : quarters)
    {
        const auto actual = ComputeUtcSessionPhase(
            midnight + std::chrono::seconds(seconds));
        AssertNear(actual.first, expected.first);
        AssertNear(actual.second, expected.second);
    }

    const auto phaseAtQuarter = ComputeUtcSessionPhase(
        midnight + std::chrono::seconds(15 * 60));
    const auto phaseNextDay = ComputeUtcSessionPhase(
        midnight + std::chrono::seconds(24 * 60 * 60 + 15 * 60));
    assert(phaseAtQuarter == phaseNextDay);

    // A normal 15-minute grid supplies materially varying values, unlike the
    // historical 15-minute-cycle channels that remain intentionally unchanged.
    std::vector<std::pair<float, float>> phases;
    for (int quarter = 0; quarter != 96; ++quarter)
        phases.push_back(ComputeUtcSessionPhase(
            midnight + std::chrono::seconds(quarter * 15 * 60)));
    assert(phases.front() != phases.at(1));
    assert(phases.at(24) != phases.at(48));
}

void TestUtcDeterminismAndTensorPlacement()
{
    const PriceTP current = PriceTP{} + std::chrono::seconds(6 * 60 * 60);
    const auto beforeTimezoneChange = ComputeUtcSessionPhase(current);
    const char* priorTz = std::getenv("TZ");
    const std::string savedTz = priorTz ? priorTz : "";
    const bool hadPriorTz = priorTz != nullptr;
    assert(setenv("TZ", "Pacific/Auckland", 1) == 0);
    tzset();
    const auto afterTimezoneChange = ComputeUtcSessionPhase(current);
    if (hadPriorTz) assert(setenv("TZ", savedTz.c_str(), 1) == 0);
    else assert(unsetenv("TZ") == 0);
    tzset();
    assert(beforeTimezoneChange == afterTimezoneChange);

    Tensor tensor{"session-phase"};
    tensor.Add(BarAt(current - std::chrono::seconds(15 * 60), 99.9f));
    tensor.Add(BarAt(current));
    const auto firstRow = BaseRowAt(tensor, 0);
    const auto firstExpected = ComputeUtcSessionPhase(
        current - std::chrono::seconds(15 * 60));
    assert(firstRow[sessionPhaseSinCol] == firstExpected.first);
    assert(firstRow[sessionPhaseCosCol] == firstExpected.second);
    const auto row = BaseRowAt(tensor, 1);
    AssertNear(row[sessionPhaseSinCol], 1.0f);
    AssertNear(row[sessionPhaseCosCol], 0.0f);
    // Historical timestamp channel semantics remain their original 900-second
    // cycle: an aligned candle is still (approximately) sin=0, cos=1.
    AssertNear(row[8], 0.0f);
    AssertNear(row[9], 1.0f);
}

void TestCausalityAndProductionParity()
{
    const PriceTP first = PriceTP{} + std::chrono::seconds(3 * 60 * 60);
    std::vector<Feature> original;
    for (std::size_t i = 0; i < 64; ++i)
        original.push_back(BarAt(first + std::chrono::seconds(i * 15 * 60),
                                 100.0f + static_cast<float>(i) * 0.01f));
    constexpr std::size_t current = 32;
    Tensor baseline{"session-phase-causality"};
    for (const Feature& bar : original) baseline.Add(bar);

    std::vector<Feature> futureChanged = original;

    // Change only information strictly after the row under test while
    // preserving the Tensor's required chronological bar ordering.
    // Shifting the complete future suffix also proves that future session
    // timestamps cannot affect the already-completed current row.
    for (std::size_t i = current + 1; i < futureChanged.size(); ++i)
        futureChanged.at(i).time += std::chrono::hours(9);

    futureChanged.at(current + 1).close = 1000.0f;
    Tensor changed{"session-phase-causality"};
    for (const Feature& bar : futureChanged) changed.Add(bar);
    const auto baselineRow = BaseRowAt(baseline, current);
    const auto changedRow = BaseRowAt(changed, current);
    assert(std::memcmp(baselineRow.data(), changedRow.data(),
                       baselineRow.size() * sizeof(float)) == 0);

    const auto training = BuildTrainingRow(baseline, current);
    const auto inference = BuildInferenceRow(baseline, current);
    assert(std::memcmp(training.data(), inference.data(),
                       training.size() * sizeof(float)) == 0);
    assert(training[sessionPhaseSinCol] == baselineRow[sessionPhaseSinCol]);
    assert(training[sessionPhaseCosCol] == baselineRow[sessionPhaseCosCol]);
}
} // namespace

int main()
{
    TestUtcPhaseFormulaAndPeriodicity();
    TestUtcDeterminismAndTensorPlacement();
    TestCausalityAndProductionParity();
    return 0;
}
