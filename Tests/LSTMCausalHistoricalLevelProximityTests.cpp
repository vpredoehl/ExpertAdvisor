#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#include <MetaNN/data/facilities/lower_access.h>

#include "CausalHistoricalLevelProximityFeatures.hpp"
#include "FeatureLayout.hpp"
#include "ModelInputContract.hpp"
#include "PricePoint.hpp"
#include "Tensor.hpp"

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled() { return false; }

namespace
{
constexpr std::int64_t kSecondsPerWeek = 7 * 24 * 60 * 60;

bool IsPivotWeek(std::size_t week)
{
    return week == 30 || week == 50 || week == 70 || week == 90;
}

float AddSyntheticWeek(CausalHistoricalLevelProximity& feature,
                       std::size_t week,
                       float scale,
                       bool repeatedLevels,
                       float observationClose = 0.0f)
{
    const float low = observationClose > 0.0f
        ? std::min(0.99f * scale, observationClose)
        : 1.00f * scale;
    const float high = observationClose > 0.0f
        ? std::max(1.21f * scale, observationClose)
        : ((repeatedLevels && IsPivotWeek(week)) ? 1.20f : 1.10f) * scale;
    const float close = observationClose > 0.0f
        ? observationClose
        : 1.05f * scale;
    return feature.AddCompletedBar(
        high, low, close, static_cast<std::int64_t>(week) * kSecondsPerWeek);
}

void AddDiscoveryHistory(CausalHistoricalLevelProximity& feature,
                         float scale,
                         bool repeatedLevels,
                         std::size_t lastWeek)
{
    for (std::size_t week = 0; week <= lastWeek; ++week)
        (void)AddSyntheticWeek(feature, week, scale, repeatedLevels);
}

Feature TensorBar(std::size_t week,
                  bool repeatedLevels,
                  float observationClose = 0.0f)
{
    const float low = observationClose > 0.0f
        ? std::min(0.99f, observationClose)
        : 1.00f;
    const float high = observationClose > 0.0f
        ? std::max(1.21f, observationClose)
        : ((repeatedLevels && IsPivotWeek(week)) ? 1.20f : 1.10f);
    const float close = observationClose > 0.0f ? observationClose : 1.05f;
    Feature bar{close, close, high, low,
                PriceTP{std::chrono::seconds{
                    static_cast<long long>(week) * kSecondsPerWeek}}};
    bar.tickVolume = 100.0f;
    return bar;
}

float RowValue(const Tensor& tensor, std::size_t row, std::size_t column)
{
    return MetaNN::LowerAccess(
        *(tensor.begin() + static_cast<std::ptrdiff_t>(row)))
        .RawMemory()[column];
}

void TestDefinitionInsufficientHistoryNearFarAndScaling()
{
    static_assert(CausalHistoricalLevelProximity::kPivotRadiusWeeks == 2);
    static_assert(CausalHistoricalLevelProximity::kBandwidthWeeks == 26);
    static_assert(CausalHistoricalLevelProximity::kMinimumHistoryWeeks == 104);
    static_assert(CausalHistoricalLevelProximity::kRetentionWeeks == 260);

    CausalHistoricalLevelProximity insufficient;
    AddDiscoveryHistory(insufficient, 1.0f, true, 103);
    assert(insufficient.CompletedWeekCount() == 103);
    assert(AddSyntheticWeek(insufficient, 103, 1.0f, true) == 0.0f);

    CausalHistoricalLevelProximity nearFeature;
    CausalHistoricalLevelProximity farFeature;
    AddDiscoveryHistory(nearFeature, 1.0f, true, 103);
    AddDiscoveryHistory(farFeature, 1.0f, true, 103);
    const float near = AddSyntheticWeek(nearFeature, 104, 1.0f, true, 1.20f);
    const float far = AddSyntheticWeek(farFeature, 104, 1.0f, true, 1.60f);
    assert(nearFeature.CandidateCount() >= 4);
    assert(std::isfinite(near) && std::isfinite(far));
    assert(near > 0.0f && near < 1.0f);
    assert(far >= 0.0f && far < near * 0.01f);

    CausalHistoricalLevelProximity scaled;
    AddDiscoveryHistory(scaled, 100.0f, true, 103);
    const float scaledNear =
        AddSyntheticWeek(scaled, 104, 100.0f, true, 120.0f);
    assert(std::fabs(near - scaledNear) < 2.0e-5f);

    CausalHistoricalLevelProximity nearbyPrice;
    AddDiscoveryHistory(nearbyPrice, 1.0f, true, 103);
    const float tinyMove = AddSyntheticWeek(
        nearbyPrice, 104, 1.0f, true,
        std::nextafter(1.20f, std::numeric_limits<float>::infinity()));
    assert(std::fabs(near - tinyMove) < 1.0e-5f);

    CausalHistoricalLevelProximity extreme;
    AddDiscoveryHistory(extreme, 1.0f, true, 103);
    const float extremeValue = AddSyntheticWeek(
        extreme, 104, 1.0f, true, std::numeric_limits<float>::max());
    assert(std::isfinite(extremeValue));
    assert(extremeValue >= 0.0f && extremeValue < 1.0f);
    assert(extreme.AddCompletedBar(
               std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f,
               105 * kSecondsPerWeek) == 0.0f);
}

void TestPerSymbolDataDerivationAndCausalHistorySensitivity()
{
    CausalHistoricalLevelProximity symbolA;
    CausalHistoricalLevelProximity symbolB;
    AddDiscoveryHistory(symbolA, 1.0f, true, 103);
    AddDiscoveryHistory(symbolB, 1.25f, true, 103);
    const float atAOwnLevel =
        AddSyntheticWeek(symbolA, 104, 1.0f, true, 1.20f);
    const float sameAbsolutePriceForB =
        AddSyntheticWeek(symbolB, 104, 1.25f, true, 1.20f);
    assert(atAOwnLevel > sameAbsolutePriceForB);

    CausalHistoricalLevelProximity oldHistoryChanged;
    AddDiscoveryHistory(oldHistoryChanged, 1.0f, false, 103);
    const float withoutOldRepeatedPivots =
        AddSyntheticWeek(oldHistoryChanged, 104, 1.0f, false, 1.20f);
    assert(atAOwnLevel > withoutOldRepeatedPivots);
}

void TestTensorCausalityParityAblationAndCompatibility()
{
    static_assert(causalRollingRangeExpansionCol == 46);
    static_assert(historicalLevelProximityCol == 47);
    static_assert(feature_size == 48);
    static_assert(EA::kCausalRollingRangeExpansionModelInputWidth == 51);
    static_assert(EA::kHistoricalLevelProximityModelInputWidth == 52);
    static_assert(EA::kCurrentModelInputWidth == 52);

    Tensor prefix{"historical-level-prefix"};
    Tensor futureA{"historical-level-future-a"};
    Tensor futureB{"historical-level-future-b"};
    for (std::size_t week = 0; week <= 104; ++week)
    {
        const float current = week == 104 ? 1.20f : 0.0f;
        prefix.Add(TensorBar(week, true, current));
        futureA.Add(TensorBar(week, true, current));
        futureB.Add(TensorBar(week, true, current));
    }
    const float atObservation =
        RowValue(prefix, 104, historicalLevelProximityCol);
    futureA.Add(TensorBar(105, false, 1.80f));
    futureB.Add(TensorBar(105, false, 1.01f));
    assert(atObservation ==
           RowValue(futureA, 104, historicalLevelProximityCol));
    assert(atObservation ==
           RowValue(futureB, 104, historicalLevelProximityCol));

    const float* tensorRow = MetaNN::LowerAccess(*(prefix.begin() + 104))
        .RawMemory();
    const auto current = EA::ResolveModelInputContract(
        EA::kCurrentModelInputWidth, feature_size);
    std::array<float, EA::kCurrentModelInputWidth> training{};
    std::array<float, EA::kCurrentModelInputWidth> inference{};
    EA::CopyTensorFeaturesForModelInput(training.data(), tensorRow, current);
    EA::CopyTensorFeaturesForModelInput(inference.data(), tensorRow, current);
    assert(std::memcmp(training.data(), inference.data(),
                       feature_size * sizeof(float)) == 0);
    assert(training[historicalLevelProximityCol] == atObservation);

    const auto mask = EA::FeatureAblationMask::Parse(
        "historical_level_proximity");
    std::array<float, EA::kCurrentModelInputWidth> ablated{};
    EA::CopyTensorFeaturesForModelInput(
        ablated.data(), tensorRow, current, mask);
    assert(ablated[historicalLevelProximityCol] == 0.0f);
    for (std::size_t column = 0; column < feature_size; ++column)
        if (column != historicalLevelProximityCol)
            assert(ablated[column] == training[column]);

    const auto historical = EA::ResolveModelInputContract(
        EA::kCausalRollingRangeExpansionModelInputWidth, feature_size);
    assert(historical.tensorFeatureCount == historicalLevelProximityCol);
    std::array<float, EA::kCausalRollingRangeExpansionModelInputWidth>
        historicalInput{};
    historicalInput.fill(-1.0f);
    EA::CopyTensorFeaturesForModelInput(
        historicalInput.data(), tensorRow, historical);
    assert(historicalInput[causalRollingRangeExpansionCol] ==
           tensorRow[causalRollingRangeExpansionCol]);
    assert(historicalInput[historicalLevelProximityCol] == -1.0f);

    bool absentAblationRejected = false;
    try
    {
        EA::CopyTensorFeaturesForModelInput(
            historicalInput.data(), tensorRow, historical, mask);
    }
    catch (const std::runtime_error&)
    {
        absentAblationRejected = true;
    }
    assert(absentAblationRejected);
}
} // namespace

int main()
{
    TestDefinitionInsufficientHistoryNearFarAndScaling();
    TestPerSymbolDataDerivationAndCausalHistorySensitivity();
    TestTensorCausalityParityAblationAndCompatibility();
    return 0;
}
