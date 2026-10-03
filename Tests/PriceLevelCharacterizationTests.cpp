#include "CausalPriceLevelEngine.hpp"
#include "PriceLevelCharacterization.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace
{
namespace PL = EA::PriceLevel;
namespace PLC = EA::PriceLevel::Characterization;
using namespace std::chrono_literals;

PL::CompletedBar Bar(std::size_t index, double high, double low, double close)
{
    return {std::chrono::sys_seconds{1'700'000'000s +
            std::chrono::seconds{static_cast<std::int64_t>(index) * 900}},
            close, high, low, close};
}

std::vector<PLC::PrecedingRangeScale::Sample> Samples(
    const std::vector<PLC::PrecedingRangeScale>& scales)
{
    std::vector<PLC::PrecedingRangeScale::Sample> result;
    for (const auto& scale : scales) result.push_back(scale.BeforeCurrentBar());
    return result;
}

void TestStrictPivotsMatchV1()
{
    const std::vector<PL::CompletedBar> bars{
        Bar(0, 4, 1, 2), Bar(1, 7, 2, 5), Bar(2, 3, 0, 1),
        Bar(3, 8, 3, 6), Bar(4, 2, -1, 0), Bar(5, 9, 4, 7),
        Bar(6, 3, 0, 1), Bar(7, 6, 2, 4), Bar(8, 1, -2, -1),
        Bar(9, 5, 1, 4), Bar(10, 2, -1, 0)};
    for (const std::size_t radius : {1U, 2U, 3U})
    {
        PLC::StrictPivotCharacterizer characterization{{radius}};
        std::vector<PLC::PrecedingRangeScale> scales;
        scales.emplace_back(3);
        PL::CausalPriceLevelEngine engine{"eurusdrmp", {radius, 0.0, 100,
            100, 100, 900s}};
        PLC::PivotCounts engineCounts;
        for (const PL::CompletedBar& bar : bars)
        {
            characterization.AddCompletedBar(bar, Samples(scales), {3},
                [](const PLC::PivotScaleSample&) {});
            for (PLC::PrecedingRangeScale& scale : scales)
                scale.AddCompletedRange(bar.high - bar.low);
            const PL::Update update = engine.AddCompletedBar(bar);
            for (const PL::Observation& observation : update.observations)
            {
                if (observation.kind == PL::InteractionKind::level_established ||
                    observation.kind == PL::InteractionKind::level_reinforced)
                {
                    const auto found = std::find_if(update.activeLevels.begin(),
                        update.activeLevels.end(), [&](const PL::Level& level) {
                            return level.identity == observation.levelIdentity;
                        });
                    // A reinforcing pivot has the original level's kind, which
                    // is sufficient here because each synthetic pivot price is
                    // unique and all observations establish a distinct level.
                    assert(found != update.activeLevels.end());
                    if (found->originatingPivot == PL::PivotKind::high)
                        ++engineCounts.highs;
                    else ++engineCounts.lows;
                }
            }
        }
        assert(characterization.counts()[0] == engineCounts);
    }
}

void TestScaleUsesPrecedingRangesOnly()
{
    PLC::PrecedingRangeScale scale{3};
    assert(scale.BeforeCurrentBar().count == 0);
    scale.AddCompletedRange(2.0);
    const auto beforeSecond = scale.BeforeCurrentBar();
    assert(beforeSecond.count == 1);
    assert(beforeSecond.mean == 2.0 && beforeSecond.median == 2.0);
    scale.AddCompletedRange(10.0);
    scale.AddCompletedRange(4.0);
    const auto beforeFourth = scale.BeforeCurrentBar();
    assert(beforeFourth.count == 3);
    assert(beforeFourth.mean == (16.0 / 3.0));
    assert(beforeFourth.median == 4.0);
    // The earlier snapshot is immutable evidence that later completed bars do
    // not revise the scale available before the second bar.
    assert(beforeSecond.count == 1 && beforeSecond.mean == 2.0);
}

void TestPivotScaleTimeAndDeterminism()
{
    const std::vector<PL::CompletedBar> bars{
        Bar(0, 3, 1, 2), Bar(1, 9, 3, 5), Bar(2, 4, 2, 3),
        Bar(3, 7, 1, 4), Bar(4, 2, -1, 0)};
    const auto replay = [&bars] {
        PLC::StrictPivotCharacterizer pivots{{1, 2}};
        std::vector<PLC::PrecedingRangeScale> scales;
        scales.emplace_back(3);
        std::vector<std::tuple<std::size_t, std::size_t, std::size_t, double, double>> result;
        for (const PL::CompletedBar& bar : bars)
        {
            pivots.AddCompletedBar(bar, Samples(scales), {3},
                [&](const PLC::PivotScaleSample& sample) {
                    result.emplace_back(sample.pivotBar, sample.pivotRadius,
                        sample.pivotTimeScale.count, sample.pivotTimeScale.mean,
                        sample.confirmationTimeScale.mean);
                });
            scales[0].AddCompletedRange(bar.high - bar.low);
        }
        return result;
    };
    const auto first = replay();
    const auto second = replay();
    assert(first == second);
    assert(!first.empty());
    // Pivot at bar one sees only bar zero's range (2), even though its
    // confirmation at bar two can see both predecessor ranges.
    const auto found = std::find_if(first.begin(), first.end(), [](const auto& value) {
        return std::get<0>(value) == 1 && std::get<1>(value) == 1;
    });
    assert(found != first.end());
    assert(std::get<2>(*found) == 1);
    assert(std::get<3>(*found) == 2.0);
    assert(std::get<4>(*found) == 4.0);
}

void TestChronologicalValidation()
{
    PLC::StrictPivotCharacterizer pivots{{1}};
    std::vector<PLC::PrecedingRangeScale::Sample> empty;
    pivots.AddCompletedBar(Bar(1, 2, 1, 1.5), empty, {},
        [](const PLC::PivotScaleSample&) {});
    bool rejected = false;
    try
    {
        pivots.AddCompletedBar(Bar(1, 2, 1, 1.5), empty, {},
            [](const PLC::PivotScaleSample&) {});
    }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

std::vector<PLC::AdaptiveObservation> AddAdaptive(
    PLC::AdaptiveWidthResearchDetector& detector, PLC::PrecedingRangeScale& scale,
    const PL::CompletedBar& bar)
{
    const PLC::AdaptiveUpdate update = detector.AddCompletedBar(bar, scale.BeforeCurrentBar());
    scale.AddCompletedRange(bar.high - bar.low);
    return update.observations;
}

const PLC::AdaptiveObservation& Observation(
    const std::vector<PLC::AdaptiveObservation>& observations, PL::InteractionKind kind)
{
    const auto found = std::find_if(observations.begin(), observations.end(),
        [kind](const PLC::AdaptiveObservation& value) { return value.kind == kind; });
    assert(found != observations.end());
    return *found;
}

void TestAdaptiveScaleBoundariesAndFrozenWidth()
{
    const PLC::AdaptiveConfiguration pivotTime{1, 3, 1.0, PLC::ScaleTiming::pivot_time,
        8, 20, 3, 900s};
    PLC::AdaptiveConfiguration confirmationTime = pivotTime;
    confirmationTime.scaleTiming = PLC::ScaleTiming::confirmation_time;
    PLC::AdaptiveWidthResearchDetector pivotDetector{"eurusdrmp", pivotTime};
    PLC::AdaptiveWidthResearchDetector confirmationDetector{"eurusdrmp", confirmationTime};
    PLC::PrecedingRangeScale pivotScale{3};
    PLC::PrecedingRangeScale confirmationScale{3};
    const std::vector<PL::CompletedBar> initial{
        Bar(0, 3, 1, 2), Bar(1, 9, 3, 5), Bar(2, 4, 2, 3)};
    std::vector<PLC::AdaptiveObservation> pivotObservations;
    std::vector<PLC::AdaptiveObservation> confirmationObservations;
    for (const PL::CompletedBar& bar : initial)
    {
        pivotObservations = AddAdaptive(pivotDetector, pivotScale, bar);
        confirmationObservations = AddAdaptive(confirmationDetector, confirmationScale, bar);
    }
    // At the pivot boundary only bar zero's range (2) exists.  At the later
    // confirmation boundary, ranges 2 and 6 exist, with median 4.
    assert(Observation(pivotObservations, PL::InteractionKind::level_established)
               .level.zoneHalfWidth == 2.0);
    assert(Observation(confirmationObservations, PL::InteractionKind::level_established)
               .level.zoneHalfWidth == 4.0);

    AddAdaptive(pivotDetector, pivotScale, Bar(3, 5, 1, 3));
    AddAdaptive(pivotDetector, pivotScale, Bar(4, 9.5, 2, 5));
    const auto reinforced = AddAdaptive(pivotDetector, pivotScale, Bar(5, 4, 1, 2));
    // Later high-low ranges differ, but reinforcement reports the original
    // frozen width rather than any current scale.
    assert(Observation(reinforced, PL::InteractionKind::level_reinforced)
               .level.zoneHalfWidth == 2.0);
}

void TestAdaptiveDeterminismChronologyAndIdentityIsolation()
{
    const PLC::AdaptiveConfiguration configuration{1, 3, 1.0,
        PLC::ScaleTiming::pivot_time, 8, 20, 3, 900s};
    const std::vector<PL::CompletedBar> bars{
        Bar(0, 3, 1, 2), Bar(1, 9, 3, 5), Bar(2, 4, 2, 3),
        Bar(3, 5, 1, 3), Bar(4, 9.5, 2, 5), Bar(5, 4, 1, 2)};
    const auto replay = [&] {
        PLC::AdaptiveWidthResearchDetector detector{"eurusdrmp", configuration};
        PLC::PrecedingRangeScale scale{3};
        std::vector<std::string> result;
        for (const PL::CompletedBar& bar : bars)
        {
            const auto observations = AddAdaptive(detector, scale, bar);
            assert(std::is_sorted(observations.begin(), observations.end(),
                [](const PLC::AdaptiveObservation& left,
                   const PLC::AdaptiveObservation& right) {
                    return left.level.identity != right.level.identity
                        ? left.level.identity < right.level.identity
                        : static_cast<int>(left.kind) < static_cast<int>(right.kind);
                }));
            for (const auto& observation : observations)
                result.push_back(std::to_string(static_cast<int>(observation.kind)) + ":" +
                                 observation.level.identity + ":" +
                                 PL::CanonicalDouble(observation.level.zoneHalfWidth));
        }
        return result;
    };
    assert(replay() == replay());
    const std::string adaptiveIdentity =
        PLC::CanonicalAdaptiveConfigurationIdentity(configuration);
    assert(adaptiveIdentity.find("causal-price-level/v1") == std::string::npos);
    assert(adaptiveIdentity.find("id=causal-price-level") == std::string::npos);
    assert(adaptiveIdentity.find(PLC::kAdaptiveStudyContract) != std::string::npos);

    PLC::AdaptiveWidthResearchDetector detector{"eurusdrmp", configuration};
    PLC::PrecedingRangeScale scale{3};
    (void)AddAdaptive(detector, scale, bars[0]);
    bool rejected = false;
    try { (void)AddAdaptive(detector, scale, bars[0]); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

void TestAdaptiveAgeBoundaryCapacityAndCensoring()
{
    // The phase-5B contract measures lifetime from availability (confirmation)
    // and preserves the v1 boundary: A + maxAgeBars remains active; expiration
    // is emitted at A + maxAgeBars + 1 before that bar's interactions.
    const PLC::AdaptiveConfiguration ageOne{1, 3, 0.0, PLC::ScaleTiming::pivot_time,
        8, 1, 3, 900s};
    PLC::AdaptiveWidthResearchDetector expiry{"eurusdrmp", ageOne};
    PLC::PrecedingRangeScale expiryScale{3};
    (void)AddAdaptive(expiry, expiryScale, Bar(0, 3, 1, 2));
    (void)AddAdaptive(expiry, expiryScale, Bar(1, 9, 3, 5));
    const auto established = AddAdaptive(expiry, expiryScale, Bar(2, 4, 2, 3));
    assert(Observation(established, PL::InteractionKind::level_established).level.availableBar == 2);
    const auto stillActive = AddAdaptive(expiry, expiryScale, Bar(3, 5, 1, 3));
    assert(std::none_of(stillActive.begin(), stillActive.end(), [](const auto& observation) {
        return observation.kind == PL::InteractionKind::level_expired;
    }));
    const auto expired = AddAdaptive(expiry, expiryScale, Bar(4, 2, 0, 1));
    assert(Observation(expired, PL::InteractionKind::level_expired).level.availableBar == 2);
    // A level present in the final active census is right-censored by the
    // harness; it is not represented by an ended-level observation.
    PLC::AdaptiveWidthResearchDetector censored{"eurusdrmp", ageOne};
    PLC::PrecedingRangeScale censoredScale{3};
    (void)AddAdaptive(censored, censoredScale, Bar(0, 3, 1, 2));
    (void)AddAdaptive(censored, censoredScale, Bar(1, 9, 3, 5));
    const PLC::AdaptiveUpdate finalUpdate = censored.AddCompletedBar(
        Bar(2, 4, 2, 3), censoredScale.BeforeCurrentBar());
    assert(finalUpdate.activeLevels.size() == 1);
    assert(std::none_of(finalUpdate.observations.begin(), finalUpdate.observations.end(),
        [](const auto& observation) { return observation.kind == PL::InteractionKind::level_expired; }));

    const PLC::AdaptiveConfiguration capacityOne{1, 3, 0.0, PLC::ScaleTiming::pivot_time,
        1, 20, 3, 900s};
    PLC::AdaptiveWidthResearchDetector capacity{"eurusdrmp", capacityOne};
    PLC::PrecedingRangeScale capacityScale{3};
    const std::vector<PL::CompletedBar> bars{
        Bar(0, 3, 1, 2), Bar(1, 9, 3, 5), Bar(2, 4, 2, 3),
        Bar(3, 5, 1, 3), Bar(4, 9.5, 2, 5), Bar(5, 4, 1, 2)};
    std::vector<PLC::AdaptiveObservation> capacityObservations;
    for (const PL::CompletedBar& bar : bars)
        capacityObservations = AddAdaptive(capacity, capacityScale, bar);
    assert(Observation(capacityObservations, PL::InteractionKind::level_evicted).kind ==
           PL::InteractionKind::level_evicted);
    assert(std::none_of(capacityObservations.begin(), capacityObservations.end(),
        [](const auto& observation) { return observation.kind == PL::InteractionKind::level_expired; }));
}

void TestAdaptiveAgeDoesNotChangePivotDetection()
{
    const std::vector<PL::CompletedBar> bars{
        Bar(0, 3, 1, 2), Bar(1, 9, 3, 5), Bar(2, 4, 2, 3),
        Bar(3, 5, 1, 3), Bar(4, 9.5, 2, 5), Bar(5, 4, 1, 2),
        Bar(6, 8, 2, 5), Bar(7, 3, 0, 1)};
    const auto pivotOutcomes = [&](std::size_t age) {
        PLC::AdaptiveWidthResearchDetector detector{"eurusdrmp", {1, 3, 0.0,
            PLC::ScaleTiming::pivot_time, 16, age, 3, 900s}};
        PLC::PrecedingRangeScale scale{3};
        std::size_t result = 0;
        for (const PL::CompletedBar& bar : bars)
        {
            for (const PLC::AdaptiveObservation& observation : AddAdaptive(detector, scale, bar))
                if (observation.kind == PL::InteractionKind::level_established ||
                    observation.kind == PL::InteractionKind::level_reinforced)
                    ++result;
        }
        return result;
    };
    assert(pivotOutcomes(1) == pivotOutcomes(20));
}

void TestPhase5CBoundsGridAndIdentity()
{
    const auto configurations = PLC::Phase5CBoundsConfirmationConfigurations();
    assert(configurations.size() == 9);
    const std::array<std::size_t, 3> activeCaps{32, 48, 64};
    const std::array<std::size_t, 3> evidenceCaps{8, 16, 32};
    std::size_t index = 0;
    for (const std::size_t active : activeCaps)
    {
        for (const std::size_t evidence : evidenceCaps)
        {
            const auto& configuration = configurations[index++];
            assert(configuration.pivotRadiusBars == 3);
            assert(configuration.scaleLookbackBars == 64);
            assert(configuration.scaleMultiplier == 1.0);
            assert(configuration.scaleTiming == PLC::ScaleTiming::pivot_time);
            assert(configuration.maxActiveLevels == active);
            assert(configuration.maxAgeBars == 512);
            assert(configuration.maxRetainedPivotEvidence == evidence);
            assert(configuration.completedBarDuration == 900s);
        }
    }
    assert(PLC::CanonicalAdaptiveConfigurationIdentity(configurations[0]) !=
           PLC::CanonicalAdaptiveConfigurationIdentity(configurations[1]));
    assert(PLC::CanonicalAdaptiveConfigurationIdentity(configurations[0]) !=
           PLC::CanonicalAdaptiveConfigurationIdentity(configurations[3]));
    assert(std::string_view{PLC::kBoundsConfirmationStudyContract} !=
           std::string_view{PLC::kAdaptiveStudyContract});
}

void TestPhase5CBoundsDoNotChangeUpstreamPivotDetection()
{
    const std::vector<PL::CompletedBar> bars{
        Bar(0, 3, 1, 2), Bar(1, 9, 3, 5), Bar(2, 4, 2, 3),
        Bar(3, 5, 1, 3), Bar(4, 9.5, 2, 5), Bar(5, 4, 1, 2),
        Bar(6, 8, 2, 5), Bar(7, 3, 0, 1), Bar(8, 7, 2, 4)};
    const auto strictPivots = [&](std::size_t active, std::size_t evidence) {
        PLC::AdaptiveWidthResearchDetector detector{"eurusdrmp", {1, 3, 0.0,
            PLC::ScaleTiming::pivot_time, active, 64, evidence, 900s}};
        PLC::PrecedingRangeScale scale{3};
        std::vector<std::size_t> result;
        for (const PL::CompletedBar& bar : bars)
        {
            const PLC::AdaptiveUpdate update = detector.AddCompletedBar(
                bar, scale.BeforeCurrentBar());
            result.push_back(update.strictPivotsConfirmed);
            scale.AddCompletedRange(bar.high - bar.low);
        }
        return result;
    };
    const auto baseline = strictPivots(32, 8);
    assert(baseline == strictPivots(64, 8));
    assert(baseline == strictPivots(32, 32));
}

void TestPhase5CBoundsCapacityAndEvidenceAccounting()
{
    const std::vector<PL::CompletedBar> capacityBars{
        Bar(0, 3, 1, 2), Bar(1, 9, 3, 5), Bar(2, 4, 2, 3),
        Bar(3, 5, 1, 3), Bar(4, 9.5, 2, 5), Bar(5, 4, 1, 2)};
    const auto capacityEvictions = [&](std::size_t maxActive) {
        PLC::AdaptiveWidthResearchDetector detector{"eurusdrmp", {1, 3, 0.0,
            PLC::ScaleTiming::pivot_time, maxActive, 64, 3, 900s}};
        PLC::PrecedingRangeScale scale{3};
        std::size_t result = 0;
        for (const PL::CompletedBar& bar : capacityBars)
            for (const auto& observation : AddAdaptive(detector, scale, bar))
                if (observation.kind == PL::InteractionKind::level_evicted) ++result;
        return result;
    };
    assert(capacityEvictions(1) > 0);
    assert(capacityEvictions(8) == 0);

    const std::vector<PL::CompletedBar> evidenceBars{
        Bar(0, 3, 1, 2), Bar(1, 10, 3, 5), Bar(2, 4, 2, 3),
        Bar(3, 5, 1, 3), Bar(4, 10, 2, 5), Bar(5, 4, 1, 2),
        Bar(6, 5, 1, 3), Bar(7, 10, 2, 5), Bar(8, 4, 1, 2)};
    const auto saturatedReinforcements = [&](std::size_t maxEvidence) {
        PLC::AdaptiveWidthResearchDetector detector{"eurusdrmp", {1, 3, 0.0,
            PLC::ScaleTiming::pivot_time, 8, 64, maxEvidence, 900s}};
        PLC::PrecedingRangeScale scale{3};
        std::size_t reinforcements = 0;
        std::size_t saturated = 0;
        for (const PL::CompletedBar& bar : evidenceBars)
        {
            for (const auto& observation : AddAdaptive(detector, scale, bar))
            {
                if (observation.kind != PL::InteractionKind::level_reinforced) continue;
                ++reinforcements;
                if (observation.retainedEvidenceAlreadySaturated) ++saturated;
            }
        }
        return std::make_pair(reinforcements, saturated);
    };
    const auto small = saturatedReinforcements(1);
    const auto large = saturatedReinforcements(3);
    assert(small.first == 2 && small.second == 2);
    assert(large.first == 2 && large.second == 0);
}

void TestAdaptiveResearchDoesNotChangeV1Replay()
{
    const std::vector<PL::CompletedBar> bars{
        Bar(0, 3, 1, 2), Bar(1, 9, 3, 5), Bar(2, 4, 2, 3),
        Bar(3, 5, 1, 3), Bar(4, 9.5, 2, 5), Bar(5, 4, 1, 2)};
    const PL::Configuration v1{1, 0.5, 8, 20, 3, 900s};
    const auto v1Replay = [&] {
        PL::CausalPriceLevelEngine engine{"eurusdrmp", v1};
        std::vector<std::string> result;
        for (const PL::CompletedBar& bar : bars)
            result.push_back(engine.AddCompletedBar(bar).CanonicalRepresentation());
        return result;
    };
    const auto before = v1Replay();
    PLC::AdaptiveWidthResearchDetector research{"eurusdrmp", {1, 3, 1.0,
        PLC::ScaleTiming::pivot_time, 8, 20, 3, 900s}};
    PLC::PrecedingRangeScale scale{3};
    for (const PL::CompletedBar& bar : bars)
        (void)AddAdaptive(research, scale, bar);
    assert(before == v1Replay());
    assert(PL::CanonicalConfigurationIdentity(v1).find("id=causal-price-level") !=
           std::string::npos);
}
} // namespace

int main()
{
    TestStrictPivotsMatchV1();
    TestScaleUsesPrecedingRangesOnly();
    TestPivotScaleTimeAndDeterminism();
    TestChronologicalValidation();
    TestAdaptiveScaleBoundariesAndFrozenWidth();
    TestAdaptiveDeterminismChronologyAndIdentityIsolation();
    TestAdaptiveAgeBoundaryCapacityAndCensoring();
    TestAdaptiveAgeDoesNotChangePivotDetection();
    TestPhase5CBoundsGridAndIdentity();
    TestPhase5CBoundsDoNotChangeUpstreamPivotDetection();
    TestPhase5CBoundsCapacityAndEvidenceAccounting();
    TestAdaptiveResearchDoesNotChangeV1Replay();
    std::cout << "PriceLevelCharacterizationTests passed\n";
}
