#include "CausalPriceLevelEngine.hpp"
#include "CausalPriceLevelV2Engine.hpp"
#include "CausalPriceLevelRawFeatures.hpp"
#include "PriceLevelCharacterization.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace
{
namespace PL = EA::PriceLevel;
namespace PLV2 = EA::PriceLevel::V2;
namespace PLC = EA::PriceLevel::Characterization;
namespace PLR = EA::PriceLevel::Raw;
using namespace std::chrono_literals;

PL::CompletedBar Bar(std::size_t index, double high, double low, double close)
{
    return {std::chrono::sys_seconds{1'700'000'000s +
            std::chrono::seconds{static_cast<std::int64_t>(index) * 900}},
            close, high, low, close};
}

bool Has(const PLV2::Update& update, PL::InteractionKind kind)
{
    return std::any_of(update.observations.begin(), update.observations.end(),
        [kind](const PLV2::Observation& observation) { return observation.kind == kind; });
}

const PLV2::Observation& Find(const PLV2::Update& update, PL::InteractionKind kind)
{
    const auto found = std::find_if(update.observations.begin(), update.observations.end(),
        [kind](const PLV2::Observation& observation) { return observation.kind == kind; });
    assert(found != update.observations.end());
    return *found;
}

PLC::AdaptiveConfiguration ResearchConfiguration(const PLV2::Configuration& value)
{
    return {value.pivotRadiusBars, value.scaleLookbackBars, value.scaleMultiplier,
            PLC::ScaleTiming::pivot_time, value.maxActiveLevels, value.maxAgeBars,
            value.maxRetainedPivotEvidence, value.completedBarDuration};
}

using Event = std::tuple<int, PL::PivotKind, PL::Role, double, double, std::size_t,
                         std::size_t, std::size_t, bool>;

std::vector<Event> Events(const PLV2::Update& update)
{
    std::vector<Event> result;
    for (const auto& observation : update.observations)
    {
        const auto& level = observation.level;
        result.emplace_back(static_cast<int>(observation.kind), level.originatingPivot,
            level.currentRole, level.anchorPrice, level.zoneHalfWidth, level.originBar,
            level.availableBar, level.retainedPivotEvidence.size(),
            observation.retainedEvidenceAlreadySaturated);
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<Event> Events(const PLC::AdaptiveUpdate& update)
{
    std::vector<Event> result;
    for (const auto& observation : update.observations)
    {
        const auto& level = observation.level;
        result.emplace_back(static_cast<int>(observation.kind), level.originatingPivot,
            level.currentRole, level.anchorPrice, level.zoneHalfWidth, level.originBar,
            level.availableBar, level.retainedPivotEvidenceCount,
            observation.retainedEvidenceAlreadySaturated);
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<Event> Active(const PLV2::Update& update)
{
    std::vector<Event> result;
    for (const auto& level : update.activeLevels)
        result.emplace_back(-1, level.originatingPivot, level.currentRole,
            level.anchorPrice, level.zoneHalfWidth, level.originBar, level.availableBar,
            level.retainedPivotEvidence.size(), false);
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<Event> Active(const PLC::AdaptiveUpdate& update)
{
    std::vector<Event> result;
    for (const auto& level : update.activeLevels)
        result.emplace_back(-1, level.originatingPivot, level.currentRole,
            level.anchorPrice, level.zoneHalfWidth, level.originBar, level.availableBar,
            level.retainedPivotEvidenceCount, false);
    std::sort(result.begin(), result.end());
    return result;
}

// Exact event-state parity against the frozen research implementation.  The
// two definitions intentionally emit distinct identities, so comparisons are
// over the semantic payload rather than research identity text.
void AssertResearchParity(const PLV2::Configuration& configuration,
                          const std::vector<PL::CompletedBar>& bars)
{
    PLV2::CausalPriceLevelEngine production{"eurusdrmp", configuration};
    PLC::AdaptiveWidthResearchDetector research{"eurusdrmp",
        ResearchConfiguration(configuration)};
    PLC::PrecedingRangeScale scale{configuration.scaleLookbackBars};
    for (const PL::CompletedBar& bar : bars)
    {
        const auto expectedScale = scale.BeforeCurrentBar();
        const PLV2::Update productionUpdate = production.AddCompletedBar(bar);
        const PLC::AdaptiveUpdate researchUpdate = research.AddCompletedBar(bar, expectedScale);
        assert(productionUpdate.scaleBeforeCurrentBar.count == expectedScale.count);
        assert(productionUpdate.scaleBeforeCurrentBar.median == expectedScale.median);
        assert(productionUpdate.strictPivotsConfirmed == researchUpdate.strictPivotsConfirmed);
        assert(Events(productionUpdate) == Events(researchUpdate));
        assert(Active(productionUpdate) == Active(researchUpdate));
        scale.AddCompletedRange(bar.high - bar.low);
    }
}

void TestFrozenContractAndIdentityBoundary()
{
    const PLV2::Configuration configuration = PLV2::ProductionConfiguration();
    assert(configuration.pivotRadiusBars == 3);
    assert(configuration.scaleLookbackBars == 64);
    assert(configuration.scaleMultiplier == 1.0);
    assert(configuration.maxAgeBars == 512);
    assert(configuration.maxActiveLevels == 48);
    assert(configuration.maxRetainedPivotEvidence == 32);
    assert(configuration.completedBarDuration == 900s);
    const std::string v2 = PLV2::CanonicalConfigurationIdentity(configuration);
    assert(v2.find("id=causal-price-level;version=v2") != std::string::npos);
    assert(v2.find("scale_statistic=median_high_low") != std::string::npos);
    assert(v2.find("scale_timing=pivot_time") != std::string::npos);
    assert(v2.find(PLC::kAdaptiveStudyContract) == std::string::npos);

    const PL::Configuration v1{1, 0.5, 4, 20, 3, 900s};
    const std::string v1Identity = PL::CanonicalConfigurationIdentity(v1);
    assert(v1Identity.find("price-level-definition-v1;id=causal-price-level;version=v1") == 0);
    assert(v1Identity.find("scale_") == std::string::npos);
    assert(v1Identity != v2);
}

void TestCausalScaleStartupAndConfirmationDelay()
{
    // At pivot bar one, only bar zero's range (2) is eligible.  The wide
    // confirmation bar cannot revise the resulting width.
    const PLV2::Configuration configuration{1, 64, 1.0, 20, 8, 3, 900s};
    const std::vector<PL::CompletedBar> first{
        Bar(0, 3, 1, 2), Bar(1, 9, 3, 5), Bar(2, 4, 2, 3)};
    AssertResearchParity(configuration, first);
    PLV2::CausalPriceLevelEngine production{"eurusdrmp", configuration};
    (void)production.AddCompletedBar(first[0]);
    const PLV2::Update pivot = production.AddCompletedBar(first[1]);
    assert(pivot.scaleBeforeCurrentBar.count == 1);
    assert(pivot.scaleBeforeCurrentBar.median == 2.0);
    const PLV2::Update confirmation = production.AddCompletedBar(first[2]);
    assert(Find(confirmation, PL::InteractionKind::level_established).
               level.zoneHalfWidth == 2.0);

    const std::vector<PL::CompletedBar> delayedA{
        Bar(0, 2, 1, 1.5), Bar(1, 3, 2, 2.5), Bar(2, 4, 3, 3.5),
        Bar(3, 20, 19, 19.5), Bar(4, 4, 3, 3.5), Bar(5, 3, 2, 2.5),
        Bar(6, 2, 1, 1.5)};
    const std::vector<PL::CompletedBar> delayedB{
        Bar(0, 2, 1, 1.5), Bar(1, 3, 2, 2.5), Bar(2, 4, 3, 3.5),
        Bar(3, 20, 19, 19.5), Bar(4, 4, -96, 3.5), Bar(5, 3, -97, 2.5),
        Bar(6, 2, -98, 1.5)};
    const auto widthAtConfirmation = [](const std::vector<PL::CompletedBar>& bars) {
        PLV2::CausalPriceLevelEngine engine{"eurusdrmp", PLV2::ProductionConfiguration()};
        double width = -1.0;
        for (const auto& bar : bars)
            for (const auto& observation : engine.AddCompletedBar(bar).observations)
                if (observation.kind == PL::InteractionKind::level_established &&
                    observation.level.originBar == 3)
                    width = observation.level.zoneHalfWidth;
        return width;
    };
    // Radius three delays availability, but both replays snapshot the same
    // predecessor prefix at pivot time (three unit ranges).
    assert(widthAtConfirmation(delayedA) == 1.0);
    assert(widthAtConfirmation(delayedA) == widthAtConfirmation(delayedB));
    AssertResearchParity(PLV2::ProductionConfiguration(), delayedA);
    AssertResearchParity(PLV2::ProductionConfiguration(), delayedB);
}

void TestFrozenWidthMergeAndInteractionParity()
{
    const PLV2::Configuration configuration{1, 3, 1.0, 20, 8, 3, 900s};
    const std::vector<PL::CompletedBar> bars{
        Bar(0, 3, 1, 2), Bar(1, 9, 3, 5), Bar(2, 4, 2, 3),
        Bar(3, 5, 2, 3), Bar(4, 9.5, 2, 5), Bar(5, 4, 2, 2),
        Bar(6, 13, 12, 13), Bar(7, 13, 8, 9)};
    AssertResearchParity(configuration, bars);

    PLV2::CausalPriceLevelEngine engine{"eurusdrmp", configuration};
    PLV2::Update last;
    for (const auto& bar : bars) last = engine.AddCompletedBar(bar);
    assert(last.activeLevels.size() == 1);
    assert(last.activeLevels[0].anchorPrice == 9.0);
    assert(last.activeLevels[0].zoneHalfWidth == 2.0);
    assert(last.activeLevels[0].pivotObservationCount == 2);
    assert(last.activeLevels[0].lower == 7.0 && last.activeLevels[0].upper == 11.0);
    assert(Has(last, PL::InteractionKind::touch));
    assert(Has(last, PL::InteractionKind::retest));
    assert(last.activeLevels[0].currentRole == PL::Role::support_like);
}

void TestNearestAnchorIdentityTieAndNoChaining()
{
    // Constant one-point ranges make every candidate width exactly one.  The
    // 11 pivot is equidistant from 10 and 12, so the earlier canonical origin
    // wins; the original 10 anchor remains fixed and 12 does not chain it.
    const PLV2::Configuration configuration{1, 3, 1.0, 20, 8, 3, 900s};
    const std::vector<PL::CompletedBar> bars{
        Bar(0, 1, 0, .5), Bar(1, 10, 9, 9.5), Bar(2, 1, 0, .5),
        Bar(3, 1, 0, .5), Bar(4, 12, 11, 11.5), Bar(5, 1, 0, .5),
        Bar(6, 1, 0, .5), Bar(7, 11, 10, 10.5), Bar(8, 1, 0, .5)};
    AssertResearchParity(configuration, bars);
    PLV2::CausalPriceLevelEngine engine{"eurusdrmp", configuration};
    PLV2::Update last;
    for (const auto& bar : bars) last = engine.AddCompletedBar(bar);
    assert(last.activeLevels.size() == 2);
    assert(last.activeLevels[0].anchorPrice == 10.0);
    assert(last.activeLevels[0].pivotObservationCount == 2);
    assert(last.activeLevels[1].anchorPrice == 12.0);
    assert(last.activeLevels[1].pivotObservationCount == 1);
}

void TestAgeAndBoundParity()
{
    const std::vector<PL::CompletedBar> ageBars{
        Bar(0, 3, 1, 2), Bar(1, 9, 3, 5), Bar(2, 4, 2, 3),
        Bar(3, 5, 1, 3), Bar(4, 2, 0, 1)};
    const PLV2::Configuration ageOne{1, 3, 0.0, 1, 8, 3, 900s};
    AssertResearchParity(ageOne, ageBars);
    PLV2::CausalPriceLevelEngine aging{"eurusdrmp", ageOne};
    (void)aging.AddCompletedBar(ageBars[0]);
    (void)aging.AddCompletedBar(ageBars[1]);
    (void)aging.AddCompletedBar(ageBars[2]);
    assert(!Has(aging.AddCompletedBar(ageBars[3]), PL::InteractionKind::level_expired));
    assert(Has(aging.AddCompletedBar(ageBars[4]), PL::InteractionKind::level_expired));

    const std::vector<PL::CompletedBar> capacityBars{
        Bar(0, 3, 1, 2), Bar(1, 9, 3, 5), Bar(2, 4, 2, 3),
        Bar(3, 5, 1, 3), Bar(4, 12, 2, 5), Bar(5, 4, 1, 2)};
    const PLV2::Configuration capacityOne{1, 3, 0.0, 64, 1, 1, 900s};
    AssertResearchParity(capacityOne, capacityBars);
    const PLV2::Configuration capacityMany{1, 3, 0.0, 64, 8, 3, 900s};
    AssertResearchParity(capacityMany, capacityBars);
    const auto evictionCount = [&](const PLV2::Configuration& configuration) {
        PLV2::CausalPriceLevelEngine engine{"eurusdrmp", configuration};
        std::size_t result = 0;
        for (const auto& bar : capacityBars)
            for (const auto& observation : engine.AddCompletedBar(bar).observations)
                if (observation.kind == PL::InteractionKind::level_evicted) ++result;
        return result;
    };
    assert(evictionCount(capacityOne) > 0);
    assert(evictionCount(capacityMany) == 0);

    const std::vector<PL::CompletedBar> evidenceBars{
        Bar(0, 3, 1, 2), Bar(1, 10, 3, 5), Bar(2, 4, 2, 3),
        Bar(3, 5, 1, 3), Bar(4, 10, 2, 5), Bar(5, 4, 1, 2),
        Bar(6, 5, 1, 3), Bar(7, 10, 2, 5), Bar(8, 4, 1, 2)};
    const PLV2::Configuration evidenceOne{1, 3, 0.0, 64, 8, 1, 900s};
    const PLV2::Configuration evidenceThree{1, 3, 0.0, 64, 8, 3, 900s};
    AssertResearchParity(evidenceOne, evidenceBars);
    AssertResearchParity(evidenceThree, evidenceBars);
    const auto evidenceAccounting = [&](const PLV2::Configuration& configuration) {
        PLV2::CausalPriceLevelEngine engine{"eurusdrmp", configuration};
        std::size_t reinforcements = 0;
        std::size_t saturated = 0;
        std::vector<std::size_t> strictPivots;
        for (const auto& bar : evidenceBars)
        {
            const PLV2::Update update = engine.AddCompletedBar(bar);
            strictPivots.push_back(update.strictPivotsConfirmed);
            for (const auto& observation : update.observations)
            {
                if (observation.kind != PL::InteractionKind::level_reinforced) continue;
                ++reinforcements;
                if (observation.retainedEvidenceAlreadySaturated) ++saturated;
            }
        }
        return std::make_tuple(reinforcements, saturated, strictPivots);
    };
    const auto smallEvidence = evidenceAccounting(evidenceOne);
    const auto largeEvidence = evidenceAccounting(evidenceThree);
    assert(std::get<0>(smallEvidence) == 2 && std::get<1>(smallEvidence) == 2);
    assert(std::get<0>(largeEvidence) == 2 && std::get<1>(largeEvidence) == 0);
    assert(std::get<2>(smallEvidence) == std::get<2>(largeEvidence));
}

PLV2::Level RawLevel(std::string identity, double anchor, double width,
                     double lower, double upper, PL::Role role,
                     std::size_t availableBar, std::size_t evidence = 0)
{
    PLV2::Level level;
    level.identity = std::move(identity);
    level.anchorPrice = anchor;
    level.zoneHalfWidth = width;
    level.lower = lower;
    level.upper = upper;
    level.currentRole = role;
    level.availableBar = availableBar;
    level.retainedPivotEvidence.resize(evidence, "evidence");
    return level;
}

PLV2::Observation RawObservation(const PLV2::Level& level,
                                 PL::InteractionKind kind,
                                 std::chrono::sys_seconds decision,
                                 bool saturated = false)
{
    PLV2::Observation observation;
    observation.levelIdentity = level.identity;
    observation.kind = kind;
    observation.availableAt = decision;
    observation.retainedEvidenceAlreadySaturated = saturated;
    return observation;
}

void TestFrozenRawModelExposureProjection()
{
    constexpr std::array<std::string_view, 11> names{{
        "available", "zone_scale_valid", "zone_gap_signed_clipped",
        "zone_relation", "current_role", "age_fraction",
        "prior_evidence_saturation", "touch_now", "cross_direction_now",
        "retest_now", "role_reversal_now"}};
    static_assert(PLR::kFeatureCount == names.size());
    assert(PLR::kFeatureNames == names);

    PLR::Producer producer;
    PLV2::Update none;
    none.decisionTime = std::chrono::sys_seconds{900s};
    assert(producer.Project(none, 1.0, 0) == PLR::FeatureVector{});

    const auto decision = std::chrono::sys_seconds{1'800s};
    const auto selected = RawLevel("selected", 10.0, 2.0, 8.0, 12.0,
                                   PL::Role::support_like, 0, 2);
    PLV2::Update update;
    update.decisionTime = decision;
    update.activeLevels = {selected};
    update.observations = {
        RawObservation(selected, PL::InteractionKind::level_reinforced, decision),
        RawObservation(selected, PL::InteractionKind::touch, decision),
        RawObservation(selected, PL::InteractionKind::cross_up, decision),
        RawObservation(selected, PL::InteractionKind::retest, decision),
        RawObservation(selected, PL::InteractionKind::role_reversal, decision)};
    const auto features = producer.Project(update, 7.0, 256);
    assert((features == PLR::FeatureVector{{1.0f, 1.0f, -0.5f, -1.0f, 1.0f,
                                             0.5f, 1.0f / 32.0f, 1.0f, 1.0f,
                                             1.0f, 1.0f}}));
    assert(producer.Project(update, 8.0, 256)[2] == 0.0f);
    assert(producer.Project(update, 20.0, 256)[2] == 1.0f);

    const auto zeroWidth = RawLevel("zero", 5.0, 0.0, 5.0, 5.0,
                                    PL::Role::resistance_like, 512);
    update.activeLevels = {zeroWidth};
    update.observations.clear();
    const auto zeroWidthFeatures = producer.Project(update, 6.0, 512);
    assert(zeroWidthFeatures[0] == 1.0f && zeroWidthFeatures[1] == 0.0f);
    assert(zeroWidthFeatures[2] == 0.0f && zeroWidthFeatures[3] == 1.0f);
    assert(zeroWidthFeatures[4] == -1.0f && zeroWidthFeatures[5] == 0.0f);

    // Retests take precedence over the nearest otherwise eligible active level.
    const auto nearest = RawLevel("nearest", 0.0, 1.0, -1.0, 1.0,
                                  PL::Role::support_like, 0);
    const auto retested = RawLevel("retested", 20.0, 1.0, 19.0, 21.0,
                                   PL::Role::resistance_like, 0);
    update.activeLevels = {nearest, retested};
    update.observations = {RawObservation(retested, PL::InteractionKind::retest, decision)};
    assert(producer.Project(update, 0.0, 1)[4] == -1.0f);

    // Fallback ranking is zone distance, anchor distance, anchor, identity.
    const auto nearerAnchor = RawLevel("nearer", -1.0, 0.0, -1.0, -1.0,
                                       PL::Role::support_like, 0);
    const auto fartherAnchor = RawLevel("farther", 2.0, 1.0, 1.0, 3.0,
                                        PL::Role::resistance_like, 0);
    update.activeLevels = {fartherAnchor, nearerAnchor};
    update.observations.clear();
    assert(producer.Project(update, 0.0, 1)[4] == 1.0f);
    const auto identityZ = RawLevel("z", 3.0, 1.0, 2.0, 4.0,
                                    PL::Role::resistance_like, 0);
    const auto identityA = RawLevel("a", 3.0, 1.0, 2.0, 4.0,
                                    PL::Role::support_like, 0);
    update.activeLevels = {identityZ, identityA};
    assert(producer.Project(update, 0.0, 1)[4] == 1.0f);
    const auto lowerAnchor = RawLevel("later-identity", -1.0, 1.0, -2.0, 0.0,
                                      PL::Role::support_like, 0);
    const auto higherAnchor = RawLevel("earlier-identity", 1.0, 1.0, 0.0, 2.0,
                                       PL::Role::resistance_like, 0);
    update.activeLevels = {higherAnchor, lowerAnchor};
    assert(producer.Project(update, 0.0, 1)[4] == 1.0f);

    // A retest for an evicted/expired identity is ineligible; fallback remains active.
    update.activeLevels = {nearest};
    update.observations = {RawObservation(retested, PL::InteractionKind::retest, decision)};
    assert(producer.Project(update, 0.0, 1)[4] == 1.0f);

    // The final selectable bar is one; an active level beyond it is malformed.
    update.activeLevels = {RawLevel("age", 1.0, 1.0, 0.0, 2.0,
                                    PL::Role::support_like, 0, 32)};
    update.observations.clear();
    assert(producer.Project(update, 1.0, 512)[5] == 1.0f);
    assert(producer.Project(update, 1.0, 512)[6] == 1.0f);
    update.activeLevels = {RawLevel("saturated", 1.0, 1.0, 0.0, 2.0,
                                    PL::Role::support_like, 512, 1)};
    update.observations = {RawObservation(update.activeLevels[0],
        PL::InteractionKind::level_reinforced, decision, true)};
    assert(producer.Project(update, 1.0, 512)[6] == 1.0f / 32.0f);

    bool malformed = false;
    update.observations = {
        RawObservation(update.activeLevels[0], PL::InteractionKind::cross_up, decision),
        RawObservation(update.activeLevels[0], PL::InteractionKind::cross_down, decision)};
    try { (void)producer.Project(update, 1.0, 512); }
    catch (const std::runtime_error&) { malformed = true; }
    assert(malformed);
}
} // namespace

int main()
{
    TestFrozenContractAndIdentityBoundary();
    TestCausalScaleStartupAndConfirmationDelay();
    TestFrozenWidthMergeAndInteractionParity();
    TestNearestAnchorIdentityTieAndNoChaining();
    TestAgeAndBoundParity();
    TestFrozenRawModelExposureProjection();
    std::cout << "CausalPriceLevelV2EngineTests passed\n";
}
