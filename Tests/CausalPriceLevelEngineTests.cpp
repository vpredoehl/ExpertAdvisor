#include "CausalPriceLevelEngine.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
namespace PL = EA::PriceLevel;
using namespace std::chrono_literals;

PL::Configuration Config(std::size_t cap = 4, std::size_t age = 20,
                         double width = 0.5)
{
    return {1, width, cap, age, 3, 900s};
}

PL::CompletedBar Bar(std::size_t index, double high, double low, double close)
{
    return {std::chrono::sys_seconds{1'700'000'000s +
            std::chrono::seconds{static_cast<std::int64_t>(index) * 900}},
            close, high, low, close};
}

bool Has(const PL::Update& update, PL::InteractionKind kind)
{
    return std::any_of(update.observations.begin(), update.observations.end(),
        [kind](const PL::Observation& value) { return value.kind == kind; });
}

const PL::Observation& Find(const PL::Update& update, PL::InteractionKind kind)
{
    const auto found = std::find_if(update.observations.begin(),
        update.observations.end(), [kind](const PL::Observation& value) {
            return value.kind == kind;
        });
    assert(found != update.observations.end());
    return *found;
}

template <typename Fn>
bool ThrowsInvalidArgument(Fn&& fn)
{
    try { fn(); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

void TestCausalEstablishmentAndNoBackdating()
{
    PL::CausalPriceLevelEngine engine{"EURUSDRMP", Config()};
    const auto first = engine.AddCompletedBar(Bar(0, 10.0, 5.0, 7.0));
    const auto pivot = engine.AddCompletedBar(Bar(1, 12.0, 6.0, 8.0));
    const auto confirmation = engine.AddCompletedBar(Bar(2, 9.0, 4.0, 6.0));
    assert(first.observations.empty());
    assert(pivot.observations.empty());
    assert(Has(confirmation, PL::InteractionKind::level_established));
    const PL::Observation& established = Find(confirmation,
        PL::InteractionKind::level_established);
    assert(established.observedAt == Bar(1, 12.0, 6.0, 8.0).barStart + 900s);
    assert(established.availableAt == Bar(2, 9.0, 4.0, 6.0).barStart + 900s);
    assert(established.observedAt < established.availableAt);
    assert(confirmation.activeLevels.size() == 1);
    assert(confirmation.activeLevels[0].originatingPivot == PL::PivotKind::high);
    assert(confirmation.activeLevels[0].currentRole == PL::Role::resistance_like);
    assert(confirmation.activeLevels[0].lower == 11.5);
    assert(confirmation.activeLevels[0].upper == 12.5);
}

void TestReplayPrefixAndImmutableEarlierOutput()
{
    const std::vector<PL::CompletedBar> bars{
        Bar(0, 10, 5, 7), Bar(1, 12, 6, 8), Bar(2, 9, 4, 6),
        Bar(3, 13, 11, 13), Bar(4, 12.4, 11.8, 12.1), Bar(5, 10, 7, 8),
    };
    PL::CausalPriceLevelEngine prefix{"eurusdrmp", Config()};
    std::vector<std::string> first;
    for (std::size_t index = 0; index != 4; ++index)
        first.push_back(prefix.AddCompletedBar(bars[index]).CanonicalRepresentation());

    PL::CausalPriceLevelEngine full{" EURUSDRMP ", Config()};
    std::vector<std::string> repeated;
    for (const auto& bar : bars)
        repeated.push_back(full.AddCompletedBar(bar).CanonicalRepresentation());
    for (std::size_t index = 0; index != first.size(); ++index)
        assert(first[index] == repeated[index]);

    PL::CausalPriceLevelEngine same{"eurusdrmp", Config()};
    for (const auto& bar : bars)
        assert(same.AddCompletedBar(bar).CanonicalRepresentation() ==
               repeated[&bar - bars.data()]);
}

void TestMergeIdentityExactToleranceAndTieBreak()
{
    // High pivots at 10.0 and 10.5 merge on the inclusive half-width boundary;
    // their fixed first anchor prevents subsequent chain recentering.
    const std::vector<PL::CompletedBar> bars{
        Bar(0, 8, 1, 4), Bar(1, 10, 2, 5), Bar(2, 8, 1, 4),
        Bar(3, 8, 1, 4), Bar(4, 10.5, 2, 5), Bar(5, 8, 1, 4),
        Bar(6, 8, 1, 4), Bar(7, 11.0, 2, 5), Bar(8, 8, 1, 4),
    };
    PL::CausalPriceLevelEngine first{"eurusdrmp", Config(4, 30, 0.5)};
    PL::CausalPriceLevelEngine replay{"eurusdrmp", Config(4, 30, 0.5)};
    PL::Update last;
    for (const auto& bar : bars)
    {
        last = first.AddCompletedBar(bar);
        assert(last.CanonicalRepresentation() == replay.AddCompletedBar(bar).
               CanonicalRepresentation());
    }
    assert(last.activeLevels.size() == 2);
    assert(last.activeLevels[0].anchorPrice == 10.0);
    assert(last.activeLevels[0].pivotObservationCount == 2);
    assert(last.activeLevels[0].retainedPivotEvidence.size() == 2);
    assert(last.activeLevels[1].anchorPrice == 11.0);
}

void TestNearestMergeTieAndInclusiveTouchBoundary()
{
    // The 11.0 pivot is exactly one unit from both fixed anchors. Canonical
    // identity resolves the tie, instead of depending on vector insertion.
    const std::vector<PL::CompletedBar> tieBars{
        Bar(0, 8, 1, 4), Bar(1, 10, 1, 5), Bar(2, 8, 1, 4),
        Bar(3, 8, 1, 4), Bar(4, 12, 1, 5), Bar(5, 8, 1, 4),
        Bar(6, 8, 1, 4), Bar(7, 11, 1, 5), Bar(8, 8, 1, 4),
    };
    PL::CausalPriceLevelEngine tie{"eurusdrmp", Config(4, 30, 1.0)};
    PL::Update last;
    for (const auto& bar : tieBars) last = tie.AddCompletedBar(bar);
    assert(last.activeLevels.size() == 2);
    assert(last.activeLevels[0].anchorPrice == 10.0);
    assert(last.activeLevels[0].pivotObservationCount == 2);
    assert(last.activeLevels[1].anchorPrice == 12.0);
    assert(last.activeLevels[1].pivotObservationCount == 1);

    PL::CausalPriceLevelEngine boundary{"eurusdrmp", Config()};
    (void)boundary.AddCompletedBar(Bar(0, 10, 5, 7));
    (void)boundary.AddCompletedBar(Bar(1, 12, 6, 8));
    (void)boundary.AddCompletedBar(Bar(2, 9, 4, 6));
    const auto exactLower = boundary.AddCompletedBar(Bar(3, 11.5, 10, 10.5));
    assert(Has(exactLower, PL::InteractionKind::touch));
    PL::CausalPriceLevelEngine outside{"eurusdrmp", Config()};
    (void)outside.AddCompletedBar(Bar(0, 10, 5, 7));
    (void)outside.AddCompletedBar(Bar(1, 12, 6, 8));
    (void)outside.AddCompletedBar(Bar(2, 9, 4, 6));
    const auto belowLower = outside.AddCompletedBar(Bar(3,
        std::nextafter(11.5, -std::numeric_limits<double>::infinity()), 9, 10));
    assert(!Has(belowLower, PL::InteractionKind::touch));
}

void TestInteractionCrossRetestAndRoleReversalTiming()
{
    PL::CausalPriceLevelEngine engine{"eurusdrmp", Config()};
    (void)engine.AddCompletedBar(Bar(0, 10, 5, 7));
    (void)engine.AddCompletedBar(Bar(1, 12, 6, 8));
    const auto established = engine.AddCompletedBar(Bar(2, 9, 4, 6));
    const std::string level = Find(established, PL::InteractionKind::level_established).
        levelIdentity;
    const auto cross = engine.AddCompletedBar(Bar(3, 13, 12.6, 13));
    assert(Has(cross, PL::InteractionKind::cross_up));
    assert(Has(cross, PL::InteractionKind::role_reversal));
    assert(!Has(cross, PL::InteractionKind::retest));
    const auto crossObservation = Find(cross, PL::InteractionKind::cross_up);
    assert(crossObservation.levelIdentity == level);
    assert(crossObservation.roleBefore == PL::Role::resistance_like);
    assert(crossObservation.roleAfter == PL::Role::support_like);
    const auto retest = engine.AddCompletedBar(Bar(4, 12.4, 11.8, 12.1));
    assert(Has(retest, PL::InteractionKind::touch));
    assert(Has(retest, PL::InteractionKind::retest));
    assert(!Has(engine.AddCompletedBar(Bar(5, 12.3, 11.9, 12.2)),
        PL::InteractionKind::retest));
}

void TestBoundedCapacityAndExpiry()
{
    PL::CausalPriceLevelEngine capped{"eurusdrmp", Config(1, 20, 0.1)};
    (void)capped.AddCompletedBar(Bar(0, 8, 1, 4));
    (void)capped.AddCompletedBar(Bar(1, 10, 2, 5));
    (void)capped.AddCompletedBar(Bar(2, 8, 1, 4));
    (void)capped.AddCompletedBar(Bar(3, 8, 1, 4));
    (void)capped.AddCompletedBar(Bar(4, 12, 2, 5));
    const auto eviction = capped.AddCompletedBar(Bar(5, 8, 1, 4));
    assert(Has(eviction, PL::InteractionKind::level_evicted));
    assert(eviction.activeLevels.size() == 1);
    assert(eviction.activeLevels[0].anchorPrice == 12.0);

    PL::CausalPriceLevelEngine aging{"eurusdrmp", Config(4, 1, 0.1)};
    (void)aging.AddCompletedBar(Bar(0, 8, 1, 4));
    (void)aging.AddCompletedBar(Bar(1, 10, 2, 5));
    (void)aging.AddCompletedBar(Bar(2, 8, 1, 4));
    (void)aging.AddCompletedBar(Bar(3, 8, 1, 4));
    const auto expired = aging.AddCompletedBar(Bar(4, 8, 1, 4));
    assert(Has(expired, PL::InteractionKind::level_expired));
    assert(expired.activeLevels.empty());
}

void TestStrictPivotsInputAndConfigurationValidation()
{
    PL::CausalPriceLevelEngine equal{"eurusdrmp", Config()};
    (void)equal.AddCompletedBar(Bar(0, 8, 2, 4));
    (void)equal.AddCompletedBar(Bar(1, 10, 2, 5));
    const auto tied = equal.AddCompletedBar(Bar(2, 10, 1, 4));
    assert(tied.activeLevels.empty());

    assert(ThrowsInvalidArgument([] {
        PL::CausalPriceLevelEngine engine{"eurusdrmp", {0, 0.1, 1, 1, 1, 900s}};
        (void)engine;
    }));
    assert(ThrowsInvalidArgument([] {
        PL::CausalPriceLevelEngine engine{"eurusdrmp", {1,
            std::numeric_limits<double>::quiet_NaN(), 1, 1, 1, 900s}};
        (void)engine;
    }));
    PL::CausalPriceLevelEngine engine{"eurusdrmp", Config()};
    assert(ThrowsInvalidArgument([&] {
        (void)engine.AddCompletedBar(Bar(0,
            std::numeric_limits<double>::infinity(), 1, 2));
    }));
    (void)engine.AddCompletedBar(Bar(1, 8, 1, 4));
    assert(ThrowsInvalidArgument([&] { (void)engine.AddCompletedBar(Bar(1, 8, 1, 4)); }));
    assert(ThrowsInvalidArgument([&] { (void)engine.AddCompletedBar(Bar(0, 8, 1, 4)); }));
}
} // namespace

int main()
{
    TestCausalEstablishmentAndNoBackdating();
    TestReplayPrefixAndImmutableEarlierOutput();
    TestMergeIdentityExactToleranceAndTieBreak();
    TestNearestMergeTieAndInclusiveTouchBoundary();
    TestInteractionCrossRetestAndRoleReversalTiming();
    TestBoundedCapacityAndExpiry();
    TestStrictPivotsInputAndConfigurationValidation();
    std::cout << "CausalPriceLevelEngineTests passed\n";
}
