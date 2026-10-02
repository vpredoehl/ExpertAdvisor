#include "PriceLevelMarketStructureObservationAdapter.hpp"
#include "ModelInputContract.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace std::chrono_literals;
namespace MS = EA::MarketStructure;
namespace Bridge = EA::MarketStructure::PriceLevelBridge;
namespace PL = EA::PriceLevel;

PL::Configuration Configuration()
{
    return {1, 0.5, 4, 20, 3, 900s};
}

PL::CompletedBar Bar(std::size_t index, double high, double low, double close)
{
    return {std::chrono::sys_seconds{1'700'000'000s +
            std::chrono::seconds{static_cast<std::int64_t>(index) * 900}},
            close, high, low, close};
}

struct SourceContract final
{
    PL::CausalPriceLevelEngine engine{" EURUSDRMP ", Configuration()};
    std::string levelIdentity = "price-level-v1;symbol=eurusdrmp;definition=" +
        engine.configurationIdentity() + ";origin=price-level-pivot-v1;kind=high;bar=1;"
        "bar_start=1700000900;price=12";
};

PL::Observation Source(SourceContract& contract, PL::InteractionKind kind,
                       std::chrono::sys_seconds observed,
                       std::chrono::sys_seconds available,
                       std::string evidence)
{
    std::optional<PL::Role> before;
    std::optional<PL::Role> after;
    if (kind == PL::InteractionKind::cross_up ||
        kind == PL::InteractionKind::role_reversal)
    {
        before = PL::Role::resistance_like;
        after = PL::Role::support_like;
    }
    else if (kind == PL::InteractionKind::cross_down)
    {
        before = PL::Role::support_like;
        after = PL::Role::resistance_like;
    }
    const std::string identity = "price-level-observation-v1;level=" +
        contract.levelIdentity + ";kind=" + std::string{PL::CanonicalInteractionKind(kind)} +
        ";observed=" + PL::CanonicalTimestamp(observed) + ";available=" +
        PL::CanonicalTimestamp(available) + ";source=" + evidence + ";before=" +
        (before ? std::string{PL::CanonicalRole(*before)} : "absent") + ";after=" +
        (after ? std::string{PL::CanonicalRole(*after)} : "absent");
    return {identity, contract.levelIdentity, kind, observed, available,
            contract.engine.sourceProvenance(), std::move(evidence), before, after};
}

template <typename Fn>
bool ThrowsInvalidArgument(Fn&& fn)
{
    try { fn(); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

std::vector<std::string> Identities(const std::vector<MS::Observation>& observations)
{
    std::vector<std::string> result;
    result.reserve(observations.size());
    for (const MS::Observation& observation : observations)
        result.push_back(MS::CanonicalObservationIdentity(observation));
    return result;
}

void TestExactEventMappingExclusionsAndProvenance()
{
    SourceContract contract;
    Bridge::ObservationAdapter adapter{"EURUSDRMP"};
    const auto observed = std::chrono::sys_seconds{1'700'100'000s};
    const auto available = observed + 900s;
    const std::vector<PL::InteractionKind> allKinds{
        PL::InteractionKind::level_established,
        PL::InteractionKind::level_reinforced,
        PL::InteractionKind::touch,
        PL::InteractionKind::cross_up,
        PL::InteractionKind::cross_down,
        PL::InteractionKind::retest,
        PL::InteractionKind::role_reversal,
        PL::InteractionKind::level_expired,
        PL::InteractionKind::level_evicted,
    };
    std::vector<PL::Observation> source;
    for (std::size_t index = 0; index < allKinds.size(); ++index)
        source.push_back(Source(contract, allKinds[index], observed,
            available + std::chrono::seconds{static_cast<std::int64_t>(index)},
            "price-level-evidence-v1;ordinal=" + std::to_string(index)));
    const auto sourceCopy = source;
    const auto observations = adapter.Adapt(std::span<const PL::Observation>{source});
    assert(source == sourceCopy);
    assert(observations.size() == 7);

    for (const MS::Observation& observation : observations)
    {
        assert(observation.familyId == "price_level");
        assert(observation.detectorVersion == "causal-price-level/v1");
        assert(observation.descriptor.schemaVersion ==
               "price-level-market-structure-observation-v1");
        assert(observation.descriptor.polarity == MS::DescriptorPolarity::neutral);
        assert(!observation.descriptor.normalizedConfidence.has_value());
        assert(MS::CanonicalObservationIdentity(observation).starts_with(
            "observation-v2;"));
        assert(observation.sourceProvenance.starts_with(
            "price-level-market-structure-observation-bridge-v2;"));
        assert(observation.sourceProvenance.find("symbol=eurusdrmp") !=
               std::string::npos);
        assert(observation.sourceProvenance.find("source_provenance=") !=
               std::string::npos);
        const auto input = std::find_if(source.begin(), source.end(),
            [&observation](const PL::Observation& value) {
                return value.identity == observation.sourceObservationId;
            });
        assert(input != source.end());
        assert(observation.correlationKey.has_value());
        assert(observation.correlationKey->type ==
               "price_level.level_identity.v1");
        assert(observation.correlationKey->value == input->levelIdentity);
        assert(observation.descriptor.role == PL::CanonicalInteractionKind(input->kind));
        assert(observation.observedAt == input->observedAt);
        assert(observation.availableAt == input->availableAt);
    }
    for (const PL::InteractionKind excluded : {PL::InteractionKind::level_expired,
                                                PL::InteractionKind::level_evicted})
    {
        const auto input = std::find_if(source.begin(), source.end(),
            [excluded](const PL::Observation& value) { return value.kind == excluded; });
        assert(input != source.end());
        assert(std::none_of(observations.begin(), observations.end(),
            [&input](const MS::Observation& value) {
                return value.sourceObservationId == input->identity;
            }));
    }
}

void TestAvailabilityPrefixReplayAndGenericAcceptance()
{
    SourceContract contract;
    Bridge::ObservationAdapter canonical{" eurusdrmp "};
    Bridge::ObservationAdapter normalized{"EURUSDRMP"};
    const auto t0 = std::chrono::sys_seconds{1'700'200'000s};
    const auto t1 = t0 + 900s;
    const auto t2 = t1 + 900s;
    const auto t3 = t2 + 900s;
    const auto delayed = Source(contract, PL::InteractionKind::level_established,
        t0, t2, "price-level-pivot-v1;bar=1");
    const auto future = Source(contract, PL::InteractionKind::touch,
        t3, t3, "price-level-bar-v1;bar=3");
    const auto prefix = canonical.Adapt(std::span<const PL::Observation>{&delayed, 1});
    const std::vector<PL::Observation> fullSource{future, delayed};
    const auto full = normalized.Adapt(std::span<const PL::Observation>{fullSource});
    assert(prefix == normalized.Adapt(std::span<const PL::Observation>{&delayed, 1}));
    assert(MS::CausallyAvailableObservations(prefix, t1).empty());
    const auto atAvailability = MS::CausallyAvailableObservations(full, t2);
    assert(atAvailability.size() == 1);
    assert(atAvailability[0].sourceObservationId == delayed.identity);
    assert(atAvailability[0].availableAt == t2);
    assert(MS::CausallyAvailableObservations(prefix, t2) == atAvailability);

    const auto first = canonical.Adapt(std::span<const PL::Observation>{fullSource});
    const auto repeated = canonical.Adapt(std::span<const PL::Observation>{fullSource});
    assert(first == repeated);
    assert(Identities(first) == Identities(full));
    const auto identities = Identities(first);
    assert(std::is_sorted(identities.begin(), identities.end()));

    MS::DescriptiveConfluenceEngine engine{{"price-level-test", "v1",
        MS::RelationKind::support, {"price_level", "level_established"},
        {"price_level", "touch"}, 2}};
    // Neutral price-level descriptions are valid generic input but cannot be
    // silently treated as a directional support relationship.
    assert(engine.Evaluate(full, t3).outputs.empty());
}

void TestInvalidSourceDuplicateAndUpdateValidation()
{
    SourceContract contract;
    Bridge::ObservationAdapter adapter{"eurusdrmp"};
    const auto t = std::chrono::sys_seconds{1'700'300'000s};
    const auto source = Source(contract, PL::InteractionKind::cross_up, t, t,
        "price-level-bar-v1;bar=7");
    const std::vector<PL::Observation> duplicate{source, source};
    assert(ThrowsInvalidArgument([&] {
        (void)adapter.Adapt(std::span<const PL::Observation>{duplicate});
    }));
    auto malformedProvenance = source;
    malformedProvenance.sourceProvenance = "bad";
    assert(ThrowsInvalidArgument([&] {
        (void)adapter.Adapt(std::span<const PL::Observation>{&malformedProvenance, 1});
    }));
    auto malformedIdentity = source;
    malformedIdentity.identity += ";mutated";
    assert(ThrowsInvalidArgument([&] {
        (void)adapter.Adapt(std::span<const PL::Observation>{&malformedIdentity, 1});
    }));
    auto malformedRole = source;
    malformedRole.roleBefore = PL::Role::support_like;
    malformedRole.identity = Source(contract, PL::InteractionKind::cross_up, t, t,
        "price-level-bar-v1;bar=7").identity;
    assert(ThrowsInvalidArgument([&] {
        (void)adapter.Adapt(std::span<const PL::Observation>{&malformedRole, 1});
    }));
    SourceContract oversizedContract;
    oversizedContract.levelIdentity.append(4096, 'x');
    const auto oversizedKeySource = Source(oversizedContract,
        PL::InteractionKind::touch, t, t, "price-level-bar-v1;bar=8");
    assert(ThrowsInvalidArgument([&] {
        (void)adapter.Adapt(std::span<const PL::Observation>{&oversizedKeySource, 1});
    }));
    auto unknown = source;
    unknown.kind = static_cast<PL::InteractionKind>(99);
    assert(ThrowsInvalidArgument([&] {
        (void)adapter.Adapt(std::span<const PL::Observation>{&unknown, 1});
    }));

    PL::CausalPriceLevelEngine detector{"eurusdrmp", Configuration()};
    (void)detector.AddCompletedBar(Bar(0, 10, 5, 7));
    (void)detector.AddCompletedBar(Bar(1, 12, 6, 8));
    const auto update = detector.AddCompletedBar(Bar(2, 9, 4, 6));
    const auto originalUpdate = update;
    const auto observations = adapter.Adapt(update);
    assert(update.observations == originalUpdate.observations);
    assert(update.activeLevels == originalUpdate.activeLevels);
    assert(update.decisionTime == originalUpdate.decisionTime);
    assert(observations.size() == 1);
    assert(observations.front().sourceObservationId == update.observations.front().identity);

    auto nonFinite = update;
    nonFinite.activeLevels.front().anchorPrice = std::numeric_limits<double>::quiet_NaN();
    assert(ThrowsInvalidArgument([&] { (void)adapter.Adapt(nonFinite); }));
    auto invalidBounds = update;
    invalidBounds.activeLevels.front().lower = invalidBounds.activeLevels.front().upper + 1.0;
    assert(ThrowsInvalidArgument([&] { (void)adapter.Adapt(invalidBounds); }));
    assert(EA::kCurrentModelInputWidth == 116);
}

} // namespace

int main()
{
    TestExactEventMappingExclusionsAndProvenance();
    TestAvailabilityPrefixReplayAndGenericAcceptance();
    TestInvalidSourceDuplicateAndUpdateValidation();
    std::cout << "PriceLevelMarketStructureObservationAdapterTests passed\n";
}
