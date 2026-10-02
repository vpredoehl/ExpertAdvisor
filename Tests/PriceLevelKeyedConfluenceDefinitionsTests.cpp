#include "PriceLevelKeyedConfluenceDefinitions.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
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

struct SourceContract final
{
    PL::CausalPriceLevelEngine engine{"EURUSDRMP", Configuration()};
    std::string levelIdentity = "price-level-v1;symbol=eurusdrmp;definition=" +
        engine.configurationIdentity() + ";origin=price-level-pivot-v1;kind=high;bar=1;"
        "bar_start=1700000900;price=12";
};

std::string AlternateLevelIdentity(const SourceContract& contract, std::size_t ordinal)
{
    return "price-level-v1;symbol=eurusdrmp;definition=" +
        contract.engine.configurationIdentity() + ";origin=price-level-pivot-v1;kind=high;bar=" +
        std::to_string(ordinal) + ";bar_start=" +
        std::to_string(1'700'000'900 + static_cast<std::int64_t>(ordinal) * 900) +
        ";price=12";
}

PL::Observation Source(const SourceContract& contract, PL::InteractionKind kind,
                       std::chrono::sys_seconds observed,
                       std::chrono::sys_seconds available, std::string evidence)
{
    const std::string identity = "price-level-observation-v1;level=" +
        contract.levelIdentity + ";kind=" + std::string{
            PL::CanonicalInteractionKind(kind)} + ";observed=" +
        PL::CanonicalTimestamp(observed) + ";available=" +
        PL::CanonicalTimestamp(available) + ";source=" + evidence +
        ";before=absent;after=absent";
    return {identity, contract.levelIdentity, kind, observed, available,
            contract.engine.sourceProvenance(), std::move(evidence), std::nullopt,
            std::nullopt};
}

MS::Observation Adapt(const Bridge::ObservationAdapter& adapter,
                      const PL::Observation& source)
{
    const auto result = adapter.Adapt(std::span<const PL::Observation>{&source, 1});
    assert(result.size() == 1);
    return result.front();
}

template <typename Fn>
bool ThrowsInvalidArgument(Fn&& fn)
{
    try { fn(); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

MS::DescriptiveConfluenceEngine Engine()
{
    const auto definitions = Bridge::FrozenKeyedConfluenceDefinitions();
    assert(definitions.size() == 1);
    const auto& definition = definitions.front();
    assert(definition.definitionId ==
           "price-level-reinforced-then-retest-cooccurrence");
    assert(definition.definitionVersion == "v1");
    assert(definition.relation == MS::RelationKind::co_occurrence);
    assert((definition.left == MS::RoleSelector{"price_level", "level_reinforced"}));
    assert((definition.right == MS::RoleSelector{"price_level", "retest"}));
    assert(definition.maxCandidatesPerSelector == 1);
    assert(definition.correlationConstraint == MS::CorrelationConstraint::exact_key_equality);
    assert(definition.temporalPredicate ==
           MS::TemporalPredicate::left_available_at_before_right_available_at);
    assert(definition.maxCorrelationKeys == 64);
    return MS::DescriptiveConfluenceEngine{definition};
}

void TestExactTypedKeyAndNeutralOrderedEmission()
{
    SourceContract contract;
    Bridge::ObservationAdapter adapter{"EURUSDRMP"};
    const auto t0 = std::chrono::sys_seconds{1'700'500'000s};
    const auto t1 = t0 + 900s;
    const auto reinforced = Adapt(adapter, Source(contract,
        PL::InteractionKind::level_reinforced, t0, t0, "pivot-reinforced"));
    const auto retest = Adapt(adapter, Source(contract, PL::InteractionKind::retest,
        t1, t1, "bar-retest"));
    const std::vector<MS::Observation> input{retest, reinforced};
    const auto original = input;
    auto engine = Engine();
    const auto replay = engine.Evaluate(input, t1);
    assert(input == original);
    assert(replay.outputs.size() == 1);
    const auto& output = replay.outputs.front();
    assert(output.descriptor.relation == MS::RelationKind::co_occurrence);
    assert(output.descriptor.leftPolarity == MS::DescriptorPolarity::neutral);
    assert(output.descriptor.rightPolarity == MS::DescriptorPolarity::neutral);
    assert(output.descriptor.correlationKey == reinforced.correlationKey);
    assert(output.descriptor.correlationKey->type ==
           "price_level.level_identity.v1");
    assert(output.descriptor.correlationKey->value == contract.levelIdentity);
    assert(output.descriptor.temporalPredicate ==
           MS::TemporalPredicate::left_available_at_before_right_available_at);
    assert(output.availableAt == t1);
    assert(output.components.size() == 2);
    assert(output.outputIdentity.starts_with("confluence-output-v2;"));
    assert(replay.CanonicalRepresentation().starts_with("confluence-replay-v2;"));

    auto differentType = retest;
    differentType.correlationKey->type = "price_level.other_identity.v1";
    assert(engine.Evaluate({reinforced, differentType}, t1).outputs.empty());

    SourceContract otherContract;
    otherContract.levelIdentity = AlternateLevelIdentity(otherContract, 2);
    const auto otherLevelRetest = Adapt(adapter, Source(otherContract,
        PL::InteractionKind::retest, t1, t1, "other-level-retest"));
    assert(engine.Evaluate({reinforced, otherLevelRetest}, t1).outputs.empty());

    auto missingKey = retest;
    missingKey.correlationKey.reset();
    assert(engine.Evaluate({reinforced, missingKey}, t1).outputs.empty());
}

void TestStrictTemporalCausalityAndDeterminism()
{
    SourceContract contract;
    Bridge::ObservationAdapter adapter{"eurusdrmp"};
    const auto t0 = std::chrono::sys_seconds{1'700'600'000s};
    const auto t1 = t0 + 900s;
    const auto reinforced = Adapt(adapter, Source(contract,
        PL::InteractionKind::level_reinforced, t0, t0, "reinforced"));
    const auto retest = Adapt(adapter, Source(contract, PL::InteractionKind::retest,
        t1, t1, "retest"));
    auto engine = Engine();

    const auto baseline = engine.Evaluate({reinforced, retest}, t1);
    const auto permuted = engine.Evaluate({retest, reinforced}, t1);
    assert(baseline.CanonicalRepresentation() == permuted.CanonicalRepresentation());

    const auto equalRetest = Adapt(adapter, Source(contract,
        PL::InteractionKind::retest, t0, t0, "equal-retest"));
    assert(engine.Evaluate({reinforced, equalRetest}, t1).outputs.empty());
    const auto laterReinforced = Adapt(adapter, Source(contract,
        PL::InteractionKind::level_reinforced, t1, t1, "later-reinforced"));
    assert(engine.Evaluate({laterReinforced, equalRetest}, t1).outputs.empty());

    auto futureMalformed = retest;
    futureMalformed.sourceObservationId = "future-malformed";
    futureMalformed.availableAt = t1 + 900s;
    futureMalformed.correlationKey->type = "Malformed key";
    const auto prefix = engine.Evaluate({reinforced, retest}, t1);
    const auto withFuture = engine.Evaluate({reinforced, retest, futureMalformed}, t1);
    assert(prefix.CanonicalRepresentation() == withFuture.CanonicalRepresentation());

    auto malformedKey = retest;
    malformedKey.correlationKey->type = "Malformed key";
    assert(ThrowsInvalidArgument([&] {
        (void)engine.Evaluate({reinforced, malformedKey}, t1);
    }));
    assert(ThrowsInvalidArgument([&] {
        (void)engine.Evaluate({reinforced, reinforced, retest}, t1);
    }));
}

void TestKeyQualificationAndDeterministicCap()
{
    Bridge::ObservationAdapter adapter{"eurusdrmp"};
    const auto t0 = std::chrono::sys_seconds{1'700'700'000s};
    const auto t1 = t0 + 900s;
    auto engine = Engine();
    std::vector<MS::Observation> saturated;
    for (std::size_t index = 0; index < 80; ++index)
    {
        SourceContract unrelated;
        unrelated.levelIdentity = AlternateLevelIdentity(unrelated, index + 100);
        saturated.push_back(Adapt(adapter, Source(unrelated,
            PL::InteractionKind::level_reinforced, t0, t0,
            "unqualified-reinforced-" + std::to_string(index))));
    }
    SourceContract valid;
    const auto reinforced = Adapt(adapter, Source(valid,
        PL::InteractionKind::level_reinforced, t0, t0, "valid-reinforced"));
    const auto retest = Adapt(adapter, Source(valid, PL::InteractionKind::retest,
        t1, t1, "valid-retest"));
    saturated.push_back(reinforced);
    saturated.push_back(retest);
    const auto qualified = engine.Evaluate(saturated, t1);
    assert(qualified.outputs.size() == 1);
    assert(qualified.outputs.front().descriptor.correlationKey ==
           reinforced.correlationKey);

    std::vector<MS::Observation> manyKeys;
    for (std::size_t index = 0; index < 65; ++index)
    {
        SourceContract contract;
        contract.levelIdentity = AlternateLevelIdentity(contract, index + 1'000);
        manyKeys.push_back(Adapt(adapter, Source(contract,
            PL::InteractionKind::level_reinforced, t0, t0,
            "capped-reinforced-" + std::to_string(index))));
        manyKeys.push_back(Adapt(adapter, Source(contract, PL::InteractionKind::retest,
            t1, t1, "capped-retest-" + std::to_string(index))));
    }
    const auto first = engine.Evaluate(manyKeys, t1);
    std::reverse(manyKeys.begin(), manyKeys.end());
    const auto second = engine.Evaluate(manyKeys, t1);
    assert(first.outputs.size() == 64);
    assert(first.CanonicalRepresentation() == second.CanonicalRepresentation());
}

} // namespace

int main()
{
    TestExactTypedKeyAndNeutralOrderedEmission();
    TestStrictTemporalCausalityAndDeterminism();
    TestKeyQualificationAndDeterministicCap();
    std::cout << "PriceLevelKeyedConfluenceDefinitionsTests passed\n";
}
