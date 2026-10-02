#include "MarketStructureRegistry.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{
using namespace std::chrono_literals;
using EA::MarketStructure::CanonicalObservationIdentity;
using EA::MarketStructure::ConfluenceDefinition;
using EA::MarketStructure::CorrelationConstraint;
using EA::MarketStructure::CorrelationKey;
using EA::MarketStructure::DescriptorPolarity;
using EA::MarketStructure::DescriptiveConfluenceEngine;
using EA::MarketStructure::Observation;
using EA::MarketStructure::RelationKind;
using EA::MarketStructure::TemporalPredicate;

template <typename Fn>
bool ThrowsInvalidArgument(Fn&& fn)
{
    try { fn(); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

Observation ObservationFor(std::string family, std::string id,
                           DescriptorPolarity polarity,
                           std::chrono::sys_seconds observedAt,
                           std::chrono::sys_seconds availableAt,
                           std::string role = "signal",
                           double confidence = 0.5)
{
    return {std::move(family), "synthetic-detector-v1", observedAt, availableAt,
            "synthetic-provenance-v1", std::move(id),
            {"market-structure-descriptor-v1", std::move(role), polarity,
             confidence}, std::nullopt};
}

ConfluenceDefinition Definition(RelationKind relation, std::string version = "v1",
                               std::size_t cap = 4)
{
    return {"synthetic-signal-relation", std::move(version), relation,
            {"synthetic.left", "signal"}, {"synthetic.right", "signal"}, cap};
}

Observation KeyedObservationFor(std::string family, std::string id,
                                DescriptorPolarity polarity,
                                std::chrono::sys_seconds observedAt,
                                std::chrono::sys_seconds availableAt,
                                std::string keyType, std::string keyValue,
                                std::string role = "signal")
{
    Observation result = ObservationFor(std::move(family), std::move(id), polarity,
                                        observedAt, availableAt, std::move(role));
    result.correlationKey = CorrelationKey{std::move(keyType), std::move(keyValue)};
    return result;
}

ConfluenceDefinition KeyedDefinition(RelationKind relation,
                                     TemporalPredicate temporal = TemporalPredicate::none,
                                     std::size_t keyCap = 2)
{
    ConfluenceDefinition result = Definition(relation, "keyed-v1", 2);
    result.correlationConstraint = CorrelationConstraint::exact_key_equality;
    result.temporalPredicate = temporal;
    result.maxCorrelationKeys = keyCap;
    if (temporal != TemporalPredicate::none)
        result.maxCandidatesPerSelector = 1;
    return result;
}

void TestCausalAvailabilityDelayedAvailabilityAndPrefixInvariance()
{
    const auto t0 = std::chrono::sys_seconds{100s};
    const auto t1 = std::chrono::sys_seconds{101s};
    const auto t2 = std::chrono::sys_seconds{102s};
    DescriptiveConfluenceEngine engine{Definition(RelationKind::support)};
    const Observation left = ObservationFor("synthetic.left", "left-1",
        DescriptorPolarity::positive, t0, t0);
    const Observation delayedRight = ObservationFor("synthetic.right", "right-1",
        DescriptorPolarity::positive, t0, t2);
    const auto before = engine.Evaluate({left, delayedRight}, t1);
    assert(before.outputs.empty());
    const auto atAvailability = engine.Evaluate({left, delayedRight}, t2);
    assert(atAvailability.outputs.size() == 1);
    assert(atAvailability.outputs[0].availableAt == t2);

    auto future = ObservationFor("synthetic.right", "future-invalid-at-t1",
        DescriptorPolarity::positive, t2, t2 + 1s);
    future.descriptor.normalizedConfidence = std::numeric_limits<double>::quiet_NaN();
    const auto prefix = engine.Evaluate({left}, t1);
    const auto withFuture = engine.Evaluate({left, future}, t1);
    assert(prefix.outputs == withFuture.outputs);
    assert(prefix.CanonicalRepresentation() == withFuture.CanonicalRepresentation());
}

void TestPermutationDeterminismAndComponentPreservation()
{
    const auto t = std::chrono::sys_seconds{200s};
    DescriptiveConfluenceEngine engine{Definition(RelationKind::support)};
    std::vector<Observation> source{
        ObservationFor("synthetic.right", "right-b", DescriptorPolarity::negative, t, t),
        ObservationFor("synthetic.left", "left-b", DescriptorPolarity::negative, t, t),
        ObservationFor("synthetic.right", "right-a", DescriptorPolarity::positive, t, t),
        ObservationFor("synthetic.left", "left-a", DescriptorPolarity::positive, t, t),
    };
    const auto original = source;
    const auto first = engine.Evaluate(source, t);
    assert(source == original);
    assert(EA::MarketStructure::CausallyAvailableObservations(source, t).size() ==
           source.size());
    std::reverse(source.begin(), source.end());
    const auto second = engine.Evaluate(source, t);
    assert(first.outputs == second.outputs);
    assert(first.CanonicalRepresentation() == second.CanonicalRepresentation());
    std::reverse(source.begin(), source.end());
    assert(source == original);
    assert(first.outputs.size() == 2);
    assert(first.outputs[0].components.size() == 2);
    assert(CanonicalObservationIdentity(first.outputs[0].components[0]) <
           CanonicalObservationIdentity(first.outputs[0].components[1]));
}

void TestCandidateCapAndExplicitOverflow()
{
    const auto t = std::chrono::sys_seconds{300s};
    DescriptiveConfluenceEngine engine{Definition(RelationKind::support, "v1", 2)};
    const auto replay = engine.Evaluate({
        ObservationFor("synthetic.left", "z", DescriptorPolarity::positive, t, t),
        ObservationFor("synthetic.left", "a", DescriptorPolarity::positive, t, t),
        ObservationFor("synthetic.left", "m", DescriptorPolarity::positive, t, t),
        ObservationFor("synthetic.right", "right", DescriptorPolarity::positive, t, t),
    }, t);
    assert(replay.left.matchingCandidateCount == 3);
    assert(replay.left.retainedCandidateCount == 2);
    assert(replay.left.overflowCandidateCount == 1);
    assert(replay.outputs.size() == 1);
    assert(replay.outputs[0].descriptor.leftComponentCount == 2);
    assert(replay.CanonicalRepresentation().find("source_id=1:z") ==
           std::string::npos);
    assert(replay.CanonicalRepresentation().find("overflow=1") != std::string::npos);
}

void TestEmptySingleAndNoQualifyingConfluence()
{
    const auto t = std::chrono::sys_seconds{400s};
    DescriptiveConfluenceEngine support{Definition(RelationKind::support)};
    assert(support.Evaluate({}, t).outputs.empty());
    assert(support.Evaluate({ObservationFor("synthetic.left", "only",
        DescriptorPolarity::positive, t, t)}, t).outputs.empty());
    assert(support.Evaluate({
        ObservationFor("synthetic.left", "left", DescriptorPolarity::positive, t, t),
        ObservationFor("synthetic.right", "right", DescriptorPolarity::negative, t, t),
    }, t).outputs.empty());
}

void TestSupportContradictionAndReplayDemonstrations()
{
    const auto t = std::chrono::sys_seconds{500s};
    const Observation leftPositive = ObservationFor("synthetic.left", "support-left",
        DescriptorPolarity::positive, t - 1s, t - 1s);
    const Observation rightPositive = ObservationFor("synthetic.right", "support-right",
        DescriptorPolarity::positive, t, t);
    const Observation rightNegative = ObservationFor("synthetic.right", "contradiction-right",
        DescriptorPolarity::negative, t, t);

    DescriptiveConfluenceEngine support{Definition(RelationKind::support)};
    const auto supportReplay = support.Evaluate({leftPositive, rightPositive}, t);
    assert(supportReplay.outputs.size() == 1);
    assert(supportReplay.outputs[0].descriptor.relation == RelationKind::support);
    assert(supportReplay.outputs[0].availableAt == t);
    assert(supportReplay.CanonicalRepresentation().find("synthetic-provenance-v1") !=
           std::string::npos);

    DescriptiveConfluenceEngine contradiction{Definition(RelationKind::contradiction)};
    const auto contradictionReplay = contradiction.Evaluate({leftPositive, rightNegative}, t);
    assert(contradictionReplay.outputs.size() == 1);
    assert(contradictionReplay.outputs[0].descriptor.relation == RelationKind::contradiction);

    const auto noQualifyingReplay = support.Evaluate({leftPositive, rightNegative}, t);
    assert(noQualifyingReplay.outputs.empty());

    // The focused test output is the required in-memory replay demonstration:
    // identity, provenance, availability, definition, and result are all in
    // the canonical representation without detector-specific interpretation.
    std::cout << "SUPPORT_REPLAY " << supportReplay.CanonicalRepresentation() << '\n';
    std::cout << "CONTRADICTION_REPLAY "
              << contradictionReplay.CanonicalRepresentation() << '\n';
    std::cout << "NO_QUALIFYING_REPLAY "
              << noQualifyingReplay.CanonicalRepresentation() << '\n';
}

void TestInvalidDescriptorsAndDefinitionVersionIdentity()
{
    const auto t = std::chrono::sys_seconds{600s};
    DescriptiveConfluenceEngine engine{Definition(RelationKind::support)};
    auto invalid = ObservationFor("synthetic.left", "nan", DescriptorPolarity::positive, t, t);
    invalid.descriptor.normalizedConfidence = std::numeric_limits<double>::quiet_NaN();
    assert(ThrowsInvalidArgument([&] { (void)engine.Evaluate({invalid}, t); }));
    auto outOfRange = ObservationFor("synthetic.left", "out-of-range",
        DescriptorPolarity::positive, t, t);
    outOfRange.descriptor.normalizedConfidence = 1.1;
    assert(ThrowsInvalidArgument([&] { (void)engine.Evaluate({outOfRange}, t); }));
    const auto duplicate = ObservationFor("synthetic.left", "duplicate",
        DescriptorPolarity::positive, t, t);
    assert(ThrowsInvalidArgument([&] {
        (void)engine.Evaluate({duplicate, duplicate}, t);
    }));
    auto selfRelation = Definition(RelationKind::support);
    selfRelation.right = selfRelation.left;
    assert(ThrowsInvalidArgument([&] {
        DescriptiveConfluenceEngine invalidDefinition{selfRelation};
        (void)invalidDefinition;
    }));

    const std::vector<Observation> inputs{
        ObservationFor("synthetic.left", "left", DescriptorPolarity::positive, t, t),
        ObservationFor("synthetic.right", "right", DescriptorPolarity::positive, t, t),
    };
    DescriptiveConfluenceEngine v1{Definition(RelationKind::support, "v1")};
    DescriptiveConfluenceEngine v2{Definition(RelationKind::support, "v2")};
    const auto first = v1.Evaluate(inputs, t);
    const auto repeated = v1.Evaluate(inputs, t);
    const auto changed = v2.Evaluate(inputs, t);
    assert(first.CanonicalRepresentation() == repeated.CanonicalRepresentation());
    assert(first.outputs[0].outputIdentity == repeated.outputs[0].outputIdentity);
    assert(v1.definitionIdentity() != v2.definitionIdentity());
    assert(first.outputs[0].outputIdentity != changed.outputs[0].outputIdentity);

    auto mutableDefinition = Definition(RelationKind::support, "frozen-v1");
    DescriptiveConfluenceEngine frozen{mutableDefinition};
    mutableDefinition.definitionVersion = "mutated-after-construction";
    assert(frozen.definition().definitionVersion == "frozen-v1");
}

void TestTypedCorrelationKeyIdentityAndValidation()
{
    const auto t = std::chrono::sys_seconds{700s};
    const Observation legacy = ObservationFor("synthetic.left", "legacy",
        DescriptorPolarity::neutral, t, t);
    const std::string legacyIdentity = CanonicalObservationIdentity(legacy);
    assert(legacyIdentity ==
        "observation-v1;family=14:synthetic.left;detector=21:synthetic-detector-v1;"
        "observed_at=700;available_at=700;provenance=23:synthetic-provenance-v1;"
        "source_id=6:legacy;descriptor_schema=30:market-structure-descriptor-v1;"
        "role=6:signal;polarity=neutral;confidence=0.5");
    auto keyed = legacy;
    keyed.correlationKey = CorrelationKey{"synthetic.level", "level-a"};
    const std::string keyedIdentity = CanonicalObservationIdentity(keyed);
    assert(keyedIdentity.starts_with("observation-v2;"));
    assert(keyedIdentity != legacyIdentity);
    assert(CanonicalObservationIdentity(keyed) == keyedIdentity);

    auto badType = keyed;
    badType.correlationKey->type = "Synthetic level";
    assert(ThrowsInvalidArgument([&] { (void)CanonicalObservationIdentity(badType); }));
    auto badValue = keyed;
    badValue.correlationKey->value = "contains space";
    assert(ThrowsInvalidArgument([&] { (void)CanonicalObservationIdentity(badValue); }));
    auto oversized = keyed;
    oversized.correlationKey->value.assign(4097, 'x');
    assert(ThrowsInvalidArgument([&] { (void)CanonicalObservationIdentity(oversized); }));

    const auto legacyDefinition = Definition(RelationKind::support);
    assert(EA::MarketStructure::CanonicalConfluenceDefinitionIdentity(legacyDefinition) ==
        "confluence-definition-v1;id=25:synthetic-signal-relation;version=2:v1;"
        "relation=support;left_family=14:synthetic.left;left_role=6:signal;"
        "right_family=15:synthetic.right;right_role=6:signal;candidate_cap=4");
    const auto keyedDefinition = KeyedDefinition(RelationKind::co_occurrence);
    assert(EA::MarketStructure::CanonicalConfluenceDefinitionIdentity(keyedDefinition)
        .starts_with("confluence-definition-v2;"));
    assert(ThrowsInvalidArgument([&] {
        auto invalid = Definition(RelationKind::co_occurrence);
        (void)DescriptiveConfluenceEngine{invalid};
    }));
}

void TestKeyedNeutralCoOccurrenceAndExactJoin()
{
    const auto t = std::chrono::sys_seconds{800s};
    DescriptiveConfluenceEngine engine{KeyedDefinition(RelationKind::co_occurrence)};
    auto left = KeyedObservationFor("synthetic.left", "left", DescriptorPolarity::neutral,
        t, t, "synthetic.level", "a");
    auto right = KeyedObservationFor("synthetic.right", "right", DescriptorPolarity::neutral,
        t, t, "synthetic.level", "a");
    left.descriptor.normalizedConfidence.reset();
    right.descriptor.normalizedConfidence.reset();
    const auto replay = engine.Evaluate({left, right}, t);
    assert(replay.outputs.size() == 1);
    const auto& output = replay.outputs.front();
    assert(output.descriptor.relation == RelationKind::co_occurrence);
    assert(output.descriptor.leftPolarity == DescriptorPolarity::neutral);
    assert(output.descriptor.rightPolarity == DescriptorPolarity::neutral);
    assert(output.descriptor.correlationKey ==
           (CorrelationKey{"synthetic.level", "a"}));
    assert(output.descriptor.temporalPredicate == TemporalPredicate::none);
    assert(output.availableAt == t);
    assert(output.outputIdentity.starts_with("confluence-output-v2;"));
    assert(replay.CanonicalRepresentation().starts_with("confluence-replay-v2;"));
    assert(!left.descriptor.normalizedConfidence.has_value());
    assert(!right.descriptor.normalizedConfidence.has_value());
    for (const Observation& component : output.components)
        assert(!component.descriptor.normalizedConfidence.has_value());

    auto differentType = right;
    differentType.sourceObservationId = "right-different-type";
    differentType.correlationKey->type = "synthetic.other";
    assert(engine.Evaluate({left, differentType}, t).outputs.empty());
    auto differentValue = right;
    differentValue.sourceObservationId = "right-different-value";
    differentValue.correlationKey->value = "b";
    assert(engine.Evaluate({left, differentValue}, t).outputs.empty());
    auto missing = right;
    missing.sourceObservationId = "right-missing";
    missing.correlationKey.reset();
    assert(engine.Evaluate({left, missing}, t).outputs.empty());
    auto missingLeft = left;
    missingLeft.sourceObservationId = "left-missing";
    missingLeft.correlationKey.reset();
    assert(engine.Evaluate({missingLeft, missing}, t).outputs.empty());
}

void TestKeyedBoundsCapOrderingAndCausality()
{
    const auto t = std::chrono::sys_seconds{900s};
    DescriptiveConfluenceEngine engine{KeyedDefinition(RelationKind::co_occurrence,
        TemporalPredicate::none, 2)};
    const auto unrelated = KeyedObservationFor("synthetic.left", "00-unrelated",
        DescriptorPolarity::neutral, t, t, "synthetic.level", "unrelated");
    const auto nonNeutralLeft = KeyedObservationFor("synthetic.left", "left-non-neutral",
        DescriptorPolarity::positive, t, t, "synthetic.level", "0");
    const auto nonNeutralRight = KeyedObservationFor("synthetic.right", "right-non-neutral",
        DescriptorPolarity::positive, t, t, "synthetic.level", "0");
    const auto leftA = KeyedObservationFor("synthetic.left", "z-left-a",
        DescriptorPolarity::neutral, t, t, "synthetic.level", "a");
    const auto leftAFirst = KeyedObservationFor("synthetic.left", "a-left-a",
        DescriptorPolarity::neutral, t, t, "synthetic.level", "a");
    const auto rightA = KeyedObservationFor("synthetic.right", "right-a",
        DescriptorPolarity::neutral, t, t, "synthetic.level", "a");
    const auto leftB = KeyedObservationFor("synthetic.left", "left-b",
        DescriptorPolarity::neutral, t, t, "synthetic.level", "b");
    const auto rightB = KeyedObservationFor("synthetic.right", "right-b",
        DescriptorPolarity::neutral, t, t, "synthetic.level", "b");
    const auto leftC = KeyedObservationFor("synthetic.left", "left-c",
        DescriptorPolarity::neutral, t, t, "synthetic.level", "c");
    const auto rightC = KeyedObservationFor("synthetic.right", "right-c",
        DescriptorPolarity::neutral, t, t, "synthetic.level", "c");
    const auto replay = engine.Evaluate({unrelated, nonNeutralLeft, leftA, rightC,
        rightA, leftB, nonNeutralRight, rightB, leftAFirst, leftC}, t);
    assert(replay.outputs.size() == 2);
    assert(replay.outputs[0].descriptor.correlationKey->value == "a");
    assert(replay.outputs[1].descriptor.correlationKey->value == "b");
    assert(replay.outputs[0].descriptor.leftComponentCount == 2);
    assert(replay.outputs[0].components[0].sourceObservationId == "a-left-a");
    assert(replay.left.matchingCandidateCount == 6);
    assert(replay.left.retainedCandidateCount == 3);
    assert(replay.left.overflowCandidateCount == 3);

    auto future = rightA;
    future.sourceObservationId = "future-malformed";
    future.availableAt = t + 1s;
    future.descriptor.normalizedConfidence = std::numeric_limits<double>::quiet_NaN();
    const auto prefix = engine.Evaluate({leftA, rightA}, t);
    const auto withFuture = engine.Evaluate({leftA, rightA, future}, t);
    assert(prefix.CanonicalRepresentation() == withFuture.CanonicalRepresentation());
}

void TestKeyedTemporalPredicate()
{
    const auto t = std::chrono::sys_seconds{1'000s};
    const auto definition = KeyedDefinition(RelationKind::co_occurrence,
        TemporalPredicate::left_available_at_before_right_available_at, 1);
    DescriptiveConfluenceEngine engine{definition};
    const auto left = KeyedObservationFor("synthetic.left", "left", DescriptorPolarity::neutral,
        t - 2s, t - 2s, "synthetic.level", "a");
    const auto right = KeyedObservationFor("synthetic.right", "right", DescriptorPolarity::neutral,
        t, t, "synthetic.level", "a");
    const auto ordered = engine.Evaluate({right, left}, t);
    assert(ordered.outputs.size() == 1);
    assert(ordered.outputs[0].components.size() == 2);
    assert(ordered.outputs[0].availableAt == t);
    assert(ordered.outputs[0].descriptor.temporalPredicate ==
           TemporalPredicate::left_available_at_before_right_available_at);

    auto equal = right;
    equal.sourceObservationId = "equal";
    equal.observedAt = left.availableAt;
    equal.availableAt = left.availableAt;
    assert(engine.Evaluate({left, equal}, t).outputs.empty());
    auto wrongOrder = left;
    wrongOrder.sourceObservationId = "wrong-left";
    wrongOrder.observedAt = t;
    wrongOrder.availableAt = t;
    assert(engine.Evaluate({wrongOrder, right}, t).outputs.empty());
    assert(ThrowsInvalidArgument([&] {
        auto invalid = KeyedDefinition(RelationKind::co_occurrence,
            TemporalPredicate::left_available_at_before_right_available_at);
        invalid.maxCandidatesPerSelector = 2;
        (void)DescriptiveConfluenceEngine{invalid};
    }));
}
} // namespace

int main()
{
    TestCausalAvailabilityDelayedAvailabilityAndPrefixInvariance();
    TestPermutationDeterminismAndComponentPreservation();
    TestCandidateCapAndExplicitOverflow();
    TestEmptySingleAndNoQualifyingConfluence();
    TestSupportContradictionAndReplayDemonstrations();
    TestInvalidDescriptorsAndDefinitionVersionIdentity();
    TestTypedCorrelationKeyIdentityAndValidation();
    TestKeyedNeutralCoOccurrenceAndExactJoin();
    TestKeyedBoundsCapOrderingAndCausality();
    TestKeyedTemporalPredicate();
    std::cout << "DescriptiveConfluenceEngineTests passed\n";
}
