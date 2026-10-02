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
using EA::MarketStructure::DescriptorPolarity;
using EA::MarketStructure::DescriptiveConfluenceEngine;
using EA::MarketStructure::Observation;
using EA::MarketStructure::RelationKind;

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
             confidence}};
}

ConfluenceDefinition Definition(RelationKind relation, std::string version = "v1",
                               std::size_t cap = 4)
{
    return {"synthetic-signal-relation", std::move(version), relation,
            {"synthetic.left", "signal"}, {"synthetic.right", "signal"}, cap};
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
} // namespace

int main()
{
    TestCausalAvailabilityDelayedAvailabilityAndPrefixInvariance();
    TestPermutationDeterminismAndComponentPreservation();
    TestCandidateCapAndExplicitOverflow();
    TestEmptySingleAndNoQualifyingConfluence();
    TestSupportContradictionAndReplayDemonstrations();
    TestInvalidDescriptorsAndDefinitionVersionIdentity();
    std::cout << "DescriptiveConfluenceEngineTests passed\n";
}
