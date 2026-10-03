#include "MarketStructureProductionObservationAdapter.hpp"
#include "ModelInputContract.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace
{
using namespace std::chrono_literals;
namespace MS = EA::MarketStructure;
namespace Production = EA::MarketStructure::Production;

EA::TG4Pulse::Pulse Pulse(std::int64_t start,
                          std::array<std::uint8_t, 3> bits)
{
    return {PriceTP{std::chrono::seconds{start}}, bits};
}

void TestExactMappingIdentityAndAvailability()
{
    Production::TG4PulseObservationAdapter adapter{"EURUSDRMP"};
    const auto pulse = Pulse(1'700'100'000, {1, 1, 1});
    const auto originalPulse = pulse;
    const auto observations = adapter.Adapt(pulse);
    assert(pulse == originalPulse);
    assert(observations.size() == 3);
    const auto completed = std::chrono::sys_seconds{pulse.barStart.time_since_epoch() +
                                                     900s};
    for (const MS::Observation& observation : observations)
    {
        assert(observation.familyId == "tg_structure");
        assert(observation.detectorVersion == "tg4-production-pulse-v1");
        assert(observation.observedAt == completed);
        assert(observation.availableAt == completed);
        assert(observation.sourceProvenance == adapter.sourceProvenance());
        assert(observation.sourceProvenance.find("symbol=eurusdrmp") !=
               std::string::npos);
        assert(observation.sourceProvenance.find("configuration_hash=fnv1a64:") !=
               std::string::npos);
        assert(!observation.descriptor.normalizedConfidence.has_value());
        assert(observation.descriptor.schemaVersion ==
               "tg4-production-pulse-observation-v1");
    }
    assert(observations[0].descriptor.role == "inner_break");
    assert(observations[0].descriptor.polarity == MS::DescriptorPolarity::positive);
    assert(observations[1].descriptor.role == "structural_eligibility");
    assert(observations[1].descriptor.polarity == MS::DescriptorPolarity::positive);
    assert(observations[2].descriptor.role == "fibonacci_retracement_relation");
    assert(observations[2].descriptor.polarity == MS::DescriptorPolarity::positive);
    assert(observations[0].sourceObservationId.find("bar_start=1700100000") !=
           std::string::npos);

    const auto ineligible = adapter.Adapt(Pulse(1'700'100'900, {1, 0, 0}));
    assert(ineligible.size() == 2);
    assert(ineligible[1].descriptor.role == "structural_eligibility");
    assert(ineligible[1].descriptor.polarity == MS::DescriptorPolarity::negative);

    const auto noRelation = adapter.Adapt(Pulse(1'700'101'800, {1, 1, 0}));
    assert(noRelation.size() == 3);
    assert(noRelation[2].descriptor.polarity == MS::DescriptorPolarity::negative);
    assert(adapter.Adapt(Pulse(1'700'102'700, {0, 0, 0})).empty());

    const auto definitions = Production::FrozenConfluenceDefinitions();
    assert(definitions[0].definitionId ==
           "tg4-structural-fibonacci-retracement-support");
    assert(definitions[0].definitionVersion == "v1");
    assert(definitions[0].relation == MS::RelationKind::support);
    assert(definitions[1].definitionId ==
           "tg4-structural-fibonacci-retracement-contradiction");
    assert(definitions[1].definitionVersion == "v1");
    assert(definitions[1].relation == MS::RelationKind::contradiction);
    for (const MS::ConfluenceDefinition& definition : definitions)
    {
        assert(definition.left.familyId == "tg_structure");
        assert(definition.left.role == "structural_eligibility");
        assert(definition.right.familyId == "tg_structure");
        assert(definition.right.role == "fibonacci_retracement_relation");
        assert(definition.maxCandidatesPerSelector == 1);
    }
}

void TestDelayedAvailabilityAndPrefixInvariance()
{
    Production::TG4ProductionConfluenceBridge bridge{"eurusdrmp"};
    const auto pulse = Pulse(1'700'200'000, {1, 1, 1});
    const auto completed = std::chrono::sys_seconds{pulse.barStart.time_since_epoch() +
                                                     900s};
    const auto before = bridge.Describe(pulse, completed - 1s);
    assert(before.outputs.empty());
    for (const MS::ConfluenceReplay& replay : before.replays)
        assert(replay.outputs.empty());
    const auto at = bridge.Describe(pulse, completed);
    assert(at.outputs.size() == 1);
    assert(at.outputs[0].availableAt == completed);
    assert(at.outputs[0].availableAt >= at.outputs[0].components[0].availableAt);
    assert(at.outputs[0].availableAt >= at.outputs[0].components[1].availableAt);

    Production::TG4PulseObservationAdapter adapter{"eurusdrmp"};
    const auto source = adapter.Adapt(pulse);
    const auto original = source;
    const auto definitions = Production::FrozenConfluenceDefinitions();
    MS::DescriptiveConfluenceEngine engine{definitions[0]};
    const auto prefix = engine.Evaluate(source, completed - 1s);
    auto future = source.front();
    future.availableAt = completed + 1s;
    future.descriptor.normalizedConfidence =
        std::numeric_limits<double>::quiet_NaN();
    const auto withFuture = engine.Evaluate({source[0], source[1], source[2], future},
                                            completed - 1s);
    assert(prefix.CanonicalRepresentation() == withFuture.CanonicalRepresentation());
    assert(source == original);
}

void TestProductionSupportContradictionAndNoMatch()
{
    Production::TG4ProductionConfluenceBridge bridge{"eurusdrmp"};
    const auto support = bridge.Describe(Pulse(1'700'300'000, {1, 1, 1}),
        std::chrono::sys_seconds{1'700'300'900s});
    assert(support.outputs.size() == 1);
    assert(support.outputs[0].definitionId ==
           "tg4-structural-fibonacci-retracement-support");
    assert(support.outputs[0].descriptor.relation == MS::RelationKind::support);
    assert(support.outputs[0].descriptor.leftPolarity ==
           MS::DescriptorPolarity::positive);
    assert(support.outputs[0].descriptor.rightPolarity ==
           MS::DescriptorPolarity::positive);

    const auto contradiction = bridge.Describe(Pulse(1'700'301'000, {1, 1, 0}),
        std::chrono::sys_seconds{1'700'301'900s});
    assert(contradiction.outputs.size() == 1);
    assert(contradiction.outputs[0].definitionId ==
           "tg4-structural-fibonacci-retracement-contradiction");
    assert(contradiction.outputs[0].descriptor.relation ==
           MS::RelationKind::contradiction);
    assert(contradiction.outputs[0].descriptor.rightPolarity ==
           MS::DescriptorPolarity::negative);

    const auto ineligible = bridge.Describe(Pulse(1'700'302'000, {1, 0, 0}),
        std::chrono::sys_seconds{1'700'302'900s});
    assert(ineligible.outputs.empty());
    const auto absent = bridge.Describe(Pulse(1'700'303'000, {0, 0, 0}),
        std::chrono::sys_seconds{1'700'303'900s});
    assert(absent.outputs.empty());
}

void TestDeterminismCapInvalidInputsAndSourcePreservation()
{
    Production::TG4PulseObservationAdapter adapter{"eurusdrmp"};
    const auto first = adapter.Adapt(Pulse(1'700'400'000, {1, 1, 1}));
    const auto second = adapter.Adapt(Pulse(1'700'400'900, {1, 1, 1}));
    const auto sourceFirst = first;
    const auto sourceSecond = second;
    const auto definitions = Production::FrozenConfluenceDefinitions();
    MS::DescriptiveConfluenceEngine engine{definitions[0]};
    const auto decision = std::chrono::sys_seconds{1'700'401'800s};
    std::vector<MS::Observation> inputs = first;
    inputs.insert(inputs.end(), second.begin(), second.end());
    const auto baseline = engine.Evaluate(inputs, decision);
    assert(baseline.left.matchingCandidateCount == 2);
    assert(baseline.left.retainedCandidateCount == 1);
    assert(baseline.left.overflowCandidateCount == 1);
    assert(baseline.right.matchingCandidateCount == 2);
    assert(baseline.right.overflowCandidateCount == 1);
    std::reverse(inputs.begin(), inputs.end());
    const auto permuted = engine.Evaluate(inputs, decision);
    assert(baseline.CanonicalRepresentation() == permuted.CanonicalRepresentation());
    const auto repeated = engine.Evaluate(inputs, decision);
    assert(permuted.CanonicalRepresentation() == repeated.CanonicalRepresentation());
    assert(first == sourceFirst);
    assert(second == sourceSecond);

    auto invalidAvailable = first.front();
    invalidAvailable.descriptor.normalizedConfidence =
        std::numeric_limits<double>::quiet_NaN();
    bool threw = false;
    try { (void)engine.Evaluate({invalidAvailable}, decision); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    threw = false;
    try { (void)adapter.Adapt(Pulse(1'700'402'000, {1, 0, 1})); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    threw = false;
    try { (void)adapter.Adapt(Pulse(1'700'402'900, {2, 0, 0})); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

Feature Bar(std::size_t index, double open, double high, double low, double close)
{
    return {static_cast<float>(open), static_cast<float>(close),
            static_cast<float>(high), static_cast<float>(low),
            PriceTP{std::chrono::seconds{
                1'700'500'000 + static_cast<std::int64_t>(index) * 900}},
            0.0f};
}

void TestActualProductionAdapterPath()
{
    std::mt19937_64 generator{0x54473450554c5345ULL};
    std::uniform_int_distribution<int> direction(-6, 6);
    std::uniform_int_distribution<int> wick(1, 8);
    EA::TG4Pulse::ProductionStreamingAdapter detector{"eurusdrmp"};
    Production::TG4ProductionConfluenceBridge bridge{"eurusdrmp"};
    bool sawSupport = false;
    bool sawContradiction = false;
    double close = 1.2000;
    for (std::size_t index = 0; index != 5'000; ++index)
    {
        const double open = close;
        close += static_cast<double>(direction(generator)) * 0.0001;
        const double high = std::max(open, close) +
            static_cast<double>(wick(generator)) * 0.0001;
        const double low = std::min(open, close) -
            static_cast<double>(wick(generator)) * 0.0001;
        const Feature bar = Bar(index, open, high, low, close);
        const EA::TG4Pulse::Pulse pulse = detector.AddCompletedCanonicalBar(bar);
        const auto description = bridge.Describe(
            pulse, std::chrono::sys_seconds{pulse.barStart.time_since_epoch() + 900s});
        for (const MS::ConfluenceObservation& output : description.outputs)
        {
            if (output.descriptor.relation == MS::RelationKind::support)
                sawSupport = true;
            if (output.descriptor.relation == MS::RelationKind::contradiction)
                sawContradiction = true;
        }
    }
    assert(sawSupport);
    assert(sawContradiction);
}

void TestFixedTensorRegistrationAndLayoutChange()
{
    assert(MS::FindFamily("confluence") != nullptr);
    assert(MS::ResolvePrefix("confluence", 8).empty());
    assert(MS::ResolvePrefix("confluence", 9).empty());
    assert(MS::ResolvePrefix("confluence", 10).empty());
    assert(MS::ResolvePrefix("confluence", 11).size() == 2);
    assert(EA::kCausalFibonacciStructuralModelInputWidth == 103);
    assert(EA::kCausalPocketRecentObservationModelInputWidth == 114);
    assert(EA::kCurrentModelInputWidth == 127);
}

} // namespace

int main()
{
    TestExactMappingIdentityAndAvailability();
    TestDelayedAvailabilityAndPrefixInvariance();
    TestProductionSupportContradictionAndNoMatch();
    TestDeterminismCapInvalidInputsAndSourcePreservation();
    TestActualProductionAdapterPath();
    TestFixedTensorRegistrationAndLayoutChange();
    std::cout << "MarketStructureProductionObservationAdapterTests passed\n";
}
