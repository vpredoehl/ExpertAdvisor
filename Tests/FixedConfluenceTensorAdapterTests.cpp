#include "FixedConfluenceTensorAdapter.hpp"

#include <array>
#include <cassert>
#include <chrono>
#include <stdexcept>

namespace
{
using namespace std::chrono_literals;
namespace MS = EA::MarketStructure;
namespace Production = EA::MarketStructure::Production;

EA::TG4Pulse::Pulse Pulse(std::int64_t barStart,
                          std::array<std::uint8_t, 3> bits)
{
    return {PriceTP{std::chrono::seconds{barStart}}, bits};
}

std::chrono::sys_seconds CompletedAt(const EA::TG4Pulse::Pulse& pulse)
{
    return std::chrono::sys_seconds{pulse.barStart.time_since_epoch() + 900s};
}

template <typename Fn>
bool ThrowsInvalidArgument(Fn&& fn)
{
    try { fn(); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

void TestExactProductionMappingAndDefaults()
{
    Production::TG4ProductionConfluenceBridge bridge{"eurusdrmp"};
    MS::TensorProjection::FixedConfluenceTensorAdapter adapter;

    const auto supportPulse = Pulse(1'700'000'000, {1, 1, 1});
    const auto supportDescription = bridge.Describe(
        supportPulse, CompletedAt(supportPulse));
    const auto support = adapter.Adapt(supportDescription, CompletedAt(supportPulse));
    assert((support == std::array<float, 2>{{1.0f, 0.0f}}));
    assert(supportDescription.outputs.size() == 1);
    assert(supportDescription.outputs[0].definitionId ==
           "tg4-structural-fibonacci-retracement-support");

    const auto contradictionPulse = Pulse(1'700'000'900, {1, 1, 0});
    const auto contradictionDescription = bridge.Describe(
        contradictionPulse, CompletedAt(contradictionPulse));
    const auto contradiction = adapter.Adapt(
        contradictionDescription, CompletedAt(contradictionPulse));
    assert((contradiction == std::array<float, 2>{{0.0f, 1.0f}}));
    assert(contradictionDescription.outputs.size() == 1);
    assert(contradictionDescription.outputs[0].definitionId ==
           "tg4-structural-fibonacci-retracement-contradiction");

    for (const auto bits : {std::array<std::uint8_t, 3>{{0, 0, 0}},
                            std::array<std::uint8_t, 3>{{1, 0, 0}}})
    {
        const auto pulse = Pulse(1'700'001'800, bits);
        const auto description = bridge.Describe(pulse, CompletedAt(pulse));
        assert((adapter.Adapt(description, CompletedAt(pulse)) ==
                std::array<float, 2>{{0.0f, 0.0f}}));
    }
}

void TestAvailabilityPrefixAndValidation()
{
    Production::TG4ProductionConfluenceBridge bridge{"eurusdrmp"};
    MS::TensorProjection::FixedConfluenceTensorAdapter adapter;
    const auto pulse = Pulse(1'700'002'700, {1, 1, 1});
    const auto completed = CompletedAt(pulse);

    // The completed bar is the first eligible decision time. No later input
    // can backfill the earlier decision prefix.
    const auto before = bridge.Describe(pulse, completed - 1s);
    assert(before.outputs.empty());
    assert((adapter.Adapt(before, completed - 1s) ==
            std::array<float, 2>{{0.0f, 0.0f}}));
    const auto at = bridge.Describe(pulse, completed);
    const auto first = adapter.Adapt(at, completed);
    const auto repeated = adapter.Adapt(at, completed);
    assert(first == repeated);
    assert(at.outputs.front().availableAt == completed);
    for (const auto& component : at.outputs.front().components)
        assert(component.availableAt <= at.outputs.front().availableAt);

    const auto sourceBefore = at.sourceObservations;
    (void)adapter.Adapt(at, completed);
    assert(at.sourceObservations == sourceBefore);

    auto delayed = at;
    delayed.outputs.front().availableAt = completed - 1s;
    assert(ThrowsInvalidArgument([&] { (void)adapter.Adapt(delayed, completed); }));
    auto future = at;
    future.outputs.front().availableAt = completed + 1s;
    assert(ThrowsInvalidArgument([&] { (void)adapter.Adapt(future, completed); }));
    auto duplicate = at;
    duplicate.outputs.push_back(duplicate.outputs.front());
    assert(ThrowsInvalidArgument([&] { (void)adapter.Adapt(duplicate, completed); }));
}

} // namespace

int main()
{
    TestExactProductionMappingAndDefaults();
    TestAvailabilityPrefixAndValidation();
}
