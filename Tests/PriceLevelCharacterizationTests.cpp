#include "CausalPriceLevelEngine.hpp"
#include "PriceLevelCharacterization.hpp"

#include <algorithm>
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
} // namespace

int main()
{
    TestStrictPivotsMatchV1();
    TestScaleUsesPrecedingRangesOnly();
    TestPivotScaleTimeAndDeterminism();
    TestChronologicalValidation();
    std::cout << "PriceLevelCharacterizationTests passed\n";
}
