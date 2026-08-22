#pragma once

#include "EconomicEventRepository.hpp"
#include "../Headers/PricePoint.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace EA::EconomicCalendar
{

constexpr std::chrono::seconds kEconomicEventBarDuration{15 * 60};

struct EventBarAlignment
{
    // Event occurs inside [barStart, barEnd).
    bool eventContainedInBar = false;

    // First bar whose completed information set may know about the event.
    //
    // This is also the containing bar when the event occurs during market
    // hours. For an event occurring during a weekend/market gap, this is the
    // first available bar after the event.
    std::optional<std::size_t> firstObservableBarIndex;

    // Present only if the event occurs inside an actual observed bar.
    std::optional<std::size_t> containingBarIndex;
};

struct MostRecentEventAtBar
{
    std::size_t eventIndex = 0;

    // Number of observed bars elapsed since the event first became observable.
    //
    // 0 means the event first becomes observable on this bar.
    std::size_t observedBarsSinceEvent = 0;

    // Wall-clock seconds between the event instant and this bar's information
    // cutoff (barStart + 15 minutes).
    std::int64_t elapsedSecondsAtBarClose = 0;

    bool eventContainedInCurrentBar = false;
};

PriceTP EconomicEventTimePoint(
    const EconomicEvent& event);

PriceTP BarInformationCutoff(
    PriceTP barStart);

EventBarAlignment AlignEventToBars(
    const EconomicEvent& event,
    const std::vector<PriceTP>& barStarts);

std::optional<MostRecentEventAtBar> FindMostRecentObservableEvent(
    const std::vector<EconomicEvent>& events,
    const std::vector<PriceTP>& barStarts,
    std::size_t barIndex);

} // namespace EA::EconomicCalendar
