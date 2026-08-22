#include "EconomicEventBarAlignment.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace EA::EconomicCalendar
{
namespace
{

void ValidateBars(
    const std::vector<PriceTP>& barStarts)
{
    for (std::size_t i = 1; i < barStarts.size(); ++i)
    {
        if (!(barStarts[i - 1] < barStarts[i]))
        {
            throw std::invalid_argument(
                "economic_event_bar_times_must_be_strictly_increasing");
        }
    }
}

} // namespace


PriceTP EconomicEventTimePoint(
    const EconomicEvent& event)
{
    // PostgreSQL repository boundary already canonicalizes the event instant
    // to Unix microseconds. PriceTP intentionally has second resolution.
    //
    // All currently imported BLS event timestamps are whole-second values.
    if (event.eventTimestampUnixMicros % 1000000LL != 0)
    {
        throw std::invalid_argument(
            "economic_event_timestamp_not_whole_second");
    }

    return PriceTP{
        std::chrono::seconds{
            event.eventTimestampUnixMicros / 1000000LL}};
}


PriceTP BarInformationCutoff(
    PriceTP barStart)
{
    return barStart + kEconomicEventBarDuration;
}


EventBarAlignment AlignEventToBars(
    const EconomicEvent& event,
    const std::vector<PriceTP>& barStarts)
{
    ValidateBars(barStarts);

    EventBarAlignment result;

    if (barStarts.empty())
        return result;

    const PriceTP eventTime =
        EconomicEventTimePoint(event);

    //
    // A completed bar knows events satisfying:
    //
    //     eventTime < barStart + 15m
    //
    // Strict '<' is deliberate. An event exactly on the next bar boundary
    // belongs to the next bar, never to the previous bar.
    //
    const auto firstObservable =
        std::lower_bound(
            barStarts.begin(),
            barStarts.end(),
            eventTime,
            [&](const PriceTP& barStart, const PriceTP& value)
            {
                return BarInformationCutoff(barStart) <= value;
            });

    if (firstObservable == barStarts.end())
        return result;

    const std::size_t index =
        static_cast<std::size_t>(
            std::distance(
                barStarts.begin(),
                firstObservable));

    result.firstObservableBarIndex = index;

    const PriceTP barStart = barStarts[index];
    const PriceTP barEnd =
        BarInformationCutoff(barStart);

    if (
        barStart <= eventTime &&
        eventTime < barEnd)
    {
        result.eventContainedInBar = true;
        result.containingBarIndex = index;
    }

    return result;
}


std::optional<MostRecentEventAtBar> FindMostRecentObservableEvent(
    const std::vector<EconomicEvent>& events,
    const std::vector<PriceTP>& barStarts,
    std::size_t barIndex)
{
    ValidateBars(barStarts);

    if (barIndex >= barStarts.size())
    {
        throw std::out_of_range(
            "economic_event_bar_index_out_of_range");
    }

    if (events.empty())
        return std::nullopt;

    const PriceTP cutoff =
        BarInformationCutoff(
            barStarts[barIndex]);

    std::optional<std::size_t> bestEventIndex;
    PriceTP bestTime{};

    for (std::size_t i = 0; i < events.size(); ++i)
    {
        const PriceTP eventTime =
            EconomicEventTimePoint(events[i]);

        // Strict cutoff is the central no-look-ahead invariant.
        if (!(eventTime < cutoff))
            continue;

        if (
            !bestEventIndex ||
            bestTime < eventTime)
        {
            bestEventIndex = i;
            bestTime = eventTime;
        }
    }

    if (!bestEventIndex)
        return std::nullopt;

    const EventBarAlignment alignment =
        AlignEventToBars(
            events[*bestEventIndex],
            barStarts);

    if (
        !alignment.firstObservableBarIndex ||
        *alignment.firstObservableBarIndex > barIndex)
    {
        return std::nullopt;
    }

    const std::int64_t elapsedSeconds =
        std::chrono::duration_cast<std::chrono::seconds>(
            cutoff - bestTime)
            .count();

    MostRecentEventAtBar result;
    result.eventIndex = *bestEventIndex;
    result.observedBarsSinceEvent =
        barIndex -
        *alignment.firstObservableBarIndex;
    result.elapsedSecondsAtBarClose =
        elapsedSeconds;
    result.eventContainedInCurrentBar =
        alignment.containingBarIndex ==
        std::optional<std::size_t>{barIndex};

    return result;
}

} // namespace EA::EconomicCalendar
