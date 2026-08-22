#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

#include "../Sources/EconomicEventBarAlignment.hpp"

using namespace EA::EconomicCalendar;

namespace
{

PriceTP At(std::int64_t seconds)
{
    return PriceTP{
        std::chrono::seconds{seconds}};
}

EconomicEvent EventAt(
    std::int64_t seconds)
{
    EconomicEvent event;
    event.currency = "USD";
    event.eventFamily = "TEST";
    event.eventTimestampUnixMicros =
        seconds * 1000000LL;
    return event;
}

} // namespace


int main()
{
    //
    // Use an arbitrary absolute epoch base. Alignment depends only on
    // canonical absolute PriceTP values, not civil timezone formatting.
    //
    constexpr std::int64_t base = 1'000'000'000;

    const std::vector<PriceTP> bars{
        At(base + 0),       // 08:15 conceptually
        At(base + 900),     // 08:30
        At(base + 1800),    // 08:45
        At(base + 2700),    // 09:00
    };

    //
    // Event exactly at 08:30 belongs to the 08:30 bar.
    //
    {
        const auto alignment =
            AlignEventToBars(
                EventAt(base + 900),
                bars);

        assert(alignment.eventContainedInBar);
        assert(
            alignment.containingBarIndex ==
            std::optional<std::size_t>{1});
        assert(
            alignment.firstObservableBarIndex ==
            std::optional<std::size_t>{1});
    }

    //
    // Event one second before 08:45 still belongs to the 08:30 bar.
    //
    {
        const auto alignment =
            AlignEventToBars(
                EventAt(base + 1799),
                bars);

        assert(alignment.eventContainedInBar);
        assert(
            alignment.containingBarIndex ==
            std::optional<std::size_t>{1});
    }

    //
    // Event exactly at 08:45 does NOT leak backward into 08:30.
    //
    {
        const auto alignment =
            AlignEventToBars(
                EventAt(base + 1800),
                bars);

        assert(alignment.eventContainedInBar);
        assert(
            alignment.containingBarIndex ==
            std::optional<std::size_t>{2});
        assert(
            alignment.firstObservableBarIndex ==
            std::optional<std::size_t>{2});
    }

    //
    // Most-recent-event lookup uses bar-close information cutoff.
    //
    {
        std::vector<EconomicEvent> events{
            EventAt(base + 900),
        };

        const auto at0830 =
            FindMostRecentObservableEvent(
                events,
                bars,
                1);

        assert(at0830);
        assert(at0830->eventIndex == 0);
        assert(
            at0830->observedBarsSinceEvent ==
            0);

        // 08:30 event -> 08:45 completed-bar cutoff = 900 seconds elapsed.
        assert(
            at0830->elapsedSecondsAtBarClose ==
            900);

        assert(
            at0830->eventContainedInCurrentBar);
    }

    //
    // An event exactly on the 08:45 boundary is invisible to the completed
    // 08:30 bar.
    //
    {
        std::vector<EconomicEvent> events{
            EventAt(base + 1800),
        };

        const auto before =
            FindMostRecentObservableEvent(
                events,
                bars,
                1);

        assert(!before);

        const auto after =
            FindMostRecentObservableEvent(
                events,
                bars,
                2);

        assert(after);
        assert(
            after->observedBarsSinceEvent ==
            0);
    }

    //
    // Weekend / market gap.
    //
    // Fri 16:45 bar, Fri 17:00 bar, then Sun 17:00 bar.
    // Event happens Saturday.
    //
    // It is not "contained" in a nonexistent bar. It becomes observable on
    // the first actual bar after the gap.
    //
    {
        const std::vector<PriceTP> gapBars{
            At(base + 0),
            At(base + 900),
            At(base + 48 * 60 * 60),
            At(base + 48 * 60 * 60 + 900),
        };

        const EconomicEvent saturday =
            EventAt(base + 24 * 60 * 60);

        const auto alignment =
            AlignEventToBars(
                saturday,
                gapBars);

        assert(!alignment.eventContainedInBar);
        assert(!alignment.containingBarIndex);

        assert(
            alignment.firstObservableBarIndex ==
            std::optional<std::size_t>{2});

        const std::vector<EconomicEvent> events{
            saturday,
        };

        const auto firstSunday =
            FindMostRecentObservableEvent(
                events,
                gapBars,
                2);

        assert(firstSunday);

        // Critical invariant:
        // weekend elapsed time does NOT become 96 synthetic 15-minute bars.
        assert(
            firstSunday->observedBarsSinceEvent ==
            0);

        assert(
            firstSunday->elapsedSecondsAtBarClose >
            24 * 60 * 60);
    }

    //
    // Observed-bar distance counts actual bars, not wall-clock / 900.
    //
    {
        const std::vector<PriceTP> gapBars{
            At(base),
            At(base + 900),
            At(base + 48 * 60 * 60),
            At(base + 48 * 60 * 60 + 900),
            At(base + 48 * 60 * 60 + 1800),
        };

        const std::vector<EconomicEvent> events{
            EventAt(base + 24 * 60 * 60),
        };

        const auto thirdAvailable =
            FindMostRecentObservableEvent(
                events,
                gapBars,
                4);

        assert(thirdAvailable);

        assert(
            thirdAvailable
                ->observedBarsSinceEvent ==
            2);
    }

    //
    // Most recent causal event wins; future event cannot leak backward.
    //
    {
        std::vector<EconomicEvent> events{
            EventAt(base + 900),
            EventAt(base + 1800),
            EventAt(base + 2700),
        };

        const auto result =
            FindMostRecentObservableEvent(
                events,
                bars,
                1);

        assert(result);
        assert(result->eventIndex == 0);
    }

    //
    // Empty history remains empty.
    //
    {
        const auto result =
            FindMostRecentObservableEvent(
                {},
                bars,
                0);

        assert(!result);
    }

    std::cout
        << "ECONOMIC_EVENT_BAR_ALIGNMENT_TEST_PASS"
        << '\n';

    return 0;
}
