#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pqxx/pqxx>

#include "../Headers/HistoricalFxTimestamp.hpp"
#include "../Sources/EconomicEventBarAlignment.hpp"
#include "../Sources/EconomicEventRepository.hpp"

using namespace EA::EconomicCalendar;

namespace
{

std::string EnvironmentOr(
    const char* name,
    const char* fallback)
{
    const char* value =
        std::getenv(name);

    return
        value && *value
            ? value
            : fallback;
}


std::string Trim(
    std::string value)
{
    const auto first =
        value.find_first_not_of(
            " \t\r\n");

    if (
        first ==
        std::string::npos)
    {
        return {};
    }

    const auto last =
        value.find_last_not_of(
            " \t\r\n");

    return value.substr(
        first,
        last - first + 1);
}


std::optional<std::string>
ExtractTimestampColumn(
    const std::string& line)
{
    const std::size_t pipe =
        line.find('|');

    if (
        pipe ==
        std::string::npos)
    {
        return std::nullopt;
    }

    const std::string value =
        Trim(
            line.substr(
                0,
                pipe));

    // Ignore psql headers, separators and footer.
    if (
        value.size() != 19 ||
        value[4] != '-' ||
        value[7] != '-' ||
        value[10] != ' ' ||
        value[13] != ':' ||
        value[16] != ':')
    {
        return std::nullopt;
    }

    return value;
}


std::vector<PriceTP>
LoadRealBarStarts(
    const std::string& path,
    const std::string& fromCivil,
    const std::string& throughCivil)
{
    std::ifstream input{
        path};

    if (!input)
    {
        throw std::runtime_error(
            "cannot_open_input15m:"
            + path);
    }

    std::vector<PriceTP> bars;

    std::string line;

    while (
        std::getline(
            input,
            line))
    {
        const auto timestamp =
            ExtractTimestampColumn(
                line);

        if (!timestamp)
            continue;

        if (*timestamp < fromCivil)
            continue;

        if (*timestamp > throughCivil)
            break;

        PriceTP time{};

        if (
            !EA::HistoricalFxTimestamp::
                ParseNewYorkCivilTimestamp(
                    *timestamp,
                    time))
        {
            throw std::runtime_error(
                "cannot_parse_input15m_timestamp:"
                + *timestamp);
        }

        bars.push_back(
            time);
    }

    if (bars.empty())
    {
        throw std::runtime_error(
            "no_input15m_bars_in_requested_range");
    }

    return bars;
}


std::size_t FindBarIndex(
    const std::vector<PriceTP>& bars,
    const std::string& civilTimestamp)
{
    PriceTP target{};

    if (
        !EA::HistoricalFxTimestamp::
            ParseNewYorkCivilTimestamp(
                civilTimestamp,
                target))
    {
        throw std::runtime_error(
            "cannot_parse_expected_bar_timestamp");
    }

    for (
        std::size_t i = 0;
        i < bars.size();
        ++i)
    {
        if (bars[i] == target)
            return i;
    }

    throw std::runtime_error(
        "expected_bar_not_found:"
        + civilTimestamp);
}


const EconomicEvent& FindEvent(
    const std::vector<EconomicEvent>& events,
    const std::string& family,
    const std::string& referencePeriod)
{
    for (
        const auto& event :
        events)
    {
        if (
            event.eventFamily ==
                family &&
            event.referencePeriod ==
                std::optional<std::string>{
                    referencePeriod})
        {
            return event;
        }
    }

    throw std::runtime_error(
        "expected_economic_event_not_found:"
        + family
        + ":"
        + referencePeriod);
}

} // namespace


int main()
{
    const std::string inputPath =
        EnvironmentOr(
            "INPUT15M_PATH",
            "input15m.txt");

    //
    // Use genuine rows from input15m.txt around the January 2010 BLS anchors.
    //
    const auto bars =
        LoadRealBarStarts(
            inputPath,
            "2010-01-08 07:45:00",
            "2010-01-15 09:00:00");

    const std::size_t employment0830 =
        FindBarIndex(
            bars,
            "2010-01-08 08:30:00");

    const std::size_t employment0815 =
        FindBarIndex(
            bars,
            "2010-01-08 08:15:00");

    const std::size_t employment0845 =
        FindBarIndex(
            bars,
            "2010-01-08 08:45:00");

    const std::size_t cpi0830 =
        FindBarIndex(
            bars,
            "2010-01-15 08:30:00");

    //
    // These values prove that input15m itself contains real bars at the
    // relevant source-local publication timestamps.
    //
    assert(
        employment0815 <
        employment0830);

    assert(
        employment0830 <
        employment0845);

    pqxx::connection connection{
        "host="
        + EnvironmentOr(
            "LSTM_DB_HOST",
            "localhost")
        + " user="
        + EnvironmentOr(
            "LSTM_DB_USER",
            "pqxx")
        + " dbname="
        + EnvironmentOr(
            "LSTM_DB_NAME",
            "LSTM")
    };

    pqxx::read_transaction transaction{
        connection};

    assert(
        EconomicEventSchemaExists(
            transaction));

    //
    // Query real persisted BLS rows.
    //
    const auto events =
        LoadEconomicEvents(
            transaction,
            "USD",
            "2010-01-08 00:00:00+00",
            "2010-01-16 00:00:00+00");

    const EconomicEvent& employment =
        FindEvent(
            events,
            "EMPLOYMENT",
            "December 2009");

    const EconomicEvent& cpi =
        FindEvent(
            events,
            "CPI",
            "December 2009");

    //
    // Production timestamp conversion of real input15m 08:30 Eastern must
    // equal the persisted BLS UTC event instant.
    //
    assert(
        bars[employment0830] ==
        EconomicEventTimePoint(
            employment));

    assert(
        bars[cpi0830] ==
        EconomicEventTimePoint(
            cpi));

    //
    // Exact-boundary semantics:
    //
    // Employment at 08:30 belongs to the actual 08:30 input15m bar.
    //
    const auto employmentAlignment =
        AlignEventToBars(
            employment,
            bars);

    assert(
        employmentAlignment
            .eventContainedInBar);

    assert(
        employmentAlignment
            .containingBarIndex ==
        std::optional<std::size_t>{
            employment0830});

    assert(
        employmentAlignment
            .firstObservableBarIndex ==
        std::optional<std::size_t>{
            employment0830});

    //
    // The completed 08:15 bar cannot know about an 08:30 event.
    //
    const auto beforeEmployment =
        FindMostRecentObservableEvent(
            std::vector<EconomicEvent>{
                employment},
            bars,
            employment0815);

    assert(
        !beforeEmployment);

    //
    // Once the actual 08:30 bar is complete, the event is causal/observable.
    //
    const auto onEmploymentBar =
        FindMostRecentObservableEvent(
            std::vector<EconomicEvent>{
                employment},
            bars,
            employment0830);

    assert(
        onEmploymentBar);

    assert(
        onEmploymentBar
            ->observedBarsSinceEvent ==
        0);

    assert(
        onEmploymentBar
            ->elapsedSecondsAtBarClose ==
        15 * 60);

    assert(
        onEmploymentBar
            ->eventContainedInCurrentBar);

    //
    // The next actual bar has one observed bar elapsed.
    //
    const auto afterEmployment =
        FindMostRecentObservableEvent(
            std::vector<EconomicEvent>{
                employment},
            bars,
            employment0845);

    assert(
        afterEmployment);

    assert(
        afterEmployment
            ->observedBarsSinceEvent ==
        1);

    assert(
        afterEmployment
            ->elapsedSecondsAtBarClose ==
        30 * 60);

    //
    // CPI receives the same exact alignment through the same production
    // timestamp conversion path.
    //
    const auto cpiAlignment =
        AlignEventToBars(
            cpi,
            bars);

    assert(
        cpiAlignment
            .containingBarIndex ==
        std::optional<std::size_t>{
            cpi0830});

    //
    // Verify a real market gap from input15m.
    //
    // The file starts Sunday 2010-01-03 17:00, and contains later weekday
    // bars. Find a gap exceeding one 15-minute period and prove observed-bar
    // distance remains based on actual rows rather than elapsed/900.
    //
    bool foundGap = false;

    for (
        std::size_t i = 1;
        i < bars.size();
        ++i)
    {
        const auto seconds =
            std::chrono::duration_cast<
                std::chrono::seconds>(
                    bars[i] -
                    bars[i - 1])
                .count();

        if (seconds > 15 * 60)
        {
            foundGap = true;

            EconomicEvent syntheticGapEvent;
            syntheticGapEvent.currency =
                "USD";
            syntheticGapEvent.eventFamily =
                "INTEGRATION_GAP_EVENT";

            const PriceTP between =
                bars[i - 1]
                + std::chrono::seconds{
                    seconds / 2};

            syntheticGapEvent
                .eventTimestampUnixMicros =
                std::chrono::duration_cast<
                    std::chrono::microseconds>(
                        between
                            .time_since_epoch())
                    .count();

            const auto alignment =
                AlignEventToBars(
                    syntheticGapEvent,
                    bars);

            assert(
                !alignment
                    .eventContainedInBar);

            assert(
                alignment
                    .firstObservableBarIndex ==
                std::optional<std::size_t>{
                    i});

            const auto recent =
                FindMostRecentObservableEvent(
                    std::vector<EconomicEvent>{
                        syntheticGapEvent},
                    bars,
                    i);

            assert(recent);

            // First real post-gap bar means zero observed bars elapsed,
            // regardless of how much wall-clock time passed.
            assert(
                recent
                    ->observedBarsSinceEvent ==
                0);

            assert(
                recent
                    ->elapsedSecondsAtBarClose >
                15 * 60);

            break;
        }
    }

    assert(foundGap);

    //
    // Training/inference parity contract:
    //
    // There is intentionally no training-specific or inference-specific
    // alignment implementation. Both consume the same vector<PriceTP> and the
    // same EconomicEvent values through these pure functions.
    //
    const auto trainingResult =
        FindMostRecentObservableEvent(
            events,
            bars,
            cpi0830);

    const auto inferenceResult =
        FindMostRecentObservableEvent(
            events,
            bars,
            cpi0830);

    assert(
        trainingResult.has_value() ==
        inferenceResult.has_value());

    assert(trainingResult);
    assert(inferenceResult);

    assert(
        trainingResult->eventIndex ==
        inferenceResult->eventIndex);

    assert(
        trainingResult
            ->observedBarsSinceEvent ==
        inferenceResult
            ->observedBarsSinceEvent);

    assert(
        trainingResult
            ->elapsedSecondsAtBarClose ==
        inferenceResult
            ->elapsedSecondsAtBarClose);

    std::cout
        << "ECONOMIC_EVENT_REAL_BAR_ALIGNMENT_INTEGRATION_PASS"
        << ",bars="
        << bars.size()
        << ",events="
        << events.size()
        << ",employment_bar="
        << employment0830
        << ",cpi_bar="
        << cpi0830
        << '\n';

    return 0;
}
