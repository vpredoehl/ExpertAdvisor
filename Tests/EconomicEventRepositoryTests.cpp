#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pqxx/pqxx>

#include "../Sources/EconomicEventRepository.hpp"

using namespace EA::EconomicCalendar;

namespace
{

std::string EnvironmentOr(
    const char* name,
    const char* fallback)
{
    const char* value = std::getenv(name);

    return value && *value
        ? value
        : fallback;
}

const EconomicEvent& FindFamily(
    const std::vector<EconomicEvent>& events,
    const std::string& family)
{
    for (const auto& event : events)
    {
        if (event.eventFamily == family)
            return event;
    }

    assert(false && "requested event family not found");
    return events.front();
}

} // namespace


int main()
{
    const std::string connectionString =
        "host=" +
        EnvironmentOr(
            "LSTM_DB_HOST",
            "localhost") +
        " user=" +
        EnvironmentOr(
            "LSTM_DB_USER",
            "pqxx") +
        " dbname=" +
        EnvironmentOr(
            "LSTM_DB_NAME",
            "LSTM");

    pqxx::connection connection{
        connectionString};

    //
    // This test is deliberately read-only.
    //
    pqxx::read_transaction transaction{
        connection};

    assert(
        EconomicEventSchemaExists(
            transaction));

    //
    // January 2010 official BLS fixtures already persisted:
    //
    // Jan 08 08:30 ET Employment
    // Jan 12 10:00 ET JOLTS
    // Jan 15 08:30 ET CPI
    // Jan 20 08:30 ET PPI
    //
    const auto january = LoadEconomicEvents(
        transaction,
        "USD",
        "2010-01-01 00:00:00+00",
        "2010-02-01 00:00:00+00");

    assert(january.size() == 4);

    assert(
        january[0].eventFamily ==
        "EMPLOYMENT");

    assert(
        january[1].eventFamily ==
        "JOLTS");

    assert(
        january[2].eventFamily ==
        "CPI");

    assert(
        january[3].eventFamily ==
        "PPI");

    const EconomicEvent& employment =
        FindFamily(
            january,
            "EMPLOYMENT");

    const EconomicEvent& jolts =
        FindFamily(
            january,
            "JOLTS");

    const EconomicEvent& cpi =
        FindFamily(
            january,
            "CPI");

    const EconomicEvent& ppi =
        FindFamily(
            january,
            "PPI");

    assert(
        employment.eventTimestampUtc ==
        "2010-01-08T13:30:00.000000Z");

    assert(
        jolts.eventTimestampUtc ==
        "2010-01-12T15:00:00.000000Z");

    assert(
        cpi.eventTimestampUtc ==
        "2010-01-15T13:30:00.000000Z");

    assert(
        ppi.eventTimestampUtc ==
        "2010-01-20T13:30:00.000000Z");

    //
    // All four releases land exactly on the 15-minute grid.
    //
    constexpr std::int64_t fifteenMinutesMicros =
        15LL * 60LL * 1000000LL;

    for (const auto& event : january)
    {
        assert(
            event.eventTimestampUnixMicros %
                fifteenMinutesMicros ==
            0);

        assert(event.currency == "USD");
        assert(event.sourceAgency == "BLS");
        assert(event.eventImportance == 3);
        assert(
            event.historicalTimeConfidence ==
            "exact");

        assert(
            event.sourceUrl.rfind(
                "https://www.bls.gov/",
                0) ==
            0);

        assert(
            event.sourceTimezone ==
            std::optional<std::string>{
                "America/New_York"});
    }

    //
    // Verify annual Employment Situation taxonomy.
    //
    const auto march = LoadEconomicEvents(
        transaction,
        "USD",
        "2010-03-01 00:00:00+00",
        "2010-04-01 00:00:00+00");

    bool foundAnnual = false;

    for (const auto& event : march)
    {
        if (
            event.eventFamily ==
            "EMPLOYMENT_ANNUAL")
        {
            foundAnnual = true;

            assert(
                event.referencePeriod ==
                std::optional<std::string>{
                    "Annual 2009"});

            assert(
                event.eventTimestampUtc ==
                "2010-03-12T15:00:00.000000Z");
        }
    }

    assert(foundAnnual);

    //
    // Half-open interval semantics:
    // an event exactly at endUtc must not be returned.
    //
    const auto beforeCpi = LoadEconomicEvents(
        transaction,
        "USD",
        "2010-01-15 00:00:00+00",
        "2010-01-15 13:30:00+00");

    assert(beforeCpi.empty());

    const auto includingCpi = LoadEconomicEvents(
        transaction,
        "USD",
        "2010-01-15 13:30:00+00",
        "2010-01-15 13:30:01+00");

    assert(includingCpi.size() == 1);
    assert(
        includingCpi.front().eventFamily ==
        "CPI");

    //
    // Session timezone must not affect the repository result.
    //
    transaction.exec(
        "SET LOCAL TIME ZONE 'UTC';");

    const auto utcEvents =
        LoadEconomicEvents(
            transaction,
            "USD",
            "2010-01-01 00:00:00+00",
            "2010-02-01 00:00:00+00");

    transaction.exec(
        "SET LOCAL TIME ZONE 'Asia/Tokyo';");

    const auto tokyoEvents =
        LoadEconomicEvents(
            transaction,
            "USD",
            "2010-01-01 00:00:00+00",
            "2010-02-01 00:00:00+00");

    assert(
        utcEvents.size() ==
        tokyoEvents.size());

    for (
        std::size_t i = 0;
        i < utcEvents.size();
        ++i)
    {
        assert(
            utcEvents[i]
                .eventTimestampUnixMicros ==
            tokyoEvents[i]
                .eventTimestampUnixMicros);

        assert(
            utcEvents[i]
                .eventTimestampUtc ==
            tokyoEvents[i]
                .eventTimestampUtc);
    }

    //
    // Invalid currency requests fail before hitting SQL.
    //
    bool lowercaseRejected = false;

    try
    {
        (void)LoadEconomicEvents(
            transaction,
            "usd",
            "2010-01-01 00:00:00+00",
            "2010-02-01 00:00:00+00");
    }
    catch (const std::invalid_argument&)
    {
        lowercaseRejected = true;
    }

    assert(lowercaseRejected);

    std::cout
        << "ECONOMIC_EVENT_REPOSITORY_TEST_PASS,"
        << "january_events="
        << january.size()
        << ",march_events="
        << march.size()
        << '\n';

    return 0;
}
