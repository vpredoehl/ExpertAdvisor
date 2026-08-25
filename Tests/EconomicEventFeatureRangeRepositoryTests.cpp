#include <cassert>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <pqxx/pqxx>

#include "EconomicEventFeatures.hpp"
#include "EconomicEventRepository.hpp"

using namespace EA::EconomicCalendar;

namespace
{

std::string EnvironmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

PriceTP At(std::int64_t seconds)
{
    return PriceTP{std::chrono::seconds{seconds}};
}

bool Near(float actual, double expected)
{
    return std::abs(static_cast<double>(actual) - expected) <= 1.0e-6;
}

} // namespace

int main()
{
    pqxx::connection connection{
        "host=" + EnvironmentOr("LSTM_DB_HOST", "127.0.0.1") +
        " user=" + EnvironmentOr("LSTM_DB_USER", "pqxx") +
        " dbname=" + EnvironmentOr("LSTM_DB_NAME", "invalid")};

    {
        pqxx::work write{connection};
        write.exec(
            "INSERT INTO economic_event "
            "(currency,event_family,event_timestamp_utc,source_agency,"
            "source_event_id,source_url,event_importance,"
            "historical_time_confidence) VALUES "
            "('USD','CPI','2023-11-14 19:00:00+00','BLS','old-cpi',"
            "'https://example.test/old-cpi',3,'exact'),"
            "('USD','CPI','2023-11-14 21:13:20+00','BLS','seed-cpi',"
            "'https://example.test/seed-cpi',3,'exact'),"
            "('USD','EMPLOYMENT','2023-11-14 20:43:20+00','BLS',"
            "'seed-employment','https://example.test/seed-employment',3,'exact'),"
            "('USD','GDP','2023-11-14 22:18:20+00','BEA','range-gdp',"
            "'https://example.test/range-gdp',3,'exact'),"
            "('USD','FOMC','2023-11-14 23:30:00+00','FEDERAL_RESERVE',"
            "'after-range','https://example.test/after-range',3,'exact');");
        write.commit();
    }

    pqxx::read_transaction read{connection};
    const auto events = LoadEconomicEventsForFeatureRange(
        read, "USD", "2023-11-14 22:13:20+00",
        "2023-11-14 22:43:20+00");

    // The old CPI row is superseded within its canonical stream. The loader
    // returns exactly the two prior stream seeds and the one in-range row.
    assert(events.size() == 3);
    assert(events[0].sourceEventId == std::optional<std::string>{"seed-employment"});
    assert(events[1].sourceEventId == std::optional<std::string>{"seed-cpi"});
    assert(events[2].sourceEventId == std::optional<std::string>{"range-gdp"});

    constexpr std::int64_t firstBarStart = 1'700'000'000;
    EconomicEventFeatureEngine engine{events};
    const auto first = engine.AdvanceCompletedBar(At(firstBarStart));
    assert(first.inflationEvent == 0.0F);
    assert(first.employmentEvent == 0.0F);
    assert(first.growthEvent == 1.0F);
    assert(Near(first.inflationRecencyDecay,
                std::exp(-4500.0 / 86400.0)));
    assert(Near(first.employmentRecencyDecay,
                std::exp(-6300.0 / 86400.0)));

    return 0;
}
