#include "MarketDataCore.hpp"
#include "TG4ProductionStreamingPulseAdapter.hpp"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <pqxx/pqxx>
#include <string>
#include <vector>

namespace
{

std::string EnvironmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : fallback;
}

std::string ForexConnectionString()
{
    const char* override = std::getenv("TG4_PULSE_FOREX_CONNECTION");
    if (override != nullptr && *override != '\0') return override;
    return "hostaddr=" + EnvironmentOr("FOREX_DB_HOST", "127.0.0.1") +
        " gssencmode=disable user=" + EnvironmentOr("FOREX_DB_USER", "pqxx") +
        " dbname=" + EnvironmentOr("FOREX_DB_NAME", "forex") +
        " application_name=tg4_production_pulse_read_only_replay";
}

std::vector<Feature> LoadCanonicalBars(const std::string& cursorName)
{
    pqxx::connection connection{ForexConnectionString()};
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ ONLY;");
    const EA::CanonicalMarketData::AbsoluteHalfOpenRange range{
        PriceTP{std::chrono::seconds{1'741'150'800}},
        PriceTP{std::chrono::seconds{1'741'237'200}}};
    const EA::MarketData::CandlestickSlice slice =
        EA::MarketData::LoadCanonicalHalfOpenCandlesticks(
            transaction, "eurusdrmp", range, cursorName);
    for (std::size_t index = 0; index < slice.rows.size(); ++index)
    {
        assert(range.Contains(slice.rows[index].time));
        if (index != 0) assert(slice.rows[index - 1].time < slice.rows[index].time);
    }
    transaction.commit();
    return slice.rows;
}

void TestBoundedCanonicalReplayIsDeterministic()
{
    const auto firstBars = LoadCanonicalBars("eurusdrmp_tg4_pulse_replay_first");
    const auto secondBars = LoadCanonicalBars("eurusdrmp_tg4_pulse_replay_second");
    assert(!firstBars.empty());
    assert(firstBars.size() == 96);
    assert(firstBars.size() == secondBars.size());
    for (std::size_t index = 0; index < firstBars.size(); ++index)
        assert(firstBars[index].time == secondBars[index].time &&
               firstBars[index].open == secondBars[index].open &&
               firstBars[index].high == secondBars[index].high &&
               firstBars[index].low == secondBars[index].low &&
               firstBars[index].close == secondBars[index].close &&
               firstBars[index].tickVolume == secondBars[index].tickVolume);

    const auto firstPulses = EA::TG4Pulse::ReplayCanonicalCompletedBars(
        "eurusdrmp", firstBars);
    const auto secondPulses = EA::TG4Pulse::ReplayCanonicalCompletedBars(
        "eurusdrmp", secondBars);
    assert(firstPulses == secondPulses);
    assert(firstPulses.size() == firstBars.size());
    for (std::size_t index = 0; index < firstPulses.size(); ++index)
        assert(firstPulses[index].barStart == firstBars[index].time);
}

} // namespace

int main()
{
    TestBoundedCanonicalReplayIsDeterministic();
    std::cout << "TG4ProductionStreamingCanonicalReplayTests passed"
              << " (96 canonical bars)\n";
}
