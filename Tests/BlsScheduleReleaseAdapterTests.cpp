#include <cassert>
#include <chrono>
#include <string>

#include "../Headers/HistoricalFxTimestamp.hpp"
#include "../Sources/BlsScheduleReleaseAdapter.hpp"

using namespace EA::EconomicCalendar;

int main()
{
    const std::string artifact =
        "bls_schedule_row_version\t1\n"
        "family\tCPI\n"
        "release_date\t2010-01-15\n"
        "release_time\t08:30:00\n"
        "title\tConsumer Price Index\n"
        "reference_period\tDecember 2009\n"
        "schedule_url\thttps://www.bls.gov/schedule/2010/home.htm\n"
        "source_event_id\tbls:cpi-december-2009\n";
    const auto first = ParseBlsScheduleReleaseArtifact(
        artifact, "https://www.bls.gov/schedule/2010/home.htm");
    const auto second = ParseBlsScheduleReleaseArtifact(
        artifact, "https://www.bls.gov/schedule/2010/home.htm");
    assert(first.sourceAgency == "BLS");
    assert(first.eventFamily == "CPI");
    assert(first.sourceEventId == "bls:cpi-december-2009");
    assert(first.referencePeriod == std::optional<std::string>{"December 2009"});
    assert(first.sourceReleaseDate == std::optional<std::string>{"2010-01-15"});
    assert(first.sourceReleaseTime == std::optional<std::string>{"08:30:00"});
    assert(first.historicalTimeConfidence == "exact");
    assert(first.sourceEventId == second.sourceEventId);

    PriceTP expected;
    assert(EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
        "2010-01-15 08:30:00", expected));
    assert(first.eventTimestampUnixMicros ==
        std::chrono::duration_cast<std::chrono::microseconds>(
            expected.time_since_epoch()).count());
    return 0;
}
