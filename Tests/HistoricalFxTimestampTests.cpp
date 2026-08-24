#include <cassert>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <string>

#include "../Headers/HistoricalFxTimestamp.hpp"

namespace
{

std::int64_t Seconds(const PriceTP& value)
{
    return std::chrono::duration_cast<std::chrono::seconds>(
        value.time_since_epoch()).count();
}

std::int64_t UtcSeconds(
    int year, int month, int day, int hour, int minute, int second)
{
    std::tm fields{};
    fields.tm_year = year - 1900;
    fields.tm_mon = month - 1;
    fields.tm_mday = day;
    fields.tm_hour = hour;
    fields.tm_min = minute;
    fields.tm_sec = second;
    return static_cast<std::int64_t>(timegm(&fields));
}

void VerifyIndependentOfHostTimezone(const char* zone)
{
    assert(setenv("TZ", zone, 1) == 0);
    tzset();
    PriceTP value;
    assert(EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
        "2010-07-22 08:30:00", value));
    assert(Seconds(value) == UtcSeconds(2010, 7, 22, 12, 30, 0));
}

} // namespace


int main()
{
    PriceTP value;

    assert(EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
        "2025-01-02 08:30:00", value));
    assert(Seconds(value) == UtcSeconds(2025, 1, 2, 13, 30, 0));

    assert(EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
        "2010-07-22 08:30:00", value));
    assert(Seconds(value) == UtcSeconds(2010, 7, 22, 12, 30, 0));

    assert(!EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
        "not-a-timestamp", value));
    assert(!EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
        "2025-02-30 08:30:00", value));
    assert(!EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
        "2025-03-09 02:30:00", value));
    assert(!EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
        "2025-11-02 01:30:00", value));
    assert(!EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
        "2025-01-02 08:30:00 trailing", value));

    VerifyIndependentOfHostTimezone("UTC");
    VerifyIndependentOfHostTimezone("Asia/Tokyo");
    VerifyIndependentOfHostTimezone("America/Chicago");
    return 0;
}
