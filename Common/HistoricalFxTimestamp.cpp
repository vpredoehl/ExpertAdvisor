#include "HistoricalFxTimestamp.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <locale>
#include <sstream>

namespace EA::HistoricalFxTimestamp
{
namespace
{

int DaysInMonth(
    int year,
    int month)
{
    static constexpr int days[] =
    {
        31, 28, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31
    };

    if (month != 2)
        return days[month - 1];

    const bool leap =
        (year % 4 == 0 && year % 100 != 0) ||
        (year % 400 == 0);

    return leap ? 29 : 28;
}


int WeekdayUtc(
    int year,
    int month,
    int day)
{
    std::tm tm{};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = 12;

    const std::time_t value =
        timegm(&tm);

    std::tm utc{};
    gmtime_r(
        &value,
        &utc);

    return utc.tm_wday;
}


int NthSundayOfMonth(
    int year,
    int month,
    int nth)
{
    const int firstWeekday =
        WeekdayUtc(
            year,
            month,
            1);

    const int firstSunday =
        1 +
        ((7 - firstWeekday) % 7);

    const int result =
        firstSunday +
        7 * (nth - 1);

    if (
        result >
        DaysInMonth(
            year,
            month))
    {
        return -1;
    }

    return result;
}


bool IsNewYorkDstCivilTime(
    const std::tm& civil)
{
    const int year =
        civil.tm_year + 1900;

    const int month =
        civil.tm_mon + 1;

    const int day =
        civil.tm_mday;

    const int hour =
        civil.tm_hour;

    if (
        month < 3 ||
        month > 11)
    {
        return false;
    }

    if (
        month > 3 &&
        month < 11)
    {
        return true;
    }

    if (month == 3)
    {
        const int transitionDay =
            NthSundayOfMonth(
                year,
                3,
                2);

        if (day < transitionDay)
            return false;

        if (day > transitionDay)
            return true;

        return hour >= 3;
    }

    const int transitionDay =
        NthSundayOfMonth(
            year,
            11,
            1);

    if (day < transitionDay)
        return true;

    if (day > transitionDay)
        return false;

    return hour < 1;
}


bool IsExceptionalNewYorkCivilTime(
    const std::tm& civil)
{
    const int year =
        civil.tm_year + 1900;

    const int month =
        civil.tm_mon + 1;

    const int day =
        civil.tm_mday;

    const int hour =
        civil.tm_hour;

    if (month == 3)
    {
        const int transitionDay =
            NthSundayOfMonth(
                year,
                3,
                2);

        return
            day == transitionDay &&
            hour == 2;
    }

    if (month == 11)
    {
        const int transitionDay =
            NthSundayOfMonth(
                year,
                11,
                1);

        return
            day == transitionDay &&
            hour == 1;
    }

    return false;
}

} // namespace


bool ParseNewYorkCivilTimestamp(
    const char* text,
    PriceTP& out)
{
    if (!text)
        return false;

    std::tm civil{};

    std::istringstream stream{text};
    stream.imbue(
        std::locale::classic());

    stream >>
        std::get_time(
            &civil,
            "%F %T");

    if (stream.fail())
        return false;

    // Reject trailing non-whitespace input.
    stream >> std::ws;

    if (!stream.eof())
        return false;

    const int parsedYear = civil.tm_year;
    const int parsedMonth = civil.tm_mon;
    const int parsedDay = civil.tm_mday;
    const int parsedHour = civil.tm_hour;
    const int parsedMinute = civil.tm_min;
    const int parsedSecond = civil.tm_sec;

    if (
        parsedYear < 0 ||
        parsedMonth < 0 ||
        parsedMonth > 11 ||
        parsedDay < 1 ||
        parsedDay > DaysInMonth(
            parsedYear + 1900,
            parsedMonth + 1) ||
        parsedHour < 0 ||
        parsedHour > 23 ||
        parsedMinute < 0 ||
        parsedMinute > 59 ||
        parsedSecond < 0 ||
        parsedSecond > 59)
    {
        return false;
    }

    if (
        IsExceptionalNewYorkCivilTime(
            civil))
    {
        return false;
    }

    std::tm utcFields =
        civil;

    utcFields.tm_isdst = 0;

    const std::time_t civilAsUtc =
        timegm(
            &utcFields);

    if (
        civilAsUtc ==
        static_cast<std::time_t>(-1))
    {
        return false;
    }

    // timegm normalizes invalid civil fields.  The explicit range checks
    // above reject most such inputs; this round trip is the final guard.
    if (
        utcFields.tm_year != parsedYear ||
        utcFields.tm_mon != parsedMonth ||
        utcFields.tm_mday != parsedDay ||
        utcFields.tm_hour != parsedHour ||
        utcFields.tm_min != parsedMinute ||
        utcFields.tm_sec != parsedSecond)
    {
        return false;
    }

    const std::time_t offsetSeconds =
        IsNewYorkDstCivilTime(civil)
            ? 4 * 60 * 60
            : 5 * 60 * 60;

    const std::time_t utcTime =
        civilAsUtc +
        offsetSeconds;

    out =
        std::chrono::time_point_cast<
            std::chrono::seconds>(
                std::chrono::system_clock::
                    from_time_t(
                        utcTime));

    return true;
}

} // namespace EA::HistoricalFxTimestamp
