//
//  ResultIter.cpp
//  ChartTest
//
//  Created by Vincent Predoehl on 6/14/19.
//  Copyright © 2019 Vincent Predoehl. All rights reserved.
//

#include "db_cursor_iterator.hpp"
#include "FxPriceSanity.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <locale>
#include <cstdlib>
#include <cerrno>
#include <cctype>

namespace
{
constexpr size_t kDiagLimit = 50;

// The historical DAT_NT_* FX files used to populate the RMP tables carry
// naive U.S. Eastern civil timestamps. Convert those wall-clock values to
// UTC deterministically instead of depending on the host process timezone.
//
// For this dataset (2009+), America/New_York follows the post-2007 U.S. DST
// rule: DST begins at 02:00 on the second Sunday in March and ends at 02:00
// on the first Sunday in November.
int DaysInMonth(int year, int month)
{
    static constexpr int days[] =
        { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

    if (month != 2)
        return days[month - 1];

    const bool leap =
        (year % 4 == 0 && year % 100 != 0) ||
        (year % 400 == 0);
    return leap ? 29 : 28;
}

int WeekdayUtc(int year, int month, int day)
{
    std::tm tm {};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = 12;
    const std::time_t t = timegm(&tm);

    std::tm utc {};
    gmtime_r(&t, &utc);
    return utc.tm_wday;
}

int NthSundayOfMonth(int year, int month, int nth)
{
    const int firstWeekday = WeekdayUtc(year, month, 1);
    const int firstSunday = 1 + ((7 - firstWeekday) % 7);
    const int result = firstSunday + 7 * (nth - 1);

    if (result > DaysInMonth(year, month))
        return -1;
    return result;
}

bool IsNewYorkDstCivilTime(const std::tm& ct)
{
    const int year = ct.tm_year + 1900;
    const int month = ct.tm_mon + 1;
    const int day = ct.tm_mday;
    const int hour = ct.tm_hour;

    if (month < 3 || month > 11)
        return false;
    if (month > 3 && month < 11)
        return true;

    if (month == 3)
    {
        const int transitionDay = NthSundayOfMonth(year, 3, 2);
        if (day < transitionDay) return false;
        if (day > transitionDay) return true;
        return hour >= 3;
    }

    const int transitionDay = NthSundayOfMonth(year, 11, 1);
    if (day < transitionDay) return true;
    if (day > transitionDay) return false;
    return hour < 1;
}

bool IsNewYorkNonexistentOrAmbiguousCivilTime(const std::tm& ct)
{
    const int year = ct.tm_year + 1900;
    const int month = ct.tm_mon + 1;
    const int day = ct.tm_mday;
    const int hour = ct.tm_hour;

    if (month == 3)
    {
        const int transitionDay = NthSundayOfMonth(year, 3, 2);
        return day == transitionDay && hour == 2;
    }

    if (month == 11)
    {
        const int transitionDay = NthSundayOfMonth(year, 11, 1);
        return day == transitionDay && hour == 1;
    }

    return false;
}

bool NewYorkCivilToUtc(const std::tm& civil, std::time_t& out)
{
    if (IsNewYorkNonexistentOrAmbiguousCivilTime(civil))
        return false;

    std::tm utcFields = civil;
    utcFields.tm_isdst = 0;

    const std::time_t civilAsUtc = timegm(&utcFields);
    if (civilAsUtc == static_cast<std::time_t>(-1))
        return false;

    const std::time_t offsetSeconds =
        IsNewYorkDstCivilTime(civil) ? 4 * 60 * 60 : 5 * 60 * 60;

    out = civilAsUtc + offsetSeconds;
    return true;
}

bool ParseTimestampStrict(const char* text, PriceTP& out)
{
    if (!text)
        return false;

    std::tm ct {};
    std::istringstream ss { text };
    ss.imbue(std::locale::classic());
    ss >> std::get_time(&ct, "%F %T");
    if (ss.fail())
        return false;

    std::time_t utcTime {};
    if (!NewYorkCivilToUtc(ct, utcTime))
        return false;

    out = std::chrono::time_point_cast<std::chrono::seconds>(
        std::chrono::system_clock::from_time_t(utcTime));
    return true;
}

bool ParseFloatStrict(const char* text, float& out)
{
    if (!text)
        return false;

    errno = 0;
    char* end = nullptr;
    const float parsed = std::strtof(text, &end);

    if (end == text || errno == ERANGE || !std::isfinite(parsed))
        return false;

    while (end && *end != '\0' && std::isspace(static_cast<unsigned char>(*end)))
        ++end;

    if (end && *end != '\0')
        return false;

    out = parsed;
    return true;
}

}


template<>
bool db_input_iterator<PricePoint>::ReadPP()
{
    pqxx::result r;
    bool lineRead = *cur >> r;
    pp = PricePoint {};

    if(lineRead)
    {
        auto bid { r[0]["bid"] }, ask { r[0]["ask"] };
        std::istringstream time { r[0]["time"].c_str() };

        time >> pp.time;  bid >> pp.bid; ask >> pp.ask;
    }
    return lineRead;
}


template<>
bool db_input_iterator<Feature>::ReadPP()
{
    pqxx::result r;
    bool lineRead = *cur >> r;
    pp = Feature {};

    if(!lineRead)
        return false;

    const size_t rowIndex = cur->nextRowIndex++;
    const std::string dtText = r[0]["dt"].c_str();
    const std::string openText = r[0]["open"].c_str();
    const std::string closeText = r[0]["close"].c_str();
    const std::string highText = r[0]["high"].c_str();
    const std::string lowText = r[0]["low"].c_str();
    const std::string volumeText = r[0]["vol"].c_str();

    Feature parsed {};
    const bool dtOk = ParseTimestampStrict(dtText.c_str(), parsed.time);
    const bool openOk = ParseFloatStrict(openText.c_str(), parsed.open);
    const bool closeOk = ParseFloatStrict(closeText.c_str(), parsed.close);
    const bool highOk = ParseFloatStrict(highText.c_str(), parsed.high);
    const bool lowOk = ParseFloatStrict(lowText.c_str(), parsed.low);
    const bool volumeOk = ParseFloatStrict(volumeText.c_str(), parsed.tickVolume) &&
        parsed.tickVolume >= 0.0f;
    const bool parseOk = dtOk && openOk && closeOk && highOk && lowOk && volumeOk;

    pp = parsed;

    if (!parseOk)
    {
        if(cur->parseFailDiagCount < kDiagLimit)
        {
            std::cout << "DIAG_DB_PARSE_FAIL"
            << ",cursor=" << cur->cursorName
            << ",row=" << rowIndex
            << ",dt_ok=" << static_cast<int>(dtOk)
            << ",open_ok=" << static_cast<int>(openOk)
            << ",close_ok=" << static_cast<int>(closeOk)
            << ",high_ok=" << static_cast<int>(highOk)
            << ",low_ok=" << static_cast<int>(lowOk)
            << ",volume_ok=" << static_cast<int>(volumeOk)
            << ",dt_text=" << dtText
            << ",open_text=" << openText
            << ",close_text=" << closeText
            << ",high_text=" << highText
            << ",low_text=" << lowText
            << ",volume_text=" << volumeText
            << ",query=" << cur->queryText
            << std::endl;
            ++cur->parseFailDiagCount;
        }
        throw std::runtime_error("DB feature parse failed");
    }
    const std::string symbol = EA::FxPriceSanity::ExtractSymbol(cur->cursorName).value_or(cur->cursorName);
    const auto sanityBounds = EA::FxPriceSanity::BoundsForSymbol(symbol);
    const bool badOpen = !EA::FxPriceSanity::IsSanePrice(parsed.open, sanityBounds);
    const bool badClose = !EA::FxPriceSanity::IsSanePrice(parsed.close, sanityBounds);
    const bool badHigh = !EA::FxPriceSanity::IsSanePrice(parsed.high, sanityBounds);
    const bool badLow = !EA::FxPriceSanity::IsSanePrice(parsed.low, sanityBounds);
    const bool highLtLow = std::isfinite(parsed.high) && std::isfinite(parsed.low) && parsed.high < parsed.low;
    if ((badOpen || badClose || badHigh || badLow || highLtLow) &&
        cur->badRowDiagCount < kDiagLimit)
    {
        std::cout << "DIAG_DB_BAD_ROW"
                  << ",cursor=" << cur->cursorName
                  << ",row=" << rowIndex
                  << ",dt=" << parsed.time
                  << ",open=" << parsed.open
                  << ",close=" << parsed.close
                  << ",high=" << parsed.high
                  << ",low=" << parsed.low
                  << ",symbol=" << symbol
                  << ",bad_open=" << static_cast<int>(badOpen)
                  << ",bad_close=" << static_cast<int>(badClose)
                  << ",bad_high=" << static_cast<int>(badHigh)
                  << ",bad_low=" << static_cast<int>(badLow)
                  << ",high_lt_low=" << static_cast<int>(highLtLow)
                  << ",sane_lower_bound=" << sanityBounds.lower
                  << ",sane_upper_bound=" << sanityBounds.upper
                  << ",query=" << cur->queryText
                  << std::endl;
        ++cur->badRowDiagCount;
    }

    return true;
}
