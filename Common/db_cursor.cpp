//
//  ResultIter.cpp
//  ChartTest
//
//  Created by Vincent Predoehl on 6/14/19.
//  Copyright © 2019 Vincent Predoehl. All rights reserved.
//

#include "db_cursor_iterator.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <ctime>

namespace
{
constexpr float kFxLowerBound = 0.2f;
constexpr float kFxUpperBound = 2.0f;
constexpr size_t kDiagLimit = 50;

bool ParseTimestampStrict(const char* text, PriceTP& out)
{
    if (!text)
        return false;

    std::tm ct {};
    std::istringstream ss { text };
    ss >> std::get_time(&ct, "%F %T");
    if (ss.fail())
        return false;

    out = std::chrono::time_point_cast<std::chrono::seconds>(
        std::chrono::system_clock::from_time_t(std::mktime(&ct)));
    return true;
}

bool ParseFloatStrict(const char* text, float& out)
{
    if (!text)
        return false;

    std::istringstream ss { text };
    ss >> out;
    ss >> std::ws;
    return !ss.fail() && ss.eof();
}

bool IsSaneFxPrice(float v)
{
    return std::isfinite(v) && v >= kFxLowerBound && v <= kFxUpperBound;
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
    const char* dtText = r[0]["dt"].c_str();
    const char* openText = r[0]["open"].c_str();
    const char* closeText = r[0]["close"].c_str();
    const char* highText = r[0]["high"].c_str();
    const char* lowText = r[0]["low"].c_str();

    Feature parsed {};
    const bool dtOk = ParseTimestampStrict(dtText, parsed.time);
    const bool openOk = ParseFloatStrict(openText, parsed.open);
    const bool closeOk = ParseFloatStrict(closeText, parsed.close);
    const bool highOk = ParseFloatStrict(highText, parsed.high);
    const bool lowOk = ParseFloatStrict(lowText, parsed.low);
    const bool parseOk = dtOk && openOk && closeOk && highOk && lowOk;

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
            << ",dt_text=" << (dtText ? dtText : "")
            << ",open_text=" << (openText ? openText : "")
            << ",close_text=" << (closeText ? closeText : "")
            << ",high_text=" << (highText ? highText : "")
            << ",low_text=" << (lowText ? lowText : "")
            << ",query=" << cur->queryText
            << std::endl;
            ++cur->parseFailDiagCount;
        }
        throw std::runtime_error("DB feature parse failed");
    }
    const bool badOpen = !IsSaneFxPrice(parsed.open);
    const bool badClose = !IsSaneFxPrice(parsed.close);
    const bool badHigh = !IsSaneFxPrice(parsed.high);
    const bool badLow = !IsSaneFxPrice(parsed.low);
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
                  << ",bad_open=" << static_cast<int>(badOpen)
                  << ",bad_close=" << static_cast<int>(badClose)
                  << ",bad_high=" << static_cast<int>(badHigh)
                  << ",bad_low=" << static_cast<int>(badLow)
                  << ",high_lt_low=" << static_cast<int>(highLtLow)
                  << ",sane_lower_bound=" << kFxLowerBound
                  << ",sane_upper_bound=" << kFxUpperBound
                  << ",query=" << cur->queryText
                  << std::endl;
        ++cur->badRowDiagCount;
    }

    return true;
}
