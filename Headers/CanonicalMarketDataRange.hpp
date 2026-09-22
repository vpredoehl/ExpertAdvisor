#pragma once

#include "PricePoint.hpp"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

// The one cross-consumer market-data boundary.  Database candlestick `dt`
// values are America/New_York civil bar starts, while callers of this API use
// absolute UTC PriceTP instants exclusively.  The SQL converts at the one
// database boundary and filters the resulting bar-start instants as [start,end).
namespace EA::CanonicalMarketData
{
inline constexpr int kCanonicalCandlePeriodMinutes = 15;
inline constexpr std::int64_t kCanonicalIntervalSeconds = 900;
inline constexpr const char* kAbsoluteHalfOpenContractVersion =
    "canonical-absolute-half-open-candlestick-v1";

struct AbsoluteHalfOpenRange final
{
    PriceTP start;
    PriceTP end;

    void Validate() const
    {
        if (!(start < end))
            throw std::invalid_argument(
                "canonical market-data range requires start < end");
    }

    bool Contains(PriceTP timestamp) const noexcept
    {
        return start <= timestamp && timestamp < end;
    }
};

inline std::string FormatAbsoluteUtc(PriceTP timestamp)
{
    const std::time_t raw = std::chrono::system_clock::to_time_t(timestamp);
    std::tm utc{};
    if (gmtime_r(&raw, &utc) == nullptr)
        throw std::invalid_argument("canonical market-data timestamp is invalid");
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::put_time(&utc, "%F %T") << "+00";
    return out.str();
}

// Arguments are already SQL-quoted by the caller's pqxx transaction.  Keeping
// quoting at that caller boundary lets both pqxx::work and read_transaction use
// this exact SQL without sharing transaction ownership.
inline std::string CanonicalHalfOpenCandlestickCte(
    const std::string& quotedSymbol,
    const std::string& quotedPeriod,
    const std::string& quotedUnit,
    const std::string& quotedStartUtc,
    const std::string& quotedEndUtc)
{
    return
        "WITH canonical_bars AS ("
        "SELECT dt,open,close,high,low,vol FROM candlestick(" +
        quotedSymbol + "::text," + quotedPeriod + "::integer," +
        quotedUnit + "::text,(" + quotedStartUtc +
        "::timestamptz AT TIME ZONE 'America/New_York'),(" +
        quotedEndUtc +
        "::timestamptz AT TIME ZONE 'America/New_York'))),"
        "bounded AS (SELECT * FROM canonical_bars WHERE "
        "(dt AT TIME ZONE 'America/New_York') >= " + quotedStartUtc +
        "::timestamptz AND (dt AT TIME ZONE 'America/New_York') < " +
        quotedEndUtc + "::timestamptz) ";
}

template <typename Transaction>
std::string CanonicalHalfOpenCandlestickCte(
    Transaction& transaction,
    const std::string& symbol,
    const AbsoluteHalfOpenRange& range,
    int candlePeriod = kCanonicalCandlePeriodMinutes,
    const std::string& candleUnit = "minute")
{
    range.Validate();
    if (symbol.empty() || candlePeriod != kCanonicalCandlePeriodMinutes ||
        candleUnit != "minute")
        throw std::invalid_argument(
            "canonical market-data API supports only 15-minute candles");
    return CanonicalHalfOpenCandlestickCte(
        transaction.quote(symbol), transaction.quote(candlePeriod),
        transaction.quote(candleUnit),
        transaction.quote(FormatAbsoluteUtc(range.start)),
        transaction.quote(FormatAbsoluteUtc(range.end)));
}

template <typename Transaction>
std::string CanonicalHalfOpenCandlestickQuery(
    Transaction& transaction,
    const std::string& symbol,
    const AbsoluteHalfOpenRange& range,
    int candlePeriod = kCanonicalCandlePeriodMinutes,
    const std::string& candleUnit = "minute")
{
    return CanonicalHalfOpenCandlestickCte(
        transaction, symbol, range, candlePeriod, candleUnit) +
        "SELECT dt,open,close,high,low,vol FROM bounded ORDER BY dt;";
}
} // namespace EA::CanonicalMarketData
