#include "TG4HistoricalMarketDataRepository.hpp"

#include "HistoricalFxTimestamp.hpp"

#include <chrono>
#include <stdexcept>
#include <string>
#include <tuple>

namespace EA::TG4
{
namespace
{

std::string CanonicalBarsCte()
{
    return
        "WITH canonical_bars AS ("
        "SELECT dt,open,close,high,low,vol FROM candlestick("
        "$1::text,$2::integer,$3::text,"
        "($4::timestamptz AT TIME ZONE 'America/New_York'),"
        "($5::timestamptz AT TIME ZONE 'America/New_York'))),"
        "bounded AS (SELECT * FROM canonical_bars WHERE "
        "(dt AT TIME ZONE 'America/New_York') >= $4::timestamptz AND "
        "(dt AT TIME ZONE 'America/New_York') < $5::timestamptz) ";
}

pqxx::params Parameters(const std::string& symbol,
                        const EvaluationConfiguration& configuration,
                        const TemporalRange& range)
{
    return pqxx::params{
        symbol,
        configuration.candlePeriod,
        configuration.candleUnit,
        FormatUtcTimestamp(range.warmupStart),
        FormatUtcTimestamp(range.outcomeEnd)};
}

std::int64_t ParseCanonicalTimestamp(const std::string& sourceTimestamp)
{
    PriceTP parsed;
    if (!HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
            sourceTimestamp, parsed))
        throw std::runtime_error(
            "TG4 canonical candle timestamp could not be parsed: " +
            sourceTimestamp);
    return parsed.time_since_epoch().count();
}

} // namespace

MarketDataPreflight HistoricalMarketDataRepository::Preflight(
    pqxx::read_transaction& transaction,
    const std::string& symbol,
    const EvaluationConfiguration& configuration,
    const TemporalRange& range)
{
    const std::string query = CanonicalBarsCte() +
        "SELECT count(*)::bigint,"
        "(count(*)-count(DISTINCT dt))::bigint,"
        "min(extract(epoch from (dt AT TIME ZONE 'America/New_York')))::bigint,"
        "max(extract(epoch from (dt AT TIME ZONE 'America/New_York')))::bigint "
        "FROM bounded;";
    const pqxx::row row = transaction.exec(query, Parameters(
        symbol, configuration, range)).one_row();
    MarketDataPreflight result;
    result.rowCount = row[0].as<std::size_t>();
    result.duplicateTimestampCount = row[1].as<std::size_t>();
    if (!row[2].is_null()) result.firstTimestamp = row[2].as<std::int64_t>();
    if (!row[3].is_null()) result.lastTimestamp = row[3].as<std::int64_t>();
    return result;
}

void HistoricalMarketDataRepository::StreamCanonicalCandles(
    pqxx::read_transaction& transaction,
    const std::string& symbol,
    const EvaluationConfiguration& configuration,
    const TemporalRange& range,
    const CandleConsumer& consumer)
{
    if (!consumer) throw std::invalid_argument("TG4 candle consumer is empty");
    const std::string query = CanonicalBarsCte() +
        "SELECT to_char(dt,'YYYY-MM-DD HH24:MI:SS'),"
        "open::double precision,high::double precision,"
        "low::double precision,close::double precision,vol::bigint "
        "FROM bounded ORDER BY dt;";
    auto stream = transaction.stream<std::string, double, double, double,
                                     double, long long>(
        query, Parameters(symbol, configuration, range));
    for (const auto& [timestamp, open, high, low, close, volume] : stream)
    {
        if (volume < 0)
            throw std::runtime_error(
                "TG4 canonical candle has negative source tick volume");
        consumer({ParseCanonicalTimestamp(timestamp), open, high, low, close});
    }
}

} // namespace EA::TG4
