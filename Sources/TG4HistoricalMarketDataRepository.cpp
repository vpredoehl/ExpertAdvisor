#include "TG4HistoricalMarketDataRepository.hpp"

#include "CanonicalMarketDataRange.hpp"
#include "HistoricalFxTimestamp.hpp"

#include <chrono>
#include <stdexcept>
#include <string>
#include <tuple>

namespace EA::TG4
{
namespace
{

std::string CanonicalBarsStreamingCte(
    const pqxx::read_transaction& transaction,
    const std::string& symbol,
    const EvaluationConfiguration& configuration,
    const TemporalRange& range)
{
    return CanonicalMarketData::CanonicalHalfOpenCandlestickCte(
        transaction, symbol,
        {PriceTP{std::chrono::seconds{range.warmupStart}},
         PriceTP{std::chrono::seconds{range.outcomeEnd}}},
        configuration.candlePeriod, configuration.candleUnit);
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
    const std::string query = CanonicalBarsStreamingCte(
        transaction, symbol, configuration, range) +
        "SELECT count(*)::bigint,"
        "(count(*)-count(DISTINCT dt))::bigint,"
        "min(extract(epoch from (dt AT TIME ZONE 'America/New_York')))::bigint,"
        "max(extract(epoch from (dt AT TIME ZONE 'America/New_York')))::bigint "
        "FROM bounded;";
    const pqxx::row row = transaction.exec(query).one_row();
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
    const std::string query = CanonicalBarsStreamingCte(
        transaction, symbol, configuration, range) +
        "SELECT to_char(dt,'YYYY-MM-DD HH24:MI:SS'),"
        "open::double precision,high::double precision,"
        "low::double precision,close::double precision,vol::bigint "
        "FROM bounded ORDER BY dt";
    auto stream = transaction.stream<std::string, double, double, double,
                                     double, long long>(query);
    for (const auto& [timestamp, open, high, low, close, volume] : stream)
    {
        if (volume < 0)
            throw std::runtime_error(
                "TG4 canonical candle has negative source tick volume");
        consumer({ParseCanonicalTimestamp(timestamp), open, high, low, close});
    }
}

} // namespace EA::TG4
