#pragma once

#include "TG4HistoricalEmpiricalEvaluation.hpp"

#include <pqxx/pqxx>

#include <cstddef>
#include <functional>
#include <optional>
#include <string>

namespace EA::TG4
{

struct MarketDataPreflight
{
    std::size_t rowCount = 0;
    std::size_t duplicateTimestampCount = 0;
    std::optional<std::int64_t> firstTimestamp;
    std::optional<std::int64_t> lastTimestamp;
};

class HistoricalMarketDataRepository
{
public:
    using CandleConsumer = std::function<void(const TG1A::Candle&)>;

    static MarketDataPreflight Preflight(
        pqxx::read_transaction& transaction,
        const std::string& symbol,
        const EvaluationConfiguration& configuration,
        const TemporalRange& range);

    static void StreamCanonicalCandles(
        pqxx::read_transaction& transaction,
        const std::string& symbol,
        const EvaluationConfiguration& configuration,
        const TemporalRange& range,
        const CandleConsumer& consumer);
};

} // namespace EA::TG4
