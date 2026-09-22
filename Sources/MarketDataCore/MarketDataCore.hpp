#pragma once

#include "PricePoint.hpp"
#include "CanonicalMarketDataRange.hpp"

#include <pqxx/pqxx>

#include <cstddef>
#include <string>
#include <vector>

namespace EA::MarketData
{
struct CandlestickRange
{
    std::string symbol;
    std::string queryStart;
    std::string outputStart;
    std::string queryEnd;
    std::string cursorName;
    bool calculateLogicalOutputStart = false;
};

struct CandlestickSlice
{
    std::vector<Feature> rows;
    std::size_t logicalOutputStartIndex = 0;
    std::string query;
};

// New callers that require a cross-consumer identity must use this absolute
// [start,end) API.  LoadCandlesticks remains the legacy civil/inclusive path
// for existing model preparation and persisted experiment reproducibility.
CandlestickSlice LoadCanonicalHalfOpenCandlesticks(
    pqxx::work& transaction,
    const std::string& symbol,
    const CanonicalMarketData::AbsoluteHalfOpenRange& range,
    const std::string& cursorName);

struct OutcomeCoverage
{
    std::string first;
    std::string last;
    long long barCount = 0;
    bool coversStart = false;
    bool coversEnd = false;
};

// All operations use the caller's transaction.  This library never creates,
// commits, or aborts a transaction.
std::vector<std::string> DiscoverRawPriceTables(pqxx::work& transaction);
CandlestickSlice LoadCandlesticks(pqxx::work& transaction,
                                  const CandlestickRange& range);
OutcomeCoverage CheckProspectiveOutcomeCoverage(pqxx::work& transaction,
                                                const std::string& symbol,
                                                const std::string& fromDate,
                                                const std::string& toDate);
}
