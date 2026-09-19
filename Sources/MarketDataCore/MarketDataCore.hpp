#pragma once

#include "PricePoint.hpp"

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
