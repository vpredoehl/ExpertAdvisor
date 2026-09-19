#include "MarketDataCore.hpp"

#include "CanonicalSymbol.hpp"
#include "db_cursor_iterator.hpp"

#include <pqxx/pqxx>
#include <stdexcept>

namespace EA::MarketData
{
std::vector<std::string> DiscoverRawPriceTables(pqxx::work& transaction)
{
    const pqxx::result tables = transaction.exec(
        "select table_name from information_schema.tables where "
        "table_schema = 'public' and table_name like '%rmp' order by table_name;");
    std::vector<std::string> symbols;
    symbols.reserve(tables.size());
    for (const auto& table : tables)
        symbols.emplace_back(CanonicalSymbol::Normalize(table[0].c_str()));
    return symbols;
}

CandlestickSlice LoadCandlesticks(pqxx::work& transaction,
                                  const CandlestickRange& range)
{
    if (range.symbol.empty() || range.queryStart.empty() || range.queryEnd.empty())
        throw std::invalid_argument("candlestick range requires symbol and dates");

    CandlestickSlice slice;
    slice.query = "select * from candlestick(" + transaction.quote(range.symbol) +
        ", 15, 'minute', " + transaction.quote(range.queryStart) + ", " +
        transaction.quote(range.queryEnd) + ") order by dt;";
    if (range.calculateLogicalOutputStart)
    {
        const std::string countQuery = "select count(*) from candlestick(" +
            transaction.quote(range.symbol) + ", 15, 'minute', " +
            transaction.quote(range.queryStart) + ", " +
            transaction.quote(range.queryEnd) + ") where dt < " +
            transaction.quote(range.outputStart) + ";";
        slice.logicalOutputStartIndex = transaction.exec(countQuery).one_row()[0]
            .as<std::size_t>();
    }
    db_cursor_stream<Feature> cursor{transaction, slice.query, range.cursorName};
    for (auto it = cursor.begin(), end = cursor.end(); it != end; ++it)
        slice.rows.push_back(*it);
    return slice;
}

OutcomeCoverage CheckProspectiveOutcomeCoverage(pqxx::work& transaction,
                                                const std::string& symbol,
                                                const std::string& fromDate,
                                                const std::string& toDate)
{
    const std::string sql = "WITH bars AS (SELECT dt FROM candlestick(" +
        transaction.quote(symbol) + ",15,'minute'," +
        transaction.quote(fromDate) + "," + transaction.quote(toDate) +
        ")) SELECT min(dt)::text,max(dt)::text,count(*),COALESCE(min(dt) <= " +
        transaction.quote(fromDate) + "::timestamp,false),COALESCE(max(dt) >= (" +
        transaction.quote(toDate) + "::timestamp - interval '15 minutes'),false) FROM bars";
    const pqxx::row row = transaction.exec(sql).one_row();
    return {row[0].is_null() ? "" : row[0].as<std::string>(),
            row[1].is_null() ? "" : row[1].as<std::string>(),
            row[2].as<long long>(), row[3].as<bool>(), row[4].as<bool>()};
}
}
