#include "ModelInputPreparation.hpp"

#include "EconomicEventRepository.hpp"
#include "LstmRuntimeLogging.hpp"
#include "MarketDataCore.hpp"
#include "ReturnFeatureHistory.hpp"

#include <iostream>
#include <pqxx/pqxx>
#include <stdexcept>
#include <utility>
#include <vector>

namespace EA::ModelInputPreparation
{
Result Prepare(const Request& request,
               const DatabaseConnectionSettings& connectionSettings)
{
    const bool fullHistoryWarmup =
        request.warmupScope == EA::FeatureWarmupScope::FullHistoryWarmup;
    const std::string queryStart = fullHistoryWarmup
        ? EA::kTensorFeatureHistoryQueryStart
        : request.outputStart;

    pqxx::connection forexConnection{connectionSettings.forexConnectionString};
    pqxx::work forexDataRead{forexConnection};
    forexDataRead.exec("SET TRANSACTION READ ONLY;");
    const auto marketData = EA::MarketData::LoadCandlesticks(
        forexDataRead,
        {request.symbol, queryStart, request.outputStart, request.outputEnd,
         request.symbol + "_candlestick_stream", fullHistoryWarmup});

    std::vector<EA::EconomicCalendar::EconomicEvent> economicEvents;
    {
        pqxx::connection lstmConnection{connectionSettings.lstmConnectionString};
        pqxx::work economicEventRead{lstmConnection};
        economicEventRead.exec("SET TRANSACTION READ ONLY;");
        economicEvents = EA::EconomicCalendar::LoadEconomicEventsForFeatureRange(
            economicEventRead,
            std::string{EA::EconomicCalendar::kEconomicEventFeatureCurrency},
            queryStart,
            request.outputEnd,
            request.calendarSnapshot);
        std::cout << "ECONOMIC_CALENDAR_CORPUS"
                  << ",behavior="
                  << (request.calendarSnapshot
                          ? "immutable_snapshot"
                          : "legacy_live_corpus")
                  << ",snapshot_id="
                  << (request.calendarSnapshot
                          ? std::to_string(request.calendarSnapshot->snapshotId)
                          : "NULL")
                  << ",content_hash="
                  << (request.calendarSnapshot
                          ? request.calendarSnapshot->contentHash
                          : "NULL")
                  << std::endl;
        economicEventRead.commit();
    }

    Tensor tensor{request.symbol, request.donchian20Mode,
                  request.donchianLookback, std::move(economicEvents)};
    if (EA::RuntimeDiagnosticLoggingEnabled())
    {
        std::cout << "Candlestick query: " << marketData.query << "\n";
        std::cout << "FEATURE_WARMUP_SCOPE"
                  << ",mode=" << EA::FeatureWarmupScopeText(request.warmupScope)
                  << ",source_start=" << queryStart
                  << ",output_start=" << request.outputStart
                  << ",output_end=" << request.outputEnd
                  << ",warmup_rows=" << marketData.logicalOutputStartIndex
                  << std::endl;
        std::cout << "Building tensor for table: " << request.symbol
                  << std::endl;
    }
    for (const auto& row : marketData.rows)
        tensor.Add(row);
    forexDataRead.commit();

    if (marketData.logicalOutputStartIndex > tensor.RowCount())
        throw std::runtime_error(
            "feature warmup query returned more rows than the source tensor");

    return {std::move(tensor), marketData.logicalOutputStartIndex,
            request.calendarSnapshot};
}
}
