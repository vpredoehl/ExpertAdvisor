#pragma once

// Production, read-only extraction boundary for the frozen incremental-
// information protocol.  It deliberately owns no feature formulas: canonical
// bars feed Tensor, Tensor feeds the normal model-input materializer, and
// TargetLabel supplies every outcome audit.

#include "../Headers/CausalFibonacciIncrementalInformation.hpp"
#include "../Headers/CanonicalMarketDataRange.hpp"
#include "../Headers/FeatureWarmupScope.hpp"
#include "../Headers/ModelInputContract.hpp"
#include "../Headers/ReturnFeatureHistory.hpp"
#include "../Headers/SupportedSymbols.hpp"
#include "../Headers/TargetLabel.hpp"
#include "../Headers/Tensor.hpp"
#include "EconomicEventRepository.hpp"
#include "HistoricalFxTimestamp.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <limits>
#include <pqxx/pqxx>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace EA::CausalFibonacciIncrementalInformation::Extraction {

inline constexpr std::string_view kAdapterIdentity =
    "causal-fibonacci-layout9-authoritative-read-only-extraction-v1";
inline constexpr std::string_view kSourceQueryIdentity =
    "canonical-absolute-half-open-candlestick-v1:full-history-through-protocol-end";
inline constexpr std::string_view kAskOhlcPriceDomain = "ask_ohlc";
inline constexpr float kTargetThreshold = 0.0008F;

struct Options
{
    std::filesystem::path outputDirectory;
    std::string codeCommit;
    EconomicCalendar::EconomicCalendarSnapshotIdentity calendarSnapshot;
    std::string forexConnectionString;
    std::string lstmConnectionString;
};

inline std::int64_t Epoch(PriceTP value)
{
    return std::chrono::duration_cast<std::chrono::seconds>(
        value.time_since_epoch()).count();
}

inline PriceTP ProtocolEndTime()
{
    return PriceTP{std::chrono::seconds{kProtocolEnd}};
}

inline void UseRepeatableReadOnlySnapshot(pqxx::read_transaction& transaction)
{
    // read_transaction is structurally read-only.  Pin one snapshot before
    // the first statement, matching the repository's established research
    // consistency pattern.
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
}

inline std::array<float, kCausalFibonacciStructuralModelInputWidth>
PhysicalModelInputAt(const Tensor& tensor, std::size_t ordinal)
{
    if (ordinal >= tensor.RowCount())
        throw std::out_of_range("fibonacci_extraction_tensor_ordinal_out_of_range");
    const auto source = MetaNN::LowerAccess(
        *(tensor.begin() + static_cast<std::ptrdiff_t>(ordinal)));
    const auto contract = ResolveModelInputContract(
        kCausalFibonacciStructuralModelInputWidth,
        feature_size);
    std::array<float, kCausalFibonacciStructuralModelInputWidth> physical{};
    CopyTensorFeaturesForModelInput(physical.data(), source.RawMemory(), contract);
    const std::size_t appended = AppendMultiHorizonReturnFeaturesAtGlobalPosition(
        ordinal, physical.data(), contract.tensorFeatureCount, kFeatureScale,
        [&tensor](std::size_t position) {
            return tensor.RawCloseAtIterator(
                tensor.begin() + static_cast<std::ptrdiff_t>(position));
        });
    if (appended != kModelReturnFeatureCount)
        throw std::runtime_error("fibonacci_extraction_return_suffix_width_mismatch");
    return physical;
}

inline TargetAudit CensoredTarget(std::string reason)
{
    TargetAudit audit;
    audit.eligible = false;
    audit.exclusionReason = std::move(reason);
    return audit;
}

inline TargetAudit AuthoritativeTargetAudit(const Tensor& tensor,
                                            std::size_t ordinal,
                                            std::size_t horizon,
                                            Partition partition)
{
    if (ordinal + horizon >= tensor.RowCount())
        return CensoredTarget("insufficient_future_canonical_observations");
    const auto start = tensor.begin() + static_cast<std::ptrdiff_t>(ordinal);
    const LookaheadClassInfo info = BuildLookaheadClassInfo(
        tensor, start, 1, horizon, kTargetThreshold);
    TargetAudit audit = ProjectAuthoritativeTargetAudit(
        info.assignedClass, Epoch(tensor.RawTimeAtIterator(
            start + static_cast<std::ptrdiff_t>(info.selectedOffset))),
        Epoch(tensor.RawTimeAtIterator(
            start + static_cast<std::ptrdiff_t>(horizon))),
        info.terminalLogReturn);
    // The target horizon, not only the first selected threshold hit, must fit
    // inside the decision partition.
    if (audit.terminalTimestamp >= PartitionEnd(partition))
    {
        audit.eligible = false;
        audit.exclusionReason = "target_horizon_crosses_partition_boundary";
    }
    return audit;
}

inline bool FinitePredictors(const Row& row)
{
    for (float value : row.baseline) if (!std::isfinite(value)) return false;
    for (float value : row.fibonacci) if (!std::isfinite(value)) return false;
    return true;
}

inline void AssertCanonicalTensorIdentity(const Tensor& tensor)
{
    std::set<std::int64_t> timestamps;
    std::int64_t previous = std::numeric_limits<std::int64_t>::min();
    for (std::size_t ordinal = 0; ordinal < tensor.RowCount(); ++ordinal)
    {
        const std::int64_t timestamp = Epoch(tensor.RawTimeAtIterator(
            tensor.begin() + static_cast<std::ptrdiff_t>(ordinal)));
        if (timestamp <= previous || !timestamps.insert(timestamp).second)
            throw std::invalid_argument(
                "fibonacci_extraction_duplicate_or_unordered_canonical_timestamp");
        previous = timestamp;
    }
}

inline void ProjectTensorToArtifact(const std::string& symbol,
                                    const Tensor& tensor,
                                    ArtifactWriter& writer)
{
    AssertCanonicalTensorIdentity(tensor);
    for (std::size_t ordinal = 0; ordinal < tensor.RowCount(); ++ordinal)
    {
        const auto iterator = tensor.begin() + static_cast<std::ptrdiff_t>(ordinal);
        const std::int64_t timestamp = Epoch(tensor.RawTimeAtIterator(iterator));
        const auto partition = PartitionFor(timestamp);
        if (!partition) continue; // Warmup and post-protocol bars are not rows.
        Row row = ProjectAuthoritativeLayout9Row(
            symbol, timestamp, ordinal, PhysicalModelInputAt(tensor, ordinal));
        if (!FinitePredictors(row))
        {
            writer.Exclude({symbol, timestamp, ordinal,
                            "nonfinite_authoritative_predictor"});
            continue;
        }
        row.h4 = AuthoritativeTargetAudit(tensor, ordinal, 4, *partition);
        row.h6 = AuthoritativeTargetAudit(tensor, ordinal, 6, *partition);
        writer.Append(row); // scale-invalid Fibonacci rows remain finite rows.
    }
}

inline Tensor MaterializeAuthoritativeTensor(
    pqxx::read_transaction& forexRead,
    pqxx::read_transaction& lstmRead,
    const std::string& symbol,
    const EconomicCalendar::EconomicCalendarSnapshotIdentity& snapshot)
{
    // Inspect first: an absent, mutable, non-finalized, or hash-mismatched
    // calendar never reaches Tensor materialization.
    (void)EconomicCalendar::InspectEconomicCalendarSnapshot(lstmRead, snapshot);
    const std::vector<EconomicCalendar::EconomicEvent> economicEvents =
        EconomicCalendar::LoadEconomicEventsForFeatureRange(
            lstmRead, std::string{EconomicCalendar::kEconomicEventFeatureCurrency},
            kTensorFeatureHistoryQueryStart,
            CanonicalMarketData::FormatAbsoluteUtc(ProtocolEndTime()), snapshot);
    Tensor tensor{symbol, kDefaultDonchian20Mode, kDefaultDonchianLookback,
                  economicEvents};
    const std::string query = CanonicalMarketData::
        CanonicalFullHistoryThroughCandlestickCte(
            forexRead, symbol, ProtocolEndTime()) +
        "SELECT to_char(dt,'YYYY-MM-DD HH24:MI:SS'),open::double precision,"
        "high::double precision,low::double precision,close::double precision,"
        "vol::bigint FROM bounded ORDER BY dt";
    auto stream = forexRead.stream<std::string, double, double, double, double,
                                  long long>(query);
    for (const auto& [sourceTimestamp, open, high, low, close, volume] : stream)
    {
        PriceTP timestamp;
        if (!HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(sourceTimestamp, timestamp))
            throw std::runtime_error("fibonacci_extraction_canonical_timestamp_parse_failed");
        if (volume < 0 || !std::isfinite(open) || !std::isfinite(high) ||
            !std::isfinite(low) || !std::isfinite(close))
            throw std::runtime_error("fibonacci_extraction_invalid_canonical_ask_ohlc");
        tensor.Add({static_cast<float>(open), static_cast<float>(close),
                    static_cast<float>(high), static_cast<float>(low), timestamp,
                    static_cast<float>(volume)});
    }
    return tensor;
}

inline ArtifactProvenance Provenance(const Options& options)
{
    return {std::string(kProtocolId), std::string(kProtocolSha256), options.codeCommit,
            std::to_string(options.calendarSnapshot.snapshotId),
            options.calendarSnapshot.contentHash, std::string(kAdapterIdentity),
            std::string(kSourceQueryIdentity), std::string(kAskOhlcPriceDomain),
            "model-input-semantic-layout-v9:tensor=99:model_input=103",
            "layout8-model-input[0..79]:tensor[0..75]+returns[99..102]",
            "layout9-tensor[76..98]", "BuildLookaheadClassInfo:window=1:threshold=0.0008:H4,H6",
            "audcadrmp,audusdrmp,eurusdrmp,gbpusdrmp,usdcadrmp,usdjpyrmp:[2010-01-01,2026-01-01)",
            "full_history_warmup"};
}

inline std::vector<std::string> VerifiedFrozenUniverse()
{
    std::vector<std::string> symbols = SupportedSymbols::TrainingSymbols();
    std::sort(symbols.begin(), symbols.end());
    const std::vector<std::string> frozen{
        "audcadrmp", "audusdrmp", "eurusdrmp", "gbpusdrmp", "usdcadrmp", "usdjpyrmp"};
    if (symbols != frozen)
        throw std::runtime_error("fibonacci_extraction_frozen_six_symbol_universe_mismatch");
    return symbols;
}

inline void Extract(const Options& options)
{
    if (options.outputDirectory.empty() || options.codeCommit.empty() ||
        options.calendarSnapshot.snapshotId <= 0 ||
        options.calendarSnapshot.contentHash.empty())
        throw std::invalid_argument("fibonacci_extraction_required_identity_missing");
    if (std::filesystem::exists(options.outputDirectory) &&
        !std::filesystem::is_empty(options.outputDirectory))
        throw std::invalid_argument("fibonacci_extraction_refuses_existing_artifact_directory");

    pqxx::connection forexConnection{options.forexConnectionString};
    pqxx::connection lstmConnection{options.lstmConnectionString};
    pqxx::read_transaction forexRead{forexConnection};
    pqxx::read_transaction lstmRead{lstmConnection};
    UseRepeatableReadOnlySnapshot(forexRead);
    UseRepeatableReadOnlySnapshot(lstmRead);
    // Fail closed before filesystem artifact generation if the immutable
    // calendar identity cannot be supplied by the authoritative repository.
    (void)EconomicCalendar::InspectEconomicCalendarSnapshot(
        lstmRead, options.calendarSnapshot);
    ArtifactWriter writer(options.outputDirectory, Provenance(options));
    const std::vector<std::string> symbols = VerifiedFrozenUniverse();
    for (const std::string& symbol : symbols)
    {
        Tensor tensor = MaterializeAuthoritativeTensor(
            forexRead, lstmRead, symbol, options.calendarSnapshot);
        ProjectTensorToArtifact(symbol, tensor, writer);
    }
    writer.Complete();
    forexRead.commit();
    lstmRead.commit();
}

} // namespace EA::CausalFibonacciIncrementalInformation::Extraction
