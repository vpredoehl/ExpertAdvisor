#include "TG4HistoricalEmpiricalEvaluation.hpp"
#include "TG4HistoricalMarketDataRepository.hpp"

#include "CanonicalSymbol.hpp"
#include "SupportedSymbols.hpp"

#include <pqxx/pqxx>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr std::string_view kTG4BaselineCommit =
    "02e414a1b9274318d95b73aa85597d6cde1a8b6d";

struct Options
{
    std::filesystem::path configurationPath;
    std::filesystem::path outputDirectory;
    std::optional<std::string> connectionString;
    std::vector<std::string> symbols;
    bool allCanonicalSymbols = false;
    std::optional<std::string> study;
    std::optional<std::string> partition;
    std::optional<std::string> warmupStart;
    std::optional<std::string> start;
    std::optional<std::string> end;
    std::optional<std::string> outcomeEnd;
};

std::string RequireNext(int argc, char* argv[], int& index,
                        const std::string& option)
{
    if (index + 1 >= argc)
        throw std::invalid_argument(option + " requires a value");
    return argv[++index];
}

void PrintUsage(std::ostream& output)
{
    output <<
        "Usage: tg4-historical-evaluation --config FILE --output-dir DIR "
        "(--symbol SYMBOL ... | --all-canonical-symbols) "
        "[--study tg4-2010-2025-v1 | --partition NAME | "
        "--warmup-start UTC --start UTC --end UTC --outcome-end UTC] "
        "[--connection CONNECTION_STRING]\n"
        "Partitions: exploratory, calibration, validation, confirmation\n"
        "All date ends are exclusive. Database access is read-only repeatable-read.\n";
}

Options ParseOptions(int argc, char* argv[])
{
    Options result;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--help")
        {
            PrintUsage(std::cout);
            std::exit(0);
        }
        if (argument == "--config")
            result.configurationPath = RequireNext(argc, argv, index, argument);
        else if (argument == "--output-dir")
            result.outputDirectory = RequireNext(argc, argv, index, argument);
        else if (argument == "--connection")
            result.connectionString = RequireNext(argc, argv, index, argument);
        else if (argument == "--symbol")
            result.symbols.push_back(EA::CanonicalSymbol::Normalize(
                RequireNext(argc, argv, index, argument)));
        else if (argument == "--all-canonical-symbols")
            result.allCanonicalSymbols = true;
        else if (argument == "--study")
            result.study = RequireNext(argc, argv, index, argument);
        else if (argument == "--partition")
            result.partition = RequireNext(argc, argv, index, argument);
        else if (argument == "--warmup-start")
            result.warmupStart = RequireNext(argc, argv, index, argument);
        else if (argument == "--start")
            result.start = RequireNext(argc, argv, index, argument);
        else if (argument == "--end")
            result.end = RequireNext(argc, argv, index, argument);
        else if (argument == "--outcome-end")
            result.outcomeEnd = RequireNext(argc, argv, index, argument);
        else
            throw std::invalid_argument("unknown TG4 option '" + argument + "'");
    }
    if (result.configurationPath.empty() || result.outputDirectory.empty())
        throw std::invalid_argument("--config and --output-dir are required");
    if (result.allCanonicalSymbols == !result.symbols.empty())
        throw std::invalid_argument(
            "choose exactly one of --symbol or --all-canonical-symbols");
    const int rangeModes = static_cast<int>(result.study.has_value()) +
        static_cast<int>(result.partition.has_value()) +
        static_cast<int>(result.start.has_value() || result.end.has_value() ||
                         result.warmupStart.has_value() ||
                         result.outcomeEnd.has_value());
    if (rangeModes != 1)
        throw std::invalid_argument(
            "choose exactly one of --study, --partition, or the explicit date range");
    return result;
}

EA::TG4::TemporalRange NamedRange(const Options& options)
{
    using EA::TG4::ParseUtcDateOrTimestamp;
    const auto range = [](std::string_view warmup, std::string_view start,
                          std::string_view end, std::string_view outcome)
    {
        return EA::TG4::TemporalRange{
            ParseUtcDateOrTimestamp(warmup), ParseUtcDateOrTimestamp(start),
            ParseUtcDateOrTimestamp(end), ParseUtcDateOrTimestamp(outcome)};
    };
    if (options.study.has_value())
    {
        if (*options.study != "tg4-2010-2025-v1")
            throw std::invalid_argument("unknown TG4 named study");
        return range("2010-01-01", "2010-01-01", "2026-01-01",
                     "2026-02-15");
    }
    if (options.partition.has_value())
    {
        if (*options.partition == "exploratory")
            return range("2010-01-01", "2010-01-01", "2020-01-01",
                         "2020-02-15");
        if (*options.partition == "calibration")
            return range("2010-01-01", "2020-01-01", "2023-01-01",
                         "2023-02-15");
        if (*options.partition == "validation")
            return range("2010-01-01", "2023-01-01", "2025-01-01",
                         "2025-02-15");
        if (*options.partition == "confirmation")
            return range("2010-01-01", "2025-01-01", "2026-01-01",
                         "2026-02-15");
        throw std::invalid_argument("unknown TG4 temporal partition");
    }
    if (!options.warmupStart.has_value() || !options.start.has_value() ||
        !options.end.has_value() || !options.outcomeEnd.has_value())
        throw std::invalid_argument(
            "explicit range requires --warmup-start, --start, --end, and --outcome-end");
    return range(*options.warmupStart, *options.start, *options.end,
                 *options.outcomeEnd);
}

std::string ConnectionString(const Options& options)
{
    if (options.connectionString.has_value()) return *options.connectionString;
    const char* host = std::getenv("FOREX_DB_HOST");
    const char* database = std::getenv("FOREX_DB_NAME");
    return "hostaddr=" + std::string(host != nullptr && *host != '\0'
        ? host : "127.0.0.1") +
        " gssencmode=disable user=pqxx dbname=" +
        std::string(database != nullptr && *database != '\0'
            ? database : "forex") +
        " application_name=tg4_read_only_historical_evaluation";
}

std::vector<std::string> SelectedSymbols(const Options& options)
{
    std::vector<std::string> symbols = options.allCanonicalSymbols
        ? EA::SupportedSymbols::TrainingSymbols() : options.symbols;
    const auto& canonical = EA::SupportedSymbols::TrainingSymbols();
    for (const std::string& symbol : symbols)
        if (std::find(canonical.begin(), canonical.end(), symbol) == canonical.end())
            throw std::invalid_argument(
                "TG4 symbol is outside the source-defined canonical universe: " +
                symbol);
    std::sort(symbols.begin(), symbols.end());
    symbols.erase(std::unique(symbols.begin(), symbols.end()), symbols.end());
    return symbols;
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        const Options options = ParseOptions(argc, argv);
        const EA::TG4::EvaluationConfiguration configuration =
            EA::TG4::LoadConfigurationFile(options.configurationPath);
        const EA::TG4::TemporalRange range = NamedRange(options);
        EA::TG4::ValidateTemporalRange(range);
        const std::vector<std::string> symbols = SelectedSymbols(options);
        EA::TG4::StudyArtifactWriter writer(
            options.outputDirectory, configuration, range,
            std::string(kTG4BaselineCommit));

        for (const std::string& symbol : symbols)
        {
            const auto started = std::chrono::steady_clock::now();
            pqxx::connection connection{ConnectionString(options)};
            pqxx::read_transaction transaction{connection};
            transaction.exec(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
            const EA::TG4::MarketDataPreflight preflight =
                EA::TG4::HistoricalMarketDataRepository::Preflight(
                    transaction, symbol, configuration, range);
            if (preflight.rowCount == 0 ||
                preflight.duplicateTimestampCount != 0)
            {
                EA::TG4::DataQualityAudit audit;
                audit.symbol = symbol;
                audit.excluded = true;
                audit.exclusionReason = preflight.rowCount == 0
                    ? "no_canonical_candles_in_query_range"
                    : "duplicate_canonical_candle_timestamps";
                audit.sourceRows = preflight.rowCount;
                audit.duplicateTimestamps = preflight.duplicateTimestampCount;
                audit.firstUsableTimestamp = preflight.firstTimestamp;
                audit.lastUsableTimestamp = preflight.lastTimestamp;
                writer.AddDataQuality(std::move(audit));
                transaction.commit();
                std::cerr << "TG4_SYMBOL_EXCLUDED symbol=" << symbol
                          << ",reason=" << (preflight.rowCount == 0
                              ? "no_canonical_candles_in_query_range"
                              : "duplicate_canonical_candle_timestamps")
                          << '\n';
                continue;
            }

            EA::TG4::HistoricalEvaluator evaluator(
                symbol, configuration, range,
                [&writer](const EA::TG4::ObservationRecord& record)
                {
                    writer.Write(record);
                });
            EA::TG4::HistoricalMarketDataRepository::StreamCanonicalCandles(
                transaction, symbol, configuration, range,
                [&evaluator](const EA::TG1A::Candle& candle)
                {
                    evaluator.AddCompletedBar(candle);
                });
            evaluator.Finalize();
            if (evaluator.DataQuality().sourceRows != preflight.rowCount)
                throw std::runtime_error(
                    "TG4 preflight/stream row-count mismatch under one snapshot");
            writer.AddDataQuality(evaluator.DataQuality());
            transaction.commit();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started);
            std::cerr << "TG4_SYMBOL_COMPLETE symbol=" << symbol
                      << ",bars=" << evaluator.DataQuality().usableRows
                      << ",observations=" << evaluator.EmittedRecordCount()
                      << ",peak_pending_records="
                      << evaluator.PeakPendingRecordCount()
                      << ",elapsed_ms=" << elapsed.count() << '\n';
        }
        writer.Complete();
        std::cerr << "TG4_STUDY_COMPLETE observations="
                  << writer.ObservationCount()
                  << ",output_dir=" << options.outputDirectory.string()
                  << ",read_only=true,parameter_optimization=false,"
                     "scheduler_started=false,workers_started=false\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "TG4_ERROR " << error.what() << '\n';
        return 1;
    }
}
