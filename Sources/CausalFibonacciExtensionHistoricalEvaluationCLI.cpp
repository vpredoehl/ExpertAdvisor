#include "CausalFibonacciExtensionHistoricalEvaluation.hpp"
#include "TG4HistoricalMarketDataRepository.hpp"

#include "SupportedSymbols.hpp"

#include <pqxx/pqxx>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace
{

constexpr std::string_view kBaselineCommit =
    "0440faa9a054d7c7c02df75e9162d3d85b33533d";

struct Options
{
    std::filesystem::path outputDirectory;
    std::filesystem::path configuration =
        "Scripts/tg4_analysis_config.frozen_v1.conf";
    std::optional<std::string> connection;
    std::string study = "pre2025";
};

Options Parse(int argc, char* argv[])
{
    Options result;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--output-dir" && index + 1 < argc)
            result.outputDirectory = argv[++index];
        else if (argument == "--config" && index + 1 < argc)
            result.configuration = argv[++index];
        else if (argument == "--connection" && index + 1 < argc)
            result.connection = argv[++index];
        else if (argument == "--study" && index + 1 < argc)
            result.study = argv[++index];
        else
            throw std::invalid_argument("unknown or incomplete option: " + argument);
    }
    if (result.outputDirectory.empty())
        throw std::invalid_argument("--output-dir is required");
    return result;
}

EA::FibonacciResearch::HistoricalStudySpecification StudySpecification(
    std::string_view selector)
{
    if (selector == "pre2025")
        return EA::FibonacciResearch::Pre2025FirstStudySpecification();
    if (selector == "confirmation2025")
        return EA::FibonacciResearch::Confirmation2025StudySpecification();
    throw std::invalid_argument("unknown Fibonacci study selector: " +
                                std::string(selector));
}

std::string ConnectionString(
    const Options& options,
    const EA::FibonacciResearch::HistoricalStudySpecification& study)
{
    if (options.connection.has_value()) return *options.connection;
    const char* host = std::getenv("FOREX_DB_HOST");
    const char* database = std::getenv("FOREX_DB_NAME");
    return "hostaddr=" + std::string(host != nullptr && *host != '\0'
        ? host : "127.0.0.1") +
        " gssencmode=disable user=pqxx dbname=" +
        std::string(database != nullptr && *database != '\0'
            ? database : "forex") +
        " application_name=fibonacci_extension_read_only_" +
        std::string(study.identity);
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        const Options options = Parse(argc, argv);
        const EA::FibonacciResearch::HistoricalStudySpecification study =
            StudySpecification(options.study);
        const EA::TG4::EvaluationConfiguration configuration =
            EA::TG4::LoadConfigurationFile(options.configuration);
        std::vector<std::string> symbols =
            EA::SupportedSymbols::TrainingSymbols();
        std::sort(symbols.begin(), symbols.end());
        const EA::TG4::TemporalRange range = study.range;
        const std::string reproduction =
            "bash Scripts/run_causal_fibonacci_extension_study.sh --output-dir " +
            options.outputDirectory.string() + " --study " + options.study;
        EA::FibonacciResearch::HistoricalArtifactWriter writer(
            options.outputDirectory, configuration, study,
            std::string(kBaselineCommit),
            symbols, reproduction);

        for (const std::string& symbol : symbols)
        {
            pqxx::connection connection{ConnectionString(options, study)};
            pqxx::read_transaction transaction{connection};
            transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
            const EA::TG4::MarketDataPreflight preflight =
                EA::TG4::HistoricalMarketDataRepository::Preflight(
                    transaction, symbol, configuration, range);
            if (preflight.rowCount == 0 ||
                preflight.duplicateTimestampCount != 0)
            {
                EA::FibonacciResearch::HistoricalDataQuality audit;
                audit.symbol = symbol;
                audit.excluded = true;
                audit.exclusionReason = preflight.rowCount == 0
                    ? "no_canonical_candles_in_query_range"
                    : "duplicate_canonical_candle_timestamps";
                audit.sourceRows = preflight.rowCount;
                audit.firstTimestamp = preflight.firstTimestamp;
                audit.lastTimestamp = preflight.lastTimestamp;
                writer.AddDataQuality(std::move(audit));
                transaction.commit();
                continue;
            }
            EA::FibonacciResearch::HistoricalEvaluator evaluator(
                symbol, configuration, study,
                [&writer](EA::FibonacciResearch::ObservationRecord record)
                { writer.AddRecord(std::move(record)); });
            EA::TG4::HistoricalMarketDataRepository::StreamCanonicalCandles(
                transaction, symbol, configuration, range,
                [&evaluator](const EA::TG1A::Candle& candle)
                { evaluator.AddCompletedBar(candle); });
            evaluator.Finalize();
            if (evaluator.DataQuality().sourceRows != preflight.rowCount)
                throw std::runtime_error(
                    "preflight/stream row-count mismatch under one snapshot");
            writer.AddDataQuality(evaluator.DataQuality());
            transaction.commit();
            std::cerr << "FIBONACCI_SYMBOL_COMPLETE symbol=" << symbol
                      << ",bars=" << evaluator.DataQuality().usableRows
                      << ",observations=" << evaluator.RecordCount() << '\n';
        }
        writer.Complete();
        std::cerr << "FIBONACCI_STUDY_COMPLETE observations="
                  << writer.ObservationCount() << ",output_dir="
                  << options.outputDirectory.string()
                  << ",read_only=true,uses_2025="
                  << (study.uses2025Bars ? "true" : "false")
                  << ",h3_closed=true\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FIBONACCI_STUDY_ERROR " << error.what() << '\n';
        return 1;
    }
}
