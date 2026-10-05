#include "CausalFibonacciRetracementLifecycleHistoricalArtifact.hpp"
#include "CausalFibonacciExtensionHistoricalEvaluation.hpp"
#include "TG4HistoricalMarketDataRepository.hpp"

#include <pqxx/pqxx>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace
{
namespace RL = EA::FibonacciResearch::RetracementLifecycle;

struct Options
{
    std::filesystem::path outputDirectory;
    std::filesystem::path configuration =
        "Scripts/tg4_analysis_config.frozen_v1.conf";
    std::optional<std::string> connection;
    std::string symbol = "audcadrmp";
    std::optional<std::string> study;
    std::optional<std::string> start;
    std::optional<std::string> end;
};

Options Parse(int argc, char* argv[])
{
    Options result;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--output-dir" && index + 1 < argc)
            result.outputDirectory = argv[++index];
        else if (argument == "--symbol" && index + 1 < argc)
            result.symbol = argv[++index];
        else if (argument == "--config" && index + 1 < argc)
            result.configuration = argv[++index];
        else if (argument == "--connection" && index + 1 < argc)
            result.connection = argv[++index];
        else if (argument == "--study" && index + 1 < argc)
            result.study = argv[++index];
        else if (argument == "--start" && index + 1 < argc)
            result.start = argv[++index];
        else if (argument == "--end" && index + 1 < argc)
            result.end = argv[++index];
        else
            throw std::invalid_argument("unknown or incomplete option: " + argument);
    }
    if (result.outputDirectory.empty())
        throw std::invalid_argument("--output-dir is required");
    if (result.study.has_value() == result.start.has_value() ||
        result.study.has_value() == result.end.has_value())
        throw std::invalid_argument(
            "supply exactly --study NAME or both --start and --end");
    return result;
}

EA::TG4::TemporalRange Range(const Options& options)
{
    if (options.study == "pre2025")
        return EA::FibonacciResearch::Pre2025FirstStudySpecification().range;
    if (options.study == "confirmation2025")
        return EA::FibonacciResearch::Confirmation2025StudySpecification().range;
    if (options.study.has_value())
        throw std::invalid_argument("unknown retracement lifecycle study: " +
                                    *options.study);
    const std::int64_t start = EA::TG4::ParseUtcDateOrTimestamp(*options.start);
    const std::int64_t end = EA::TG4::ParseUtcDateOrTimestamp(*options.end);
    return {start, start, end, end};
}

std::string ConnectionString(const Options& options)
{
    if (options.connection.has_value()) return *options.connection;
    const char* host = std::getenv("FOREX_DB_HOST");
    const char* database = std::getenv("FOREX_DB_NAME");
    return "hostaddr=" + std::string(host != nullptr && *host != '\0'
        ? host : "127.0.0.1") +
        " gssencmode=disable user=pqxx dbname=" +
        std::string(database != nullptr && *database != '\0'
            ? database : "forex") +
        " application_name=fibonacci_retracement_lifecycle_read_only";
}

std::string BaselineCommit()
{
    const char* value = std::getenv("GIT_COMMIT");
    return value != nullptr && *value != '\0' ? value : "unrecorded_worktree";
}
} // namespace

int main(int argc, char* argv[])
{
    try
    {
        const Options options = Parse(argc, argv);
        const EA::TG4::TemporalRange range = Range(options);
        EA::TG4::ValidateTemporalRange(range);
        const EA::TG4::EvaluationConfiguration configuration =
            EA::TG4::LoadConfigurationFile(options.configuration);
        if (configuration.timeframe != "15m" || configuration.candlePeriod != 15 ||
            configuration.candleUnit != "minute")
            throw std::invalid_argument(
                "retracement lifecycle runner requires canonical 15-minute bars");
        EA::TG3::Configuration fibonacci = configuration.fibonacci;
        fibonacci.absolutePriceTolerance =
            EA::TG4::EffectiveFibonacciAbsolutePriceTolerance(configuration,
                                                               options.symbol);
        const std::string requestedRange = "[" +
            EA::TG4::FormatUtcTimestamp(range.warmupStart) + "," +
            EA::TG4::FormatUtcTimestamp(range.outcomeEnd) + ")";
        const std::string reproduction =
            "bash Scripts/run_causal_fibonacci_retracement_lifecycle_study.sh "
            "--output-dir " + options.outputDirectory.string() + " --symbol " +
            options.symbol + " --start " + std::to_string(range.warmupStart) +
            " --end " + std::to_string(range.outcomeEnd);
        RL::HistoricalArtifactWriter writer(options.outputDirectory,
            {options.symbol, configuration.timeframe, options.configuration,
             EA::TG4::ConfigurationFingerprint(configuration), BaselineCommit(),
             requestedRange, reproduction});

        pqxx::connection connection{ConnectionString(options)};
        pqxx::read_transaction transaction{connection};
        transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
        const EA::TG4::MarketDataPreflight preflight =
            EA::TG4::HistoricalMarketDataRepository::Preflight(
                transaction, options.symbol, configuration, range);
        if (preflight.rowCount == 0 || preflight.duplicateTimestampCount != 0)
            throw std::runtime_error(preflight.rowCount == 0
                ? "no canonical candles in requested range"
                : "duplicate canonical candle timestamps in requested range");

        RL::HistoricalEvaluator evaluator(
            EA::TG1B::CalibrationConfiguration(configuration.referenceBarScale),
            fibonacci, configuration.geometry, configuration.behavior,
            {options.symbol, configuration.timeframe},
            [&writer](RL::HistoricalRecord record) { writer.AddRecord(std::move(record)); });
        std::size_t streamed = 0;
        EA::TG4::HistoricalMarketDataRepository::StreamCanonicalCandles(
            transaction, options.symbol, configuration, range,
            [&evaluator, &streamed](const EA::TG1A::Candle& candle)
            {
                ++streamed;
                evaluator.AddCompletedBar(candle);
            });
        evaluator.Finalize();
        if (streamed != preflight.rowCount)
            throw std::runtime_error(
                "preflight/stream row-count mismatch under one snapshot");
        writer.SetDataQuality({preflight.rowCount, streamed,
            preflight.duplicateTimestampCount, preflight.firstTimestamp,
            preflight.lastTimestamp});
        writer.Complete();
        transaction.commit();
        std::cerr << "FIBONACCI_RETRACEMENT_LIFECYCLE_COMPLETE symbol="
                  << options.symbol << ",bars=" << streamed
                  << ",observations=" << writer.ObservationCount()
                  << ",read_only=true\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FIBONACCI_RETRACEMENT_LIFECYCLE_ERROR " << error.what() << '\n';
        return 1;
    }
}
