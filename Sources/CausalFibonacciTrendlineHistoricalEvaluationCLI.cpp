#include "CausalFibonacciTrendlineHistoricalArtifact.hpp"
#include "RuntimeDatabaseConnection.hpp"
#include "TG4HistoricalMarketDataRepository.hpp"
#include <cstdlib>
#include <iostream>

#ifndef EA_RESEARCH_LIVE_ROOT
#error Immutable standalone build must embed live repository boundary
#endif
#ifndef EA_RESEARCH_SNAPSHOT_ROOT
#error Immutable standalone build must embed snapshot boundary
#endif
namespace
{
namespace S = EA::FibonacciResearch::TrendlineHistorical;
std::string Environment(const char* key)
{
    const char* value = std::getenv(key);
    if (!value || !*value) throw std::invalid_argument(std::string("missing immutable-build provenance: ") + key);
    return value;
}
bool Within(const std::filesystem::path& child, const std::filesystem::path& parent)
{
    auto a = child.begin();
    for (auto b = parent.begin(); b != parent.end(); ++a, ++b)
        if (a == child.end() || *a != *b) return false;
    return true;
}
}
int main(int argc, char** argv)
{
    try
    {
        S::Protocol protocol; std::filesystem::path output;
        bool help = false, havePartition = false;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--help") help = true;
            else if (arg == "--output-dir" && i + 1 < argc && output.empty()) output = argv[++i];
            else if (arg == "--partition" && i + 1 < argc && !havePartition)
            {
                havePartition = true; const std::string p = argv[++i];
                if (p == "development") protocol.partition = S::Partition::Development;
                else if (p == "calibration") protocol.partition = S::Partition::Calibration;
                else throw std::invalid_argument("only development/calibration allowed; 2025 disabled before detector/database access");
            }
            else throw std::invalid_argument("unknown, duplicate or incomplete option: " + arg);
        }
        if (help)
        {
            std::cout << "Phase 2B v2 standalone research: fixed UTC bar-start minute 45, complete market-state aggregation.\n"
                << "--partition development|calibration --output-dir NEW_EXTERNAL_DIRECTORY\n"
                << "Replay origin 2010; warmup 2048; calibration embargo 4; no 2025.\n"
                << "Calendar UNVERIFIED. Statistical estimation NOT_EXECUTED. --help never accesses a database.\n";
            return 0;
        }
        if (output.empty()) throw std::invalid_argument("--output-dir required");
        const auto root = std::filesystem::canonical(EA_RESEARCH_LIVE_ROOT);
        const auto snapshot = std::filesystem::canonical(EA_RESEARCH_SNAPSHOT_ROOT);
        output = std::filesystem::weakly_canonical(std::filesystem::absolute(output));
        if (Within(output, root) || Within(output, snapshot) || std::filesystem::exists(output))
            throw std::invalid_argument("research output must be a new directory outside repository and immutable snapshot");
        S::Provenance provenance;
        provenance.gitCommit = Environment("GIT_COMMIT");
        provenance.sourceSHA256 = Environment("EA_RESEARCH_SOURCE_SHA256");
        provenance.executableSHA256 = Environment("EA_RESEARCH_EXECUTABLE_SHA256");
        provenance.configurationFileSHA256 = Environment("EA_RESEARCH_CONFIG_SHA256");
        provenance.compiler = Environment("EA_RESEARCH_COMPILER");
        provenance.libraryVersion = Environment("EA_RESEARCH_LIBRARIES");
        provenance.sourceManifest = Environment("EA_RESEARCH_SOURCE_MANIFEST");
        provenance.sourceArchive = Environment("EA_RESEARCH_SOURCE_ARCHIVE");
        provenance.executable = Environment("EA_RESEARCH_EXECUTABLE");
        provenance.configurationFile = snapshot / "Scripts/tg4_analysis_config.frozen_v1.conf";
        if (S::FileHash(provenance.sourceManifest) != provenance.sourceSHA256 ||
            S::FileHash(provenance.executable) != provenance.executableSHA256 ||
            S::FileHash(provenance.configurationFile) != provenance.configurationFileSHA256)
            throw std::runtime_error("immutable build provenance mismatch, before database access");
        auto config = EA::TG4::LoadConfigurationFile(provenance.configurationFile);
        config.fibonacci = S::HistoricalEvaluator::ResearchFibonacci(config.fibonacci);
        config.fibonacci.absolutePriceTolerance = EA::TG4::EffectiveFibonacciAbsolutePriceTolerance(config, provenance.sourceSymbol);
        const auto [start, end] = S::Bounds(protocol.partition);
        const EA::TG4::TemporalRange range{S::kOrigin, start, end, end};
        EA::TG4::ValidateTemporalRange(range);
        pqxx::connection connection{EA::RuntimeDatabaseConnection::ForexConnectionString() + " application_name=fib_trendline_v2_read_only"};
        pqxx::read_transaction transaction{connection};
        transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
        const auto preflight = EA::TG4::HistoricalMarketDataRepository::Preflight(transaction, provenance.sourceSymbol, config, range);
        if (!preflight.rowCount || preflight.duplicateTimestampCount)
            throw std::runtime_error("empty or duplicate canonical source; study aborted");
        provenance.preflightRows = preflight.rowCount;
        S::ArtifactWriter writer(output);
        S::HistoricalEvaluator evaluator(protocol, EA::TG1B::CalibrationConfiguration(config.referenceBarScale),
            config.fibonacci, config.geometry, config.behavior, EA::TG4::ConfigurationFingerprint(config), writer.GetSinks());
        EA::TG4::HistoricalMarketDataRepository::StreamCanonicalCandles(transaction, provenance.sourceSymbol, config, range,
            [&](const EA::TG1A::Candle& c) { evaluator.AddCompletedBar(c); });
        evaluator.Finalize(); transaction.commit();
        writer.Complete(evaluator.Engine(), config, provenance);
        std::cout << "processing_complete candles=" << evaluator.Engine().GetAudit().candles
                  << " decisions=" << evaluator.Engine().GetAudit().scheduled
                  << " statistical_estimation=NOT_EXECUTED coverage=UNVERIFIED\n";
        return 0;
    }
    catch (const std::exception& error)
    { std::cerr << "fib_trendline_v2_error: " << error.what() << '\n'; return 1; }
}
