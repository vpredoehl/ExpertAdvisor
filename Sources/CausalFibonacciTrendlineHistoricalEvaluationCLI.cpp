#include "CausalFibonacciTrendlineHistoricalArtifact.hpp"
#include "RuntimeDatabaseConnection.hpp"
#include "TG4HistoricalMarketDataRepository.hpp"
#include <chrono>
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
std::int64_t ParseUtcBoundary(std::string_view text)
{
    // A single canonical spelling avoids process-local TZ, offset ambiguity,
    // normalized invalid dates, fractional seconds and leap-second handling.
    if (text.size() != 20 || text[4] != '-' || text[7] != '-' || text[10] != 'T' ||
        text[13] != ':' || text[16] != ':' || text[19] != 'Z')
        throw std::invalid_argument("bounds require YYYY-MM-DDTHH:MM:SSZ UTC timestamps");
    const auto digits = [&](std::size_t first, std::size_t count)
    {
        int value = 0;
        for (std::size_t i = first; i < first + count; ++i)
        {
            if (text[i] < '0' || text[i] > '9')
                throw std::invalid_argument("bounds require YYYY-MM-DDTHH:MM:SSZ UTC timestamps");
            value = value * 10 + text[i] - '0';
        }
        return value;
    };
    const std::chrono::year_month_day date{std::chrono::year{digits(0, 4)},
        std::chrono::month{static_cast<unsigned>(digits(5, 2))},
        std::chrono::day{static_cast<unsigned>(digits(8, 2))}};
    const int hour = digits(11, 2), minute = digits(14, 2), second = digits(17, 2);
    if (!date.ok() || hour > 23 || minute > 59 || second > 59)
        throw std::invalid_argument("invalid UTC calendar boundary");
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::sys_days{date}.time_since_epoch()).count()
        + hour * 3600 + minute * 60 + second;
}
}
int main(int argc, char** argv)
{
    try
    {
        S::Protocol protocol; std::filesystem::path output, auditOutput;
        bool help = false, havePartition = false, boundedDevelopment = false, haveTier = false;
        S::EpisodePublication tier = S::EpisodePublication::Summary;
        std::optional<std::int64_t> startUtc, endUtc;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--help") help = true;
            else if (arg == "--bounded-development" && !boundedDevelopment) boundedDevelopment = true;
            else if (arg == "--start-utc" && i + 1 < argc && !startUtc) startUtc = ParseUtcBoundary(argv[++i]);
            else if (arg == "--end-utc" && i + 1 < argc && !endUtc) endUtc = ParseUtcBoundary(argv[++i]);
            else if (arg == "--history-shadow-audit-dir")
            {
                if (i + 1 >= argc || !auditOutput.empty() || argv[i + 1][0] == '-' || argv[i + 1][0] == '\0')
                    throw std::invalid_argument("unknown, duplicate or incomplete option: " + arg);
                auditOutput = argv[++i];
            }
            else if (arg == "--output-dir" && i + 1 < argc && output.empty()) output = argv[++i];
            else if (arg == "--episode-publication" && i + 1 < argc && !haveTier)
            {
                haveTier = true; const std::string value = argv[++i];
                if (value == "summary") tier = S::EpisodePublication::Summary;
                else if (value == "full") tier = S::EpisodePublication::Full;
                else throw std::invalid_argument("episode publication requires summary|full");
            }
            else if (arg == "--partition" && i + 1 < argc && !havePartition)
            {
                havePartition = true; const std::string p = argv[++i];
                if (p == "development") protocol.partition = S::Partition::Development;
                else if (p == "calibration") protocol.partition = S::Partition::Calibration;
                else throw std::invalid_argument("only development/calibration allowed; 2025 disabled before detector/database access");
            }
            else throw std::invalid_argument("unknown, duplicate or incomplete option: " + arg);
        }
        const auto [partitionStart, partitionEnd] = S::Bounds(protocol.partition);
        S::ScoreRange score{partitionStart, partitionEnd};
        if (boundedDevelopment)
        {
            if (protocol.partition != S::Partition::Development)
                throw std::invalid_argument("bounded mode permits development only; calibration and 2025 disabled");
            if (!startUtc || !endUtc)
                throw std::invalid_argument("bounded development requires both --start-utc and --end-utc");
            if (*startUtc >= *endUtc)
                throw std::invalid_argument("bounded development requires start < end");
            if (*startUtc < S::kOrigin || *startUtc >= S::kCalibrationStart ||
                *endUtc < S::kOrigin || *endUtc >= S::kCalibrationStart)
                throw std::invalid_argument("both bounds must lie within development [2010-01-01,2020-01-01) UTC");
            score = {*startUtc, *endUtc};
        }
        else if (startUtc || endUtc)
            throw std::invalid_argument("UTC bounds require --bounded-development");
        // ScoreRange purges t+4 >= score.end and rejects replay bars >= end.
        // Consequently every accepted horizon is contained in this input
        // range; never replace kOrigin with score.start or append tail bars.
        const EA::TG4::TemporalRange range{S::kOrigin, score.start, score.end, score.end};
        EA::TG4::ValidateTemporalRange(range);
        if (help)
        {
            std::cout << "Phase 2B v2 standalone research: fixed UTC bar-start minute 45, complete market-state aggregation.\n"
                << "--partition development|calibration --output-dir NEW_EXTERNAL_DIRECTORY\n"
                << "--bounded-development --start-utc YYYY-MM-DDTHH:MM:SSZ --end-utc YYYY-MM-DDTHH:MM:SSZ --output-dir NEW_EXTERNAL_DIRECTORY\n"
                << "--history-shadow-audit-dir NEW_EXTERNAL_DIRECTORY (optional; includes causal decision-geometry counterfactual extension).\n"
                << "--episode-publication summary|full (default summary); artifact format v3, semantic protocol v2.\n"
                << "Deterministic lossless gzip geometry; episode tracking unchanged; summary omits raw episode detail.\n"
                << "episode_publication=" << S::Name(tier) << "\n"
                << "Replay origin 2010; warmup 2048; calibration embargo 4; no 2025.\n"
                << "Bounded ranges are half-open; both bounds strictly inside development; final four-bar horizons are purged at range end.\n"
                << "Calendar UNVERIFIED. Statistical estimation NOT_EXECUTED. --help never accesses a database.\n";
            if (boundedDevelopment)
                std::cout << "bounded_development_plan replay_origin_utc_epoch=" << range.warmupStart
                    << " score_start_utc_epoch=" << score.start << " score_end_exclusive_utc_epoch=" << score.end
                    << " input_end_exclusive_utc_epoch=" << range.outcomeEnd << "\n";
            return 0;
        }
        if (output.empty()) throw std::invalid_argument("--output-dir required");
        const auto root = std::filesystem::canonical(EA_RESEARCH_LIVE_ROOT);
        const auto snapshot = std::filesystem::canonical(EA_RESEARCH_SNAPSHOT_ROOT);
        output = std::filesystem::weakly_canonical(std::filesystem::absolute(output));
        if (Within(output, root) || Within(output, snapshot) || std::filesystem::exists(output))
            throw std::invalid_argument("research output must be a new directory outside repository and immutable snapshot");
        if (!auditOutput.empty())
        {
            auditOutput = std::filesystem::weakly_canonical(std::filesystem::absolute(auditOutput));
            if (Within(auditOutput, root) || Within(auditOutput, snapshot) || Within(auditOutput, output) ||
                Within(output, auditOutput) || std::filesystem::exists(auditOutput))
                throw std::invalid_argument("audit output must be a new separate external directory");
        }
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
        std::unique_ptr<EA::FibonacciResearch::HistoryShadow::Audit> shadowAudit;
        if (!auditOutput.empty())
        {
            EA::FibonacciResearch::HistoryShadow::ValidateFrozenSourceSnapshot(snapshot);
            EA::FibonacciResearch::HistoryShadow::DecisionAuditConfiguration auditConfig{
                S::Hash(S::CanonicalProtocol(protocol,score,EA::TG4::ConfigurationFingerprint(config),{})),score.start,score.end,
                protocol.partition==S::Partition::Calibration};
            shadowAudit = std::make_unique<EA::FibonacciResearch::HistoryShadow::Audit>(config.fibonacci, config.geometry, auditOutput,
                EA::FibonacciResearch::HistoryShadow::Budgets{},EA::FibonacciResearch::HistoryShadow::PublicationHook{},auditConfig);
        }
        EA::FibonacciResearch::HistoryShadow::Observer observer;
        if (shadowAudit) observer = [&](const auto& observation) { shadowAudit->Observe(observation); };
        pqxx::connection connection{EA::RuntimeDatabaseConnection::ForexConnectionString() + " application_name=fib_trendline_v2_read_only"};
        pqxx::read_transaction transaction{connection};
        transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
        const auto preflight = EA::TG4::HistoricalMarketDataRepository::Preflight(transaction, provenance.sourceSymbol, config, range);
        if (!preflight.rowCount || preflight.duplicateTimestampCount)
            throw std::runtime_error("empty or duplicate canonical source; study aborted");
        provenance.preflightRows = preflight.rowCount;
        S::ArtifactWriter writer(output, tier);
        auto sinks=writer.GetSinks();
        if(shadowAudit)
        {
            auto writeDecision=sinks.decision;
            sinks.decision=[&,writeDecision](const S::Decision& d){
                writeDecision(d);
                shadowAudit->ObservePrimaryDecision({d.identity,d.status,d.cohort?S::Name(*d.cohort):"",d.timestamp,d.close.has_value(),
                    d.aggregate.complete,d.aggregate.evaluated,d.aggregate.abs,d.aggregate.lines,
                    d.aggregate.fibNear,d.aggregate.lineNear,d.aggregate.intersection});
            };
        }
        S::HistoricalEvaluator evaluator(protocol, EA::TG1B::CalibrationConfiguration(config.referenceBarScale),
            config.fibonacci, config.geometry, config.behavior, EA::TG4::ConfigurationFingerprint(config), std::move(sinks), score, {}, std::move(observer));
        EA::TG4::HistoricalMarketDataRepository::StreamCanonicalCandles(transaction, provenance.sourceSymbol, config, range,
            [&](const EA::TG1A::Candle& c) { evaluator.AddCompletedBar(c); });
        evaluator.Finalize(); transaction.commit();
        writer.Complete(evaluator.Engine(), config, provenance);
        if (shadowAudit && !shadowAudit->Complete({evaluator.Engine().Fingerprint(), provenance.gitCommit,
            provenance.sourceSHA256, evaluator.Engine().CandleDigest(), output}))
            throw std::runtime_error("primary publication complete; history shadow audit INCOMPLETE: " + shadowAudit->GetSummary().failure);
        std::cout << "processing_complete candles=" << evaluator.Engine().GetAudit().candles
                  << " decisions=" << evaluator.Engine().GetAudit().scheduled
                  << " statistical_estimation=NOT_EXECUTED coverage=UNVERIFIED\n";
        return 0;
    }
    catch (const std::exception& error)
    { std::cerr << "fib_trendline_v2_error: " << error.what() << '\n'; return 1; }
}
