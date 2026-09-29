#pragma once

// The Phase Pocket 4 command is deliberately isolated ahead of all scheduler
// and training CLI parsing.  It has no database write path and its source end
// is fixed at the Phase Pocket 3 preconfirmation firewall.

#include "../Headers/CanonicalMarketDataRange.hpp"
#include "../Headers/HistoricalFxTimestamp.hpp"
#include "../Headers/PocketProspectiveEvaluator.hpp"

#include <pqxx/pqxx>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string_view>

namespace EA::Pocket::Prospective::Cli
{
struct Options final
{
    std::filesystem::path configuration;
    std::filesystem::path output;
    std::optional<std::string> connection;
    std::string gitCommit;
    std::string executableIdentity;
    bool validateOnly = false;
    bool execute = false;
};

inline std::string DefaultConnection()
{
    const char* host = std::getenv("FOREX_DB_HOST");
    const char* database = std::getenv("FOREX_DB_NAME");
    return "hostaddr=" + std::string(host && *host ? host : "127.0.0.1") +
        " gssencmode=disable user=pqxx dbname=" +
        std::string(database && *database ? database : "forex") +
        " application_name=pocket_prospective_read_only_preconfirmation";
}

inline Options Parse(int argc, const char* const argv[])
{
    Options result;
    for (int index = 2; index < argc; ++index)
    {
        const std::string_view option(argv[index]);
        const auto require = [&]() -> std::string {
            if (++index >= argc) throw std::invalid_argument("POCKET_CLI_OPTION_VALUE_MISSING:" + std::string(option));
            return argv[index];
        };
        if (option == "--config") result.configuration = require();
        else if (option == "--output-dir") result.output = require();
        else if (option == "--connection") result.connection = require();
        else if (option == "--git-commit") result.gitCommit = require();
        else if (option == "--executable-identity") result.executableIdentity = require();
        else if (option == "--validate-only") result.validateOnly = true;
        else if (option == "--execute") result.execute = true;
        else throw std::invalid_argument("POCKET_CLI_UNKNOWN_OPTION:" + std::string(option));
    }
    if (result.configuration.empty() || result.output.empty())
        throw std::invalid_argument("POCKET_CLI_CONFIG_AND_OUTPUT_REQUIRED");
    if (result.validateOnly == result.execute)
        throw std::invalid_argument("POCKET_CLI_CHOOSE_EXACTLY_ONE_OF_VALIDATE_OR_EXECUTE");
    if (result.gitCommit.size() != 40 || !std::all_of(result.gitCommit.begin(), result.gitCommit.end(), [](char c) {
            return std::isxdigit(static_cast<unsigned char>(c)); }))
        throw std::invalid_argument("POCKET_CLI_FULL_GIT_COMMIT_REQUIRED");
    if (!IsHexSha256(result.executableIdentity))
        throw std::invalid_argument("POCKET_CLI_EXECUTABLE_SHA256_REQUIRED");
    return result;
}

inline std::int64_t ParseTimestamp(const std::string& source)
{
    PriceTP parsed;
    if (!HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(source, parsed))
        throw std::invalid_argument("POCKET_SOURCE_TIMESTAMP_UNPARSEABLE");
    return parsed.time_since_epoch().count();
}

template <typename Transaction>
inline std::string ReadOnlyBarsStreamQuery(Transaction& transaction,
    const SymbolSource& source)
{
    // This finite upper bound is the firewall.  No preconfirmation command
    // constructs a query that can request a 2025 confirmation/outcome bar.
    const auto end = PriceTP{std::chrono::seconds{kPreconfirmationEnd}};
    // pqxx::stream wraps this SELECT in COPY (...).  It must therefore remain
    // a query expression, with no terminating statement semicolon.
    return CanonicalMarketData::CanonicalFullHistoryThroughCandlestickCte(
        transaction, source.table, end) +
        "SELECT to_char(dt,'YYYY-MM-DD HH24:MI:SS'),open::double precision,high::double precision,"
        "low::double precision,close::double precision FROM bounded ORDER BY dt";
}

inline std::vector<CompletedBar> ReadOnlyBars(pqxx::read_transaction& transaction,
    const SymbolSource& source)
{
    const std::string query = ReadOnlyBarsStreamQuery(transaction, source);
    std::vector<CompletedBar> bars;
    auto stream = transaction.stream<std::string, double, double, double, double>(query);
    for (const auto& [timestamp, open, high, low, close] : stream)
        bars.push_back({ParseTimestamp(timestamp), open, high, low, close});
    return bars;
}

inline std::string DatabaseIdentity(pqxx::read_transaction& transaction)
{
    const pqxx::row row = transaction.exec("SELECT current_database(), current_setting('server_version_num');").one_row();
    const std::string identity = row[0].as<std::string>() + ":postgresql:" + row[1].as<std::string>();
    if (identity.empty()) throw std::invalid_argument("POCKET_SOURCE_PROVENANCE_MISSING");
    return identity;
}

inline std::optional<int> TryRun(int argc, const char* const argv[])
{
    if (argc < 2) return std::nullopt;
    const std::string_view command(argv[1]);
    if (command == "--verify-pocket-prospective-artifact")
    {
        try {
            if (argc != 3) throw std::invalid_argument("usage: --verify-pocket-prospective-artifact ARTIFACT_DIRECTORY");
            ImmutableArtifactWriter::VerifyDirectory(argv[2]);
            std::cout << "POCKET_PROSPECTIVE_ARTIFACT_VERIFIED\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "POCKET_PROSPECTIVE_ERROR " << error.what() << '\n'; return 1;
        }
    }
    if (command != "--pocket-prospective-evaluator") return std::nullopt;
    try {
        const Options options = Parse(argc, argv);
        const RunConfiguration configuration = LoadAndValidateConfiguration(options.configuration);
        VerifyFrozenProtocolDocument();
        if (std::filesystem::exists(options.output))
            throw std::invalid_argument("POCKET_OUTPUT_TARGET_EXISTS_OR_EMPTY");
        pqxx::connection connection(options.connection.value_or(DefaultConnection()));
        pqxx::read_transaction transaction(connection);
        transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
        const std::string database = DatabaseIdentity(transaction);
        std::vector<EvaluatedObservation> records;
        std::vector<SourceAudit> audits;
        std::vector<std::pair<SymbolSource, std::vector<CompletedBar>>> preparedSources;
        for (const SymbolSource& source : configuration.symbols)
        {
            std::vector<CompletedBar> bars = ReadOnlyBars(transaction, source);
            audits.push_back(PreflightCompletedBars(configuration, source.symbol, source.table, database, bars));
            // Warmup is checked without reading a post-firewall outcome: a
            // structurally scored confirmation cannot be admitted until 21
            // contiguous completed bars precede its earliest possible point.
            bool warmupFound = false;
            for (std::size_t i = configuration.warmupBars; i < bars.size(); ++i)
                if (bars[i].timestamp >= configuration.partitions.front().start) {
                    warmupFound = true;
                    for (std::size_t p = i - configuration.warmupBars + 1; p < i; ++p)
                        if (bars[p].timestamp - bars[p - 1].timestamp != configuration.cadenceSeconds)
                            throw std::invalid_argument("POCKET_INSUFFICIENT_OR_DISCONTINUOUS_WARMUP");
                    break;
                }
            if (!warmupFound) throw std::invalid_argument("POCKET_INSUFFICIENT_WARMUP");
            preparedSources.emplace_back(source, std::move(bars));
        }
        transaction.commit();
        std::cout << "POCKET_PRECONFIRMATION_PREFLIGHT study=" << configuration.study
                  << ",configuration_sha256=" << configuration.configurationSha256
                  << ",protocol=" << configuration.protocol << ",detector=" << configuration.detector
                  << ",source_end_exclusive=2025-01-01T00:00:00Z,output=" << options.output.string()
                  << ",outcome_blind=" << (options.validateOnly ? "true" : "false") << '\n';
        if (options.validateOnly) return 0;
        // This begins only after the complete, outcome-blind review line has
        // been emitted above.  Labels receive the already-authorized bounded
        // pre-2025 source vectors, never a live source reader.
        for (const auto& [source, bars] : preparedSources) {
            auto symbolRecords = ReplayCausallyAndLabelBounded(configuration, source.symbol, bars);
            records.insert(records.end(), std::make_move_iterator(symbolRecords.begin()),
                std::make_move_iterator(symbolRecords.end()));
        }
        std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) { return a.identity < b.identity; });
        std::ostringstream provenance;
        provenance << "git=" << options.gitCommit << ";executable_sha256=" << options.executableIdentity
                   << ";database=" << database << ";read_only=true;isolation=repeatable_read";
        for (const SourceAudit& audit : audits)
            provenance << ";" << audit.symbol << ':' << audit.table << ':' << audit.rowCount << ':'
                       << audit.firstTimestamp << ':' << audit.lastTimestamp;
        ImmutableArtifactWriter(options.output).Publish(configuration, provenance.str(), records);
        std::cout << "POCKET_PRECONFIRMATION_EVALUATION_PUBLISHED observations=" << records.size()
                  << ",confirmation_2025_sealed=true\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "POCKET_PROSPECTIVE_ERROR " << error.what() << '\n';
        return 1;
    }
}
} // namespace EA::Pocket::Prospective::Cli
