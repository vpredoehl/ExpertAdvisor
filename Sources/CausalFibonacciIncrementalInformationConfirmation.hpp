#pragma once

// Isolated one-time confirmation boundary.  This reader deliberately does not
// reuse ParsePre2025Row: development and confirmation are admitted through
// separate partition-first parsers, and are only joined at evaluation.

#include "CausalFibonacciIncrementalInformationAnalysis.hpp"

#include <chrono>

namespace EA::CausalFibonacciIncrementalInformation::Confirmation {

namespace A = Analysis;

inline constexpr std::string_view kRunnerIdentity =
    "causal-fibonacci-layout9-confirmation-2025-v1";
inline constexpr std::string_view kFrozenExtractionManifestSha256 =
    "864879e0a8ce444da684af402fc70dcb7622e29c4f602bd0983e83458a4abc90";
inline constexpr std::string_view kFrozenExtractionRowsSha256 =
    "e3644fc742f1c970d993e6410fb11cfdd228c916f8c044891cf9321d1ce2e39d";
inline constexpr std::string_view kFrozenPre2025ManifestSha256 =
    "4e05ac7137784a3eb57c415f97fa86197266013c5b707172871a506bdd5d3ea0";
inline constexpr std::string_view kFrozenPre2025ChecksumsSha256 =
    "84c229faa734605ba0da582d55143137c06d30f7143335542ce789b3df11f267";
inline constexpr std::string_view kFrozenPre2025RunnerIdentity =
    "causal-fibonacci-layout9-pre2025-artifact-analysis-v2";
inline constexpr std::string_view kFrozenPre2025CodeCommit =
    "222598c9655bd3398aa70bd7909bca1e871be35d";

struct VerificationContract {
    std::string extractionProtocolId = std::string(kV1SourceArtifactProtocolId);
    std::string extractionProtocolSha256 = std::string(kV1SourceArtifactProtocolSha256);
    std::string extractionManifestSha256 = std::string(kFrozenExtractionManifestSha256);
    std::string extractionRowsSha256 = std::string(kFrozenExtractionRowsSha256);
    std::string pre2025ManifestSha256 = std::string(kFrozenPre2025ManifestSha256);
    std::string pre2025ChecksumsSha256 = std::string(kFrozenPre2025ChecksumsSha256);
    std::string pre2025RunnerIdentity = std::string(kFrozenPre2025RunnerIdentity);
    std::string pre2025CodeCommit = std::string(kFrozenPre2025CodeCommit);
};

inline double JsonNumberField(const std::string& contents, std::string_view name)
{
    const std::string needle = "\"" + std::string(name) + "\":";
    const auto start = contents.find(needle);
    if (start == std::string::npos) throw std::invalid_argument("fibonacci_confirmation_manifest_missing:" + std::string(name));
    const auto valueStart = start + needle.size();
    const auto valueEnd = contents.find_first_of(",}\n\r", valueStart);
    if (valueEnd == std::string::npos) throw std::invalid_argument("fibonacci_confirmation_manifest_malformed:" + std::string(name));
    const std::string value = contents.substr(valueStart, valueEnd - valueStart);
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (end != value.c_str() + value.size() || !std::isfinite(parsed))
        throw std::invalid_argument("fibonacci_confirmation_manifest_invalid_number:" + std::string(name));
    return parsed;
}

inline std::string ReadFile(const std::filesystem::path& path, const char* reason)
{
    std::ifstream input(path);
    if (!input) throw std::invalid_argument(std::string(reason) + ":" + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}

inline ArtifactProvenance ExtractionProvenance(const std::filesystem::path& directory)
{
    const std::string manifest = ReadFile(directory / "manifest.json", "fibonacci_confirmation_extraction_manifest_unreadable");
    return {JsonStringField(manifest,"protocol_id"), JsonStringField(manifest,"protocol_sha256"), JsonStringField(manifest,"code_commit"),
            JsonStringField(manifest,"economic_calendar_snapshot_id"), JsonStringField(manifest,"economic_calendar_snapshot_sha256"),
            JsonStringField(manifest,"source_adapter_identity"), JsonStringField(manifest,"source_query_identity"), JsonStringField(manifest,"source_price_domain"),
            JsonStringField(manifest,"layout_identity"), JsonStringField(manifest,"baseline_schema_identity"), JsonStringField(manifest,"fibonacci_schema_identity"),
            JsonStringField(manifest,"target_identity"), JsonStringField(manifest,"symbol_range_identity"), JsonStringField(manifest,"warmup_identity")};
}

inline void VerifyChecksumFile(const std::filesystem::path& directory,
                               const std::vector<std::string>& names)
{
    std::ifstream sums(directory / "sha256sums.txt");
    if (!sums) throw std::invalid_argument("fibonacci_confirmation_pre2025_sums_unreadable");
    std::set<std::string> seen;
    std::string digest, filename;
    while (sums >> digest >> filename) {
        const std::string observed = FileSha256(directory / filename);
        if (!seen.insert(filename).second || std::find(names.begin(), names.end(), filename) == names.end() || digest != observed)
            throw std::invalid_argument("fibonacci_confirmation_pre2025_checksum_mismatch:" + filename + ":expected=" + digest + ",observed=" + observed);
    }
    if (seen.size() != names.size())
        throw std::invalid_argument("fibonacci_confirmation_pre2025_checksums_incomplete");
}

inline void VerifyExtractionArtifact(const std::filesystem::path& directory,
                                     const VerificationContract& contract)
{
    VerifyArtifactDirectory(directory);
    const std::string manifest = ReadFile(directory / "manifest.json", "fibonacci_confirmation_extraction_manifest_unreadable");
    if (JsonStringField(manifest, "protocol_id") != contract.extractionProtocolId ||
        JsonStringField(manifest, "protocol_sha256") != contract.extractionProtocolSha256)
        throw std::invalid_argument("fibonacci_confirmation_extraction_protocol_mismatch");
    if (FileSha256(directory / "manifest.json") != contract.extractionManifestSha256)
        throw std::invalid_argument("fibonacci_confirmation_extraction_manifest_hash_mismatch");
    if (FileSha256(directory / "rows.csv") != contract.extractionRowsSha256)
        throw std::invalid_argument("fibonacci_confirmation_extraction_rows_hash_mismatch");
}

inline void VerifyFrozenPre2025ResultDirectory(const std::filesystem::path& directory,
                                                const VerificationContract& contract)
{
    if (FileSha256(directory / "manifest.json") != contract.pre2025ManifestSha256)
        throw std::invalid_argument("fibonacci_confirmation_pre2025_manifest_hash_mismatch");
    if (FileSha256(directory / "sha256sums.txt") != contract.pre2025ChecksumsSha256)
        throw std::invalid_argument("fibonacci_confirmation_pre2025_checksum_identity_mismatch");
    const std::string manifest = ReadFile(directory / "manifest.json", "fibonacci_confirmation_pre2025_manifest_unreadable");
    if (JsonStringField(manifest, "runner_identity") != contract.pre2025RunnerIdentity ||
        JsonStringField(manifest, "protocol_id") != kProtocolId ||
        JsonStringField(manifest, "protocol_sha256") != kProtocolSha256 ||
        JsonStringField(manifest, "input_manifest_sha256") != contract.extractionManifestSha256 ||
        JsonStringField(manifest, "input_rows_sha256") != contract.extractionRowsSha256 ||
        JsonStringField(manifest, "code_commit") != contract.pre2025CodeCommit ||
        JsonStringField(manifest, "confirmation_2025") != "sealed_and_discarded_before_parsing")
        throw std::invalid_argument("fibonacci_confirmation_pre2025_manifest_identity_mismatch");
    if (JsonNumberField(manifest, "lambda") != 1.0 ||
        JsonNumberField(manifest, "lbfgs_max_iterations") != 4000.0 ||
        JsonNumberField(manifest, "lbfgs_gradient_infinity_tolerance") != 1e-8 ||
        JsonNumberField(manifest, "lbfgs_relative_objective_tolerance") != 1e-12)
        throw std::invalid_argument("fibonacci_confirmation_pre2025_solver_settings_mismatch");
    const std::vector<std::string> names{
        "associations.csv", "conditional_incremental.csv", "coverage_degeneracy.csv",
        "cross_symbol_equal_summary.csv", "fibonacci_ledgers.csv", "manifest.json",
        "monthly_deltas.csv", "reconstruction.csv", "structural_ledgers.csv"};
    VerifyChecksumFile(directory, names);
}

struct Options {
    std::filesystem::path artifactDirectory;
    std::filesystem::path frozenPre2025ResultDirectory;
    std::filesystem::path outputDirectory;
    std::string codeCommit;
    bool explicitlyAuthorized = false;
    bool noRefit = false;
    VerificationContract verification;
};

inline std::optional<A::ParsedRow> ParseDevelopmentReconstructionRow(std::string_view line)
{
    constexpr std::size_t kPartitionField = 4;
    const auto partition = A::ParsedPartition(A::FieldAt(line, kPartitionField));
    if (!partition) throw std::invalid_argument("fibonacci_confirmation_unknown_partition");
    if (*partition != Partition::Development) return std::nullopt;
    A::ParsedRow row;
    row.partition = *partition;
    row.symbol = std::string(A::FieldAt(line, 1));
    row.timestamp = A::Decimal<std::int64_t>(A::FieldAt(line, 2), "timestamp");
    row.ordinal = A::Decimal<std::uint64_t>(A::FieldAt(line, 3), "ordinal");
    if (PartitionFor(row.timestamp) != row.partition)
        throw std::invalid_argument("fibonacci_confirmation_partition_timestamp_mismatch");
    row.h4 = {A::Decimal<int>(A::FieldAt(line, 5), "h4_class"), A::DecimalFloat(A::FieldAt(line, 8), "h4_terminal_return"), A::DecimalBool(A::FieldAt(line, 9), "h4_eligible")};
    row.h6 = {A::Decimal<int>(A::FieldAt(line, 11), "h6_class"), A::DecimalFloat(A::FieldAt(line, 14), "h6_terminal_return"), A::DecimalBool(A::FieldAt(line, 15), "h6_eligible")};
    for (std::size_t i = 0; i < kBaselineWidth; ++i) row.baseline[i] = A::DecimalFloat(A::FieldAt(line, 17 + i), "baseline");
    for (std::size_t i = 0; i < kFibonacciWidth; ++i) row.fibonacci[i] = A::DecimalFloat(A::FieldAt(line, 17 + kBaselineWidth + i), "fibonacci");
    return row;
}

inline std::optional<A::ParsedRow> ParseConfirmationEvaluationRow(std::string_view line)
{
    constexpr std::size_t kPartitionField = 4;
    const auto partition = A::ParsedPartition(A::FieldAt(line, kPartitionField));
    if (!partition) throw std::invalid_argument("fibonacci_confirmation_unknown_partition");
    if (*partition != Partition::Confirmation2025) return std::nullopt;
    A::ParsedRow row;
    row.partition = *partition;
    row.symbol = std::string(A::FieldAt(line, 1));
    row.timestamp = A::Decimal<std::int64_t>(A::FieldAt(line, 2), "timestamp");
    row.ordinal = A::Decimal<std::uint64_t>(A::FieldAt(line, 3), "ordinal");
    if (PartitionFor(row.timestamp) != row.partition)
        throw std::invalid_argument("fibonacci_confirmation_partition_timestamp_mismatch");
    row.h4 = {A::Decimal<int>(A::FieldAt(line, 5), "h4_class"), A::DecimalFloat(A::FieldAt(line, 8), "h4_terminal_return"), A::DecimalBool(A::FieldAt(line, 9), "h4_eligible")};
    row.h6 = {A::Decimal<int>(A::FieldAt(line, 11), "h6_class"), A::DecimalFloat(A::FieldAt(line, 14), "h6_terminal_return"), A::DecimalBool(A::FieldAt(line, 15), "h6_eligible")};
    for (std::size_t i = 0; i < kBaselineWidth; ++i) row.baseline[i] = A::DecimalFloat(A::FieldAt(line, 17 + i), "baseline");
    for (std::size_t i = 0; i < kFibonacciWidth; ++i) row.fibonacci[i] = A::DecimalFloat(A::FieldAt(line, 17 + kBaselineWidth + i), "fibonacci");
    return row;
}

struct InputRows { std::map<std::string, std::vector<A::ParsedRow>> development, confirmation; };

inline void ValidateConfirmationRowsHeader(const std::string& header)
{
    A::ValidateRowsHeader(header);
    if (A::FieldAt(header, 5) != "h4_class" || A::FieldAt(header, 11) != "h6_class" ||
        A::FieldAt(header, 17 + kBaselineWidth + kFibonacciWidth - 1) != "fibonacci_22")
        throw std::invalid_argument("fibonacci_confirmation_unexpected_horizon_or_schema");
    try { (void)A::FieldAt(header, 17 + kBaselineWidth + kFibonacciWidth); }
    catch (const std::invalid_argument&) { return; }
    throw std::invalid_argument("fibonacci_confirmation_unexpected_horizon_or_schema");
}

inline InputRows LoadIsolatedRows(const std::filesystem::path& artifact)
{
    std::ifstream input(artifact / "rows.csv");
    if (!input) throw std::runtime_error("fibonacci_confirmation_rows_unreadable");
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("fibonacci_confirmation_rows_header_missing");
    ValidateConfirmationRowsHeader(line);
    InputRows result;
    while (std::getline(input, line)) {
        if (const auto development = ParseDevelopmentReconstructionRow(line))
            result.development[development->symbol].push_back(std::move(*development));
        if (const auto confirmation = ParseConfirmationEvaluationRow(line))
            result.confirmation[confirmation->symbol].push_back(std::move(*confirmation));
    }
    const std::array<std::string, 6> expected{{"audcadrmp","audusdrmp","eurusdrmp","gbpusdrmp","usdcadrmp","usdjpyrmp"}};
    for (const auto& symbol : expected) {
        if (!result.development.contains(symbol) || !result.confirmation.contains(symbol))
            throw std::invalid_argument("fibonacci_confirmation_symbol_universe_mismatch:" + symbol);
    }
    if (result.development.size() != expected.size() || result.confirmation.size() != expected.size())
        throw std::invalid_argument("fibonacci_confirmation_unexpected_symbol");
    return result;
}

inline std::vector<A::ParsedRow> JoinDevelopmentAndConfirmation(const std::vector<A::ParsedRow>& development,
                                                                 const std::vector<A::ParsedRow>& confirmation)
{
    std::vector<A::ParsedRow> joined;
    joined.reserve(development.size() + confirmation.size());
    joined.insert(joined.end(), development.begin(), development.end());
    joined.insert(joined.end(), confirmation.begin(), confirmation.end());
    return joined;
}

inline void WriteConfirmationStructuralRows(std::ofstream& out, const std::string& symbol,
                                            const std::vector<A::ParsedRow>& rows)
{
    for (const bool h6 : {false, true}) {
        std::array<A::StateAccumulator, 3> state{};
        for (const auto& row : rows) {
            const auto& target = h6 ? row.h6 : row.h4;
            if (target.eligible)
                state[static_cast<std::size_t>(StateFor(Row{row.symbol,row.timestamp,row.ordinal,row.baseline,row.fibonacci,{},{}}))].Add(target);
        }
        for (std::size_t index = 0; index < state.size(); ++index) {
            const auto& value = state[index]; double mean = 0;
            for (const double item : value.returns) mean += item;
            if (!value.returns.empty()) mean /= value.returns.size();
            out << symbol << ",confirmation_2025," << (h6 ? "H6" : "H4") << ',' << index << ',' << value.rows << ','
                << value.classes[0] << ',' << value.classes[1] << ',' << value.classes[2] << ',' << A::Number(mean) << ',' << A::Number(Median(value.returns)) << '\n';
        }
    }
}

inline void WriteConfirmationFibonacciLedgers(std::ofstream& out, const std::string& symbol,
                                              const std::vector<A::ParsedRow>& joined)
{
    for (std::size_t feature = 0; feature < kFibonacciWidth; ++feature) {
        for (const bool h6 : {false, true}) {
            const auto edges = A::FrozenDevelopmentBinEdges(joined, feature, h6);
            if (edges.empty()) { out << symbol << ",confirmation_2025," << (h6 ? "H6" : "H4") << ',' << feature << ",,0,0,0,0,,,insufficient_distinct_development_values_after_tied_edge_collapse\n"; continue; }
            std::vector<A::StateAccumulator> bins(edges.size());
            for (const auto& row : joined) if (row.partition == Partition::Confirmation2025) {
                const auto& target = h6 ? row.h6 : row.h4;
                if (target.eligible) bins[A::FrozenBinIndex(edges, row.fibonacci[feature])].Add(target);
            }
            for (std::size_t bin = 0; bin < bins.size(); ++bin) {
                const auto& value = bins[bin]; double mean = 0;
                for (const double item : value.returns) mean += item;
                if (!value.returns.empty()) mean /= value.returns.size();
                out << symbol << ",confirmation_2025," << (h6 ? "H6" : "H4") << ',' << feature << ',' << bin << ',' << A::Number(edges[bin]) << ',' << value.rows << ',' << value.classes[0] << ',' << value.classes[1] << ',' << value.classes[2] << ',' << A::Number(mean) << ',' << A::Number(Median(value.returns)) << ",\n";
            }
        }
    }
}

inline void WriteConfirmationCoverage(std::ofstream& out, const std::string& symbol,
                                      const std::vector<A::ParsedRow>& confirmation,
                                      const A::BinaryRowsModel& binary,
                                      const std::array<A::RidgeModel, kFibonacciWidth - 1>& ridge)
{
    for (std::size_t feature = 0; feature < kFibonacciWidth; ++feature) {
        std::size_t scaleValid=0, finite=0, nonzero=0;
        for (const auto& row : confirmation) { scaleValid += row.fibonacci[0] > .5f; finite += std::isfinite(row.fibonacci[feature]); nonzero += row.fibonacci[feature] != 0.0f; }
        const bool validity = feature == 0, count = A::IsFibonacciCountColumn(feature);
        const double variance = validity ? std::numeric_limits<double>::quiet_NaN() : A::FibonacciTargetVariance(confirmation, Partition::Confirmation2025, feature);
        const std::string reason = validity ? A::ReconstructionUnavailableReason(binary, confirmation.size()) : A::ReconstructionUnavailableReason(ridge[feature - 1], confirmation.size(), variance);
        out << symbol << ",confirmation_2025," << feature << ',' << (validity ? std::to_string(scaleValid) : "") << ',' << (validity ? std::to_string(confirmation.size()) : "") << ',' << (validity ? A::Number(confirmation.empty() ? std::numeric_limits<double>::quiet_NaN() : static_cast<double>(scaleValid)/confirmation.size()) : "") << ',' << (count ? std::to_string(nonzero) : "") << ',' << (count ? std::to_string(confirmation.size()) : "") << ',' << (count ? A::Number(confirmation.empty() ? std::numeric_limits<double>::quiet_NaN() : static_cast<double>(nonzero)/confirmation.size()) : "") << ',' << finite << ',' << confirmation.size() << ',' << A::Number(variance) << ',' << A::Csv(reason) << '\n';
    }
}

inline void WriteConfirmationAssociations(std::ofstream& out, const std::string& symbol,
                                          const std::vector<A::ParsedRow>& confirmation)
{
    std::vector<Row> converted; converted.reserve(confirmation.size());
    for (const auto& row : confirmation) converted.push_back({row.symbol,row.timestamp,row.ordinal,row.baseline,row.fibonacci,{},{}});
    const auto matrix = CompleteAssociationMatrix(converted, symbol, Partition::Confirmation2025);
    for (std::size_t base = 0; base < kBaselineWidth; ++base)
        out << symbol << ",confirmation_2025,0," << base << ",point_biserial," << A::Number(matrix.scaleValidityPointBiserial[base]) << ',' << matrix.nearestBaselineProxy[0] << '\n';
    for (std::size_t fib = 1; fib < kFibonacciWidth; ++fib) for (std::size_t base = 0; base < kBaselineWidth; ++base)
        out << symbol << ",confirmation_2025," << fib << ',' << base << ",spearman," << A::Number(matrix.spearman[fib-1][base]) << ',' << matrix.nearestBaselineProxy[fib] << '\n';
}

inline void WriteConfirmationEqualSymbolSummary(std::ofstream& out, const std::vector<A::ConditionalRecord>& records)
{
    out << "partition,horizon,metric,symbols,available,positive,zero_or_negative,unavailable,median,min,max\n";
    for (const bool h6 : {false, true}) for (const bool brier : {false, true}) {
        std::vector<double> values; std::string symbols; std::size_t positive=0, nonpositive=0, unavailable=0;
        for (const auto& record : records) if (record.h6 == h6) { if (!symbols.empty()) symbols += ';'; symbols += record.symbol; if (!record.available) { ++unavailable; continue; } const double value = brier ? record.brier : record.logLoss; values.push_back(value); value > 0 ? ++positive : ++nonpositive; }
        out << "confirmation_2025," << (h6 ? "H6" : "H4") << ',' << (brier ? "brier" : "log_loss") << ',' << A::Csv(symbols) << ',' << values.size() << ',' << positive << ',' << nonpositive << ',' << unavailable << ',' << A::Number(Median(values)) << ',' << A::Number(values.empty() ? std::numeric_limits<double>::quiet_NaN() : *std::min_element(values.begin(), values.end())) << ',' << A::Number(values.empty() ? std::numeric_limits<double>::quiet_NaN() : *std::max_element(values.begin(), values.end())) << '\n';
    }
}

inline void VerifyConfirmationOutputDirectory(const std::filesystem::path& directory)
{
    const std::string manifest = ReadFile(directory / "manifest.json", "fibonacci_confirmation_output_manifest_unreadable");
    if (JsonStringField(manifest, "runner_identity") != kRunnerIdentity || JsonStringField(manifest, "protocol_id") != kProtocolId || JsonStringField(manifest, "protocol_sha256") != kProtocolSha256 || JsonStringField(manifest, "confirmation_partition") != "confirmation_2025" || JsonStringField(manifest, "no_refit") != "enforced" || JsonStringField(manifest, "explicit_authorization") != "asserted" || JsonNumberField(manifest, "lambda") != 1.0 || JsonNumberField(manifest, "lbfgs_max_iterations") != 4000.0 || JsonNumberField(manifest, "lbfgs_gradient_infinity_tolerance") != 1e-8 || JsonNumberField(manifest, "lbfgs_relative_objective_tolerance") != 1e-12)
        throw std::invalid_argument("fibonacci_confirmation_output_manifest_identity_mismatch");
    const std::vector<std::string> names{"associations.csv","conditional_incremental.csv","completion.txt","coverage_degeneracy.csv","cross_symbol_equal_summary.csv","fibonacci_ledgers.csv","manifest.json","monthly_deltas.csv","reconstruction.csv","structural_ledgers.csv"};
    VerifyChecksumFile(directory, names);
    std::set<std::string> expected(names.begin(), names.end()); expected.insert("sha256sums.txt");
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        if (!entry.is_regular_file() || !expected.contains(entry.path().filename().string()))
            throw std::invalid_argument("fibonacci_confirmation_output_unexpected_artifact");
    if (ReadFile(directory / "completion.txt", "fibonacci_confirmation_completion_marker_unreadable") != "FIBONACCI_INCREMENTAL_CONFIRMATION_2025_COMPLETE\n")
        throw std::invalid_argument("fibonacci_confirmation_completion_marker_invalid");
}

inline std::filesystem::path StagingDirectory(const std::filesystem::path& output)
{
    return output.string() + ".incomplete-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
}

inline void Run(const Options& options)
{
    if (options.artifactDirectory.empty() || options.frozenPre2025ResultDirectory.empty() || options.outputDirectory.empty() || options.codeCommit.empty())
        throw std::invalid_argument("fibonacci_confirmation_required_argument_missing");
    if (!options.explicitlyAuthorized) throw std::invalid_argument("fibonacci_confirmation_requires_explicit_authorization");
    if (!options.noRefit) throw std::invalid_argument("fibonacci_confirmation_refit_forbidden");
    ConfirmationAuthorization authorization;
    authorization.explicitlyAuthorized = options.explicitlyAuthorized;
    authorization.frozenPre2025ResultAvailable = !options.frozenPre2025ResultDirectory.empty();
    authorization.noRefit = options.noRefit;
    authorization.artifactDirectory = options.artifactDirectory;
    authorization.developmentFitProvenance = ExtractionProvenance(options.artifactDirectory);
    RequireConfirmation2025(authorization);
    if (std::filesystem::exists(options.outputDirectory)) throw std::invalid_argument("fibonacci_confirmation_refuses_existing_output_directory");
    VerifyFrozenProtocolDocument("docs/phases/target-generation/FibonacciExtensions/FIBONACCI_LAYOUT9_INCREMENTAL_INFORMATION_PROTOCOL.md");
    VerifyExtractionArtifact(options.artifactDirectory, options.verification);
    VerifyFrozenPre2025ResultDirectory(options.frozenPre2025ResultDirectory, options.verification);
    const auto inputs = LoadIsolatedRows(options.artifactDirectory);
    const auto staging = StagingDirectory(options.outputDirectory);
    std::filesystem::create_directories(staging);
    try {
        std::ofstream structural(staging/"structural_ledgers.csv"), fibonacci(staging/"fibonacci_ledgers.csv"), reconstruction(staging/"reconstruction.csv"), conditional(staging/"conditional_incremental.csv"), monthly(staging/"monthly_deltas.csv"), association(staging/"associations.csv"), coverage(staging/"coverage_degeneracy.csv"), summary(staging/"cross_symbol_equal_summary.csv");
        if (!structural || !fibonacci || !reconstruction || !conditional || !monthly || !association || !coverage || !summary) throw std::runtime_error("fibonacci_confirmation_result_open_failed");
        structural << "symbol,partition,horizon,structural_state,rows,down,neutral,up,mean_terminal_log_return,median_terminal_log_return\n";
        fibonacci << "symbol,partition,horizon,fibonacci_column,bin,upper_edge,rows,down,neutral,up,mean_terminal_log_return,median_terminal_log_return,unavailable_reason\n";
        reconstruction << "symbol,partition,fibonacci_column,diagnostic,available,unavailable_reason,rmse,r_squared,log_loss,brier\n";
        conditional << "symbol,partition,horizon,subset,rows,baseline_log_loss,augmented_log_loss,delta_log_loss,baseline_brier,augmented_brier,delta_brier,baseline_accuracy,augmented_accuracy,delta_accuracy,available,unavailable_reason\n";
        monthly << "symbol,partition,horizon,metric,calendar_month,rows,mean_delta,median_delta,p10_delta,p90_delta\n";
        association << "symbol,partition,fibonacci_column,baseline_column,association_type,value,nearest_baseline_proxy\n";
        coverage << "symbol,partition,fibonacci_column,scale_valid_numerator,scale_valid_denominator,scale_valid_prevalence,nonzero_count_numerator,nonzero_count_denominator,nonzero_count_prevalence,finite_value_count,finite_value_denominator,target_variance,reconstruction_unavailable_reason\n";
        std::vector<A::ConditionalRecord> records;
        for (const auto& [symbol, development] : inputs.development) {
            const auto& confirmation = inputs.confirmation.at(symbol);
            const auto joined = JoinDevelopmentAndConfirmation(development, confirmation);
            const auto transform = A::FitDevelopmentInputTransform(joined);
            const auto developmentAll = A::EligibleRows(joined, Partition::Development, [](const auto&) { return A::ParsedTarget{0, 0, true}; });
            const auto binary = A::FitBinaryRows(joined, developmentAll, transform.baseline);
            std::array<A::RidgeModel, kFibonacciWidth - 1> ridge{};
            for (std::size_t feature = 1; feature < kFibonacciWidth; ++feature) ridge[feature - 1] = A::FitRidge(joined, developmentAll, transform.baseline, feature);
            WriteConfirmationStructuralRows(structural, symbol, confirmation);
            WriteConfirmationFibonacciLedgers(fibonacci, symbol, joined);
            WriteConfirmationCoverage(coverage, symbol, confirmation, binary, ridge);
            const std::vector<std::size_t> confirmationAll = A::EligibleRows(joined, Partition::Confirmation2025, [](const auto&) { return A::ParsedTarget{0, 0, true}; });
            const auto binaryMetric = A::EvaluateBinaryRows(binary, joined, confirmationAll, transform.baseline);
            reconstruction << symbol << ",confirmation_2025,0,scale_validity_logistic," << binaryMetric.available << ',' << A::Csv(binaryMetric.unavailableReason) << ",,," << A::Number(binaryMetric.logLoss) << ',' << A::Number(binaryMetric.brier) << '\n';
            for (std::size_t feature = 1; feature < kFibonacciWidth; ++feature) { const auto metric = A::EvaluateRidge(ridge[feature-1], joined, confirmationAll, transform.baseline, feature); reconstruction << symbol << ",confirmation_2025," << feature << ",ridge," << metric.available << ',' << A::Csv(metric.unavailableReason) << ',' << A::Number(metric.rmse) << ',' << A::Number(metric.rSquared) << ",,\n"; }
            WriteConfirmationAssociations(association, symbol, confirmation);
            for (const bool h6 : {false, true}) {
                const auto label = [h6](const A::ParsedRow& row) -> const A::ParsedTarget& { return h6 ? row.h6 : row.h4; };
                const auto developmentRows = A::EligibleRows(joined, Partition::Development, label);
                const auto baseline = A::FitMultiRows(joined, developmentRows, transform.baseline, label);
                const auto augmented = A::FitMultiRows(joined, developmentRows, transform.augmented, label);
                const auto selected = A::EligibleRows(joined, Partition::Confirmation2025, label);
                const auto baselineMetric = A::EvaluateMultiRows(baseline, joined, selected, transform.baseline, label);
                const auto augmentedMetric = A::EvaluateMultiRows(augmented, joined, selected, transform.augmented, label);
                const bool available = baselineMetric.available && augmentedMetric.available && baselineMetric.rowLoss.size() == augmentedMetric.rowLoss.size();
                const std::string reason = available ? "" : (!baselineMetric.available ? baselineMetric.unavailableReason : augmentedMetric.unavailableReason);
                const double deltaLoss = available ? baselineMetric.logLoss - augmentedMetric.logLoss : std::numeric_limits<double>::quiet_NaN();
                const double deltaBrier = available ? baselineMetric.brier - augmentedMetric.brier : std::numeric_limits<double>::quiet_NaN();
                const double deltaAccuracy = available ? baselineMetric.accuracy - augmentedMetric.accuracy : std::numeric_limits<double>::quiet_NaN();
                conditional << symbol << ",confirmation_2025," << (h6 ? "H6" : "H4") << ",all," << selected.size() << ',' << A::Number(baselineMetric.logLoss) << ',' << A::Number(augmentedMetric.logLoss) << ',' << A::Number(deltaLoss) << ',' << A::Number(baselineMetric.brier) << ',' << A::Number(augmentedMetric.brier) << ',' << A::Number(deltaBrier) << ',' << A::Number(baselineMetric.accuracy) << ',' << A::Number(augmentedMetric.accuracy) << ',' << A::Number(deltaAccuracy) << ',' << available << ',' << A::Csv(reason) << '\n';
                records.push_back({symbol, Partition::Confirmation2025, h6, deltaLoss, deltaBrier, available});
                std::vector<std::size_t> event; for (const auto index : selected) if (A::EventState(joined[index])) event.push_back(index);
                const auto eventBaseline = A::EvaluateMultiRows(baseline, joined, event, transform.baseline, label);
                const auto eventAugmented = A::EvaluateMultiRows(augmented, joined, event, transform.augmented, label);
                const bool eventAvailable = eventBaseline.available && eventAugmented.available && eventBaseline.rowLoss.size() == eventAugmented.rowLoss.size();
                conditional << symbol << ",confirmation_2025," << (h6 ? "H6" : "H4") << ",event_state," << event.size() << ',' << A::Number(eventBaseline.logLoss) << ',' << A::Number(eventAugmented.logLoss) << ',' << A::Number(eventAvailable ? eventBaseline.logLoss-eventAugmented.logLoss : std::numeric_limits<double>::quiet_NaN()) << ',' << A::Number(eventBaseline.brier) << ',' << A::Number(eventAugmented.brier) << ',' << A::Number(eventAvailable ? eventBaseline.brier-eventAugmented.brier : std::numeric_limits<double>::quiet_NaN()) << ',' << A::Number(eventBaseline.accuracy) << ',' << A::Number(eventAugmented.accuracy) << ',' << A::Number(eventAvailable ? eventBaseline.accuracy-eventAugmented.accuracy : std::numeric_limits<double>::quiet_NaN()) << ',' << eventAvailable << ',' << A::Csv(eventAvailable ? "" : (!eventBaseline.available ? eventBaseline.unavailableReason : eventAugmented.unavailableReason)) << '\n';
                if (available) { std::vector<double> loss, brier; for (std::size_t i=0; i<baselineMetric.rowLoss.size(); ++i) { loss.push_back(baselineMetric.rowLoss[i]-augmentedMetric.rowLoss[i]); brier.push_back(baselineMetric.rowBrier[i]-augmentedMetric.rowBrier[i]); } for (const auto& [metric, values] : std::array<std::pair<const char*,std::vector<double>>,2>{{{"log_loss",loss},{"brier",brier}}}) for (const auto& block : A::Monthly(joined, selected, values)) monthly << symbol << ",confirmation_2025," << (h6 ? "H6" : "H4") << ',' << metric << ',' << block.calendarMonth << ',' << block.rows << ',' << A::Number(block.mean) << ',' << A::Number(block.median) << ',' << A::Number(block.p10) << ',' << A::Number(block.p90) << '\n'; }
            }
        }
        WriteConfirmationEqualSymbolSummary(summary, records);
        structural.close(); fibonacci.close(); reconstruction.close(); conditional.close(); monthly.close(); association.close(); coverage.close(); summary.close();
        const auto extractionManifest = FileSha256(options.artifactDirectory / "manifest.json");
        const auto extractionRows = FileSha256(options.artifactDirectory / "rows.csv");
        const auto preManifest = FileSha256(options.frozenPre2025ResultDirectory / "manifest.json");
        const auto preSums = FileSha256(options.frozenPre2025ResultDirectory / "sha256sums.txt");
        std::ofstream manifest(staging / "manifest.json");
        manifest << "{\n\"runner_identity\":\"" << kRunnerIdentity << "\",\n\"protocol_id\":\"" << kProtocolId << "\",\n\"protocol_sha256\":\"" << kProtocolSha256 << "\",\n\"extraction_manifest_sha256\":\"" << extractionManifest << "\",\n\"extraction_rows_sha256\":\"" << extractionRows << "\",\n\"frozen_pre2025_manifest_sha256\":\"" << preManifest << "\",\n\"frozen_pre2025_checksums_sha256\":\"" << preSums << "\",\n\"code_commit\":\"" << options.codeCommit << "\",\n\"explicit_authorization\":\"asserted\",\n\"no_refit\":\"enforced\",\n\"confirmation_partition\":\"confirmation_2025\",\n\"lambda\":1.0,\n\"lbfgs_max_iterations\":4000,\n\"lbfgs_gradient_infinity_tolerance\":1e-8,\n\"lbfgs_relative_objective_tolerance\":1e-12\n}\n";
        manifest.close();
        std::ofstream completion(staging / "completion.txt"); completion << "FIBONACCI_INCREMENTAL_CONFIRMATION_2025_COMPLETE\n"; completion.close();
        const std::vector<std::string> names{"associations.csv","conditional_incremental.csv","completion.txt","coverage_degeneracy.csv","cross_symbol_equal_summary.csv","fibonacci_ledgers.csv","manifest.json","monthly_deltas.csv","reconstruction.csv","structural_ledgers.csv"};
        std::ofstream sums(staging / "sha256sums.txt"); for (const auto& name : names) sums << FileSha256(staging / name) << "  " << name << '\n'; sums.close();
        VerifyConfirmationOutputDirectory(staging);
        std::filesystem::rename(staging, options.outputDirectory);
    } catch (...) { std::error_code ignored; std::filesystem::remove_all(staging, ignored); throw; }
}

} // namespace EA::CausalFibonacciIncrementalInformation::Confirmation
