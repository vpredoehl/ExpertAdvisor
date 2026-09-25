#include "CausalFibonacciStructuralDiagnostic.hpp"
#include "TG4HistoricalMarketDataRepository.hpp"
#include "SupportedSymbols.hpp"

#include <pqxx/pqxx>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace
{
constexpr std::string_view kBaselineCommit = "91b87aa4de2695eeeac3629326aa160d53b6da71";
constexpr std::int64_t kStart = 1'262'304'000;
constexpr std::int64_t kEnd = 1'767'225'600;

void RequireOpen(const std::ofstream& stream, const std::filesystem::path& path)
{
    if (!stream) throw std::runtime_error("could not write " + path.string());
}
}

int main(int argc, char* argv[])
{
    try
    {
        if (argc != 3 || std::string_view(argv[1]) != "--output-dir")
            throw std::invalid_argument("usage: --output-dir DIR");
        const std::filesystem::path output = argv[2];
        std::filesystem::create_directories(output);
        const EA::TG4::EvaluationConfiguration configuration =
            EA::TG4::LoadConfigurationFile("Scripts/tg4_analysis_config.frozen_v1.conf");
        const EA::TG4::TemporalRange range{kStart, kStart, kEnd, kEnd};
        EA::TG4::ValidateTemporalRange(range);
        std::vector<std::string> symbols = EA::SupportedSymbols::TrainingSymbols();
        std::sort(symbols.begin(), symbols.end());

        const auto csvPath = output / "per_symbol.csv";
        std::ofstream csv(csvPath); RequireOpen(csv, csvPath);
        csv << EA::FibonacciDiagnostic::Summary::CsvHeader() << '\n';
        const auto distancePath = output / "distance_distributions.csv";
        std::ofstream distances(distancePath); RequireOpen(distances, distancePath);
        distances << EA::FibonacciDiagnostic::Summary::DistanceCsvHeader() << '\n';
        std::vector<EA::FibonacciDiagnostic::Summary> results;
        for (const std::string& symbol : symbols)
        {
            std::cerr << "FIBONACCI_STRUCTURAL_DIAGNOSTIC_SYMBOL_START symbol=" << symbol << '\n';
            pqxx::connection connection;
            pqxx::read_transaction transaction{connection};
            transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY");
            EA::FibonacciDiagnostic::StructuralDiagnostic diagnostic(symbol, configuration, range.scoreStart);
            EA::TG4::HistoricalMarketDataRepository::StreamCanonicalCandles(
                transaction, symbol, configuration, range,
                [&diagnostic](const EA::TG1A::Candle& candle) { diagnostic.AddCompletedBar(candle); });
            results.push_back(diagnostic.Finalize());
            if (results.back().bars == 0)
                throw std::runtime_error("zero measurement bars for " + symbol);
            csv << results.back().ToCsv() << '\n';
            distances << results.back().DistanceCsvRows();
            csv.flush();
            std::cerr << "FIBONACCI_STRUCTURAL_DIAGNOSTIC_SYMBOL_COMPLETE symbol=" << symbol
                      << ",bars=" << results.back().bars << '\n';
            transaction.commit();
        }
        EA::FibonacciDiagnostic::Summary combined;
        combined.symbol = "combined";
        for (const auto& result : results) combined.Merge(result);
        csv << combined.ToCsv() << '\n';
        distances << combined.DistanceCsvRows();

        const auto manifestPath = output / "manifest.json";
        std::ofstream manifest(manifestPath); RequireOpen(manifest, manifestPath);
        manifest << "{\n  \"schema\": \"causal-fibonacci-model-feature-structural-diagnostic-v1\",\n"
                 << "  \"source_baseline_commit\": \"" << kBaselineCommit << "\",\n"
                 << "  \"configuration_identity\": \"causal-fibonacci-h1-h2-symmetric-structural-diagnostic-v1\",\n"
                 << "  \"source_configuration\": \"" << configuration.name << "\",\n"
                 << "  \"source_configuration_schema\": \"" << configuration.configurationSchema << "\",\n"
                 << "  \"symbols_in_actual_order\": [";
        for (std::size_t i = 0; i < results.size(); ++i)
            manifest << (i ? ", " : "") << "\"" << results[i].symbol << "\"";
        manifest << "],\n  \"requested_warmup_range\": null,\n"
                 << "  \"warmup_policy\": \"none; repository stream begins at measurement_start, so no pre-measurement bars are replayed\",\n"
                 << "  \"requested_measurement_range\": \"[2010-01-01T00:00:00Z,2026-01-01T00:00:00Z)\",\n"
                 << "  \"actual_measurement_timestamps_per_symbol\": [\n";
        for (std::size_t i = 0; i < results.size(); ++i)
            manifest << "    {\"symbol\": \"" << results[i].symbol
                     << "\", \"first_measurement_epoch\": " << results[i].firstMeasurementTimestamp
                     << ", \"last_measurement_epoch\": " << results[i].lastMeasurementTimestamp
                     << "}" << (i + 1 == results.size() ? "\n" : ",\n");
        manifest << "  ],\n"
                 << "  \"actual_combined_first_measurement_epoch\": " << combined.firstMeasurementTimestamp << ",\n"
                 << "  \"actual_combined_last_measurement_epoch\": " << combined.lastMeasurementTimestamp << ",\n"
                 << "  \"read_only\": true,\n  \"outcome_blind\": true,\n"
                 << "  \"maxABAgeBars\": " << configuration.fibonacci.maxABAgeBars << ",\n"
                 << "  \"maxActiveABStructures\": " << configuration.fibonacci.maxActiveABStructures << ",\n"
                 << "  \"both_directions\": \"symmetric_directional_diagnostic_hypothesis\",\n"
                 << "  \"normalization\": \"direction_sign*(level-close)/max(completed_bar_wilder_atr14_raw,canonical_symbol_pip_size)\",\n"
                 << "  \"quantiles\": \"exact value-frequency nearest-rank for counts and ages; mergeable signed logarithmic bins for finite normalized distances, zero exact; bin gamma=2*log(1.001), geometric-centre representatives have relative value error <=0.1%\",\n"
                 << "  \"reproduction_command\": \"bash Scripts/run_causal_fibonacci_structural_diagnostic.sh --output-dir Artifacts/causal-fibonacci-model-feature-structural-diagnostic-v1\"\n}\n";

        const auto summaryPath = output / "summary.md";
        std::ofstream summary(summaryPath); RequireOpen(summary, summaryPath);
        summary << "# Outcome-blind causal Fibonacci structural diagnostic\n\n"
                << "The replay uses only TG1A causal fractal confirmation and the actual bounded TG3 A/B tracker. It does not instantiate a target tracker or consume target resolution, outcome, censoring, profitability, or model state. `per_symbol.csv` has scalar columns and a true combined row built by merging sufficient statistics; `distance_distributions.csv` keeps geometric and 20-bar event-relevant populations separate. Touch, beyond, and rejection state flags intentionally overlap.\n";
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "FIBONACCI_STRUCTURAL_DIAGNOSTIC_ERROR " << exception.what() << '\n';
        return 1;
    }
}
