#pragma once

// Operational boundary for the frozen research harness.  Extraction is a
// separate, explicit read-only command; it only generates a source artifact.

#include "CausalFibonacciIncrementalInformation.hpp"
#include "CausalFibonacciIncrementalInformationExtraction.hpp"
#include "CausalFibonacciIncrementalInformationAnalysis.hpp"
#include "CausalFibonacciIncrementalInformationConfirmation.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace EA::CausalFibonacciIncrementalInformation::Cli {
inline std::string DefaultForexConnection()
{
    const char* host = std::getenv("FOREX_DB_HOST");
    const char* database = std::getenv("FOREX_DB_NAME");
    return "hostaddr=" + std::string(host && *host ? host : "127.0.0.1") +
        " gssencmode=disable user=pqxx dbname=" +
        std::string(database && *database ? database : "forex");
}

inline std::string DefaultLstmConnection()
{
    const char* host = std::getenv("LSTM_DB_HOST");
    const char* database = std::getenv("LSTM_DB_NAME");
    return "hostaddr=" + std::string(host && *host ? host : "127.0.0.1") +
        " gssencmode=disable user=pqxx dbname=" +
        std::string(database && *database ? database : "LSTM");
}

inline std::optional<int> TryRun(int argc, const char* const argv[])
{
    if (argc < 2) return std::nullopt;
    const std::string_view command(argv[1]);
    if (command != "--fibonacci-incremental-information-verify-artifact" &&
        command != "--fibonacci-incremental-information-extract" &&
        command != "--fibonacci-incremental-information-analyze-pre2025" &&
        command != "--fibonacci-incremental-information-confirm-2025" &&
        command != "--fibonacci-incremental-information-diagnose-pre2025-multinomial" &&
        command != "--fibonacci-incremental-information-qualify-solver-budget")
        return std::nullopt;
    try {
        VerifyFrozenProtocolDocument("docs/phases/target-generation/FibonacciExtensions/FIBONACCI_LAYOUT9_INCREMENTAL_INFORMATION_PROTOCOL.md");
        if (command == "--fibonacci-incremental-information-verify-artifact") {
            if (argc != 3) throw std::invalid_argument("usage: --fibonacci-incremental-information-verify-artifact ARTIFACT_DIRECTORY");
            VerifyArtifactDirectory(std::filesystem::path(argv[2]));
            std::cout << "FIBONACCI_INCREMENTAL_ARTIFACT_VERIFIED protocol=" << kProtocolId << '\n';
            return 0;
        }
        if (command == "--fibonacci-incremental-information-diagnose-pre2025-multinomial") {
            if ((argc != 4 && argc != 5) || std::string_view(argv[2]) != "--artifact-dir" ||
                (argc == 5 && std::string_view(argv[4]) != "--continuation-5000"))
                throw std::invalid_argument("usage: --fibonacci-incremental-information-diagnose-pre2025-multinomial --artifact-dir ARTIFACT_DIRECTORY [--continuation-5000]");
            const auto mode=argc==5?Analysis::AudcadH4BaselineMultinomialDiagnosticMode::Continuation5000:Analysis::AudcadH4BaselineMultinomialDiagnosticMode::Frozen250;
            Analysis::RunAudcadH4BaselineMultinomialDiagnostic(std::filesystem::path(argv[3]), std::cerr, mode);
            return 0;
        }
        if (command == "--fibonacci-incremental-information-qualify-solver-budget") {
            if (argc != 4 || std::string_view(argv[2]) != "--artifact-dir")
                throw std::invalid_argument("usage: --fibonacci-incremental-information-qualify-solver-budget --artifact-dir ARTIFACT_DIRECTORY");
            Analysis::RunSolverBudgetQualification(std::filesystem::path(argv[3]), std::cerr);
            return 0;
        }
        if (command == "--fibonacci-incremental-information-analyze-pre2025") {
            Analysis::Options options;
            for (int index = 2; index < argc; ++index) {
                const std::string_view option(argv[index]);
                if (option == "--artifact-dir" && index + 1 < argc) options.artifactDirectory = argv[++index];
                else if (option == "--output-dir" && index + 1 < argc) options.outputDirectory = argv[++index];
                else if (option == "--code-commit" && index + 1 < argc) options.codeCommit = argv[++index];
                else throw std::invalid_argument("unknown or incomplete Fibonacci pre-2025 analysis option: " + std::string(option));
            }
            Analysis::Run(options);
            std::cout << "FIBONACCI_INCREMENTAL_PRE2025_ANALYSIS_COMPLETE protocol=" << kProtocolId
                      << ",confirmation_2025=sealed\n";
            return 0;
        }
        if (command == "--fibonacci-incremental-information-confirm-2025") {
            Confirmation::Options options;
            for (int index = 2; index < argc; ++index) {
                const std::string_view option(argv[index]);
                if (option == "--artifact-dir" && index + 1 < argc) options.artifactDirectory = argv[++index];
                else if (option == "--frozen-pre2025-result-dir" && index + 1 < argc) options.frozenPre2025ResultDirectory = argv[++index];
                else if (option == "--output-dir" && index + 1 < argc) options.outputDirectory = argv[++index];
                else if (option == "--code-commit" && index + 1 < argc) options.codeCommit = argv[++index];
                else if (option == "--authorize-fibonacci-confirmation-2025") options.explicitlyAuthorized = true;
                else if (option == "--enforce-no-refit") options.noRefit = true;
                else throw std::invalid_argument("unknown or incomplete Fibonacci confirmation option: " + std::string(option));
            }
            Confirmation::Run(options);
            std::cout << "FIBONACCI_INCREMENTAL_CONFIRMATION_2025_COMPLETE protocol=" << kProtocolId
                      << ",confirmation_2025=evaluated_once_by_explicit_authorization\n";
            return 0;
        }
        Extraction::Options options;
        options.forexConnectionString = DefaultForexConnection();
        options.lstmConnectionString = DefaultLstmConnection();
        for (int index = 2; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (option == "--output-dir" && index + 1 < argc) options.outputDirectory = argv[++index];
            else if (option == "--code-commit" && index + 1 < argc) options.codeCommit = argv[++index];
            else if (option == "--economic-calendar-snapshot-id" && index + 1 < argc) options.calendarSnapshot.snapshotId = std::stoll(argv[++index]);
            else if (option == "--economic-calendar-snapshot-sha256" && index + 1 < argc) options.calendarSnapshot.contentHash = argv[++index];
            else if (option == "--forex-connection" && index + 1 < argc) options.forexConnectionString = argv[++index];
            else if (option == "--lstm-connection" && index + 1 < argc) options.lstmConnectionString = argv[++index];
            else throw std::invalid_argument("unknown or incomplete Fibonacci extraction option: " + std::string(option));
        }
        Extraction::Extract(options);
        std::cout << "FIBONACCI_INCREMENTAL_EXTRACTION_COMPLETE protocol=" << kProtocolId
                  << ",read_only=true,diagnostics_run=false,confirmation_2025_evaluated=false\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FIBONACCI_INCREMENTAL_ARTIFACT_ERROR " << error.what() << '\n';
        return 1;
    }
}
} // namespace EA::CausalFibonacciIncrementalInformation::Cli
