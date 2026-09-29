#pragma once

#include "../Headers/PocketProspectiveDerivedAnalyzer.hpp"

#include <iostream>
#include <optional>
#include <string_view>

namespace EA::Pocket::Prospective::Derived::Cli
{
inline std::optional<int> TryRun(int argc, const char* const argv[])
{
    if (argc < 2) return std::nullopt;
    const std::string_view command(argv[1]);
    try {
        if (command == "--verify-pocket-prospective-derived-report") {
            if (argc != 3) throw std::invalid_argument("usage: --verify-pocket-prospective-derived-report REPORT_DIRECTORY");
            Writer::Verify(argv[2]); std::cout << "POCKET_DERIVED_REPORT_VERIFIED\n"; return 0;
        }
        if (command != "--derive-pocket-prospective-report") return std::nullopt;
        std::filesystem::path source, output; std::string git, executable;
        for (int index=2; index<argc; ++index) {
            const std::string_view option(argv[index]);
            const auto value=[&]() -> std::string { if(++index>=argc) throw std::invalid_argument("POCKET_DERIVED_OPTION_VALUE_MISSING"); return argv[index]; };
            if(option=="--artifact-dir") source=value(); else if(option=="--output-dir") output=value();
            else if(option=="--git-commit") git=value(); else if(option=="--executable-identity") executable=value();
            else throw std::invalid_argument("POCKET_DERIVED_UNKNOWN_OPTION:"+std::string(option));
        }
        if(source.empty()||output.empty()) throw std::invalid_argument("POCKET_DERIVED_ARTIFACT_AND_OUTPUT_REQUIRED");
        Analyze(source,output,git,executable); std::cout << "POCKET_DERIVED_REPORT_PUBLISHED\n"; return 0;
    } catch(const std::exception& error) { std::cerr << "POCKET_DERIVED_ERROR " << error.what() << '\n'; return 1; }
}
} // namespace EA::Pocket::Prospective::Derived::Cli
