#pragma once

#include "../Headers/PocketConfirmationFreeze.hpp"

#include <iostream>
#include <optional>
#include <string_view>

namespace EA::Pocket::Prospective::Confirmation::Cli
{
inline std::optional<int> TryRun(int argc, const char* const argv[])
{
    if (argc < 2) return std::nullopt;
    const std::string_view command(argv[1]);
    try {
        if (command == "--print-pocket-confirmation-freeze") {
            if (argc != 2) throw std::invalid_argument("usage: --print-pocket-confirmation-freeze");
            std::cout << CanonicalConfigurationText() << ValidationSummary(FrozenConfiguration()); return 0;
        }
        if (command != "--validate-pocket-confirmation-configuration") return std::nullopt;
        std::filesystem::path config, output;
        for (int index=2; index<argc; ++index) {
            const std::string_view option(argv[index]);
            if (++index >= argc) throw std::invalid_argument("POCKET_CONFIRMATION_OPTION_VALUE_MISSING");
            if (option == "--config") config=argv[index]; else if (option == "--output-dir") output=argv[index];
            else throw std::invalid_argument("POCKET_CONFIRMATION_UNKNOWN_OPTION:"+std::string(option));
        }
        if (config.empty() || output.empty()) throw std::invalid_argument("POCKET_CONFIRMATION_CONFIG_AND_OUTPUT_REQUIRED");
        const Configuration frozen=LoadAndValidateConfiguration(config); ValidateAbsentOutputTarget(output);
        std::cout << ValidationSummary(frozen); return 0;
    } catch (const std::exception& error) {
        std::cerr << "POCKET_CONFIRMATION_ERROR " << error.what() << '\n'; return 1;
    }
}
} // namespace EA::Pocket::Prospective::Confirmation::Cli
