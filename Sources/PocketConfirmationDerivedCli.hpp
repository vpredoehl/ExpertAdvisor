#pragma once

#include "../Headers/PocketConfirmationDerived.hpp"

#include <iostream>
#include <optional>

namespace EA::Pocket::Prospective::Confirmation::DerivedReport::Cli
{
inline std::optional<int> TryRun(int argc, const char* const argv[])
{
    if(argc<2)return std::nullopt; const std::string_view command(argv[1]);
    try {
        if(command=="--verify-pocket-confirmation-derived-report") { if(argc!=3) throw std::invalid_argument("usage: --verify-pocket-confirmation-derived-report REPORT_DIRECTORY"); Verify(argv[2]);std::cout<<"POCKET_CONFIRMATION_DERIVED_REPORT_VERIFIED\n";return 0; }
        if(command!="--derive-pocket-confirmation-report")return std::nullopt;
        std::filesystem::path source,output;std::string git,executable;
        for(int index=2;index<argc;++index){const std::string_view option(argv[index]);if(++index>=argc)throw std::invalid_argument("POCKET_CONFIRMATION_DERIVED_OPTION_VALUE_MISSING");if(option=="--artifact-dir")source=argv[index];else if(option=="--output-dir")output=argv[index];else if(option=="--git-commit")git=argv[index];else if(option=="--executable-identity")executable=argv[index];else throw std::invalid_argument("POCKET_CONFIRMATION_DERIVED_UNKNOWN_OPTION:"+std::string(option));}
        if(source.empty()||output.empty())throw std::invalid_argument("POCKET_CONFIRMATION_DERIVED_ARTIFACT_AND_OUTPUT_REQUIRED");Analyze(source,output,git,executable);std::cout<<"POCKET_CONFIRMATION_DERIVED_REPORT_PUBLISHED\n";return 0;
    } catch(const std::exception& error){std::cerr<<"POCKET_CONFIRMATION_DERIVED_ERROR "<<error.what()<<'\n';return 1;}
}
} // namespace EA::Pocket::Prospective::Confirmation::DerivedReport::Cli
