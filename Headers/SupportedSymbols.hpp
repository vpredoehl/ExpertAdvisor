#pragma once

#include <string>
#include <vector>

namespace EA::SupportedSymbols
{

inline const std::vector<std::string>& TrainingSymbols()
{
    static const std::vector<std::string> symbols{
        "eurusdrmp",
        "gbpusdrmp",
        "usdcadrmp",
        "usdjpyrmp",
        "audusdrmp",
        "audcadrmp"
    };
    return symbols;
}

} // namespace EA::SupportedSymbols
