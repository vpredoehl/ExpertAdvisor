#pragma once

#include "AuthoritativeEconomicEventCandidate.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace EA::EconomicCalendar
{

// parserType is one of fomc_statement, fomc_minutes, or beige_book.  The
// declared type is checked against the first-party artifact rather than used
// as an unchecked classification hint.
AuthoritativeEconomicEventCandidate
ParseFederalReserveEconomicReleaseArtifact(
    const std::string& artifact,
    const std::string& canonicalSourceUrl,
    const std::string& parserType);

// Loads the shared version-1 local manifest, verifies all declared hashes and
// provenance, and emits exactly one raw publication per manifest entry.
std::vector<AuthoritativeEconomicEventCandidate>
LoadFederalReserveEconomicReleaseManifest(
    const std::filesystem::path& manifestPath);

} // namespace EA::EconomicCalendar
