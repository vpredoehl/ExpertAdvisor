#pragma once

#include "AuthoritativeEconomicEventCandidate.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace EA::EconomicCalendar
{

std::string NormalizeDolEtaReleaseId(
    const std::string& releaseId);

AuthoritativeEconomicEventCandidate ParseDolEtaWeeklyClaimsArtifact(
    const std::string& artifact,
    const std::string& canonicalSourceUrl);

// Loads a deterministic local-only manifest, verifies every declared SHA-256
// digest, and parses one raw WEEKLY_CLAIMS occurrence per entry.
std::vector<AuthoritativeEconomicEventCandidate>
LoadDolEtaWeeklyClaimsManifest(
    const std::filesystem::path& manifestPath);

} // namespace EA::EconomicCalendar
