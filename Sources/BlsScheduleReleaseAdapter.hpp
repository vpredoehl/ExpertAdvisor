#pragma once

#include "AuthoritativeEconomicEventCandidate.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace EA::EconomicCalendar
{

AuthoritativeEconomicEventCandidate ParseBlsScheduleReleaseArtifact(
    const std::string& artifact,
    const std::string& canonicalSourceUrl);

std::vector<AuthoritativeEconomicEventCandidate>
LoadBlsScheduleReleaseManifest(const std::filesystem::path& manifestPath);

} // namespace EA::EconomicCalendar
