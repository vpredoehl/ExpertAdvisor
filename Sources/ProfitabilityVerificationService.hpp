#pragma once

#include "ProfitabilityVerification.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace EA::ProfitabilityVerification
{

int RunVerificationCommand(const std::string& connectionString,
                           const std::vector<long long>& experimentIds,
                           std::ostream& output,
                           std::ostream& errors);

int RunCampaignReadinessCommand(const std::string& connectionString,
                                long long rankingSnapshotId,
                                std::ostream& output,
                                std::ostream& errors);

} // namespace EA::ProfitabilityVerification
