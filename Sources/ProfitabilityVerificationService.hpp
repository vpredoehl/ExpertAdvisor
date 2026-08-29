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

int RunCampaignShadowRankingCommand(
    const std::string& connectionString,
    long long rankingSnapshotId,
    const std::vector<double>& shadowWeights,
    std::ostream& output,
    std::ostream& errors);

int RunCampaignProfitabilityCalibrationCommand(
    const std::string& connectionString,
    long long rankingSnapshotId,
    std::ostream& output,
    std::ostream& errors);

int RunCampaignProfitabilityTemporalValidationCommand(
    const std::string& connectionString,
    std::ostream& output,
    std::ostream& errors);

int RunCampaignProfitabilityForwardValidationPrecommitCommand(
    const std::string& connectionString,
    long long rankingSnapshotId,
    const std::string& expectedOutcomeStart,
    const std::string& expectedOutcomeEnd,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ProfitabilityVerification
