#pragma once

#include <iosfwd>
#include <string>

namespace EA::ExperimentRecommendation
{

int RunExecuteApprovedRecommendationConversionProposalCommand(
    const std::string& connectionString,
    long long proposalId,
    std::ostream& output,
    std::ostream& errors);

int RunRecommendationConversionExecutionStatusCommand(
    const std::string& connectionString,
    long long proposalId,
    std::ostream& output);

} // namespace EA::ExperimentRecommendation
