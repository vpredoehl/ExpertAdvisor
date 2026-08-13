#pragma once

#include <iosfwd>
#include <string>

namespace EA::ExperimentRecommendation
{

int RunActivateRecommendationConversionExecutionCommand(
    const std::string& connectionString,
    long long executionId,
    std::ostream& output,
    std::ostream& errors);

int RunRecommendationConversionActivationStatusCommand(
    const std::string& connectionString,
    long long activationId,
    std::ostream& output);

} // namespace EA::ExperimentRecommendation
