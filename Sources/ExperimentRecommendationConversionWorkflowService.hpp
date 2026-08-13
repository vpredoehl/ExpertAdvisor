#pragma once

#include "ExperimentRecommendationConversionWorkflow.hpp"

#include <optional>
#include <ostream>
#include <string>

namespace EA::ExperimentRecommendation
{

int RunRecommendationConversionWorkflowCommand(
    const std::string& connectionString,
    long long proposalId,
    std::ostream& output,
    std::ostream& errors);

int RunListRecommendationConversionWorkflowsCommand(
    const std::string& connectionString,
    std::optional<RecommendationConversionWorkflowState> state,
    int limit,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
