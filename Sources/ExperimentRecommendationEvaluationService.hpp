#pragma once

#include "ExperimentRecommendationEvaluationRepository.hpp"

#include <iosfwd>
#include <optional>
#include <string>

namespace EA::ExperimentRecommendation
{

struct RecommendationEvaluationCommandRequest
{
    RecommendationEvaluationPolicy policy;
    std::optional<long long> recommendationScanId;
    std::optional<long long> recommendationId;
    int limit = 100;
    bool dryRun = false;
};

int RunEvaluateExperimentRecommendationsCommand(
    const std::string& connectionString,
    const RecommendationEvaluationCommandRequest& request,
    std::ostream& output,
    std::ostream& errors);
int RunListExperimentRecommendationEvaluationsCommand(
    const std::string& connectionString,
    const RecommendationEvaluationFilters& filters,
    std::ostream& output);
int RunExperimentRecommendationEvaluationStatusCommand(
    const std::string& connectionString,
    long long evaluationResultId,
    std::ostream& output);
int RunExplainExperimentRecommendationEvaluationCommand(
    const std::string& connectionString,
    long long evaluationResultId,
    std::ostream& output);
int RunListExperimentRecommendationEvaluationRunsCommand(
    const std::string& connectionString,
    int limit,
    std::ostream& output);
int RunExperimentRecommendationEvaluationRunStatusCommand(
    const std::string& connectionString,
    long long evaluationRunId,
    std::ostream& output);

} // namespace EA::ExperimentRecommendation
