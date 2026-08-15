#pragma once

#include "ExperimentRecommendationCandidateGenerator.hpp"
#include "ExperimentRecommendationReview.hpp"
#include "ExperimentRecommendationScoring.hpp"

#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

struct RecommendationGenerationCommandRequest
{
    RecommendationPolicy policy;
    std::optional<std::string> symbol;
    std::optional<int> predictionHorizon;
    std::optional<long long> sourceExperimentId;
    std::optional<int> requestedMaximum;
};

struct RecommendationListCommandRequest
{
    std::optional<std::string> status;
    std::optional<std::string> symbol;
    std::optional<int> predictionHorizon;
    std::optional<long long> recommendationScanId;
    int limit = 100;
};

struct RecommendationScoringCommandRequest
{
    RecommendationScoringPolicy policy;
    std::optional<std::string> status;
    std::optional<std::string> symbol;
    std::optional<int> predictionHorizon;
    std::optional<long long> recommendationScanId;
    std::optional<long long> recommendationId;
    std::optional<int> requestedLimit;
};

struct RecommendationScoreListCommandRequest
{
    std::optional<long long> scoreRunId;
    std::optional<long long> recommendationId;
    std::optional<std::string> symbol;
    std::optional<int> predictionHorizon;
    std::optional<double> minimumScore;
    int limit = 100;
};

struct RecommendationReviewCommandRequest
{
    long long recommendationId = -1;
    RecommendationReviewRequest review;
};

struct RecommendationReviewListCommandRequest
{
    std::optional<long long> recommendationId;
    std::optional<RecommendationReviewAction> action;
    int limit = 100;
};

struct RecommendationSourceSelectionRecord
{
    RecommendationSource source;
    std::string groupKey;
    int rankWithinGroup = 0;
};

struct RecommendationSourceSelectionResult
{
    std::vector<RecommendationSourceSelectionRecord> selected;
    std::vector<std::pair<long long, std::string>> skipped;
};

// Pure deterministic source grouping used by the persistence orchestration.
RecommendationSourceSelectionResult SelectRecommendationSources(
    const RecommendationPolicy& policy,
    std::vector<RecommendationSource> eligibleSources);

// Percent-encode one machine-record text value. NULL is reserved for absent
// optionals by the record format; a concrete "NULL" value is encoded fully.
std::string RecommendationMachineText(const std::string& value);

// Preserve ordinary human-readable text while rendering terminal control
// bytes visibly so review provenance cannot inject additional output lines or
// terminal escape sequences.
std::string RecommendationHumanText(const std::string& value);

int RunGenerateExperimentRecommendationsCommand(
    const std::string& connectionString,
    const RecommendationGenerationCommandRequest& request,
    std::ostream& output,
    std::ostream& errors);
int RunListExperimentRecommendationsCommand(
    const std::string& connectionString,
    const RecommendationListCommandRequest& request,
    std::ostream& output);
int RunExperimentRecommendationStatusCommand(
    const std::string& connectionString,
    long long recommendationId,
    std::ostream& output);
int RunListExperimentRecommendationScansCommand(
    const std::string& connectionString,
    int limit,
    std::ostream& output);
int RunExperimentRecommendationScanStatusCommand(
    const std::string& connectionString,
    long long scanId,
    std::ostream& output);
int RunScoreExperimentRecommendationsCommand(
    const std::string& connectionString,
    const RecommendationScoringCommandRequest& request,
    std::ostream& output,
    std::ostream& errors);
int RunListExperimentRecommendationScoresCommand(
    const std::string& connectionString,
    const RecommendationScoreListCommandRequest& request,
    std::ostream& output);
int RunExperimentRecommendationScoreStatusCommand(
    const std::string& connectionString,
    long long scoreId,
    std::ostream& output);
int RunListExperimentRecommendationScoreRunsCommand(
    const std::string& connectionString,
    int limit,
    std::ostream& output);
int RunExperimentRecommendationScoreRunStatusCommand(
    const std::string& connectionString,
    long long scoreRunId,
    std::ostream& output);
int RunExplainExperimentRecommendationScoreCommand(
    const std::string& connectionString,
    long long scoreId,
    std::ostream& output);
int RunExperimentRecommendationReviewCommand(
    const std::string& connectionString,
    const RecommendationReviewCommandRequest& request,
    std::ostream& output,
    std::ostream& errors);
int RunListExperimentRecommendationReviewsCommand(
    const std::string& connectionString,
    const RecommendationReviewListCommandRequest& request,
    std::ostream& output);
int RunExperimentRecommendationReviewStatusCommand(
    const std::string& connectionString,
    long long reviewEventId,
    std::ostream& output);
int RunExperimentRecommendationReviewHistoryCommand(
    const std::string& connectionString,
    long long recommendationId,
    std::ostream& output);

} // namespace EA::ExperimentRecommendation
