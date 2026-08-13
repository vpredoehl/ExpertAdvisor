#pragma once

#include "ExperimentRecommendationEvaluation.hpp"

#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

struct RecommendationEvaluationFilters
{
    std::optional<long long> recommendationScanId;
    std::optional<long long> recommendationId;
    std::optional<RecommendationEvaluationDisposition> disposition;
    int limit = 100;
};

struct RecommendationEvaluationLoadResult
{
    long long recommendationId = -1;
    RecommendationEvaluationInput input;
};

struct RecommendationEvaluationRunRequest
{
    RecommendationEvaluationPolicy policy;
    RecommendationEvaluationFilters filters;
    std::string runIdentityCanonical;
    std::string runIdentityHash;
    std::string evidenceSnapshotCanonical;
    std::string evidenceSnapshotHash;
};

struct RecommendationEvaluationRunBeginResult
{
    long long evaluationRunId = -1;
    bool created = false;
    std::string status;
};

struct RecommendationEvaluationRunCounters
{
    int recommendationsConsidered = 0;
    int recommendationsEvaluated = 0;
    int recommendationsEligible = 0;
    int recommendationsBlocked = 0;
    int evaluationErrors = 0;
};

struct RecommendationEvaluationPersistenceRequest
{
    long long evaluationRunId = -1;
    RecommendationEvaluationInput input;
    RecommendationEvaluationResult result;
};

struct RecommendationEvaluationPersistResult
{
    long long evaluationResultId = -1;
    bool created = false;
};

struct PersistedRecommendationEvaluationSummary
{
    long long evaluationResultId = -1;
    long long evaluationRunId = -1;
    long long recommendationId = -1;
    long long recommendationScanId = -1;
    long long sourceExperimentId = -1;
    std::string sourceSymbol;
    int sourcePredictionHorizon = 0;
    std::string changedParameter;
    std::string recommendationSemanticHash;
    std::string evaluationIdentityHash;
    std::string evaluationPolicyHash;
    int evaluationVersion = 0;
    int evaluatorVersion = 0;
    std::string scoringPolicyHash;
    int scoringVersion = 0;
    RecommendationEligibility eligibility = RecommendationEligibility::ineligible;
    RecommendationEvaluationDisposition disposition =
        RecommendationEvaluationDisposition::invalidPersistedEvidence;
    std::string reasonCode;
    std::string explanation;
    std::optional<double> finalScore;
    int componentCount = 0;
    int missingEvidenceCount = 0;
    int rankingOrdinal = 0;
    std::string createdAt;
};

struct PersistedRecommendationEvaluationDetail
    : PersistedRecommendationEvaluationSummary
{
    std::optional<long long> sourceModelId;
    std::optional<long long> sourceAnalysisId;
    std::string evaluationIdentityCanonical;
    std::string recommendationSemanticCanonical;
    std::string recommendationPolicyCanonical;
    std::string recommendationPolicyHash;
    std::string evidenceCanonical;
    std::string evidenceHash;
    std::optional<double> rawPositiveScore;
    std::optional<double> rawPenaltyScore;
    std::optional<double> rawTotalScore;
    std::vector<RecommendationScoreComponent> components;
};

struct PersistedRecommendationEvaluationRun
{
    long long evaluationRunId = -1;
    std::string status;
    std::string runIdentityHash;
    std::string evaluationPolicyHash;
    int evaluationVersion = 0;
    int evaluatorVersion = 0;
    std::string scoringPolicyHash;
    int scoringVersion = 0;
    std::optional<long long> recommendationScanFilter;
    std::optional<long long> recommendationIdFilter;
    std::optional<int> requestedLimit;
    RecommendationEvaluationRunCounters counters;
    std::string startedAt;
    std::optional<std::string> completedAt;
    std::optional<std::string> errorMessage;
};

bool RecommendationEvaluationSchemaExists(pqxx::connection& connection);
std::vector<RecommendationEvaluationLoadResult>
LoadRecommendationsForEvaluation(
    pqxx::connection& connection,
    const RecommendationEvaluationFilters& filters);
RecommendationEvaluationRunBeginResult BeginOrFindRecommendationEvaluationRun(
    pqxx::connection& connection,
    const RecommendationEvaluationRunRequest& request);
RecommendationEvaluationPersistResult PersistRecommendationEvaluation(
    pqxx::connection& connection,
    const RecommendationEvaluationPersistenceRequest& request);
void CompleteRecommendationEvaluationRun(
    pqxx::connection& connection,
    long long evaluationRunId,
    const RecommendationEvaluationRunCounters& counters);
void FailRecommendationEvaluationRun(
    pqxx::connection& connection,
    long long evaluationRunId,
    const RecommendationEvaluationRunCounters& counters,
    const std::string& errorMessage);
std::vector<PersistedRecommendationEvaluationSummary>
ListRecommendationEvaluations(
    pqxx::connection& connection,
    const RecommendationEvaluationFilters& filters);
std::optional<PersistedRecommendationEvaluationDetail>
FindRecommendationEvaluation(
    pqxx::connection& connection,
    long long evaluationResultId);
std::vector<PersistedRecommendationEvaluationRun>
ListRecommendationEvaluationRuns(pqxx::connection& connection, int limit);
std::optional<PersistedRecommendationEvaluationRun>
FindRecommendationEvaluationRun(
    pqxx::connection& connection,
    long long evaluationRunId);

} // namespace EA::ExperimentRecommendation
