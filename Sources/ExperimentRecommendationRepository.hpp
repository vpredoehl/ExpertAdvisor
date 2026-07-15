#pragma once

#include "ExperimentRecommendationCandidateGenerator.hpp"

#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

inline constexpr const char* kRecommendationDateTimeZone = "America/Chicago";

struct RecommendationSourceFilters
{
    std::optional<std::string> symbol;
    std::optional<int> predictionHorizon;
    std::optional<long long> sourceExperimentId;
};

struct RecommendationSourceLoadResult
{
    long long experimentId = -1;
    std::optional<RecommendationSource> source;
    std::string skipReason;
};

struct RecommendationScanRequest
{
    RecommendationPolicy policy;
    RecommendationSourceFilters filters;
    std::optional<int> requestedMaximum;
};

struct RecommendationScanCounters
{
    int sourcesScanned = 0;
    int sourcesEligible = 0;
    int sourcesSkipped = 0;
    int candidatesGenerated = 0;
    int candidatesRejected = 0;
    int duplicatesExistingExperiment = 0;
    int duplicatesTerminalExperiment = 0;
    int duplicatesActiveRecommendation = 0;
    int duplicatesHistoricalRecommendation = 0;
    int hashCollisions = 0;
    int recommendationsCreated = 0;
    int recommendationsAlreadyExisting = 0;
    int persistenceErrors = 0;
};

struct ExperimentDuplicateMatch
{
    RecommendationDuplicateType type = RecommendationDuplicateType::noDuplicate;
    std::optional<long long> experimentId;
    std::string experimentStatus;
    bool semanticExactMatch = false;
    bool invocationExactMatch = false;
};

enum class PersistedRecommendationMatchKind
{
    none,
    active,
    historical
};

struct RecommendationDuplicateMatch
{
    PersistedRecommendationMatchKind kind =
        PersistedRecommendationMatchKind::none;
    std::optional<long long> recommendationId;
    std::string status;
    bool semanticExactMatch = false;
    bool invocationExactMatch = false;
    bool policyExactMatch = false;
    struct HashCollision
    {
        std::string identityKind;
        RecommendationCandidateHashCollision collision;
    };
    std::vector<HashCollision> hashCollisions;
};

struct RecommendationPersistenceRequest
{
    long long recommendationScanId = -1;
    RecommendationPolicy policy;
    RecommendationSource source;
    GeneratedRecommendationCandidate candidate;
    int generationOrdinal = 0;
    int structuralRank = 0;
};

enum class RecommendationPersistOutcome
{
    created,
    existingExperiment,
    terminalExperiment,
    activeRecommendation,
    historicalRecommendation
};

struct RecommendationPersistResult
{
    RecommendationPersistOutcome outcome =
        RecommendationPersistOutcome::created;
    std::optional<long long> recommendationId;
    ExperimentDuplicateMatch experimentMatch;
    RecommendationDuplicateMatch recommendationMatch;
};

struct RecommendationListFilters
{
    std::optional<std::string> status;
    std::optional<std::string> symbol;
    std::optional<int> predictionHorizon;
    std::optional<long long> recommendationScanId;
    int limit = 100;
};

struct PersistedRecommendationSummary
{
    long long recommendationId = -1;
    long long recommendationScanId = -1;
    std::string status;
    long long sourceExperimentId = -1;
    std::optional<long long> sourceModelId;
    std::optional<long long> sourceAnalysisId;
    std::string sourceSymbol;
    int sourcePredictionHorizon = 0;
    std::string changedParameter;
    std::string sourceValueCanonical;
    std::string proposedValueCanonical;
    std::string semanticHash;
    std::string invocationHash;
    std::string policyHash;
    int generationOrdinal = 0;
    int structuralRank = 0;
    std::string reason;
    std::string createdAt;
};

struct PersistedRecommendationDetail : PersistedRecommendationSummary
{
    double sourceLeaderScore = 0.0;
    double sourceInferAccuracy = 0.0;
    std::optional<double> sourcePredictedNeutralProportion;
    long long sourceEvidenceCount = 0;
    double absoluteDelta = 0.0;
    std::optional<double> relativeDelta;
    std::optional<int> horizonDelta;
    std::string semanticConfigurationCanonical;
    std::string invocationConfigurationCanonical;
    std::string policyCanonical;
    std::string duplicateType;
    std::optional<long long> matchedExperimentId;
    std::optional<long long> matchedRecommendationId;
    std::optional<long long> approvedExperimentId;
    std::optional<std::string> rejectedAt;
    std::optional<std::string> rejectedReason;
    std::optional<std::string> expiredAt;
};

struct PersistedRecommendationScanSummary
{
    long long recommendationScanId = -1;
    std::string status;
    std::string policyHash;
    int policyVersion = 0;
    std::optional<std::string> symbolFilter;
    std::optional<int> horizonFilter;
    std::optional<long long> sourceExperimentFilter;
    std::optional<int> requestedMaximum;
    RecommendationScanCounters counters;
    std::string startedAt;
    std::optional<std::string> completedAt;
    std::optional<std::string> errorMessage;
};

struct PersistedRecommendationScanDetail : PersistedRecommendationScanSummary
{
    std::string policyCanonical;
};

bool RecommendationSchemaExists(pqxx::connection& connection);
long long BeginRecommendationScan(
    pqxx::connection& connection,
    const RecommendationScanRequest& request);
std::vector<RecommendationSourceLoadResult> LoadRecommendationSources(
    pqxx::connection& connection,
    const RecommendationSourceFilters& filters);
std::vector<RecommendationSourceLoadResult> LoadRecommendationSources(
    pqxx::transaction_base& transaction,
    const RecommendationSourceFilters& filters);

ExperimentDuplicateMatch FindExperimentDuplicate(
    pqxx::transaction_base& transaction,
    const RecommendationCandidateIdentity& semanticIdentity,
    const RecommendationInvocationIdentity& invocationIdentity,
    const RecommendationPolicy& policy);
RecommendationDuplicateMatch FindRecommendationDuplicate(
    pqxx::transaction_base& transaction,
    const RecommendationCandidateIdentity& semanticIdentity,
    const RecommendationInvocationIdentity& invocationIdentity,
    const RecommendationPolicy& policy);

RecommendationPersistResult PersistRecommendationIdempotently(
    pqxx::connection& connection,
    const RecommendationPersistenceRequest& request);
void CompleteRecommendationScan(
    pqxx::connection& connection,
    long long scanId,
    const RecommendationScanCounters& counters);
void FailRecommendationScan(
    pqxx::connection& connection,
    long long scanId,
    const RecommendationScanCounters& counters,
    const std::string& errorMessage);

std::vector<PersistedRecommendationSummary> ListRecommendations(
    pqxx::connection& connection,
    const RecommendationListFilters& filters);
std::optional<PersistedRecommendationDetail> FindRecommendation(
    pqxx::connection& connection,
    long long recommendationId);
std::vector<PersistedRecommendationScanSummary> ListRecommendationScans(
    pqxx::connection& connection,
    int limit);
std::optional<PersistedRecommendationScanDetail> FindRecommendationScan(
    pqxx::connection& connection,
    long long scanId);

std::string PersistedRecommendationMatchKindText(
    PersistedRecommendationMatchKind kind);
std::string RecommendationPersistOutcomeText(
    RecommendationPersistOutcome outcome);

} // namespace EA::ExperimentRecommendation
