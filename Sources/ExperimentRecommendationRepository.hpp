#pragma once

#include "ExperimentRecommendationCandidateGenerator.hpp"
#include "ExperimentRecommendationReview.hpp"
#include "ExperimentRecommendationScoring.hpp"

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
    int sourceRank = 0;
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
    std::optional<int> sourceRank;
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
    std::optional<std::string> approvedAt;
    std::optional<std::string> rejectedAt;
    std::optional<std::string> rejectedReason;
    std::optional<std::string> expiredAt;
};

struct RecommendationScoringFilters
{
    std::string status = "proposed";
    std::optional<std::string> symbol;
    std::optional<int> predictionHorizon;
    std::optional<long long> recommendationScanId;
    std::optional<long long> recommendationId;
    std::optional<double> minimumScore;
    std::optional<long long> scoreRunId;
    int limit = 100;
};

struct RecommendationScoreRunRequest
{
    RecommendationScoringPolicy policy;
    RecommendationScoringFilters filters;
    std::optional<int> requestedLimit;
};

struct RecommendationScoreRunCounters
{
    int recommendationsConsidered = 0;
    int recommendationsScored = 0;
    int recommendationsSkipped = 0;
    int scoringErrors = 0;
    int hashCollisions = 0;
};

struct RecommendationScoringLoadResult
{
    long long recommendationId = -1;
    std::optional<RecommendationScoringInput> input;
    std::string skipReason;
};

struct RecommendationScorePersistenceRequest
{
    long long scoreRunId = -1;
    RankedRecommendationScore ranked;
};

struct RecommendationScorePersistResult
{
    long long recommendationScoreId = -1;
    bool created = false;
};

struct PersistedRecommendationScoreSummary
{
    long long recommendationScoreId = -1;
    long long scoreRunId = -1;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    std::string scoringPolicyHash;
    int scoringVersion = 0;
    double finalScore = 0.0;
    double rawPositiveScore = 0.0;
    double rawPenaltyScore = 0.0;
    double rawTotalScore = 0.0;
    double structuralDistance = 0.0;
    int scoreRank = 0;
    int tieGroup = 0;
    int rankingOrdinal = 0;
    std::string reasonCode;
    std::string explanation;
    std::string createdAt;
};

struct PersistedRecommendationScoreDetail : PersistedRecommendationScoreSummary
{
    std::string scoringPolicyCanonical;
    std::string recommendationSemanticCanonical;
    std::string recommendationPolicyCanonical;
    std::vector<RecommendationScoreComponent> components;
};

struct PersistedRecommendationScoreRunSummary
{
    long long scoreRunId = -1;
    std::string status;
    std::string scoringPolicyHash;
    int scoringVersion = 0;
    std::optional<std::string> recommendationStatusFilter;
    std::optional<std::string> symbolFilter;
    std::optional<int> horizonFilter;
    std::optional<long long> recommendationScanFilter;
    std::optional<long long> recommendationIdFilter;
    std::optional<int> requestedLimit;
    RecommendationScoreRunCounters counters;
    std::string startedAt;
    std::optional<std::string> completedAt;
    std::optional<std::string> errorMessage;
};

struct PersistedRecommendationScoreRunDetail : PersistedRecommendationScoreRunSummary
{
    std::string scoringPolicyCanonical;
};

struct RecommendationReviewPersistenceRequest
{
    long long recommendationId = -1;
    RecommendationReviewRequest review;
};

struct RecommendationReviewFilters
{
    std::optional<long long> recommendationId;
    std::optional<RecommendationReviewAction> action;
    int limit = 100;
};

struct PersistedRecommendationReviewEvent
{
    long long recommendationReviewEventId = -1;
    long long recommendationId = -1;
    std::optional<long long> recommendationScoreId;
    RecommendationReviewAction action = RecommendationReviewAction::approve;
    std::string previousStatus;
    std::string resultingStatus;
    std::string reasonCode;
    std::optional<std::string> reasonText;
    std::optional<std::string> reviewer;
    std::optional<std::string> note;
    std::string recommendationSemanticCanonical;
    std::string recommendationSemanticHash;
    std::string recommendationPolicyCanonical;
    std::string recommendationPolicyHash;
    long long recommendationScanId = -1;
    long long sourceExperimentId = -1;
    std::string createdAt;
};

struct RecommendationReviewPersistResult
{
    PersistedRecommendationReviewEvent event;
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

bool RecommendationScoringSchemaExists(pqxx::connection& connection);
long long BeginRecommendationScoreRun(
    pqxx::connection& connection,
    const RecommendationScoreRunRequest& request);
std::optional<std::string> FindRecommendationScoringPolicyHashCollision(
    pqxx::connection& connection,
    const RecommendationScoringPolicy& policy);
std::vector<RecommendationScoringLoadResult> LoadRecommendationsForScoring(
    pqxx::connection& connection,
    const RecommendationScoringFilters& filters);
RecommendationScorePersistResult PersistRecommendationScore(
    pqxx::connection& connection,
    const RecommendationScorePersistenceRequest& request);
void CompleteRecommendationScoreRun(
    pqxx::connection& connection,
    long long scoreRunId,
    const RecommendationScoreRunCounters& counters);
void FailRecommendationScoreRun(
    pqxx::connection& connection,
    long long scoreRunId,
    const RecommendationScoreRunCounters& counters,
    const std::string& errorMessage);
std::vector<PersistedRecommendationScoreSummary> ListRecommendationScores(
    pqxx::connection& connection,
    const RecommendationScoringFilters& filters);
std::optional<PersistedRecommendationScoreDetail> FindRecommendationScore(
    pqxx::connection& connection,
    long long scoreId);
std::vector<PersistedRecommendationScoreRunSummary> ListRecommendationScoreRuns(
    pqxx::connection& connection,
    int limit);
std::optional<PersistedRecommendationScoreRunDetail> FindRecommendationScoreRun(
    pqxx::connection& connection,
    long long scoreRunId);

bool RecommendationReviewSchemaExists(pqxx::connection& connection);
RecommendationReviewPersistResult ReviewRecommendation(
    pqxx::connection& connection,
    const RecommendationReviewPersistenceRequest& request);
std::vector<PersistedRecommendationReviewEvent> ListRecommendationReviewEvents(
    pqxx::connection& connection,
    const RecommendationReviewFilters& filters);
std::optional<PersistedRecommendationReviewEvent> FindRecommendationReviewEvent(
    pqxx::connection& connection,
    long long reviewEventId);
std::vector<PersistedRecommendationReviewEvent>
ListReviewHistoryForRecommendation(
    pqxx::connection& connection,
    long long recommendationId);

std::string PersistedRecommendationMatchKindText(
    PersistedRecommendationMatchKind kind);
std::string RecommendationPersistOutcomeText(
    RecommendationPersistOutcome outcome);

} // namespace EA::ExperimentRecommendation
