#include "ExperimentRecommendationService.hpp"

#include "CanonicalSymbol.hpp"
#include "ExperimentRecommendationRepository.hpp"

#include <algorithm>
#include <iomanip>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace EA::ExperimentRecommendation
{
std::string RecommendationMachineText(const std::string& value)
{
    std::ostringstream escaped;
    escaped.imbue(std::locale::classic());
    escaped << std::uppercase << std::hex;
    for (const unsigned char ch : value)
    {
        const bool asciiAlphanumeric =
            (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9');
        const bool safe = asciiAlphanumeric || ch == '-' || ch == '_' ||
                          ch == '.' || ch == ':' || ch == '/' || ch == ';';
        if (safe)
            escaped << static_cast<char>(ch);
        else
            escaped << '%' << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(ch);
    }
    // NULL is the explicit absent-value token.  Encode a concrete value with
    // the same spelling so null and text remain observably distinct.
    return escaped.str() == "NULL" ? "%4E%55%4C%4C" : escaped.str();
}

namespace
{

std::string OptionalText(const std::optional<std::string>& value)
{
    return value ? RecommendationMachineText(*value) : "NULL";
}

template <typename Value>
std::string OptionalNumber(const std::optional<Value>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

int ParameterOrder(RecommendationMutationParameter parameter)
{
    switch (parameter)
    {
        case RecommendationMutationParameter::coreLrMult: return 0;
        case RecommendationMutationParameter::headLrMult: return 1;
        case RecommendationMutationParameter::labelThreshold: return 2;
        case RecommendationMutationParameter::predictionHorizon: return 3;
    }
    return 4;
}

struct CandidateWork
{
    RecommendationSource source;
    std::string groupKey;
    int sourceRank = 0;
    int generationOrdinal = 0;
    GeneratedRecommendationCandidate candidate;
};

bool CandidateWorkLess(const CandidateWork& lhs, const CandidateWork& rhs)
{
    return std::tuple{
               lhs.groupKey, lhs.sourceRank,
               ParameterOrder(lhs.candidate.parameter),
               lhs.candidate.proposedValue,
               lhs.candidate.semanticIdentity.canonicalText,
               lhs.candidate.invocationIdentity.canonicalText,
               lhs.source.experimentId} <
           std::tuple{
               rhs.groupKey, rhs.sourceRank,
               ParameterOrder(rhs.candidate.parameter),
               rhs.candidate.proposedValue,
               rhs.candidate.semanticIdentity.canonicalText,
               rhs.candidate.invocationIdentity.canonicalText,
               rhs.source.experimentId};
}

void PrintScanCounters(
    std::ostream& output,
    const char* event,
    long long scanId,
    const RecommendationScanCounters& counters)
{
    output << event
           << ",scan_id=" << scanId
           << ",sources_scanned=" << counters.sourcesScanned
           << ",sources_eligible=" << counters.sourcesEligible
           << ",sources_skipped=" << counters.sourcesSkipped
           << ",candidates_generated=" << counters.candidatesGenerated
           << ",candidates_rejected=" << counters.candidatesRejected
           << ",duplicates_existing_experiment="
           << counters.duplicatesExistingExperiment
           << ",duplicates_terminal_experiment="
           << counters.duplicatesTerminalExperiment
           << ",duplicates_active_recommendation="
           << counters.duplicatesActiveRecommendation
           << ",duplicates_historical_recommendation="
           << counters.duplicatesHistoricalRecommendation
           << ",hash_collisions=" << counters.hashCollisions
           << ",recommendations_created=" << counters.recommendationsCreated
           << ",recommendations_already_existing="
           << counters.recommendationsAlreadyExisting
           << ",persistence_errors=" << counters.persistenceErrors
           << '\n';
}

void PrintRecommendationSummary(
    std::ostream& output,
    const PersistedRecommendationSummary& summary)
{
    output << "EXPERIMENT_RECOMMENDATION"
           << ",recommendation_id=" << summary.recommendationId
           << ",scan_id=" << summary.recommendationScanId
           << ",status=" << summary.status
           << ",source_experiment_id=" << summary.sourceExperimentId
           << ",source_model_id=" << OptionalNumber(summary.sourceModelId)
           << ",source_analysis_id=" << OptionalNumber(summary.sourceAnalysisId)
           << ",symbol=" << summary.sourceSymbol
           << ",horizon=" << summary.sourcePredictionHorizon
           << ",changed_parameter=" << summary.changedParameter
           << ",source_value=" << summary.sourceValueCanonical
           << ",proposed_value=" << summary.proposedValueCanonical
           << ",semantic_hash=" << summary.semanticHash
           << ",invocation_hash=" << summary.invocationHash
           << ",policy_hash=" << summary.policyHash
           << ",generation_ordinal=" << summary.generationOrdinal
           << ",structural_rank=" << summary.structuralRank
           << ",reason=" << RecommendationMachineText(summary.reason)
           << ",created_at=" << RecommendationMachineText(summary.createdAt)
           << '\n';
}

void PrintScanSummary(
    std::ostream& output,
    const PersistedRecommendationScanSummary& summary)
{
    output << "EXPERIMENT_RECOMMENDATION_SCAN"
           << ",scan_id=" << summary.recommendationScanId
           << ",status=" << summary.status
           << ",policy_hash=" << summary.policyHash
           << ",policy_version=" << summary.policyVersion
           << ",symbol_filter=" << OptionalText(summary.symbolFilter)
           << ",horizon_filter=" << OptionalNumber(summary.horizonFilter)
           << ",source_experiment_filter="
           << OptionalNumber(summary.sourceExperimentFilter)
           << ",requested_maximum=" << OptionalNumber(summary.requestedMaximum)
           << ",sources_scanned=" << summary.counters.sourcesScanned
           << ",sources_eligible=" << summary.counters.sourcesEligible
           << ",sources_skipped=" << summary.counters.sourcesSkipped
           << ",candidates_generated=" << summary.counters.candidatesGenerated
           << ",candidates_rejected=" << summary.counters.candidatesRejected
           << ",recommendations_created="
           << summary.counters.recommendationsCreated
           << ",recommendations_already_existing="
           << summary.counters.recommendationsAlreadyExisting
           << ",hash_collisions=" << summary.counters.hashCollisions
           << ",persistence_errors=" << summary.counters.persistenceErrors
           << ",started_at=" << RecommendationMachineText(summary.startedAt)
           << ",completed_at=" << OptionalText(summary.completedAt)
           << ",error=" << OptionalText(summary.errorMessage)
           << '\n';
}

} // namespace

RecommendationSourceSelectionResult SelectRecommendationSources(
    const RecommendationPolicy& policy,
    std::vector<RecommendationSource> eligibleSources)
{
    using GroupKey = std::tuple<std::string, int>;
    std::map<GroupKey, std::vector<RecommendationSource>> groups;
    const auto groupFor = [&](const RecommendationSource& source) {
        const std::string symbol = source.invocation.configuration.symbol;
        const int horizon = source.invocation.configuration.predictionHorizon;
        switch (policy.sourceScope)
        {
            case RecommendationSourceScope::symbolHorizon:
                return GroupKey{symbol, horizon};
            case RecommendationSourceScope::symbol:
                return GroupKey{symbol, -1};
            case RecommendationSourceScope::global:
                return GroupKey{"", -1};
        }
        return GroupKey{"", -1};
    };
    for (RecommendationSource& source : eligibleSources)
        groups[groupFor(source)].push_back(std::move(source));

    RecommendationSourceSelectionResult result;
    for (auto& [key, sources] : groups)
    {
        std::sort(sources.begin(), sources.end(),
                  [](const RecommendationSource& lhs,
                     const RecommendationSource& rhs) {
            return std::tuple{
                       -*lhs.leaderScore, -*lhs.inferenceAccuracy,
                       -lhs.evidenceCount, lhs.experimentId} <
                   std::tuple{
                       -*rhs.leaderScore, -*rhs.inferenceAccuracy,
                       -rhs.evidenceCount, rhs.experimentId};
        });
        const std::string groupText = std::get<0>(key).empty()
            ? "global"
            : (std::get<1>(key) < 0
                ? std::get<0>(key)
                : std::get<0>(key) + ":" + std::to_string(std::get<1>(key)));
        for (std::size_t index = 0; index < sources.size(); ++index)
        {
            if (index >= static_cast<std::size_t>(policy.topSourcesPerScope))
            {
                result.skipped.emplace_back(
                    sources[index].experimentId,
                    "outside_top_sources_per_scope");
                continue;
            }
            result.selected.push_back(RecommendationSourceSelectionRecord{
                std::move(sources[index]), groupText,
                static_cast<int>(index + 1)});
        }
    }
    return result;
}

int RunGenerateExperimentRecommendationsCommand(
    const std::string& connectionString,
    const RecommendationGenerationCommandRequest& request,
    std::ostream& output,
    std::ostream& errors)
{
    if (const auto error = ValidateRecommendationPolicy(request.policy))
        throw std::invalid_argument(*error);
    if (request.requestedMaximum && *request.requestedMaximum <= 0)
        throw std::invalid_argument("recommendation_maximum_must_be_positive");

    pqxx::connection connection{connectionString};
    if (!RecommendationSchemaExists(connection))
        throw std::runtime_error(
            "recommendation schema required; run ./migrate_lstm_db.sh");

    RecommendationScanRequest scanRequest;
    scanRequest.policy = request.policy;
    scanRequest.filters.symbol = request.symbol;
    scanRequest.filters.predictionHorizon = request.predictionHorizon;
    scanRequest.filters.sourceExperimentId = request.sourceExperimentId;
    scanRequest.requestedMaximum = request.requestedMaximum;
    const long long scanId = BeginRecommendationScan(connection, scanRequest);
    RecommendationScanCounters counters;
    output << "EXPERIMENT_RECOMMENDATION_SCAN_START"
           << ",scan_id=" << scanId
           << ",policy_hash=" << RecommendationPolicyHash(request.policy)
           << ",symbol=" << OptionalText(request.symbol)
           << ",horizon=" << OptionalNumber(request.predictionHorizon)
           << ",source_experiment_id="
           << OptionalNumber(request.sourceExperimentId)
           << ",requested_maximum=" << OptionalNumber(request.requestedMaximum)
           << '\n';
    output.flush();

    try
    {
        const std::vector<RecommendationSourceLoadResult> loaded =
            LoadRecommendationSources(connection, scanRequest.filters);
        counters.sourcesScanned = static_cast<int>(loaded.size());
        std::vector<RecommendationSource> eligibleSources;
        for (const RecommendationSourceLoadResult& item : loaded)
        {
            if (!item.source)
            {
                ++counters.sourcesSkipped;
                output << "EXPERIMENT_RECOMMENDATION_SOURCE_SKIPPED"
                       << ",scan_id=" << scanId
                       << ",source_experiment_id=" << item.experimentId
                       << ",reason=" << RecommendationMachineText(item.skipReason) << '\n';
                continue;
            }
            const RecommendationSourceEligibilityResult eligibility =
                EvaluateRecommendationSource(request.policy, *item.source);
            if (!eligibility.eligible)
            {
                ++counters.sourcesSkipped;
                output << "EXPERIMENT_RECOMMENDATION_SOURCE_SKIPPED"
                       << ",scan_id=" << scanId
                       << ",source_experiment_id=" << item.experimentId
                       << ",reason="
                       << RecommendationSourceEligibilityReasonText(
                              eligibility.reason)
                       << ",detail=" << RecommendationMachineText(eligibility.detail) << '\n';
                continue;
            }
            eligibleSources.push_back(*item.source);
        }

        RecommendationSourceSelectionResult selection =
            SelectRecommendationSources(request.policy,
                                        std::move(eligibleSources));
        for (const auto& [experimentId, reason] : selection.skipped)
        {
            ++counters.sourcesSkipped;
            output << "EXPERIMENT_RECOMMENDATION_SOURCE_SKIPPED"
                   << ",scan_id=" << scanId
                   << ",source_experiment_id=" << experimentId
                   << ",reason=" << reason << '\n';
        }
        counters.sourcesEligible = static_cast<int>(selection.selected.size());

        std::vector<CandidateWork> work;
        for (const RecommendationSourceSelectionRecord& selected :
             selection.selected)
        {
            output << "EXPERIMENT_RECOMMENDATION_SOURCE_ELIGIBLE"
                   << ",scan_id=" << scanId
                   << ",source_experiment_id=" << selected.source.experimentId
                   << ",group=" << selected.groupKey
                   << ",source_rank=" << selected.rankWithinGroup
                   << ",leader_score="
                   << CanonicalRecommendationDouble(*selected.source.leaderScore)
                   << ",infer_accuracy="
                   << CanonicalRecommendationDouble(
                          *selected.source.inferenceAccuracy)
                   << ",evidence_count=" << selected.source.evidenceCount
                   << '\n';
            RecommendationCandidateGenerationResult generated =
                GenerateRecommendationCandidates(request.policy,
                                                 selected.source);
            counters.candidatesGenerated +=
                static_cast<int>(generated.candidates.size());
            counters.candidatesRejected +=
                static_cast<int>(generated.rejected.size());
            counters.hashCollisions +=
                static_cast<int>(generated.collisions.size());
            for (const RejectedRecommendationCandidate& rejected :
                 generated.rejected)
            {
                output << "EXPERIMENT_RECOMMENDATION_CANDIDATE_REJECTED"
                       << ",scan_id=" << scanId
                       << ",source_experiment_id="
                       << selected.source.experimentId
                       << ",parameter="
                       << RecommendationMutationParameterText(rejected.parameter)
                       << ",proposed_value=" << rejected.proposedValue
                       << ",reason="
                       << RecommendationCandidateRejectionReasonText(
                              rejected.reason) << '\n';
            }
            for (const RecommendationCandidateHashCollision& collision :
                 generated.collisions)
            {
                output << "EXPERIMENT_RECOMMENDATION_HASH_COLLISION"
                       << ",scan_id=" << scanId
                       << ",source_experiment_id="
                       << selected.source.experimentId
                       << ",identity_kind=semantic"
                       << ",hash=" << collision.hash
                       << ",first_canonical="
                       << RecommendationMachineText(collision.firstCanonicalText)
                       << ",second_canonical="
                       << RecommendationMachineText(collision.secondCanonicalText) << '\n';
            }
            for (std::size_t index = 0;
                 index < generated.candidates.size(); ++index)
            {
                work.push_back(CandidateWork{
                    selected.source, selected.groupKey,
                    selected.rankWithinGroup, static_cast<int>(index + 1),
                    std::move(generated.candidates[index])});
            }
        }

        std::sort(work.begin(), work.end(), CandidateWorkLess);
        const int scanLimit = request.requestedMaximum.value_or(
            request.policy.maximumRecommendationsPerScan);
        if (work.size() > static_cast<std::size_t>(scanLimit))
        {
            for (std::size_t index = static_cast<std::size_t>(scanLimit);
                 index < work.size(); ++index)
            {
                ++counters.candidatesRejected;
                output << "EXPERIMENT_RECOMMENDATION_CANDIDATE_REJECTED"
                       << ",scan_id=" << scanId
                       << ",source_experiment_id="
                       << work[index].source.experimentId
                       << ",parameter="
                       << RecommendationMutationParameterText(
                              work[index].candidate.parameter)
                       << ",proposed_value="
                       << CanonicalRecommendationDouble(
                              work[index].candidate.proposedValue)
                       << ",reason=scan_limit" << '\n';
            }
            work.resize(static_cast<std::size_t>(scanLimit));
        }

        for (std::size_t index = 0; index < work.size(); ++index)
        {
            const int structuralRank = static_cast<int>(index + 1);
            const CandidateWork& item = work[index];
            output << "EXPERIMENT_RECOMMENDATION_CANDIDATE"
                   << ",scan_id=" << scanId
                   << ",source_experiment_id=" << item.source.experimentId
                   << ",source_rank=" << item.sourceRank
                   << ",parameter="
                   << RecommendationMutationParameterText(
                          item.candidate.parameter)
                   << ",source_value="
                   << CanonicalRecommendationDouble(item.candidate.sourceValue)
                   << ",proposed_value="
                   << CanonicalRecommendationDouble(
                          item.candidate.proposedValue)
                   << ",semantic_hash=" << item.candidate.semanticIdentity.hash
                   << ",invocation_hash="
                   << item.candidate.invocationIdentity.hash
                   << ",policy_hash="
                   << RecommendationPolicyHash(request.policy)
                   << ",structural_rank=" << structuralRank << '\n';
            try
            {
                RecommendationPersistenceRequest persistence;
                persistence.recommendationScanId = scanId;
                persistence.policy = request.policy;
                persistence.source = item.source;
                persistence.candidate = item.candidate;
                persistence.sourceRank = item.sourceRank;
                persistence.generationOrdinal = item.generationOrdinal;
                persistence.structuralRank = structuralRank;
                const RecommendationPersistResult persisted =
                    PersistRecommendationIdempotently(connection, persistence);
                counters.hashCollisions += static_cast<int>(
                    persisted.recommendationMatch.hashCollisions.size());
                for (const RecommendationDuplicateMatch::HashCollision& observed :
                     persisted.recommendationMatch.hashCollisions)
                {
                    const RecommendationCandidateHashCollision& collision =
                        observed.collision;
                    output << "EXPERIMENT_RECOMMENDATION_HASH_COLLISION"
                           << ",scan_id=" << scanId
                           << ",source_experiment_id="
                           << item.source.experimentId
                           << ",identity_kind=" << observed.identityKind
                           << ",hash=" << collision.hash
                           << ",first_canonical="
                           << RecommendationMachineText(collision.firstCanonicalText)
                           << ",second_canonical="
                           << RecommendationMachineText(collision.secondCanonicalText) << '\n';
                }
                switch (persisted.outcome)
                {
                    case RecommendationPersistOutcome::created:
                        ++counters.recommendationsCreated;
                        output << "EXPERIMENT_RECOMMENDATION_CREATED"
                               << ",scan_id=" << scanId
                               << ",recommendation_id="
                               << *persisted.recommendationId
                               << ",source_experiment_id="
                               << item.source.experimentId
                               << ",semantic_hash="
                               << item.candidate.semanticIdentity.hash << '\n';
                        break;
                    case RecommendationPersistOutcome::existingExperiment:
                        ++counters.duplicatesExistingExperiment;
                        output << "EXPERIMENT_RECOMMENDATION_DUPLICATE"
                               << ",scan_id=" << scanId
                               << ",source_experiment_id="
                               << item.source.experimentId
                               << ",duplicate_type=existing_experiment"
                               << ",matched_experiment_id="
                               << OptionalNumber(
                                      persisted.experimentMatch.experimentId)
                               << ",invocation_exact="
                               << (persisted.experimentMatch.invocationExactMatch
                                      ? 1 : 0) << '\n';
                        break;
                    case RecommendationPersistOutcome::terminalExperiment:
                        ++counters.duplicatesTerminalExperiment;
                        output << "EXPERIMENT_RECOMMENDATION_DUPLICATE"
                               << ",scan_id=" << scanId
                               << ",source_experiment_id="
                               << item.source.experimentId
                               << ",duplicate_type=excluded_terminal_experiment"
                               << ",matched_experiment_id="
                               << OptionalNumber(
                                      persisted.experimentMatch.experimentId)
                               << '\n';
                        break;
                    case RecommendationPersistOutcome::activeRecommendation:
                        ++counters.duplicatesActiveRecommendation;
                        ++counters.recommendationsAlreadyExisting;
                        output << "EXPERIMENT_RECOMMENDATION_ALREADY_EXISTS"
                               << ",scan_id=" << scanId
                               << ",recommendation_id="
                               << OptionalNumber(persisted.recommendationId)
                               << ",source_experiment_id="
                               << item.source.experimentId
                               << ",duplicate_type=active_recommendation"
                               << ",invocation_exact="
                               << (persisted.recommendationMatch.invocationExactMatch
                                      ? 1 : 0) << '\n';
                        break;
                    case RecommendationPersistOutcome::historicalRecommendation:
                        ++counters.duplicatesHistoricalRecommendation;
                        output << "EXPERIMENT_RECOMMENDATION_DUPLICATE"
                               << ",scan_id=" << scanId
                               << ",source_experiment_id="
                               << item.source.experimentId
                               << ",duplicate_type=historical_recommendation"
                               << ",matched_recommendation_id="
                               << OptionalNumber(persisted.recommendationId)
                               << '\n';
                        break;
                }
            }
            catch (const std::exception& error)
            {
                output.flush();
                ++counters.persistenceErrors;
                errors << "EXPERIMENT_RECOMMENDATION_PERSISTENCE_ERROR"
                       << ",scan_id=" << scanId
                       << ",source_experiment_id=" << item.source.experimentId
                       << ",semantic_hash="
                       << item.candidate.semanticIdentity.hash
                       << ",error=" << RecommendationMachineText(error.what()) << '\n';
            }
        }
        CompleteRecommendationScan(connection, scanId, counters);
        PrintScanCounters(output, "EXPERIMENT_RECOMMENDATION_SCAN_COMPLETE",
                          scanId, counters);
        return counters.persistenceErrors == 0 ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        ++counters.persistenceErrors;
        try
        {
            FailRecommendationScan(connection, scanId, counters, error.what());
        }
        catch (const std::exception& finalizationError)
        {
            errors << "EXPERIMENT_RECOMMENDATION_SCAN_FINALIZATION_FAILED"
                   << ",scan_id=" << scanId
                   << ",original_error="
                   << RecommendationMachineText(error.what())
                   << ",finalization_error="
                   << RecommendationMachineText(finalizationError.what())
                   << '\n';
        }
        errors << "EXPERIMENT_RECOMMENDATION_SCAN_FAILED"
               << ",scan_id=" << scanId
               << ",error=" << RecommendationMachineText(error.what()) << '\n';
        return 1;
    }
}

int RunListExperimentRecommendationsCommand(
    const std::string& connectionString,
    const RecommendationListCommandRequest& request,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    if (!RecommendationSchemaExists(connection))
        throw std::runtime_error("recommendation schema required; run ./migrate_lstm_db.sh");
    RecommendationListFilters filters;
    filters.status = request.status;
    filters.symbol = request.symbol;
    filters.predictionHorizon = request.predictionHorizon;
    filters.recommendationScanId = request.recommendationScanId;
    filters.limit = request.limit;
    const auto recommendations = ListRecommendations(connection, filters);
    for (const auto& recommendation : recommendations)
        PrintRecommendationSummary(output, recommendation);
    output << "EXPERIMENT_RECOMMENDATION_LIST_COMPLETE,count="
           << recommendations.size() << '\n';
    return 0;
}

int RunExperimentRecommendationStatusCommand(
    const std::string& connectionString,
    long long recommendationId,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    if (!RecommendationSchemaExists(connection))
        throw std::runtime_error("recommendation schema required; run ./migrate_lstm_db.sh");
    const auto detail = FindRecommendation(connection, recommendationId);
    if (!detail)
    {
        output << "EXPERIMENT_RECOMMENDATION_NOT_FOUND,recommendation_id="
               << recommendationId << '\n';
        return 1;
    }
    PrintRecommendationSummary(output, *detail);
    output << "EXPERIMENT_RECOMMENDATION_DETAIL"
           << ",recommendation_id=" << detail->recommendationId
           << ",leader_score="
           << CanonicalRecommendationDouble(detail->sourceLeaderScore)
           << ",infer_accuracy="
           << CanonicalRecommendationDouble(detail->sourceInferAccuracy)
           << ",predicted_neutral="
           << (detail->sourcePredictedNeutralProportion
                ? CanonicalRecommendationDouble(
                      *detail->sourcePredictedNeutralProportion)
                : "NULL")
           << ",evidence_count=" << detail->sourceEvidenceCount
           << ",semantic_canonical="
           << RecommendationMachineText(detail->semanticConfigurationCanonical)
           << ",invocation_canonical="
           << RecommendationMachineText(detail->invocationConfigurationCanonical)
           << ",policy_canonical=" << RecommendationMachineText(detail->policyCanonical)
           << '\n';
    return 0;
}

int RunListExperimentRecommendationScansCommand(
    const std::string& connectionString,
    int limit,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    if (!RecommendationSchemaExists(connection))
        throw std::runtime_error("recommendation schema required; run ./migrate_lstm_db.sh");
    const auto scans = ListRecommendationScans(connection, limit);
    for (const auto& scan : scans) PrintScanSummary(output, scan);
    output << "EXPERIMENT_RECOMMENDATION_SCAN_LIST_COMPLETE,count="
           << scans.size() << '\n';
    return 0;
}

int RunExperimentRecommendationScanStatusCommand(
    const std::string& connectionString,
    long long scanId,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    if (!RecommendationSchemaExists(connection))
        throw std::runtime_error("recommendation schema required; run ./migrate_lstm_db.sh");
    const auto detail = FindRecommendationScan(connection, scanId);
    if (!detail)
    {
        output << "EXPERIMENT_RECOMMENDATION_SCAN_NOT_FOUND,scan_id="
               << scanId << '\n';
        return 1;
    }
    PrintScanSummary(output, *detail);
    output << "EXPERIMENT_RECOMMENDATION_SCAN_DETAIL"
           << ",scan_id=" << detail->recommendationScanId
           << ",policy_canonical=" << RecommendationMachineText(detail->policyCanonical)
           << '\n';
    return 0;
}

} // namespace EA::ExperimentRecommendation
