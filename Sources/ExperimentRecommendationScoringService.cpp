#include "ExperimentRecommendationService.hpp"

#include "ExperimentRecommendationRepository.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

template <typename Value>
std::string OptionalNumber(const std::optional<Value>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalText(const std::optional<std::string>& value)
{
    return value ? RecommendationMachineText(*value) : "NULL";
}

void PrintScoreRunCounters(
    std::ostream& output,
    const char* event,
    long long runId,
    const RecommendationScoreRunCounters& counters)
{
    output << event
           << ",score_run_id=" << runId
           << ",recommendations_considered="
           << counters.recommendationsConsidered
           << ",recommendations_scored=" << counters.recommendationsScored
           << ",recommendations_skipped=" << counters.recommendationsSkipped
           << ",scoring_errors=" << counters.scoringErrors
           << ",hash_collisions=" << counters.hashCollisions << '\n';
}

void PrintScoreSummary(
    std::ostream& output,
    const PersistedRecommendationScoreSummary& score)
{
    output << "EXPERIMENT_RECOMMENDATION_SCORE"
           << ",score_id=" << score.recommendationScoreId
           << ",score_run_id=" << score.scoreRunId
           << ",recommendation_id=" << score.recommendationId
           << ",source_experiment_id=" << score.sourceExperimentId
           << ",scoring_policy_hash="
           << RecommendationMachineText(score.scoringPolicyHash)
           << ",scoring_version=" << score.scoringVersion
           << ",final_score=" << CanonicalRecommendationDouble(score.finalScore)
           << ",raw_positive_score="
           << CanonicalRecommendationDouble(score.rawPositiveScore)
           << ",raw_penalty_score="
           << CanonicalRecommendationDouble(score.rawPenaltyScore)
           << ",raw_total_score="
           << CanonicalRecommendationDouble(score.rawTotalScore)
           << ",structural_distance="
           << CanonicalRecommendationDouble(score.structuralDistance)
           << ",score_rank=" << score.scoreRank
           << ",tie_group=" << score.tieGroup
           << ",ranking_ordinal=" << score.rankingOrdinal
           << ",reason_code=" << RecommendationMachineText(score.reasonCode)
           << ",explanation=" << RecommendationMachineText(score.explanation)
           << ",created_at=" << RecommendationMachineText(score.createdAt)
           << '\n';
}

void PrintScoreRunSummary(
    std::ostream& output,
    const PersistedRecommendationScoreRunSummary& run)
{
    output << "EXPERIMENT_RECOMMENDATION_SCORE_RUN"
           << ",score_run_id=" << run.scoreRunId
           << ",status=" << RecommendationMachineText(run.status)
           << ",scoring_policy_hash="
           << RecommendationMachineText(run.scoringPolicyHash)
           << ",scoring_version=" << run.scoringVersion
           << ",recommendation_status_filter="
           << OptionalText(run.recommendationStatusFilter)
           << ",symbol_filter=" << OptionalText(run.symbolFilter)
           << ",horizon_filter=" << OptionalNumber(run.horizonFilter)
           << ",recommendation_scan_filter="
           << OptionalNumber(run.recommendationScanFilter)
           << ",recommendation_id_filter="
           << OptionalNumber(run.recommendationIdFilter)
           << ",requested_limit=" << OptionalNumber(run.requestedLimit)
           << ",recommendations_considered="
           << run.counters.recommendationsConsidered
           << ",recommendations_scored="
           << run.counters.recommendationsScored
           << ",recommendations_skipped="
           << run.counters.recommendationsSkipped
           << ",scoring_errors=" << run.counters.scoringErrors
           << ",hash_collisions=" << run.counters.hashCollisions
           << ",started_at=" << RecommendationMachineText(run.startedAt)
           << ",completed_at=" << OptionalText(run.completedAt)
           << ",error_message=" << OptionalText(run.errorMessage) << '\n';
}

} // namespace

int RunScoreExperimentRecommendationsCommand(
    const std::string& connectionString,
    const RecommendationScoringCommandRequest& request,
    std::ostream& output,
    std::ostream& errors)
{
    if (const auto validation = ValidateRecommendationScoringPolicy(request.policy))
        throw std::invalid_argument(*validation);
    pqxx::connection connection{connectionString};
    if (!RecommendationScoringSchemaExists(connection))
        throw std::runtime_error("recommendation_scoring_schema_unavailable");

    RecommendationScoringFilters filters;
    filters.status = request.status.value_or("proposed");
    filters.symbol = request.symbol;
    filters.predictionHorizon = request.predictionHorizon;
    filters.recommendationScanId = request.recommendationScanId;
    filters.recommendationId = request.recommendationId;
    RecommendationScoreRunRequest runRequest;
    runRequest.policy = request.policy;
    runRequest.filters = filters;
    runRequest.requestedLimit = request.requestedLimit;
    const long long runId = BeginRecommendationScoreRun(connection, runRequest);
    RecommendationScoreRunCounters counters;
    output << "EXPERIMENT_RECOMMENDATION_SCORE_RUN_START"
           << ",score_run_id=" << runId
           << ",scoring_policy_hash="
           << RecommendationScoringPolicyHash(request.policy)
           << ",scoring_version=" << request.policy.scoringVersion << '\n';

    try
    {
        if (const auto collidedCanonical =
                FindRecommendationScoringPolicyHashCollision(
                    connection, request.policy))
        {
            ++counters.hashCollisions;
            output << "EXPERIMENT_RECOMMENDATION_SCORE_HASH_COLLISION"
                   << ",score_run_id=" << runId
                   << ",identity_kind=scoring_policy"
                   << ",hash="
                   << RecommendationScoringPolicyHash(request.policy)
                   << ",current_canonical="
                   << RecommendationMachineText(
                          RecommendationScoringPolicyCanonicalText(
                              request.policy))
                   << ",existing_canonical="
                   << RecommendationMachineText(*collidedCanonical) << '\n';
        }
        const std::vector<RecommendationScoringLoadResult> loaded =
            LoadRecommendationsForScoring(connection, filters);
        std::vector<RankedRecommendationScore> scoreable;
        for (const RecommendationScoringLoadResult& item : loaded)
        {
            ++counters.recommendationsConsidered;
            if (!item.input)
            {
                ++counters.recommendationsSkipped;
                output << "EXPERIMENT_RECOMMENDATION_SCORE_SKIPPED"
                       << ",score_run_id=" << runId
                       << ",recommendation_id=" << item.recommendationId
                       << ",reason=" << RecommendationMachineText(item.skipReason)
                       << '\n';
                continue;
            }
            output << "EXPERIMENT_RECOMMENDATION_SCORE_INPUT"
                   << ",score_run_id=" << runId
                   << ",recommendation_id=" << item.recommendationId
                   << ",source_experiment_id="
                   << item.input->sourceExperimentId << '\n';
            RecommendationScoreResult score =
                ScoreExperimentRecommendation(request.policy, *item.input);
            if (!score.valid)
            {
                ++counters.recommendationsSkipped;
                output << "EXPERIMENT_RECOMMENDATION_SCORE_SKIPPED"
                       << ",score_run_id=" << runId
                       << ",recommendation_id=" << item.recommendationId
                       << ",reason="
                       << RecommendationMachineText(score.reasonCode) << '\n';
                continue;
            }
            scoreable.push_back(RankedRecommendationScore{
                *item.input, std::move(score), 0, 0, 0});
        }

        std::vector<RankedRecommendationScore> ranked =
            RankRecommendationScores(std::move(scoreable));
        if (request.requestedLimit &&
            ranked.size() > static_cast<std::size_t>(*request.requestedLimit))
        {
            for (std::size_t index = static_cast<std::size_t>(
                     *request.requestedLimit); index < ranked.size(); ++index)
            {
                ++counters.recommendationsSkipped;
                output << "EXPERIMENT_RECOMMENDATION_SCORE_SKIPPED"
                       << ",score_run_id=" << runId
                       << ",recommendation_id="
                       << ranked[index].input.recommendationId
                       << ",reason=score_run_limit" << '\n';
            }
            ranked.resize(static_cast<std::size_t>(*request.requestedLimit));
        }

        for (const RankedRecommendationScore& item : ranked)
        {
            try
            {
                const RecommendationScorePersistResult persisted =
                    PersistRecommendationScore(
                        connection, RecommendationScorePersistenceRequest{
                            runId, item});
                ++counters.recommendationsScored;
                for (std::size_t index = 0;
                     index < item.score.components.size(); ++index)
                {
                    const RecommendationScoreComponent& component =
                        item.score.components[index];
                    output << "EXPERIMENT_RECOMMENDATION_SCORE_COMPONENT"
                           << ",score_run_id=" << runId
                           << ",score_id=" << persisted.recommendationScoreId
                           << ",recommendation_id="
                           << item.input.recommendationId
                           << ",component_ordinal=" << (index + 1)
                           << ",component_name="
                           << RecommendationMachineText(component.componentName)
                           << ",reason_code="
                           << RecommendationMachineText(component.reasonCode)
                           << ",normalized_value="
                           << CanonicalRecommendationDouble(
                                  component.normalizedValue)
                           << ",weight="
                           << CanonicalRecommendationDouble(component.weight)
                           << ",weighted_contribution="
                           << CanonicalRecommendationDouble(
                                  component.weightedContribution)
                           << ",is_penalty=" << (component.penalty ? 1 : 0)
                           << '\n';
                }
                output << "EXPERIMENT_RECOMMENDATION_SCORED"
                       << ",score_run_id=" << runId
                       << ",score_id=" << persisted.recommendationScoreId
                       << ",recommendation_id="
                       << item.input.recommendationId
                       << ",created=" << (persisted.created ? 1 : 0)
                       << ",final_score="
                       << CanonicalRecommendationDouble(item.score.finalScore)
                       << ",raw_positive_score="
                       << CanonicalRecommendationDouble(
                              item.score.rawPositiveScore)
                       << ",raw_penalty_score="
                       << CanonicalRecommendationDouble(
                              item.score.rawPenaltyScore)
                       << ",raw_total_score="
                       << CanonicalRecommendationDouble(
                              item.score.rawTotalScore)
                       << ",score_rank=" << item.scoreRank
                       << ",tie_group=" << item.tieGroup
                       << ",ranking_ordinal=" << item.rankingOrdinal << '\n';
            }
            catch (const std::exception& error)
            {
                ++counters.scoringErrors;
                errors << "EXPERIMENT_RECOMMENDATION_SCORE_ERROR"
                       << ",score_run_id=" << runId
                       << ",recommendation_id="
                       << item.input.recommendationId
                       << ",error=" << RecommendationMachineText(error.what())
                       << '\n';
            }
        }
        CompleteRecommendationScoreRun(connection, runId, counters);
        PrintScoreRunCounters(output,
            "EXPERIMENT_RECOMMENDATION_SCORE_RUN_COMPLETE", runId, counters);
        return counters.scoringErrors == 0 ? 0 : 2;
    }
    catch (const std::exception& error)
    {
        try
        {
            FailRecommendationScoreRun(connection, runId, counters, error.what());
        }
        catch (const std::exception& finalizationError)
        {
            errors << "EXPERIMENT_RECOMMENDATION_SCORE_RUN_FAILED"
                   << ",score_run_id=" << runId
                   << ",error=" << RecommendationMachineText(error.what())
                   << ",finalization_error="
                   << RecommendationMachineText(finalizationError.what()) << '\n';
            throw std::runtime_error(
                std::string{"recommendation_score_run_failed:"} + error.what() +
                ";finalization_failed:" + finalizationError.what());
        }
        errors << "EXPERIMENT_RECOMMENDATION_SCORE_RUN_FAILED"
               << ",score_run_id=" << runId
               << ",error=" << RecommendationMachineText(error.what()) << '\n';
        return 1;
    }
}

int RunListExperimentRecommendationScoresCommand(
    const std::string& connectionString,
    const RecommendationScoreListCommandRequest& request,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    RecommendationScoringFilters filters;
    filters.scoreRunId = request.scoreRunId;
    filters.recommendationId = request.recommendationId;
    filters.symbol = request.symbol;
    filters.predictionHorizon = request.predictionHorizon;
    filters.minimumScore = request.minimumScore;
    filters.limit = request.limit;
    const auto scores = ListRecommendationScores(connection, filters);
    for (const auto& score : scores) PrintScoreSummary(output, score);
    output << "EXPERIMENT_RECOMMENDATION_SCORE_LIST_COMPLETE,count="
           << scores.size() << '\n';
    return 0;
}

int RunExperimentRecommendationScoreStatusCommand(
    const std::string& connectionString,
    long long scoreId,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto detail = FindRecommendationScore(connection, scoreId);
    if (!detail) return 1;
    PrintScoreSummary(output, *detail);
    for (std::size_t index = 0; index < detail->components.size(); ++index)
    {
        const auto& component = detail->components[index];
        output << "EXPERIMENT_RECOMMENDATION_SCORE_COMPONENT"
               << ",score_id=" << scoreId
               << ",component_ordinal=" << (index + 1)
               << ",component_name="
               << RecommendationMachineText(component.componentName)
               << ",reason_code="
               << RecommendationMachineText(component.reasonCode)
               << ",input="
               << RecommendationMachineText(component.inputCanonical)
               << ",normalized_value="
               << CanonicalRecommendationDouble(component.normalizedValue)
               << ",weight=" << CanonicalRecommendationDouble(component.weight)
               << ",weighted_contribution="
               << CanonicalRecommendationDouble(component.weightedContribution)
               << ",is_penalty=" << (component.penalty ? 1 : 0)
               << ",explanation="
               << RecommendationMachineText(component.explanation) << '\n';
    }
    return 0;
}

int RunListExperimentRecommendationScoreRunsCommand(
    const std::string& connectionString,
    int limit,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto runs = ListRecommendationScoreRuns(connection, limit);
    for (const auto& run : runs) PrintScoreRunSummary(output, run);
    output << "EXPERIMENT_RECOMMENDATION_SCORE_RUN_LIST_COMPLETE,count="
           << runs.size() << '\n';
    return 0;
}

int RunExperimentRecommendationScoreRunStatusCommand(
    const std::string& connectionString,
    long long scoreRunId,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto run = FindRecommendationScoreRun(connection, scoreRunId);
    if (!run) return 1;
    PrintScoreRunSummary(output, *run);
    output << "EXPERIMENT_RECOMMENDATION_SCORE_RUN_POLICY"
           << ",score_run_id=" << scoreRunId
           << ",scoring_policy_canonical="
           << RecommendationMachineText(run->scoringPolicyCanonical) << '\n';
    return 0;
}

int RunExplainExperimentRecommendationScoreCommand(
    const std::string& connectionString,
    long long scoreId,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto score = FindRecommendationScore(connection, scoreId);
    if (!score) return 1;
    const auto recommendation =
        FindRecommendation(connection, score->recommendationId);
    const auto run = FindRecommendationScoreRun(connection, score->scoreRunId);
    if (!recommendation || !run) return 1;

    output << "Recommendation " << recommendation->recommendationId
           << " proposes changing " << recommendation->changedParameter
           << " from " << recommendation->sourceValueCanonical
           << " to " << recommendation->proposedValueCanonical << ".\n\n"
           << "Advisory score: "
           << CanonicalRecommendationDouble(score->finalScore) << "\n"
           << "Rank: " << score->scoreRank << " of "
           << run->counters.recommendationsScored
           << " (run " << score->scoreRunId << ")\n"
           << "Source experiment: " << score->sourceExperimentId << "\n\n"
           << "Components:\n";
    for (const RecommendationScoreComponent& component : score->components)
    {
        output << "- " << (component.penalty ? "Penalty: " : "Positive: ")
               << component.componentName << " — " << component.explanation
               << " (value "
               << CanonicalRecommendationDouble(component.normalizedValue)
               << ", weight "
               << CanonicalRecommendationDouble(component.weight) << ")\n";
    }
    output << "\nThis score is a deterministic advisory prioritization "
              "heuristic. It does not approve, queue, or predict the "
              "profitability of the experiment.\n";
    return 0;
}

} // namespace EA::ExperimentRecommendation
