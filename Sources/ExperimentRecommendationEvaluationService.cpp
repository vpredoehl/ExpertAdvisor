#include "ExperimentRecommendationEvaluationService.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationService.hpp"

#include <algorithm>
#include <locale>
#include <sstream>
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

std::string OptionalScore(const std::optional<double>& value)
{
    return value ? CanonicalRecommendationDouble(*value) : "NULL";
}

void PrintProfitability(
    std::ostream& output,
    const std::optional<RecommendationSource::FinalProfitabilityEvidence>&
        evidence,
    const std::string* evidenceHash)
{
    output << ",final_profitability_evidence="
           << (!evidence ? "legacy"
                         : evidence->Available() ? "available" : "unavailable")
           << ",final_profitability_unavailable_reason="
           << (!evidence || evidence->unavailableReason.empty()
                   ? "NULL"
                   : RecommendationMachineText(evidence->unavailableReason))
           << ",final_inference_eval_result_id="
           << (evidence
                   ? OptionalNumber(evidence->finalInferenceEvalResultId)
                   : "NULL")
           << ",final_profitability_observation_id="
           << (evidence
                   ? OptionalNumber(evidence->profitabilityObservationId)
                   : "NULL")
           << ",final_profitability_inference_scope="
           << (evidence
                   ? RecommendationMachineText(evidence->inferenceScope)
                   : "NULL")
           << ",final_profitability_actionable_count="
           << (evidence
                   ? OptionalNumber(evidence->actionablePredictionCount)
                   : "NULL")
           << ",final_profitability_aggregate_terminal_horizon_log_return_sum="
           << (evidence && evidence->aggregateTerminalHorizonLogReturnSum
                   ? CanonicalRecommendationDouble(
                         *evidence->aggregateTerminalHorizonLogReturnSum)
                   : "NULL")
           << ",final_profitability_average_terminal_horizon_log_return_per_actionable_prediction="
           << (evidence &&
                       evidence->averageTerminalHorizonLogReturnPerActionablePrediction
                   ? CanonicalRecommendationDouble(
                         *evidence->averageTerminalHorizonLogReturnPerActionablePrediction)
                   : "NULL")
           << ",profitability_evidence_hash="
           << (evidenceHash ? RecommendationMachineText(*evidenceHash) : "NULL")
           << ",profitability_weight=0,profitability_score_contribution=0";
}

std::string SnapshotCanonical(
    const RecommendationEvaluationPolicy& policy,
    const RecommendationEvaluationFilters& filters,
    const std::vector<RecommendationEvaluationResult>& results)
{
    std::vector<std::string> identities;
    identities.reserve(results.size());
    for (const auto& result : results)
        identities.push_back(result.evaluationIdentityCanonical);
    std::sort(identities.begin(), identities.end());
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_evaluation_snapshot_v1"
        << ";policy=" << RecommendationEvaluationPolicyCanonicalText(policy).size()
        << ":" << RecommendationEvaluationPolicyCanonicalText(policy)
        << ";scan=" << OptionalNumber(filters.recommendationScanId)
        << ";recommendation=" << OptionalNumber(filters.recommendationId)
        << ";limit=" << filters.limit
        << ";count=" << identities.size();
    for (std::size_t index = 0; index < identities.size(); ++index)
        out << ";identity[" << index << "]=" << identities[index].size()
            << ":" << identities[index];
    return out.str();
}

void PrintSafety(std::ostream& output)
{
    output << "experiment_created=false,experiment_queued=false,"
              "scheduler_modified=false";
}

void PrintEvaluation(
    std::ostream& output,
    const RecommendationEvaluationResult& result,
    std::optional<long long> runId,
    std::optional<long long> resultId,
    bool persisted)
{
    output << "EXPERIMENT_RECOMMENDATION_EVALUATION"
           << ",evaluation_run_id=" << OptionalNumber(runId)
           << ",evaluation_result_id=" << OptionalNumber(resultId)
           << ",recommendation_id=" << result.recommendationId
           << ",recommendation_semantic_hash="
           << RecommendationMachineText(result.recommendationSemanticHash)
           << ",evaluation_identity_hash="
           << RecommendationMachineText(result.evaluationIdentityHash)
           << ",evaluation_policy_hash="
           << RecommendationMachineText(result.evaluationPolicyHash)
           << ",evaluation_version=" << result.evaluationVersion
           << ",evaluator_version=" << result.evaluatorVersion
           << ",scoring_policy_hash="
           << RecommendationMachineText(result.scoringPolicyHash)
           << ",scoring_policy_version=" << result.scoringVersion
           << ",eligibility="
           << RecommendationEligibilityText(result.eligibility)
           << ",disposition="
           << RecommendationEvaluationDispositionText(result.disposition)
           << ",final_score=" << OptionalScore(result.finalScore)
           << ",component_count=" << result.components.size()
           << ",missing_evidence_count=" << result.missingEvidenceCount
           << ",block_reason="
           << (result.eligibility == RecommendationEligibility::ineligible
                   ? RecommendationMachineText(result.reasonCode) : "NULL")
           << ",explanation="
           << RecommendationMachineText(result.explanation)
           << ",persisted=" << (persisted ? "true" : "false");
    PrintProfitability(output, result.finalProfitabilityEvidence,
                       &result.profitabilityEvidenceHash);
    output << ',';
    PrintSafety(output);
    output << '\n';
}

void PrintPersistedSummary(
    std::ostream& output,
    const PersistedRecommendationEvaluationSummary& result)
{
    output << "EXPERIMENT_RECOMMENDATION_EVALUATION"
           << ",evaluation_run_id=" << result.evaluationRunId
           << ",evaluation_result_id=" << result.evaluationResultId
           << ",recommendation_id=" << result.recommendationId
           << ",recommendation_scan_id=" << result.recommendationScanId
           << ",source_experiment_id=" << result.sourceExperimentId
           << ",symbol=" << RecommendationMachineText(result.sourceSymbol)
           << ",horizon=" << result.sourcePredictionHorizon
           << ",family=" << RecommendationMachineText(result.changedParameter)
           << ",recommendation_semantic_hash="
           << RecommendationMachineText(result.recommendationSemanticHash)
           << ",evaluation_identity_hash="
           << RecommendationMachineText(result.evaluationIdentityHash)
           << ",evaluation_policy_hash="
           << RecommendationMachineText(result.evaluationPolicyHash)
           << ",evaluation_version=" << result.evaluationVersion
           << ",evaluator_version=" << result.evaluatorVersion
           << ",scoring_policy_hash="
           << RecommendationMachineText(result.scoringPolicyHash)
           << ",scoring_policy_version=" << result.scoringVersion
           << ",eligibility=" << RecommendationEligibilityText(result.eligibility)
           << ",disposition="
           << RecommendationEvaluationDispositionText(result.disposition)
           << ",final_score=" << OptionalScore(result.finalScore)
           << ",component_count=" << result.componentCount
           << ",missing_evidence_count=" << result.missingEvidenceCount
           << ",block_reason="
           << (result.eligibility == RecommendationEligibility::ineligible
                   ? RecommendationMachineText(result.reasonCode) : "NULL")
           << ",explanation=" << RecommendationMachineText(result.explanation)
           << ",ranking_ordinal=" << result.rankingOrdinal
           << ",created_at=" << RecommendationMachineText(result.createdAt)
           << ",persisted=true";
    const std::string* profitabilityHash = result.profitabilityEvidenceHash
        ? &*result.profitabilityEvidenceHash : nullptr;
    PrintProfitability(output, result.finalProfitabilityEvidence,
                       profitabilityHash);
    output << ',';
    PrintSafety(output);
    output << '\n';
}

void PrintHumanSummary(
    std::ostream& output,
    const RecommendationEvaluationResult& result,
    const RecommendationEvaluationInput& input)
{
    output << "Recommendation " << result.recommendationId
           << " was evaluated for advisory use.\n\n"
           << "Symbol: " << RecommendationHumanText(input.sourceSymbol) << '\n'
           << "Horizon: " << input.scoringInput.sourcePredictionHorizon << '\n'
           << "Family: "
           << RecommendationHumanText(input.scoringInput.changedParameter) << '\n'
           << "Source experiment: " << input.scoringInput.sourceExperimentId << '\n'
           << "Source model: " << OptionalNumber(input.sourceModelId) << '\n'
           << "Source analysis: " << OptionalNumber(input.sourceAnalysisId) << '\n'
           << "Evaluation status: "
           << RecommendationEvaluationDispositionText(result.disposition) << '\n'
           << "Final advisory score: " << OptionalScore(result.finalScore) << '\n'
           << "Missing evidence: " << result.missingEvidenceCount << '\n'
           << "Explanation: " << RecommendationHumanText(result.explanation)
           << '\n';
    bool wrotePositive = false;
    for (const auto& component : result.components)
    {
        if (component.penalty) continue;
        output << (wrotePositive ? ", " : "Positive components: ")
               << RecommendationHumanText(component.componentName) << '='
               << CanonicalRecommendationDouble(component.weightedContribution);
        wrotePositive = true;
    }
    if (wrotePositive) output << '\n';
    bool wrotePenalty = false;
    for (const auto& component : result.components)
    {
        if (!component.penalty) continue;
        output << (wrotePenalty ? ", " : "Penalties: ")
               << RecommendationHumanText(component.componentName) << '='
               << CanonicalRecommendationDouble(component.weightedContribution);
        wrotePenalty = true;
    }
    if (wrotePenalty) output << '\n';
    output << "\nAdvisory evaluation only. No experiment was created or queued; "
              "scheduler state was not changed.\n";
}

void PrintRun(
    std::ostream& output,
    const PersistedRecommendationEvaluationRun& run)
{
    output << "EXPERIMENT_RECOMMENDATION_EVALUATION_RUN"
           << ",evaluation_run_id=" << run.evaluationRunId
           << ",status=" << RecommendationMachineText(run.status)
           << ",run_identity_hash=" << RecommendationMachineText(run.runIdentityHash)
           << ",evaluation_policy_hash="
           << RecommendationMachineText(run.evaluationPolicyHash)
           << ",evaluation_version=" << run.evaluationVersion
           << ",evaluator_version=" << run.evaluatorVersion
           << ",scoring_policy_hash="
           << RecommendationMachineText(run.scoringPolicyHash)
           << ",scoring_version=" << run.scoringVersion
           << ",recommendation_scan_filter="
           << OptionalNumber(run.recommendationScanFilter)
           << ",recommendation_id_filter="
           << OptionalNumber(run.recommendationIdFilter)
           << ",requested_limit=" << OptionalNumber(run.requestedLimit)
           << ",recommendations_considered="
           << run.counters.recommendationsConsidered
           << ",recommendations_evaluated="
           << run.counters.recommendationsEvaluated
           << ",recommendations_eligible="
           << run.counters.recommendationsEligible
           << ",recommendations_blocked="
           << run.counters.recommendationsBlocked
           << ",evaluation_errors=" << run.counters.evaluationErrors
           << ",started_at=" << RecommendationMachineText(run.startedAt)
           << ",completed_at="
           << (run.completedAt ? RecommendationMachineText(*run.completedAt) : "NULL")
           << ",error_message="
           << (run.errorMessage ? RecommendationMachineText(*run.errorMessage) : "NULL")
           << ',';
    PrintSafety(output);
    output << '\n';
}

const RecommendationEvaluationInput& FindInput(
    const std::vector<RecommendationEvaluationLoadResult>& loaded,
    long long recommendationId)
{
    const auto found = std::find_if(loaded.begin(), loaded.end(),
        [recommendationId](const auto& item) {
            return item.recommendationId == recommendationId;
        });
    if (found == loaded.end())
        throw std::runtime_error("recommendation_evaluation_input_not_found");
    return found->input;
}

} // namespace

int RunEvaluateExperimentRecommendationsCommand(
    const std::string& connectionString,
    const RecommendationEvaluationCommandRequest& request,
    std::ostream& output,
    std::ostream& errors)
{
    if (const auto error = ValidateRecommendationEvaluationPolicy(request.policy))
        throw std::invalid_argument(*error);
    RecommendationEvaluationFilters filters;
    filters.recommendationScanId = request.recommendationScanId;
    filters.recommendationId = request.recommendationId;
    filters.limit = request.limit;
    pqxx::connection connection{connectionString};
    if (!request.dryRun && !RecommendationEvaluationSchemaExists(connection))
        throw std::runtime_error("recommendation_evaluation_schema_unavailable");
    const auto loaded = LoadRecommendationsForEvaluation(connection, filters);
    std::vector<RecommendationEvaluationResult> evaluated;
    evaluated.reserve(loaded.size());
    for (const auto& item : loaded)
        evaluated.push_back(EvaluateExperimentRecommendation(
            request.policy, item.input));
    evaluated = RankRecommendationEvaluations(std::move(evaluated));
    const std::string snapshot = SnapshotCanonical(request.policy, filters, evaluated);
    const std::string snapshotHash = RecommendationEvaluationCanonicalHash(snapshot);
    const std::string runIdentity =
        "experiment_recommendation_evaluation_run_identity_v1;snapshot=" +
        std::to_string(snapshot.size()) + ":" + snapshot;
    const std::string runHash = RecommendationEvaluationCanonicalHash(runIdentity);

    output << "EXPERIMENT_RECOMMENDATION_EVALUATION_START"
           << ",evaluation_run_id=NULL,evaluation_policy_hash="
           << RecommendationEvaluationPolicyHash(request.policy)
           << ",scoring_policy_version="
           << request.policy.scoringPolicy.scoringVersion
           << ",evaluator_version=" << request.policy.evaluatorVersion
           << ",dry_run=" << (request.dryRun ? "true" : "false") << ',';
    PrintSafety(output);
    output << '\n';

    if (request.dryRun)
    {
        for (std::size_t index = 0; index < evaluated.size(); ++index)
        {
            PrintEvaluation(output, evaluated[index], std::nullopt,
                            std::nullopt, false);
            PrintHumanSummary(output, evaluated[index],
                              FindInput(loaded, evaluated[index].recommendationId));
        }
        output << "EXPERIMENT_RECOMMENDATION_EVALUATION_COMPLETE"
               << ",evaluation_run_id=NULL,count=" << evaluated.size()
               << ",persisted=false,";
        PrintSafety(output);
        output << '\n';
        return 0;
    }

    RecommendationEvaluationRunCounters counters;
    RecommendationEvaluationRunBeginResult run;
    try
    {
        run = BeginOrFindRecommendationEvaluationRun(connection, {
            request.policy, filters, runIdentity, runHash, snapshot, snapshotHash});
        output << "EXPERIMENT_RECOMMENDATION_EVALUATION_RUN_START"
               << ",evaluation_run_id=" << run.evaluationRunId
               << ",created=" << (run.created ? "true" : "false")
               << ",status=" << RecommendationMachineText(run.status) << ',';
        PrintSafety(output);
        output << '\n';
        if (run.status == "failed")
            throw std::runtime_error("recommendation_evaluation_run_failed");
        if (run.status != "running" && run.status != "completed")
            throw std::runtime_error(
                "invalid_recommendation_evaluation_run_status");
        for (std::size_t index = 0; index < evaluated.size(); ++index)
        {
            ++counters.recommendationsConsidered;
            const RecommendationEvaluationPersistResult persisted =
                PersistRecommendationEvaluation(connection, {
                    run.evaluationRunId,
                    FindInput(loaded, evaluated[index].recommendationId),
                    evaluated[index]});
            ++counters.recommendationsEvaluated;
            if (evaluated[index].eligibility == RecommendationEligibility::eligible)
                ++counters.recommendationsEligible;
            else
                ++counters.recommendationsBlocked;
            PrintEvaluation(output, evaluated[index], run.evaluationRunId,
                            persisted.evaluationResultId, true);
            PrintHumanSummary(output, evaluated[index],
                              FindInput(loaded, evaluated[index].recommendationId));
        }
        CompleteRecommendationEvaluationRun(
            connection, run.evaluationRunId, counters);
        output << "EXPERIMENT_RECOMMENDATION_EVALUATION_COMPLETE"
               << ",evaluation_run_id=" << run.evaluationRunId
               << ",count=" << counters.recommendationsEvaluated
               << ",persisted=true,";
        PrintSafety(output);
        output << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        ++counters.evaluationErrors;
        if (run.evaluationRunId > 0 && run.status == "running")
        {
            try { FailRecommendationEvaluationRun(
                connection, run.evaluationRunId, counters, error.what()); }
            catch (...) {}
        }
        errors << "EXPERIMENT_RECOMMENDATION_EVALUATION_FAILED"
               << ",evaluation_run_id="
               << (run.evaluationRunId > 0 ? std::to_string(run.evaluationRunId)
                                           : "NULL")
               << ",reason=" << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors);
        errors << '\n';
        return 2;
    }
}

int RunListExperimentRecommendationEvaluationsCommand(
    const std::string& connectionString,
    const RecommendationEvaluationFilters& filters,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto values = ListRecommendationEvaluations(connection, filters);
    for (const auto& value : values) PrintPersistedSummary(output, value);
    output << "EXPERIMENT_RECOMMENDATION_EVALUATION_LIST_COMPLETE,count="
           << values.size() << ',';
    PrintSafety(output);
    output << '\n';
    return 0;
}

int RunExperimentRecommendationEvaluationStatusCommand(
    const std::string& connectionString,
    long long evaluationResultId,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto value = FindRecommendationEvaluation(connection, evaluationResultId);
    if (!value)
    {
        output << "EXPERIMENT_RECOMMENDATION_EVALUATION_NOT_FOUND,evaluation_result_id="
               << evaluationResultId << '\n';
        return 3;
    }
    PrintPersistedSummary(output, *value);
    return 0;
}

int RunExplainExperimentRecommendationEvaluationCommand(
    const std::string& connectionString,
    long long evaluationResultId,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto value = FindRecommendationEvaluation(connection, evaluationResultId);
    if (!value) return 3;
    PrintPersistedSummary(output, *value);
    int ordinal = 0;
    for (const auto& component : value->components)
        output << "EXPERIMENT_RECOMMENDATION_EVALUATION_COMPONENT"
               << ",evaluation_result_id=" << evaluationResultId
               << ",component_ordinal=" << ++ordinal
               << ",component_name="
               << RecommendationMachineText(component.componentName)
               << ",reason_code=" << RecommendationMachineText(component.reasonCode)
               << ",input=" << RecommendationMachineText(component.inputCanonical)
               << ",normalized_value="
               << CanonicalRecommendationDouble(component.normalizedValue)
               << ",weight=" << CanonicalRecommendationDouble(component.weight)
               << ",contribution="
               << CanonicalRecommendationDouble(component.weightedContribution)
               << ",is_penalty=" << (component.penalty ? "true" : "false")
               << ",is_missing=false,explanation="
               << RecommendationMachineText(component.explanation) << '\n';
    output << "EXPERIMENT_RECOMMENDATION_EVALUATION_EXPLANATION_COMPLETE"
           << ",evaluation_result_id=" << evaluationResultId
           << ",component_count=" << value->components.size() << ',';
    PrintSafety(output);
    output << '\n';
    return 0;
}

int RunListExperimentRecommendationEvaluationRunsCommand(
    const std::string& connectionString,
    int limit,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto runs = ListRecommendationEvaluationRuns(connection, limit);
    for (const auto& run : runs) PrintRun(output, run);
    output << "EXPERIMENT_RECOMMENDATION_EVALUATION_RUN_LIST_COMPLETE,count="
           << runs.size() << '\n';
    return 0;
}

int RunExperimentRecommendationEvaluationRunStatusCommand(
    const std::string& connectionString,
    long long evaluationRunId,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    const auto run = FindRecommendationEvaluationRun(connection, evaluationRunId);
    if (!run) return 3;
    PrintRun(output, *run);
    return 0;
}

} // namespace EA::ExperimentRecommendation
