#include "ExperimentRecommendationCampaignPlanningRepository.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationConversionWorkflowRepository.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

template <typename Value>
std::optional<Value> OptionalValue(const pqxx::row& row, const char* column)
{
    return row[column].is_null()
        ? std::nullopt
        : std::optional<Value>{row[column].as<Value>()};
}

void RequireIdentity(
    const std::string& canonical,
    const std::string& hash,
    const char* error)
{
    if (canonical.empty() || hash != RecommendationCanonicalHash(canonical))
        throw std::runtime_error(error);
}

std::optional<RecommendationSource::FinalProfitabilityEvidence>
MapFinalProfitabilityEvidence(const pqxx::row& row)
{
    const auto version = OptionalValue<int>(
        row, "final_profitability_provenance_version");
    if (!version) return std::nullopt;
    RecommendationSource::FinalProfitabilityEvidence evidence;
    evidence.provenanceVersion = *version;
    evidence.finalInferenceEvalResultId = OptionalValue<long long>(
        row, "source_final_inference_eval_result_id");
    evidence.profitabilityObservationId = OptionalValue<long long>(
        row, "source_final_profitability_observation_id");
    evidence.inferenceScope = row[
        "source_final_profitability_inference_scope"].as<std::string>();
    evidence.inferenceStart = OptionalValue<std::string>(
        row, "source_final_profitability_inference_start");
    evidence.inferenceEnd = OptionalValue<std::string>(
        row, "source_final_profitability_inference_end");
    evidence.actionablePredictionCount = OptionalValue<long long>(
        row, "source_final_profitability_actionable_count");
    evidence.aggregateTerminalHorizonLogReturnSum = OptionalValue<double>(
        row, "source_final_profitability_aggregate_return");
    evidence.averageTerminalHorizonLogReturnPerActionablePrediction =
        OptionalValue<double>(row,
            "source_final_profitability_average_return");
    evidence.metricDefinitionHash = OptionalValue<std::string>(
        row, "source_final_profitability_metric_definition_hash");
    evidence.sourceContentHash = OptionalValue<std::string>(
        row, "source_final_profitability_source_content_hash");
    evidence.observationIdentityHash = OptionalValue<std::string>(
        row, "source_final_profitability_observation_identity_hash");
    evidence.unavailableReason = row[
        "source_final_profitability_unavailable_reason"].is_null()
        ? ""
        : row["source_final_profitability_unavailable_reason"]
              .as<std::string>();
    if (const auto error =
            ValidateRecommendationFinalProfitabilityEvidence(evidence))
        throw std::runtime_error(*error);
    return evidence;
}

RecommendationCampaignWorkflowEvidence MapWorkflowEvidence(
    const RecommendationConversionWorkflowView& view)
{
    RecommendationCampaignWorkflowEvidence evidence;
    evidence.proposalId = view.proposalId;
    evidence.proposalIdentityCanonical = view.proposalIdentityCanonical;
    evidence.proposalIdentityHash = view.proposalIdentityHash;
    if (view.latestReview)
    {
        evidence.latestReviewDecisionId = view.latestReview->reviewDecisionId;
        evidence.latestReviewDisposition = view.latestReview->decision == "approve"
            ? "approved"
            : view.latestReview->decision == "reject" ? "rejected" : "invalid";
    }
    if (view.execution)
    {
        evidence.executionId = view.execution->executionId;
        evidence.executionIdentityCanonical = view.execution->identityCanonical;
        evidence.executionIdentityHash = view.execution->identityHash;
        evidence.convertedExperimentId = view.execution->experimentId;
    }
    if (view.activation)
    {
        evidence.activationId = view.activation->activationId;
        evidence.activationIdentityCanonical = view.activation->identityCanonical;
        evidence.activationIdentityHash = view.activation->identityHash;
    }
    evidence.state = view.derivation.state;
    evidence.integrity = view.derivation.integrity;
    evidence.diagnosticCodes = view.derivation.diagnosticCodes;
    return evidence;
}

} // namespace

bool RecommendationCampaignPlanningSchemasExist(pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return RecommendationCampaignPlanningSchemasExist(transaction);
}

bool RecommendationCampaignPlanningSchemasExist(
    pqxx::transaction_base& transaction)
{
    return transaction.exec(R"SQL(
SELECT to_regclass('experiment_recommendation_ranking_snapshot') IS NOT NULL
   AND to_regclass('experiment_recommendation_ranking_member') IS NOT NULL
   AND to_regclass('experiment_recommendation') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_proposal') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_review_decision') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_execution') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_activation') IS NOT NULL
   AND to_regclass('experiment') IS NOT NULL;
)SQL").one_row()[0].as<bool>();
}

RecommendationCampaignPlanInput LoadRecommendationCampaignPlanInput(
    pqxx::connection& connection,
    const RecommendationCampaignPlanningPolicy& policy,
    const RecommendationCampaignPlanningScope& scope)
{
    pqxx::read_transaction transaction{connection};
    return LoadRecommendationCampaignPlanInput(transaction, policy, scope);
}

RecommendationCampaignPlanInput LoadRecommendationCampaignPlanInput(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignPlanningPolicy& policy,
    const RecommendationCampaignPlanningScope& scope)
{
    if (const auto error = ValidateRecommendationCampaignPlanningPolicy(policy))
        throw std::invalid_argument(*error);
    if (const auto error = ValidateRecommendationCampaignPlanningScope(scope))
        throw std::invalid_argument(*error);

    const pqxx::result snapshots = transaction.exec(R"SQL(
SELECT status,ranking_snapshot_identity_canonical,
       ranking_snapshot_identity_hash,ranking_policy_canonical,
       ranking_policy_hash,ranking_version,population_semantic_state,
       scoring_semantic_canonical,scoring_semantic_hash,
       scoring_semantic_version,evaluation_semantic_canonical,
       evaluation_semantic_hash,evaluation_semantic_version,
       distinct_scoring_semantic_count,distinct_evaluation_semantic_count,
       homogeneity_validation_result,member_count,
       transaction_timestamp()::text AS read_at
FROM experiment_recommendation_ranking_snapshot
WHERE recommendation_ranking_snapshot_id=$1;
)SQL", pqxx::params{scope.rankingSnapshotId});
    if (snapshots.empty())
        throw std::runtime_error("recommendation_campaign_ranking_snapshot_not_found");
    const pqxx::row snapshot = snapshots.one_row();
    if (snapshot["status"].as<std::string>() != "completed")
        throw std::runtime_error(
            "recommendation_campaign_ranking_snapshot_not_completed");
    const std::string populationState =
        snapshot["population_semantic_state"].as<std::string>();
    const bool emptyPopulation = populationState == "empty";
    if (populationState != "verified_homogeneous" && !emptyPopulation)
        throw std::runtime_error(
            "recommendation_campaign_ranking_semantics_not_verified");
    if (emptyPopulation)
    {
        if (snapshot["member_count"].as<int>() != 0 ||
            !snapshot["scoring_semantic_canonical"].is_null() ||
            !snapshot["evaluation_semantic_canonical"].is_null() ||
            snapshot["distinct_scoring_semantic_count"].as<int>() != 0 ||
            snapshot["distinct_evaluation_semantic_count"].as<int>() != 0 ||
            snapshot["homogeneity_validation_result"].as<std::string>() !=
                "empty_population")
            throw std::runtime_error(
                "recommendation_campaign_empty_ranking_semantics_invalid");
    }
    else
    {
        RequireIdentity(
            snapshot["scoring_semantic_canonical"].as<std::string>(),
            snapshot["scoring_semantic_hash"].as<std::string>(),
            "recommendation_campaign_scoring_semantic_identity_invalid");
        RequireIdentity(
            snapshot["evaluation_semantic_canonical"].as<std::string>(),
            snapshot["evaluation_semantic_hash"].as<std::string>(),
            "recommendation_campaign_evaluation_semantic_identity_invalid");
        if (snapshot["scoring_semantic_version"].as<int>() <= 0 ||
            snapshot["evaluation_semantic_version"].as<int>() <= 0 ||
            snapshot["distinct_scoring_semantic_count"].as<int>() != 1 ||
            snapshot["distinct_evaluation_semantic_count"].as<int>() != 1 ||
            snapshot["homogeneity_validation_result"].as<std::string>() !=
                "verified_homogeneous")
            throw std::runtime_error(
                "recommendation_campaign_ranking_semantic_shape_invalid");
    }

    RecommendationCampaignPlanInput input;
    input.policy = policy;
    input.scope = scope;
    input.rankingSnapshotIdentityCanonical =
        snapshot["ranking_snapshot_identity_canonical"].as<std::string>();
    input.rankingSnapshotIdentityHash =
        snapshot["ranking_snapshot_identity_hash"].as<std::string>();
    input.generatedAt = snapshot["read_at"].as<std::string>();
    const std::string rankingPolicyCanonical =
        snapshot["ranking_policy_canonical"].as<std::string>();
    const std::string rankingPolicyHash =
        snapshot["ranking_policy_hash"].as<std::string>();
    const int rankingVersion = snapshot["ranking_version"].as<int>();
    RequireIdentity(
        input.rankingSnapshotIdentityCanonical,
        input.rankingSnapshotIdentityHash,
        "recommendation_campaign_snapshot_identity_invalid");
    RequireIdentity(
        rankingPolicyCanonical,
        rankingPolicyHash,
        "recommendation_campaign_ranking_policy_identity_invalid");

    std::string sql = R"SQL(
SELECT rm.recommendation_ranking_member_id,rm.global_ordinal,rm.bucket,
       rm.final_score,rm.recommendation_id,rm.source_experiment_id,
       rm.source_model_id,
       rm.symbol,rm.horizon,rm.family,e.target_epochs,
       r.source_experiment_id AS recommendation_source_experiment_id,
       r.source_model_id AS recommendation_source_model_id,
       r.source_symbol AS recommendation_symbol,
       r.source_prediction_horizon AS recommendation_horizon,
       r.changed_parameter AS recommendation_family,
       r.source_leader_score,r.source_infer_accuracy,
       r.source_predicted_neutral_proportion,
       r.semantic_configuration_canonical,r.semantic_hash,
       r.invocation_configuration_canonical,r.invocation_hash,
       r.final_profitability_provenance_version,
       r.source_final_inference_eval_result_id,
       r.source_final_profitability_observation_id,
       r.source_final_profitability_unavailable_reason,
       r.source_final_profitability_inference_scope,
       r.source_final_profitability_inference_start,
       r.source_final_profitability_inference_end,
       r.source_final_profitability_actionable_count,
       r.source_final_profitability_aggregate_return,
       r.source_final_profitability_average_return,
       r.source_final_profitability_metric_definition_hash,
       r.source_final_profitability_source_content_hash,
       r.source_final_profitability_observation_identity_hash
FROM experiment_recommendation_ranking_member rm
JOIN experiment_recommendation r ON r.recommendation_id=rm.recommendation_id
JOIN experiment e ON e.experiment_id=rm.source_experiment_id
WHERE rm.recommendation_ranking_snapshot_id=$1
)SQL";
    pqxx::params parameters{scope.rankingSnapshotId};
    int parameter = 2;
    if (scope.symbol)
    {
        sql += " AND rm.symbol=$" + std::to_string(parameter++);
        parameters.append(*scope.symbol);
    }
    if (scope.horizon)
    {
        sql += " AND rm.horizon=$" + std::to_string(parameter++);
        parameters.append(*scope.horizon);
    }
    if (scope.recommendationId)
    {
        sql += " AND rm.recommendation_id=$" + std::to_string(parameter++);
        parameters.append(*scope.recommendationId);
    }
    sql += " ORDER BY rm.global_ordinal,rm.recommendation_id LIMIT 1001;";

    const pqxx::result rows = transaction.exec(sql, parameters);
    if (rows.size() > kMaximumRecommendationCampaignCandidates)
        throw std::runtime_error(
            "recommendation_campaign_repository_input_limit_exceeded");

    std::vector<long long> recommendationIds;
    recommendationIds.reserve(rows.size());
    for (const pqxx::row& row : rows)
    {
        RecommendationCampaignCandidateInput candidate;
        candidate.rankingSnapshotId = scope.rankingSnapshotId;
        candidate.rankingSnapshotIdentityCanonical =
            input.rankingSnapshotIdentityCanonical;
        candidate.rankingSnapshotIdentityHash =
            input.rankingSnapshotIdentityHash;
        candidate.rankingPolicyCanonical = rankingPolicyCanonical;
        candidate.rankingPolicyHash = rankingPolicyHash;
        candidate.rankingVersion = rankingVersion;
        candidate.rankingMemberId =
            row["recommendation_ranking_member_id"].as<long long>();
        candidate.rankingPosition = row["global_ordinal"].as<int>();
        candidate.rankingBucket = row["bucket"].as<std::string>();
        candidate.rankingScore = OptionalValue<double>(row, "final_score");
        candidate.recommendationId = row["recommendation_id"].as<long long>();
        candidate.sourceExperimentId =
            row["source_experiment_id"].as<long long>();
        candidate.sourceModelId = OptionalValue<long long>(
            row, "source_model_id");
        candidate.symbol = row["symbol"].as<std::string>();
        candidate.predictionHorizon = row["horizon"].as<int>();
        candidate.family = row["family"].as<std::string>();
        candidate.targetEpochs = OptionalValue<int>(row, "target_epochs");
        candidate.leaderScore = row["source_leader_score"].as<double>();
        candidate.inferenceAccuracy =
            row["source_infer_accuracy"].as<double>();
        candidate.predictedNeutralProportion =
            OptionalValue<double>(row, "source_predicted_neutral_proportion");
        candidate.recommendationSemanticCanonical =
            row["semantic_configuration_canonical"].as<std::string>();
        candidate.recommendationSemanticHash =
            row["semantic_hash"].as<std::string>();
        candidate.recommendationInvocationCanonical =
            row["invocation_configuration_canonical"].as<std::string>();
        candidate.recommendationInvocationHash =
            row["invocation_hash"].as<std::string>();
        candidate.persistedProvenanceValid =
            candidate.sourceExperimentId ==
                row["recommendation_source_experiment_id"].as<long long>() &&
            candidate.sourceModelId == OptionalValue<long long>(
                row, "recommendation_source_model_id") &&
            candidate.symbol ==
                row["recommendation_symbol"].as<std::string>() &&
            candidate.predictionHorizon ==
                row["recommendation_horizon"].as<int>() &&
            candidate.family ==
                row["recommendation_family"].as<std::string>();
        candidate.finalProfitabilityEvidence =
            MapFinalProfitabilityEvidence(row);
        recommendationIds.push_back(candidate.recommendationId);
        input.candidates.push_back(std::move(candidate));
    }

    std::sort(recommendationIds.begin(), recommendationIds.end());
    recommendationIds.erase(
        std::unique(recommendationIds.begin(), recommendationIds.end()),
        recommendationIds.end());
    const auto workflows =
        ListRecommendationConversionWorkflowsForRecommendations(
            transaction, recommendationIds);
    std::map<long long, std::vector<RecommendationCampaignWorkflowEvidence>>
        workflowsByRecommendation;
    for (const auto& workflow : workflows)
        workflowsByRecommendation[workflow.recommendationId].push_back(
            MapWorkflowEvidence(workflow));
    for (auto& candidate : input.candidates)
    {
        const auto found = workflowsByRecommendation.find(
            candidate.recommendationId);
        if (found != workflowsByRecommendation.end())
            candidate.workflows = found->second;
    }
    return input;
}

} // namespace EA::ExperimentRecommendation
