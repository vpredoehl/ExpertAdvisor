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
    if (const auto error = ValidateRecommendationCampaignPlanningPolicy(policy))
        throw std::invalid_argument(*error);
    if (const auto error = ValidateRecommendationCampaignPlanningScope(scope))
        throw std::invalid_argument(*error);

    pqxx::read_transaction transaction{connection};
    const pqxx::result snapshots = transaction.exec(R"SQL(
SELECT status,ranking_snapshot_identity_canonical,
       ranking_snapshot_identity_hash,ranking_policy_canonical,
       ranking_policy_hash,ranking_version,transaction_timestamp()::text AS read_at
FROM experiment_recommendation_ranking_snapshot
WHERE recommendation_ranking_snapshot_id=$1;
)SQL", pqxx::params{scope.rankingSnapshotId});
    if (snapshots.empty())
        throw std::runtime_error("recommendation_campaign_ranking_snapshot_not_found");
    const pqxx::row snapshot = snapshots.one_row();
    if (snapshot["status"].as<std::string>() != "completed")
        throw std::runtime_error(
            "recommendation_campaign_ranking_snapshot_not_completed");

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
       rm.symbol,rm.horizon,rm.family,e.target_epochs,
       r.source_experiment_id AS recommendation_source_experiment_id,
       r.source_symbol AS recommendation_symbol,
       r.source_prediction_horizon AS recommendation_horizon,
       r.changed_parameter AS recommendation_family,
       r.source_leader_score,r.source_infer_accuracy,
       r.source_predicted_neutral_proportion,
       r.semantic_configuration_canonical,r.semantic_hash,
       r.invocation_configuration_canonical,r.invocation_hash
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
            candidate.symbol ==
                row["recommendation_symbol"].as<std::string>() &&
            candidate.predictionHorizon ==
                row["recommendation_horizon"].as<int>() &&
            candidate.family ==
                row["recommendation_family"].as<std::string>();
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
