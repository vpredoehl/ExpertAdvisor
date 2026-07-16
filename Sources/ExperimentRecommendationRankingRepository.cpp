#include "ExperimentRecommendationRankingRepository.hpp"

#include <algorithm>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

template <typename Value>
std::optional<Value> OptionalValue(const pqxx::row& row, const char* name)
{
    return row[name].is_null() ? std::nullopt
                               : std::optional<Value>{row[name].as<Value>()};
}

bool TableExists(pqxx::transaction_base& transaction, const std::string& table)
{
    return transaction.exec(
        "SELECT to_regclass(current_schema() || '.' || $1) IS NOT NULL;",
        pqxx::params{table}).one_row()[0].as<bool>();
}

RecommendationEligibility ParseEligibility(const std::string& value)
{
    if (value == "eligible") return RecommendationEligibility::eligible;
    if (value == "ineligible") return RecommendationEligibility::ineligible;
    throw std::runtime_error("invalid_persisted_ranking_eligibility");
}

RecommendationRankingScopeType ParseScopeType(const std::string& value)
{
    if (value == "evaluation_run") return RecommendationRankingScopeType::evaluationRun;
    if (value == "recommendation_scan") return RecommendationRankingScopeType::recommendationScan;
    if (value == "symbol") return RecommendationRankingScopeType::symbol;
    if (value == "horizon") return RecommendationRankingScopeType::horizon;
    if (value == "family") return RecommendationRankingScopeType::family;
    if (value == "symbol_horizon") return RecommendationRankingScopeType::symbolHorizon;
    if (value == "global") return RecommendationRankingScopeType::global;
    throw std::runtime_error("invalid_persisted_ranking_scope_type");
}

std::string EvaluationColumns()
{
    return "er.recommendation_evaluation_result_id,"
        "er.recommendation_evaluation_run_id,er.recommendation_id,"
        "er.recommendation_scan_id,er.source_experiment_id,er.source_model_id,"
        "er.source_analysis_id,r.source_symbol,r.source_prediction_horizon,"
        "r.changed_parameter,r.source_value_canonical,r.proposed_value_canonical,"
        "er.recommendation_semantic_hash,er.evaluation_identity_canonical,"
        "er.evaluation_identity_hash,run.evaluation_policy_canonical,"
        "run.evaluation_policy_hash,run.evaluation_version,run.evaluator_version,"
        "run.scoring_policy_canonical,run.scoring_policy_hash,run.scoring_version,"
        "er.eligibility,er.disposition,er.reason_code,er.explanation,"
        "er.final_score,er.component_count,er.missing_evidence_count,"
        "ec.component_ordinal,ec.component_name,ec.reason_code AS component_reason_code,"
        "ec.input_canonical,ec.normalized_value,ec.weight,"
        "ec.weighted_contribution,ec.is_penalty,ec.explanation AS component_explanation";
}

RecommendationRankingEvaluation MapEvaluation(const pqxx::row& row)
{
    RecommendationRankingEvaluation value;
    value.evaluationResultId =
        row["recommendation_evaluation_result_id"].as<long long>();
    value.evaluationRunId =
        row["recommendation_evaluation_run_id"].as<long long>();
    value.recommendationId = row["recommendation_id"].as<long long>();
    value.recommendationScanId = row["recommendation_scan_id"].as<long long>();
    value.sourceExperimentId = row["source_experiment_id"].as<long long>();
    value.sourceModelId = OptionalValue<long long>(row, "source_model_id");
    value.sourceAnalysisId = OptionalValue<long long>(row, "source_analysis_id");
    value.symbol = row["source_symbol"].as<std::string>();
    value.horizon = row["source_prediction_horizon"].as<int>();
    value.family = row["changed_parameter"].as<std::string>();
    value.sourceValueCanonical = row["source_value_canonical"].as<std::string>();
    value.proposedValueCanonical = row["proposed_value_canonical"].as<std::string>();
    value.recommendationSemanticHash =
        row["recommendation_semantic_hash"].as<std::string>();
    value.evaluationIdentityCanonical =
        row["evaluation_identity_canonical"].as<std::string>();
    value.evaluationIdentityHash =
        row["evaluation_identity_hash"].as<std::string>();
    value.evaluationPolicyCanonical =
        row["evaluation_policy_canonical"].as<std::string>();
    value.evaluationPolicyHash = row["evaluation_policy_hash"].as<std::string>();
    value.evaluationVersion = row["evaluation_version"].as<int>();
    value.evaluatorVersion = row["evaluator_version"].as<int>();
    value.scoringPolicyCanonical =
        row["scoring_policy_canonical"].as<std::string>();
    value.scoringPolicyHash = row["scoring_policy_hash"].as<std::string>();
    value.scoringVersion = row["scoring_version"].as<int>();
    value.eligibility = ParseEligibility(row["eligibility"].as<std::string>());
    const auto disposition = ParseRecommendationEvaluationDisposition(
        row["disposition"].as<std::string>());
    if (!disposition)
        throw std::runtime_error("invalid_persisted_ranking_disposition");
    value.disposition = *disposition;
    value.reasonCode = row["reason_code"].as<std::string>();
    value.explanation = row["explanation"].as<std::string>();
    value.finalScore = OptionalValue<double>(row, "final_score");
    value.componentCount = row["component_count"].as<int>();
    value.missingEvidenceCount = row["missing_evidence_count"].as<int>();
    return value;
}

void AddComponent(RecommendationRankingEvaluation& value, const pqxx::row& row)
{
    if (row["component_ordinal"].is_null()) return;
    value.components.push_back({
        row["component_name"].as<std::string>(),
        row["component_reason_code"].as<std::string>(),
        row["input_canonical"].as<std::string>(),
        row["normalized_value"].as<double>(),
        row["weight"].as<double>(),
        row["weighted_contribution"].as<double>(),
        row["is_penalty"].as<bool>(),
        row["component_explanation"].as<std::string>()});
}

std::vector<RecommendationRankingEvaluation> MapEvaluations(
    const pqxx::result& rows)
{
    std::vector<RecommendationRankingEvaluation> values;
    for (const pqxx::row& row : rows)
    {
        const long long id =
            row["recommendation_evaluation_result_id"].as<long long>();
        if (values.empty() || values.back().evaluationResultId != id)
        {
            if (values.size() >=
                static_cast<std::size_t>(kMaximumRecommendationRankingInputs))
                throw std::invalid_argument(
                    "recommendation_ranking_input_limit_exceeded");
            values.push_back(MapEvaluation(row));
        }
        AddComponent(values.back(), row);
    }
    return values;
}

std::string EvaluationFromClause()
{
    return " FROM experiment_recommendation_evaluation_result er "
        "JOIN experiment_recommendation_evaluation_run run ON "
        "run.recommendation_evaluation_run_id=er.recommendation_evaluation_run_id "
        "JOIN experiment_recommendation r ON r.recommendation_id=er.recommendation_id "
        "LEFT JOIN experiment_recommendation_evaluation_component ec ON "
        "ec.recommendation_evaluation_result_id=er.recommendation_evaluation_result_id ";
}

RecommendationRankingScope MapScope(const pqxx::row& row)
{
    RecommendationRankingScope scope;
    scope.type = ParseScopeType(row["scope_type"].as<std::string>());
    scope.evaluationRunId = OptionalValue<long long>(row, "evaluation_run_filter");
    scope.recommendationScanId = OptionalValue<long long>(
        row, "recommendation_scan_filter");
    scope.symbol = OptionalValue<std::string>(row, "symbol_filter");
    scope.horizon = OptionalValue<int>(row, "horizon_filter");
    scope.family = OptionalValue<std::string>(row, "family_filter");
    if (const auto error = ValidateRecommendationRankingScope(scope))
        throw std::runtime_error("invalid_persisted_ranking_scope:" + *error);
    return scope;
}

std::string SnapshotColumns()
{
    return "recommendation_ranking_snapshot_id,status,"
        "ranking_snapshot_identity_canonical,ranking_snapshot_identity_hash,"
        "ranking_policy_canonical,ranking_policy_hash,ranking_version,"
        "scope_type,scope_canonical,scope_hash,evaluation_run_filter,"
        "recommendation_scan_filter,symbol_filter,horizon_filter,family_filter,"
        "requested_limit,source_membership_canonical,source_membership_hash,"
        "member_count,advisory_ready_count,blocked_count,non_actionable_count,"
        "started_at::text AS started_at,completed_at::text AS completed_at,"
        "error_message";
}

PersistedRecommendationRankingSnapshot MapSnapshot(const pqxx::row& row)
{
    PersistedRecommendationRankingSnapshot value;
    value.snapshotId = row["recommendation_ranking_snapshot_id"].as<long long>();
    value.status = row["status"].as<std::string>();
    value.snapshotIdentityCanonical =
        row["ranking_snapshot_identity_canonical"].as<std::string>();
    value.snapshotIdentityHash =
        row["ranking_snapshot_identity_hash"].as<std::string>();
    value.rankingPolicyCanonical =
        row["ranking_policy_canonical"].as<std::string>();
    value.rankingPolicyHash = row["ranking_policy_hash"].as<std::string>();
    value.rankingVersion = row["ranking_version"].as<int>();
    value.scope = MapScope(row);
    value.scopeCanonical = row["scope_canonical"].as<std::string>();
    value.scopeHash = row["scope_hash"].as<std::string>();
    value.requestedLimit = row["requested_limit"].as<int>();
    value.membershipCanonical =
        row["source_membership_canonical"].as<std::string>();
    value.membershipHash = row["source_membership_hash"].as<std::string>();
    value.counts.memberCount = row["member_count"].as<int>();
    value.counts.advisoryReadyCount = row["advisory_ready_count"].as<int>();
    value.counts.blockedCount = row["blocked_count"].as<int>();
    value.counts.nonActionableCount = row["non_actionable_count"].as<int>();
    value.startedAt = row["started_at"].as<std::string>();
    value.completedAt = OptionalValue<std::string>(row, "completed_at");
    value.errorMessage = OptionalValue<std::string>(row, "error_message");
    return value;
}

bool MemberRowEqual(const pqxx::row& row,
                    const RecommendationRankingMember& member)
{
    const auto& value = member.evaluation;
    return row["recommendation_evaluation_result_id"].as<long long>() ==
            value.evaluationResultId &&
        row["recommendation_id"].as<long long>() == value.recommendationId &&
        row["recommendation_semantic_hash"].as<std::string>() ==
            value.recommendationSemanticHash &&
        row["evaluation_identity_hash"].as<std::string>() ==
            value.evaluationIdentityHash &&
        row["bucket"].as<std::string>() ==
            RecommendationRankingBucketText(member.bucket) &&
        row["bucket_rank"].as<int>() == member.bucketRank &&
        row["global_ordinal"].as<int>() == member.globalOrdinal &&
        OptionalValue<double>(row, "final_score") == value.finalScore &&
        row["disposition"].as<std::string>() ==
            RecommendationEvaluationDispositionText(value.disposition) &&
        row["tie_break_primary"].as<std::string>() == member.tieBreakPrimary &&
        row["tie_break_semantic_hash"].as<std::string>() ==
            member.tieBreakSemanticHash &&
        row["tie_break_evaluation_hash"].as<std::string>() ==
            member.tieBreakEvaluationHash &&
        row["inclusion_reason"].as<std::string>() == member.inclusionReason &&
        OptionalValue<std::string>(row, "block_reason") == member.blockReason &&
        OptionalValue<std::string>(row, "top_positive_component") ==
            member.topPositiveComponent &&
        OptionalValue<std::string>(row, "top_penalty_component") ==
            member.topPenaltyComponent &&
        row["symbol"].as<std::string>() == value.symbol &&
        row["horizon"].as<int>() == value.horizon &&
        row["family"].as<std::string>() == value.family &&
        row["source_value_canonical"].as<std::string>() ==
            value.sourceValueCanonical &&
        row["proposed_value_canonical"].as<std::string>() ==
            value.proposedValueCanonical &&
        row["source_experiment_id"].as<long long>() == value.sourceExperimentId &&
        OptionalValue<long long>(row, "source_model_id") == value.sourceModelId &&
        OptionalValue<long long>(row, "source_analysis_id") == value.sourceAnalysisId &&
        row["recommendation_scan_id"].as<long long>() ==
            value.recommendationScanId;
}

std::string MemberPersistenceColumns()
{
    return "recommendation_ranking_member_id,"
        "recommendation_evaluation_result_id,recommendation_id,"
        "recommendation_semantic_hash,evaluation_identity_hash,bucket,"
        "bucket_rank,global_ordinal,final_score,disposition,tie_break_primary,"
        "tie_break_semantic_hash,tie_break_evaluation_hash,inclusion_reason,"
        "block_reason,top_positive_component,top_penalty_component,symbol,"
        "horizon,family,source_value_canonical,proposed_value_canonical,"
        "source_experiment_id,source_model_id,source_analysis_id,"
        "recommendation_scan_id";
}

std::string MemberLoadColumns()
{
    return "rm.recommendation_ranking_member_id,"
        "rm.recommendation_ranking_snapshot_id,rm.bucket,rm.bucket_rank,"
        "rm.global_ordinal,rm.tie_break_primary,rm.tie_break_semantic_hash,"
        "rm.tie_break_evaluation_hash,rm.inclusion_reason,rm.block_reason,"
        "rm.top_positive_component,rm.top_penalty_component,"
        "rm.symbol AS member_symbol,rm.horizon AS member_horizon,"
        "rm.family AS member_family,"
        "rm.source_value_canonical AS member_source_value,"
        "rm.proposed_value_canonical AS member_proposed_value,"
        "rm.source_experiment_id AS member_source_experiment_id,"
        "rm.source_model_id AS member_source_model_id,"
        "rm.source_analysis_id AS member_source_analysis_id,"
        "rm.recommendation_scan_id AS member_scan_id,"
        "rm.created_at::text AS member_created_at," + EvaluationColumns();
}

std::vector<PersistedRecommendationRankingMember> MapPersistedMembers(
    const pqxx::result& rows)
{
    std::vector<PersistedRecommendationRankingMember> values;
    for (const pqxx::row& row : rows)
    {
        const long long memberId =
            row["recommendation_ranking_member_id"].as<long long>();
        if (values.empty() || values.back().memberId != memberId)
        {
            PersistedRecommendationRankingMember value;
            value.memberId = memberId;
            value.snapshotId =
                row["recommendation_ranking_snapshot_id"].as<long long>();
            value.createdAt = row["member_created_at"].as<std::string>();
            value.member.evaluation = MapEvaluation(row);
            value.member.evaluation.symbol = row["member_symbol"].as<std::string>();
            value.member.evaluation.horizon = row["member_horizon"].as<int>();
            value.member.evaluation.family = row["member_family"].as<std::string>();
            value.member.evaluation.sourceValueCanonical =
                row["member_source_value"].as<std::string>();
            value.member.evaluation.proposedValueCanonical =
                row["member_proposed_value"].as<std::string>();
            value.member.evaluation.sourceExperimentId =
                row["member_source_experiment_id"].as<long long>();
            value.member.evaluation.sourceModelId = OptionalValue<long long>(
                row, "member_source_model_id");
            value.member.evaluation.sourceAnalysisId = OptionalValue<long long>(
                row, "member_source_analysis_id");
            value.member.evaluation.recommendationScanId =
                row["member_scan_id"].as<long long>();
            const auto bucket = ParseRecommendationRankingBucket(
                row["bucket"].as<std::string>());
            if (!bucket)
                throw std::runtime_error("invalid_persisted_ranking_bucket");
            value.member.bucket = *bucket;
            value.member.bucketRank = row["bucket_rank"].as<int>();
            value.member.globalOrdinal = row["global_ordinal"].as<int>();
            value.member.tieBreakPrimary =
                row["tie_break_primary"].as<std::string>();
            value.member.tieBreakSemanticHash =
                row["tie_break_semantic_hash"].as<std::string>();
            value.member.tieBreakEvaluationHash =
                row["tie_break_evaluation_hash"].as<std::string>();
            value.member.inclusionReason =
                row["inclusion_reason"].as<std::string>();
            value.member.blockReason = OptionalValue<std::string>(
                row, "block_reason");
            value.member.topPositiveComponent = OptionalValue<std::string>(
                row, "top_positive_component");
            value.member.topPenaltyComponent = OptionalValue<std::string>(
                row, "top_penalty_component");
            values.push_back(std::move(value));
        }
        AddComponent(values.back().member.evaluation, row);
    }
    return values;
}

} // namespace

bool RecommendationRankingSchemaExists(pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return TableExists(transaction, "experiment_recommendation_ranking_snapshot") &&
           TableExists(transaction, "experiment_recommendation_ranking_member");
}

std::vector<RecommendationRankingEvaluation> LoadEvaluationsForRanking(
    pqxx::connection& connection,
    const RecommendationRankingScope& scope)
{
    if (const auto error = ValidateRecommendationRankingScope(scope))
        throw std::invalid_argument(*error);
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "WITH selected_evaluations AS (SELECT "
        "er.recommendation_evaluation_result_id,"
        "er.evaluation_identity_canonical FROM "
        "experiment_recommendation_evaluation_result er "
        "JOIN experiment_recommendation r ON "
        "r.recommendation_id=er.recommendation_id "
        "WHERE ($1::bigint IS NULL OR er.recommendation_evaluation_run_id=$1) "
        "AND ($2::bigint IS NULL OR er.recommendation_scan_id=$2) "
        "AND ($3::text IS NULL OR r.source_symbol=$3) "
        "AND ($4::integer IS NULL OR r.source_prediction_horizon=$4) "
        "AND ($5::text IS NULL OR r.changed_parameter=$5) "
        "ORDER BY er.evaluation_identity_canonical ASC,"
        "er.recommendation_evaluation_result_id ASC LIMIT " +
        std::to_string(kMaximumRecommendationRankingInputs + 1) + ") "
        "SELECT " + EvaluationColumns() +
        " FROM selected_evaluations selected "
        "JOIN experiment_recommendation_evaluation_result er ON "
        "er.recommendation_evaluation_result_id="
        "selected.recommendation_evaluation_result_id "
        "JOIN experiment_recommendation_evaluation_run run ON "
        "run.recommendation_evaluation_run_id=er.recommendation_evaluation_run_id "
        "JOIN experiment_recommendation r ON "
        "r.recommendation_id=er.recommendation_id "
        "LEFT JOIN experiment_recommendation_evaluation_component ec ON "
        "ec.recommendation_evaluation_result_id="
        "er.recommendation_evaluation_result_id "
        "ORDER BY selected.evaluation_identity_canonical ASC,"
        "er.recommendation_evaluation_result_id ASC,ec.component_ordinal ASC;",
        pqxx::params{scope.evaluationRunId, scope.recommendationScanId,
                     scope.symbol, scope.horizon, scope.family});
    return MapEvaluations(rows);
}

RecommendationRankingSnapshotBeginResult BeginOrFindRecommendationRankingSnapshot(
    pqxx::connection& connection,
    const RecommendationRankingSnapshotRequest& request)
{
    if (const auto error = ValidateRecommendationRankingPolicy(request.policy))
        throw std::invalid_argument(*error);
    if (const auto error = ValidateRecommendationRankingScope(request.scope))
        throw std::invalid_argument(*error);
    if (request.limit <= 0 || request.limit > kMaximumRecommendationRankingMembers ||
        request.snapshotIdentityCanonical.empty() ||
        request.snapshotIdentityHash.empty() || request.membershipCanonical.empty() ||
        request.membershipHash.empty())
        throw std::invalid_argument("invalid_recommendation_ranking_snapshot_request");
    const std::string policyCanonical =
        RecommendationRankingPolicyCanonicalText(request.policy);
    const std::string policyHash =
        RecommendationRankingCanonicalHash(policyCanonical);
    const std::string scopeCanonical =
        RecommendationRankingScopeCanonicalText(request.scope);
    const std::string scopeHash = RecommendationRankingCanonicalHash(scopeCanonical);
    const std::string expectedMembershipHash =
        RecommendationRankingCanonicalHash(request.membershipCanonical);
    const std::string expectedSnapshotIdentity =
        RecommendationRankingSnapshotIdentityCanonicalTextFromMembership(
            request.policy, request.scope, request.limit,
            request.membershipCanonical);
    if (request.membershipHash != expectedMembershipHash ||
        request.snapshotIdentityCanonical != expectedSnapshotIdentity ||
        request.snapshotIdentityHash !=
            RecommendationRankingCanonicalHash(expectedSnapshotIdentity))
        throw std::invalid_argument(
            "invalid_recommendation_ranking_snapshot_identity");
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");
    const pqxx::result inserted = transaction.exec(
        "INSERT INTO experiment_recommendation_ranking_snapshot (status,"
        "ranking_snapshot_identity_canonical,ranking_snapshot_identity_hash,"
        "ranking_policy_canonical,ranking_policy_hash,ranking_version,scope_type,"
        "scope_canonical,scope_hash,evaluation_run_filter,recommendation_scan_filter,"
        "symbol_filter,horizon_filter,family_filter,requested_limit,"
        "source_membership_canonical,source_membership_hash) VALUES ('running',"
        "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16) "
        "ON CONFLICT (ranking_snapshot_identity_canonical) DO NOTHING "
        "RETURNING recommendation_ranking_snapshot_id,status;",
        pqxx::params{request.snapshotIdentityCanonical,
            request.snapshotIdentityHash, policyCanonical, policyHash,
            request.policy.rankingVersion,
            RecommendationRankingScopeTypeText(request.scope.type), scopeCanonical,
            scopeHash, request.scope.evaluationRunId,
            request.scope.recommendationScanId, request.scope.symbol,
            request.scope.horizon, request.scope.family, request.limit,
            request.membershipCanonical, request.membershipHash});
    RecommendationRankingSnapshotBeginResult result;
    if (!inserted.empty())
    {
        result.snapshotId = inserted.one_row()[0].as<long long>();
        result.status = inserted.one_row()[1].as<std::string>();
        result.created = true;
    }
    else
    {
        const pqxx::row row = transaction.exec(
            "SELECT " + SnapshotColumns() +
            " FROM experiment_recommendation_ranking_snapshot "
            "WHERE ranking_snapshot_identity_canonical=$1;",
            pqxx::params{request.snapshotIdentityCanonical}).one_row();
        const PersistedRecommendationRankingSnapshot existing = MapSnapshot(row);
        if (existing.snapshotIdentityHash != request.snapshotIdentityHash ||
            existing.rankingPolicyCanonical != policyCanonical ||
            existing.rankingPolicyHash != policyHash ||
            existing.rankingVersion != request.policy.rankingVersion ||
            existing.scopeCanonical != scopeCanonical ||
            existing.scopeHash != scopeHash || existing.requestedLimit != request.limit ||
            existing.membershipCanonical != request.membershipCanonical ||
            existing.membershipHash != request.membershipHash)
            throw std::runtime_error("recommendation_ranking_snapshot_retry_mismatch");
        result.snapshotId = existing.snapshotId;
        result.status = existing.status;
    }
    transaction.commit();
    return result;
}

std::vector<PersistedRecommendationRankingMember> PersistRecommendationRankingMembers(
    pqxx::connection& connection,
    long long snapshotId,
    const std::vector<RecommendationRankingMember>& members)
{
    if (snapshotId <= 0 || members.size() >
        static_cast<std::size_t>(kMaximumRecommendationRankingMembers))
        throw std::invalid_argument("invalid_recommendation_ranking_members_request");
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");
    const pqxx::result snapshotRows = transaction.exec(
        "SELECT status FROM experiment_recommendation_ranking_snapshot "
        "WHERE recommendation_ranking_snapshot_id=$1 FOR UPDATE;",
        pqxx::params{snapshotId});
    if (snapshotRows.empty())
        throw std::runtime_error("recommendation_ranking_snapshot_not_found");
    const std::string status = snapshotRows.one_row()[0].as<std::string>();
    if (status == "failed")
        throw std::runtime_error("recommendation_ranking_snapshot_failed");
    std::vector<PersistedRecommendationRankingMember> persisted;
    persisted.reserve(members.size());
    for (const auto& member : members)
    {
        const auto& value = member.evaluation;
        const pqxx::result inserted = transaction.exec(
            "INSERT INTO experiment_recommendation_ranking_member ("
            "recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,"
            "recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,"
            "bucket,bucket_rank,global_ordinal,final_score,disposition,"
            "tie_break_primary,tie_break_semantic_hash,tie_break_evaluation_hash,"
            "inclusion_reason,block_reason,top_positive_component,"
            "top_penalty_component,symbol,horizon,family,source_value_canonical,"
            "proposed_value_canonical,source_experiment_id,source_model_id,"
            "source_analysis_id,recommendation_scan_id) SELECT $1,$2,$3,$4,$5,"
            "$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,"
            "$23,$24,$25,$26 FROM experiment_recommendation_ranking_snapshot guard "
            "JOIN experiment_recommendation_evaluation_result er ON "
            "er.recommendation_evaluation_result_id=$2 "
            "JOIN experiment_recommendation r ON r.recommendation_id=er.recommendation_id "
            "WHERE guard.recommendation_ranking_snapshot_id=$1 "
            "AND guard.status='running' "
            "AND er.recommendation_id=$3 "
            "AND er.recommendation_semantic_hash=$4 "
            "AND er.evaluation_identity_hash=$5 "
            "AND er.recommendation_scan_id=$26 "
            "AND er.source_experiment_id=$23 "
            "AND er.source_model_id IS NOT DISTINCT FROM $24::bigint "
            "AND er.source_analysis_id IS NOT DISTINCT FROM $25::bigint "
            "AND r.source_symbol=$18 AND r.source_prediction_horizon=$19 "
            "AND r.changed_parameter=$20 AND r.source_value_canonical=$21 "
            "AND r.proposed_value_canonical=$22 "
            "ON CONFLICT (recommendation_ranking_snapshot_id,"
            "recommendation_evaluation_result_id) DO NOTHING "
            "RETURNING recommendation_ranking_member_id;",
            pqxx::params{snapshotId, value.evaluationResultId,
                value.recommendationId, value.recommendationSemanticHash,
                value.evaluationIdentityHash,
                RecommendationRankingBucketText(member.bucket), member.bucketRank,
                member.globalOrdinal, value.finalScore,
                RecommendationEvaluationDispositionText(value.disposition),
                member.tieBreakPrimary, member.tieBreakSemanticHash,
                member.tieBreakEvaluationHash, member.inclusionReason,
                member.blockReason, member.topPositiveComponent,
                member.topPenaltyComponent, value.symbol, value.horizon,
                value.family, value.sourceValueCanonical,
                value.proposedValueCanonical, value.sourceExperimentId,
                value.sourceModelId, value.sourceAnalysisId,
                value.recommendationScanId});
        const pqxx::row row = transaction.exec(
            "SELECT " + MemberPersistenceColumns() +
            " FROM experiment_recommendation_ranking_member "
            "WHERE recommendation_ranking_snapshot_id=$1 "
            "AND recommendation_evaluation_result_id=$2;",
            pqxx::params{snapshotId, value.evaluationResultId}).one_row();
        if (!MemberRowEqual(row, member))
            throw std::runtime_error("recommendation_ranking_member_retry_mismatch");
        PersistedRecommendationRankingMember item;
        item.memberId = row["recommendation_ranking_member_id"].as<long long>();
        item.snapshotId = snapshotId;
        item.member = member;
        persisted.push_back(std::move(item));
        (void)inserted;
    }
    const long long count = transaction.exec(
        "SELECT count(*) FROM experiment_recommendation_ranking_member "
        "WHERE recommendation_ranking_snapshot_id=$1;",
        pqxx::params{snapshotId}).one_row()[0].as<long long>();
    if (count != static_cast<long long>(members.size()))
        throw std::runtime_error("recommendation_ranking_membership_retry_mismatch");
    transaction.commit();
    return persisted;
}

void CompleteRecommendationRankingSnapshot(
    pqxx::connection& connection,
    long long snapshotId,
    const RecommendationRankingCounts& counts)
{
    if (snapshotId <= 0 || counts.memberCount < 0 ||
        counts.advisoryReadyCount < 0 || counts.blockedCount < 0 ||
        counts.nonActionableCount < 0 ||
        counts.memberCount != counts.advisoryReadyCount + counts.blockedCount +
                              counts.nonActionableCount)
        throw std::invalid_argument("invalid_recommendation_ranking_counts");
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");
    const pqxx::row persistedCounts = transaction.exec(
        "SELECT count(*) AS member_count,"
        "count(*) FILTER (WHERE bucket='advisory_ready') AS ready_count,"
        "count(*) FILTER (WHERE bucket='blocked') AS blocked_count,"
        "count(*) FILTER (WHERE bucket='non_actionable') AS non_actionable_count "
        "FROM experiment_recommendation_ranking_member "
        "WHERE recommendation_ranking_snapshot_id=$1;",
        pqxx::params{snapshotId}).one_row();
    if (persistedCounts["member_count"].as<int>() != counts.memberCount ||
        persistedCounts["ready_count"].as<int>() != counts.advisoryReadyCount ||
        persistedCounts["blocked_count"].as<int>() != counts.blockedCount ||
        persistedCounts["non_actionable_count"].as<int>() !=
            counts.nonActionableCount)
        throw std::runtime_error("recommendation_ranking_member_count_mismatch");
    const pqxx::result updated = transaction.exec(
        "UPDATE experiment_recommendation_ranking_snapshot SET status='completed',"
        "member_count=$2,advisory_ready_count=$3,blocked_count=$4,"
        "non_actionable_count=$5,completed_at=now(),error_message=NULL,"
        "updated_at=now() WHERE recommendation_ranking_snapshot_id=$1 "
        "AND status='running';",
        pqxx::params{snapshotId, counts.memberCount, counts.advisoryReadyCount,
                     counts.blockedCount, counts.nonActionableCount});
    if (updated.affected_rows() == 0)
    {
        const pqxx::row row = transaction.exec(
            "SELECT status,member_count,advisory_ready_count,blocked_count,"
            "non_actionable_count FROM experiment_recommendation_ranking_snapshot "
            "WHERE recommendation_ranking_snapshot_id=$1;",
            pqxx::params{snapshotId}).one_row();
        if (row["status"].as<std::string>() != "completed" ||
            row["member_count"].as<int>() != counts.memberCount ||
            row["advisory_ready_count"].as<int>() != counts.advisoryReadyCount ||
            row["blocked_count"].as<int>() != counts.blockedCount ||
            row["non_actionable_count"].as<int>() != counts.nonActionableCount)
            throw std::runtime_error("recommendation_ranking_snapshot_not_running");
    }
    transaction.commit();
}

void FailRecommendationRankingSnapshot(
    pqxx::connection& connection,
    long long snapshotId,
    const std::string& errorMessage)
{
    if (snapshotId <= 0)
        throw std::invalid_argument("recommendation_ranking_snapshot_id_invalid");
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");
    const long long memberCount = transaction.exec(
        "SELECT count(*) FROM experiment_recommendation_ranking_member "
        "WHERE recommendation_ranking_snapshot_id=$1;",
        pqxx::params{snapshotId}).one_row()[0].as<long long>();
    if (memberCount != 0)
        throw std::runtime_error("recommendation_ranking_failed_snapshot_has_members");
    const std::string persisted = errorMessage.empty()
        ? "unknown_recommendation_ranking_failure" : errorMessage;
    const pqxx::result updated = transaction.exec(
        "UPDATE experiment_recommendation_ranking_snapshot SET status='failed',"
        "completed_at=now(),error_message=$2,updated_at=now() "
        "WHERE recommendation_ranking_snapshot_id=$1 AND status='running';",
        pqxx::params{snapshotId, persisted});
    if (updated.affected_rows() == 0)
        throw std::runtime_error("recommendation_ranking_snapshot_not_running");
    transaction.commit();
}

std::vector<PersistedRecommendationRankingSnapshot>
ListRecommendationRankingSnapshots(pqxx::connection& connection, int limit)
{
    if (limit <= 0 || limit > kMaximumRecommendationRankingListLimit)
        throw std::invalid_argument("recommendation_ranking_list_limit_invalid");
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT " + SnapshotColumns() +
        " FROM experiment_recommendation_ranking_snapshot "
        "ORDER BY recommendation_ranking_snapshot_id DESC LIMIT $1;",
        pqxx::params{limit});
    std::vector<PersistedRecommendationRankingSnapshot> values;
    values.reserve(rows.size());
    for (const pqxx::row& row : rows) values.push_back(MapSnapshot(row));
    return values;
}

std::optional<PersistedRecommendationRankingSnapshot>
FindRecommendationRankingSnapshot(pqxx::connection& connection,
                                  long long snapshotId)
{
    if (snapshotId <= 0)
        throw std::invalid_argument("recommendation_ranking_snapshot_id_invalid");
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT " + SnapshotColumns() +
        " FROM experiment_recommendation_ranking_snapshot "
        "WHERE recommendation_ranking_snapshot_id=$1;",
        pqxx::params{snapshotId});
    if (rows.empty()) return std::nullopt;
    return MapSnapshot(rows.one_row());
}

std::vector<PersistedRecommendationRankingMember>
ListRecommendationRankingMembers(
    pqxx::connection& connection,
    long long snapshotId,
    std::optional<RecommendationRankingBucket> bucket,
    int limit)
{
    if (snapshotId <= 0)
        throw std::invalid_argument("recommendation_ranking_snapshot_id_invalid");
    if (limit <= 0 || limit > kMaximumRecommendationRankingListLimit)
        throw std::invalid_argument("recommendation_ranking_list_limit_invalid");
    const std::optional<std::string> bucketText = bucket
        ? std::optional<std::string>{RecommendationRankingBucketText(*bucket)}
        : std::nullopt;
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "WITH selected AS (SELECT * FROM "
        "experiment_recommendation_ranking_member WHERE "
        "recommendation_ranking_snapshot_id=$1 "
        "AND ($2::text IS NULL OR bucket=$2) "
        "ORDER BY global_ordinal LIMIT $3) SELECT " + MemberLoadColumns() +
        " FROM selected rm "
        "JOIN experiment_recommendation_evaluation_result er ON "
        "er.recommendation_evaluation_result_id=rm.recommendation_evaluation_result_id "
        "JOIN experiment_recommendation_evaluation_run run ON "
        "run.recommendation_evaluation_run_id=er.recommendation_evaluation_run_id "
        "JOIN experiment_recommendation r ON r.recommendation_id=er.recommendation_id "
        "LEFT JOIN experiment_recommendation_evaluation_component ec ON "
        "ec.recommendation_evaluation_result_id=er.recommendation_evaluation_result_id "
        "ORDER BY rm.global_ordinal,ec.component_ordinal;",
        pqxx::params{snapshotId, bucketText, limit});
    return MapPersistedMembers(rows);
}

std::optional<PersistedRecommendationRankingMember>
FindRecommendationRankingMember(pqxx::connection& connection,
                                long long memberId)
{
    if (memberId <= 0)
        throw std::invalid_argument("recommendation_ranking_member_id_invalid");
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT " + MemberLoadColumns() +
        " FROM experiment_recommendation_ranking_member rm "
        "JOIN experiment_recommendation_evaluation_result er ON "
        "er.recommendation_evaluation_result_id=rm.recommendation_evaluation_result_id "
        "JOIN experiment_recommendation_evaluation_run run ON "
        "run.recommendation_evaluation_run_id=er.recommendation_evaluation_run_id "
        "JOIN experiment_recommendation r ON r.recommendation_id=er.recommendation_id "
        "LEFT JOIN experiment_recommendation_evaluation_component ec ON "
        "ec.recommendation_evaluation_result_id=er.recommendation_evaluation_result_id "
        "WHERE rm.recommendation_ranking_member_id=$1 "
        "ORDER BY ec.component_ordinal;",
        pqxx::params{memberId});
    if (rows.empty()) return std::nullopt;
    return MapPersistedMembers(rows).front();
}

std::optional<RecommendationRankingEvaluation> FindRankingEvaluation(
    pqxx::connection& connection,
    long long evaluationResultId)
{
    if (evaluationResultId <= 0)
        throw std::invalid_argument("recommendation_evaluation_result_id_invalid");
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT " + EvaluationColumns() + EvaluationFromClause() +
        "WHERE er.recommendation_evaluation_result_id=$1 "
        "ORDER BY ec.component_ordinal;",
        pqxx::params{evaluationResultId});
    if (rows.empty()) return std::nullopt;
    return MapEvaluations(rows).front();
}

} // namespace EA::ExperimentRecommendation
