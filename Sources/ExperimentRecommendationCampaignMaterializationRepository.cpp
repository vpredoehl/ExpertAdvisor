#include "ExperimentRecommendationCampaignMaterializationRepository.hpp"

#include "ExperimentRecommendationEvaluation.hpp"

#include <algorithm>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace EA::ExperimentRecommendation
{
namespace
{

template <typename Value>
std::optional<Value> OptionalValue(const pqxx::row& row, const char* column)
{
    const auto field = row[column];
    if (field.is_null()) return std::nullopt;
    return field.as<Value>();
}

std::string LengthText(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

std::string AuthorizationCanonical(const pqxx::row& row)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_conversion_review_authorization_v1"
        << ";review_event_id="
        << row["recommendation_review_event_id"].as<long long>()
        << ";recommendation_id=" << row["recommendation_id"].as<long long>()
        << ";action=" << LengthText(row["review_action"].as<std::string>())
        << ";resulting_status="
        << LengthText(row["review_resulting_status"].as<std::string>())
        << ";recommendation_semantic="
        << LengthText(row["review_semantic_canonical"].as<std::string>())
        << ";recommendation_semantic_hash="
        << LengthText(row["review_semantic_hash"].as<std::string>())
        << ";source_experiment_id="
        << row["review_source_experiment_id"].as<long long>();
    return out.str();
}

std::string MaterializationColumns()
{
    return
        "recommendation_campaign_materialization_id,"
        "recommendation_campaign_approval_id,"
        "materialization_contract_version,approval_identity_hash,"
        "recommendation_ranking_snapshot_id,ranking_snapshot_identity_hash,"
        "planning_policy_hash,campaign_plan_identity_hash,"
        "campaign_review_identity_hash,materialized_by,"
        "materialization_reason_text,selected_member_count,"
        "initially_created_proposal_count,initially_reused_proposal_count,"
        "materialization_identity_canonical,materialization_identity_hash,"
        "created_at::text AS created_at";
}

std::string MemberColumns()
{
    return
        "recommendation_campaign_materialization_member_id,member_ordinal,"
        "recommendation_ranking_member_id,recommendation_id,"
        "source_experiment_id,ranking_position,"
        "selected_member_identity_canonical,selected_member_identity_hash,"
        "recommendation_conversion_proposal_id,"
        "proposal_identity_canonical,proposal_identity_hash,"
        "created_at::text AS created_at";
}

PersistedRecommendationCampaignMaterializationMember MapMember(
    const pqxx::row& row)
{
    PersistedRecommendationCampaignMaterializationMember value;
    value.materializationMemberId = row[0].as<long long>();
    value.memberOrdinal = row[1].as<int>();
    value.rankingMemberId = row[2].as<long long>();
    value.recommendationId = row[3].as<long long>();
    value.sourceExperimentId = row[4].as<long long>();
    value.rankingPosition = row[5].as<int>();
    value.selectedMemberIdentityCanonical = row[6].as<std::string>();
    value.selectedMemberIdentityHash = row[7].as<std::string>();
    value.conversionProposalId = row[8].as<long long>();
    value.proposalIdentityCanonical = row[9].as<std::string>();
    value.proposalIdentityHash = row[10].as<std::string>();
    value.createdAt = row[11].as<std::string>();
    if (value.memberOrdinal <= 0 || value.rankingMemberId <= 0 ||
        value.recommendationId <= 0 || value.sourceExperimentId <= 0 ||
        value.rankingPosition <= 0 || value.conversionProposalId <= 0 ||
        value.selectedMemberIdentityHash != RecommendationCanonicalHash(
            value.selectedMemberIdentityCanonical) ||
        value.proposalIdentityHash != RecommendationCanonicalHash(
            value.proposalIdentityCanonical))
        throw std::runtime_error(
            "invalid_persisted_recommendation_campaign_materialization_member");
    return value;
}

PersistedRecommendationCampaignMaterialization MapMaterialization(
    pqxx::transaction_base& transaction,
    const pqxx::row& row)
{
    PersistedRecommendationCampaignMaterialization value;
    value.materializationId = row[0].as<long long>();
    value.campaignApprovalId = row[1].as<long long>();
    value.contractVersion = row[2].as<int>();
    value.approvalIdentityHash = row[3].as<std::string>();
    value.rankingSnapshotId = row[4].as<long long>();
    value.rankingSnapshotIdentityHash = row[5].as<std::string>();
    value.planningPolicyHash = row[6].as<std::string>();
    value.campaignPlanIdentityHash = row[7].as<std::string>();
    value.campaignReviewIdentityHash = row[8].as<std::string>();
    value.materializedBy = row[9].as<std::string>();
    value.reasonText = row[10].as<std::string>();
    value.selectedMemberCount = row[11].as<int>();
    value.initiallyCreatedProposalCount = row[12].as<int>();
    value.initiallyReusedProposalCount = row[13].as<int>();
    value.identityCanonical = row[14].as<std::string>();
    value.identityHash = row[15].as<std::string>();
    value.createdAt = row[16].as<std::string>();
    if (value.materializationId <= 0 || value.campaignApprovalId <= 0 ||
        value.contractVersion !=
            kRecommendationCampaignMaterializationContractVersion ||
        value.selectedMemberCount <= 0 ||
        value.identityHash != RecommendationCanonicalHash(
            value.identityCanonical))
        throw std::runtime_error(
            "invalid_persisted_recommendation_campaign_materialization");
    const pqxx::result members = transaction.exec(
        "SELECT " + MemberColumns() + " FROM "
        "experiment_recommendation_campaign_materialization_member WHERE "
        "recommendation_campaign_materialization_id=$1 "
        "ORDER BY member_ordinal ASC;",
        pqxx::params{value.materializationId});
    value.members.reserve(members.size());
    for (const auto& member : members) value.members.push_back(MapMember(member));
    if (static_cast<int>(value.members.size()) != value.selectedMemberCount)
        throw std::runtime_error(
            "incomplete_persisted_recommendation_campaign_materialization");
    for (std::size_t index = 0; index < value.members.size(); ++index)
        if (value.members[index].memberOrdinal != static_cast<int>(index) + 1)
            throw std::runtime_error(
                "invalid_persisted_recommendation_campaign_materialization_order");
    return value;
}

bool Matches(
    const PersistedRecommendationCampaignMaterialization& persisted,
    const RecommendationCampaignMaterializationEvidence& evidence)
{
    if (persisted.campaignApprovalId != evidence.campaignApprovalId ||
        persisted.contractVersion != evidence.materializationContractVersion ||
        persisted.identityCanonical != evidence.materializationIdentityCanonical ||
        persisted.identityHash != evidence.materializationIdentityHash ||
        persisted.materializedBy != evidence.operatorIdentity ||
        persisted.reasonText != evidence.reasonText ||
        persisted.selectedMemberCount != evidence.selectedMemberCount ||
        persisted.members.size() != evidence.members.size())
        return false;
    for (std::size_t index = 0; index < persisted.members.size(); ++index)
    {
        const auto& left = persisted.members[index];
        const auto& right = evidence.members[index];
        if (left.memberOrdinal != right.memberOrdinal ||
            left.rankingMemberId != right.rankingMemberId ||
            left.recommendationId != right.recommendationId ||
            left.sourceExperimentId != right.sourceExperimentId ||
            left.rankingPosition != right.rankingPosition ||
            left.selectedMemberIdentityCanonical !=
                right.selectedMemberIdentityCanonical ||
            left.selectedMemberIdentityHash != right.selectedMemberIdentityHash ||
            left.proposalIdentityCanonical !=
                right.proposal.conversionIdentityCanonical ||
            left.proposalIdentityHash != right.proposal.conversionIdentityHash)
            return false;
    }
    return true;
}

std::optional<PersistedRecommendationCampaignMaterialization> FindById(
    pqxx::transaction_base& transaction,
    long long id)
{
    const auto rows = transaction.exec(
        "SELECT " + MaterializationColumns() + " FROM "
        "experiment_recommendation_campaign_materialization WHERE "
        "recommendation_campaign_materialization_id=$1;", pqxx::params{id});
    if (rows.empty()) return std::nullopt;
    return MapMaterialization(transaction, rows.one_row());
}

} // namespace

bool RecommendationCampaignMaterializationSchemaExists(
    pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return RecommendationCampaignMaterializationSchemaExists(transaction);
}

bool RecommendationCampaignMaterializationSchemaExists(
    pqxx::transaction_base& transaction)
{
    const auto row = transaction.exec(
        "SELECT to_regclass('experiment_recommendation_campaign_materialization') "
        "IS NOT NULL AND to_regclass('experiment_recommendation_campaign_"
        "materialization_member') IS NOT NULL;").one_row();
    return row[0].as<bool>();
}

RecommendationConversionRequest LoadRecommendationCampaignConversionRequest(
    pqxx::transaction_base& transaction,
    long long rankingSnapshotId,
    long long rankingMemberId,
    long long recommendationId)
{
    if (rankingSnapshotId <= 0 || rankingMemberId <= 0 || recommendationId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_materialization_member_id_invalid");
    const pqxx::result rows = transaction.exec(
        "SELECT r.recommendation_id,r.status,r.source_experiment_id,"
        "r.semantic_configuration_canonical,r.semantic_hash,"
        "r.invocation_configuration_canonical,r.invocation_hash,"
        "r.changed_parameter,r.source_value_canonical,r.proposed_value_canonical,"
        "m.recommendation_evaluation_result_id,m.bucket AS ranking_bucket,"
        "m.recommendation_semantic_hash AS ranking_semantic_hash,"
        "m.source_experiment_id AS ranking_source_experiment_id,"
        "er.recommendation_id AS evaluation_recommendation_id,"
        "er.recommendation_semantic_canonical AS evaluation_semantic_canonical,"
        "er.recommendation_semantic_hash AS evaluation_semantic_hash,"
        "er.evaluation_identity_canonical,"
        "er.evaluation_identity_hash,er.eligibility,er.disposition,"
        "er.source_experiment_id AS evaluation_source_experiment_id,"
        "eru.status AS evaluation_run_status,eru.evaluation_policy_canonical,"
        "eru.evaluation_policy_hash,eru.scoring_policy_hash AS evaluation_scoring_hash,"
        "re.recommendation_review_event_id,re.action AS review_action,"
        "re.resulting_status AS review_resulting_status,"
        "re.recommendation_semantic_canonical AS review_semantic_canonical,"
        "re.recommendation_semantic_hash AS review_semantic_hash,"
        "re.source_experiment_id AS review_source_experiment_id,"
        "s.recommendation_id AS score_recommendation_id,"
        "s.recommendation_semantic_canonical AS score_semantic_canonical,"
        "s.source_experiment_id AS score_source_experiment_id,s.final_score,"
        "s.score_status,s.scoring_policy_canonical,s.scoring_policy_hash,"
        "sr.status AS score_run_status,e.symbol,e.prediction_horizon,"
        "e.c_next_threshold,e.core_lr_mult,e.head_lr_mult,e.target_epochs,"
        "e.checkpoint_interval,e.donchian20_mode,"
        "to_char(e.train_start AT TIME ZONE 'America/Chicago',"
        "'YYYY-MM-DD') AS train_start_date,"
        "to_char(e.train_end AT TIME ZONE 'America/Chicago',"
        "'YYYY-MM-DD') AS train_end_date,"
        "CASE WHEN e.infer_start IS NULL THEN NULL ELSE "
        "to_char(e.infer_start AT TIME ZONE 'America/Chicago',"
        "'YYYY-MM-DD') END AS infer_start_date,"
        "CASE WHEN e.infer_end IS NULL THEN NULL ELSE "
        "to_char(e.infer_end AT TIME ZONE 'America/Chicago',"
        "'YYYY-MM-DD') END AS infer_end_date,e.resume_model_id,"
        "((e.train_start AT TIME ZONE 'America/Chicago') = "
        "date_trunc('day',e.train_start AT TIME ZONE 'America/Chicago') AND "
        "(e.train_end AT TIME ZONE 'America/Chicago') = "
        "date_trunc('day',e.train_end AT TIME ZONE 'America/Chicago') AND "
        "(e.infer_start IS NULL OR "
        "(e.infer_start AT TIME ZONE 'America/Chicago') = "
        "date_trunc('day',e.infer_start AT TIME ZONE 'America/Chicago')) AND "
        "(e.infer_end IS NULL OR "
        "(e.infer_end AT TIME ZONE 'America/Chicago') = "
        "date_trunc('day',e.infer_end AT TIME ZONE 'America/Chicago'))) "
        "AS date_mapping_valid "
        "FROM experiment_recommendation_ranking_member m "
        "JOIN experiment_recommendation r ON r.recommendation_id=m.recommendation_id "
        "JOIN experiment_recommendation_evaluation_result er ON "
        "er.recommendation_evaluation_result_id=m.recommendation_evaluation_result_id "
        "JOIN experiment_recommendation_evaluation_run eru ON "
        "eru.recommendation_evaluation_run_id=er.recommendation_evaluation_run_id "
        "JOIN experiment e ON e.experiment_id=r.source_experiment_id "
        "LEFT JOIN LATERAL (SELECT review.* FROM "
        "experiment_recommendation_review_event review WHERE "
        "review.recommendation_id=r.recommendation_id ORDER BY "
        "review.recommendation_review_event_id DESC LIMIT 1) re ON true "
        "LEFT JOIN LATERAL (SELECT candidate.recommendation_score_id FROM "
        "experiment_recommendation_score candidate JOIN "
        "experiment_recommendation_score_run candidate_run ON "
        "candidate_run.recommendation_score_run_id="
        "candidate.recommendation_score_run_id WHERE "
        "candidate.recommendation_id=r.recommendation_id AND "
        "candidate.score_status='scored' AND candidate_run.status='completed' "
        "AND (re.recommendation_score_id IS NULL OR "
        "candidate.recommendation_score_id=re.recommendation_score_id) "
        "ORDER BY candidate.recommendation_score_run_id DESC,"
        "candidate.recommendation_score_id DESC LIMIT 1) selected_score ON true "
        "LEFT JOIN experiment_recommendation_score s ON "
        "s.recommendation_score_id=selected_score.recommendation_score_id "
        "LEFT JOIN experiment_recommendation_score_run sr ON "
        "sr.recommendation_score_run_id=s.recommendation_score_run_id "
        "WHERE m.recommendation_ranking_snapshot_id=$1 AND "
        "m.recommendation_ranking_member_id=$2 AND m.recommendation_id=$3;",
        pqxx::params{rankingSnapshotId, rankingMemberId, recommendationId});
    if (rows.empty())
        throw std::runtime_error(
            "recommendation_campaign_materialization_member_not_found");
    const auto& row = rows.one_row();
    RecommendationConversionRequest request;
    request.recommendationExists = true;
    request.recommendationId = row["recommendation_id"].as<long long>();
    const auto status = ParseRecommendationStatus(row["status"].as<std::string>());
    if (!status) throw std::runtime_error("invalid_persisted_recommendation_status");
    request.recommendationStatus = *status;
    request.sourceExperimentId = row["source_experiment_id"].as<long long>();
    request.recommendationSourceExperimentId = request.sourceExperimentId;
    request.recommendationSemanticCanonical =
        row["semantic_configuration_canonical"].as<std::string>();
    request.recommendationSemanticHash = row["semantic_hash"].as<std::string>();
    request.recommendationInvocationCanonical =
        row["invocation_configuration_canonical"].as<std::string>();
    request.recommendationInvocationHash = row["invocation_hash"].as<std::string>();
    const auto semanticVersion =
        RecommendationSemanticConfigurationVersionFromCanonicalText(
            request.recommendationSemanticCanonical);
    if (!semanticVersion)
        throw std::runtime_error(
            "invalid_persisted_recommendation_semantic_configuration_version");
    if (row["ranking_bucket"].as<std::string>() != "advisory_ready" ||
        row["ranking_semantic_hash"].as<std::string>() !=
            request.recommendationSemanticHash ||
        row["ranking_source_experiment_id"].as<long long>() !=
            request.sourceExperimentId ||
        row["evaluation_recommendation_id"].as<long long>() !=
            request.recommendationId ||
        row["evaluation_semantic_canonical"].as<std::string>() !=
            request.recommendationSemanticCanonical ||
        row["evaluation_semantic_hash"].as<std::string>() !=
            request.recommendationSemanticHash)
        throw std::runtime_error(
            "inconsistent_persisted_recommendation_campaign_member_provenance");
    if (!row["recommendation_review_event_id"].is_null())
    {
        if (row["review_semantic_canonical"].as<std::string>() !=
                request.recommendationSemanticCanonical ||
            row["review_semantic_hash"].as<std::string>() !=
                request.recommendationSemanticHash ||
            row["review_source_experiment_id"].as<long long>() !=
                request.sourceExperimentId)
            throw std::runtime_error(
                "inconsistent_persisted_recommendation_review_provenance");
        auto& authorization = request.reviewAuthorization;
        authorization.present = true;
        authorization.recommendationId = request.recommendationId;
        const auto action = ParseRecommendationReviewAction(
            row["review_action"].as<std::string>());
        const auto resulting = ParseRecommendationStatus(
            row["review_resulting_status"].as<std::string>());
        if (!action || !resulting)
            throw std::runtime_error("invalid_persisted_recommendation_review");
        authorization.latestAction = *action;
        authorization.resultingStatus = *resulting;
        authorization.latestActionEffective = true;
        authorization.superseded = false;
        authorization.authorizationCanonical = AuthorizationCanonical(row);
        authorization.authorizationHash = RecommendationCanonicalHash(
            authorization.authorizationCanonical);
    }
    auto& evaluation = request.evaluation;
    evaluation.state = row["evaluation_run_status"].as<std::string>() == "completed"
        ? RecommendationConversionEvidenceState::completed
        : RecommendationConversionEvidenceState::pending;
    evaluation.valid = evaluation.state ==
                           RecommendationConversionEvidenceState::completed;
    evaluation.recommendationId = request.recommendationId;
    evaluation.sourceExperimentId =
        row["evaluation_source_experiment_id"].as<long long>();
    evaluation.eligibility = row["eligibility"].as<std::string>() == "eligible"
        ? RecommendationEligibility::eligible : RecommendationEligibility::ineligible;
    const auto disposition = ParseRecommendationEvaluationDisposition(
        row["disposition"].as<std::string>());
    if (!disposition)
        throw std::runtime_error("invalid_persisted_recommendation_disposition");
    evaluation.disposition = *disposition;
    evaluation.evaluationIdentityCanonical =
        row["evaluation_identity_canonical"].as<std::string>();
    evaluation.evaluationIdentityHash =
        row["evaluation_identity_hash"].as<std::string>();
    evaluation.evaluationPolicyCanonical =
        row["evaluation_policy_canonical"].as<std::string>();
    evaluation.evaluationPolicyHash =
        row["evaluation_policy_hash"].as<std::string>();
    evaluation.scoringPolicyHash =
        row["evaluation_scoring_hash"].as<std::string>();
    auto& score = request.score;
    if (!row["score_recommendation_id"].is_null())
    {
        if (row["score_semantic_canonical"].as<std::string>() !=
                request.recommendationSemanticCanonical ||
            row["score_source_experiment_id"].as<long long>() !=
                request.sourceExperimentId)
            throw std::runtime_error(
                "inconsistent_persisted_recommendation_score_provenance");
        score.state = row["score_run_status"].as<std::string>() == "completed"
            ? RecommendationConversionEvidenceState::completed
            : RecommendationConversionEvidenceState::pending;
        score.valid = score.state == RecommendationConversionEvidenceState::completed &&
                      row["score_status"].as<std::string>() == "scored";
        score.recommendationId =
            row["score_recommendation_id"].as<long long>();
        score.finalScore = row["final_score"].as<double>();
        score.scoringPolicyCanonical =
            row["scoring_policy_canonical"].as<std::string>();
        score.scoringPolicyHash =
            row["scoring_policy_hash"].as<std::string>();
    }
    auto& invocation = request.sourceInvocation;
    if (!row["date_mapping_valid"].as<bool>())
        throw std::runtime_error(
            "invalid_persisted_recommendation_source_date_mapping");
    invocation.configuration.symbol = row["symbol"].as<std::string>();
    invocation.configuration.predictionHorizon =
        row["prediction_horizon"].as<int>();
    invocation.configuration.labelThreshold =
        row["c_next_threshold"].as<double>();
    invocation.configuration.coreLrMult = OptionalValue<double>(row, "core_lr_mult");
    invocation.configuration.headLrMult = OptionalValue<double>(row, "head_lr_mult");
    invocation.configuration.targetEpochs = row["target_epochs"].as<int>();
    invocation.configuration.trainStartDate = row["train_start_date"].as<std::string>();
    invocation.configuration.trainEndDate = row["train_end_date"].as<std::string>();
    invocation.configuration.inferStartDate = OptionalValue<std::string>(row, "infer_start_date");
    invocation.configuration.inferEndDate = OptionalValue<std::string>(row, "infer_end_date");
    // A v4 identity makes this persisted source mode part of its exact
    // scientific provenance.  Do not substitute the configuration default.
    if (*semanticVersion == RecommendationSemanticConfigurationVersion::v4)
        invocation.configuration.donchian20Mode = ParseDonchian20Mode(
            row["donchian20_mode"].as<std::string>());
    invocation.checkpointInterval = row["checkpoint_interval"].as<int>();
    invocation.resumeModelId = OptionalValue<long long>(row, "resume_model_id");
    request.mutations.push_back({
        row["changed_parameter"].as<std::string>(),
        row["source_value_canonical"].as<std::string>(),
        row["proposed_value_canonical"].as<std::string>()});
    return request;
}

std::optional<PersistedRecommendationCampaignMaterialization>
FindRecommendationCampaignMaterializationByApproval(
    pqxx::transaction_base& transaction,
    long long campaignApprovalId)
{
    if (campaignApprovalId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_materialization_approval_id_invalid");
    const auto rows = transaction.exec(
        "SELECT " + MaterializationColumns() + " FROM "
        "experiment_recommendation_campaign_materialization WHERE "
        "recommendation_campaign_approval_id=$1;",
        pqxx::params{campaignApprovalId});
    if (rows.empty()) return std::nullopt;
    return MapMaterialization(transaction, rows.one_row());
}

RecommendationCampaignMaterializationPersistResult
PersistRecommendationCampaignMaterialization(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignMaterializationEvidence& evidence)
{
    ValidateRecommendationCampaignMaterializationEvidence(evidence);
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, "
        "-7046029254386353131));",
        pqxx::params{evidence.approval.approvalIdentityHash});
    if (auto existing = FindRecommendationCampaignMaterializationByApproval(
            transaction, evidence.campaignApprovalId))
    {
        if (!Matches(*existing, evidence))
            throw std::runtime_error(
                "recommendation_campaign_materialization_conflict");
        return {RecommendationCampaignMaterializationPersistOutcome::existingIdentical,
                std::move(*existing), 0, evidence.selectedMemberCount};
    }

    std::vector<std::size_t> order(evidence.members.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        const auto& a = evidence.members[left].proposal;
        const auto& b = evidence.members[right].proposal;
        return std::tie(a.conversionIdentityHash, a.conversionIdentityCanonical) <
               std::tie(b.conversionIdentityHash, b.conversionIdentityCanonical);
    });
    std::vector<RecommendationConversionProposalPersistResult> proposals(
        evidence.members.size());
    int created = 0;
    for (const std::size_t index : order)
    {
        proposals[index] = PersistRecommendationConversionProposal(
            transaction, evidence.members[index].proposal);
        if (proposals[index].outcome ==
            RecommendationConversionProposalPersistOutcome::existingIdentical)
            continue;
        ++created;
    }
    const int reused = evidence.selectedMemberCount - created;
    const auto& approval = evidence.approval;
    const auto row = transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_materialization ("
        "recommendation_campaign_approval_id,materialization_contract_version,"
        "approval_identity_canonical,approval_identity_hash,"
        "recommendation_ranking_snapshot_id,ranking_snapshot_identity_canonical,"
        "ranking_snapshot_identity_hash,planning_policy_canonical,"
        "planning_policy_hash,planning_scope_canonical,"
        "campaign_plan_identity_canonical,campaign_plan_identity_hash,"
        "campaign_review_identity_canonical,campaign_review_identity_hash,"
        "approval_decision,approval_reviewer_identity,approval_reason_text,"
        "materialized_by,materialization_reason_text,selected_member_count,"
        "initially_created_proposal_count,initially_reused_proposal_count,"
        "materialization_identity_canonical,materialization_identity_hash) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,"
        "'approved',$15,$16,$17,$18,$19,$20,$21,$22,$23) RETURNING " +
        MaterializationColumns() + ";",
        pqxx::params{evidence.campaignApprovalId,
            evidence.materializationContractVersion,
            approval.approvalIdentityCanonical, approval.approvalIdentityHash,
            approval.rankingSnapshotId, approval.rankingSnapshotIdentityCanonical,
            approval.rankingSnapshotIdentityHash, approval.planningPolicyCanonical,
            approval.planningPolicyHash, approval.planningScopeCanonical,
            approval.campaignPlanIdentityCanonical,
            approval.campaignPlanIdentityHash,
            approval.campaignReviewIdentityCanonical,
            approval.campaignReviewIdentityHash, approval.reviewerIdentity,
            approval.reasonText, evidence.operatorIdentity, evidence.reasonText,
            evidence.selectedMemberCount, created, reused,
            evidence.materializationIdentityCanonical,
            evidence.materializationIdentityHash}).one_row();
    const long long materializationId = row[0].as<long long>();
    for (std::size_t index = 0; index < evidence.members.size(); ++index)
    {
        const auto& member = evidence.members[index];
        transaction.exec(
            "INSERT INTO experiment_recommendation_campaign_materialization_member ("
            "recommendation_campaign_materialization_id,member_ordinal,"
            "recommendation_ranking_member_id,recommendation_id,"
            "source_experiment_id,ranking_position,"
            "selected_member_identity_canonical,selected_member_identity_hash,"
            "recommendation_conversion_proposal_id,proposal_identity_canonical,"
            "proposal_identity_hash) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11);",
            pqxx::params{materializationId, member.memberOrdinal,
                member.rankingMemberId, member.recommendationId,
                member.sourceExperimentId, member.rankingPosition,
                member.selectedMemberIdentityCanonical,
                member.selectedMemberIdentityHash,
                proposals[index].persisted.proposalId,
                member.proposal.conversionIdentityCanonical,
                member.proposal.conversionIdentityHash});
    }
    auto persisted = FindById(transaction, materializationId);
    if (!persisted || !Matches(*persisted, evidence))
        throw std::runtime_error(
            "recommendation_campaign_materialization_persistence_mismatch");
    return {RecommendationCampaignMaterializationPersistOutcome::recorded,
            std::move(*persisted), created, reused};
}

std::optional<PersistedRecommendationCampaignMaterialization>
FindRecommendationCampaignMaterialization(
    pqxx::connection& connection,
    long long materializationId)
{
    pqxx::read_transaction transaction{connection};
    return FindRecommendationCampaignMaterialization(
        transaction, materializationId);
}

std::optional<PersistedRecommendationCampaignMaterialization>
FindRecommendationCampaignMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId)
{
    if (materializationId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_materialization_id_invalid");
    return FindById(transaction, materializationId);
}

std::vector<PersistedRecommendationCampaignMaterialization>
ListRecommendationCampaignMaterializations(
    pqxx::connection& connection,
    std::optional<long long> campaignApprovalId,
    int limit)
{
    pqxx::read_transaction transaction{connection};
    return ListRecommendationCampaignMaterializations(
        transaction, campaignApprovalId, limit);
}

std::vector<PersistedRecommendationCampaignMaterialization>
ListRecommendationCampaignMaterializations(
    pqxx::transaction_base& transaction,
    std::optional<long long> campaignApprovalId,
    int limit)
{
    if ((campaignApprovalId && *campaignApprovalId <= 0) || limit <= 0 ||
        limit > kMaximumRecommendationCampaignMaterializationListLimit)
        throw std::invalid_argument(
            "recommendation_campaign_materialization_list_argument_invalid");
    const auto rows = transaction.exec(
        "SELECT " + MaterializationColumns() + " FROM "
        "experiment_recommendation_campaign_materialization WHERE "
        "($1::bigint IS NULL OR recommendation_campaign_approval_id=$1) "
        "ORDER BY recommendation_campaign_materialization_id ASC LIMIT $2;",
        pqxx::params{campaignApprovalId, limit});
    std::vector<PersistedRecommendationCampaignMaterialization> values;
    values.reserve(rows.size());
    for (const auto& row : rows)
        values.push_back(MapMaterialization(transaction, row));
    return values;
}

} // namespace EA::ExperimentRecommendation
