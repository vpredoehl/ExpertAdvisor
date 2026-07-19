#include "ExperimentRecommendationCampaignApprovalRepository.hpp"

#include <stdexcept>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string ApprovalColumns()
{
    return
        "recommendation_campaign_approval_id,approval_contract_version,"
        "recommendation_ranking_snapshot_id,"
        "ranking_snapshot_identity_canonical,ranking_snapshot_identity_hash,"
        "planning_policy_canonical,planning_policy_hash,"
        "planning_scope_canonical,campaign_plan_identity_canonical,"
        "campaign_plan_identity_hash,review_contract_version,"
        "campaign_review_identity_canonical,campaign_review_identity_hash,"
        "campaign_review_hash_collision_ordinal,candidate_count,selected_count,"
        "excluded_count,duplicate_group_count,duplicate_candidate_count,"
        "considered_family_count,selected_family_count,considered_symbol_count,"
        "selected_symbol_count,considered_horizon_count,selected_horizon_count,"
        "deterministic_ordering_verified,decision,reviewer_identity,reason_text,"
        "approval_identity_canonical,approval_identity_hash,"
        "created_at::text AS created_at";
}

PersistedRecommendationCampaignApproval MapApproval(const pqxx::row& row)
{
    PersistedRecommendationCampaignApproval persisted;
    persisted.campaignApprovalId =
        row["recommendation_campaign_approval_id"].as<long long>();
    auto& value = persisted.evidence;
    value.approvalContractVersion =
        row["approval_contract_version"].as<int>();
    value.rankingSnapshotId =
        row["recommendation_ranking_snapshot_id"].as<long long>();
    value.rankingSnapshotIdentityCanonical =
        row["ranking_snapshot_identity_canonical"].as<std::string>();
    value.rankingSnapshotIdentityHash =
        row["ranking_snapshot_identity_hash"].as<std::string>();
    value.planningPolicyCanonical =
        row["planning_policy_canonical"].as<std::string>();
    value.planningPolicyHash =
        row["planning_policy_hash"].as<std::string>();
    value.planningScopeCanonical =
        row["planning_scope_canonical"].as<std::string>();
    value.campaignPlanIdentityCanonical =
        row["campaign_plan_identity_canonical"].as<std::string>();
    value.campaignPlanIdentityHash =
        row["campaign_plan_identity_hash"].as<std::string>();
    value.reviewContractVersion = row["review_contract_version"].as<int>();
    value.campaignReviewIdentityCanonical =
        row["campaign_review_identity_canonical"].as<std::string>();
    value.campaignReviewIdentityHash =
        row["campaign_review_identity_hash"].as<std::string>();
    persisted.reviewHashCollisionOrdinal =
        row["campaign_review_hash_collision_ordinal"].as<int>();
    value.summary.candidateCount = row["candidate_count"].as<int>();
    value.summary.selectedCount = row["selected_count"].as<int>();
    value.summary.excludedCount = row["excluded_count"].as<int>();
    value.summary.duplicateGroupCount =
        row["duplicate_group_count"].as<int>();
    value.summary.duplicateCandidateCount =
        row["duplicate_candidate_count"].as<int>();
    value.summary.consideredFamilyCount =
        row["considered_family_count"].as<int>();
    value.summary.selectedFamilyCount =
        row["selected_family_count"].as<int>();
    value.summary.consideredSymbolCount =
        row["considered_symbol_count"].as<int>();
    value.summary.selectedSymbolCount =
        row["selected_symbol_count"].as<int>();
    value.summary.consideredHorizonCount =
        row["considered_horizon_count"].as<int>();
    value.summary.selectedHorizonCount =
        row["selected_horizon_count"].as<int>();
    value.summary.deterministicOrderingVerified =
        row["deterministic_ordering_verified"].as<bool>();
    const auto decision = ParseRecommendationCampaignApprovalDecision(
        row["decision"].as<std::string>());
    if (!decision)
        throw std::runtime_error(
            "recommendation_campaign_approval_persisted_decision_invalid");
    value.decision = *decision;
    value.reviewerIdentity = row["reviewer_identity"].as<std::string>();
    value.reasonText = row["reason_text"].as<std::string>();
    value.approvalIdentityCanonical =
        row["approval_identity_canonical"].as<std::string>();
    value.approvalIdentityHash =
        row["approval_identity_hash"].as<std::string>();
    persisted.createdAt = row["created_at"].as<std::string>();
    ValidateRecommendationCampaignApprovalEvidence(value);
    return persisted;
}

std::optional<PersistedRecommendationCampaignApproval> FindByReviewCanonical(
    pqxx::transaction_base& transaction,
    const std::string& canonical)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + ApprovalColumns() + " FROM "
        "experiment_recommendation_campaign_approval WHERE "
        "campaign_review_identity_canonical=$1 LIMIT 2;",
        pqxx::params{canonical});
    if (rows.empty()) return std::nullopt;
    if (rows.size() != 1)
        throw std::runtime_error(
            "recommendation_campaign_approval_duplicate_review_identity");
    return MapApproval(rows.one_row());
}

bool Matches(
    const RecommendationCampaignApprovalEvidence& left,
    const RecommendationCampaignApprovalEvidence& right)
{
    return left.approvalContractVersion == right.approvalContractVersion &&
           left.rankingSnapshotId == right.rankingSnapshotId &&
           left.rankingSnapshotIdentityCanonical ==
               right.rankingSnapshotIdentityCanonical &&
           left.rankingSnapshotIdentityHash == right.rankingSnapshotIdentityHash &&
           left.planningPolicyCanonical == right.planningPolicyCanonical &&
           left.planningPolicyHash == right.planningPolicyHash &&
           left.planningScopeCanonical == right.planningScopeCanonical &&
           left.campaignPlanIdentityCanonical ==
               right.campaignPlanIdentityCanonical &&
           left.campaignPlanIdentityHash == right.campaignPlanIdentityHash &&
           left.reviewContractVersion == right.reviewContractVersion &&
           left.campaignReviewIdentityCanonical ==
               right.campaignReviewIdentityCanonical &&
           left.campaignReviewIdentityHash == right.campaignReviewIdentityHash &&
           left.summary.candidateCount == right.summary.candidateCount &&
           left.summary.selectedCount == right.summary.selectedCount &&
           left.summary.excludedCount == right.summary.excludedCount &&
           left.summary.duplicateGroupCount ==
               right.summary.duplicateGroupCount &&
           left.summary.duplicateCandidateCount ==
               right.summary.duplicateCandidateCount &&
           left.summary.consideredFamilyCount ==
               right.summary.consideredFamilyCount &&
           left.summary.selectedFamilyCount ==
               right.summary.selectedFamilyCount &&
           left.summary.consideredSymbolCount ==
               right.summary.consideredSymbolCount &&
           left.summary.selectedSymbolCount ==
               right.summary.selectedSymbolCount &&
           left.summary.consideredHorizonCount ==
               right.summary.consideredHorizonCount &&
           left.summary.selectedHorizonCount ==
               right.summary.selectedHorizonCount &&
           left.summary.deterministicOrderingVerified ==
               right.summary.deterministicOrderingVerified &&
           left.decision == right.decision &&
           left.reviewerIdentity == right.reviewerIdentity &&
           left.reasonText == right.reasonText &&
           left.approvalIdentityCanonical == right.approvalIdentityCanonical &&
           left.approvalIdentityHash == right.approvalIdentityHash;
}

void ValidateLimit(int limit)
{
    if (limit <= 0 || limit > kMaximumRecommendationCampaignApprovalListLimit)
        throw std::invalid_argument(
            "recommendation_campaign_approval_limit_invalid");
}

} // namespace

std::string RecommendationCampaignApprovalPersistOutcomeText(
    RecommendationCampaignApprovalPersistOutcome outcome)
{
    switch (outcome)
    {
        case RecommendationCampaignApprovalPersistOutcome::recorded:
            return "recorded";
        case RecommendationCampaignApprovalPersistOutcome::existingIdentical:
            return "existing_identical";
    }
    throw std::invalid_argument(
        "recommendation_campaign_approval_outcome_invalid");
}

bool RecommendationCampaignApprovalSchemaExists(pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return RecommendationCampaignApprovalSchemaExists(transaction);
}

bool RecommendationCampaignApprovalSchemaExists(
    pqxx::transaction_base& transaction)
{
    return transaction.exec(
        "SELECT to_regclass('experiment_recommendation_campaign_approval') "
        "IS NOT NULL;").one_row()[0].as<bool>();
}

RecommendationCampaignApprovalPersistResult PersistRecommendationCampaignApproval(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignApprovalEvidence& evidence)
{
    ValidateRecommendationCampaignApprovalEvidence(evidence);
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, "
        "4354685564936845355));",
        pqxx::params{evidence.campaignReviewIdentityHash});
    if (auto existing = FindByReviewCanonical(
            transaction, evidence.campaignReviewIdentityCanonical))
    {
        if (!Matches(existing->evidence, evidence))
            throw std::runtime_error(
                "recommendation_campaign_approval_review_conflict");
        return {
            RecommendationCampaignApprovalPersistOutcome::existingIdentical,
            std::move(*existing)};
    }

    const int collisionOrdinal = transaction.exec(
        "SELECT coalesce(max(campaign_review_hash_collision_ordinal),-1)+1 "
        "FROM experiment_recommendation_campaign_approval WHERE "
        "campaign_review_identity_hash=$1;",
        pqxx::params{evidence.campaignReviewIdentityHash})
        .one_row()[0].as<int>();
    const auto& summary = evidence.summary;
    const pqxx::row row = transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_approval ("
        "approval_contract_version,recommendation_ranking_snapshot_id,"
        "ranking_snapshot_identity_canonical,ranking_snapshot_identity_hash,"
        "planning_policy_canonical,planning_policy_hash,"
        "planning_scope_canonical,campaign_plan_identity_canonical,"
        "campaign_plan_identity_hash,review_contract_version,"
        "campaign_review_identity_canonical,campaign_review_identity_hash,"
        "campaign_review_hash_collision_ordinal,candidate_count,selected_count,"
        "excluded_count,duplicate_group_count,duplicate_candidate_count,"
        "considered_family_count,selected_family_count,considered_symbol_count,"
        "selected_symbol_count,considered_horizon_count,selected_horizon_count,"
        "deterministic_ordering_verified,decision,reviewer_identity,reason_text,"
        "approval_identity_canonical,approval_identity_hash) VALUES ("
        "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,"
        "$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30) RETURNING " +
        ApprovalColumns() + ";",
        pqxx::params{
            evidence.approvalContractVersion, evidence.rankingSnapshotId,
            evidence.rankingSnapshotIdentityCanonical,
            evidence.rankingSnapshotIdentityHash,
            evidence.planningPolicyCanonical, evidence.planningPolicyHash,
            evidence.planningScopeCanonical,
            evidence.campaignPlanIdentityCanonical,
            evidence.campaignPlanIdentityHash, evidence.reviewContractVersion,
            evidence.campaignReviewIdentityCanonical,
            evidence.campaignReviewIdentityHash, collisionOrdinal,
            summary.candidateCount, summary.selectedCount,
            summary.excludedCount, summary.duplicateGroupCount,
            summary.duplicateCandidateCount, summary.consideredFamilyCount,
            summary.selectedFamilyCount, summary.consideredSymbolCount,
            summary.selectedSymbolCount, summary.consideredHorizonCount,
            summary.selectedHorizonCount,
            summary.deterministicOrderingVerified,
            RecommendationCampaignApprovalDecisionText(evidence.decision),
            evidence.reviewerIdentity, evidence.reasonText,
            evidence.approvalIdentityCanonical,
            evidence.approvalIdentityHash}).one_row();
    return {
        RecommendationCampaignApprovalPersistOutcome::recorded,
        MapApproval(row)};
}

std::optional<PersistedRecommendationCampaignApproval>
FindRecommendationCampaignApproval(
    pqxx::connection& connection,
    long long campaignApprovalId)
{
    if (campaignApprovalId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_approval_id_invalid");
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT " + ApprovalColumns() + " FROM "
        "experiment_recommendation_campaign_approval WHERE "
        "recommendation_campaign_approval_id=$1;",
        pqxx::params{campaignApprovalId});
    if (rows.empty()) return std::nullopt;
    return MapApproval(rows.one_row());
}

std::optional<PersistedRecommendationCampaignApproval>
FindRecommendationCampaignApprovalByReviewIdentity(
    pqxx::connection& connection,
    const std::string& campaignReviewIdentityCanonical)
{
    if (campaignReviewIdentityCanonical.empty())
        throw std::invalid_argument(
            "recommendation_campaign_approval_review_identity_invalid");
    pqxx::read_transaction transaction{connection};
    return FindByReviewCanonical(transaction, campaignReviewIdentityCanonical);
}

std::vector<PersistedRecommendationCampaignApproval>
ListRecommendationCampaignApprovals(
    pqxx::connection& connection,
    std::optional<RecommendationCampaignApprovalDecision> decision,
    int limit)
{
    ValidateLimit(limit);
    const std::optional<std::string> decisionText = decision
        ? std::optional<std::string>{
              RecommendationCampaignApprovalDecisionText(*decision)}
        : std::nullopt;
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT " + ApprovalColumns() + " FROM "
        "experiment_recommendation_campaign_approval WHERE "
        "($1::text IS NULL OR decision=$1) ORDER BY "
        "recommendation_campaign_approval_id ASC LIMIT $2;",
        pqxx::params{decisionText, limit});
    std::vector<PersistedRecommendationCampaignApproval> values;
    values.reserve(rows.size());
    for (const pqxx::row& row : rows) values.push_back(MapApproval(row));
    return values;
}

} // namespace EA::ExperimentRecommendation
