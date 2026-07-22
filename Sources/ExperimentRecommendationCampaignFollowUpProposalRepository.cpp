#include "ExperimentRecommendationCampaignFollowUpProposalRepository.hpp"

#include "ExperimentRecommendation.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace EA::ExperimentRecommendation
{

struct RecommendationCampaignFollowUpProposalHydrationData
{
    int proposalContractVersion = 0;
    std::string proposalIdentityCanonical;
    std::string proposalIdentityHash;

    int assessmentContractVersion = 0;
    std::string assessmentCanonicalText;
    std::string assessmentIdentityHash;
    int policyContractVersion = 0;
    std::string policyCanonicalText;
    std::string policyIdentityHash;
    int policyDecisionContractVersion = 0;
    std::string policyDecisionCanonicalText;
    std::string policyDecisionIdentityHash;

    long long campaignApprovalId = 0;
    std::string campaignIdentityCanonical;
    std::string campaignIdentityHash;
    long long materializationId = 0;
    long long materializationCampaignApprovalId = 0;
    std::string materializationCampaignIdentityHash;
    int materializationContractVersion = 0;
    int memberCount = 0;
    std::string materializationIdentityCanonical;
    std::string materializationIdentityHash;

    RecommendationCampaignOutcomePolicyEvidenceSufficiency
        evidenceSufficiency =
            RecommendationCampaignOutcomePolicyEvidenceSufficiency::
                Insufficient;
    RecommendationCampaignOutcomePolicyCampaignInterpretation
        campaignInterpretation =
            RecommendationCampaignOutcomePolicyCampaignInterpretation::
                Inconclusive;
    RecommendationCampaignOutcomePolicyFollowUpEligibility
        followUpEligibility =
            RecommendationCampaignOutcomePolicyFollowUpEligibility::
                NotEligible;
    bool followUpAuthorized = false;
    RecommendationCampaignFollowUpProposalReason reason =
        RecommendationCampaignFollowUpProposalReason::
            EligibleFavorablePolicyDecision;
    std::vector<RecommendationCampaignOutcomeAssessmentMemberIdentity>
        members;
};

struct RecommendationCampaignFollowUpProposalPersistenceBuilder
{
    static RecommendationCampaignFollowUpProposal Hydrate(
        RecommendationCampaignFollowUpProposalHydrationData data)
    {
        RecommendationCampaignFollowUpProposalValidationView validation;
        validation.assessmentContractVersion =
            data.assessmentContractVersion;
        validation.assessmentCanonicalText = data.assessmentCanonicalText;
        validation.assessmentIdentityHash = data.assessmentIdentityHash;
        validation.policyContractVersion = data.policyContractVersion;
        validation.policyCanonicalText = data.policyCanonicalText;
        validation.policyIdentityHash = data.policyIdentityHash;
        validation.policyDecisionContractVersion =
            data.policyDecisionContractVersion;
        validation.policyDecisionCanonicalText =
            data.policyDecisionCanonicalText;
        validation.policyDecisionIdentityHash =
            data.policyDecisionIdentityHash;
        validation.decisionAssessmentContractVersion =
            data.assessmentContractVersion;
        validation.decisionAssessmentCanonicalText =
            data.assessmentCanonicalText;
        validation.decisionAssessmentIdentityHash =
            data.assessmentIdentityHash;
        validation.assessmentCampaignIdentity = {
            data.campaignApprovalId, data.campaignIdentityCanonical,
            data.campaignIdentityHash};
        validation.decisionCampaignIdentity =
            validation.assessmentCampaignIdentity;
        validation.assessmentMaterializationIdentity = {
            data.materializationId,
            data.materializationCampaignApprovalId,
            data.materializationCampaignIdentityHash,
            data.materializationContractVersion, data.memberCount,
            data.materializationIdentityCanonical,
            data.materializationIdentityHash};
        validation.decisionMaterializationIdentity =
            validation.assessmentMaterializationIdentity;
        validation.assessmentSummaryMemberCount = data.memberCount;
        validation.decisionSummaryMemberCount = data.memberCount;
        validation.evidenceSufficiency = data.evidenceSufficiency;
        validation.campaignInterpretation = data.campaignInterpretation;
        validation.followUpEligibility = data.followUpEligibility;
        validation.followUpAuthorized = data.followUpAuthorized;
        validation.assessmentMembers.reserve(data.members.size());
        for (const auto& member : data.members)
            validation.assessmentMembers.push_back({member.memberOrdinal,
                member.materializationMemberId, member.rankingMemberId,
                member.recommendationId, member.sourceExperimentId,
                member.proposalId, member.expectedExperimentId});
        validation.decisionMembers = validation.assessmentMembers;
        ValidateRecommendationCampaignFollowUpProposalInput(validation);

        std::vector<RecommendationCampaignFollowUpProposalMember> members;
        members.reserve(data.members.size());
        for (auto& member : data.members)
            members.push_back(RecommendationCampaignFollowUpProposalMember(
                std::move(member)));
        RecommendationCampaignFollowUpProposalSummary summary(
            data.evidenceSufficiency, data.campaignInterpretation,
            data.followUpEligibility, data.followUpAuthorized,
            data.memberCount, {data.reason});
        RecommendationCampaignFollowUpProposalIdentity identity(
            data.proposalContractVersion,
            std::move(data.proposalIdentityCanonical),
            std::move(data.proposalIdentityHash));
        RecommendationCampaignOutcomeAssessmentCampaignIdentity campaign(
            data.campaignApprovalId,
            std::move(data.campaignIdentityCanonical),
            std::move(data.campaignIdentityHash));
        RecommendationCampaignOutcomeAssessmentMaterializationIdentity
            materialization(data.materializationId,
                data.materializationCampaignApprovalId,
                std::move(data.materializationCampaignIdentityHash),
                data.materializationContractVersion, data.memberCount,
                std::move(data.materializationIdentityCanonical),
                std::move(data.materializationIdentityHash));
        return RecommendationCampaignFollowUpProposal(std::move(identity),
            data.assessmentContractVersion,
            std::move(data.assessmentCanonicalText),
            std::move(data.assessmentIdentityHash),
            data.policyContractVersion, std::move(data.policyCanonicalText),
            std::move(data.policyIdentityHash),
            data.policyDecisionContractVersion,
            std::move(data.policyDecisionCanonicalText),
            std::move(data.policyDecisionIdentityHash), std::move(campaign),
            std::move(materialization), data.memberCount,
            std::move(summary), std::move(members));
    }
};

namespace
{

using EvidenceSufficiency =
    RecommendationCampaignOutcomePolicyEvidenceSufficiency;
using Interpretation =
    RecommendationCampaignOutcomePolicyCampaignInterpretation;
using Eligibility = RecommendationCampaignOutcomePolicyFollowUpEligibility;
using ProposalReason = RecommendationCampaignFollowUpProposalReason;
using Persisted = PersistedRecommendationCampaignFollowUpProposal;

std::string ProposalColumns()
{
    return
        "recommendation_campaign_follow_up_proposal_id,"
        "proposal_contract_version,proposal_identity_canonical,"
        "proposal_identity_hash,proposal_identity_hash_collision_ordinal,"
        "assessment_contract_version,assessment_identity_canonical,"
        "assessment_identity_hash,policy_contract_version,"
        "policy_identity_canonical,policy_identity_hash,"
        "policy_decision_contract_version,"
        "policy_decision_identity_canonical,policy_decision_identity_hash,"
        "campaign_approval_id,campaign_identity_canonical,"
        "campaign_identity_hash,materialization_id,"
        "materialization_campaign_approval_id,"
        "materialization_campaign_identity_hash,"
        "materialization_contract_version,"
        "materialization_identity_canonical,materialization_identity_hash,"
        "evidence_sufficiency,campaign_interpretation,"
        "follow_up_eligibility,follow_up_authorized,proposal_reason,"
        "member_count,created_at::text AS created_at";
}

std::string MemberColumns()
{
    return
        "member_ordinal,materialization_member_id,ranking_member_id,"
        "recommendation_id,source_experiment_id,conversion_proposal_id,"
        "expected_experiment_id";
}

RecommendationCampaignFollowUpProposalHydrationData ReadHydrationData(
    pqxx::transaction_base& transaction,
    const pqxx::row& row)
{
    RecommendationCampaignFollowUpProposalHydrationData data;
    data.proposalContractVersion =
        row["proposal_contract_version"].as<int>();
    data.proposalIdentityCanonical =
        row["proposal_identity_canonical"].as<std::string>();
    data.proposalIdentityHash =
        row["proposal_identity_hash"].as<std::string>();
    data.assessmentContractVersion =
        row["assessment_contract_version"].as<int>();
    data.assessmentCanonicalText =
        row["assessment_identity_canonical"].as<std::string>();
    data.assessmentIdentityHash =
        row["assessment_identity_hash"].as<std::string>();
    data.policyContractVersion = row["policy_contract_version"].as<int>();
    data.policyCanonicalText =
        row["policy_identity_canonical"].as<std::string>();
    data.policyIdentityHash = row["policy_identity_hash"].as<std::string>();
    data.policyDecisionContractVersion =
        row["policy_decision_contract_version"].as<int>();
    data.policyDecisionCanonicalText =
        row["policy_decision_identity_canonical"].as<std::string>();
    data.policyDecisionIdentityHash =
        row["policy_decision_identity_hash"].as<std::string>();
    data.campaignApprovalId = row["campaign_approval_id"].as<long long>();
    data.campaignIdentityCanonical =
        row["campaign_identity_canonical"].as<std::string>();
    data.campaignIdentityHash =
        row["campaign_identity_hash"].as<std::string>();
    data.materializationId = row["materialization_id"].as<long long>();
    data.materializationCampaignApprovalId =
        row["materialization_campaign_approval_id"].as<long long>();
    data.materializationCampaignIdentityHash =
        row["materialization_campaign_identity_hash"].as<std::string>();
    data.materializationContractVersion =
        row["materialization_contract_version"].as<int>();
    data.memberCount = row["member_count"].as<int>();
    data.materializationIdentityCanonical =
        row["materialization_identity_canonical"].as<std::string>();
    data.materializationIdentityHash =
        row["materialization_identity_hash"].as<std::string>();

    if (row["evidence_sufficiency"].as<std::string>() != "sufficient" ||
        row["campaign_interpretation"].as<std::string>() != "favorable" ||
        row["follow_up_eligibility"].as<std::string>() !=
            "eligible_for_operator_review" ||
        row["proposal_reason"].as<std::string>() !=
            "eligible_favorable_policy_decision")
        throw std::runtime_error("persisted_classification_invalid");
    data.evidenceSufficiency = EvidenceSufficiency::Sufficient;
    data.campaignInterpretation = Interpretation::Favorable;
    data.followUpEligibility = Eligibility::EligibleForOperatorReview;
    data.followUpAuthorized = row["follow_up_authorized"].as<bool>();
    data.reason = ProposalReason::EligibleFavorablePolicyDecision;

    const long long persistedId =
        row["recommendation_campaign_follow_up_proposal_id"].as<long long>();
    const pqxx::result memberRows = transaction.exec(
        "SELECT " + MemberColumns() + " FROM "
        "experiment_recommendation_campaign_follow_up_proposal_member "
        "WHERE recommendation_campaign_follow_up_proposal_id=$1 "
        "ORDER BY member_ordinal ASC;", pqxx::params{persistedId});
    data.members.reserve(memberRows.size());
    for (const auto& member : memberRows)
    {
        const std::optional<long long> expectedExperimentId =
            member["expected_experiment_id"].is_null()
                ? std::nullopt
                : std::optional<long long>{
                      member["expected_experiment_id"].as<long long>()};
        data.members.emplace_back(member["member_ordinal"].as<int>(),
            member["materialization_member_id"].as<long long>(),
            member["ranking_member_id"].as<long long>(),
            member["recommendation_id"].as<long long>(),
            member["source_experiment_id"].as<long long>(),
            member["conversion_proposal_id"].as<long long>(),
            expectedExperimentId);
    }
    if (static_cast<int>(data.members.size()) != data.memberCount)
        throw std::runtime_error("persisted_member_count_mismatch");
    return data;
}

Persisted MapProposal(
    pqxx::transaction_base& transaction,
    const pqxx::row& row)
{
    try
    {
        const long long id =
            row["recommendation_campaign_follow_up_proposal_id"]
                .as<long long>();
        const int collisionOrdinal =
            row["proposal_identity_hash_collision_ordinal"].as<int>();
        const std::string createdAt = row["created_at"].as<std::string>();
        if (id <= 0 || collisionOrdinal < 0 || createdAt.empty())
            throw std::runtime_error("persisted_metadata_invalid");
        auto proposal =
            RecommendationCampaignFollowUpProposalPersistenceBuilder::
                Hydrate(ReadHydrationData(transaction, row));
        return Persisted(id, collisionOrdinal, std::move(proposal),
            createdAt);
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(
            "invalid_persisted_recommendation_campaign_follow_up_proposal:" +
            std::string(error.what()));
    }
}

std::optional<Persisted> FindByIdentity(
    pqxx::transaction_base& transaction,
    const std::string& canonical)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + ProposalColumns() + " FROM "
        "experiment_recommendation_campaign_follow_up_proposal WHERE "
        "proposal_identity_canonical=$1 LIMIT 2;",
        pqxx::params{canonical});
    if (rows.empty()) return std::nullopt;
    if (rows.size() != 1)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_duplicate_identity");
    return MapProposal(transaction, rows.one_row());
}

} // namespace

PersistedRecommendationCampaignFollowUpProposal::
    PersistedRecommendationCampaignFollowUpProposal(
        long long followUpProposalIdValue,
        int identityHashCollisionOrdinalValue,
        RecommendationCampaignFollowUpProposal proposalValue,
        std::string createdAtValue)
    : followUpProposalId(followUpProposalIdValue),
      identityHashCollisionOrdinal(identityHashCollisionOrdinalValue),
      proposal(std::move(proposalValue)),
      createdAt(std::move(createdAtValue))
{
}

RecommendationCampaignFollowUpProposalPersistResult::
    RecommendationCampaignFollowUpProposalPersistResult(
        RecommendationCampaignFollowUpProposalPersistOutcome outcomeValue,
        PersistedRecommendationCampaignFollowUpProposal persistedValue)
    : outcome(outcomeValue), persisted(std::move(persistedValue))
{
}

std::string RecommendationCampaignFollowUpProposalPersistOutcomeText(
    RecommendationCampaignFollowUpProposalPersistOutcome outcome)
{
    switch (outcome)
    {
        case RecommendationCampaignFollowUpProposalPersistOutcome::recorded:
            return "recorded";
        case RecommendationCampaignFollowUpProposalPersistOutcome::
                existingIdentical:
            return "existing_identical";
    }
    throw std::invalid_argument(
        "recommendation_campaign_follow_up_proposal_persist_outcome_invalid");
}

bool RecommendationCampaignFollowUpProposalSchemaExists(
    pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return RecommendationCampaignFollowUpProposalSchemaExists(transaction);
}

bool RecommendationCampaignFollowUpProposalSchemaExists(
    pqxx::transaction_base& transaction)
{
    return transaction.exec(
        "SELECT to_regclass('experiment_recommendation_campaign_follow_up_"
        "proposal') IS NOT NULL AND to_regclass('experiment_recommendation_"
        "campaign_follow_up_proposal_member') IS NOT NULL;")
        .one_row()[0].as<bool>();
}

RecommendationCampaignFollowUpProposalPersistResult
PersistRecommendationCampaignFollowUpProposal(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignFollowUpProposal& proposal)
{
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, "
        "6062001007003009));",
        pqxx::params{proposal.identity.hash});
    if (auto existing = FindByIdentity(
            transaction, proposal.identity.canonicalText))
    {
        if (existing->proposal != proposal)
            throw std::runtime_error(
                "recommendation_campaign_follow_up_proposal_identity_conflict");
        return {
            RecommendationCampaignFollowUpProposalPersistOutcome::
                existingIdentical,
            std::move(*existing)};
    }

    const int collisionOrdinal = transaction.exec(
        "SELECT coalesce(max(proposal_identity_hash_collision_ordinal),-1)+1 "
        "FROM experiment_recommendation_campaign_follow_up_proposal WHERE "
        "proposal_identity_hash=$1;", pqxx::params{proposal.identity.hash})
        .one_row()[0].as<int>();
    const auto& campaign = proposal.campaignIdentity;
    const auto& materialization = proposal.materializationIdentity;
    const auto& summary = proposal.summary;
    const long long id = transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_follow_up_proposal ("
        "proposal_contract_version,proposal_identity_canonical,"
        "proposal_identity_hash,proposal_identity_hash_collision_ordinal,"
        "assessment_contract_version,assessment_identity_canonical,"
        "assessment_identity_hash,policy_contract_version,"
        "policy_identity_canonical,policy_identity_hash,"
        "policy_decision_contract_version,"
        "policy_decision_identity_canonical,policy_decision_identity_hash,"
        "campaign_approval_id,campaign_identity_canonical,"
        "campaign_identity_hash,materialization_id,"
        "materialization_campaign_approval_id,"
        "materialization_campaign_identity_hash,"
        "materialization_contract_version,"
        "materialization_identity_canonical,materialization_identity_hash,"
        "evidence_sufficiency,campaign_interpretation,"
        "follow_up_eligibility,follow_up_authorized,proposal_reason,"
        "member_count) VALUES ("
        "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,"
        "$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28) RETURNING "
        "recommendation_campaign_follow_up_proposal_id;",
        pqxx::params{proposal.identity.contractVersion,
            proposal.identity.canonicalText, proposal.identity.hash,
            collisionOrdinal, proposal.assessmentContractVersion,
            proposal.assessmentCanonicalText,
            proposal.assessmentIdentityHash, proposal.policyContractVersion,
            proposal.policyCanonicalText, proposal.policyIdentityHash,
            proposal.policyDecisionContractVersion,
            proposal.policyDecisionCanonicalText,
            proposal.policyDecisionIdentityHash,
            campaign.campaignApprovalId, campaign.identityCanonical,
            campaign.identityHash, materialization.materializationId,
            materialization.campaignApprovalId,
            materialization.campaignIdentityHash,
            materialization.contractVersion,
            materialization.identityCanonical,
            materialization.identityHash,
            RecommendationCampaignOutcomePolicyEvidenceSufficiencyText(
                summary.evidenceSufficiency),
            RecommendationCampaignOutcomePolicyCampaignInterpretationText(
                summary.campaignInterpretation),
            RecommendationCampaignOutcomePolicyFollowUpEligibilityText(
                summary.followUpEligibility), summary.followUpAuthorized,
            RecommendationCampaignFollowUpProposalReasonText(
                summary.reasons.front()),
            proposal.memberCount})
        .one_row()[0].as<long long>();

    for (const auto& member : proposal.members)
    {
        transaction.exec(
            "INSERT INTO experiment_recommendation_campaign_follow_up_"
            "proposal_member (recommendation_campaign_follow_up_proposal_id,"
            "member_ordinal,materialization_member_id,ranking_member_id,"
            "recommendation_id,source_experiment_id,conversion_proposal_id,"
            "expected_experiment_id) VALUES ($1,$2,$3,$4,$5,$6,$7,$8);",
            pqxx::params{id, member.identity.memberOrdinal,
                member.identity.materializationMemberId,
                member.identity.rankingMemberId,
                member.identity.recommendationId,
                member.identity.sourceExperimentId,
                member.identity.proposalId,
                member.identity.expectedExperimentId});
    }
    auto persisted = FindRecommendationCampaignFollowUpProposal(
        transaction, id);
    if (!persisted)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_insert_missing");
    return {RecommendationCampaignFollowUpProposalPersistOutcome::recorded,
        std::move(*persisted)};
}

std::optional<PersistedRecommendationCampaignFollowUpProposal>
FindRecommendationCampaignFollowUpProposal(
    pqxx::connection& connection,
    long long followUpProposalId)
{
    pqxx::read_transaction transaction{connection};
    return FindRecommendationCampaignFollowUpProposal(
        transaction, followUpProposalId);
}

std::optional<PersistedRecommendationCampaignFollowUpProposal>
FindRecommendationCampaignFollowUpProposal(
    pqxx::transaction_base& transaction,
    long long followUpProposalId)
{
    if (followUpProposalId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_follow_up_proposal_id_invalid");
    const pqxx::result rows = transaction.exec(
        "SELECT " + ProposalColumns() + " FROM "
        "experiment_recommendation_campaign_follow_up_proposal WHERE "
        "recommendation_campaign_follow_up_proposal_id=$1;",
        pqxx::params{followUpProposalId});
    if (rows.empty()) return std::nullopt;
    return MapProposal(transaction, rows.one_row());
}

std::optional<PersistedRecommendationCampaignFollowUpProposal>
FindRecommendationCampaignFollowUpProposalByIdentity(
    pqxx::connection& connection,
    const std::string& proposalIdentityCanonical)
{
    pqxx::read_transaction transaction{connection};
    return FindRecommendationCampaignFollowUpProposalByIdentity(
        transaction, proposalIdentityCanonical);
}

std::optional<PersistedRecommendationCampaignFollowUpProposal>
FindRecommendationCampaignFollowUpProposalByIdentity(
    pqxx::transaction_base& transaction,
    const std::string& proposalIdentityCanonical)
{
    if (proposalIdentityCanonical.empty() ||
        proposalIdentityCanonical.size() >
            kMaximumRecommendationCampaignFollowUpProposalCanonicalTextBytes)
        throw std::invalid_argument(
            "recommendation_campaign_follow_up_proposal_identity_invalid");
    return FindByIdentity(transaction, proposalIdentityCanonical);
}

} // namespace EA::ExperimentRecommendation
