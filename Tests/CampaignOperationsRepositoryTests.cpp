#include "../Sources/CampaignOperationsRepository.hpp"
#include "../Sources/CampaignOperationsService.hpp"

#include "../Sources/ExperimentRecommendation.hpp"
#include "../Sources/ExperimentRecommendationCampaignFollowUpProposalRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignMaterializationRepository.hpp"

#include <atomic>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#include <pqxx/pqxx>

using namespace EA::CampaignOperations;

namespace EA::ExperimentRecommendation
{

struct RecommendationCampaignFollowUpProposalPersistenceBuilder
{
    static RecommendationCampaignFollowUpProposal BuildFixture(
        long long proposalId, long long materializationId,
        int materializationContractVersion,
        const std::string& materializationCanonical,
        const std::string& materializationHash, int memberCount,
        const std::string& proposalCanonical,
        const std::string& proposalHash)
    {
        (void)proposalId;
        const long long campaignApprovalId = materializationId + 1000;
        const std::string campaignCanonical =
            "test-campaign-" + std::to_string(materializationId);
        const std::string campaignHash =
            RecommendationCanonicalHash(campaignCanonical);
        std::vector<RecommendationCampaignFollowUpProposalMember> members;
        members.reserve(static_cast<std::size_t>(memberCount));
        for (int ordinal = 1; ordinal <= memberCount; ++ordinal)
        {
            members.push_back(RecommendationCampaignFollowUpProposalMember(
                RecommendationCampaignOutcomeAssessmentMemberIdentity(
                    ordinal, materializationId * 100 + ordinal,
                    materializationId * 1000 + ordinal,
                    materializationId * 2000 + ordinal,
                    materializationId * 3000 + ordinal,
                    materializationId * 4000 + ordinal, std::nullopt)));
        }
        return RecommendationCampaignFollowUpProposal(
            RecommendationCampaignFollowUpProposalIdentity(
                kRecommendationCampaignFollowUpProposalContractVersion,
                proposalCanonical, proposalHash),
            kRecommendationCampaignOutcomeAssessmentContractVersion,
            "test-assessment", RecommendationCanonicalHash("test-assessment"),
            1, "test-policy", RecommendationCanonicalHash("test-policy"), 1,
            "test-policy-decision",
            RecommendationCanonicalHash("test-policy-decision"),
            RecommendationCampaignOutcomeAssessmentCampaignIdentity(
                campaignApprovalId, campaignCanonical, campaignHash),
            RecommendationCampaignOutcomeAssessmentMaterializationIdentity(
                materializationId, campaignApprovalId, campaignHash,
                materializationContractVersion, memberCount,
                materializationCanonical, materializationHash),
            memberCount, RecommendationCampaignFollowUpProposalSummary(
                RecommendationCampaignOutcomePolicyEvidenceSufficiency::
                    Sufficient,
                RecommendationCampaignOutcomePolicyCampaignInterpretation::
                    Favorable,
                RecommendationCampaignOutcomePolicyFollowUpEligibility::
                    EligibleForOperatorReview,
                true, memberCount,
                {RecommendationCampaignFollowUpProposalReason::
                    EligibleFavorablePolicyDecision}),
            std::move(members));
    }
};

RecommendationCampaignOutcomeAssessmentCampaignIdentity::
    RecommendationCampaignOutcomeAssessmentCampaignIdentity(
        long long campaignApprovalIdValue, std::string identityCanonicalValue,
        std::string identityHashValue)
    : campaignApprovalId(campaignApprovalIdValue),
      identityCanonical(std::move(identityCanonicalValue)),
      identityHash(std::move(identityHashValue))
{
}

RecommendationCampaignOutcomeAssessmentMaterializationIdentity::
    RecommendationCampaignOutcomeAssessmentMaterializationIdentity(
        long long materializationIdValue, long long campaignApprovalIdValue,
        std::string campaignIdentityHashValue, int contractVersionValue,
        int memberCountValue, std::string identityCanonicalValue,
        std::string identityHashValue)
    : materializationId(materializationIdValue),
      campaignApprovalId(campaignApprovalIdValue),
      campaignIdentityHash(std::move(campaignIdentityHashValue)),
      contractVersion(contractVersionValue), memberCount(memberCountValue),
      identityCanonical(std::move(identityCanonicalValue)),
      identityHash(std::move(identityHashValue))
{
}

RecommendationCampaignOutcomeAssessmentMemberIdentity::
    RecommendationCampaignOutcomeAssessmentMemberIdentity(
        int memberOrdinalValue, long long materializationMemberIdValue,
        long long rankingMemberIdValue, long long recommendationIdValue,
        long long sourceExperimentIdValue, long long proposalIdValue,
        std::optional<long long> expectedExperimentIdValue)
    : memberOrdinal(memberOrdinalValue),
      materializationMemberId(materializationMemberIdValue),
      rankingMemberId(rankingMemberIdValue),
      recommendationId(recommendationIdValue),
      sourceExperimentId(sourceExperimentIdValue),
      proposalId(proposalIdValue),
      expectedExperimentId(std::move(expectedExperimentIdValue))
{
}

RecommendationCampaignFollowUpProposalIdentity::
    RecommendationCampaignFollowUpProposalIdentity(
        int contractVersionValue, std::string canonicalTextValue,
        std::string hashValue)
    : contractVersion(contractVersionValue),
      canonicalText(std::move(canonicalTextValue)),
      hash(std::move(hashValue))
{
}

RecommendationCampaignFollowUpProposalMember::
    RecommendationCampaignFollowUpProposalMember(
        RecommendationCampaignOutcomeAssessmentMemberIdentity identityValue)
    : identity(std::move(identityValue))
{
}

RecommendationCampaignFollowUpProposalSummary::
    RecommendationCampaignFollowUpProposalSummary(
        RecommendationCampaignOutcomePolicyEvidenceSufficiency
            evidenceSufficiencyValue,
        RecommendationCampaignOutcomePolicyCampaignInterpretation
            campaignInterpretationValue,
        RecommendationCampaignOutcomePolicyFollowUpEligibility
            followUpEligibilityValue,
        bool followUpAuthorizedValue, int memberCountValue,
        std::vector<RecommendationCampaignFollowUpProposalReason> reasonsValue)
    : evidenceSufficiency(evidenceSufficiencyValue),
      campaignInterpretation(campaignInterpretationValue),
      followUpEligibility(followUpEligibilityValue),
      followUpAuthorized(followUpAuthorizedValue),
      memberCount(memberCountValue), reasons(std::move(reasonsValue))
{
}

RecommendationCampaignFollowUpProposal::
    RecommendationCampaignFollowUpProposal(
        RecommendationCampaignFollowUpProposalIdentity identityValue,
        int assessmentContractVersionValue,
        std::string assessmentCanonicalTextValue,
        std::string assessmentIdentityHashValue,
        int policyContractVersionValue, std::string policyCanonicalTextValue,
        std::string policyIdentityHashValue,
        int policyDecisionContractVersionValue,
        std::string policyDecisionCanonicalTextValue,
        std::string policyDecisionIdentityHashValue,
        RecommendationCampaignOutcomeAssessmentCampaignIdentity
            campaignIdentityValue,
        RecommendationCampaignOutcomeAssessmentMaterializationIdentity
            materializationIdentityValue,
        int memberCountValue,
        RecommendationCampaignFollowUpProposalSummary summaryValue,
        std::vector<RecommendationCampaignFollowUpProposalMember> membersValue)
    : identity(std::move(identityValue)),
      assessmentContractVersion(assessmentContractVersionValue),
      assessmentCanonicalText(std::move(assessmentCanonicalTextValue)),
      assessmentIdentityHash(std::move(assessmentIdentityHashValue)),
      policyContractVersion(policyContractVersionValue),
      policyCanonicalText(std::move(policyCanonicalTextValue)),
      policyIdentityHash(std::move(policyIdentityHashValue)),
      policyDecisionContractVersion(policyDecisionContractVersionValue),
      policyDecisionCanonicalText(
          std::move(policyDecisionCanonicalTextValue)),
      policyDecisionIdentityHash(std::move(policyDecisionIdentityHashValue)),
      campaignIdentity(std::move(campaignIdentityValue)),
      materializationIdentity(std::move(materializationIdentityValue)),
      memberCount(memberCountValue), summary(std::move(summaryValue)),
      members(std::move(membersValue))
{
}

PersistedRecommendationCampaignFollowUpProposal::
    PersistedRecommendationCampaignFollowUpProposal(
        long long followUpProposalIdValue,
        int identityHashCollisionOrdinalValue,
        RecommendationCampaignFollowUpProposal proposalValue,
        std::string createdAtValue)
    : followUpProposalId(followUpProposalIdValue),
      identityHashCollisionOrdinal(identityHashCollisionOrdinalValue),
      proposal(std::move(proposalValue)), createdAt(std::move(createdAtValue))
{
}

struct RecommendationCampaignFollowUpProposalRatificationBuilder
{
    static RecommendationCampaignFollowUpProposalRatification BuildFixture(
        long long ratificationEventId, int ratificationContractVersion,
        long long reviewEventId, int reviewContractVersion,
        const std::string& reviewCanonical, const std::string& reviewHash,
        long long proposalId, int proposalContractVersion,
        const std::string& proposalCanonical, const std::string& proposalHash,
        const std::string& ratificationCanonical,
        const std::string& ratificationHash)
    {
        return RecommendationCampaignFollowUpProposalRatification(
            RecommendationCampaignFollowUpProposalRatificationIdentity(
                ratificationContractVersion, ratificationCanonical,
                ratificationHash),
            reviewEventId, reviewContractVersion, reviewCanonical, reviewHash,
            proposalId, proposalContractVersion, proposalCanonical,
            proposalHash,
            RecommendationCampaignFollowUpProposalReviewDecision::approved,
            "reviewer@example.test",
            kRecommendationCampaignFollowUpProposalRatificationAuthorityRole,
            RecommendationCampaignFollowUpProposalRatificationDecision::
                ratified,
            "ratifier@example.test",
            "Focused Campaign Operations upstream fixture " +
                std::to_string(ratificationEventId) + ".");
    }
};

RecommendationCampaignFollowUpProposalRatificationIdentity::
    RecommendationCampaignFollowUpProposalRatificationIdentity(
        int contractVersionValue, std::string canonicalTextValue,
        std::string hashValue)
    : contractVersion(contractVersionValue),
      canonicalText(std::move(canonicalTextValue)),
      hash(std::move(hashValue))
{
}

RecommendationCampaignFollowUpProposalRatification::
    RecommendationCampaignFollowUpProposalRatification(
        RecommendationCampaignFollowUpProposalRatificationIdentity identityValue,
        long long reviewEventIdValue, int reviewContractVersionValue,
        std::string reviewCanonicalTextValue,
        std::string reviewIdentityHashValue, long long followUpProposalIdValue,
        int proposalContractVersionValue,
        std::string proposalCanonicalTextValue,
        std::string proposalIdentityHashValue,
        RecommendationCampaignFollowUpProposalReviewDecision
            reviewDecisionValue,
        std::string reviewerIdentityValue,
        std::string ratificationAuthorityRoleValue,
        RecommendationCampaignFollowUpProposalRatificationDecision
            decisionValue,
        std::string ratifierIdentityValue,
        std::string ratificationBasisValue)
    : identity(std::move(identityValue)), reviewEventId(reviewEventIdValue),
      reviewContractVersion(reviewContractVersionValue),
      reviewCanonicalText(std::move(reviewCanonicalTextValue)),
      reviewIdentityHash(std::move(reviewIdentityHashValue)),
      followUpProposalId(followUpProposalIdValue),
      proposalContractVersion(proposalContractVersionValue),
      proposalCanonicalText(std::move(proposalCanonicalTextValue)),
      proposalIdentityHash(std::move(proposalIdentityHashValue)),
      reviewDecision(reviewDecisionValue),
      reviewerIdentity(std::move(reviewerIdentityValue)),
      ratificationAuthorityRole(std::move(ratificationAuthorityRoleValue)),
      decision(decisionValue), ratifierIdentity(std::move(ratifierIdentityValue)),
      ratificationBasis(std::move(ratificationBasisValue))
{
}

PersistedRecommendationCampaignFollowUpProposalRatification::
    PersistedRecommendationCampaignFollowUpProposalRatification(
        long long ratificationEventIdValue,
        RecommendationCampaignFollowUpProposalRatification ratificationValue,
        std::string createdAtValue)
    : ratificationEventId(ratificationEventIdValue),
      ratification(std::move(ratificationValue)),
      createdAt(std::move(createdAtValue))
{
}

// The focused repository test supplies functional upstream read adapters.
// Production links the established repositories.
std::optional<PersistedRecommendationCampaignMaterialization>
FindRecommendationCampaignMaterialization(
    pqxx::transaction_base& transaction, long long materializationId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT recommendation_campaign_materialization_id,"
        "recommendation_campaign_approval_id,materialization_contract_version,"
        "approval_identity_hash,recommendation_ranking_snapshot_id,"
        "ranking_snapshot_identity_hash,planning_policy_hash,"
        "campaign_plan_identity_hash,campaign_review_identity_hash,"
        "materialized_by,materialization_reason_text,selected_member_count,"
        "initially_created_proposal_count,initially_reused_proposal_count,"
        "materialization_identity_canonical,materialization_identity_hash,"
        "created_at::text FROM "
        "experiment_recommendation_campaign_materialization WHERE "
        "recommendation_campaign_materialization_id=$1;",
        pqxx::params{materializationId});
    if (rows.empty()) return std::nullopt;
    const auto& row = rows.one_row();
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
    const pqxx::result members = transaction.exec(
        "SELECT recommendation_campaign_materialization_member_id,"
        "member_ordinal,recommendation_ranking_member_id,recommendation_id,"
        "source_experiment_id,ranking_position,"
        "selected_member_identity_canonical,selected_member_identity_hash,"
        "recommendation_conversion_proposal_id,proposal_identity_canonical,"
        "proposal_identity_hash,created_at::text FROM "
        "experiment_recommendation_campaign_materialization_member WHERE "
        "recommendation_campaign_materialization_id=$1 "
        "ORDER BY member_ordinal;",
        pqxx::params{materializationId});
    for (const auto& member : members)
    {
        PersistedRecommendationCampaignMaterializationMember mapped;
        mapped.materializationMemberId = member[0].as<long long>();
        mapped.memberOrdinal = member[1].as<int>();
        mapped.rankingMemberId = member[2].as<long long>();
        mapped.recommendationId = member[3].as<long long>();
        mapped.sourceExperimentId = member[4].as<long long>();
        mapped.rankingPosition = member[5].as<int>();
        mapped.selectedMemberIdentityCanonical = member[6].as<std::string>();
        mapped.selectedMemberIdentityHash = member[7].as<std::string>();
        mapped.conversionProposalId = member[8].as<long long>();
        mapped.proposalIdentityCanonical = member[9].as<std::string>();
        mapped.proposalIdentityHash = member[10].as<std::string>();
        mapped.createdAt = member[11].as<std::string>();
        value.members.push_back(std::move(mapped));
    }
    if (value.identityHash != RecommendationCanonicalHash(
            value.identityCanonical) ||
        static_cast<int>(value.members.size()) != value.selectedMemberCount)
        throw std::runtime_error("invalid_test_materialization");
    return value;
}

std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
FindRecommendationCampaignFollowUpProposalRatification(
    pqxx::transaction_base& transaction, long long ratificationEventId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT ratification_contract_version,"
        "recommendation_campaign_follow_up_proposal_review_event_id,"
        "review_contract_version,review_identity_canonical,"
        "review_identity_hash,"
        "recommendation_campaign_follow_up_proposal_id,"
        "proposal_contract_version,proposal_identity_canonical,"
        "proposal_identity_hash,ratification_identity_canonical,"
        "ratification_identity_hash,created_at::text FROM "
        "experiment_recommendation_campaign_follow_up_ratification_event "
        "WHERE recommendation_campaign_follow_up_ratification_event_id=$1;",
        pqxx::params{ratificationEventId});
    if (rows.empty()) return std::nullopt;
    const auto& row = rows.one_row();
    return PersistedRecommendationCampaignFollowUpProposalRatification(
        ratificationEventId,
        RecommendationCampaignFollowUpProposalRatificationBuilder::
            BuildFixture(ratificationEventId, row[0].as<int>(),
                row[1].as<long long>(), row[2].as<int>(),
                row[3].as<std::string>(), row[4].as<std::string>(),
                row[5].as<long long>(), row[6].as<int>(),
                row[7].as<std::string>(), row[8].as<std::string>(),
                row[9].as<std::string>(), row[10].as<std::string>()),
        row[11].as<std::string>());
}

std::optional<PersistedRecommendationCampaignFollowUpProposal>
FindRecommendationCampaignFollowUpProposal(
    pqxx::transaction_base& transaction, long long proposalId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT materialization_id,materialization_contract_version,"
        "materialization_identity_canonical,materialization_identity_hash,"
        "member_count,proposal_contract_version,"
        "proposal_identity_canonical,proposal_identity_hash,"
        "created_at::text FROM "
        "experiment_recommendation_campaign_follow_up_proposal WHERE "
        "recommendation_campaign_follow_up_proposal_id=$1;",
        pqxx::params{proposalId});
    if (rows.empty()) return std::nullopt;
    const auto& row = rows.one_row();
    return PersistedRecommendationCampaignFollowUpProposal(proposalId, 0,
        RecommendationCampaignFollowUpProposalPersistenceBuilder::BuildFixture(
            proposalId, row[0].as<long long>(), row[1].as<int>(),
            row[2].as<std::string>(), row[3].as<std::string>(),
            row[4].as<int>(), row[6].as<std::string>(),
            row[7].as<std::string>()),
        row[8].as<std::string>());
}

} // namespace EA::ExperimentRecommendation

namespace
{

std::string EnvironmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

std::string ReadFile(const std::string& path)
{
    std::ifstream input{path};
    if (!input) throw std::runtime_error("unable_to_read_" + path);
    return {std::istreambuf_iterator<char>{input},
        std::istreambuf_iterator<char>{}};
}

std::string Lowercase(std::string value)
{
    for (char& character : value)
        character = static_cast<char>(
            std::tolower(static_cast<unsigned char>(character)));
    return value;
}

bool SafeDisposableDatabaseName(const std::string& value)
{
    if (value.empty() || Lowercase(value) == "lstm") return false;
    const std::string lower = Lowercase(value);
    if (lower.find("test") == std::string::npos &&
        lower.find("tmp") == std::string::npos &&
        lower.find("disposable") == std::string::npos)
        return false;
    for (const unsigned char character : value)
        if (!std::isalnum(character) && character != '_' && character != '-')
            return false;
    return true;
}

void SetSearchPath(
    pqxx::transaction_base& transaction, const std::string& schema)
{
    transaction.exec(
        "SET LOCAL search_path TO " + transaction.quote_name(schema) + ";");
}

void SetCampaignCreatorRole(pqxx::transaction_base& transaction)
{
    transaction.exec("SET LOCAL ROLE campaign_operations_campaign_creator;");
}

void SetAuthorizerRole(pqxx::transaction_base& transaction)
{
    transaction.exec("SET LOCAL ROLE campaign_operations_authorizer;");
}

void SetBudgetAdministratorRole(pqxx::transaction_base& transaction)
{
    transaction.exec(
        "SET LOCAL ROLE campaign_operations_budget_administrator;");
}

void SetRequestAcceptorRole(pqxx::transaction_base& transaction)
{
    transaction.exec(
        "SET LOCAL ROLE campaign_operations_request_acceptor;");
}

void ApplyFile(pqxx::connection& connection, const std::string& schema,
    const std::string& path)
{
    pqxx::work transaction{connection};
    SetSearchPath(transaction, schema);
    transaction.exec(ReadFile(path));
    transaction.commit();
}

void CreateBaseSchema(
    pqxx::connection& connection, const std::string& schema)
{
    pqxx::work transaction{connection};
    transaction.exec("CREATE SCHEMA " + transaction.quote_name(schema) + ";");
    SetSearchPath(transaction, schema);
    transaction.exec(R"SQL(
CREATE TABLE experiment_recommendation_campaign_materialization(
    recommendation_campaign_materialization_id bigint PRIMARY KEY,
    recommendation_campaign_approval_id bigint NOT NULL,
    materialization_contract_version integer NOT NULL,
    approval_identity_hash text NOT NULL,
    recommendation_ranking_snapshot_id bigint NOT NULL,
    ranking_snapshot_identity_hash text NOT NULL,
    planning_policy_hash text NOT NULL,
    campaign_plan_identity_hash text NOT NULL,
    campaign_review_identity_hash text NOT NULL,
    materialized_by text NOT NULL,
    materialization_reason_text text NOT NULL,
    selected_member_count integer NOT NULL,
    initially_created_proposal_count integer NOT NULL,
    initially_reused_proposal_count integer NOT NULL,
    materialization_identity_canonical text NOT NULL,
    materialization_identity_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE experiment_recommendation_campaign_materialization_member(
    recommendation_campaign_materialization_member_id bigint PRIMARY KEY,
    recommendation_campaign_materialization_id bigint NOT NULL,
    member_ordinal integer NOT NULL,
    recommendation_ranking_member_id bigint NOT NULL,
    recommendation_id bigint NOT NULL,
    source_experiment_id bigint NOT NULL,
    ranking_position integer NOT NULL,
    selected_member_identity_canonical text NOT NULL,
    selected_member_identity_hash text NOT NULL,
    recommendation_conversion_proposal_id bigint NOT NULL,
    proposal_identity_canonical text NOT NULL,
    proposal_identity_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE experiment_recommendation_campaign_follow_up_proposal(
    recommendation_campaign_follow_up_proposal_id bigint PRIMARY KEY,
    materialization_id bigint NOT NULL,
    materialization_contract_version integer NOT NULL,
    materialization_identity_canonical text NOT NULL,
    materialization_identity_hash text NOT NULL,
    member_count integer NOT NULL,
    proposal_contract_version integer NOT NULL,
    proposal_identity_canonical text NOT NULL,
    proposal_identity_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE experiment_recommendation_campaign_follow_up_proposal_member(
    recommendation_campaign_follow_up_proposal_member_id bigint PRIMARY KEY);
CREATE TABLE experiment_recommendation_campaign_follow_up_proposal_review_event(
    recommendation_campaign_follow_up_proposal_review_event_id bigint
        PRIMARY KEY);
CREATE TABLE experiment_recommendation_campaign_follow_up_ratification_event(
    recommendation_campaign_follow_up_ratification_event_id bigint PRIMARY KEY,
    ratification_contract_version integer NOT NULL,
    ratification_identity_canonical text NOT NULL,
    ratification_identity_hash text NOT NULL,
    recommendation_campaign_follow_up_proposal_review_event_id bigint NOT NULL,
    review_contract_version integer NOT NULL,
    review_identity_canonical text NOT NULL,
    review_identity_hash text NOT NULL,
    recommendation_campaign_follow_up_proposal_id bigint NOT NULL,
    proposal_contract_version integer NOT NULL,
    proposal_identity_canonical text NOT NULL,
    proposal_identity_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now());
)SQL");
    transaction.commit();
}

OperationalCampaign InsertMaterialization(pqxx::connection& connection,
    const std::string& schema, long long materializationId, int memberCount,
    std::string materializationCanonical = {})
{
    if (materializationCanonical.empty())
        materializationCanonical =
            "campaign-operations-test-materialization-" +
            std::to_string(materializationId);
    const std::string materializationHash =
        EA::ExperimentRecommendation::RecommendationCanonicalHash(
            materializationCanonical);
    pqxx::work transaction{connection};
    SetSearchPath(transaction, schema);
    transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_materialization "
        "VALUES($1,$2,1,$3,1,$3,$3,$3,$3,'materializer','reason',$4,"
        "$4,0,$5,$6,transaction_timestamp());",
        pqxx::params{materializationId, materializationId + 1000,
            materializationHash, memberCount, materializationCanonical,
            materializationHash});
    for (int ordinal = 1; ordinal <= memberCount; ++ordinal)
    {
        const std::string selected =
            "selected-" + std::to_string(materializationId) + "-" +
            std::to_string(ordinal);
        const std::string proposal =
            "proposal-" + std::to_string(materializationId) + "-" +
            std::to_string(ordinal);
        transaction.exec(
            "INSERT INTO "
            "experiment_recommendation_campaign_materialization_member "
            "VALUES($1,$2,$3,$4,$5,$6,$3,$7,$8,$9,$10,$11,"
            "transaction_timestamp());",
            pqxx::params{materializationId * 100 + ordinal,
                materializationId, ordinal,
                materializationId * 1000 + ordinal,
                materializationId * 2000 + ordinal,
                materializationId * 3000 + ordinal, selected,
                EA::ExperimentRecommendation::RecommendationCanonicalHash(
                    selected),
                materializationId * 4000 + ordinal, proposal,
                EA::ExperimentRecommendation::RecommendationCanonicalHash(
                    proposal)});
    }
    transaction.commit();
    return BuildOperationalCampaign(materializationId, 1,
        materializationCanonical, materializationHash, memberCount);
}

std::string DeterministicIncompressiblePayload(
    std::size_t size, std::uint64_t seed)
{
    static constexpr char alphabet[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";
    std::string value;
    value.reserve(size);
    std::uint64_t state = seed;
    for (std::size_t index = 0; index < size; ++index)
    {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const std::uint64_t mixed = state * 2685821657736338717ULL;
        value.push_back(alphabet[mixed & 63U]);
    }
    return value;
}

GovernanceProvenanceEvent InsertGovernanceFixture(
    pqxx::connection& connection, const std::string& schema,
    OperationalCampaignId campaignId, const OperationalCampaign& campaign,
    long long ratificationEventId, PrerequisitePolicy prerequisitePolicy,
    std::string payload = {})
{
    const long long reviewEventId = ratificationEventId + 100000;
    const long long proposalId = ratificationEventId + 200000;
    const std::string proposalCanonical = payload.empty()
        ? "campaign-operations-test-proposal-" + std::to_string(proposalId)
        : payload + "-proposal";
    const std::string reviewCanonical = payload.empty()
        ? "campaign-operations-test-review-" + std::to_string(reviewEventId)
        : payload + "-review";
    const std::string ratificationCanonical = payload.empty()
        ? "campaign-operations-test-ratification-" +
            std::to_string(ratificationEventId)
        : payload + "-ratification";
    const std::string proposalHash =
        EA::ExperimentRecommendation::RecommendationCanonicalHash(
            proposalCanonical);
    const std::string reviewHash =
        EA::ExperimentRecommendation::RecommendationCanonicalHash(
            reviewCanonical);
    const std::string ratificationHash =
        EA::ExperimentRecommendation::RecommendationCanonicalHash(
            ratificationCanonical);

    pqxx::work transaction{connection};
    SetSearchPath(transaction, schema);
    transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_follow_up_proposal "
        "VALUES($1,$2,1,$3,$4,$5,1,$6,$7,transaction_timestamp());",
        pqxx::params{proposalId, campaign.materializationId,
            campaign.materializationCanonicalText,
            campaign.materializationIdentityHash, campaign.memberCount,
            proposalCanonical, proposalHash});
    transaction.exec(
        "INSERT INTO "
        "experiment_recommendation_campaign_follow_up_proposal_review_event "
        "VALUES($1);",
        pqxx::params{reviewEventId});
    transaction.exec(
        "INSERT INTO "
        "experiment_recommendation_campaign_follow_up_ratification_event ("
        "recommendation_campaign_follow_up_ratification_event_id,"
        "ratification_contract_version,ratification_identity_canonical,"
        "ratification_identity_hash,"
        "recommendation_campaign_follow_up_proposal_review_event_id,"
        "review_contract_version,review_identity_canonical,"
        "review_identity_hash,"
        "recommendation_campaign_follow_up_proposal_id,"
        "proposal_contract_version,proposal_identity_canonical,"
        "proposal_identity_hash) "
        "VALUES($1,1,$8,$9,$2,1,$3,$4,$5,1,$6,$7);",
        pqxx::params{ratificationEventId, reviewEventId, reviewCanonical,
            reviewHash, proposalId, proposalCanonical, proposalHash,
            ratificationCanonical, ratificationHash});
    transaction.commit();

    return BuildGovernanceProvenanceEvent(campaignId,
        campaign.identity.canonicalText(), ratificationEventId, 1,
        ratificationCanonical, ratificationHash, reviewEventId, 1,
        reviewCanonical, reviewHash, proposalId, 1, proposalCanonical,
        proposalHash, prerequisitePolicy);
}

long long CountRows(pqxx::connection& connection, const std::string& schema,
    const std::string& table)
{
    pqxx::read_transaction transaction{connection};
    SetSearchPath(transaction, schema);
    return transaction.exec("SELECT count(*) FROM " +
               transaction.quote_name(table) + ";")
        .one_row()[0]
        .as<long long>();
}

long long CountRowsForCampaign(pqxx::connection& connection,
    const std::string& schema, const std::string& table,
    OperationalCampaignId campaignId)
{
    pqxx::read_transaction transaction{connection};
    SetSearchPath(transaction, schema);
    return transaction.exec(
        "SELECT count(*) FROM " + transaction.quote_name(table) +
            " WHERE operational_campaign_id=$1;",
        pqxx::params{campaignId.value()})
        .one_row()[0]
        .as<long long>();
}

long long InsertAuthorizationDirect(pqxx::transaction_base& transaction,
    const OperationalAuthorizationEvent& event)
{
    return transaction.exec(
        "INSERT INTO campaign_operations_authorization_event ("
        "operational_campaign_id,campaign_identity_canonical,"
        "previous_event_id,previous_event_identity_canonical,"
        "previous_event_identity_hash,chain_version,event_kind,action_kind,"
        "action_contract_version,scope_kind,scope_contract_version,"
        "prerequisite_policy,governance_provenance_event_id,"
        "provenance_identity_canonical,provenance_identity_hash,"
        "authorization_role,actor_identity,reason,not_before,expires_at,"
        "authorization_contract_version,authorization_identity_canonical,"
        "authorization_identity_hash) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,"
        "$17,$18,$19::timestamptz,$20::timestamptz,$21,$22,$23) "
        "RETURNING authorization_event_id;",
        pqxx::params{event.campaignId.value(), event.campaignCanonicalText,
            event.previousEventId
                ? std::optional<long long>(event.previousEventId->value())
                : std::nullopt,
            event.previousEventCanonicalText,
            event.previousEventIdentityHash, event.chainVersion,
            ToText(event.eventKind), ToText(event.actionKind),
            event.actionContractVersion, ToText(event.scopeKind),
            event.scopeContractVersion, ToText(event.prerequisitePolicy),
            event.provenanceEventId
                ? std::optional<long long>(event.provenanceEventId->value())
                : std::nullopt,
            event.provenanceCanonicalText, event.provenanceIdentityHash,
            event.authorizationRole, event.actor.value(), event.reason.value(),
            event.notBefore.value(),
            event.expiresAt
                ? std::optional<std::string>(event.expiresAt->value())
                : std::nullopt,
            event.identity.contractVersion(), event.identity.canonicalText(),
            event.identity.hash()})
        .one_row()[0]
        .as<long long>();
}

PersistedOperationalCampaign PersistCampaignFixture(
    pqxx::connection& connection, const std::string& schema,
    const OperationalCampaign& campaign)
{
    pqxx::work transaction{connection};
    SetSearchPath(transaction, schema);
    SetCampaignCreatorRole(transaction);
    auto result = PersistOperationalCampaign(transaction, campaign,
        ActorIdentity("creator@example.test"),
        Reason("Create focused Campaign Operations repository fixture."));
    assert(result.outcome == PersistOutcome::recorded);
    PersistedOperationalCampaign persisted(result.persisted.campaignId,
        result.persisted.campaign, result.persisted.createdAt);
    transaction.commit();
    return persisted;
}

void TestProvenanceAndPrerequisitePolicy(pqxx::connection& owner,
    const std::string& connectionString, const std::string& schema)
{
    (void)connectionString;
    const OperationalCampaign campaign =
        InsertMaterialization(owner, schema, 44, 2);
    const PersistedOperationalCampaign persistedCampaign =
        PersistCampaignFixture(owner, schema, campaign);
    const GovernanceProvenanceEvent matching =
        InsertGovernanceFixture(owner, schema, persistedCampaign.campaignId,
            campaign, 701,
            PrerequisitePolicy::
                phase4dMaterializationPlusExactPhase6dRatificationV1);

    std::optional<PersistedGovernanceProvenanceEvent> persistedProvenance;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        const auto recorded = PersistGovernanceProvenanceEvent(transaction,
            matching, ActorIdentity("authorizer@example.test"),
            Reason("Record matching exact Phase 6D provenance."));
        assert(recorded.outcome == PersistOutcome::recorded);
        assert(recorded.persisted.event == matching);
        persistedProvenance.emplace(recorded.persisted.provenanceEventId,
            recorded.persisted.event, recorded.persisted.createdAt);
        transaction.exec("RESET ROLE;");
        assert(transaction.exec(
            "SELECT count(*) FROM "
            "campaign_operations_governance_provenance_event WHERE "
            "governance_provenance_event_id=$1;",
            pqxx::params{recorded.persisted.provenanceEventId.value()})
                   .one_row()[0]
                   .as<int>() == 1);
        assert(transaction.exec(
            "SELECT count(*) FROM "
            "campaign_operations_audit_reference_event WHERE "
            "governance_provenance_event_id=$1;",
            pqxx::params{recorded.persisted.provenanceEventId.value()})
                   .one_row()[0]
                   .as<int>() == 1);
        transaction.commit();
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_governance_provenance_event",
               persistedCampaign.campaignId) == 1);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_audit_reference_event",
               persistedCampaign.campaignId) == 2);

    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        const auto loaded = FindGovernanceProvenanceEvent(
            transaction, persistedProvenance->provenanceEventId);
        assert(loaded);
        assert(loaded->event == matching);
        const auto replay = PersistGovernanceProvenanceEvent(transaction,
            matching, ActorIdentity("retry.authorizer@example.test"),
            Reason("Exact provenance retry."));
        assert(replay.outcome == PersistOutcome::existingIdentical);
        assert(replay.persisted.provenanceEventId ==
            persistedProvenance->provenanceEventId);
        transaction.commit();
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_audit_reference_event",
               persistedCampaign.campaignId) == 2);

    const GovernanceProvenanceEvent conflicting =
        BuildGovernanceProvenanceEvent(matching.campaignId,
            matching.campaignCanonicalText, matching.ratificationEventId,
            matching.ratificationContractVersion,
            matching.ratificationCanonicalText,
            matching.ratificationIdentityHash, matching.reviewEventId,
            matching.reviewContractVersion, matching.reviewCanonicalText,
            matching.reviewIdentityHash, matching.proposalId,
            matching.proposalContractVersion, matching.proposalCanonicalText,
            matching.proposalIdentityHash,
            PrerequisitePolicy::phase4dMaterializationOnlyV1);
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        bool conflicted = false;
        try
        {
            (void)PersistGovernanceProvenanceEvent(transaction, conflicting,
                ActorIdentity("authorizer@example.test"),
                Reason("Differing canonical must conflict."));
        }
        catch (const Error& error)
        {
            conflicted = error.code() == ErrorCode::persistenceConflict &&
                std::string(error.what()) ==
                    "campaign_operations_governance_provenance_conflict";
        }
        assert(conflicted);
        transaction.abort();
    }

    const GovernanceProvenanceEvent rollback =
        InsertGovernanceFixture(owner, schema, persistedCampaign.campaignId,
            campaign, 702,
            PrerequisitePolicy::
                phase4dMaterializationPlusExactPhase6dRatificationV1);
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        const auto recorded = PersistGovernanceProvenanceEvent(transaction,
            rollback, ActorIdentity("authorizer@example.test"),
            Reason("Rollback provenance and audit together."));
        assert(recorded.outcome == PersistOutcome::recorded);
        transaction.abort();
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_governance_provenance_event",
               persistedCampaign.campaignId) == 1);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_audit_reference_event",
               persistedCampaign.campaignId) == 2);

    const OperationalAuthorizationEvent matchingAuthorization =
        BuildOperationalAuthorizationEvent(persistedCampaign.campaignId,
            campaign.identity.canonicalText(), std::nullopt, std::nullopt,
            std::nullopt, 1, AuthorizationEventKind::granted,
            campaign.actionKind, campaign.actionContractVersion,
            campaign.scopeKind, campaign.scopeContractVersion,
            matching.prerequisitePolicy,
            persistedProvenance->provenanceEventId,
            persistedProvenance->event.identity.canonicalText(),
            persistedProvenance->event.identity.hash(),
            kCampaignOperationsAuthorizationRole,
            ActorIdentity("authorizer@example.test"),
            Reason("Authorize with matching exact Phase 6D provenance."),
            UtcTimestamp("2026-07-24T12:00:00.000000Z"), std::nullopt);
    std::optional<AuthorizationEventId> matchingAuthorizationId;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        const auto recorded = PersistOperationalAuthorizationEvent(
            transaction, matchingAuthorization);
        assert(recorded.outcome == PersistOutcome::recorded);
        assert(recorded.persisted.event == matchingAuthorization);
        matchingAuthorizationId.emplace(
            recorded.persisted.authorizationEventId);
        transaction.exec("RESET ROLE;");
        assert(transaction.exec(
            "SELECT count(*) FROM "
            "campaign_operations_audit_reference_event WHERE "
            "authorization_event_id=$1 AND "
            "governance_provenance_event_id IS NULL;",
            pqxx::params{recorded.persisted.authorizationEventId.value()})
                   .one_row()[0]
                   .as<int>() == 1);
        transaction.commit();
    }
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        const auto loaded = FindOperationalAuthorizationEvent(
            transaction, *matchingAuthorizationId);
        assert(loaded);
        assert(loaded->event == matchingAuthorization);
        transaction.commit();
    }

    const GovernanceProvenanceEvent noPrerequisite =
        InsertGovernanceFixture(owner, schema, persistedCampaign.campaignId,
            campaign, 703,
            PrerequisitePolicy::phase4dMaterializationOnlyV1);
    std::optional<PersistedGovernanceProvenanceEvent> persistedNoPrerequisite;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        const auto recorded = PersistGovernanceProvenanceEvent(transaction,
            noPrerequisite, ActorIdentity("authorizer@example.test"),
            Reason("Record no-prerequisite provenance fixture."));
        persistedNoPrerequisite.emplace(recorded.persisted.provenanceEventId,
            recorded.persisted.event, recorded.persisted.createdAt);
        transaction.commit();
    }

    const auto mismatchedAuthorization =
        [&](const PersistedGovernanceProvenanceEvent& provenance,
            PrerequisitePolicy policy, const std::string& actor)
    {
        return BuildOperationalAuthorizationEvent(
            persistedCampaign.campaignId, campaign.identity.canonicalText(),
            std::nullopt, std::nullopt, std::nullopt, 1,
            AuthorizationEventKind::granted,
            OperationalActionKind::adoptExistingPendingAndControl,
            kCampaignOperationsActionContractVersion, campaign.scopeKind,
            campaign.scopeContractVersion, policy,
            provenance.provenanceEventId,
            provenance.event.identity.canonicalText(),
            provenance.event.identity.hash(),
            kCampaignOperationsAuthorizationRole, ActorIdentity(actor),
            Reason("Prerequisite-policy mismatch must be rejected."),
            UtcTimestamp("2026-07-24T13:00:00.000000Z"), std::nullopt);
    };
    const OperationalAuthorizationEvent repositoryMismatchRatification =
        mismatchedAuthorization(*persistedProvenance,
            PrerequisitePolicy::phase4dMaterializationOnlyV1,
            "mismatch.none@example.test");
    const OperationalAuthorizationEvent repositoryMismatchNoPrerequisite =
        mismatchedAuthorization(*persistedNoPrerequisite,
            PrerequisitePolicy::
                phase4dMaterializationPlusExactPhase6dRatificationV1,
            "mismatch.ratification@example.test");
    for (const auto* mismatch : {&repositoryMismatchRatification,
             &repositoryMismatchNoPrerequisite})
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        bool rejected = false;
        try
        {
            (void)PersistOperationalAuthorizationEvent(
                transaction, *mismatch);
        }
        catch (const Error& error)
        {
            rejected = error.code() == ErrorCode::persistenceConflict &&
                std::string(error.what()) ==
                    "campaign_operations_authorization_provenance_conflict";
        }
        assert(rejected);
        transaction.abort();
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_authorization_event",
               persistedCampaign.campaignId) == 1);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_audit_reference_event",
               persistedCampaign.campaignId) == 4);

    const OperationalCampaign triggerCampaign =
        InsertMaterialization(owner, schema, 45, 2);
    const PersistedOperationalCampaign persistedTriggerCampaign =
        PersistCampaignFixture(owner, schema, triggerCampaign);
    const GovernanceProvenanceEvent triggerRequiresRatification =
        InsertGovernanceFixture(owner, schema,
            persistedTriggerCampaign.campaignId, triggerCampaign, 711,
            PrerequisitePolicy::
                phase4dMaterializationPlusExactPhase6dRatificationV1);
    const GovernanceProvenanceEvent triggerNoPrerequisite =
        InsertGovernanceFixture(owner, schema,
            persistedTriggerCampaign.campaignId, triggerCampaign, 712,
            PrerequisitePolicy::phase4dMaterializationOnlyV1);
    std::optional<PersistedGovernanceProvenanceEvent>
        persistedTriggerRequiresRatification;
    std::optional<PersistedGovernanceProvenanceEvent>
        persistedTriggerNoPrerequisite;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        const auto first = PersistGovernanceProvenanceEvent(transaction,
            triggerRequiresRatification,
            ActorIdentity("authorizer@example.test"),
            Reason("Direct-trigger ratification provenance fixture."));
        const auto second = PersistGovernanceProvenanceEvent(transaction,
            triggerNoPrerequisite, ActorIdentity("authorizer@example.test"),
            Reason("Direct-trigger no-prerequisite provenance fixture."));
        persistedTriggerRequiresRatification.emplace(
            first.persisted.provenanceEventId, first.persisted.event,
            first.persisted.createdAt);
        persistedTriggerNoPrerequisite.emplace(
            second.persisted.provenanceEventId, second.persisted.event,
            second.persisted.createdAt);
        transaction.commit();
    }
    const auto directMismatch =
        [&](const PersistedGovernanceProvenanceEvent& provenance,
            PrerequisitePolicy policy, const std::string& actor)
    {
        return BuildOperationalAuthorizationEvent(
            persistedTriggerCampaign.campaignId,
            triggerCampaign.identity.canonicalText(), std::nullopt,
            std::nullopt, std::nullopt, 1, AuthorizationEventKind::granted,
            triggerCampaign.actionKind,
            triggerCampaign.actionContractVersion,
            triggerCampaign.scopeKind, triggerCampaign.scopeContractVersion,
            policy, provenance.provenanceEventId,
            provenance.event.identity.canonicalText(),
            provenance.event.identity.hash(),
            kCampaignOperationsAuthorizationRole, ActorIdentity(actor),
            Reason("Database trigger prerequisite mismatch."),
            UtcTimestamp("2026-07-24T14:00:00.000000Z"), std::nullopt);
    };
    const OperationalAuthorizationEvent directMismatchRatification =
        directMismatch(*persistedTriggerRequiresRatification,
            PrerequisitePolicy::phase4dMaterializationOnlyV1,
            "direct.none@example.test");
    const OperationalAuthorizationEvent directMismatchNoPrerequisite =
        directMismatch(*persistedTriggerNoPrerequisite,
            PrerequisitePolicy::
                phase4dMaterializationPlusExactPhase6dRatificationV1,
            "direct.ratification@example.test");
    for (const auto* mismatch :
        {&directMismatchRatification, &directMismatchNoPrerequisite})
    {
        bool rejected = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            (void)InsertAuthorizationDirect(transaction, *mismatch);
            transaction.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            rejected = error.sqlstate() == "23514";
        }
        assert(rejected);
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_authorization_event",
                   persistedTriggerCampaign.campaignId) == 0);
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_audit_reference_event",
                   persistedTriggerCampaign.campaignId) == 3);
    }

    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_owner;");
        transaction.exec(
            "ALTER TABLE campaign_operations_authorization_event "
            "DISABLE TRIGGER campaign_operations_authorization_chain_trigger;");
        transaction.exec("RESET ROLE;");
        const AuthorizationEventId malformedId(
            InsertAuthorizationDirect(
                transaction, directMismatchRatification));
        bool rejected = false;
        try
        {
            (void)FindOperationalAuthorizationEvent(
                transaction, malformedId);
        }
        catch (const Error& error)
        {
            rejected = error.code() == ErrorCode::persistenceCorruption &&
                std::string(error.what()) ==
                    "campaign_operations_authorization_provenance_mismatch";
        }
        assert(rejected);
        transaction.abort();
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_authorization_event",
               persistedTriggerCampaign.campaignId) == 0);
}

template <pqxx::isolation_level Isolation>
void TestConcurrentAuthorizationSuccessors(pqxx::connection& owner,
    const std::string& connectionString, const std::string& schema,
    long long materializationId)
{
    const OperationalCampaign campaign =
        InsertMaterialization(owner, schema, materializationId, 2);
    const PersistedOperationalCampaign persistedCampaign =
        PersistCampaignFixture(owner, schema, campaign);
    const OperationalAuthorizationEvent grant =
        BuildOperationalAuthorizationEvent(persistedCampaign.campaignId,
            campaign.identity.canonicalText(), std::nullopt, std::nullopt,
            std::nullopt, 1, AuthorizationEventKind::granted,
            campaign.actionKind, campaign.actionContractVersion,
            campaign.scopeKind, campaign.scopeContractVersion,
            PrerequisitePolicy::phase4dMaterializationOnlyV1, std::nullopt,
            std::nullopt, std::nullopt, kCampaignOperationsAuthorizationRole,
            ActorIdentity("authorizer@example.test"),
            Reason("Concurrent successor fixture grant."),
            UtcTimestamp("2026-07-24T15:00:00.000000Z"), std::nullopt);
    std::optional<PersistedOperationalAuthorizationEvent> persistedGrant;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        const auto recorded =
            PersistOperationalAuthorizationEvent(transaction, grant);
        persistedGrant.emplace(recorded.persisted.authorizationEventId,
            recorded.persisted.event, recorded.persisted.createdAt);
        transaction.commit();
    }
    const auto successor =
        [&](AuthorizationEventKind kind, const std::string& actor,
            const std::string& reason)
    {
        return BuildOperationalAuthorizationEvent(
            persistedCampaign.campaignId, campaign.identity.canonicalText(),
            persistedGrant->authorizationEventId,
            persistedGrant->event.identity.canonicalText(),
            persistedGrant->event.identity.hash(), 2, kind,
            campaign.actionKind, campaign.actionContractVersion,
            campaign.scopeKind, campaign.scopeContractVersion,
            PrerequisitePolicy::phase4dMaterializationOnlyV1, std::nullopt,
            std::nullopt, std::nullopt, kCampaignOperationsAuthorizationRole,
            ActorIdentity(actor), Reason(reason),
            UtcTimestamp("2026-07-24T16:00:00.000000Z"), std::nullopt);
    };
    const OperationalAuthorizationEvent firstSuccessor = successor(
        AuthorizationEventKind::granted, "successor.one@example.test",
        "Concurrent successor one.");
    const OperationalAuthorizationEvent secondSuccessor = successor(
        AuthorizationEventKind::revoked, "successor.two@example.test",
        "Concurrent successor two.");
    struct Attempt
    {
        bool recorded = false;
        bool deterministicConflict = false;
        std::exception_ptr unexpected;
    };
    Attempt first;
    Attempt second;
    std::atomic<bool> start{false};
    const auto persist =
        [&](const OperationalAuthorizationEvent& event, Attempt& attempt)
    {
        try
        {
            pqxx::connection connection{connectionString};
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            pqxx::transaction<Isolation> transaction{connection};
            SetSearchPath(transaction, schema);
            SetAuthorizerRole(transaction);
            try
            {
                const auto result =
                    PersistOperationalAuthorizationEvent(transaction, event);
                attempt.recorded = result.outcome == PersistOutcome::recorded;
                transaction.commit();
            }
            catch (const Error& error)
            {
                attempt.deterministicConflict =
                    error.code() == ErrorCode::persistenceConflict &&
                    std::string(error.what()) ==
                        "campaign_operations_authorization_conflict";
                transaction.abort();
            }
        }
        catch (...)
        {
            attempt.unexpected = std::current_exception();
        }
    };
    std::thread firstThread(
        persist, std::cref(firstSuccessor), std::ref(first));
    std::thread secondThread(
        persist, std::cref(secondSuccessor), std::ref(second));
    start.store(true, std::memory_order_release);
    firstThread.join();
    secondThread.join();
    if (first.unexpected) std::rethrow_exception(first.unexpected);
    if (second.unexpected) std::rethrow_exception(second.unexpected);
    assert(first.recorded != second.recorded);
    assert(first.deterministicConflict != second.deterministicConflict);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_authorization_event",
               persistedCampaign.campaignId) == 2);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_audit_reference_event",
               persistedCampaign.campaignId) == 3);
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        assert(transaction.exec(
            "SELECT count(*) FROM campaign_operations_authorization_event "
            "head WHERE operational_campaign_id=$1 AND NOT EXISTS ("
            "SELECT 1 FROM campaign_operations_authorization_event successor "
            "WHERE successor.previous_event_id=head.authorization_event_id);",
            pqxx::params{persistedCampaign.campaignId.value()})
                   .one_row()[0]
                   .as<int>() == 1);
        assert(transaction.exec(
            "SELECT count(*) FROM campaign_operations_audit_reference_event "
            "WHERE operational_campaign_id=$1 AND "
            "cause_kind='authorization_recorded';",
            pqxx::params{persistedCampaign.campaignId.value()})
                   .one_row()[0]
                   .as<int>() == 2);
    }
}

void TestOversizedCanonicals(pqxx::connection& owner,
    const std::string& schema)
{
    constexpr std::size_t oversizedBytes = 16U * 1024U;
    const OperationalCampaign campaign = InsertMaterialization(owner, schema,
        47, 2, DeterministicIncompressiblePayload(oversizedBytes, 0x4701U));
    assert(campaign.identity.canonicalText().size() > 8192U);
    assert(campaign.identity.canonicalText().size() <
        kCampaignOperationsCanonicalMaximumBytes);
    const PersistedOperationalCampaign persistedCampaign =
        PersistCampaignFixture(owner, schema, campaign);
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetCampaignCreatorRole(transaction);
        const auto loaded = FindOperationalCampaign(
            transaction, persistedCampaign.campaignId);
        assert(loaded);
        assert(loaded->campaign.materializationCanonicalText ==
            campaign.materializationCanonicalText);
        assert(loaded->campaign.identity.canonicalText() ==
            campaign.identity.canonicalText());
        const auto replay = PersistOperationalCampaign(transaction, campaign,
            ActorIdentity("creator@example.test"),
            Reason("Replay oversized operational campaign."));
        assert(replay.outcome == PersistOutcome::existingIdentical);
        transaction.commit();
    }
    const OperationalCampaign conflictingCampaign = BuildOperationalCampaign(
        campaign.materializationId, campaign.materializationContractVersion,
        campaign.materializationCanonicalText,
        campaign.materializationIdentityHash, campaign.memberCount + 1);
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetCampaignCreatorRole(transaction);
        bool conflicted = false;
        try
        {
            (void)PersistOperationalCampaign(transaction, conflictingCampaign,
                ActorIdentity("creator@example.test"),
                Reason("Oversized campaign conflict."));
        }
        catch (const Error& error)
        {
            conflicted = error.code() == ErrorCode::persistenceConflict;
        }
        assert(conflicted);
        transaction.abort();
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_campaign",
               persistedCampaign.campaignId) == 1);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_audit_reference_event",
               persistedCampaign.campaignId) == 1);

    const OperationalCampaign provenanceCampaign =
        InsertMaterialization(owner, schema, 48, 2);
    const PersistedOperationalCampaign persistedProvenanceCampaign =
        PersistCampaignFixture(owner, schema, provenanceCampaign);
    const GovernanceProvenanceEvent provenance =
        InsertGovernanceFixture(owner, schema,
            persistedProvenanceCampaign.campaignId, provenanceCampaign, 721,
            PrerequisitePolicy::
                phase4dMaterializationPlusExactPhase6dRatificationV1,
            DeterministicIncompressiblePayload(
                oversizedBytes, 0x4702U));
    assert(provenance.identity.canonicalText().size() > 8192U);
    assert(provenance.identity.canonicalText().size() <
        kCampaignOperationsCanonicalMaximumBytes);
    std::optional<PersistedGovernanceProvenanceEvent> persistedProvenance;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        const auto recorded = PersistGovernanceProvenanceEvent(transaction,
            provenance, ActorIdentity("authorizer@example.test"),
            Reason("Persist oversized governance provenance."));
        assert(recorded.outcome == PersistOutcome::recorded);
        assert(recorded.persisted.event == provenance);
        persistedProvenance.emplace(recorded.persisted.provenanceEventId,
            recorded.persisted.event, recorded.persisted.createdAt);
        transaction.commit();
    }
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        const auto loaded = FindGovernanceProvenanceEvent(
            transaction, persistedProvenance->provenanceEventId);
        assert(loaded);
        assert(loaded->event == provenance);
        const auto replay = PersistGovernanceProvenanceEvent(transaction,
            provenance, ActorIdentity("authorizer@example.test"),
            Reason("Replay oversized governance provenance."));
        assert(replay.outcome == PersistOutcome::existingIdentical);
        transaction.commit();
    }
    const GovernanceProvenanceEvent conflictingProvenance =
        BuildGovernanceProvenanceEvent(provenance.campaignId,
            provenance.campaignCanonicalText, provenance.ratificationEventId,
            provenance.ratificationContractVersion,
            provenance.ratificationCanonicalText,
            provenance.ratificationIdentityHash, provenance.reviewEventId,
            provenance.reviewContractVersion, provenance.reviewCanonicalText,
            provenance.reviewIdentityHash, provenance.proposalId,
            provenance.proposalContractVersion,
            provenance.proposalCanonicalText, provenance.proposalIdentityHash,
            PrerequisitePolicy::phase4dMaterializationOnlyV1);
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        bool conflicted = false;
        try
        {
            (void)PersistGovernanceProvenanceEvent(transaction,
                conflictingProvenance,
                ActorIdentity("authorizer@example.test"),
                Reason("Oversized provenance conflict."));
        }
        catch (const Error& error)
        {
            conflicted = error.code() == ErrorCode::persistenceConflict;
        }
        assert(conflicted);
        transaction.abort();
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_governance_provenance_event",
               persistedProvenanceCampaign.campaignId) == 1);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_audit_reference_event",
               persistedProvenanceCampaign.campaignId) == 2);

    const OperationalCampaign authorizationCampaign =
        InsertMaterialization(owner, schema, 49, 2,
            DeterministicIncompressiblePayload(
                oversizedBytes, 0x4703U));
    const PersistedOperationalCampaign persistedAuthorizationCampaign =
        PersistCampaignFixture(owner, schema, authorizationCampaign);
    const std::string oversizedAuthorizationReason =
        DeterministicIncompressiblePayload(
            kCampaignOperationsReasonMaximumBytes, 0x4704U);
    const OperationalAuthorizationEvent authorization =
        BuildOperationalAuthorizationEvent(
            persistedAuthorizationCampaign.campaignId,
            authorizationCampaign.identity.canonicalText(), std::nullopt,
            std::nullopt,
            std::nullopt, 1, AuthorizationEventKind::granted,
            authorizationCampaign.actionKind,
            authorizationCampaign.actionContractVersion,
            authorizationCampaign.scopeKind,
            authorizationCampaign.scopeContractVersion,
            PrerequisitePolicy::phase4dMaterializationOnlyV1,
            std::nullopt, std::nullopt, std::nullopt,
            kCampaignOperationsAuthorizationRole,
            ActorIdentity("authorizer@example.test"),
            Reason(oversizedAuthorizationReason),
            UtcTimestamp("2026-07-24T17:00:00.000000Z"), std::nullopt);
    assert(authorization.identity.canonicalText().size() > 8192U);
    assert(authorization.identity.canonicalText().size() <
        kCampaignOperationsCanonicalMaximumBytes);
    std::optional<PersistedOperationalAuthorizationEvent>
        persistedAuthorization;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        const auto recorded =
            PersistOperationalAuthorizationEvent(transaction, authorization);
        assert(recorded.outcome == PersistOutcome::recorded);
        assert(recorded.persisted.event == authorization);
        persistedAuthorization.emplace(
            recorded.persisted.authorizationEventId,
            recorded.persisted.event, recorded.persisted.createdAt);
        transaction.commit();
    }
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        const auto loaded = FindOperationalAuthorizationEvent(
            transaction, persistedAuthorization->authorizationEventId);
        assert(loaded);
        assert(loaded->event == authorization);
        const auto replay =
            PersistOperationalAuthorizationEvent(transaction, authorization);
        assert(replay.outcome == PersistOutcome::existingIdentical);
        transaction.commit();
    }
    const OperationalAuthorizationEvent conflictingAuthorization =
        BuildOperationalAuthorizationEvent(
            persistedAuthorizationCampaign.campaignId,
            authorizationCampaign.identity.canonicalText(), std::nullopt,
            std::nullopt,
            std::nullopt, 1, AuthorizationEventKind::granted,
            authorizationCampaign.actionKind,
            authorizationCampaign.actionContractVersion,
            authorizationCampaign.scopeKind,
            authorizationCampaign.scopeContractVersion,
            PrerequisitePolicy::phase4dMaterializationOnlyV1,
            std::nullopt, std::nullopt, std::nullopt,
            kCampaignOperationsAuthorizationRole,
            ActorIdentity("different.authorizer@example.test"),
            Reason(oversizedAuthorizationReason),
            UtcTimestamp("2026-07-24T17:00:00.000000Z"), std::nullopt);
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        bool conflicted = false;
        try
        {
            (void)PersistOperationalAuthorizationEvent(
                transaction, conflictingAuthorization);
        }
        catch (const Error& error)
        {
            conflicted = error.code() == ErrorCode::persistenceConflict;
        }
        assert(conflicted);
        transaction.abort();
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_authorization_event",
               persistedAuthorizationCampaign.campaignId) == 1);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_audit_reference_event",
               persistedAuthorizationCampaign.campaignId) == 2);
}

PersistedOperationalAuthorizationEvent PersistActiveAuthorizationFixture(
    pqxx::connection& owner, const std::string& schema,
    const PersistedOperationalCampaign& campaign)
{
    const auto grant = BuildOperationalAuthorizationEvent(
        campaign.campaignId, campaign.campaign.identity.canonicalText(),
        std::nullopt, std::nullopt, std::nullopt, 1,
        AuthorizationEventKind::granted, campaign.campaign.actionKind,
        campaign.campaign.actionContractVersion,
        campaign.campaign.scopeKind,
        campaign.campaign.scopeContractVersion,
        PrerequisitePolicy::phase4dMaterializationOnlyV1, std::nullopt,
        std::nullopt, std::nullopt, kCampaignOperationsAuthorizationRole,
        ActorIdentity("phase2.authorizer@example.test"),
        Reason("Authorize durable full-materialization request acceptance."),
        UtcTimestamp("2020-01-01T00:00:00.000000Z"), std::nullopt);
    pqxx::work transaction{owner};
    SetSearchPath(transaction, schema);
    SetAuthorizerRole(transaction);
    auto result = PersistOperationalAuthorizationEvent(transaction, grant);
    transaction.commit();
    return std::move(result.persisted);
}

struct Phase2AcceptanceFixture final
{
    PersistedOperationalCampaign campaign;
    PersistedOperationalAuthorizationEvent authorization;
    PersistedBudgetLedgerEntry budget;
};

Phase2AcceptanceFixture CreatePhase2AcceptanceFixture(
    pqxx::connection& owner, const std::string& budgetConnectionString,
    const std::string& schema, long long materializationId, int memberCount)
{
    const OperationalCampaign campaign =
        InsertMaterialization(owner, schema, materializationId, memberCount);
    const auto persistedCampaign =
        PersistCampaignFixture(owner, schema, campaign);
    const auto authorization =
        PersistActiveAuthorizationFixture(owner, schema, persistedCampaign);
    BudgetAdministrationRequest grant;
    grant.campaignId = persistedCampaign.campaignId.value();
    grant.expectedLedgerVersion = 0;
    grant.kind = BudgetLedgerEntryKind::grant;
    grant.value = memberCount;
    grant.actorIdentity = "direct.budget.admin@example.test";
    grant.reason = "Fund direct capability integrity fixture.";
    pqxx::connection budgetConnection{budgetConnectionString};
    auto budget = AdministerCampaignBudget(budgetConnection, grant);
    return {persistedCampaign, authorization, std::move(budget.persisted)};
}

long long InsertDirectReservation(pqxx::transaction_base& transaction,
    const Phase2AcceptanceFixture& fixture,
    const std::optional<UtcTimestamp>& expiresAt)
{
    const LogicalOperation operation = BuildLogicalOperation(
        fixture.campaign.campaignId, fixture.campaign.campaign);
    const Reservation reservation = BuildReservation(operation,
        fixture.authorization.authorizationEventId,
        fixture.authorization.event.identity.canonicalText(),
        fixture.authorization.event.identity.hash(),
        fixture.budget.budgetLedgerEntryId, fixture.budget.entry.ledgerVersion,
        fixture.budget.entry.identity.canonicalText(),
        fixture.budget.entry.identity.hash(),
        fixture.campaign.campaign.memberCount,
        fixture.campaign.campaign.memberCount,
        BudgetUnit::materializedMemberDispatch, expiresAt);
    return transaction.exec(
        "INSERT INTO campaign_operations_reservation ("
        "operational_campaign_id,campaign_identity_canonical,"
        "logical_operation_contract_version,logical_operation_canonical,"
        "logical_operation_hash,authorization_event_id,"
        "authorization_identity_canonical,authorization_identity_hash,"
        "budget_ledger_entry_id,budget_ledger_version,"
        "budget_identity_canonical,budget_identity_hash,action_kind,"
        "action_contract_version,recommendation_campaign_materialization_id,"
        "materialization_contract_version,materialization_identity_canonical,"
        "materialization_identity_hash,scope_kind,scope_contract_version,"
        "materialization_member_count,amount,budget_unit,expires_at,"
        "reservation_contract_version,reservation_identity_canonical,"
        "reservation_identity_hash,reservation_state,state_version) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,"
        "$17,$18,$19,$20,$21,$22,$23,$24::timestamptz,$25,$26,$27,"
        "'held',1) RETURNING reservation_id;",
        pqxx::params{fixture.campaign.campaignId.value(),
            fixture.campaign.campaign.identity.canonicalText(),
            operation.identity.contractVersion(),
            operation.identity.canonicalText(), operation.identity.hash(),
            fixture.authorization.authorizationEventId.value(),
            fixture.authorization.event.identity.canonicalText(),
            fixture.authorization.event.identity.hash(),
            fixture.budget.budgetLedgerEntryId.value(),
            fixture.budget.entry.ledgerVersion,
            fixture.budget.entry.identity.canonicalText(),
            fixture.budget.entry.identity.hash(), ToText(operation.actionKind),
            operation.actionContractVersion, operation.materializationId,
            operation.materializationContractVersion,
            operation.materializationCanonicalText,
            operation.materializationIdentityHash,
            ToText(operation.scopeKind), operation.scopeContractVersion,
            fixture.campaign.campaign.memberCount,
            fixture.campaign.campaign.memberCount,
            ToText(BudgetUnit::materializedMemberDispatch),
            expiresAt
                ? std::optional<std::string>(expiresAt->value())
                : std::nullopt,
            reservation.identity.contractVersion(),
            reservation.identity.canonicalText(), reservation.identity.hash()})
        .one_row()[0]
        .as<long long>();
}

long long InsertDirectRequest(pqxx::transaction_base& transaction,
    const Phase2AcceptanceFixture& fixture, long long reservationId,
    PrerequisitePolicy prerequisitePolicy,
    const std::optional<std::string>& provenanceCanonical,
    const std::optional<std::string>& provenanceHash)
{
    const LogicalOperation operation = BuildLogicalOperation(
        fixture.campaign.campaignId, fixture.campaign.campaign);
    const auto reservation = FindReservation(
        transaction, ReservationId(reservationId));
    assert(reservation);
    const OperationalRequest request = BuildOperationalRequest(operation,
        fixture.authorization.authorizationEventId,
        fixture.authorization.event.identity.canonicalText(),
        fixture.authorization.event.identity.hash(),
        reservation->reservationId,
        reservation->reservation.identity.canonicalText(),
        reservation->reservation.identity.hash(),
        fixture.campaign.campaign.memberCount,
        fixture.campaign.campaign.materializationIdentityHash,
        ActorIdentity("direct.requester@example.test"),
        Reason("Attempt malformed direct request evidence."),
        prerequisitePolicy, provenanceCanonical, provenanceHash);
    return transaction.exec(
        "INSERT INTO campaign_operations_operational_request ("
        "operational_campaign_id,campaign_identity_canonical,"
        "logical_operation_contract_version,logical_operation_canonical,"
        "logical_operation_hash,authorization_event_id,"
        "authorization_identity_canonical,authorization_identity_hash,"
        "reservation_id,reservation_identity_canonical,"
        "reservation_identity_hash,action_kind,action_contract_version,"
        "recommendation_campaign_materialization_id,"
        "materialization_contract_version,materialization_identity_canonical,"
        "materialization_identity_hash,ordered_scope_digest,"
        "materialization_member_count,accepting_actor_identity,reason,"
        "prerequisite_policy,provenance_identity_canonical,"
        "provenance_identity_hash,request_contract_version,"
        "request_identity_canonical,request_identity_hash,request_state,"
        "state_version) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,"
        "$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,"
        "'ready',1) RETURNING operational_request_id;",
        pqxx::params{fixture.campaign.campaignId.value(),
            fixture.campaign.campaign.identity.canonicalText(),
            operation.identity.contractVersion(),
            operation.identity.canonicalText(), operation.identity.hash(),
            fixture.authorization.authorizationEventId.value(),
            fixture.authorization.event.identity.canonicalText(),
            fixture.authorization.event.identity.hash(), reservationId,
            reservation->reservation.identity.canonicalText(),
            reservation->reservation.identity.hash(),
            ToText(operation.actionKind), operation.actionContractVersion,
            operation.materializationId,
            operation.materializationContractVersion,
            operation.materializationCanonicalText,
            operation.materializationIdentityHash,
            fixture.campaign.campaign.materializationIdentityHash,
            fixture.campaign.campaign.memberCount,
            request.acceptingActor.value(), request.reason.value(),
            ToText(prerequisitePolicy), provenanceCanonical, provenanceHash,
            request.identity.contractVersion(),
            request.identity.canonicalText(), request.identity.hash()})
        .one_row()[0]
        .as<long long>();
}

void InsertDirectBudgetGrantWithAudit(
    pqxx::transaction_base& transaction,
    const PersistedOperationalCampaign& campaign,
    const std::optional<std::string>& auditActor,
    const std::optional<std::string>& auditReason)
{
    const BudgetLedgerEntry entry = BuildBudgetLedgerEntry(
        campaign.campaignId, campaign.campaign.identity.canonicalText(),
        std::nullopt, std::nullopt, std::nullopt, 1,
        BudgetLedgerEntryKind::grant, BudgetLedgerStatus::active,
        BudgetUnit::materializedMemberDispatch, 1, 0, 1,
        ActorIdentity("authoritative.budget.admin@example.test"),
        Reason("Authoritative direct budget reason."));
    const long long entryId = transaction.exec(
        "INSERT INTO campaign_operations_budget_ledger_entry ("
        "operational_campaign_id,campaign_identity_canonical,"
        "previous_entry_id,previous_entry_identity_canonical,"
        "previous_entry_identity_hash,ledger_version,entry_kind,"
        "ledger_status,budget_unit,delta,prior_total,resulting_total,"
        "administrator_identity,reason,budget_contract_version,"
        "budget_identity_canonical,budget_identity_hash) "
        "VALUES($1,$2,NULL,NULL,NULL,1,'grant','active',$3,1,0,1,$4,$5,"
        "$6,$7,$8) RETURNING budget_ledger_entry_id;",
        pqxx::params{campaign.campaignId.value(),
            campaign.campaign.identity.canonicalText(), ToText(entry.unit),
            entry.administrator.value(), entry.reason.value(),
            entry.identity.contractVersion(), entry.identity.canonicalText(),
            entry.identity.hash()})
        .one_row()[0]
        .as<long long>();
    if (auditActor && auditReason)
        transaction.exec(
            "INSERT INTO campaign_operations_audit_reference_event ("
            "operational_campaign_id,budget_ledger_entry_id,cause_kind,"
            "actor_identity,capability,reason,prior_version,resulting_version,"
            "outcome,replay_disposition) VALUES($1,$2,"
            "'budget_ledger_recorded',$3,"
            "'campaign_operations_budget_administrator',$4,NULL,1,"
            "'recorded','recorded');",
            pqxx::params{campaign.campaignId.value(), entryId, *auditActor,
                *auditReason});
}

struct DirectAcceptanceAuditOverrides final
{
    std::string actor = "direct.requester@example.test";
    std::string reason = "Attempt malformed direct request evidence.";
    std::string capability = "campaign_operations_request_acceptor";
    std::optional<long long> budgetLedgerEntryId;
    std::optional<long long> authorizationEventId;
    std::optional<int> resultingVersion;
    bool includeAudit = true;
};

void InsertDirectAcquisitionAndAudit(
    pqxx::transaction_base& transaction,
    const Phase2AcceptanceFixture& fixture, long long reservationId,
    long long requestId, bool includeAcquisition,
    const DirectAcceptanceAuditOverrides& overrides)
{
    const auto reservation =
        FindReservation(transaction, ReservationId(reservationId));
    const auto request =
        FindOperationalRequest(transaction, OperationalRequestId(requestId));
    assert(reservation);
    assert(request);
    if (!includeAcquisition) return;
    const ReservationEvent event = BuildReservationAcquisitionEvent(
        reservation->reservationId,
        reservation->reservation.identity.canonicalText(),
        request->requestId, request->request.identity.canonicalText(),
        reservation->reservation.amount);
    const long long eventId = transaction.exec(
        "INSERT INTO campaign_operations_reservation_event ("
        "reservation_id,reservation_identity_canonical,transition_kind,"
        "expected_state,resulting_state,expected_version,resulting_version,"
        "operational_request_id,request_identity_canonical,amount,"
        "reservation_event_contract_version,"
        "reservation_event_identity_canonical,"
        "reservation_event_identity_hash) VALUES($1,$2,'acquired',NULL,"
        "'held',0,1,$3,$4,$5,$6,$7,$8) RETURNING reservation_event_id;",
        pqxx::params{reservationId,
            reservation->reservation.identity.canonicalText(), requestId,
            request->request.identity.canonicalText(),
            reservation->reservation.amount,
            event.identity.contractVersion(), event.identity.canonicalText(),
            event.identity.hash()})
        .one_row()[0]
        .as<long long>();
    if (!overrides.includeAudit) return;
    transaction.exec(
        "INSERT INTO campaign_operations_audit_reference_event ("
        "operational_campaign_id,authorization_event_id,"
        "budget_ledger_entry_id,reservation_id,reservation_event_id,"
        "operational_request_id,cause_kind,actor_identity,capability,reason,"
        "prior_version,resulting_version,outcome,replay_disposition) "
        "VALUES($1,$2,$3,$4,$5,$6,'reservation_request_accepted',$7,$8,$9,"
        "NULL,$10,'recorded','recorded');",
        pqxx::params{fixture.campaign.campaignId.value(),
            overrides.authorizationEventId.value_or(
                fixture.authorization.authorizationEventId.value()),
            overrides.budgetLedgerEntryId.value_or(
                fixture.budget.budgetLedgerEntryId.value()),
            reservationId, eventId, requestId, overrides.actor,
            overrides.capability, overrides.reason,
            overrides.resultingVersion.value_or(1)});
}

template <typename Function>
void ExpectSqlFailure(Function&& function)
{
    bool failed = false;
    try
    {
        function();
    }
    catch (const pqxx::sql_error&)
    {
        failed = true;
    }
    assert(failed);
}

void TestDirectCapabilityIntegrity(pqxx::connection& owner,
    const std::string& connectionString, const std::string& schema)
{
    const std::string budgetConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_budget_administrator'";
    const std::string requestConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_request_acceptor'";
    const auto policyFixture = CreatePhase2AcceptanceFixture(owner,
        budgetConnectionString, schema, 70, 1);
    const CanonicalIdentity fakeProvenance =
        CanonicalIdentity::Create(1, "direct-fake-provenance");

    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        const long long reservationId =
            InsertDirectReservation(transaction, policyFixture, std::nullopt);
        (void)InsertDirectRequest(transaction, policyFixture, reservationId,
            PrerequisitePolicy::
                phase4dMaterializationPlusExactPhase6dRatificationV1,
            fakeProvenance.canonicalText(), fakeProvenance.hash());
        transaction.commit();
    });
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_reservation",
               policyFixture.campaign.campaignId) == 0);

    const auto provenanceFixture = CreatePhase2AcceptanceFixture(owner,
        budgetConnectionString, schema, 71, 1);
    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        const long long reservationId =
            InsertDirectReservation(
                transaction, provenanceFixture, std::nullopt);
        (void)InsertDirectRequest(transaction, provenanceFixture,
            reservationId,
            PrerequisitePolicy::phase4dMaterializationOnlyV1,
            fakeProvenance.canonicalText(), fakeProvenance.hash());
        transaction.commit();
    });
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_reservation",
               provenanceFixture.campaign.campaignId) == 0);

    const auto expiryFixture = CreatePhase2AcceptanceFixture(owner,
        budgetConnectionString, schema, 72, 1);
    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        (void)InsertDirectReservation(transaction, expiryFixture,
            UtcTimestamp("2000-01-01T00:00:00.000000Z"));
        transaction.commit();
    });
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_reservation",
               expiryFixture.campaign.campaignId) == 0);

    for (const bool actorMismatch : {true, false})
    {
        const long long materializationId = actorMismatch ? 73 : 74;
        const OperationalCampaign campaign =
            InsertMaterialization(owner, schema, materializationId, 1);
        const auto persisted =
            PersistCampaignFixture(owner, schema, campaign);
        ExpectSqlFailure([&]
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            SetBudgetAdministratorRole(transaction);
            InsertDirectBudgetGrantWithAudit(transaction, persisted,
                actorMismatch
                    ? "false.audit.actor@example.test"
                    : "authoritative.budget.admin@example.test",
                actorMismatch
                    ? "Authoritative direct budget reason."
                    : "False audit reason.");
            transaction.commit();
        });
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_budget_ledger_entry",
                   persisted.campaignId) == 0);
    }

    const auto reservationOnlyFixture = CreatePhase2AcceptanceFixture(owner,
        budgetConnectionString, schema, 76, 1);
    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        (void)InsertDirectReservation(
            transaction, reservationOnlyFixture, std::nullopt);
        transaction.commit();
    });
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_reservation",
               reservationOnlyFixture.campaign.campaignId) == 0);

    const auto requestOnlyFixture = CreatePhase2AcceptanceFixture(owner,
        budgetConnectionString, schema, 77, 1);
    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        const long long reservationId =
            InsertDirectReservation(
                transaction, requestOnlyFixture, std::nullopt);
        (void)InsertDirectRequest(transaction, requestOnlyFixture,
            reservationId,
            PrerequisitePolicy::phase4dMaterializationOnlyV1,
            std::nullopt, std::nullopt);
        transaction.commit();
    });
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_operational_request",
               requestOnlyFixture.campaign.campaignId) == 0);

    const auto missingAuditFixture = CreatePhase2AcceptanceFixture(owner,
        budgetConnectionString, schema, 78, 1);
    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        const long long reservationId =
            InsertDirectReservation(
                transaction, missingAuditFixture, std::nullopt);
        const long long requestId = InsertDirectRequest(transaction,
            missingAuditFixture, reservationId,
            PrerequisitePolicy::phase4dMaterializationOnlyV1,
            std::nullopt, std::nullopt);
        DirectAcceptanceAuditOverrides overrides;
        overrides.includeAudit = false;
        InsertDirectAcquisitionAndAudit(transaction, missingAuditFixture,
            reservationId, requestId, true, overrides);
        transaction.commit();
    });
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_operational_request",
               missingAuditFixture.campaign.campaignId) == 0);

    const OperationalCampaign missingBudgetAuditCampaign =
        InsertMaterialization(owner, schema, 79, 1);
    const auto missingBudgetAuditPersisted =
        PersistCampaignFixture(owner, schema, missingBudgetAuditCampaign);
    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetBudgetAdministratorRole(transaction);
        InsertDirectBudgetGrantWithAudit(transaction,
            missingBudgetAuditPersisted, std::nullopt, std::nullopt);
        transaction.commit();
    });
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_budget_ledger_entry",
               missingBudgetAuditPersisted.campaignId) == 0);

    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        transaction.exec(
            "INSERT INTO campaign_operations_reservation_event ("
            "reservation_id,reservation_identity_canonical,transition_kind,"
            "expected_state,resulting_state,expected_version,"
            "resulting_version,operational_request_id,"
            "request_identity_canonical,amount,"
            "reservation_event_contract_version,"
            "reservation_event_identity_canonical,"
            "reservation_event_identity_hash) VALUES(999999,"
            "'missing-reservation','acquired',NULL,'held',0,1,999999,"
            "'missing-request',1,1,'missing-event',"
            "'fnv1a64:0000000000000000');");
        transaction.commit();
    });

    for (int mismatch = 0; mismatch < 6; ++mismatch)
    {
        const auto fixture = CreatePhase2AcceptanceFixture(owner,
            budgetConnectionString, schema, 81 + mismatch, 1);
        ExpectSqlFailure([&]
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            SetRequestAcceptorRole(transaction);
            const long long reservationId =
                InsertDirectReservation(transaction, fixture, std::nullopt);
            const long long requestId = InsertDirectRequest(transaction,
                fixture, reservationId,
                PrerequisitePolicy::phase4dMaterializationOnlyV1,
                std::nullopt, std::nullopt);
            DirectAcceptanceAuditOverrides overrides;
            if (mismatch == 0)
                overrides.actor = "false.audit.actor@example.test";
            else if (mismatch == 1)
                overrides.reason = "False acceptance audit reason.";
            else if (mismatch == 2)
                overrides.capability =
                    "campaign_operations_budget_administrator";
            else if (mismatch == 3)
                overrides.budgetLedgerEntryId =
                    policyFixture.budget.budgetLedgerEntryId.value();
            else if (mismatch == 4)
                overrides.authorizationEventId =
                    policyFixture.authorization.authorizationEventId.value();
            else
                overrides.resultingVersion = 2;
            InsertDirectAcquisitionAndAudit(transaction, fixture,
                reservationId, requestId, true, overrides);
            transaction.commit();
        });
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_operational_request",
                   fixture.campaign.campaignId) == 0);
    }

    const auto crossFixture = CreatePhase2AcceptanceFixture(owner,
        budgetConnectionString, schema, 75, 1);
    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        transaction.exec(
            "INSERT INTO campaign_operations_budget_ledger_entry "
            "(operational_campaign_id) VALUES($1);",
            pqxx::params{crossFixture.campaign.campaignId.value()});
    });
    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        transaction.exec(
            "INSERT INTO campaign_operations_reservation "
            "(reservation_id,operational_campaign_id) VALUES(999999,$1);",
            pqxx::params{crossFixture.campaign.campaignId.value()});
    });
    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        transaction.exec(
            "UPDATE campaign_operations_reservation SET state_version=2;");
    });
    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        transaction.exec("DELETE FROM campaign_operations_reservation;");
    });
    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        transaction.exec("TRUNCATE campaign_operations_reservation;");
    });
    ExpectSqlFailure([&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        transaction.exec(
            "ALTER SEQUENCE campaign_operations_reservation_reservation_id_seq "
            "RESTART WITH 1;");
    });

    const auto hydrationFixture = CreatePhase2AcceptanceFixture(owner,
        budgetConnectionString, schema, 87, 1);
    PersistResult<AcceptedOperationalRequest> accepted = [&]
    {
        pqxx::connection connection{requestConnectionString};
        return AcceptOperationalRequest(connection,
            {hydrationFixture.campaign.campaignId.value(),
                "hydration.requester@example.test",
                "Persist evidence for fail-closed hydration.",
                std::nullopt});
    }();
    const OperationalRequest& persistedRequest =
        accepted.persisted.request.request;
    const OperationalRequest malformed = BuildOperationalRequest(
        persistedRequest.logicalOperation,
        persistedRequest.acceptingAuthorizationEventId,
        persistedRequest.acceptingAuthorizationCanonicalText,
        persistedRequest.acceptingAuthorizationIdentityHash,
        persistedRequest.reservationId,
        persistedRequest.reservationCanonicalText,
        persistedRequest.reservationIdentityHash,
        persistedRequest.memberCount, persistedRequest.orderedScopeDigest,
        persistedRequest.acceptingActor, persistedRequest.reason,
        PrerequisitePolicy::
            phase4dMaterializationPlusExactPhase6dRatificationV1,
        fakeProvenance.canonicalText(), fakeProvenance.hash());
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec("SET LOCAL ROLE campaign_operations_owner;");
        transaction.exec(
            "UPDATE campaign_operations_operational_request SET "
            "prerequisite_policy=$1,provenance_identity_canonical=$2,"
            "provenance_identity_hash=$3,request_identity_canonical=$4,"
            "request_identity_hash=$5 WHERE operational_request_id=$6;",
            pqxx::params{ToText(malformed.prerequisitePolicy),
                malformed.provenanceCanonicalText,
                malformed.provenanceIdentityHash,
                malformed.identity.canonicalText(), malformed.identity.hash(),
                accepted.persisted.request.requestId.value()});
        transaction.commit();
    }
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        bool rejected = false;
        try
        {
            (void)FindOperationalRequest(transaction,
                accepted.persisted.request.requestId);
        }
        catch (const Error& error)
        {
            rejected =
                error.code() == ErrorCode::persistenceCorruption &&
                std::string(error.what()) ==
                    "campaign_operations_request_authorization_evidence_mismatch";
        }
        assert(rejected);
    }
    {
        pqxx::connection statusConnection{requestConnectionString};
        bool rejected = false;
        try
        {
            (void)LoadOperationalRequestStatus(statusConnection,
                accepted.persisted.request.requestId);
        }
        catch (const Error& error)
        {
            rejected =
                error.code() == ErrorCode::persistenceCorruption &&
                std::string(error.what()) ==
                    "campaign_operations_reservation_reconciliation_required";
        }
        assert(rejected);
    }
}

void TestPhase2CorrectionConcurrency(pqxx::connection& owner,
    const std::string& connectionString, const std::string& schema)
{
    const std::string budgetConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_budget_administrator'";
    const std::string requestConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_request_acceptor'";

    {
        const auto fixture = CreatePhase2AcceptanceFixture(owner,
            budgetConnectionString, schema, 88, 2);
        const OperationalRequestAcceptanceRequest firstRequest{
            fixture.campaign.campaignId.value(),
            "race.requester@example.test",
            "Competing changed payload one.", std::nullopt};
        OperationalRequestAcceptanceRequest secondRequest = firstRequest;
        secondRequest.reason = "Competing changed payload two.";
        std::atomic<bool> start{false};
        bool firstRecorded = false;
        bool secondRecorded = false;
        bool firstConflict = false;
        bool secondConflict = false;
        std::exception_ptr firstUnexpected;
        std::exception_ptr secondUnexpected;
        const auto accept = [&](const OperationalRequestAcceptanceRequest& value,
                                bool& recorded, bool& conflict,
                                std::exception_ptr& unexpected)
        {
            try
            {
                pqxx::connection connection{requestConnectionString};
                while (!start.load(std::memory_order_acquire))
                    std::this_thread::yield();
                const auto result = AcceptOperationalRequest(connection, value);
                recorded = result.outcome == PersistOutcome::recorded;
            }
            catch (const Error& error)
            {
                conflict = error.code() == ErrorCode::persistenceConflict;
                if (!conflict) unexpected = std::current_exception();
            }
            catch (...)
            {
                unexpected = std::current_exception();
            }
        };
        std::thread first(accept, std::cref(firstRequest),
            std::ref(firstRecorded), std::ref(firstConflict),
            std::ref(firstUnexpected));
        std::thread second(accept, std::cref(secondRequest),
            std::ref(secondRecorded), std::ref(secondConflict),
            std::ref(secondUnexpected));
        start.store(true, std::memory_order_release);
        first.join();
        second.join();
        if (firstUnexpected) std::rethrow_exception(firstUnexpected);
        if (secondUnexpected) std::rethrow_exception(secondUnexpected);
        assert(firstRecorded != secondRecorded);
        assert(firstConflict != secondConflict);
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_operational_request",
                   fixture.campaign.campaignId) == 1);
    }

    {
        const auto fixture = CreatePhase2AcceptanceFixture(owner,
            budgetConnectionString, schema, 89, 2);
        BudgetAdministrationRequest first;
        first.campaignId = fixture.campaign.campaignId.value();
        first.expectedLedgerVersion = 1;
        first.kind = BudgetLedgerEntryKind::amend;
        first.value = 1;
        first.actorIdentity = "race.budget.admin@example.test";
        first.reason = "Concurrent successor budget one.";
        BudgetAdministrationRequest second = first;
        second.value = 2;
        second.reason = "Concurrent successor budget two.";
        std::atomic<bool> start{false};
        bool firstRecorded = false;
        bool secondRecorded = false;
        bool firstConflict = false;
        bool secondConflict = false;
        std::exception_ptr firstUnexpected;
        std::exception_ptr secondUnexpected;
        const auto amend = [&](const BudgetAdministrationRequest& value,
                               bool& recorded, bool& conflict,
                               std::exception_ptr& unexpected)
        {
            try
            {
                pqxx::connection connection{budgetConnectionString};
                while (!start.load(std::memory_order_acquire))
                    std::this_thread::yield();
                const auto result = AdministerCampaignBudget(connection, value);
                recorded = result.outcome == PersistOutcome::recorded;
            }
            catch (const Error& error)
            {
                conflict = error.code() == ErrorCode::persistenceConflict;
                if (!conflict) unexpected = std::current_exception();
            }
            catch (...)
            {
                unexpected = std::current_exception();
            }
        };
        std::thread firstThread(amend, std::cref(first),
            std::ref(firstRecorded), std::ref(firstConflict),
            std::ref(firstUnexpected));
        std::thread secondThread(amend, std::cref(second),
            std::ref(secondRecorded), std::ref(secondConflict),
            std::ref(secondUnexpected));
        start.store(true, std::memory_order_release);
        firstThread.join();
        secondThread.join();
        if (firstUnexpected) std::rethrow_exception(firstUnexpected);
        if (secondUnexpected) std::rethrow_exception(secondUnexpected);
        assert(firstRecorded != secondRecorded);
        assert(firstConflict != secondConflict);
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_budget_ledger_entry",
                   fixture.campaign.campaignId) == 2);
    }

    for (const bool revokeBudget : {false, true})
    {
        const long long materializationId = revokeBudget ? 91 : 90;
        const auto fixture = CreatePhase2AcceptanceFixture(owner,
            budgetConnectionString, schema, materializationId, 2);
        OperationalRequestAcceptanceRequest acceptance{
            fixture.campaign.campaignId.value(),
            "budget.race.requester@example.test",
            revokeBudget
                ? "Race budget revocation against acceptance."
                : "Race budget amendment against acceptance.",
            std::nullopt};
        BudgetAdministrationRequest mutation;
        mutation.campaignId = fixture.campaign.campaignId.value();
        mutation.expectedLedgerVersion = 1;
        mutation.kind = revokeBudget
            ? BudgetLedgerEntryKind::revoke
            : BudgetLedgerEntryKind::amend;
        mutation.value = revokeBudget
            ? std::nullopt
            : std::optional<long long>(-1);
        mutation.actorIdentity = "budget.race.admin@example.test";
        mutation.reason = revokeBudget
            ? "Revoke concurrently with request acceptance."
            : "Reduce concurrently below request requirement.";
        std::atomic<bool> start{false};
        bool accepted = false;
        bool acceptanceDenied = false;
        bool budgetRecorded = false;
        bool budgetDenied = false;
        std::exception_ptr acceptanceUnexpected;
        std::exception_ptr budgetUnexpected;
        std::thread acceptThread([&]
        {
            try
            {
                pqxx::connection connection{requestConnectionString};
                while (!start.load(std::memory_order_acquire))
                    std::this_thread::yield();
                accepted = AcceptOperationalRequest(connection, acceptance)
                               .outcome == PersistOutcome::recorded;
            }
            catch (const Error& error)
            {
                acceptanceDenied = error.code() == ErrorCode::budgetDenied;
                if (!acceptanceDenied)
                    acceptanceUnexpected = std::current_exception();
            }
            catch (...)
            {
                acceptanceUnexpected = std::current_exception();
            }
        });
        std::thread budgetThread([&]
        {
            try
            {
                pqxx::connection connection{budgetConnectionString};
                while (!start.load(std::memory_order_acquire))
                    std::this_thread::yield();
                budgetRecorded =
                    AdministerCampaignBudget(connection, mutation).outcome ==
                    PersistOutcome::recorded;
            }
            catch (const Error& error)
            {
                budgetDenied = error.code() == ErrorCode::budgetDenied;
                if (!budgetDenied)
                    budgetUnexpected = std::current_exception();
            }
            catch (...)
            {
                budgetUnexpected = std::current_exception();
            }
        });
        start.store(true, std::memory_order_release);
        acceptThread.join();
        budgetThread.join();
        if (acceptanceUnexpected)
            std::rethrow_exception(acceptanceUnexpected);
        if (budgetUnexpected) std::rethrow_exception(budgetUnexpected);
        assert(accepted != acceptanceDenied);
        assert(budgetRecorded != budgetDenied);
        if (!revokeBudget)
            assert(accepted != budgetRecorded);
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_operational_request",
                   fixture.campaign.campaignId) == (accepted ? 1 : 0));
    }

    for (const bool supersede : {false, true})
    {
        const long long materializationId = supersede ? 93 : 92;
        const auto fixture = CreatePhase2AcceptanceFixture(owner,
            budgetConnectionString, schema, materializationId, 1);
        const AuthorizationEventKind successorKind = supersede
            ? AuthorizationEventKind::granted
            : AuthorizationEventKind::revoked;
        const OperationalAuthorizationEvent successor =
            BuildOperationalAuthorizationEvent(
                fixture.campaign.campaignId,
                fixture.campaign.campaign.identity.canonicalText(),
                fixture.authorization.authorizationEventId,
                fixture.authorization.event.identity.canonicalText(),
                fixture.authorization.event.identity.hash(), 2, successorKind,
                fixture.campaign.campaign.actionKind,
                fixture.campaign.campaign.actionContractVersion,
                fixture.campaign.campaign.scopeKind,
                fixture.campaign.campaign.scopeContractVersion,
                PrerequisitePolicy::phase4dMaterializationOnlyV1,
                std::nullopt, std::nullopt, std::nullopt,
                kCampaignOperationsAuthorizationRole,
                ActorIdentity("authorization.race@example.test"),
                Reason(supersede
                        ? "Supersede concurrently with acceptance."
                        : "Revoke concurrently with acceptance."),
                UtcTimestamp("2020-01-02T00:00:00.000000Z"),
                std::nullopt);
        OperationalRequestAcceptanceRequest acceptance{
            fixture.campaign.campaignId.value(),
            "authorization.race.requester@example.test",
            supersede
                ? "Race authorization supersession."
                : "Race authorization revocation.",
            std::nullopt};
        std::atomic<bool> start{false};
        bool accepted = false;
        bool denied = false;
        bool successorRecorded = false;
        std::exception_ptr acceptanceUnexpected;
        std::exception_ptr successorUnexpected;
        std::thread acceptThread([&]
        {
            try
            {
                pqxx::connection connection{requestConnectionString};
                while (!start.load(std::memory_order_acquire))
                    std::this_thread::yield();
                accepted = AcceptOperationalRequest(connection, acceptance)
                               .outcome == PersistOutcome::recorded;
            }
            catch (const Error& error)
            {
                denied = error.code() == ErrorCode::authorizationDenied;
                if (!denied) acceptanceUnexpected = std::current_exception();
            }
            catch (...)
            {
                acceptanceUnexpected = std::current_exception();
            }
        });
        std::thread successorThread([&]
        {
            try
            {
                pqxx::connection connection{connectionString};
                while (!start.load(std::memory_order_acquire))
                    std::this_thread::yield();
                pqxx::work transaction{connection};
                SetSearchPath(transaction, schema);
                SetAuthorizerRole(transaction);
                successorRecorded =
                    PersistOperationalAuthorizationEvent(
                        transaction, successor).outcome ==
                    PersistOutcome::recorded;
                transaction.commit();
            }
            catch (...)
            {
                successorUnexpected = std::current_exception();
            }
        });
        start.store(true, std::memory_order_release);
        acceptThread.join();
        successorThread.join();
        if (acceptanceUnexpected)
            std::rethrow_exception(acceptanceUnexpected);
        if (successorUnexpected)
            std::rethrow_exception(successorUnexpected);
        assert(successorRecorded);
        assert(accepted != denied);
        if (supersede) assert(accepted);
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_operational_request",
                   fixture.campaign.campaignId) == (accepted ? 1 : 0));
    }

    {
        const auto fixture = CreatePhase2AcceptanceFixture(owner,
            budgetConnectionString, schema, 94, 1);
        const OperationalAuthorizationEvent revoke =
            BuildOperationalAuthorizationEvent(
                fixture.campaign.campaignId,
                fixture.campaign.campaign.identity.canonicalText(),
                fixture.authorization.authorizationEventId,
                fixture.authorization.event.identity.canonicalText(),
                fixture.authorization.event.identity.hash(), 2,
                AuthorizationEventKind::revoked,
                fixture.campaign.campaign.actionKind,
                fixture.campaign.campaign.actionContractVersion,
                fixture.campaign.campaign.scopeKind,
                fixture.campaign.campaign.scopeContractVersion,
                PrerequisitePolicy::phase4dMaterializationOnlyV1,
                std::nullopt, std::nullopt, std::nullopt,
                kCampaignOperationsAuthorizationRole,
                ActorIdentity("direct.authorization.race@example.test"),
                Reason("Direct authorization must share the acceptance lock."),
                UtcTimestamp("2020-01-02T00:00:00.000000Z"),
                std::nullopt);
        pqxx::work lockHolder{owner};
        SetSearchPath(lockHolder, schema);
        LockOperationalAuthorizationDomain(
            lockHolder, fixture.campaign.campaign);

        bool lockTimedOut = false;
        std::exception_ptr unexpected;
        std::thread directWriter([&]
        {
            try
            {
                pqxx::connection connection{connectionString};
                pqxx::work transaction{connection};
                SetSearchPath(transaction, schema);
                SetAuthorizerRole(transaction);
                transaction.exec("SET LOCAL lock_timeout='250ms';");
                (void)InsertAuthorizationDirect(transaction, revoke);
                transaction.abort();
            }
            catch (const pqxx::sql_error& error)
            {
                lockTimedOut = error.sqlstate() == "55P03";
                if (!lockTimedOut) unexpected = std::current_exception();
            }
            catch (...)
            {
                unexpected = std::current_exception();
            }
        });
        directWriter.join();
        if (unexpected) std::rethrow_exception(unexpected);
        assert(lockTimedOut);
        lockHolder.abort();
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_authorization_event",
                   fixture.campaign.campaignId) == 1);
    }
}

void TestBudgetReservationAndRequestAcceptance(
    pqxx::connection& owner, const std::string& connectionString,
    const std::string& schema)
{
    assert(CampaignOperationsMachineText("comma,\"quote\"\nline\tcontrol") ==
        "comma%2C%22quote%22%0Aline%09control");
    assert(CampaignOperationsMachineText("NULL") == "%4E%55%4C%4C");
    const std::string budgetConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_budget_administrator'";
    const std::string requestConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_request_acceptor'";
    const std::string readerConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_reader'";
    const OperationalCampaign campaign =
        InsertMaterialization(owner, schema, 60, 3);
    const auto persistedCampaign =
        PersistCampaignFixture(owner, schema, campaign);
    const auto authorization = PersistActiveAuthorizationFixture(
        owner, schema, persistedCampaign);
    assert(authorization.event.eventKind ==
        AuthorizationEventKind::granted);

    {
        pqxx::connection connection{requestConnectionString};
        bool denied = false;
        try
        {
            (void)AcceptOperationalRequest(connection,
                {persistedCampaign.campaignId.value(),
                    "requester@example.test",
                    "Accept one durable complete-materialization request.",
                    std::nullopt});
        }
        catch (const Error& error)
        {
            denied = error.code() == ErrorCode::budgetDenied;
        }
        assert(denied);
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_reservation",
               persistedCampaign.campaignId) == 0);
    BudgetAdministrationRequest grantRequest;
    grantRequest.campaignId = persistedCampaign.campaignId.value();
    grantRequest.expectedLedgerVersion = 0;
    grantRequest.kind = BudgetLedgerEntryKind::grant;
    grantRequest.value = 3;
    grantRequest.actorIdentity = "budget.admin@example.test";
    grantRequest.reason = "Fund the exact three-member materialization.";
    {
        pqxx::connection connection{budgetConnectionString};
        const auto granted =
            AdministerCampaignBudget(connection, grantRequest);
        assert(granted.outcome == PersistOutcome::recorded);
        assert(granted.persisted.entry.resultingTotal == 3);
        assert(granted.persisted.entry.ledgerVersion == 1);
    }
    {
        pqxx::connection connection{budgetConnectionString};
        const auto replay =
            AdministerCampaignBudget(connection, grantRequest);
        assert(replay.outcome == PersistOutcome::existingIdentical);
        BudgetAdministrationRequest changed = grantRequest;
        changed.reason = "A changed reason conflicts with version one.";
        bool conflicted = false;
        try
        {
            (void)AdministerCampaignBudget(connection, changed);
        }
        catch (const Error& error)
        {
            conflicted = error.code() == ErrorCode::persistenceConflict;
        }
        assert(conflicted);
    }

    OperationalRequestAcceptanceRequest acceptance;
    acceptance.campaignId = persistedCampaign.campaignId.value();
    acceptance.actorIdentity = "requester@example.test";
    acceptance.reason =
        "Accept one durable complete-materialization request.";
    {
        pqxx::connection connection{requestConnectionString};
        const auto recorded = AcceptOperationalRequest(connection, acceptance);
        assert(recorded.outcome == PersistOutcome::recorded);
        assert(recorded.persisted.reservation.state ==
            ReservationState::held);
        assert(recorded.persisted.request.state == RequestState::ready);
        assert(!recorded.persisted.request.productionDispatchEnabled);
        assert(recorded.persisted.reservation.reservation.amount == 3);
        assert(recorded.persisted.acquisitionEvent.event.eventKind ==
            ReservationEventKind::acquired);
    }
    {
        std::ostringstream output;
        std::ostringstream errors;
        const int rc = RunCampaignOperationalRequestAcceptanceCommand(
            requestConnectionString, acceptance, output, errors);
        assert(rc == 0);
        assert(errors.str().empty());
        for (const std::string& field : {
                 "operational_campaign_id=",
                 "authorization_event_id=",
                 "budget_ledger_entry_id=",
                 "budget_ledger_version=",
                 "reservation_id=",
                 "reservation_event_id=",
                 "operational_request_id=",
                 "reservation_state=held",
                 "reservation_state_version=1",
                 "request_state=ready",
                 "request_state_version=1",
                 "replay_disposition=existing_identical",
                 "production_dispatch_enabled=false"})
            assert(output.str().find(field) != std::string::npos);
    }
    {
        pqxx::connection connection{requestConnectionString};
        const auto replay = AcceptOperationalRequest(connection, acceptance);
        assert(replay.outcome == PersistOutcome::existingIdentical);
        OperationalRequestAcceptanceRequest changed = acceptance;
        changed.actorIdentity = "different.requester@example.test";
        bool conflicted = false;
        try
        {
            (void)AcceptOperationalRequest(connection, changed);
        }
        catch (const Error& error)
        {
            conflicted = error.code() == ErrorCode::persistenceConflict;
        }
        assert(conflicted);
        const auto status = LoadCampaignBudgetStatus(
            connection, persistedCampaign.campaignId);
        assert(status.accountingConsistent);
        assert(status.everReserved == 3);
        assert(status.held == 3);
        assert(status.reservable == 0);
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_reservation",
               persistedCampaign.campaignId) == 1);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_operational_request",
               persistedCampaign.campaignId) == 1);
    assert(CountRows(owner, schema,
               "campaign_operations_reservation_event") == 1);

    BudgetAdministrationRequest belowHold = grantRequest;
    belowHold.expectedLedgerVersion = 1;
    belowHold.kind = BudgetLedgerEntryKind::amend;
    belowHold.value = -1;
    belowHold.reason = "This amendment must not reduce below the held units.";
    {
        pqxx::connection connection{budgetConnectionString};
        bool denied = false;
        try
        {
            (void)AdministerCampaignBudget(connection, belowHold);
        }
        catch (const Error& error)
        {
            denied = error.code() == ErrorCode::budgetDenied;
        }
        assert(denied);
    }
    BudgetAdministrationRequest revoke = grantRequest;
    revoke.expectedLedgerVersion = 1;
    revoke.kind = BudgetLedgerEntryKind::revoke;
    revoke.value.reset();
    revoke.reason = "Revoke availability while preserving the exact hold.";
    {
        pqxx::connection connection{budgetConnectionString};
        const auto revoked = AdministerCampaignBudget(connection, revoke);
        assert(revoked.persisted.entry.status ==
            BudgetLedgerStatus::revoked);
        assert(revoked.persisted.entry.resultingTotal == 3);
        bool denied = false;
        try
        {
            (void)AcceptOperationalRequest(connection, acceptance);
        }
        catch (const Error& error)
        {
            denied = error.code() == ErrorCode::authorizationDenied;
        }
        assert(denied);
    }
    {
        pqxx::connection connection{requestConnectionString};
        bool conflicted = false;
        try
        {
            (void)AcceptOperationalRequest(connection, acceptance);
        }
        catch (const Error& error)
        {
            conflicted = error.code() == ErrorCode::persistenceConflict;
        }
        assert(conflicted);
    }
    BudgetAdministrationRequest supersede = grantRequest;
    supersede.expectedLedgerVersion = 2;
    supersede.kind = BudgetLedgerEntryKind::supersede;
    supersede.value = 5;
    supersede.reason =
        "Explicitly supersede the revoked head with five units.";
    {
        pqxx::connection connection{budgetConnectionString};
        const auto restored =
            AdministerCampaignBudget(connection, supersede);
        assert(restored.persisted.entry.status ==
            BudgetLedgerStatus::active);
        const auto status = LoadCampaignBudgetStatus(
            connection, persistedCampaign.campaignId);
        assert(status.held == 3);
        assert(status.reservable == 2);
    }

    {
        std::ostringstream output;
        std::ostringstream errors;
        const int rc = RunCampaignBudgetStatusCommand(
            readerConnectionString, persistedCampaign.campaignId,
            output, errors);
        assert(rc == 0);
        assert(errors.str().empty());
        assert(output.str().find("accounting_consistent=true") !=
            std::string::npos);
    }
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        const auto request = FindOperationalRequestByCampaignAction(
            transaction, persistedCampaign.campaignId,
            OperationalActionKind::dispatchFullMaterialization, 1);
        assert(request);
        std::ostringstream output;
        std::ostringstream errors;
        const int rc = RunCampaignOperationalRequestStatusCommand(
            readerConnectionString, request->requestId, output, errors);
        assert(rc == 0);
        assert(errors.str().empty());
        assert(output.str().find(
            "production_dispatch_enabled=false") != std::string::npos);
        for (const std::string& field : {
                 "authorization_event_id=",
                 "budget_ledger_entry_id=",
                 "budget_ledger_version=",
                 "reservation_id=",
                 "reservation_state=held",
                 "reservation_state_version=1",
                 "reservation_amount=3",
                 "reservation_budget_unit=materialized_member_dispatch",
                 "reservation_identity_hash="})
            assert(output.str().find(field) != std::string::npos);
    }

    const OperationalCampaign rollbackCampaign =
        InsertMaterialization(owner, schema, 61, 2);
    const auto persistedRollback =
        PersistCampaignFixture(owner, schema, rollbackCampaign);
    (void)PersistActiveAuthorizationFixture(
        owner, schema, persistedRollback);
    BudgetAdministrationRequest rollbackBudget = grantRequest;
    rollbackBudget.campaignId = persistedRollback.campaignId.value();
    rollbackBudget.value = 2;
    rollbackBudget.reason = "Fund rollback acceptance fixture.";
    {
        pqxx::connection connection{budgetConnectionString};
        (void)AdministerCampaignBudget(connection, rollbackBudget);
    }
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetRequestAcceptorRole(transaction);
        const auto result = PersistAcceptedOperationalRequest(transaction,
            persistedRollback.campaignId,
            ActorIdentity("rollback.requester@example.test"),
            Reason("The outer transaction intentionally rolls back."),
            std::nullopt);
        assert(result.outcome == PersistOutcome::recorded);
        transaction.abort();
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_reservation",
               persistedRollback.campaignId) == 0);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_operational_request",
               persistedRollback.campaignId) == 0);

    const OperationalCampaign concurrentCampaign =
        InsertMaterialization(owner, schema, 63, 2);
    const auto persistedConcurrent =
        PersistCampaignFixture(owner, schema, concurrentCampaign);
    const auto concurrentAuthorization =
        PersistActiveAuthorizationFixture(
            owner, schema, persistedConcurrent);
    BudgetAdministrationRequest concurrentBudget = grantRequest;
    concurrentBudget.campaignId = persistedConcurrent.campaignId.value();
    concurrentBudget.value = 2;
    concurrentBudget.reason = "Fund concurrent duplicate acceptance.";
    {
        pqxx::connection connection{budgetConnectionString};
        (void)AdministerCampaignBudget(connection, concurrentBudget);
    }
    const OperationalRequestAcceptanceRequest concurrentAcceptance{
        persistedConcurrent.campaignId.value(),
        "concurrent.requester@example.test",
        "Accept one exact operation under concurrent retry.",
        std::nullopt};
    std::atomic<bool> start{false};
    PersistOutcome firstOutcome = PersistOutcome::recorded;
    PersistOutcome secondOutcome = PersistOutcome::recorded;
    long long firstRequestId = 0;
    long long secondRequestId = 0;
    std::exception_ptr firstError;
    std::exception_ptr secondError;
    const auto accept = [&](PersistOutcome& outcome, long long& requestId,
                            std::exception_ptr& error)
    {
        try
        {
            pqxx::connection connection{requestConnectionString};
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            const auto result =
                AcceptOperationalRequest(connection, concurrentAcceptance);
            outcome = result.outcome;
            requestId = result.persisted.request.requestId.value();
        }
        catch (...)
        {
            error = std::current_exception();
        }
    };
    std::thread first(accept, std::ref(firstOutcome),
        std::ref(firstRequestId), std::ref(firstError));
    std::thread second(accept, std::ref(secondOutcome),
        std::ref(secondRequestId), std::ref(secondError));
    start.store(true, std::memory_order_release);
    first.join();
    second.join();
    if (firstError) std::rethrow_exception(firstError);
    if (secondError) std::rethrow_exception(secondError);
    assert(firstRequestId == secondRequestId);
    assert((firstOutcome == PersistOutcome::recorded &&
               secondOutcome == PersistOutcome::existingIdentical) ||
        (secondOutcome == PersistOutcome::recorded &&
            firstOutcome == PersistOutcome::existingIdentical));
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_reservation",
               persistedConcurrent.campaignId) == 1);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_operational_request",
               persistedConcurrent.campaignId) == 1);
    const auto revokeAuthorization = BuildOperationalAuthorizationEvent(
        persistedConcurrent.campaignId,
        concurrentCampaign.identity.canonicalText(),
        concurrentAuthorization.authorizationEventId,
        concurrentAuthorization.event.identity.canonicalText(),
        concurrentAuthorization.event.identity.hash(), 2,
        AuthorizationEventKind::revoked,
        concurrentCampaign.actionKind,
        concurrentCampaign.actionContractVersion,
        concurrentCampaign.scopeKind,
        concurrentCampaign.scopeContractVersion,
        PrerequisitePolicy::phase4dMaterializationOnlyV1, std::nullopt,
        std::nullopt, std::nullopt,
        kCampaignOperationsAuthorizationRole,
        ActorIdentity("phase2.authorizer@example.test"),
        Reason("Revoke the accepting grant after durable acceptance."),
        UtcTimestamp("2020-01-02T00:00:00.000000Z"), std::nullopt);
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetAuthorizerRole(transaction);
        (void)PersistOperationalAuthorizationEvent(
            transaction, revokeAuthorization);
        transaction.commit();
    }
    {
        pqxx::connection connection{requestConnectionString};
        bool conflicted = false;
        try
        {
            (void)AcceptOperationalRequest(
                connection, concurrentAcceptance);
        }
        catch (const Error& error)
        {
            conflicted = error.code() == ErrorCode::persistenceConflict;
        }
        assert(conflicted);
    }

    const OperationalCampaign unauthorizedCampaign =
        InsertMaterialization(owner, schema, 62, 1);
    const auto persistedUnauthorized =
        PersistCampaignFixture(owner, schema, unauthorizedCampaign);
    BudgetAdministrationRequest unauthorizedBudget = grantRequest;
    unauthorizedBudget.campaignId =
        persistedUnauthorized.campaignId.value();
    unauthorizedBudget.value = 1;
    unauthorizedBudget.reason =
        "Budget does not substitute for operational authorization.";
    {
        pqxx::connection connection{budgetConnectionString};
        (void)AdministerCampaignBudget(connection, unauthorizedBudget);
    }
    {
        pqxx::connection connection{requestConnectionString};
        bool denied = false;
        try
        {
            (void)AcceptOperationalRequest(connection,
                {persistedUnauthorized.campaignId.value(),
                    "requester@example.test",
                    "Authorization remains independently mandatory.",
                    std::nullopt});
        }
        catch (const Error& error)
        {
            denied = error.code() == ErrorCode::authorizationDenied;
        }
        assert(denied);
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_reservation",
               persistedUnauthorized.campaignId) == 0);

    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        SetBudgetAdministratorRole(transaction);
        bool requestInsertDenied = false;
        try
        {
            transaction.exec(
                "INSERT INTO campaign_operations_operational_request "
                "(operational_campaign_id) VALUES($1);",
                pqxx::params{persistedCampaign.campaignId.value()});
        }
        catch (const pqxx::sql_error& error)
        {
            requestInsertDenied = error.sqlstate() == "42501" ||
                std::string(error.what()).find("permission denied") !=
                    std::string::npos;
        }
        assert(requestInsertDenied);
        transaction.abort();
    }
}

void DropSchema(pqxx::connection& connection, const std::string& schema)
{
    pqxx::work transaction{connection};
    transaction.exec("DROP SCHEMA IF EXISTS " +
        transaction.quote_name(schema) + " CASCADE;");
    transaction.commit();
}

} // namespace

int main()
{
    const char* requestedDatabase = std::getenv("LSTM_TEST_DB_NAME");
    if (!requestedDatabase || !*requestedDatabase)
    {
        std::cerr << "LSTM_TEST_DB_NAME_required\n";
        return 2;
    }
    const std::string database = requestedDatabase;
    if (!SafeDisposableDatabaseName(database))
    {
        std::cerr << "clearly_disposable_non_LSTM_database_required\n";
        return 2;
    }
    const std::string host = EnvironmentOr("LSTM_TEST_DB_HOST", "127.0.0.1");
    const std::string port = EnvironmentOr("LSTM_TEST_DB_PORT", "5432");
    const std::string user = EnvironmentOr(
        "LSTM_TEST_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string schema =
        "campaign_operations_foundation_" + std::to_string(getpid());
    const std::string connectionString = "host=" + host + " port=" + port +
        " user=" + user + " dbname=" + database;
    pqxx::connection owner{connectionString};

    try
    {
        {
            pqxx::read_transaction verify{owner};
            const std::string actualDatabase = verify.exec(
                "SELECT current_database();").one_row()[0].as<std::string>();
            if (actualDatabase != database || Lowercase(actualDatabase) == "lstm")
                throw std::runtime_error("test_database_target_mismatch");
        }
        CreateBaseSchema(owner, schema);
        ApplyFile(owner, schema,
            "Database/migrations/045_campaign_operations_foundation.sql");
        ApplyFile(owner, schema,
            "Database/migrations/045_campaign_operations_foundation.sql");
        ApplyFile(owner, schema,
            "Database/migrations/047_campaign_operations_budget_request_acceptance.sql");
        ApplyFile(owner, schema,
            "Database/migrations/047_campaign_operations_budget_request_acceptance.sql");
        ApplyFile(owner, schema,
            "Tests/CampaignOperationsMigrationTests.sql");
        ApplyFile(owner, schema,
            "Tests/CampaignOperationsPhase2MigrationTests.sql");

        const OperationalCampaign campaign =
            InsertMaterialization(owner, schema, 41, 3);
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            SetCampaignCreatorRole(transaction);
            assert(SchemaExists(transaction));
            const auto recorded = PersistOperationalCampaign(transaction,
                campaign, ActorIdentity("creator@example.test"),
                Reason("Create exact Phase 4D operational envelope."));
            assert(recorded.outcome == PersistOutcome::recorded);
            assert(recorded.persisted.campaign == campaign);
            assert(InitialAdministrativeState(recorded.persisted.campaign) ==
                AdministrativeCampaignState::awaitingOperationalAuthorization);
            transaction.commit();
        }
        assert(CountRows(owner, schema, "campaign_operations_campaign") == 1);
        assert(CountRows(owner, schema,
                   "campaign_operations_audit_reference_event") == 1);

        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            SetCampaignCreatorRole(transaction);
            const auto replay = PersistOperationalCampaign(transaction,
                campaign, ActorIdentity("different.creator"),
                Reason("A retry does not create another campaign."));
            assert(replay.outcome == PersistOutcome::existingIdentical);
            const auto byMaterialization =
                FindOperationalCampaignByMaterialization(transaction, 41);
            assert(byMaterialization);
            assert(byMaterialization->campaign == campaign);
            assert(ListOperationalCampaigns(transaction, 1).size() == 1U);
            transaction.commit();
        }
        assert(CountRows(owner, schema,
                   "campaign_operations_audit_reference_event") == 1);

        const OperationalCampaign changedPayload = BuildOperationalCampaign(
            campaign.materializationId, campaign.materializationContractVersion,
            campaign.materializationCanonicalText,
            campaign.materializationIdentityHash, campaign.memberCount + 1);
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            SetCampaignCreatorRole(transaction);
            bool conflicted = false;
            try
            {
                (void)PersistOperationalCampaign(transaction, changedPayload,
                    ActorIdentity("creator@example.test"),
                    Reason("Changed payload must conflict."));
            }
            catch (const Error& error)
            {
                conflicted = error.code() == ErrorCode::persistenceConflict &&
                    std::string(error.what()) ==
                        "campaign_operations_campaign_conflict";
            }
            assert(conflicted);
            transaction.abort();
        }

        const OperationalCampaign rollbackCampaign =
            InsertMaterialization(owner, schema, 42, 2);
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            SetCampaignCreatorRole(transaction);
            const auto result = PersistOperationalCampaign(transaction,
                rollbackCampaign, ActorIdentity("creator@example.test"),
                Reason("Rollback verification."));
            assert(result.outcome == PersistOutcome::recorded);
            transaction.abort();
        }
        assert(CountRows(owner, schema, "campaign_operations_campaign") == 1);
        assert(CountRows(owner, schema,
                   "campaign_operations_audit_reference_event") == 1);

        const OperationalCampaign concurrentCampaign =
            InsertMaterialization(owner, schema, 43, 2);
        std::atomic<bool> start{false};
        PersistOutcome firstOutcome = PersistOutcome::recorded;
        PersistOutcome secondOutcome = PersistOutcome::recorded;
        long long firstId = 0;
        long long secondId = 0;
        std::exception_ptr firstError;
        std::exception_ptr secondError;
        const auto persist = [&](PersistOutcome& outcome, long long& id,
                                 std::exception_ptr& error)
        {
            try
            {
                pqxx::connection connection{connectionString};
                while (!start.load(std::memory_order_acquire))
                    std::this_thread::yield();
                pqxx::work transaction{connection};
                SetSearchPath(transaction, schema);
                SetCampaignCreatorRole(transaction);
                const auto result = PersistOperationalCampaign(transaction,
                    concurrentCampaign, ActorIdentity("creator@example.test"),
                    Reason("Concurrent exact creation."));
                outcome = result.outcome;
                id = result.persisted.campaignId.value();
                transaction.commit();
            }
            catch (...)
            {
                error = std::current_exception();
            }
        };
        std::thread first(persist, std::ref(firstOutcome), std::ref(firstId),
            std::ref(firstError));
        std::thread second(persist, std::ref(secondOutcome),
            std::ref(secondId), std::ref(secondError));
        start.store(true, std::memory_order_release);
        first.join();
        second.join();
        if (firstError) std::rethrow_exception(firstError);
        if (secondError) std::rethrow_exception(secondError);
        assert(firstId == secondId);
        assert((firstOutcome == PersistOutcome::recorded &&
                   secondOutcome == PersistOutcome::existingIdentical) ||
            (secondOutcome == PersistOutcome::recorded &&
                firstOutcome == PersistOutcome::existingIdentical));
        assert(CountRows(owner, schema, "campaign_operations_campaign") == 2);
        assert(CountRows(owner, schema,
                   "campaign_operations_audit_reference_event") == 2);

        TestProvenanceAndPrerequisitePolicy(
            owner, connectionString, schema);
        TestConcurrentAuthorizationSuccessors<
            pqxx::isolation_level::read_committed>(
            owner, connectionString, schema, 46);
        TestConcurrentAuthorizationSuccessors<
            pqxx::isolation_level::repeatable_read>(
            owner, connectionString, schema, 50);
        TestOversizedCanonicals(owner, schema);
        TestBudgetReservationAndRequestAcceptance(
            owner, connectionString, schema);
        TestDirectCapabilityIntegrity(owner, connectionString, schema);
        TestPhase2CorrectionConcurrency(owner, connectionString, schema);

        const OperationalCampaignId campaignId = [&]
        {
            pqxx::read_transaction transaction{owner};
            SetSearchPath(transaction, schema);
            const auto persisted =
                FindOperationalCampaignByMaterialization(transaction, 41);
            assert(persisted);
            return persisted->campaignId;
        }();
        const OperationalAuthorizationEvent grant =
            BuildOperationalAuthorizationEvent(campaignId,
                campaign.identity.canonicalText(), std::nullopt, std::nullopt,
                std::nullopt, 1, AuthorizationEventKind::granted,
                campaign.actionKind, campaign.actionContractVersion,
                campaign.scopeKind, campaign.scopeContractVersion,
                PrerequisitePolicy::phase4dMaterializationOnlyV1,
                std::nullopt, std::nullopt, std::nullopt,
                kCampaignOperationsAuthorizationRole,
                ActorIdentity("authorizer@example.test"),
                Reason("Authorize the exact materialization envelope."),
                UtcTimestamp("2026-07-22T12:00:00.000000Z"), std::nullopt);
        std::optional<PersistedOperationalAuthorizationEvent> persistedGrant;
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            SetAuthorizerRole(transaction);
            const auto recorded =
                PersistOperationalAuthorizationEvent(transaction, grant);
            assert(recorded.outcome == PersistOutcome::recorded);
            assert(recorded.persisted.event == grant);
            persistedGrant.emplace(
                recorded.persisted.authorizationEventId,
                recorded.persisted.event, recorded.persisted.createdAt);
            transaction.commit();
        }
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            SetAuthorizerRole(transaction);
            const auto replay =
                PersistOperationalAuthorizationEvent(transaction, grant);
            assert(replay.outcome == PersistOutcome::existingIdentical);
            assert(replay.persisted.authorizationEventId ==
                persistedGrant->authorizationEventId);
            transaction.commit();
        }

        const OperationalAuthorizationEvent successorGrant =
            BuildOperationalAuthorizationEvent(campaignId,
                campaign.identity.canonicalText(),
                persistedGrant->authorizationEventId,
                persistedGrant->event.identity.canonicalText(),
                persistedGrant->event.identity.hash(), 2,
                AuthorizationEventKind::granted, campaign.actionKind,
                campaign.actionContractVersion, campaign.scopeKind,
                campaign.scopeContractVersion,
                PrerequisitePolicy::phase4dMaterializationOnlyV1,
                std::nullopt, std::nullopt, std::nullopt,
                kCampaignOperationsAuthorizationRole,
                ActorIdentity("successor.authorizer@example.test"),
                Reason("Supersede with a successor grant."),
                UtcTimestamp("2026-07-23T12:00:00.000000Z"), std::nullopt);
        std::optional<PersistedOperationalAuthorizationEvent>
            persistedSuccessor;
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            SetAuthorizerRole(transaction);
            const auto recorded = PersistOperationalAuthorizationEvent(
                transaction, successorGrant);
            assert(recorded.outcome == PersistOutcome::recorded);
            persistedSuccessor.emplace(
                recorded.persisted.authorizationEventId,
                recorded.persisted.event, recorded.persisted.createdAt);
            transaction.commit();
        }

        const OperationalAuthorizationEvent staleSuccessor =
            BuildOperationalAuthorizationEvent(campaignId,
                campaign.identity.canonicalText(),
                persistedGrant->authorizationEventId,
                persistedGrant->event.identity.canonicalText(),
                persistedGrant->event.identity.hash(), 2,
                AuthorizationEventKind::revoked, campaign.actionKind,
                campaign.actionContractVersion, campaign.scopeKind,
                campaign.scopeContractVersion,
                PrerequisitePolicy::phase4dMaterializationOnlyV1,
                std::nullopt, std::nullopt, std::nullopt,
                kCampaignOperationsAuthorizationRole,
                ActorIdentity("authorizer@example.test"),
                Reason("A competing successor must conflict."),
                UtcTimestamp("2026-07-23T12:00:00.000000Z"), std::nullopt);
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            SetAuthorizerRole(transaction);
            bool conflicted = false;
            try
            {
                (void)PersistOperationalAuthorizationEvent(
                    transaction, staleSuccessor);
            }
            catch (const Error& error)
            {
                conflicted = error.code() == ErrorCode::persistenceConflict;
            }
            assert(conflicted);
            transaction.abort();
        }

        const OperationalAuthorizationEvent revoke =
            BuildOperationalAuthorizationEvent(campaignId,
                campaign.identity.canonicalText(),
                persistedSuccessor->authorizationEventId,
                persistedSuccessor->event.identity.canonicalText(),
                persistedSuccessor->event.identity.hash(), 3,
                AuthorizationEventKind::revoked, campaign.actionKind,
                campaign.actionContractVersion, campaign.scopeKind,
                campaign.scopeContractVersion,
                PrerequisitePolicy::phase4dMaterializationOnlyV1,
                std::nullopt, std::nullopt, std::nullopt,
                kCampaignOperationsAuthorizationRole,
                ActorIdentity("authorizer@example.test"),
                Reason("Revoke operational authority."),
                UtcTimestamp("2026-07-24T12:00:00.000000Z"), std::nullopt);
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            SetAuthorizerRole(transaction);
            const auto recorded =
                PersistOperationalAuthorizationEvent(transaction, revoke);
            assert(recorded.outcome == PersistOutcome::recorded);
            const auto head = FindOperationalAuthorizationHead(transaction,
                campaignId, campaign.actionKind,
                campaign.actionContractVersion, campaign.scopeKind,
                campaign.scopeContractVersion);
            assert(head);
            assert(head->authorizationEventId ==
                recorded.persisted.authorizationEventId);
            assert(head->event.eventKind == AuthorizationEventKind::revoked);
            const auto chain = LoadOperationalAuthorizationChain(transaction,
                campaignId, campaign.actionKind,
                campaign.actionContractVersion, campaign.scopeKind,
                campaign.scopeContractVersion, 10);
            assert(chain.size() == 3U);
            assert(chain[0].event.chainVersion == 1);
            assert(chain[1].event.chainVersion == 2);
            assert(chain[2].event.chainVersion == 3);
            transaction.commit();
        }
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_authorization_event",
                   campaignId) == 3);
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_audit_reference_event",
                   campaignId) == 4);

        DropSchema(owner, schema);
    }
    catch (...)
    {
        try
        {
            DropSchema(owner, schema);
        }
        catch (...)
        {
        }
        throw;
    }
    return 0;
}
