#include "../Sources/CampaignOperationsRepository.hpp"

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
            "Tests/CampaignOperationsMigrationTests.sql");

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
