#include "../Sources/CampaignOperationsRepository.hpp"
#include "../Sources/CampaignOperationsService.hpp"
#include "../Sources/CampaignOperationsDispatchRepository.hpp"
#include "../Sources/CampaignOperationsControlRepository.hpp"
#include "../Sources/CampaignOperationsControlService.hpp"
#include "../Sources/CampaignOperationsCompletionRepository.hpp"
#include "../Sources/CampaignOperationsCompletionService.hpp"

#include "../Sources/ExperimentRecommendation.hpp"
#include "../Sources/ExperimentRecommendationCampaignFollowUpProposalRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignMaterializationRepository.hpp"

#include <atomic>
#include <array>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
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

std::vector<PersistedRecommendationCampaignMaterialization>
ListRecommendationCampaignMaterializations(
    pqxx::transaction_base&, std::optional<long long>, int)
{
    // The focused Campaign Operations suite never lists upstream
    // materializations; this adapter satisfies the handoff repository's
    // link-time surface without granting that unrelated behavior authority.
    return {};
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
CREATE TABLE experiment(
    experiment_id bigserial PRIMARY KEY,
    symbol text NOT NULL DEFAULT 'TEST',
    prediction_horizon integer NOT NULL DEFAULT 1,
    c_next_threshold double precision NOT NULL DEFAULT 0,
    core_lr_mult double precision,
    head_lr_mult double precision,
    target_epochs integer NOT NULL DEFAULT 1,
    checkpoint_interval integer NOT NULL DEFAULT 20,
    train_start timestamptz NOT NULL DEFAULT now(),
    train_end timestamptz NOT NULL DEFAULT now(),
    infer_start timestamptz,
    infer_end timestamptz,
    status text NOT NULL DEFAULT 'paused',
    phase text NOT NULL DEFAULT 'train',
    invocation_mode text,
    resume_model_id bigint,
    duplicate_nonce bigint NOT NULL DEFAULT 0,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE experiment_recommendation_conversion_proposal(
    recommendation_conversion_proposal_id bigint PRIMARY KEY);
CREATE TABLE experiment_recommendation_conversion_review_decision(
    recommendation_conversion_review_decision_id bigint PRIMARY KEY);
CREATE TABLE experiment_recommendation_conversion_execution(
    recommendation_conversion_execution_id bigserial PRIMARY KEY,
    recommendation_conversion_proposal_id bigint NOT NULL UNIQUE,
    recommendation_conversion_review_decision_id bigint NOT NULL,
    experiment_id bigint NOT NULL UNIQUE,
    execution_contract_version integer NOT NULL,
    authorization_decision text NOT NULL,
    execution_identity_canonical text NOT NULL,
    execution_identity_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE experiment_recommendation_conversion_activation(
    recommendation_conversion_activation_id bigserial PRIMARY KEY,
    recommendation_conversion_execution_id bigint NOT NULL UNIQUE,
    recommendation_conversion_proposal_id bigint NOT NULL,
    recommendation_conversion_review_decision_id bigint NOT NULL,
    experiment_id bigint NOT NULL UNIQUE,
    activation_contract_version integer NOT NULL,
    previous_status text NOT NULL,
    previous_phase text NOT NULL,
    resulting_status text NOT NULL,
    resulting_phase text NOT NULL,
    activation_identity_canonical text NOT NULL,
    activation_identity_hash text NOT NULL,
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

long long Scalar(pqxx::connection& connection, const std::string& schema,
    const std::string& query)
{
    pqxx::read_transaction transaction{connection};
    SetSearchPath(transaction, schema);
    return transaction.exec(query).one_row()[0].as<long long>();
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

void TestPhase3LeaseAcquisition(pqxx::connection& owner,
    const std::string& connectionString, const std::string& schema)
{
    const std::string budgetConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_budget_administrator'";
    const std::string requestConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_request_acceptor'";
    const auto fixture = CreatePhase2AcceptanceFixture(
        owner, budgetConnectionString, schema, 190, 2);
    pqxx::connection acceptor{requestConnectionString};
    const auto accepted = AcceptOperationalRequest(acceptor,
        {fixture.campaign.campaignId.value(),
            "phase3.requester@example.test",
            "Accept exact Phase 3 lease fixture.", std::nullopt});
    const auto requestId = accepted.persisted.request.requestId;
    const LeaseTokenDigest digest = LeaseTokenDigest::Derive(
        "phase3-lease-token-0123456789abcdef");
    {
        pqxx::read_transaction first{owner};
        SetSearchPath(first, schema);
        first.exec("SET LOCAL ROLE campaign_operations_dispatcher;");
        const auto firstCandidates =
            SelectDispatchCandidatesForIsolatedTest(first, 100);
        pqxx::connection secondConnection{connectionString};
        pqxx::read_transaction second{secondConnection};
        SetSearchPath(second, schema);
        second.exec("SET LOCAL ROLE campaign_operations_dispatcher;");
        const auto secondCandidates =
            SelectDispatchCandidatesForIsolatedTest(second, 100);
        assert(std::is_sorted(
            firstCandidates.begin(), firstCandidates.end(),
            [](OperationalRequestId left, OperationalRequestId right)
            {
                return left.value() < right.value();
            }));
        assert(std::find(firstCandidates.begin(), firstCandidates.end(),
                   requestId) != firstCandidates.end());
        assert(std::find(secondCandidates.begin(), secondCandidates.end(),
                   requestId) != secondCandidates.end());
    }

    DispatchLease lease = [&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_dispatcher;");
        const auto acquired = AcquireDispatchLeaseInTransaction(
            transaction, requestId, 1, digest,
            ActorIdentity("phase3.dispatcher@example.test"));
        transaction.commit();
        return acquired;
    }();
    assert(lease.acquisition.attemptOrdinal == 1);
    assert(lease.acquisition.expectedRequestVersion == 1);
    assert(lease.acquisition.resultingRequestVersion == 2);
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        const auto row = transaction.exec(
            "SELECT request_state,state_version,lease_token_hash,"
            "dispatcher_identity,production_dispatch_enabled "
            "FROM campaign_operations_operational_request "
            "WHERE operational_request_id=$1;",
            pqxx::params{requestId.value()}).one_row();
        assert(row[0].as<std::string>() == "dispatching");
        assert(row[1].as<int>() == 2);
        assert(row[2].as<std::string>() == digest.value());
        assert(row[2].as<std::string>().find("phase3-lease-token") ==
            std::string::npos);
        assert(row[3].as<std::string>() ==
            "phase3.dispatcher@example.test");
        assert(!row[4].as<bool>());
        assert(transaction.exec(
            "SELECT count(*) FROM campaign_operations_dispatch_attempt "
            "WHERE operational_request_id=$1;",
            pqxx::params{requestId.value()}).one_row()[0].as<int>() == 1);
    }
    {
        bool lost = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec(
                "SET LOCAL ROLE campaign_operations_dispatcher;");
            (void)AcquireDispatchLeaseInTransaction(
                transaction, requestId, 1, digest,
                ActorIdentity("phase3.loser@example.test"));
            transaction.commit();
        }
        catch (const std::exception&)
        {
            lost = true;
        }
        assert(lost);
    }

    const auto rollbackFixture = CreatePhase2AcceptanceFixture(
        owner, budgetConnectionString, schema, 191, 1);
    const auto rollbackAccepted = AcceptOperationalRequest(acceptor,
        {rollbackFixture.campaign.campaignId.value(),
            "phase3.rollback@example.test",
            "Accept exact rollback lease fixture.", std::nullopt});
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_dispatcher;");
        (void)AcquireDispatchLeaseInTransaction(transaction,
            rollbackAccepted.persisted.request.requestId, 1,
            LeaseTokenDigest::Derive(
                "phase3-rollback-token-0123456789abc"),
            ActorIdentity("phase3.dispatcher@example.test"));
        transaction.abort();
    }
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        const auto row = transaction.exec(
            "SELECT request_state,state_version,lease_token_hash IS NULL "
            "FROM campaign_operations_operational_request "
            "WHERE operational_request_id=$1;",
            pqxx::params{
                rollbackAccepted.persisted.request.requestId.value()})
            .one_row();
        assert(row[0].as<std::string>() == "ready");
        assert(row[1].as<int>() == 1);
        assert(row[2].as<bool>());
        assert(transaction.exec(
            "SELECT count(*) FROM campaign_operations_dispatch_attempt "
            "WHERE operational_request_id=$1;",
            pqxx::params{
                rollbackAccepted.persisted.request.requestId.value()})
            .one_row()[0].as<int>() == 0);
    }

    const auto contentionFixture = CreatePhase2AcceptanceFixture(
        owner, budgetConnectionString, schema, 192, 1);
    const auto contentionAccepted = AcceptOperationalRequest(acceptor,
        {contentionFixture.campaign.campaignId.value(),
            "phase3.contention@example.test",
            "Accept exact contention lease fixture.", std::nullopt});
    const auto contentionRequestId =
        contentionAccepted.persisted.request.requestId;
    std::barrier start{3};
    std::atomic<int> winners{0};
    std::atomic<int> losers{0};
    auto acquire = [&](const char* token, const char* actor)
    {
        pqxx::connection connection{connectionString};
        start.arrive_and_wait();
        try
        {
            pqxx::work transaction{connection};
            SetSearchPath(transaction, schema);
            transaction.exec(
                "SET LOCAL ROLE campaign_operations_dispatcher;");
            (void)AcquireDispatchLeaseInTransaction(transaction,
                contentionRequestId, 1, LeaseTokenDigest::Derive(token),
                ActorIdentity(actor));
            transaction.commit();
            ++winners;
        }
        catch (const std::exception&)
        {
            ++losers;
        }
    };
    std::thread firstAcquirer{acquire,
        "phase3-contention-token-aaaaaaaaaaaaaaaa",
        "phase3.first@example.test"};
    std::thread secondAcquirer{acquire,
        "phase3-contention-token-bbbbbbbbbbbbbbbb",
        "phase3.second@example.test"};
    start.arrive_and_wait();
    firstAcquirer.join();
    secondAcquirer.join();
    assert(winners == 1);
    assert(losers == 1);
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        const auto row = transaction.exec(
            "SELECT request_state,state_version,"
            "(SELECT count(*) FROM campaign_operations_dispatch_attempt "
            "WHERE operational_request_id=$1),"
            "(SELECT min(attempt_ordinal) FROM "
            "campaign_operations_dispatch_attempt "
            "WHERE operational_request_id=$1) "
            "FROM campaign_operations_operational_request "
            "WHERE operational_request_id=$1;",
            pqxx::params{contentionRequestId.value()}).one_row();
        assert(row[0].as<std::string>() == "dispatching");
        assert(row[1].as<int>() == 2);
        assert(row[2].as<int>() == 1);
        assert(row[3].as<int>() == 1);
    }
}

PersistResult<AcceptedOperationalRequest> AcceptPhase4Fixture(
    pqxx::connection& owner, const std::string& connectionString,
    const std::string& schema, long long materializationId,
    int memberCount,
    std::optional<std::string> expiresAt = std::nullopt)
{
    const std::string budgetConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_budget_administrator'";
    const std::string requestConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_request_acceptor'";
    const auto fixture = CreatePhase2AcceptanceFixture(owner,
        budgetConnectionString, schema, materializationId, memberCount);
    pqxx::connection acceptor{requestConnectionString};
    return AcceptOperationalRequest(acceptor,
        {fixture.campaign.campaignId.value(),
            "phase4.requester@example.test",
            "Accept exact Phase F control fixture.",
            std::move(expiresAt)});
}

void ExpireDispatchLeaseForTest(pqxx::connection& owner,
    const std::string& schema, OperationalRequestId requestId,
    bool shortenWhileActive = false)
{
    const auto setAcquisitionTrigger = [&](const char* state)
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec("SET LOCAL ROLE campaign_operations_owner;");
        transaction.exec(
            std::string(
                "ALTER TABLE campaign_operations_operational_request ") +
            state + " TRIGGER "
            "campaign_operations_dispatch_acquisition_complete_trigger;");
        transaction.commit();
    };
    setAcquisitionTrigger("DISABLE");
    try
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec("SET LOCAL ROLE campaign_operations_owner;");
        transaction.exec(
            "UPDATE campaign_operations_operational_request "
            "SET lease_expires_at=transaction_timestamp()" +
            std::string(shortenWhileActive ? "+" : "-") +
            "interval '" +
            std::string(shortenWhileActive ? "5 seconds" : "1 second") +
            "' "
            "WHERE operational_request_id=$1;",
            pqxx::params{requestId.value()});
        transaction.exec(
            "UPDATE campaign_operations_dispatch_attempt "
            "SET lease_expires_at=transaction_timestamp()" +
            std::string(shortenWhileActive ? "+" : "-") +
            "interval '" +
            std::string(shortenWhileActive ? "5 seconds" : "1 second") +
            "' "
            "WHERE operational_request_id=$1;",
            pqxx::params{requestId.value()});
        transaction.commit();
    }
    catch (...)
    {
        setAcquisitionTrigger("ENABLE");
        throw;
    }
    setAcquisitionTrigger("ENABLE");
}

bool WaitForApplicationLockWait(
    pqxx::connection& owner, const std::string& applicationName)
{
    for (int attempt = 0; attempt < 2500; ++attempt)
    {
        pqxx::read_transaction transaction{owner};
        const bool waiting = transaction.exec(
            "SELECT EXISTS(SELECT 1 FROM pg_stat_activity "
            "WHERE application_name=$1 AND wait_event_type='Lock');",
            pqxx::params{applicationName}).one_row()[0].as<bool>();
        if (waiting) return true;
        transaction.abort();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
}

void TestPhase4ControlsCancellationAndReconciliation(
    pqxx::connection& owner, const std::string& connectionString,
    const std::string& schema)
{
    const std::string runtimeConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c lock_timeout=5s -c statement_timeout=15s'";
    const std::string budgetConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_budget_administrator'";
    const std::string requestConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_request_acceptor'";

    const auto acceptanceGateFixture = CreatePhase2AcceptanceFixture(
        owner, budgetConnectionString, schema, 209, 1);
    pqxx::connection controlConnection{runtimeConnectionString};
    CampaignControlRequest gatePause{
        acceptanceGateFixture.campaign.campaignId.value(), 0,
        ControlEventKind::pause, "phase4.operator@example.test",
        "Pause before request acceptance."};
    (void)ControlCampaign(controlConnection, gatePause);
    bool pausedAcceptanceBlocked = false;
    try
    {
        pqxx::connection acceptor{requestConnectionString};
        (void)AcceptOperationalRequest(acceptor,
            {acceptanceGateFixture.campaign.campaignId.value(),
                "phase4.requester@example.test",
                "Paused acceptance must fail.", std::nullopt});
    }
    catch (const Error& error)
    {
        pausedAcceptanceBlocked =
            error.code() == ErrorCode::persistenceConflict;
    }
    assert(pausedAcceptanceBlocked);
    CampaignControlRequest gateResume{
        acceptanceGateFixture.campaign.campaignId.value(), 1,
        ControlEventKind::resume, "phase4.operator@example.test",
        "Resume before request acceptance."};
    (void)ControlCampaign(controlConnection, gateResume);
    {
        pqxx::connection acceptor{requestConnectionString};
        assert(AcceptOperationalRequest(acceptor,
                   {acceptanceGateFixture.campaign.campaignId.value(),
                       "phase4.requester@example.test",
                       "Resumed acceptance must succeed.", std::nullopt})
                   .outcome == PersistOutcome::recorded);
    }

    const auto pausedAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 210, 1);
    const auto pausedCampaignId =
        pausedAccepted.persisted.request.request.logicalOperation.campaignId;
    CampaignControlRequest pause{pausedCampaignId.value(), 0,
        ControlEventKind::pause, "phase4.operator@example.test",
        "Pause only future Campaign Operations actions."};
    const auto paused = ControlCampaign(controlConnection, pause);
    assert(paused.replay == ControlReplayDisposition::recorded);
    const auto pauseReplay = ControlCampaign(controlConnection, pause);
    assert(pauseReplay.replay ==
        ControlReplayDisposition::existingIdentical);
    assert(pauseReplay.persisted.controlEventId ==
        paused.persisted.controlEventId);
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_dispatcher;");
        const auto candidates =
            SelectDispatchCandidatesForIsolatedTest(transaction, 100);
        assert(std::find(candidates.begin(), candidates.end(),
                   pausedAccepted.persisted.request.requestId) ==
            candidates.end());
    }
    bool changedReplayConflicted = false;
    try
    {
        CampaignControlRequest changed = pause;
        changed.reason = "A changed duplicate must conflict.";
        (void)ControlCampaign(controlConnection, changed);
    }
    catch (const Error& error)
    {
        changedReplayConflicted =
            error.code() == ErrorCode::persistenceConflict;
    }
    assert(changedReplayConflicted);
    CampaignControlRequest resume{pausedCampaignId.value(), 1,
        ControlEventKind::resume, "phase4.operator@example.test",
        "Resume future Campaign Operations actions."};
    const auto resumed = ControlCampaign(controlConnection, resume);
    assert(resumed.replay == ControlReplayDisposition::recorded);
    assert(ControlCampaign(controlConnection, resume).replay ==
        ControlReplayDisposition::existingIdentical);
    const auto controlStatus =
        LoadCampaignControlStatus(controlConnection, pausedCampaignId);
    assert(!controlStatus.paused);
    assert(controlStatus.controlVersion == 2);
    CampaignCancellationCommandRequest statusSnapshotCancellation;
    statusSnapshotCancellation.campaignId = pausedCampaignId.value();
    statusSnapshotCancellation.requestId =
        pausedAccepted.persisted.request.requestId.value();
    statusSnapshotCancellation.expectedRequestVersion = 1;
    statusSnapshotCancellation.operationKey =
        "phase4-status-repeatable-read";
    statusSnapshotCancellation.actorIdentity =
        "phase4.operator@example.test";
    statusSnapshotCancellation.reason =
        "Commit cancellation between status projection reads.";
    bool statusSnapshotMutationCommitted = false;
    const auto statusBeforeConcurrentCancellation =
        LoadCampaignControlStatus(controlConnection, pausedCampaignId,
            [&](CampaignOperationsControlTestInjectionPoint point,
                pqxx::transaction_base& transaction)
            {
                if (point !=
                    CampaignOperationsControlTestInjectionPoint::
                        afterStatusControlReadBeforeCancellationRead)
                    return;
                const auto settings = transaction.exec(
                    "SELECT current_setting('transaction_isolation'),"
                    "current_setting('transaction_read_only');").one_row();
                assert(settings[0].as<std::string>() == "repeatable read");
                assert(settings[1].as<std::string>() == "on");
                const auto concurrent = CancelCampaign(
                    runtimeConnectionString, statusSnapshotCancellation);
                assert(concurrent.progress ==
                    CancellationProgress::settled);
                statusSnapshotMutationCommitted = true;
            });
    assert(statusSnapshotMutationCommitted);
    assert(!statusBeforeConcurrentCancellation.cancellationRequested);
    const auto statusAfterConcurrentCancellation =
        LoadCampaignControlStatus(controlConnection, pausedCampaignId);
    assert(statusAfterConcurrentCancellation.cancellationRequested);
    assert(statusAfterConcurrentCancellation.cancellationSettled);

    const auto rollbackAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 211, 1);
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_controller;");
        const auto rollbackCampaign =
            FindOperationalCampaign(transaction,
                rollbackAccepted.persisted.request.request.logicalOperation
                    .campaignId);
        assert(rollbackCampaign);
        const auto event = BuildCampaignControlEvent(
            rollbackCampaign->campaignId,
            rollbackCampaign->campaign.identity.canonicalText(),
            std::nullopt, std::nullopt, 1, ControlEventKind::pause,
            ActorIdentity("phase4.rollback@example.test"),
            Reason("Rollback must leave no partial control evidence."));
        const auto persisted =
            PersistCampaignControlEvent(transaction, event);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_owner;");
        transaction.exec(
            "DELETE FROM campaign_operations_control_audit_reference_event "
            "WHERE control_event_id=$1;",
            pqxx::params{persisted.controlEventId.value()});
        bool completenessRejected = false;
        try
        {
            transaction.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            completenessRejected = error.sqlstate() == "23514";
        }
        assert(completenessRejected);
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_control_event",
               rollbackAccepted.persisted.request.request.logicalOperation
                   .campaignId) == 0);
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE "
            "campaign_operations_cancellation_coordinator;");
        transaction.exec(
            "SELECT transition_campaign_operations_request_cancelled("
            "$1,1);",
            pqxx::params{
                rollbackAccepted.persisted.request.requestId.value()});
        bool incompleteTransitionRejected = false;
        try
        {
            transaction.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            incompleteTransitionRejected = error.sqlstate() == "23514";
        }
        assert(incompleteTransitionRejected);
    }
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        assert(transaction.exec(
            "SELECT request_state||':'||state_version::text "
            "FROM campaign_operations_operational_request "
            "WHERE operational_request_id=$1;",
            pqxx::params{
                rollbackAccepted.persisted.request.requestId.value()})
                   .one_row()[0]
                   .as<std::string>() == "ready:1");
    }

    const auto cancellationAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 212, 2);
    CampaignCancellationCommandRequest cancellation;
    cancellation.campaignId =
        cancellationAccepted.persisted.request.request.logicalOperation
            .campaignId.value();
    cancellation.requestId =
        cancellationAccepted.persisted.request.requestId.value();
    cancellation.expectedRequestVersion = 1;
    cancellation.operationKey = "phase4-ready-cancel";
    cancellation.actorIdentity = "phase4.operator@example.test";
    cancellation.reason =
        "Cancel the unbound request and release its held units.";
    const auto cancelled =
        CancelCampaign(runtimeConnectionString, cancellation);
    assert(cancelled.replay == ControlReplayDisposition::recorded);
    assert(cancelled.progress == CancellationProgress::settled);
    assert(cancelled.settlement);
    assert(cancelled.settlement->settlement.disposition ==
        CancellationSettlementDisposition::unboundCancelled);
    const auto cancelledReplay =
        CancelCampaign(runtimeConnectionString, cancellation);
    assert(cancelledReplay.replay ==
        ControlReplayDisposition::existingIdentical);
    assert(cancelledReplay.progress == CancellationProgress::settled);
    assert(cancelledReplay.settlement);
    assert(cancelledReplay.settlement->cancellationSettlementId ==
        cancelled.settlement->cancellationSettlementId);
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        const auto row = transaction.exec(
            "SELECT request.request_state,request.state_version,"
            "reservation.reservation_state,reservation.state_version "
            "FROM campaign_operations_operational_request request "
            "JOIN campaign_operations_reservation reservation "
            "ON reservation.reservation_id=request.reservation_id "
            "WHERE request.operational_request_id=$1;",
            pqxx::params{
                cancellationAccepted.persisted.request.requestId.value()})
            .one_row();
        assert(row[0].as<std::string>() == "cancelled");
        assert(row[1].as<int>() == 2);
        assert(row[2].as<std::string>() == "released");
        assert(row[3].as<int>() == 2);
    }

    const auto concurrentCancellationAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 215, 1);
    CampaignCancellationCommandRequest concurrentCancellation;
    concurrentCancellation.campaignId =
        concurrentCancellationAccepted.persisted.request.request
            .logicalOperation.campaignId.value();
    concurrentCancellation.requestId =
        concurrentCancellationAccepted.persisted.request.requestId.value();
    concurrentCancellation.expectedRequestVersion = 1;
    concurrentCancellation.operationKey =
        "phase4-concurrent-identical-cancel";
    concurrentCancellation.actorIdentity =
        "phase4.operator@example.test";
    concurrentCancellation.reason =
        "Concurrent exact cancellation calls must converge.";
    std::barrier concurrentCancellationStart{3};
    std::optional<CampaignCancellationResult> firstConcurrentCancellation;
    std::optional<CampaignCancellationResult> secondConcurrentCancellation;
    std::exception_ptr firstConcurrentCancellationError;
    std::exception_ptr secondConcurrentCancellationError;
    const auto cancelConcurrently =
        [&](std::optional<CampaignCancellationResult>& result,
            std::exception_ptr& error)
        {
            concurrentCancellationStart.arrive_and_wait();
            try
            {
                result.emplace(CancelCampaign(
                    runtimeConnectionString, concurrentCancellation));
            }
            catch (...)
            {
                error = std::current_exception();
            }
        };
    std::thread firstConcurrentCancellationThread(cancelConcurrently,
        std::ref(firstConcurrentCancellation),
        std::ref(firstConcurrentCancellationError));
    std::thread secondConcurrentCancellationThread(cancelConcurrently,
        std::ref(secondConcurrentCancellation),
        std::ref(secondConcurrentCancellationError));
    concurrentCancellationStart.arrive_and_wait();
    firstConcurrentCancellationThread.join();
    secondConcurrentCancellationThread.join();
    if (firstConcurrentCancellationError)
        std::rethrow_exception(firstConcurrentCancellationError);
    if (secondConcurrentCancellationError)
        std::rethrow_exception(secondConcurrentCancellationError);
    assert(firstConcurrentCancellation);
    assert(secondConcurrentCancellation);
    assert(firstConcurrentCancellation->progress ==
        CancellationProgress::settled);
    assert(secondConcurrentCancellation->progress ==
        CancellationProgress::settled);
    assert(firstConcurrentCancellation->settlement);
    assert(secondConcurrentCancellation->settlement);
    assert(firstConcurrentCancellation->request.cancellationRequestId ==
        secondConcurrentCancellation->request.cancellationRequestId);
    assert(firstConcurrentCancellation->settlement->
            cancellationSettlementId ==
        secondConcurrentCancellation->settlement->
            cancellationSettlementId);

    const auto retriedCancellationAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 217, 1);
    CampaignCancellationCommandRequest retriedCancellationCommand;
    retriedCancellationCommand.campaignId =
        retriedCancellationAccepted.persisted.request.request
            .logicalOperation.campaignId.value();
    retriedCancellationCommand.requestId =
        retriedCancellationAccepted.persisted.request.requestId.value();
    retriedCancellationCommand.expectedRequestVersion = 1;
    retriedCancellationCommand.operationKey =
        "phase4-cancellation-intent-retry";
    retriedCancellationCommand.actorIdentity =
        "phase4.operator@example.test";
    retriedCancellationCommand.reason =
        "Retry the whole cancellation transaction after intent insertion.";
    int cancellationIntentAttempts = 0;
    const auto retriedCancellation = CancelCampaign(
        runtimeConnectionString, retriedCancellationCommand,
        [&](CampaignOperationsControlTestInjectionPoint point,
            pqxx::transaction_base& transaction)
        {
            if (point != CampaignOperationsControlTestInjectionPoint::
                    afterCancellationIntentInsertion)
                return;
            ++cancellationIntentAttempts;
            if (cancellationIntentAttempts == 1)
                transaction.exec(
                    "DO $phase4_retry$ BEGIN RAISE EXCEPTION "
                    "'deterministic Phase 4 cancellation retry' "
                    "USING ERRCODE='40P01'; END $phase4_retry$;");
        });
    assert(cancellationIntentAttempts == 2);
    assert(retriedCancellation.replay ==
        ControlReplayDisposition::recorded);
    assert(retriedCancellation.progress ==
        CancellationProgress::settled);
    assert(retriedCancellation.settlement);
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        const auto row = transaction.exec(
            "SELECT count(*),count(settlement.cancellation_settlement_id),"
            "count(DISTINCT cancellation.cancellation_request_id),"
            "count(DISTINCT settlement.cancellation_settlement_id) "
            "FROM campaign_operations_cancellation_request cancellation "
            "LEFT JOIN campaign_operations_cancellation_settlement settlement "
            "USING(cancellation_request_id) "
            "WHERE cancellation.operation_key=$1;",
            pqxx::params{retriedCancellationCommand.operationKey}).one_row();
        assert(row[0].as<int>() == 1);
        assert(row[1].as<int>() == 1);
        assert(row[2].as<int>() == 1);
        assert(row[3].as<int>() == 1);
        assert(transaction.exec(
            "SELECT count(*) FROM campaign_operations_reservation_event "
            "WHERE cancellation_request_id=$1;",
            pqxx::params{
                retriedCancellation.request.cancellationRequestId.value()})
            .one_row()[0].as<int>() == 1);
    }

    const auto leasedAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 213, 1);
    const auto leasedRequestId =
        leasedAccepted.persisted.request.requestId;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_dispatcher;");
        (void)AcquireDispatchLeaseInTransaction(transaction,
            leasedRequestId, 1,
            LeaseTokenDigest::Derive(
                "phase4-cancellation-lease-token-0001"),
            ActorIdentity("phase4.dispatcher@example.test"));
        transaction.commit();
    }
    ExpireDispatchLeaseForTest(
        owner, schema, leasedRequestId, true);
    CampaignCancellationCommandRequest leasedCancellation;
    leasedCancellation.campaignId =
        leasedAccepted.persisted.request.request.logicalOperation
            .campaignId.value();
    leasedCancellation.requestId = leasedRequestId.value();
    leasedCancellation.expectedRequestVersion = 2;
    leasedCancellation.operationKey = "phase4-leased-cancel";
    leasedCancellation.actorIdentity = "phase4.operator@example.test";
    leasedCancellation.reason =
        "Persist intent without stealing active dispatch ownership.";
    const auto waiting =
        CancelCampaign(runtimeConnectionString, leasedCancellation);
    assert(waiting.progress ==
        CancellationProgress::waitingForLeaseExpiry);
    assert(!waiting.settlement);
    assert(CancelCampaign(runtimeConnectionString, leasedCancellation)
               .progress ==
        CancellationProgress::waitingForLeaseExpiry);
    auto overlappingCancellation = leasedCancellation;
    overlappingCancellation.operationKey =
        "phase4-leased-overlapping-cancel";
    overlappingCancellation.reason =
        "A distinct operation cannot acquire duplicate cancellation ownership.";
    bool overlappingCancellationRejected = false;
    try
    {
        (void)CancelCampaign(
            runtimeConnectionString, overlappingCancellation);
    }
    catch (const Error& error)
    {
        overlappingCancellationRejected =
            error.code() == ErrorCode::persistenceConflict &&
            std::string(error.what()) ==
                "campaign_operations_cancellation_target_already_owned";
    }
    assert(overlappingCancellationRejected);
    assert(Scalar(owner, schema,
        "SELECT count(*) FROM campaign_operations_cancellation_request "
        "WHERE operational_request_id=" +
        std::to_string(leasedRequestId.value())) == 1);
    ReconciliationObserveRequest observeActiveLeaseCancellation;
    observeActiveLeaseCancellation.runKey =
        "phase4-active-lease-cancellation-observation";
    observeActiveLeaseCancellation.afterRequestId =
        leasedRequestId.value() - 1;
    observeActiveLeaseCancellation.limit = 1;
    const auto activeLeaseCancellationObservation =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, observeActiveLeaseCancellation);
    assert(activeLeaseCancellationObservation.selectedCount == 1);
    assert(activeLeaseCancellationObservation.observationCount == 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(5200));
    std::barrier cancellationReplayStart{3};
    std::optional<CampaignCancellationResult> firstCancellationReplay;
    std::optional<CampaignCancellationResult> secondCancellationReplay;
    std::exception_ptr firstCancellationReplayError;
    std::exception_ptr secondCancellationReplayError;
    const auto replayCancellation =
        [&](std::optional<CampaignCancellationResult>& result,
            std::exception_ptr& error)
        {
            cancellationReplayStart.arrive_and_wait();
            try
            {
                result.emplace(CancelCampaign(
                    runtimeConnectionString, leasedCancellation));
            }
            catch (...)
            {
                error = std::current_exception();
            }
        };
    std::thread firstCancellationReplayThread(replayCancellation,
        std::ref(firstCancellationReplay),
        std::ref(firstCancellationReplayError));
    std::thread secondCancellationReplayThread(replayCancellation,
        std::ref(secondCancellationReplay),
        std::ref(secondCancellationReplayError));
    cancellationReplayStart.arrive_and_wait();
    firstCancellationReplayThread.join();
    secondCancellationReplayThread.join();
    if (firstCancellationReplayError)
        std::rethrow_exception(firstCancellationReplayError);
    if (secondCancellationReplayError)
        std::rethrow_exception(secondCancellationReplayError);
    assert(firstCancellationReplay);
    assert(secondCancellationReplay);
    assert(firstCancellationReplay->replay ==
        ControlReplayDisposition::existingIdentical);
    assert(secondCancellationReplay->replay ==
        ControlReplayDisposition::existingIdentical);
    assert(firstCancellationReplay->progress ==
        CancellationProgress::settled);
    assert(secondCancellationReplay->progress ==
        CancellationProgress::settled);
    assert(firstCancellationReplay->request.cancellationRequestId ==
        waiting.request.cancellationRequestId);
    assert(secondCancellationReplay->request.cancellationRequestId ==
        waiting.request.cancellationRequestId);
    assert(firstCancellationReplay->settlement);
    assert(secondCancellationReplay->settlement);
    assert(firstCancellationReplay->settlement->
            cancellationSettlementId ==
        secondCancellationReplay->settlement->
            cancellationSettlementId);
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        assert(transaction.exec(
            "SELECT count(*) FROM "
            "campaign_operations_cancellation_request "
            "WHERE operation_key=$1;",
            pqxx::params{leasedCancellation.operationKey})
            .one_row()[0].as<int>() == 1);
        assert(transaction.exec(
            "SELECT count(*) FROM "
            "campaign_operations_cancellation_settlement "
            "WHERE cancellation_request_id=$1;",
            pqxx::params{waiting.request.cancellationRequestId.value()})
            .one_row()[0].as<int>() == 1);
        const auto resolution = transaction.exec(
            "SELECT count(*),"
            "min(resolution.owning_capability),"
            "min(resolution.resolution_disposition),"
            "min(resolution.transition_identity_hash),"
            "bool_and(resolution.cancellation_settlement_id IS NOT NULL),"
            "bool_and(resolution.dispatch_attempt_outcome_id IS NULL) "
            "FROM campaign_operations_reconciliation_observation observation "
            "LEFT JOIN campaign_operations_reconciliation_resolution resolution "
            "USING(reconciliation_observation_id) "
            "WHERE observation.operational_request_id=$1 "
            "AND observation.reason_code="
            "'cancellation_settlement_pending';",
            pqxx::params{leasedRequestId.value()}).one_row();
        assert(resolution[0].as<int>() == 1);
        assert(resolution[1].as<std::string>() ==
            "campaign_operations_cancellation_coordinator");
        assert(resolution[2].as<std::string>() ==
            "cancellation_settled");
        assert(resolution[3].as<std::string>() ==
            firstCancellationReplay->settlement->settlement.identity.hash());
        assert(resolution[4].as<bool>());
        assert(resolution[5].as<bool>());
        assert(transaction.exec(
            "SELECT count(*) FROM "
            "campaign_operations_reconciliation_observation observation "
            "LEFT JOIN campaign_operations_reconciliation_resolution resolution "
            "USING(reconciliation_observation_id) "
            "WHERE observation.operational_request_id=$1 "
            "AND observation.reason_code="
            "'cancellation_settlement_pending' "
            "AND resolution.reconciliation_resolution_id IS NULL;",
            pqxx::params{leasedRequestId.value()})
            .one_row()[0].as<int>() == 0);
    }

    const auto expiredAmbiguousAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 224, 1);
    const auto expiredAmbiguousRequestId =
        expiredAmbiguousAccepted.persisted.request.requestId;
    DispatchLease expiredAmbiguousLease = [&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_dispatcher;");
        auto lease = AcquireDispatchLeaseInTransaction(transaction,
            expiredAmbiguousRequestId, 1,
            LeaseTokenDigest::Derive(
                "phase4-expired-ambiguous-outcome-token"),
            ActorIdentity("phase4.dispatcher@example.test"));
        transaction.commit();
        return lease;
    }();
    ExpireDispatchLeaseForTest(
        owner, schema, expiredAmbiguousRequestId);
    {
        const auto outcome = BuildDispatchAttemptOutcomeEvidence(
            expiredAmbiguousLease.attemptId,
            expiredAmbiguousLease.acquisition.identity.canonicalText(),
            DispatchResultClassification::reconciliationRequired,
            DownstreamEvidenceClassification::causallyAmbiguous,
            SemanticConflictClassification::none,
            UncertainCommitRecoveryClassification::ambiguousEvidence,
            "dispatch_outcome_unknown", 2, 2, 1, 1,
            std::nullopt, std::nullopt);
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_phase5_transactional;");
        const auto outcomeId = transaction.exec(
            "INSERT INTO campaign_operations_dispatch_attempt_outcome("
            "dispatch_attempt_id,attempt_identity_canonical,"
            "result_classification,downstream_evidence_classification,"
            "semantic_conflict_classification,"
            "uncertain_commit_recovery_classification,diagnostic_code,"
            "expected_request_version,resulting_request_version,"
            "expected_reservation_version,resulting_reservation_version,"
            "outcome_contract_version,outcome_identity_canonical,"
            "outcome_identity_hash) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,"
            "$10,$11,1,$12,$13) "
            "RETURNING dispatch_attempt_outcome_id;",
            pqxx::params{expiredAmbiguousLease.attemptId.value(),
                expiredAmbiguousLease.acquisition.identity.canonicalText(),
                ToText(outcome.result), ToText(outcome.downstreamEvidence),
                ToText(outcome.conflict), ToText(outcome.recovery),
                outcome.diagnosticCode, outcome.expectedRequestVersion,
                outcome.resultingRequestVersion,
                outcome.expectedReservationVersion,
                outcome.resultingReservationVersion,
                outcome.identity.canonicalText(),
                outcome.identity.hash()}).one_row()[0].as<long long>();
        transaction.exec(
            "INSERT INTO campaign_operations_dispatch_audit_reference_event("
            "operational_campaign_id,operational_request_id,"
            "dispatch_attempt_id,dispatch_attempt_outcome_id,cause_kind,"
            "actor_identity,capability,prior_version,resulting_version,"
            "outcome,replay_disposition,diagnostic_code) VALUES($1,$2,$3,$4,"
            "'dispatch_handoff_failed','phase4.dispatcher@example.test',"
            "'campaign_operations_phase5_transactional',2,2,"
            "'reconciliation_required','reconciliation_required',"
            "'dispatch_outcome_unknown');",
            pqxx::params{
                expiredAmbiguousAccepted.persisted.request.request
                    .logicalOperation.campaignId.value(),
                expiredAmbiguousRequestId.value(),
                expiredAmbiguousLease.attemptId.value(), outcomeId});
        transaction.commit();
    }
    CampaignCancellationCommandRequest expiredAmbiguousCancellation;
    expiredAmbiguousCancellation.campaignId =
        expiredAmbiguousAccepted.persisted.request.request.logicalOperation
            .campaignId.value();
    expiredAmbiguousCancellation.requestId =
        expiredAmbiguousRequestId.value();
    expiredAmbiguousCancellation.expectedRequestVersion = 2;
    expiredAmbiguousCancellation.operationKey =
        "phase4-expired-ambiguous-cancellation";
    expiredAmbiguousCancellation.actorIdentity =
        "phase4.operator@example.test";
    expiredAmbiguousCancellation.reason =
        "Expired downstream evidence requires reconciliation.";
    const auto expiredAmbiguousResult = CancelCampaign(
        runtimeConnectionString, expiredAmbiguousCancellation);
    assert(expiredAmbiguousResult.progress ==
        CancellationProgress::reconciliationRequired);
    assert(!expiredAmbiguousResult.settlement);

    const auto recoveryAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 214, 1);
    const auto recoveryRequestId =
        recoveryAccepted.persisted.request.requestId;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_dispatcher;");
        (void)AcquireDispatchLeaseInTransaction(transaction,
            recoveryRequestId, 1,
            LeaseTokenDigest::Derive(
                "phase4-reconciliation-lease-token-01"),
            ActorIdentity("phase4.dispatcher@example.test"));
        transaction.commit();
    }
    ExpireDispatchLeaseForTest(owner, schema, recoveryRequestId);
    std::string utcReconciliationEvidence;
    std::string chicagoReconciliationEvidence;
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec("SET LOCAL TIME ZONE 'UTC';");
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_reconciler;");
        utcReconciliationEvidence =
            BuildReconciliationEvidenceCanonical(
                LoadReconciliationCandidate(
                    transaction, recoveryRequestId));
    }
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL TIME ZONE 'America/Chicago';");
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_reconciler;");
        chicagoReconciliationEvidence =
            BuildReconciliationEvidenceCanonical(
                LoadReconciliationCandidate(
                    transaction, recoveryRequestId));
    }
    assert(utcReconciliationEvidence ==
        chicagoReconciliationEvidence);
    assert(utcReconciliationEvidence.find("-05") ==
        std::string::npos);
    assert(utcReconciliationEvidence.find("-06") ==
        std::string::npos);
    ReconciliationObserveRequest observe;
    observe.runKey = "phase4-restart-recovery";
    observe.afterRequestId = recoveryRequestId.value() - 1;
    observe.limit = 1;
    observe.resolveSafeTransitions = false;
    const auto observationOnly =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, observe);
    assert(observationOnly.observationCount == 1);
    assert(observationOnly.resolutionCount == 0);
    long long recoveryObservationId = 0;
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        recoveryObservationId = transaction.exec(
            "SELECT reconciliation_observation_id "
            "FROM campaign_operations_reconciliation_observation "
            "WHERE run_key=$1 AND operational_request_id=$2;",
            pqxx::params{observe.runKey, recoveryRequestId.value()})
            .one_row()[0].as<long long>();
    }
    {
        pqxx::connection connection{runtimeConnectionString};
        pqxx::work transaction{connection};
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_recovery;");
        bool fabricatedCauseRejected = false;
        try
        {
            transaction.exec(
                "SELECT append_campaign_operations_recovery_resolution("
                "$1,'fabricated-transition','fnv1a64:0000000000000000',"
                "'request_returned_ready','fabricated-resolution',"
                "'fnv1a64:0000000000000000');",
                pqxx::params{recoveryObservationId});
        }
        catch (const pqxx::sql_error& error)
        {
            fabricatedCauseRejected = error.sqlstate() == "23514";
        }
        assert(fabricatedCauseRejected);
        transaction.abort();
    }
    {
        pqxx::connection connection{runtimeConnectionString};
        pqxx::work transaction{connection};
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_recovery;");
        bool foreignCapabilityRejected = false;
        try
        {
            transaction.exec(
                "SELECT append_campaign_operations_cancellation_resolution("
                "$1,'fabricated-transition','fnv1a64:0000000000000000',"
                "'fabricated-resolution','fnv1a64:0000000000000000');",
                pqxx::params{recoveryObservationId});
        }
        catch (const pqxx::sql_error&)
        {
            foreignCapabilityRejected = true;
        }
        assert(foreignCapabilityRejected);
        transaction.abort();
    }
    observe.resolveSafeTransitions = true;
    const auto recovered =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, observe);
    assert(recovered.observationCount == 1);
    assert(recovered.resolutionCount == 1);
    const auto replayAfterRestart =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, observe);
    assert(replayAfterRestart.selectedCount == 1);
    assert(replayAfterRestart.observationCount == 1);
    assert(replayAfterRestart.resolutionCount == 1);
    assert(replayAfterRestart.lastTargetId == recoveryRequestId.value());
    ReconciliationObserveRequest completedCursor = observe;
    completedCursor.afterRequestId = recoveryRequestId.value();
    const auto completedBatch =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, completedCursor);
    assert(completedBatch.selectedCount == 0);
    assert(completedBatch.lastTargetId == recoveryRequestId.value());
    const auto completedBatchReplay =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, completedCursor);
    assert(completedBatchReplay.selectedCount == 0);
    assert(completedBatchReplay.lastTargetId == recoveryRequestId.value());
    {
        pqxx::connection connection{runtimeConnectionString};
        pqxx::work transaction{connection};
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_reconciler;");
        bool changedCursorConflicted = false;
        try
        {
            (void)PersistReconciliationBatch(transaction, observe.runKey,
                observe.afterRequestId, observe.limit + 1, {});
        }
        catch (const Error& error)
        {
            changedCursorConflicted =
                error.code() == ErrorCode::persistenceConflict;
        }
        assert(changedCursorConflicted);
        transaction.abort();
    }
    for (const auto& [runKey, lastTargetId, selectedCount] :
         std::vector<std::tuple<std::string, long long, int>>{
             {"phase4-malformed-cursor-missing-observation",
                 recoveryRequestId.value(), 1},
             {"phase4-malformed-cursor-last-target",
                 recoveryRequestId.value(), 0}})
    {
        pqxx::connection connection{runtimeConnectionString};
        pqxx::work transaction{connection};
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_reconciler;");
        transaction.exec(
            "INSERT INTO campaign_operations_reconciliation_cursor_event("
            "run_key,prior_target_id,last_target_id,requested_limit,"
            "selected_count) VALUES($1,$2,$3,1,$4);",
            pqxx::params{runKey, recoveryRequestId.value() - 1,
                lastTargetId, selectedCount});
        bool malformedCursorRejected = false;
        try
        {
            transaction.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            malformedCursorRejected = error.sqlstate() == "23514";
        }
        assert(malformedCursorRejected);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM "
            "campaign_operations_reconciliation_cursor_event "
            "WHERE run_key='" + runKey + "'") == 0);
    }
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        const auto row = transaction.exec(
            "SELECT request_state,state_version,lease_token_hash IS NULL,"
            "lease_expires_at IS NULL,dispatcher_identity IS NULL "
            "FROM campaign_operations_operational_request "
            "WHERE operational_request_id=$1;",
            pqxx::params{recoveryRequestId.value()}).one_row();
        assert(row[0].as<std::string>() == "ready");
        assert(row[1].as<int>() == 3);
        assert(row[2].as<bool>());
        assert(row[3].as<bool>());
        assert(row[4].as<bool>());
        assert(transaction.exec(
            "SELECT count(*) FROM "
            "campaign_operations_reconciliation_observation "
            "WHERE operational_request_id=$1;",
            pqxx::params{recoveryRequestId.value()})
            .one_row()[0].as<int>() == 1);
        assert(transaction.exec(
            "SELECT count(*) FROM "
            "campaign_operations_reconciliation_resolution resolution "
            "JOIN campaign_operations_reconciliation_observation observation "
            "USING(reconciliation_observation_id) "
            "WHERE observation.operational_request_id=$1;",
            pqxx::params{recoveryRequestId.value()})
            .one_row()[0].as<int>() == 1);
        const auto causalOwnership = transaction.exec(
            "SELECT resolution.owning_capability,"
            "resolution.cancellation_settlement_id IS NULL,"
            "resolution.dispatch_attempt_outcome_id IS NOT NULL "
            "FROM campaign_operations_reconciliation_resolution resolution "
            "WHERE resolution.reconciliation_observation_id=$1;",
            pqxx::params{recoveryObservationId}).one_row();
        assert(causalOwnership[0].as<std::string>() ==
            "campaign_operations_recovery");
        assert(causalOwnership[1].as<bool>());
        assert(causalOwnership[2].as<bool>());
    }

    const auto crashBatchAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 218, 1);
    const auto crashBatchRequestId =
        crashBatchAccepted.persisted.request.requestId;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_dispatcher;");
        (void)AcquireDispatchLeaseInTransaction(transaction,
            crashBatchRequestId, 1,
            LeaseTokenDigest::Derive(
                "phase4-atomic-batch-crash-token-0001"),
            ActorIdentity("phase4.dispatcher@example.test"));
        transaction.commit();
    }
    ExpireDispatchLeaseForTest(owner, schema, crashBatchRequestId);
    ReconciliationObserveRequest crashBatchObserve;
    crashBatchObserve.runKey = "phase4-atomic-batch-crash";
    crashBatchObserve.afterRequestId = crashBatchRequestId.value() - 1;
    crashBatchObserve.limit = 1;
    bool batchCrashInjected = false;
    try
    {
        (void)ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, crashBatchObserve,
            [&](CampaignOperationsControlTestInjectionPoint point,
                pqxx::transaction_base&)
            {
                if (point == CampaignOperationsControlTestInjectionPoint::
                        beforeReconciliationBatchCommit)
                {
                    batchCrashInjected = true;
                    throw std::runtime_error(
                        "phase4_atomic_batch_crash_injected");
                }
            });
    }
    catch (const std::runtime_error& error)
    {
        assert(std::string(error.what()) ==
            "phase4_atomic_batch_crash_injected");
    }
    assert(batchCrashInjected);
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        assert(transaction.exec(
            "SELECT count(*) FROM "
            "campaign_operations_reconciliation_cursor_event "
            "WHERE run_key=$1;",
            pqxx::params{crashBatchObserve.runKey})
            .one_row()[0].as<int>() == 0);
        assert(transaction.exec(
            "SELECT count(*) FROM "
            "campaign_operations_reconciliation_observation "
            "WHERE run_key=$1;",
            pqxx::params{crashBatchObserve.runKey})
            .one_row()[0].as<int>() == 0);
    }
    const auto crashBatchRestart =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, crashBatchObserve);
    const auto crashBatchReplay =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, crashBatchObserve);
    assert(crashBatchRestart.selectedCount == 1);
    assert(crashBatchRestart.observationCount == 1);
    assert(crashBatchReplay.selectedCount == 1);
    assert(crashBatchReplay.observationCount == 1);
    assert(crashBatchReplay.lastTargetId ==
        crashBatchRequestId.value());

    const auto overlapFiller = AcceptPhase4Fixture(
        owner, connectionString, schema, 219, 1);
    const auto overlapAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 220, 1);
    const auto overlapRequestId =
        overlapAccepted.persisted.request.requestId;
    assert(overlapFiller.persisted.request.requestId.value() ==
        overlapRequestId.value() - 1);
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_dispatcher;");
        (void)AcquireDispatchLeaseInTransaction(transaction,
            overlapRequestId, 1,
            LeaseTokenDigest::Derive(
                "phase4-overlapping-cursor-token-0001"),
            ActorIdentity("phase4.dispatcher@example.test"));
        transaction.commit();
    }
    ExpireDispatchLeaseForTest(owner, schema, overlapRequestId);
    ReconciliationObserveRequest firstOverlap;
    firstOverlap.runKey = "phase4-overlapping-exact-cursors";
    firstOverlap.afterRequestId = overlapRequestId.value() - 1;
    firstOverlap.limit = 1;
    const auto firstOverlapBatch =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, firstOverlap);
    ReconciliationObserveRequest secondOverlap = firstOverlap;
    secondOverlap.afterRequestId = overlapRequestId.value() - 2;
    const auto secondOverlapBatch =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, secondOverlap);
    assert(firstOverlapBatch.selectedCount == 1);
    assert(firstOverlapBatch.observationCount == 1);
    assert(secondOverlapBatch.selectedCount == 1);
    assert(secondOverlapBatch.observationCount == 1);
    long long firstOverlapCursorId = 0;
    long long secondOverlapCursorId = 0;
    long long firstOverlapObservationId = 0;
    long long secondOverlapObservationId = 0;
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        firstOverlapCursorId = transaction.exec(
            "SELECT reconciliation_cursor_event_id FROM "
            "campaign_operations_reconciliation_cursor_event "
            "WHERE run_key=$1 AND prior_target_id=$2;",
            pqxx::params{firstOverlap.runKey,
                firstOverlap.afterRequestId})
            .one_row()[0].as<long long>();
        secondOverlapCursorId = transaction.exec(
            "SELECT reconciliation_cursor_event_id FROM "
            "campaign_operations_reconciliation_cursor_event "
            "WHERE run_key=$1 AND prior_target_id=$2;",
            pqxx::params{secondOverlap.runKey,
                secondOverlap.afterRequestId})
            .one_row()[0].as<long long>();
        assert(firstOverlapCursorId != secondOverlapCursorId);
        firstOverlapObservationId = transaction.exec(
            "SELECT reconciliation_observation_id FROM "
            "campaign_operations_reconciliation_observation "
            "WHERE reconciliation_cursor_event_id=$1;",
            pqxx::params{firstOverlapCursorId})
            .one_row()[0].as<long long>();
        secondOverlapObservationId = transaction.exec(
            "SELECT reconciliation_observation_id FROM "
            "campaign_operations_reconciliation_observation "
            "WHERE reconciliation_cursor_event_id=$1;",
            pqxx::params{secondOverlapCursorId})
            .one_row()[0].as<long long>();
        assert(firstOverlapObservationId != secondOverlapObservationId);
        assert(transaction.exec(
            "SELECT count(*) FROM "
            "campaign_operations_reconciliation_observation "
            "WHERE run_key=$1 AND operational_request_id=$2;",
            pqxx::params{firstOverlap.runKey,
                overlapRequestId.value()})
            .one_row()[0].as<int>() == 2);
    }
    firstOverlap.resolveSafeTransitions = true;
    const auto resolvedFirstOverlap =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, firstOverlap);
    assert(resolvedFirstOverlap.resolutionCount == 1);
    secondOverlap.resolveSafeTransitions = true;
    const auto exactChangedStateReplay =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, secondOverlap);
    assert(exactChangedStateReplay.selectedCount == 1);
    assert(exactChangedStateReplay.observationCount == 1);
    assert(exactChangedStateReplay.resolutionCount == 1);
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        assert(transaction.exec(
            "SELECT reconciliation_observation_id FROM "
            "campaign_operations_reconciliation_observation "
            "WHERE reconciliation_cursor_event_id=$1;",
            pqxx::params{firstOverlapCursorId})
            .one_row()[0].as<long long>() ==
            firstOverlapObservationId);
        assert(transaction.exec(
            "SELECT reconciliation_observation_id FROM "
            "campaign_operations_reconciliation_observation "
            "WHERE reconciliation_cursor_event_id=$1;",
            pqxx::params{secondOverlapCursorId})
            .one_row()[0].as<long long>() ==
            secondOverlapObservationId);
        const auto converged = transaction.exec(
            "SELECT count(*),count(DISTINCT "
            "resolution.transition_identity_canonical),"
            "count(DISTINCT resolution.transition_identity_hash),"
            "count(*) FILTER (WHERE "
            "resolution.resolution_disposition='request_returned_ready'),"
            "count(*) FILTER (WHERE "
            "resolution.resolution_disposition='already_resolved') "
            "FROM campaign_operations_reconciliation_resolution resolution "
            "JOIN campaign_operations_reconciliation_observation observation "
            "USING(reconciliation_observation_id) "
            "WHERE observation.operational_request_id=$1 "
            "AND observation.run_key=$2;",
            pqxx::params{overlapRequestId.value(),
                firstOverlap.runKey}).one_row();
        assert(converged[0].as<int>() == 2);
        assert(converged[1].as<int>() == 1);
        assert(converged[2].as<int>() == 1);
        assert(converged[3].as<int>() == 1);
        assert(converged[4].as<int>() == 1);
    }

    const auto cancellationLockRaceAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 221, 1);
    const auto cancellationLockRaceRequestId =
        cancellationLockRaceAccepted.persisted.request.requestId;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_dispatcher;");
        (void)AcquireDispatchLeaseInTransaction(transaction,
            cancellationLockRaceRequestId, 1,
            LeaseTokenDigest::Derive(
                "phase4-observe-cancel-lock-race-0001"),
            ActorIdentity("phase4.dispatcher@example.test"));
        transaction.commit();
    }
    ExpireDispatchLeaseForTest(
        owner, schema, cancellationLockRaceRequestId);
    ReconciliationObserveRequest cancellationLockRaceObserve;
    cancellationLockRaceObserve.runKey =
        "phase4-observe-cancellation-lock-order";
    cancellationLockRaceObserve.afterRequestId =
        cancellationLockRaceRequestId.value() - 1;
    cancellationLockRaceObserve.limit = 1;
    CampaignCancellationCommandRequest cancellationLockRaceCommand;
    cancellationLockRaceCommand.campaignId =
        cancellationLockRaceAccepted.persisted.request.request
            .logicalOperation.campaignId.value();
    cancellationLockRaceCommand.requestId =
        cancellationLockRaceRequestId.value();
    cancellationLockRaceCommand.expectedRequestVersion = 2;
    cancellationLockRaceCommand.operationKey =
        "phase4-observe-cancellation-lock-order";
    cancellationLockRaceCommand.actorIdentity =
        "phase4.operator@example.test";
    cancellationLockRaceCommand.reason =
        "Race cancellation against ordered reconciliation persistence.";
    std::atomic<bool> cancellationRaceCampaignLocked{false};
    std::atomic<bool> releaseCancellationRace{false};
    std::optional<ReconciliationBatchResult> cancellationRaceObservation;
    std::optional<CampaignCancellationResult> cancellationRaceResult;
    std::exception_ptr cancellationRaceObservationError;
    std::exception_ptr cancellationRaceError;
    const std::string cancellationRaceObserveApplication =
        "phase4_observe_cancel_observer";
    const std::string cancellationRaceCancelApplication =
        "phase4_observe_cancel_canceller";
    std::thread cancellationRaceObservationThread([&]
    {
        try
        {
            cancellationRaceObservation.emplace(
                ObserveAndRecoverCampaignOperations(
                    runtimeConnectionString + " application_name=" +
                        cancellationRaceObserveApplication,
                    cancellationLockRaceObserve,
                    [&](CampaignOperationsControlTestInjectionPoint point,
                        pqxx::transaction_base&)
                    {
                        if (point !=
                            CampaignOperationsControlTestInjectionPoint::
                                afterReconciliationCampaignLocksBeforeRequestLocks)
                            return;
                        cancellationRaceCampaignLocked.store(
                            true, std::memory_order_release);
                        while (!releaseCancellationRace.load(
                            std::memory_order_acquire))
                            std::this_thread::yield();
                    }));
        }
        catch (...)
        {
            cancellationRaceObservationError =
                std::current_exception();
        }
    });
    for (int attempt = 0;
         attempt < 2500 &&
         !cancellationRaceCampaignLocked.load(std::memory_order_acquire);
         ++attempt)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    assert(cancellationRaceCampaignLocked.load(
        std::memory_order_acquire));
    std::thread cancellationRaceThread([&]
    {
        try
        {
            cancellationRaceResult.emplace(CancelCampaign(
                runtimeConnectionString + " application_name=" +
                    cancellationRaceCancelApplication,
                cancellationLockRaceCommand));
        }
        catch (...)
        {
            cancellationRaceError = std::current_exception();
        }
    });
    assert(WaitForApplicationLockWait(
        owner, cancellationRaceCancelApplication));
    releaseCancellationRace.store(true, std::memory_order_release);
    cancellationRaceObservationThread.join();
    cancellationRaceThread.join();
    if (cancellationRaceObservationError)
        std::rethrow_exception(cancellationRaceObservationError);
    if (cancellationRaceError)
        std::rethrow_exception(cancellationRaceError);
    assert(cancellationRaceObservation);
    assert(cancellationRaceObservation->observationCount == 1);
    assert(cancellationRaceResult);
    assert(cancellationRaceResult->progress ==
        CancellationProgress::settled);

    const auto recoveryLockRaceFiller = AcceptPhase4Fixture(
        owner, connectionString, schema, 222, 1);
    const auto recoveryLockRaceAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 223, 1);
    const auto recoveryLockRaceRequestId =
        recoveryLockRaceAccepted.persisted.request.requestId;
    assert(recoveryLockRaceFiller.persisted.request.requestId.value() ==
        recoveryLockRaceRequestId.value() - 1);
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_dispatcher;");
        (void)AcquireDispatchLeaseInTransaction(transaction,
            recoveryLockRaceRequestId, 1,
            LeaseTokenDigest::Derive(
                "phase4-observe-recovery-lock-race-01"),
            ActorIdentity("phase4.dispatcher@example.test"));
        transaction.commit();
    }
    ExpireDispatchLeaseForTest(
        owner, schema, recoveryLockRaceRequestId);
    ReconciliationObserveRequest recoveryLockRaceFirst;
    recoveryLockRaceFirst.runKey =
        "phase4-observe-recovery-lock-order";
    recoveryLockRaceFirst.afterRequestId =
        recoveryLockRaceRequestId.value() - 1;
    recoveryLockRaceFirst.limit = 1;
    const auto recoveryLockRaceFirstResult =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, recoveryLockRaceFirst);
    assert(recoveryLockRaceFirstResult.observationCount == 1);
    std::optional<PersistedReconciliationObservation>
        recoveryLockRaceEvidence;
    {
        pqxx::connection connection{runtimeConnectionString};
        pqxx::read_transaction transaction{connection};
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_reconciler;");
        const auto cursor = FindReconciliationCursor(transaction,
            recoveryLockRaceFirst.runKey,
            recoveryLockRaceFirst.afterRequestId);
        assert(cursor);
        const auto persisted = LoadReconciliationObservationsForCursor(
            transaction, cursor->reconciliationCursorEventId);
        assert(persisted.size() == 1);
        recoveryLockRaceEvidence.emplace(persisted.front());
    }
    ReconciliationObserveRequest recoveryLockRaceSecond =
        recoveryLockRaceFirst;
    recoveryLockRaceSecond.afterRequestId =
        recoveryLockRaceRequestId.value() - 2;
    std::atomic<bool> recoveryRaceCampaignLocked{false};
    std::atomic<bool> releaseRecoveryRace{false};
    std::optional<ReconciliationBatchResult> recoveryRaceObservation;
    std::optional<PersistedReconciliationResolution> recoveryRaceResolution;
    std::exception_ptr recoveryRaceObservationError;
    std::exception_ptr recoveryRaceError;
    const std::string recoveryRaceObserveApplication =
        "phase4_observe_recovery_observer";
    const std::string recoveryRaceRecoveryApplication =
        "phase4_observe_recovery_worker";
    std::thread recoveryRaceObservationThread([&]
    {
        try
        {
            recoveryRaceObservation.emplace(
                ObserveAndRecoverCampaignOperations(
                    runtimeConnectionString + " application_name=" +
                        recoveryRaceObserveApplication,
                    recoveryLockRaceSecond,
                    [&](CampaignOperationsControlTestInjectionPoint point,
                        pqxx::transaction_base&)
                    {
                        if (point !=
                            CampaignOperationsControlTestInjectionPoint::
                                afterReconciliationCampaignLocksBeforeRequestLocks)
                            return;
                        recoveryRaceCampaignLocked.store(
                            true, std::memory_order_release);
                        while (!releaseRecoveryRace.load(
                            std::memory_order_acquire))
                            std::this_thread::yield();
                    }));
        }
        catch (...)
        {
            recoveryRaceObservationError =
                std::current_exception();
        }
    });
    for (int attempt = 0;
         attempt < 2500 &&
         !recoveryRaceCampaignLocked.load(std::memory_order_acquire);
         ++attempt)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    assert(recoveryRaceCampaignLocked.load(std::memory_order_acquire));
    std::thread recoveryRaceThread([&]
    {
        try
        {
            pqxx::connection connection{
                runtimeConnectionString + " application_name=" +
                    recoveryRaceRecoveryApplication};
            pqxx::work transaction{connection};
            transaction.exec(
                "SET LOCAL ROLE campaign_operations_recovery;");
            recoveryRaceResolution.emplace(
                RecoverExpiredDispatchLeaseInTransaction(
                    transaction, *recoveryLockRaceEvidence));
            transaction.commit();
        }
        catch (...)
        {
            recoveryRaceError = std::current_exception();
        }
    });
    assert(WaitForApplicationLockWait(
        owner, recoveryRaceRecoveryApplication));
    releaseRecoveryRace.store(true, std::memory_order_release);
    recoveryRaceObservationThread.join();
    recoveryRaceThread.join();
    if (recoveryRaceObservationError)
        std::rethrow_exception(recoveryRaceObservationError);
    if (recoveryRaceError)
        std::rethrow_exception(recoveryRaceError);
    assert(recoveryRaceObservation);
    assert(recoveryRaceObservation->observationCount == 1);
    assert(recoveryRaceResolution);
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        assert(transaction.exec(
            "SELECT request_state FROM "
            "campaign_operations_operational_request "
            "WHERE operational_request_id=$1;",
            pqxx::params{recoveryLockRaceRequestId.value()})
            .one_row()[0].as<std::string>() == "ready");
        assert(transaction.exec(
            "SELECT count(DISTINCT reconciliation_cursor_event_id) "
            "FROM campaign_operations_reconciliation_observation "
            "WHERE run_key=$1 AND operational_request_id=$2;",
            pqxx::params{recoveryLockRaceFirst.runKey,
                recoveryLockRaceRequestId.value()})
            .one_row()[0].as<int>() == 2);
    }

    std::optional<ReconciliationObserveRequest>
        retryableRecoveryObservation;
    for (const auto& [sqlState, materializationId] :
         std::vector<std::pair<std::string, long long>>{
             {"40001", 225}, {"40P01", 226}})
    {
        const auto retryAccepted = AcceptPhase4Fixture(
            owner, connectionString, schema, materializationId, 1);
        const auto retryRequestId =
            retryAccepted.persisted.request.requestId;
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec(
                "SET LOCAL ROLE campaign_operations_dispatcher;");
            (void)AcquireDispatchLeaseInTransaction(transaction,
                retryRequestId, 1,
                LeaseTokenDigest::Derive(
                    "phase4-reconciliation-retry-" + sqlState),
                ActorIdentity("phase4.dispatcher@example.test"));
            transaction.commit();
        }
        ExpireDispatchLeaseForTest(owner, schema, retryRequestId);
        ReconciliationObserveRequest retryObserve;
        retryObserve.runKey =
            "phase4-reconciliation-retry-" + sqlState;
        retryObserve.afterRequestId = retryRequestId.value() - 1;
        retryObserve.limit = 1;
        int batchAttempts = 0;
        const auto retriedBatch =
            ObserveAndRecoverCampaignOperations(
                runtimeConnectionString, retryObserve,
                [&](CampaignOperationsControlTestInjectionPoint point,
                    pqxx::transaction_base& transaction)
                {
                    if (point !=
                        CampaignOperationsControlTestInjectionPoint::
                            beforeReconciliationBatchCommit)
                        return;
                    ++batchAttempts;
                    if (batchAttempts == 1)
                        transaction.exec(
                            "DO $phase4_reconciliation_retry$ BEGIN "
                            "RAISE EXCEPTION "
                            "'deterministic reconciliation retry' "
                            "USING ERRCODE='" + sqlState + "'; "
                            "END $phase4_reconciliation_retry$;");
                });
        assert(batchAttempts == 2);
        assert(retriedBatch.selectedCount == 1);
        assert(retriedBatch.observationCount == 1);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM "
            "campaign_operations_reconciliation_cursor_event "
            "WHERE run_key='" + retryObserve.runKey + "'") == 1);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM "
            "campaign_operations_reconciliation_observation "
            "WHERE run_key='" + retryObserve.runKey + "'") == 1);
        if (sqlState == "40001")
            retryableRecoveryObservation.emplace(retryObserve);
    }
    assert(retryableRecoveryObservation);
    retryableRecoveryObservation->resolveSafeTransitions = true;
    int recoveryAttempts = 0;
    const auto retriedRecovery =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, *retryableRecoveryObservation,
            [&](CampaignOperationsControlTestInjectionPoint point,
                pqxx::transaction_base& transaction)
            {
                if (point !=
                    CampaignOperationsControlTestInjectionPoint::
                        beforeReconciliationRecoveryCommit)
                    return;
                ++recoveryAttempts;
                if (recoveryAttempts == 1)
                    transaction.exec(
                        "DO $phase4_recovery_retry$ BEGIN "
                        "RAISE EXCEPTION "
                        "'deterministic recovery retry' "
                        "USING ERRCODE='40001'; "
                        "END $phase4_recovery_retry$;");
            });
    assert(recoveryAttempts == 2);
    assert(retriedRecovery.resolutionCount == 1);

    const auto uncertainCommitAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 227, 1);
    const auto uncertainCommitRequestId =
        uncertainCommitAccepted.persisted.request.requestId;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_dispatcher;");
        (void)AcquireDispatchLeaseInTransaction(transaction,
            uncertainCommitRequestId, 1,
            LeaseTokenDigest::Derive(
                "phase4-reconciliation-uncertain-commit"),
            ActorIdentity("phase4.dispatcher@example.test"));
        transaction.commit();
    }
    ExpireDispatchLeaseForTest(
        owner, schema, uncertainCommitRequestId);
    ReconciliationObserveRequest uncertainCommitObserve;
    uncertainCommitObserve.runKey =
        "phase4-reconciliation-uncertain-commit";
    uncertainCommitObserve.afterRequestId =
        uncertainCommitRequestId.value() - 1;
    uncertainCommitObserve.limit = 1;
    int batchCommitResponseLosses = 0;
    const auto uncertainBatch =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, uncertainCommitObserve,
            [&](CampaignOperationsControlTestInjectionPoint point,
                pqxx::transaction_base&)
            {
                if (point ==
                        CampaignOperationsControlTestInjectionPoint::
                            afterReconciliationBatchCommitBeforeResponse &&
                    batchCommitResponseLosses++ == 0)
                    throw pqxx::broken_connection(
                        "deterministic reconciliation commit response loss");
            });
    assert(batchCommitResponseLosses == 1);
    assert(uncertainBatch.observationCount == 1);
    assert(Scalar(owner, schema,
        "SELECT count(*) FROM "
        "campaign_operations_reconciliation_cursor_event "
        "WHERE run_key='" + uncertainCommitObserve.runKey + "'") == 1);
    assert(Scalar(owner, schema,
        "SELECT count(*) FROM "
        "campaign_operations_reconciliation_observation "
        "WHERE run_key='" + uncertainCommitObserve.runKey + "'") == 1);

    uncertainCommitObserve.resolveSafeTransitions = true;
    int recoveryCommitResponseLosses = 0;
    const auto uncertainRecovery =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, uncertainCommitObserve,
            [&](CampaignOperationsControlTestInjectionPoint point,
                pqxx::transaction_base&)
            {
                if (point ==
                        CampaignOperationsControlTestInjectionPoint::
                            afterReconciliationRecoveryCommitBeforeResponse &&
                    recoveryCommitResponseLosses++ == 0)
                    throw pqxx::broken_connection(
                        "deterministic recovery commit response loss");
            });
    assert(recoveryCommitResponseLosses == 1);
    assert(uncertainRecovery.resolutionCount == 1);
    assert(Scalar(owner, schema,
        "SELECT count(*) FROM "
        "campaign_operations_reconciliation_resolution resolution "
        "JOIN campaign_operations_reconciliation_observation observation "
        "USING(reconciliation_observation_id) "
        "WHERE observation.run_key='" +
        uncertainCommitObserve.runKey + "'") == 1);

    std::string reservationExpiry;
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        reservationExpiry = transaction.exec(
            "SELECT to_char(("
            "clock_timestamp()+interval '1 second') AT TIME ZONE 'UTC',"
            "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"');")
            .one_row()[0].as<std::string>();
    }
    const auto expiringReservationAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 216, 1, reservationExpiry);
    const auto expiringReservationRequestId =
        expiringReservationAccepted.persisted.request.requestId;
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    ReconciliationObserveRequest observeExpiredReservation;
    observeExpiredReservation.runKey =
        "phase4-expired-reservation-observation";
    observeExpiredReservation.afterRequestId =
        expiringReservationRequestId.value() - 1;
    observeExpiredReservation.limit = 1;
    observeExpiredReservation.resolveSafeTransitions = true;
    const auto expiredReservationObservation =
        ObserveAndRecoverCampaignOperations(
            runtimeConnectionString, observeExpiredReservation);
    assert(expiredReservationObservation.selectedCount == 1);
    assert(expiredReservationObservation.observationCount == 1);
    assert(expiredReservationObservation.resolutionCount == 0);
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        const auto row = transaction.exec(
            "SELECT request.request_state,reservation.reservation_state,"
            "observation.reason_code,observation.recommended_service,"
            "observation.recommended_action "
            "FROM campaign_operations_operational_request request "
            "JOIN campaign_operations_reservation reservation "
            "ON reservation.reservation_id=request.reservation_id "
            "JOIN campaign_operations_reconciliation_observation observation "
            "ON observation.operational_request_id="
            "request.operational_request_id "
            "WHERE request.operational_request_id=$1;",
            pqxx::params{expiringReservationRequestId.value()})
            .one_row();
        assert(row[0].as<std::string>() == "ready");
        assert(row[1].as<std::string>() == "held");
        assert(row[2].as<std::string>() ==
            "reservation_expired_no_downstream_evidence");
        assert(row[3].as<std::string>() ==
            "campaign_operations_reservation_service");
        assert(row[4].as<std::string>() ==
            "expire_held_reservation");
    }
}

void TestPhase5OperationalCompletion(pqxx::connection& owner,
    const std::string& connectionString, const std::string& schema)
{
    const std::string runtimeConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c lock_timeout=5s -c statement_timeout=15s'";
    const auto assertBlockedBy = [&](OperationalCampaignId targetCampaignId,
                                     const std::string& operationKey,
                                     const std::vector<std::string>& codes)
    {
        pqxx::connection connection{runtimeConnectionString};
        const auto result = CompleteCampaignIfSettled(connection,
            {targetCampaignId.value(), operationKey,
                "phase5.completer@example.test",
                "Verify one exact completion prerequisite remains blocked."});
        assert(result.disposition == CompletionAttemptDisposition::blocked);
        for (const auto& code : codes)
            assert(std::any_of(result.blockers.begin(),
                result.blockers.end(), [&](const CompletionBlocker& blocker)
                {
                    return blocker.code == code;
                }));
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_completion_event",
                   targetCampaignId) == 0);
    };

    const auto missingAuthorityCampaign = PersistCampaignFixture(
        owner, schema, InsertMaterialization(owner, schema, 242, 1));
    assertBlockedBy(missingAuthorityCampaign.campaignId,
        "phase-g-missing-authority",
        {"authorization_missing", "budget_missing", "request_missing"});
    {
        bool prematureBoundaryCloseRejected = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec(
                "UPDATE campaign_operations_campaign "
                "SET completion_boundary_closed=true "
                "WHERE operational_campaign_id=$1;",
                pqxx::params{missingAuthorityCampaign.campaignId.value()});
            transaction.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            prematureBoundaryCloseRejected =
                error.sqlstate() == "55000";
        }
        assert(prematureBoundaryCloseRejected);
        pqxx::read_transaction verify{owner};
        SetSearchPath(verify, schema);
        assert(!verify.exec(
            "SELECT completion_boundary_closed FROM "
            "campaign_operations_campaign WHERE operational_campaign_id=$1;",
            pqxx::params{missingAuthorityCampaign.campaignId.value()})
            .one_row()[0].as<bool>());
    }

    const std::string budgetConnectionString = connectionString +
        " options='-c search_path=" + schema +
        " -c role=campaign_operations_budget_administrator'";
    const auto noRequestFixture = CreatePhase2AcceptanceFixture(
        owner, budgetConnectionString, schema, 243, 1);
    assertBlockedBy(noRequestFixture.campaign.campaignId,
        "phase-g-missing-request",
        {"active_authorization_unexhausted", "request_missing"});

    const auto leaseAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 244, 1);
    const auto leaseCampaignId =
        leaseAccepted.persisted.request.request.logicalOperation.campaignId;
    const auto leaseRequestId = leaseAccepted.persisted.request.requestId;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_dispatcher;");
        (void)AcquireDispatchLeaseInTransaction(transaction, leaseRequestId, 1,
            LeaseTokenDigest::Derive(
                "phase-g-blocker-lease-0123456789abcdef"),
            ActorIdentity("phase5.dispatcher@example.test"));
        transaction.commit();
    }
    assertBlockedBy(leaseCampaignId, "phase-g-active-lease",
        {"request_dispatching", "active_dispatch_lease",
            "incomplete_dispatch_attempt"});
    CampaignCancellationCommandRequest unsettledCancellation;
    unsettledCancellation.campaignId = leaseCampaignId.value();
    unsettledCancellation.requestId = leaseRequestId.value();
    unsettledCancellation.expectedRequestVersion = 2;
    unsettledCancellation.operationKey = "phase-g-unsettled-cancel";
    unsettledCancellation.actorIdentity = "phase5.operator@example.test";
    unsettledCancellation.reason =
        "Record intent while an active dispatch lease prevents settlement.";
    assert(CancelCampaign(runtimeConnectionString, unsettledCancellation)
               .progress == CancellationProgress::waitingForLeaseExpiry);
    assertBlockedBy(leaseCampaignId, "phase-g-cancellation-unsettled",
        {"cancellation_unsettled"});
    ExpireDispatchLeaseForTest(owner, schema, leaseRequestId);
    const auto observations = ObserveAndRecoverCampaignOperations(
        runtimeConnectionString,
        {"phase-g-blocking-observation", 0, 500, false});
    assert(observations.observationCount > 0);
    assertBlockedBy(leaseCampaignId, "phase-g-reconciliation-unsettled",
        {"blocking_reconciliation_observation"});

    const auto reconciliationRaceAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 246, 1);
    const auto reconciliationRaceCampaignId =
        reconciliationRaceAccepted.persisted.request.request.logicalOperation.
            campaignId;
    const auto reconciliationRaceRequestId =
        reconciliationRaceAccepted.persisted.request.requestId;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_dispatcher;");
        (void)AcquireDispatchLeaseInTransaction(transaction,
            reconciliationRaceRequestId, 1,
            LeaseTokenDigest::Derive(
                "phase-g-reconciliation-race-lease-token"),
            ActorIdentity("phase5.dispatcher@example.test"));
        transaction.commit();
    }
    ExpireDispatchLeaseForTest(
        owner, schema, reconciliationRaceRequestId);
    std::barrier reconciliationRaceStart{3};
    std::optional<CompleteIfSettledResult> reconciliationCompletionResult;
    std::optional<ReconciliationBatchResult> reconciliationRaceResult;
    std::exception_ptr reconciliationCompletionError;
    std::exception_ptr reconciliationRaceError;
    std::thread reconciliationCompletion{[&]
    {
        try
        {
            pqxx::connection connection{runtimeConnectionString};
            reconciliationRaceStart.arrive_and_wait();
            reconciliationCompletionResult.emplace(
                CompleteCampaignIfSettled(connection,
                    {reconciliationRaceCampaignId.value(),
                        "phase-g-reconciliation-race-completion",
                        "phase5.completer@example.test",
                        "Completion must serialize with reconciliation."}));
        }
        catch (...)
        {
            reconciliationCompletionError = std::current_exception();
        }
    }};
    std::thread reconciliationWorker{[&]
    {
        try
        {
            reconciliationRaceStart.arrive_and_wait();
            reconciliationRaceResult.emplace(
                ObserveAndRecoverCampaignOperations(
                    runtimeConnectionString,
                    {"phase-g-reconciliation-race", 0, 500, true}));
        }
        catch (...)
        {
            reconciliationRaceError = std::current_exception();
        }
    }};
    reconciliationRaceStart.arrive_and_wait();
    reconciliationCompletion.join();
    reconciliationWorker.join();
    if (reconciliationCompletionError)
        std::rethrow_exception(reconciliationCompletionError);
    if (reconciliationRaceError)
        std::rethrow_exception(reconciliationRaceError);
    assert(reconciliationCompletionResult);
    assert(reconciliationCompletionResult->disposition ==
        CompletionAttemptDisposition::blocked);
    assert(reconciliationRaceResult);
    assert(reconciliationRaceResult->observationCount > 0);
    assert(reconciliationRaceResult->resolutionCount > 0);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_completion_event",
               reconciliationRaceCampaignId) == 0);

    const auto accepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 240, 2);
    const auto campaignId =
        accepted.persisted.request.request.logicalOperation.campaignId;
    const auto requestId = accepted.persisted.request.requestId;
    CompleteIfSettledRequest completionRequest{
        campaignId.value(), "phase-g-cancelled-completion",
        "phase5.completer@example.test",
        "Record exact settled all-scope cancellation."};

    {
        pqxx::connection connection{runtimeConnectionString};
        const auto blocked =
            CompleteCampaignIfSettled(connection, completionRequest);
        assert(blocked.disposition ==
            CompletionAttemptDisposition::blocked);
        assert(std::any_of(blocked.blockers.begin(),
            blocked.blockers.end(), [](const CompletionBlocker& blocker)
            {
                return blocker.code == "request_ready";
            }));
        assert(std::any_of(blocked.blockers.begin(),
            blocked.blockers.end(), [](const CompletionBlocker& blocker)
            {
                return blocker.code == "reservation_held";
            }));
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_completion_event", campaignId) == 0);

    {
        pqxx::connection connection{runtimeConnectionString};
        const auto paused = ControlCampaign(connection,
            {campaignId.value(), 0, ControlEventKind::pause,
                "phase5.operator@example.test",
                "Pause must never masquerade as completion."});
        assert(paused.persisted.event.eventKind == ControlEventKind::pause);
        const auto blocked =
            CompleteCampaignIfSettled(connection, completionRequest);
        assert(blocked.disposition ==
            CompletionAttemptDisposition::blocked);
        assert(std::any_of(blocked.blockers.begin(),
            blocked.blockers.end(), [](const CompletionBlocker& blocker)
            {
                return blocker.code == "campaign_paused";
            }));
        const auto resumed = ControlCampaign(connection,
            {campaignId.value(), 1, ControlEventKind::resume,
                "phase5.operator@example.test",
                "Resume only removes the Campaign Operations pause gate."});
        assert(resumed.persisted.event.eventKind == ControlEventKind::resume);
    }

    CampaignCancellationCommandRequest cancellation;
    cancellation.campaignId = campaignId.value();
    cancellation.requestId = requestId.value();
    cancellation.expectedRequestVersion = 1;
    cancellation.operationKey = "phase-g-settle-cancellation";
    cancellation.actorIdentity = "phase5.operator@example.test";
    cancellation.reason =
        "Cancel never-dispatched scope and settle the held reservation.";
    const auto cancelled = CancelCampaign(
        runtimeConnectionString, cancellation);
    assert(cancelled.progress == CancellationProgress::settled);
    assert(cancelled.settlement);
    assert(cancelled.settlement->settlement.disposition ==
        CancellationSettlementDisposition::unboundCancelled);

    // The mutex witness and immutable completion/audit facts must share one
    // transaction boundary.  Observe all three changes before abort, then prove
    // that rollback leaves no closed boundary and no partial authoritative row.
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_completion_writer;");
        const auto campaign = LockCompletionDomains(transaction, campaignId);
        assert(LoadCompletionBlockers(transaction, campaignId).empty());
        const auto rollbackEvent = BuildCompletionEventFromAuthority(
            transaction, campaign, "phase-g-completion-rollback",
            ActorIdentity("phase5.completer@example.test"),
            Reason("Abort completion to prove the durable boundary is atomic."));
        (void)PersistCompletionEvent(transaction, rollbackEvent);
        assert(transaction.exec(
            "SELECT completion_boundary_closed FROM "
            "campaign_operations_campaign WHERE operational_campaign_id=$1;",
            pqxx::params{campaignId.value()}).one_row()[0].as<bool>());
        assert(transaction.exec(
            "SELECT count(*) FROM campaign_operations_completion_event "
            "WHERE operational_campaign_id=$1;",
            pqxx::params{campaignId.value()}).one_row()[0].as<int>() == 1);
        assert(transaction.exec(
            "SELECT count(*) FROM "
            "campaign_operations_completion_audit_reference_event "
            "WHERE operational_campaign_id=$1;",
            pqxx::params{campaignId.value()}).one_row()[0].as<int>() == 1);
        transaction.abort();
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_completion_event", campaignId) == 0);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_completion_audit_reference_event",
               campaignId) == 0);
    {
        pqxx::read_transaction verify{owner};
        SetSearchPath(verify, schema);
        assert(!verify.exec(
            "SELECT completion_boundary_closed FROM "
            "campaign_operations_campaign WHERE operational_campaign_id=$1;",
            pqxx::params{campaignId.value()}).one_row()[0].as<bool>());
    }

    std::optional<PersistedCompletionEvent> recorded;
    {
        int preCommitBreakCount = 0;
        int absentInDoubtCount = 0;
        int persistedInDoubtCount = 0;
        const auto result = CompleteCampaignIfSettled(
            [&runtimeConnectionString]
            {
                return std::make_unique<pqxx::connection>(
                    runtimeConnectionString);
            },
            completionRequest,
            [&preCommitBreakCount, &absentInDoubtCount,
                &persistedInDoubtCount](
                CompletionTestInjectionPoint point)
            {
                if (point == CompletionTestInjectionPoint::
                        afterLocksBeforeEvidence &&
                    preCommitBreakCount == 0)
                {
                    ++preCommitBreakCount;
                    throw pqxx::broken_connection(
                        "deterministic pre-append connection loss");
                }
                if (point == CompletionTestInjectionPoint::beforeCommit &&
                    absentInDoubtCount == 0)
                {
                    ++absentInDoubtCount;
                    throw pqxx::in_doubt_error(
                        "deterministic absent uncertain completion commit");
                }
                if (point == CompletionTestInjectionPoint::
                        afterCommitBeforeResponse &&
                    persistedInDoubtCount == 0)
                {
                    ++persistedInDoubtCount;
                    throw pqxx::in_doubt_error(
                        "deterministic persisted uncertain completion commit");
                }
            });
        assert(result.disposition ==
            CompletionAttemptDisposition::existingIdentical);
        assert(preCommitBreakCount == 1);
        assert(absentInDoubtCount == 1);
        assert(persistedInDoubtCount == 1);
        assert(result.completion);
        recorded.emplace(*result.completion);
        assert(recorded->event.terminalState ==
            AdministrativeCampaignState::terminalCancelled);
        assert(recorded->event.classification ==
            CompletionClassification::allScopeCancelled);
        assert(recorded->event.scopeMemberCount == 2);
        assert(recorded->event.cancelledOrNeverDispatchedMemberCount == 2);
        assert(recorded->event.budgetHeld == 0);
        assert(recorded->event.budgetReleasedOrExpired == 2);
    }
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_completion_event", campaignId) == 1);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_completion_audit_reference_event",
               campaignId) == 1);

    // Shared C++/PostgreSQL golden vectors exercise byte-oriented UTF-8,
    // unsigned modulo-2^64 FNV-1a, fixed-width lower-case rendering, and the
    // complete V1 completion field order/framing contract.
    {
        const std::vector<std::string> hashVectors{
            "", "ordinary-ascii", std::string("\xC3\xA9\xE2\x98\x83", 5),
            "embedded;semicolon:colon=equals|pipe,comma",
            std::string(32768, 'x') + ";long:payload",
            "a", "0123456789abcdef0123456789abcdef"};
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        for (const auto& value : hashVectors)
        {
            const std::string cppHash =
                EA::ExperimentRecommendation::RecommendationCanonicalHash(
                    value);
            const std::string sqlHash = transaction.exec(
                "SELECT campaign_operations_tagged_fnv1a64($1);",
                pqxx::params{value}).one_row()[0].as<std::string>();
            assert(sqlHash == cppHash);
            assert(sqlHash.size() == 24U);
            assert(sqlHash.starts_with("fnv1a64:"));
        }
        assert(EA::ExperimentRecommendation::RecommendationCanonicalHash("a") ==
            "fnv1a64:af63dc4c8601ec8c");
        assert(EA::ExperimentRecommendation::RecommendationCanonicalHash(
                   "0123456789abcdef0123456789abcdef") ==
            "fnv1a64:01527c9731f0ff55");
    }
    {
        const std::array<CanonicalIdentity, 8> evidence{
            CanonicalIdentity::Create(1,
                "authorization;head=1:value:\xC3\xA9"),
            CanonicalIdentity::Create(1, "budget;empty_value=:;units=0"),
            CanonicalIdentity::Create(1,
                std::string(32768, 'r') + ";reservation:long"),
            CanonicalIdentity::Create(1, "request;delimiters=:;,|/"),
            CanonicalIdentity::Create(1, "binding;canonical=5:a:b;c"),
            CanonicalIdentity::Create(1,
                "lifecycle;status=completed;phase=train"),
            CanonicalIdentity::Create(1, "cancellation;count=0;items="),
            CanonicalIdentity::Create(1,
                "reconciliation;observations=0;resolutions=0")};
        const auto& source = recorded->event;
        const auto goldenCompletion = BuildCompletionEvent(source.campaignId,
            source.campaignCanonicalText, "golden:key/with:delimiters",
            source.terminalState, source.classification,
            source.budgetLedgerEntryId, source.budgetLedgerVersion,
            source.budgetResultingTotal, source.budgetEverReserved,
            source.budgetCommitted, source.budgetReleasedOrExpired,
            source.budgetHeld, source.budgetUnallocated,
            source.scopeMemberCount, source.completedMemberCount,
            source.failedMemberCount,
            source.cancelledOrNeverDispatchedMemberCount,
            source.reservationCount, source.requestCount, source.bindingCount,
            source.controlOwnerCount, source.cancellationRequestCount,
            source.cancellationSettlementCount,
            source.unresolvedBlockingObservationCount,
            evidence[0], evidence[1], evidence[2], evidence[3], evidence[4],
            evidence[5], evidence[6], evidence[7],
            ActorIdentity("golden.actor@example.test"),
            Reason("Golden reason; framing=: includes UTF-8 \xE2\x98\x83."));
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        const auto row = transaction.exec(
            "SELECT campaign_operations_completion_identity_valid("
            "jsonb_populate_record(NULL::campaign_operations_completion_event,"
            "to_jsonb(source)||jsonb_build_object("
            "'operation_key',$2::text,'authorization_evidence_canonical',$3::text,"
            "'authorization_evidence_hash',$4::text,"
            "'budget_evidence_canonical',$5::text,"
            "'budget_evidence_hash',$6::text,"
            "'reservation_evidence_canonical',$7::text,"
            "'reservation_evidence_hash',$8::text,"
            "'request_evidence_canonical',$9::text,"
            "'request_evidence_hash',$10::text,"
            "'binding_evidence_canonical',$11::text,"
            "'binding_evidence_hash',$12::text,"
            "'lifecycle_evidence_canonical',$13::text,"
            "'lifecycle_evidence_hash',$14::text,"
            "'cancellation_evidence_canonical',$15::text,"
            "'cancellation_evidence_hash',$16::text,"
            "'reconciliation_evidence_canonical',$17::text,"
            "'reconciliation_evidence_hash',$18::text,"
            "'actor_identity',$19::text,'reason',$20::text,"
            "'completion_identity_canonical',$21::text,"
            "'completion_identity_hash',$22::text))) AS identity_valid,"
            "campaign_operations_tagged_fnv1a64($21) AS sql_hash "
            "FROM campaign_operations_completion_event source "
            "WHERE source.operational_campaign_id=$1;",
            pqxx::params{campaignId.value(), goldenCompletion.operationKey,
                evidence[0].canonicalText(), evidence[0].hash(),
                evidence[1].canonicalText(), evidence[1].hash(),
                evidence[2].canonicalText(), evidence[2].hash(),
                evidence[3].canonicalText(), evidence[3].hash(),
                evidence[4].canonicalText(), evidence[4].hash(),
                evidence[5].canonicalText(), evidence[5].hash(),
                evidence[6].canonicalText(), evidence[6].hash(),
                evidence[7].canonicalText(), evidence[7].hash(),
                goldenCompletion.actor.value(), goldenCompletion.reason.value(),
                goldenCompletion.identity.canonicalText(),
                goldenCompletion.identity.hash()}).one_row();
        assert(row[0].as<bool>());
        assert(row[1].as<std::string>() == goldenCompletion.identity.hash());
    }

    // PostgreSQL must reject malformed identity material in its BEFORE INSERT
    // trigger, before a forged immutable row can consume either the campaign
    // or completion-event uniqueness domain.  Reusing the valid C++ row keeps
    // every unrelated authoritative field exact.
    const auto assertMalformedCompletionCopyDenied =
        [&](const std::string& field, const std::string& value)
    {
        bool deniedByIdentityValidation = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec(
                "SET LOCAL ROLE campaign_operations_completion_writer;");
            transaction.exec(
                "WITH candidate AS (SELECT jsonb_populate_record(NULL::"
                "campaign_operations_completion_event,to_jsonb(source)||"
                "jsonb_build_object($2::text,$3::text)) AS candidate_row "
                "FROM campaign_operations_completion_event source "
                "WHERE source.operational_campaign_id=$1) "
                "INSERT INTO campaign_operations_completion_event("
                "operational_campaign_id,campaign_identity_canonical,"
                "operation_key,administrative_terminal_state,"
                "completion_classification,budget_ledger_entry_id,"
                "budget_ledger_version,budget_resulting_total,"
                "budget_ever_reserved,budget_committed,"
                "budget_released_or_expired,budget_held,budget_unallocated,"
                "scope_member_count,completed_member_count,failed_member_count,"
                "cancelled_or_never_dispatched_member_count,reservation_count,"
                "request_count,binding_count,control_owner_count,"
                "cancellation_request_count,cancellation_settlement_count,"
                "unresolved_blocking_observation_count,"
                "authorization_evidence_canonical,authorization_evidence_hash,"
                "budget_evidence_canonical,budget_evidence_hash,"
                "reservation_evidence_canonical,reservation_evidence_hash,"
                "request_evidence_canonical,request_evidence_hash,"
                "binding_evidence_canonical,binding_evidence_hash,"
                "lifecycle_evidence_canonical,lifecycle_evidence_hash,"
                "cancellation_evidence_canonical,cancellation_evidence_hash,"
                "reconciliation_evidence_canonical,"
                "reconciliation_evidence_hash,actor_identity,capability,reason,"
                "completion_contract_version,completion_identity_canonical,"
                "completion_identity_hash) SELECT "
                "(candidate_row).operational_campaign_id,"
                "(candidate_row).campaign_identity_canonical,"
                "(candidate_row).operation_key,"
                "(candidate_row).administrative_terminal_state,"
                "(candidate_row).completion_classification,"
                "(candidate_row).budget_ledger_entry_id,"
                "(candidate_row).budget_ledger_version,"
                "(candidate_row).budget_resulting_total,"
                "(candidate_row).budget_ever_reserved,"
                "(candidate_row).budget_committed,"
                "(candidate_row).budget_released_or_expired,"
                "(candidate_row).budget_held,"
                "(candidate_row).budget_unallocated,"
                "(candidate_row).scope_member_count,"
                "(candidate_row).completed_member_count,"
                "(candidate_row).failed_member_count,"
                "(candidate_row).cancelled_or_never_dispatched_member_count,"
                "(candidate_row).reservation_count,"
                "(candidate_row).request_count,(candidate_row).binding_count,"
                "(candidate_row).control_owner_count,"
                "(candidate_row).cancellation_request_count,"
                "(candidate_row).cancellation_settlement_count,"
                "(candidate_row).unresolved_blocking_observation_count,"
                "(candidate_row).authorization_evidence_canonical,"
                "(candidate_row).authorization_evidence_hash,"
                "(candidate_row).budget_evidence_canonical,"
                "(candidate_row).budget_evidence_hash,"
                "(candidate_row).reservation_evidence_canonical,"
                "(candidate_row).reservation_evidence_hash,"
                "(candidate_row).request_evidence_canonical,"
                "(candidate_row).request_evidence_hash,"
                "(candidate_row).binding_evidence_canonical,"
                "(candidate_row).binding_evidence_hash,"
                "(candidate_row).lifecycle_evidence_canonical,"
                "(candidate_row).lifecycle_evidence_hash,"
                "(candidate_row).cancellation_evidence_canonical,"
                "(candidate_row).cancellation_evidence_hash,"
                "(candidate_row).reconciliation_evidence_canonical,"
                "(candidate_row).reconciliation_evidence_hash,"
                "(candidate_row).actor_identity,(candidate_row).capability,"
                "(candidate_row).reason,"
                "(candidate_row).completion_contract_version,"
                "(candidate_row).completion_identity_canonical,"
                "(candidate_row).completion_identity_hash FROM candidate;",
                pqxx::params{campaignId.value(), field, value});
            transaction.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            deniedByIdentityValidation = error.sqlstate() == "23514" &&
                std::string(error.what()).find(
                    "completion canonical or hash mismatch") !=
                    std::string::npos;
        }
        assert(deniedByIdentityValidation);
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_completion_event", campaignId) == 1);
    };
    for (const auto* hashField : {
             "authorization_evidence_hash", "budget_evidence_hash",
             "reservation_evidence_hash", "request_evidence_hash",
             "binding_evidence_hash", "lifecycle_evidence_hash",
             "cancellation_evidence_hash", "reconciliation_evidence_hash"})
        assertMalformedCompletionCopyDenied(
            hashField, "fnv1a64:0000000000000000");
    assertMalformedCompletionCopyDenied("completion_identity_canonical",
        recorded->event.identity.canonicalText() + ";changed=true");
    assertMalformedCompletionCopyDenied(
        "completion_identity_hash", "fnv1a64:0000000000000000");
    // Keeping the original hash while changing full canonical text proves that
    // hash equality is never accepted as completion identity.
    assertMalformedCompletionCopyDenied("completion_identity_canonical",
        recorded->event.identity.canonicalText() +
            ";same_hash_different_canonical=true");
    assertMalformedCompletionCopyDenied("completion_identity_canonical",
        "campaign_operations_completion_v1");

    std::string boundaryXminBeforeReplay;
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        const auto boundary = transaction.exec(
            "SELECT xmin::text,completion_boundary_closed,("
            "SELECT count(*) FROM campaign_operations_completion_event event "
            "WHERE event.operational_campaign_id=campaign.operational_campaign_id) "
            "FROM campaign_operations_campaign campaign "
            "WHERE operational_campaign_id=$1;",
            pqxx::params{campaignId.value()}).one_row();
        boundaryXminBeforeReplay = boundary[0].as<std::string>();
        assert(boundary[1].as<bool>());
        assert(boundary[2].as<int>() == 1);
    }
    {
        pqxx::connection connection{runtimeConnectionString};
        const auto replay =
            CompleteCampaignIfSettled(connection, completionRequest);
        assert(replay.disposition ==
            CompletionAttemptDisposition::existingIdentical);
        assert(replay.completion);
        assert(replay.completion->completionEventId ==
            recorded->completionEventId);
        auto conflicting = completionRequest;
        conflicting.operationKey = "phase-g-conflicting-completion";
        const auto conflict =
            CompleteCampaignIfSettled(connection, conflicting);
        assert(conflict.disposition ==
            CompletionAttemptDisposition::conflictingReplay);
        assert(conflict.completion);
        assert(conflict.completion->completionEventId ==
            recorded->completionEventId);
        auto actorConflict = completionRequest;
        actorConflict.actorIdentity = "different.completer@example.test";
        assert(CompleteCampaignIfSettled(connection, actorConflict).
                   disposition ==
            CompletionAttemptDisposition::conflictingReplay);
        auto reasonConflict = completionRequest;
        reasonConflict.reason =
            "A changed completion reason must not mutate terminal truth.";
        assert(CompleteCampaignIfSettled(connection, reasonConflict).
                   disposition ==
            CompletionAttemptDisposition::conflictingReplay);
        const auto status =
            LoadCampaignCompletionStatus(connection, campaignId);
        assert(status.completionRecorded);
        assert(status.completion);
        assert(status.currentOperationalState ==
            AdministrativeCampaignState::terminalCancelled);
        assert(!status.postCompletionLifecycleChanged);
        assert(status.completion->event.classification ==
            CompletionClassification::allScopeCancelled);
    }
    {
        pqxx::read_transaction transaction{owner};
        SetSearchPath(transaction, schema);
        const auto boundary = transaction.exec(
            "SELECT xmin::text,completion_boundary_closed FROM "
            "campaign_operations_campaign WHERE operational_campaign_id=$1;",
            pqxx::params{campaignId.value()}).one_row();
        assert(boundary[0].as<std::string>() == boundaryXminBeforeReplay);
        assert(boundary[1].as<bool>());
    }

    // A fresh process has no retained attempted event.  The connection-factory
    // path must rebuild the complete authoritative candidate before deciding
    // either exact replay or conflict.
    {
        const auto freshReplay = CompleteCampaignIfSettled(
            [&runtimeConnectionString]
            {
                return std::make_unique<pqxx::connection>(
                    runtimeConnectionString);
            }, completionRequest);
        assert(freshReplay.disposition ==
            CompletionAttemptDisposition::existingIdentical);
        auto changed = completionRequest;
        changed.reason =
            "A restart must compare the complete changed completion event.";
        const auto freshConflict = CompleteCampaignIfSettled(
            [&runtimeConnectionString]
            {
                return std::make_unique<pqxx::connection>(
                    runtimeConnectionString);
            }, changed);
        assert(freshConflict.disposition ==
            CompletionAttemptDisposition::conflictingReplay);
    }

    // Repeated failures while resolving an uncertain outcome fail closed.  No
    // append is retried because the durable campaign lookup never completes.
    {
        int lookupFailures = 0;
        bool ambiguous = false;
        try
        {
            (void)CompleteCampaignIfSettled(
                [&runtimeConnectionString]
                {
                    return std::make_unique<pqxx::connection>(
                        runtimeConnectionString);
                }, completionRequest,
                [&lookupFailures](CompletionTestInjectionPoint point)
                {
                    if (point == CompletionTestInjectionPoint::
                            duringOutcomeLookup)
                    {
                        ++lookupFailures;
                        throw pqxx::in_doubt_error(
                            "deterministic repeated outcome lookup failure");
                    }
                });
        }
        catch (const Error& error)
        {
            ambiguous = error.code() == ErrorCode::persistenceConflict &&
                std::string(error.what()) ==
                    "campaign_operations_completion_outcome_ambiguous";
        }
        assert(ambiguous);
        assert(lookupFailures == 4);
        assert(CountRowsForCampaign(owner, schema,
                   "campaign_operations_completion_event", campaignId) == 1);
    }

    {
        bool insertDenied = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec("SET LOCAL ROLE campaign_operations_reader;");
            transaction.exec(
                "INSERT INTO campaign_operations_completion_event "
                "DEFAULT VALUES;");
            transaction.commit();
        }
        catch (const pqxx::sql_error&)
        {
            insertDenied = true;
        }
        assert(insertDenied);
        bool updateDenied = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec("SET LOCAL ROLE campaign_operations_reader;");
            transaction.exec(
                "UPDATE campaign_operations_completion_event "
                "SET reason='forbidden' WHERE completion_event_id=$1;",
                pqxx::params{recorded->completionEventId.value()});
            transaction.commit();
        }
        catch (const pqxx::sql_error&)
        {
            updateDenied = true;
        }
        assert(updateDenied);
        bool deleteDenied = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec("SET LOCAL ROLE campaign_operations_reader;");
            transaction.exec(
                "DELETE FROM campaign_operations_completion_event "
                "WHERE completion_event_id=$1;",
                pqxx::params{recorded->completionEventId.value()});
            transaction.commit();
        }
        catch (const pqxx::sql_error&)
        {
            deleteDenied = true;
        }
        assert(deleteDenied);
        bool ownerMutationRejected = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec(
                "UPDATE campaign_operations_completion_event "
                "SET reason='owner mutation forbidden' "
                "WHERE completion_event_id=$1;",
                pqxx::params{recorded->completionEventId.value()});
            transaction.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            ownerMutationRejected = error.sqlstate() == "55000";
        }
        assert(ownerMutationRejected);
        bool ownerBoundaryReopenRejected = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec(
                "UPDATE campaign_operations_campaign "
                "SET completion_boundary_closed=false "
                "WHERE operational_campaign_id=$1;",
                pqxx::params{campaignId.value()});
            transaction.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            ownerBoundaryReopenRejected = error.sqlstate() == "55000";
        }
        assert(ownerBoundaryReopenRejected);
        bool ownerCompletedCampaignDeleteRejected = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec(
                "DELETE FROM campaign_operations_campaign "
                "WHERE operational_campaign_id=$1;",
                pqxx::params{campaignId.value()});
            transaction.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            ownerCompletedCampaignDeleteRejected =
                error.sqlstate() == "55000";
        }
        assert(ownerCompletedCampaignDeleteRejected);
        bool ownerTruncateRejected = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec(
                "TRUNCATE campaign_operations_completion_audit_reference_event, "
                "campaign_operations_completion_event;");
            transaction.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            ownerTruncateRejected = error.sqlstate() == "55000";
        }
        assert(ownerTruncateRejected);
        bool ownerCampaignTruncateRejected = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec("TRUNCATE campaign_operations_campaign CASCADE;");
            transaction.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            ownerCampaignTruncateRejected = error.sqlstate() == "55000";
        }
        assert(ownerCampaignTruncateRejected);
    }

    {
        pqxx::connection connection{runtimeConnectionString};
        bool controlBlocked = false;
        try
        {
            (void)ControlCampaign(connection,
                {campaignId.value(), 2, ControlEventKind::pause,
                    "phase5.operator@example.test",
                    "A completed campaign cannot be reopened by pause."});
        }
        catch (const pqxx::sql_error& error)
        {
            controlBlocked = error.sqlstate() == "23514";
        }
        assert(controlBlocked);
    }

    // Database defense-in-depth for every mutation family represented by the
    // settled cancellation fixture.  Exact service replay was proven above;
    // any attempted new row or guarded projection write must fail at the
    // immutable completion boundary before a uniqueness or state constraint.
    const auto assertCompletionGate = [&](const std::string& statement)
    {
        bool denied = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec(statement, pqxx::params{campaignId.value()});
            transaction.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            denied = error.sqlstate() == "23514";
        }
        assert(denied);
    };
    for (const auto* table : {
             "campaign_operations_authorization_event",
             "campaign_operations_budget_ledger_entry",
             "campaign_operations_reservation",
             "campaign_operations_operational_request",
             "campaign_operations_control_event",
             "campaign_operations_cancellation_request",
             "campaign_operations_audit_reference_event",
             "campaign_operations_control_audit_reference_event"})
        assertCompletionGate("INSERT INTO " + std::string(table) +
            " SELECT * FROM " + table +
            " WHERE operational_campaign_id=$1 LIMIT 1;");
    assertCompletionGate(
        "INSERT INTO campaign_operations_reservation_event "
        "SELECT event.* FROM campaign_operations_reservation_event event "
        "JOIN campaign_operations_reservation reservation USING(reservation_id) "
        "WHERE reservation.operational_campaign_id=$1 LIMIT 1;");
    assertCompletionGate(
        "INSERT INTO campaign_operations_cancellation_settlement "
        "SELECT settlement.* "
        "FROM campaign_operations_cancellation_settlement settlement "
        "JOIN campaign_operations_cancellation_request cancellation "
        "USING(cancellation_request_id) "
        "WHERE cancellation.operational_campaign_id=$1 LIMIT 1;");
    assertCompletionGate(
        "UPDATE campaign_operations_reservation SET state_version=state_version "
        "WHERE operational_campaign_id=$1;");
    assertCompletionGate(
        "UPDATE campaign_operations_operational_request "
        "SET state_version=state_version WHERE operational_campaign_id=$1;");

    const auto terminalGateRaceAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 247, 1);
    const auto terminalGateRaceCampaignId = terminalGateRaceAccepted.persisted.
        request.request.logicalOperation.campaignId;
    CampaignCancellationCommandRequest terminalGateRaceCancellation;
    terminalGateRaceCancellation.campaignId =
        terminalGateRaceCampaignId.value();
    terminalGateRaceCancellation.requestId = terminalGateRaceAccepted.
        persisted.request.requestId.value();
    terminalGateRaceCancellation.expectedRequestVersion = 1;
    terminalGateRaceCancellation.operationKey = "phase-g-terminal-gate-race";
    terminalGateRaceCancellation.actorIdentity =
        "phase5.operator@example.test";
    terminalGateRaceCancellation.reason =
        "Settle the deterministic terminal gate race fixture.";
    assert(CancelCampaign(runtimeConnectionString,
               terminalGateRaceCancellation).progress ==
        CancellationProgress::settled);
    const CompleteIfSettledRequest terminalGateCompletionRequest{
        terminalGateRaceCampaignId.value(),
        "phase-g-terminal-gate-race-completion",
        "phase5.completer@example.test",
        "Hold completion locks while Phase F mutation paths contend."};
    std::atomic<bool> completionLocksHeld{false};
    std::atomic<bool> releaseCompletion{false};
    std::atomic<int> mutationStarted{0};
    std::array<std::atomic<int>, 4> mutationBackendPids{};
    std::atomic<int> mutationDenied{0};
    std::exception_ptr terminalGateCompletionError;
    std::optional<CompleteIfSettledResult> terminalGateCompletion;
    std::thread terminalCompleter{[&]
    {
        try
        {
            terminalGateCompletion.emplace(CompleteCampaignIfSettled(
                [&runtimeConnectionString]
                {
                    return std::make_unique<pqxx::connection>(
                        runtimeConnectionString);
                },
                terminalGateCompletionRequest,
                [&](CompletionTestInjectionPoint point)
                {
                    if (point != CompletionTestInjectionPoint::
                            afterLocksBeforeEvidence)
                        return;
                    completionLocksHeld.store(true,
                        std::memory_order_release);
                    while (!releaseCompletion.load(
                        std::memory_order_acquire))
                        std::this_thread::yield();
                }));
        }
        catch (...)
        {
            terminalGateCompletionError = std::current_exception();
        }
    }};
    while (!completionLocksHeld.load(std::memory_order_acquire))
        std::this_thread::yield();
    const std::array<std::string, 4> racingMutations{
        "UPDATE campaign_operations_reservation SET state_version=state_version "
        "WHERE operational_campaign_id=$1;",
        "UPDATE campaign_operations_operational_request "
        "SET state_version=state_version WHERE operational_campaign_id=$1;",
        "INSERT INTO campaign_operations_reservation_event "
        "SELECT event.* FROM campaign_operations_reservation_event event "
        "JOIN campaign_operations_reservation reservation USING(reservation_id) "
        "WHERE reservation.operational_campaign_id=$1 LIMIT 1;",
        "INSERT INTO campaign_operations_cancellation_settlement "
        "SELECT settlement.* FROM "
        "campaign_operations_cancellation_settlement settlement "
        "JOIN campaign_operations_cancellation_request cancellation "
        "USING(cancellation_request_id) "
        "WHERE cancellation.operational_campaign_id=$1 LIMIT 1;"};
    std::vector<std::thread> mutationThreads;
    mutationThreads.reserve(racingMutations.size());
    for (std::size_t index = 0; index < racingMutations.size(); ++index)
        mutationThreads.emplace_back([&, index]
        {
            try
            {
                pqxx::connection connection{runtimeConnectionString};
                mutationBackendPids[index].store(
                    connection.backendpid(), std::memory_order_release);
                pqxx::work transaction{connection};
                mutationStarted.fetch_add(1, std::memory_order_release);
                transaction.exec(racingMutations[index],
                    pqxx::params{terminalGateRaceCampaignId.value()});
                transaction.commit();
            }
            catch (const pqxx::sql_error& error)
            {
                if (error.sqlstate() == "23514")
                    mutationDenied.fetch_add(1,
                        std::memory_order_release);
            }
        });
    while (mutationStarted.load(std::memory_order_acquire) !=
           static_cast<int>(racingMutations.size()))
        std::this_thread::yield();
    bool allMutationsObservedWaitingOnLock = false;
    for (int observation = 0;
         observation < 500 && !allMutationsObservedWaitingOnLock;
         ++observation)
    {
        pqxx::read_transaction observer{owner};
        SetSearchPath(observer, schema);
        allMutationsObservedWaitingOnLock = true;
        for (const auto& backendPid : mutationBackendPids)
            allMutationsObservedWaitingOnLock =
                allMutationsObservedWaitingOnLock && observer.exec(
                    "SELECT cardinality(pg_blocking_pids($1)) > 0;",
                    pqxx::params{backendPid.load(std::memory_order_acquire)})
                    .one_row()[0].as<bool>();
        if (!allMutationsObservedWaitingOnLock)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    releaseCompletion.store(true, std::memory_order_release);
    terminalCompleter.join();
    for (auto& mutation : mutationThreads) mutation.join();
    assert(allMutationsObservedWaitingOnLock);
    if (terminalGateCompletionError)
        std::rethrow_exception(terminalGateCompletionError);
    assert(terminalGateCompletion);
    assert(terminalGateCompletion->disposition ==
        CompletionAttemptDisposition::recorded);
    assert(mutationDenied.load(std::memory_order_acquire) ==
        static_cast<int>(racingMutations.size()));
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_completion_event",
               terminalGateRaceCampaignId) == 1);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_completion_audit_reference_event",
               terminalGateRaceCampaignId) == 1);

    const auto concurrentAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 241, 1);
    const auto concurrentCampaignId =
        concurrentAccepted.persisted.request.request.logicalOperation.campaignId;
    CampaignCancellationCommandRequest concurrentCancellation;
    concurrentCancellation.campaignId = concurrentCampaignId.value();
    concurrentCancellation.requestId =
        concurrentAccepted.persisted.request.requestId.value();
    concurrentCancellation.expectedRequestVersion = 1;
    concurrentCancellation.operationKey = "phase-g-concurrent-cancel";
    concurrentCancellation.actorIdentity =
        "phase5.operator@example.test";
    concurrentCancellation.reason =
        "Settle the concurrent completion fixture.";
    assert(CancelCampaign(runtimeConnectionString, concurrentCancellation)
               .progress == CancellationProgress::settled);
    CompleteIfSettledRequest concurrentRequest{
        concurrentCampaignId.value(), "phase-g-concurrent-completion",
        "phase5.completer@example.test",
        "Concurrent exact completion must converge."};
    std::barrier start{3};
    std::optional<CompleteIfSettledResult> firstResult;
    std::optional<CompleteIfSettledResult> secondResult;
    std::exception_ptr firstError;
    std::exception_ptr secondError;
    const auto complete = [&](std::optional<CompleteIfSettledResult>& result,
                              std::exception_ptr& error)
    {
        try
        {
            pqxx::connection connection{runtimeConnectionString};
            start.arrive_and_wait();
            result.emplace(
                CompleteCampaignIfSettled(connection, concurrentRequest));
        }
        catch (...)
        {
            error = std::current_exception();
        }
    };
    std::thread first{complete, std::ref(firstResult),
        std::ref(firstError)};
    std::thread second{complete, std::ref(secondResult),
        std::ref(secondError)};
    start.arrive_and_wait();
    first.join();
    second.join();
    if (firstError) std::rethrow_exception(firstError);
    if (secondError) std::rethrow_exception(secondError);
    assert(firstResult && secondResult);
    assert(firstResult->completion && secondResult->completion);
    assert(firstResult->completion->completionEventId ==
        secondResult->completion->completionEventId);
    assert((firstResult->disposition ==
                CompletionAttemptDisposition::recorded &&
               secondResult->disposition ==
                CompletionAttemptDisposition::existingIdentical) ||
        (secondResult->disposition ==
                CompletionAttemptDisposition::recorded &&
            firstResult->disposition ==
                CompletionAttemptDisposition::existingIdentical));
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_completion_event",
               concurrentCampaignId) == 1);

    const auto cancellationRaceAccepted = AcceptPhase4Fixture(
        owner, connectionString, schema, 245, 1);
    const auto cancellationRaceCampaignId =
        cancellationRaceAccepted.persisted.request.request.logicalOperation.
            campaignId;
    CampaignCancellationCommandRequest cancellationRaceRequest;
    cancellationRaceRequest.campaignId =
        cancellationRaceCampaignId.value();
    cancellationRaceRequest.requestId =
        cancellationRaceAccepted.persisted.request.requestId.value();
    cancellationRaceRequest.expectedRequestVersion = 1;
    cancellationRaceRequest.operationKey = "phase-g-cancellation-race";
    cancellationRaceRequest.actorIdentity =
        "phase5.operator@example.test";
    cancellationRaceRequest.reason =
        "Race exact cancellation settlement with completion evaluation.";
    CompleteIfSettledRequest completionRaceRequest{
        cancellationRaceCampaignId.value(), "phase-g-completion-race",
        "phase5.completer@example.test",
        "Completion must serialize with cancellation settlement."};
    std::barrier cancellationRaceStart{3};
    std::optional<CompleteIfSettledResult> completionRaceResult;
    std::optional<CampaignCancellationResult> cancellationRaceResult;
    std::exception_ptr completionRaceError;
    std::exception_ptr cancellationRaceError;
    std::thread completionRacer{[&]
    {
        try
        {
            pqxx::connection connection{runtimeConnectionString};
            cancellationRaceStart.arrive_and_wait();
            completionRaceResult.emplace(
                CompleteCampaignIfSettled(connection,
                    completionRaceRequest));
        }
        catch (...)
        {
            completionRaceError = std::current_exception();
        }
    }};
    std::thread cancellationRacer{[&]
    {
        try
        {
            cancellationRaceStart.arrive_and_wait();
            cancellationRaceResult.emplace(
                CancelCampaign(runtimeConnectionString,
                    cancellationRaceRequest));
        }
        catch (...)
        {
            cancellationRaceError = std::current_exception();
        }
    }};
    cancellationRaceStart.arrive_and_wait();
    completionRacer.join();
    cancellationRacer.join();
    if (completionRaceError) std::rethrow_exception(completionRaceError);
    if (cancellationRaceError) std::rethrow_exception(cancellationRaceError);
    assert(cancellationRaceResult);
    assert(cancellationRaceResult->progress ==
        CancellationProgress::settled);
    assert(completionRaceResult);
    if (completionRaceResult->disposition ==
        CompletionAttemptDisposition::blocked)
    {
        pqxx::connection connection{runtimeConnectionString};
        completionRaceResult.emplace(
            CompleteCampaignIfSettled(connection, completionRaceRequest));
    }
    assert(completionRaceResult->disposition ==
            CompletionAttemptDisposition::recorded ||
        completionRaceResult->disposition ==
            CompletionAttemptDisposition::existingIdentical);
    assert(CountRowsForCampaign(owner, schema,
               "campaign_operations_completion_event",
               cancellationRaceCampaignId) == 1);
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
        ApplyFile(owner, schema,
            "Database/migrations/048_campaign_operations_durable_dispatch_handoff.sql");
        ApplyFile(owner, schema,
            "Database/migrations/048_campaign_operations_durable_dispatch_handoff.sql");
        ApplyFile(owner, schema,
            "Tests/CampaignOperationsPhase3MigrationTests.sql");
        ApplyFile(owner, schema,
            "Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql");
        ApplyFile(owner, schema,
            "Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql");
        ApplyFile(owner, schema,
            "Tests/CampaignOperationsPhase4MigrationTests.sql");
        ApplyFile(owner, schema,
            "Database/migrations/054_campaign_operations_completion_and_audit.sql");
        ApplyFile(owner, schema,
            "Database/migrations/054_campaign_operations_completion_and_audit.sql");
        ApplyFile(owner, schema,
            "Tests/CampaignOperationsPhase5MigrationTests.sql");

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
        TestPhase3LeaseAcquisition(owner, connectionString, schema);
        TestPhase4ControlsCancellationAndReconciliation(
            owner, connectionString, schema);
        TestPhase5OperationalCompletion(
            owner, connectionString, schema);
        // Simulate the upgrade shape produced by the earlier Phase G draft:
        // completion rows exist, but the durable tuple witness was not yet
        // protected.  Replaying 054 must backfill the witness before restoring
        // the invariant triggers.
        {
            pqxx::work upgradeFixture{owner};
            SetSearchPath(upgradeFixture, schema);
            upgradeFixture.exec(
                "DROP TRIGGER campaign_operations_completion_boundary_update_guard "
                "ON campaign_operations_campaign;"
                "DROP TRIGGER campaign_operations_completion_boundary_truncate_guard "
                "ON campaign_operations_campaign;"
                "DROP TRIGGER "
                "campaign_operations_completion_boundary_consistency_trigger "
                "ON campaign_operations_campaign;"
                "UPDATE campaign_operations_campaign campaign "
                "SET completion_boundary_closed=false WHERE EXISTS ("
                "SELECT 1 FROM campaign_operations_completion_event completion "
                "WHERE completion.operational_campaign_id="
                "campaign.operational_campaign_id);" );
            upgradeFixture.commit();
        }
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM campaign_operations_completion_event") > 0);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM campaign_operations_campaign campaign "
            "WHERE NOT completion_boundary_closed AND EXISTS ("
            "SELECT 1 FROM campaign_operations_completion_event completion "
            "WHERE completion.operational_campaign_id="
            "campaign.operational_campaign_id)") > 0);
        ApplyFile(owner, schema,
            "Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql");
        ApplyFile(owner, schema,
            "Tests/CampaignOperationsPhase4MigrationTests.sql");
        ApplyFile(owner, schema,
            "Database/migrations/054_campaign_operations_completion_and_audit.sql");
        ApplyFile(owner, schema,
            "Tests/CampaignOperationsPhase5MigrationTests.sql");
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM campaign_operations_campaign campaign "
            "WHERE completion_boundary_closed <> EXISTS ("
            "SELECT 1 FROM campaign_operations_completion_event completion "
            "WHERE completion.operational_campaign_id="
            "campaign.operational_campaign_id)") == 0);
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
    catch (const std::exception& error)
    {
        std::cerr << "CampaignOperationsRepositoryTests_failed: "
                  << error.what() << '\n';
        try
        {
            DropSchema(owner, schema);
        }
        catch (...)
        {
        }
        return 1;
    }
    return 0;
}
