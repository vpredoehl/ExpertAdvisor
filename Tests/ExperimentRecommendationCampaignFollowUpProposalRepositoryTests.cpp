#include "../Sources/ExperimentRecommendationCampaignFollowUpProposalPreview.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include <pqxx/pqxx>

using namespace EA::ExperimentRecommendation;

namespace
{

using Assessment = RecommendationCampaignOutcomeAssessment;
using CampaignIdentity =
    RecommendationCampaignOutcomeAssessmentCampaignIdentity;
using Context = RecommendationCampaignOutcomeAssessmentComparisonContext;
using Lifecycle = RecommendationCampaignOutcomeAssessmentLifecycleState;
using MaterializationIdentity =
    RecommendationCampaignOutcomeAssessmentMaterializationIdentity;
using MemberEvidence = RecommendationCampaignOutcomeAssessmentMemberEvidence;
using MemberIdentity = RecommendationCampaignOutcomeAssessmentMemberIdentity;
using Metric = RecommendationCampaignOutcomeAssessmentMetric;
using MetricCollection =
    RecommendationCampaignOutcomeAssessmentMetricCollection;
using MetricSupport =
    RecommendationCampaignOutcomeAssessmentMetricComparisonSupport;
using PolicyInput = RecommendationCampaignOutcomePolicyInput;
using Proposal = RecommendationCampaignFollowUpProposal;
using ResultEvidence =
    RecommendationCampaignOutcomeAssessmentResultEvidence;
using ResultIdentity =
    RecommendationCampaignOutcomeAssessmentResultIdentity;
using SourceEvidence =
    RecommendationCampaignOutcomeAssessmentSourceEvidence;

static_assert(!std::is_copy_assignable_v<Proposal>);
static_assert(!std::is_move_assignable_v<Proposal>);
static_assert(!std::is_copy_assignable_v<
    PersistedRecommendationCampaignFollowUpProposal>);
static_assert(!std::is_move_assignable_v<
    PersistedRecommendationCampaignFollowUpProposal>);

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

bool SafeDisposableDatabaseName(const std::string& value)
{
    if (value.empty()) return false;
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](unsigned char character)
        { return static_cast<char>(std::tolower(character)); });
    if (lower == "lstm") return false;
    return std::all_of(value.begin(), value.end(),
        [](unsigned char character)
        { return std::isalnum(character) || character == '_' ||
                 character == '-'; });
}

bool PermissionDenied(const pqxx::sql_error& error)
{
    return error.sqlstate() == "42501" ||
        (error.sqlstate().empty() &&
            std::string(error.what()).find("permission denied") !=
                std::string::npos);
}

Context ComparisonContext(int ordinal)
{
    return {"eurusd", 12, 0.01, 24,
        "phase6b-label-" + std::to_string(ordinal), "2026-06-01",
        "2026-06-30"};
}

Metric Numeric(std::string identity, double value)
{
    return {std::move(identity), value, MetricSupport::NumericDelta};
}

MemberEvidence FavorableMember(int ordinal)
{
    const long long offset = ordinal - 1;
    MemberIdentity identity(ordinal, 101 + offset, 201 + offset,
        301 + offset, 401 + offset, 501 + offset, 601 + offset);
    return {identity, Lifecycle::Succeeded,
        SourceEvidence(identity.sourceExperimentId, 701 + offset,
            801 + offset, ComparisonContext(ordinal),
            MetricCollection({Numeric("inference_accuracy", 0.70),
                Numeric("leader_score", 0.80)})),
        ResultEvidence(identity.expectedExperimentId, 901 + offset,
            {ResultIdentity{"inference", 1001 + offset}},
            ComparisonContext(ordinal),
            MetricCollection({Numeric("inference_accuracy", 0.75),
                Numeric("leader_score", 0.85)}))};
}

Proposal BuildProposal()
{
    const std::string campaignCanonical = "phase6b-campaign";
    CampaignIdentity campaign(10, campaignCanonical,
        RecommendationCanonicalHash(campaignCanonical));
    const std::string materializationCanonical = "phase6b-materialization";
    MaterializationIdentity materialization(20, campaign.campaignApprovalId,
        campaign.identityHash, 1, 2, materializationCanonical,
        RecommendationCanonicalHash(materializationCanonical));
    const Assessment assessment =
        BuildRecommendationCampaignOutcomeAssessment(campaign,
            materialization, "2026-07-20 12:00:00+00",
            {FavorableMember(1), FavorableMember(2)});
    const auto policy = BuildRecommendationCampaignOutcomePolicy(PolicyInput{});
    const auto decision = ApplyRecommendationCampaignOutcomePolicy(
        policy, assessment);
    return BuildRecommendationCampaignFollowUpProposal(assessment, decision);
}

void SetSearchPath(pqxx::transaction_base& transaction,
    const std::string& schema)
{
    transaction.exec(
        "SET LOCAL search_path TO " + transaction.quote_name(schema) + ";");
}

std::vector<long long> Phase6BSequenceValues(
    pqxx::connection& owner,
    const std::string& schema)
{
    pqxx::read_transaction transaction{owner};
    const pqxx::result rows = transaction.exec(
        "SELECT last_value FROM pg_sequences WHERE schemaname=$1 AND "
        "sequencename<>'phase6b_preview_read_sentinel' ORDER BY "
        "sequencename;", pqxx::params{schema});
    std::vector<long long> values;
    values.reserve(rows.size());
    for (const auto& row : rows) values.push_back(row[0].as<long long>());
    return values;
}

template <typename Corruption>
void AssertMalformedPersistenceRejected(pqxx::connection& owner,
    const std::string& schema, long long id, Corruption&& corrupt)
{
    bool rejected = false;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        corrupt(transaction);
        try
        {
            (void)FindRecommendationCampaignFollowUpProposal(transaction, id);
        }
        catch (const std::runtime_error& error)
        {
            rejected = std::string(error.what()).starts_with(
                "invalid_persisted_recommendation_campaign_follow_up_proposal:");
        }
    }
    assert(rejected);
}

} // namespace

int main()
{
    const char* testDatabase = std::getenv("LSTM_TEST_DB_NAME");
    if (testDatabase == nullptr || *testDatabase == '\0')
    {
        std::cerr << "LSTM_TEST_DB_NAME_required\n";
        return 2;
    }
    const std::string database = testDatabase;
    if (!SafeDisposableDatabaseName(database))
    {
        std::cerr << "active_LSTM_database_forbidden\n";
        return 2;
    }
    const std::string host = EnvironmentOr("LSTM_TEST_DB_HOST", "127.0.0.1");
    const std::string port = EnvironmentOr("LSTM_TEST_DB_PORT", "5432");
    const std::string ownerUser = EnvironmentOr(
        "LSTM_TEST_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string schema =
        "phase6b_follow_up_proposal_" + std::to_string(getpid());
    const std::string ownerConnectionString =
        "host=" + host + " port=" + port + " user=" + ownerUser +
        " dbname=" + database;
    const std::string runtimeConnectionString =
        "host=" + host + " port=" + port + " user=pqxx dbname=" + database +
        " options='-c search_path=" + schema + "'";
    pqxx::connection owner{ownerConnectionString};

    try
    {
        {
            pqxx::work setup{owner};
            setup.exec("CREATE SCHEMA " + setup.quote_name(schema) + ";");
            SetSearchPath(setup, schema);
            setup.exec(R"SQL(
CREATE SEQUENCE phase6b_preview_read_sentinel;
CREATE TABLE experiment_recommendation_campaign_approval(
    recommendation_campaign_approval_id bigint PRIMARY KEY,
    approval_identity_canonical text NOT NULL,
    approval_identity_hash text NOT NULL);
CREATE TABLE experiment_recommendation_campaign_materialization(
    recommendation_campaign_materialization_id bigint PRIMARY KEY,
    recommendation_campaign_approval_id bigint NOT NULL,
    materialization_contract_version integer NOT NULL,
    approval_identity_hash text NOT NULL,
    selected_member_count integer NOT NULL,
    materialization_identity_canonical text NOT NULL,
    materialization_identity_hash text NOT NULL);
CREATE TABLE experiment_recommendation_campaign_materialization_member(
    recommendation_campaign_materialization_member_id bigint PRIMARY KEY,
    recommendation_campaign_materialization_id bigint NOT NULL,
    member_ordinal integer NOT NULL,
    recommendation_ranking_member_id bigint NOT NULL,
    recommendation_id bigint NOT NULL,
    source_experiment_id bigint NOT NULL,
    recommendation_conversion_proposal_id bigint NOT NULL);
)SQL");
            setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                " TO pqxx; GRANT SELECT ON ALL TABLES IN SCHEMA " +
                setup.quote_name(schema) + " TO pqxx;");

            const std::string migration = ReadFile(
                "Database/migrations/042_experiment_recommendation_campaign_follow_up_proposal.sql");
            setup.exec(migration);
            setup.exec(migration);
            setup.exec(ReadFile(
                "Tests/ExperimentRecommendationCampaignFollowUpProposalMigrationTests.sql"));

            const Proposal proposal = BuildProposal();
            setup.exec(
                "INSERT INTO experiment_recommendation_campaign_approval "
                "VALUES($1,$2,$3);",
                pqxx::params{proposal.campaignIdentity.campaignApprovalId,
                    proposal.campaignIdentity.identityCanonical,
                    proposal.campaignIdentity.identityHash});
            setup.exec(
                "INSERT INTO experiment_recommendation_campaign_materialization "
                "VALUES($1,$2,$3,$4,$5,$6,$7);",
                pqxx::params{
                    proposal.materializationIdentity.materializationId,
                    proposal.materializationIdentity.campaignApprovalId,
                    proposal.materializationIdentity.contractVersion,
                    proposal.materializationIdentity.campaignIdentityHash,
                    proposal.materializationIdentity.memberCount,
                    proposal.materializationIdentity.identityCanonical,
                    proposal.materializationIdentity.identityHash});
            for (const auto& member : proposal.members)
                setup.exec(
                    "INSERT INTO experiment_recommendation_campaign_"
                    "materialization_member VALUES($1,$2,$3,$4,$5,$6,$7);",
                    pqxx::params{member.identity.materializationMemberId,
                        proposal.materializationIdentity.materializationId,
                        member.identity.memberOrdinal,
                        member.identity.rankingMemberId,
                        member.identity.recommendationId,
                        member.identity.sourceExperimentId,
                        member.identity.proposalId});
            setup.commit();
        }

        pqxx::connection runtime{runtimeConnectionString};
        assert(RecommendationCampaignFollowUpProposalSchemaExists(runtime));
        const Proposal proposal = BuildProposal();

        using PersistOutcome =
            RecommendationCampaignFollowUpProposalPersistOutcome;
        PersistOutcome firstOutcome = PersistOutcome::recorded;
        PersistOutcome secondOutcome = PersistOutcome::recorded;
        long long firstId = -1;
        long long secondId = -1;
        int firstOrdinal = -1;
        int secondOrdinal = -1;
        bool firstProposalMatched = false;
        bool secondProposalMatched = false;
        std::exception_ptr firstError;
        std::exception_ptr secondError;
        std::atomic<bool> startConcurrentPersistence{false};
        const auto persistConcurrently = [&](PersistOutcome& outcome,
                                             long long& id,
                                             int& ordinal,
                                             bool& proposalMatched,
                                             std::exception_ptr& error)
        {
            try
            {
                pqxx::connection connection{runtimeConnectionString};
                pqxx::work transaction{connection};
                while (!startConcurrentPersistence.load(
                    std::memory_order_acquire))
                    std::this_thread::yield();
                const auto result =
                    PersistRecommendationCampaignFollowUpProposal(
                        transaction, proposal);
                outcome = result.outcome;
                id = result.persisted.followUpProposalId;
                ordinal = result.persisted.identityHashCollisionOrdinal;
                proposalMatched = result.persisted.proposal == proposal;
                transaction.commit();
            }
            catch (...)
            {
                error = std::current_exception();
            }
        };
        std::thread firstThread(persistConcurrently, std::ref(firstOutcome),
            std::ref(firstId), std::ref(firstOrdinal),
            std::ref(firstProposalMatched), std::ref(firstError));
        std::thread secondThread(persistConcurrently, std::ref(secondOutcome),
            std::ref(secondId), std::ref(secondOrdinal),
            std::ref(secondProposalMatched), std::ref(secondError));
        startConcurrentPersistence.store(true, std::memory_order_release);
        firstThread.join();
        secondThread.join();
        if (firstError) std::rethrow_exception(firstError);
        if (secondError) std::rethrow_exception(secondError);
        assert((firstOutcome == PersistOutcome::recorded &&
                   secondOutcome == PersistOutcome::existingIdentical) ||
            (secondOutcome == PersistOutcome::recorded &&
                firstOutcome == PersistOutcome::existingIdentical));
        assert(firstId > 0 && firstId == secondId);
        assert(firstOrdinal == 0 && secondOrdinal == 0);
        assert(firstProposalMatched && secondProposalMatched);
        const long long persistedId = firstId;

        const auto reloaded = FindRecommendationCampaignFollowUpProposal(
            runtime, persistedId);
        assert(reloaded);
        assert(reloaded->proposal == proposal);
        assert(reloaded->proposal.identity.contractVersion ==
            proposal.identity.contractVersion);
        assert(reloaded->proposal.identity.canonicalText ==
            proposal.identity.canonicalText);
        assert(reloaded->proposal.identity.hash == proposal.identity.hash);
        assert(reloaded->proposal.assessmentCanonicalText ==
            proposal.assessmentCanonicalText);
        assert(reloaded->proposal.policyCanonicalText ==
            proposal.policyCanonicalText);
        assert(reloaded->proposal.policyDecisionCanonicalText ==
            proposal.policyDecisionCanonicalText);
        assert(reloaded->proposal.members == proposal.members);

        const auto byIdentity =
            FindRecommendationCampaignFollowUpProposalByIdentity(
                runtime, proposal.identity.canonicalText);
        assert(byIdentity && byIdentity->followUpProposalId == persistedId);

        {
            pqxx::work transaction{runtime};
            const auto replay = PersistRecommendationCampaignFollowUpProposal(
                transaction, proposal);
            assert(replay.outcome ==
                RecommendationCampaignFollowUpProposalPersistOutcome::
                    existingIdentical);
            assert(replay.persisted.followUpProposalId == persistedId);
            assert(replay.persisted.proposal == proposal);
            transaction.commit();
        }

        bool wrongStoredHashLookupRejected = false;
        bool wrongStoredHashPersistenceRejected = false;
        {
            pqxx::work corrupted{owner};
            SetSearchPath(corrupted, schema);
            corrupted.exec(
                "UPDATE experiment_recommendation_campaign_follow_up_proposal "
                "SET proposal_identity_hash='fnv1a64:0000000000000000' "
                "WHERE recommendation_campaign_follow_up_proposal_id=$1;",
                pqxx::params{persistedId});
            try
            {
                (void)FindRecommendationCampaignFollowUpProposalByIdentity(
                    corrupted, proposal.identity.canonicalText);
            }
            catch (const std::runtime_error& error)
            {
                wrongStoredHashLookupRejected =
                    std::string(error.what()).starts_with(
                        "invalid_persisted_recommendation_campaign_follow_up_"
                        "proposal:");
            }
            try
            {
                (void)PersistRecommendationCampaignFollowUpProposal(
                    corrupted, proposal);
            }
            catch (const std::runtime_error& error)
            {
                wrongStoredHashPersistenceRejected =
                    std::string(error.what()).starts_with(
                        "invalid_persisted_recommendation_campaign_follow_up_"
                        "proposal:");
            }
        }
        assert(wrongStoredHashLookupRejected);
        assert(wrongStoredHashPersistenceRejected);

        const std::vector<long long> sequenceValuesBeforePreview =
            Phase6BSequenceValues(owner, schema);
        assert(sequenceValuesBeforePreview.size() == 2);
        std::ostringstream preview;
        WriteRecommendationCampaignFollowUpProposalPreview(
            preview, *reloaded);
        const std::string previewText = preview.str();
        assert(previewText.find(
            "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_PREVIEW") !=
            std::string::npos);
        assert(previewText.find("proposal_identity_hash=" +
            proposal.identity.hash) != std::string::npos);
        assert(previewText.find("member_count=2") != std::string::npos);
        assert(previewText.find("read_only=true") != std::string::npos);
        assert(previewText.find("operator_approved=false") !=
            std::string::npos);
        assert(previewText.find("follow_up_authorized=false") !=
            std::string::npos);
        assert(previewText.find("queued=false") != std::string::npos);
        assert(previewText.find("scheduled=false") != std::string::npos);
        assert(previewText.find("operator_approved=true") ==
            std::string::npos);
        assert(previewText.find("execution_authorized=true") ==
            std::string::npos);

        std::ostringstream commandOutput;
        std::ostringstream commandErrors;
        assert(RunPreviewRecommendationCampaignFollowUpProposalCommand(
            runtimeConnectionString, persistedId, commandOutput,
            commandErrors) == 0);
        assert(commandOutput.str() == previewText);
        assert(commandErrors.str().empty());
        commandOutput.str({});
        commandOutput.clear();
        assert(RunPreviewRecommendationCampaignFollowUpProposalCommand(
            runtimeConnectionString, 999999, commandOutput,
            commandErrors) == 1);
        assert(commandErrors.str().find(
            "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_NOT_FOUND") !=
            std::string::npos);
        assert(Phase6BSequenceValues(owner, schema) ==
            sequenceValuesBeforePreview);

        AssertMalformedPersistenceRejected(owner, schema, persistedId,
            [&](pqxx::transaction_base& transaction)
            {
                transaction.exec(
                    "UPDATE experiment_recommendation_campaign_follow_up_"
                    "proposal SET policy_identity_hash="
                    "'fnv1a64:0000000000000000' WHERE "
                    "recommendation_campaign_follow_up_proposal_id=$1;",
                    pqxx::params{persistedId});
            });
        AssertMalformedPersistenceRejected(owner, schema, persistedId,
            [&](pqxx::transaction_base& transaction)
            {
                transaction.exec(
                    "DELETE FROM experiment_recommendation_campaign_follow_"
                    "up_proposal_member WHERE "
                    "recommendation_campaign_follow_up_proposal_id=$1 AND "
                    "member_ordinal=2;", pqxx::params{persistedId});
            });
        AssertMalformedPersistenceRejected(owner, schema, persistedId,
            [&](pqxx::transaction_base& transaction)
            {
                transaction.exec(
                    "UPDATE experiment_recommendation_campaign_follow_up_"
                    "proposal_member SET expected_experiment_id=999 WHERE "
                    "recommendation_campaign_follow_up_proposal_id=$1 AND "
                    "member_ordinal=1;", pqxx::params{persistedId});
            });

        bool wrongMemberOrderRejected = false;
        try
        {
            pqxx::work malformedMember{runtime};
            const auto& member = proposal.members.front().identity;
            malformedMember.exec(
                "INSERT INTO experiment_recommendation_campaign_follow_up_"
                "proposal_member (recommendation_campaign_follow_up_proposal_"
                "id,member_ordinal,materialization_member_id,ranking_member_id,"
                "recommendation_id,source_experiment_id,conversion_proposal_id,"
                "expected_experiment_id) VALUES ($1,3,$2,$3,$4,$5,$6,$7);",
                pqxx::params{persistedId, member.materializationMemberId,
                    member.rankingMemberId, member.recommendationId,
                    member.sourceExperimentId, member.proposalId,
                    member.expectedExperimentId});
            malformedMember.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            wrongMemberOrderRejected = error.sqlstate() == "23514";
        }
        assert(wrongMemberOrderRejected);

        bool duplicateRejected = false;
        bool syntheticCollisionSeparated = false;
        bool syntheticCollisionRejectedOnLoad = false;
        bool incompleteMemberSetRejected = false;
        {
            pqxx::work duplicate{owner};
            SetSearchPath(duplicate, schema);
            const long long duplicateId = duplicate.exec(R"SQL(
INSERT INTO experiment_recommendation_campaign_follow_up_proposal(
    proposal_contract_version,proposal_identity_canonical,
    proposal_identity_hash,proposal_identity_hash_collision_ordinal,
    assessment_contract_version,assessment_identity_canonical,
    assessment_identity_hash,policy_contract_version,
    policy_identity_canonical,policy_identity_hash,
    policy_decision_contract_version,policy_decision_identity_canonical,
    policy_decision_identity_hash,campaign_approval_id,
    campaign_identity_canonical,campaign_identity_hash,materialization_id,
    materialization_campaign_approval_id,
    materialization_campaign_identity_hash,materialization_contract_version,
    materialization_identity_canonical,materialization_identity_hash,
    evidence_sufficiency,campaign_interpretation,follow_up_eligibility,
    follow_up_authorized,proposal_reason,member_count)
SELECT proposal_contract_version,proposal_identity_canonical,
    proposal_identity_hash,proposal_identity_hash_collision_ordinal + 1,
    assessment_contract_version,assessment_identity_canonical,
    assessment_identity_hash,policy_contract_version,
    policy_identity_canonical,policy_identity_hash,
    policy_decision_contract_version,policy_decision_identity_canonical,
    policy_decision_identity_hash,campaign_approval_id,
    campaign_identity_canonical,campaign_identity_hash,materialization_id,
    materialization_campaign_approval_id,
    materialization_campaign_identity_hash,materialization_contract_version,
    materialization_identity_canonical,materialization_identity_hash,
    evidence_sufficiency,campaign_interpretation,follow_up_eligibility,
    follow_up_authorized,proposal_reason,member_count
FROM experiment_recommendation_campaign_follow_up_proposal
WHERE recommendation_campaign_follow_up_proposal_id=$1
RETURNING recommendation_campaign_follow_up_proposal_id;
)SQL", pqxx::params{persistedId}).one_row()[0].as<long long>();
            duplicate.exec(R"SQL(
INSERT INTO experiment_recommendation_campaign_follow_up_proposal_member(
    recommendation_campaign_follow_up_proposal_id,member_ordinal,
    materialization_member_id,ranking_member_id,recommendation_id,
    source_experiment_id,conversion_proposal_id,expected_experiment_id)
SELECT $1,member_ordinal,materialization_member_id,ranking_member_id,
    recommendation_id,source_experiment_id,conversion_proposal_id,
    expected_experiment_id
FROM experiment_recommendation_campaign_follow_up_proposal_member
WHERE recommendation_campaign_follow_up_proposal_id=$2;
)SQL", pqxx::params{duplicateId, persistedId});
            try
            {
                (void)FindRecommendationCampaignFollowUpProposalByIdentity(
                    duplicate, proposal.identity.canonicalText);
            }
            catch (const std::runtime_error& error)
            {
                duplicateRejected = std::string(error.what()) ==
                    "recommendation_campaign_follow_up_proposal_duplicate_identity";
            }
            const std::string syntheticCollisionCanonical =
                "synthetic_distinct_canonical_for_phase6b_collision";
            duplicate.exec(
                "UPDATE experiment_recommendation_campaign_follow_up_proposal "
                "SET proposal_identity_canonical=$1 WHERE "
                "recommendation_campaign_follow_up_proposal_id=$2;",
                pqxx::params{syntheticCollisionCanonical, duplicateId});
            const pqxx::row collisionBucket = duplicate.exec(
                "SELECT count(*) AS row_count,count(DISTINCT "
                "proposal_identity_canonical) AS canonical_count,"
                "min(proposal_identity_hash_collision_ordinal) AS min_ordinal,"
                "max(proposal_identity_hash_collision_ordinal) AS max_ordinal "
                "FROM experiment_recommendation_campaign_follow_up_proposal "
                "WHERE proposal_identity_hash=$1;",
                pqxx::params{proposal.identity.hash}).one_row();
            syntheticCollisionSeparated =
                collisionBucket["row_count"].as<int>() == 2 &&
                collisionBucket["canonical_count"].as<int>() == 2 &&
                collisionBucket["min_ordinal"].as<int>() == 0 &&
                collisionBucket["max_ordinal"].as<int>() == 1;
            const auto authoritativeIdentity =
                FindRecommendationCampaignFollowUpProposalByIdentity(
                    duplicate, proposal.identity.canonicalText);
            assert(authoritativeIdentity &&
                authoritativeIdentity->followUpProposalId == persistedId);
            try
            {
                (void)FindRecommendationCampaignFollowUpProposalByIdentity(
                    duplicate, syntheticCollisionCanonical);
            }
            catch (const std::runtime_error& error)
            {
                syntheticCollisionRejectedOnLoad =
                    std::string(error.what()).starts_with(
                        "invalid_persisted_recommendation_campaign_follow_up_"
                        "proposal:");
            }
            duplicate.exec(
                "DELETE FROM experiment_recommendation_campaign_follow_up_"
                "proposal_member WHERE recommendation_campaign_follow_up_"
                "proposal_id=$1 AND member_ordinal=2;",
                pqxx::params{duplicateId});
            try
            {
                duplicate.commit();
            }
            catch (const pqxx::sql_error& error)
            {
                incompleteMemberSetRejected = error.sqlstate() == "23514";
            }
        }
        assert(duplicateRejected);
        assert(syntheticCollisionSeparated);
        assert(syntheticCollisionRejectedOnLoad);
        assert(incompleteMemberSetRejected);

        try
        {
            pqxx::work forbidden{runtime};
            forbidden.exec(
                "UPDATE experiment_recommendation_campaign_follow_up_proposal "
                "SET member_count=1;");
            assert(false);
        }
        catch (const pqxx::sql_error& error)
        {
            assert(PermissionDenied(error));
        }
        try
        {
            pqxx::work forbidden{runtime};
            forbidden.exec(
                "DELETE FROM experiment_recommendation_campaign_follow_up_"
                "proposal_member;");
            assert(false);
        }
        catch (const pqxx::sql_error& error)
        {
            assert(PermissionDenied(error));
        }
        try
        {
            pqxx::work forbidden{runtime};
            forbidden.exec(
                "TRUNCATE experiment_recommendation_campaign_follow_up_"
                "proposal;");
            assert(false);
        }
        catch (const pqxx::sql_error& error)
        {
            assert(PermissionDenied(error));
        }

        pqxx::read_transaction verify{owner};
        SetSearchPath(verify, schema);
        assert(verify.exec(
            "SELECT count(*) FROM experiment_recommendation_campaign_follow_"
            "up_proposal;").one_row()[0].as<int>() == 1);
        assert(verify.exec(
            "SELECT count(*) FROM experiment_recommendation_campaign_follow_"
            "up_proposal_member;").one_row()[0].as<int>() == 2);
        const pqxx::row sentinel = verify.exec(
            "SELECT last_value,is_called FROM phase6b_preview_read_sentinel;")
            .one_row();
        assert(sentinel["last_value"].as<long long>() == 1);
        assert(!sentinel["is_called"].as<bool>());
        assert(verify.exec(
            "SELECT to_regclass('experiment_recommendation_campaign_follow_"
            "up_approval') IS NULL AND "
            "to_regclass('experiment_recommendation_campaign_follow_up_"
            "activation') IS NULL AND "
            "to_regclass('experiment_recommendation_campaign_follow_up_"
            "execution') IS NULL;").one_row()[0].as<bool>());
    }
    catch (...)
    {
        pqxx::work cleanup{owner};
        cleanup.exec("DROP SCHEMA IF EXISTS " + cleanup.quote_name(schema) +
            " CASCADE;");
        cleanup.commit();
        throw;
    }

    pqxx::work cleanup{owner};
    cleanup.exec("DROP SCHEMA " + cleanup.quote_name(schema) + " CASCADE;");
    cleanup.commit();
    return 0;
}
