#include "../Sources/ExperimentRecommendationCampaignFollowUpProposalReviewPresentation.hpp"

#include "../Sources/ExperimentRecommendation.hpp"
#include "../Sources/ExperimentRecommendationCampaignFollowUpProposalRepository.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

#include <pqxx/pqxx>

using namespace EA::ExperimentRecommendation;

namespace
{

using Assessment = RecommendationCampaignOutcomeAssessment;
using CampaignIdentity =
    RecommendationCampaignOutcomeAssessmentCampaignIdentity;
using Context = RecommendationCampaignOutcomeAssessmentComparisonContext;
using Decision = RecommendationCampaignFollowUpProposalReviewDecision;
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
using PersistOutcome =
    RecommendationCampaignFollowUpProposalReviewPersistOutcome;
using PolicyInput = RecommendationCampaignOutcomePolicyInput;
using Proposal = RecommendationCampaignFollowUpProposal;
using Review = RecommendationCampaignFollowUpProposalReview;
using ResultEvidence = RecommendationCampaignOutcomeAssessmentResultEvidence;
using ResultIdentity = RecommendationCampaignOutcomeAssessmentResultIdentity;
using SourceEvidence = RecommendationCampaignOutcomeAssessmentSourceEvidence;

static_assert(!std::is_copy_assignable_v<
    PersistedRecommendationCampaignFollowUpProposalReview>);
static_assert(!std::is_move_assignable_v<
    PersistedRecommendationCampaignFollowUpProposalReview>);

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
        std::string(error.what()).find("permission denied") !=
            std::string::npos;
}

void SetSearchPath(
    pqxx::transaction_base& transaction,
    const std::string& schema)
{
    transaction.exec(
        "SET LOCAL search_path TO " + transaction.quote_name(schema) + ";");
}

Context ComparisonContext(int seed)
{
    return {"eurusd", 12, 0.01, 24,
        "phase6c-label-" + std::to_string(seed), "2026-06-01",
        "2026-06-30"};
}

Metric Numeric(std::string identity, double value)
{
    return {std::move(identity), value, MetricSupport::NumericDelta};
}

MemberEvidence FavorableMember(int seed)
{
    MemberIdentity identity(1, 1000 + seed, 2000 + seed, 3000 + seed,
        4000 + seed, 5000 + seed, 6000 + seed);
    return {identity, Lifecycle::Succeeded,
        SourceEvidence(identity.sourceExperimentId, 7000 + seed,
            8000 + seed, ComparisonContext(seed),
            MetricCollection({Numeric("inference_accuracy", 0.70),
                Numeric("leader_score", 0.80)})),
        ResultEvidence(identity.expectedExperimentId, 9000 + seed,
            {ResultIdentity{"inference", 10000 + seed}},
            ComparisonContext(seed),
            MetricCollection({Numeric("inference_accuracy", 0.75),
                Numeric("leader_score", 0.85)}))};
}

Proposal BuildProposal(int seed)
{
    const std::string campaignCanonical =
        "phase6c-campaign-" + std::to_string(seed);
    CampaignIdentity campaign(100 + seed, campaignCanonical,
        RecommendationCanonicalHash(campaignCanonical));
    const std::string materializationCanonical =
        "phase6c-materialization-" + std::to_string(seed);
    MaterializationIdentity materialization(200 + seed,
        campaign.campaignApprovalId, campaign.identityHash, 1, 1,
        materializationCanonical,
        RecommendationCanonicalHash(materializationCanonical));
    const Assessment assessment =
        BuildRecommendationCampaignOutcomeAssessment(campaign,
            materialization, "2026-07-21 12:00:00+00",
            {FavorableMember(seed)});
    const auto policy = BuildRecommendationCampaignOutcomePolicy(PolicyInput{});
    const auto policyDecision = ApplyRecommendationCampaignOutcomePolicy(
        policy, assessment);
    return BuildRecommendationCampaignFollowUpProposal(
        assessment, policyDecision);
}

long long InsertAndPersistProposal(
    pqxx::connection& owner,
    const std::string& schema,
    const std::string& runtimeConnectionString,
    const Proposal& proposal)
{
    {
        pqxx::work setup{owner};
        SetSearchPath(setup, schema);
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
        const auto& member = proposal.members.front();
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
    pqxx::work transaction{runtime};
    const auto result = PersistRecommendationCampaignFollowUpProposal(
        transaction, proposal);
    assert(result.outcome ==
        RecommendationCampaignFollowUpProposalPersistOutcome::recorded);
    transaction.commit();
    return result.persisted.followUpProposalId;
}

Review MakeReview(
    long long proposalId,
    const Proposal& proposal,
    Decision decision,
    std::string reviewer,
    std::string reason)
{
    return BuildRecommendationCampaignFollowUpProposalReview(proposalId,
        proposal, decision, std::move(reviewer), std::move(reason));
}

std::string TableSnapshot(
    pqxx::connection& owner,
    const std::string& schema,
    const std::string& table,
    const std::string& orderColumn)
{
    pqxx::read_transaction transaction{owner};
    SetSearchPath(transaction, schema);
    return transaction.exec(
        "SELECT coalesce(string_agg(row_to_json(snapshot)::text,'|' ORDER BY " +
        orderColumn + "),'') FROM (SELECT * FROM " +
        transaction.quote_name(table) + ") snapshot;")
        .one_row()[0].as<std::string>();
}

std::pair<long long, bool> ReviewSequenceState(
    pqxx::connection& owner,
    const std::string& schema)
{
    pqxx::read_transaction transaction{owner};
    SetSearchPath(transaction, schema);
    const std::string sequence = transaction.exec(
        "SELECT relation.relname FROM pg_class relation WHERE relation.oid="
        "pg_get_serial_sequence('experiment_recommendation_campaign_follow_"
        "up_proposal_review_event','recommendation_campaign_follow_up_"
        "proposal_review_event_id')::regclass;")
                                     .one_row()[0].as<std::string>();
    const auto row = transaction.exec(
        "SELECT last_value,is_called FROM " +
        transaction.quote_name(sequence) + ";")
                         .one_row();
    return {row[0].as<long long>(), row[1].as<bool>()};
}

template <typename Corruption>
void AssertMalformedReloadRejected(
    pqxx::connection& owner,
    const std::string& schema,
    long long reviewEventId,
    Corruption&& corruption,
    const std::string& expectedFragment)
{
    bool rejected = false;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        corruption(transaction);
        try
        {
            (void)FindRecommendationCampaignFollowUpProposalReview(
                transaction, reviewEventId);
        }
        catch (const std::runtime_error& error)
        {
            const std::string message = error.what();
            rejected = message.starts_with(
                           "invalid_persisted_recommendation_campaign_"
                           "follow_up_proposal_review:") &&
                message.find(expectedFragment) != std::string::npos;
        }
    }
    assert(rejected);
}

template <typename Function>
void AssertInvalidArgument(
    Function&& function,
    const std::string& expected)
{
    bool rejected = false;
    try
    {
        function();
    }
    catch (const std::invalid_argument& error)
    {
        rejected = error.what() == expected;
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
        "phase6c_follow_up_review_" + std::to_string(getpid());
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
CREATE TABLE experiment(placeholder text PRIMARY KEY);
CREATE TABLE experiment_scheduler(placeholder text PRIMARY KEY);
INSERT INTO experiment VALUES('unchanged');
INSERT INTO experiment_scheduler VALUES('unchanged');
)SQL");
            setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                " TO pqxx; GRANT SELECT ON ALL TABLES IN SCHEMA " +
                setup.quote_name(schema) + " TO pqxx;");
            setup.exec(ReadFile(
                "Database/migrations/042_experiment_recommendation_campaign_follow_up_proposal.sql"));
            const std::string migration = ReadFile(
                "Database/migrations/043_experiment_recommendation_campaign_follow_up_proposal_review.sql");
            setup.exec(migration);
            setup.exec(migration);
            setup.exec(ReadFile(
                "Tests/ExperimentRecommendationCampaignFollowUpProposalReviewMigrationTests.sql"));
            setup.commit();
        }
        {
            pqxx::nontransaction session{owner};
            session.exec(
                "SET search_path TO " + session.quote_name(schema) + ";");
        }

        const Proposal proposal1 = BuildProposal(1);
        const Proposal proposal2 = BuildProposal(2);
        const Proposal proposal3 = BuildProposal(3);
        const Proposal proposal4 = BuildProposal(4);
        const long long proposalId1 = InsertAndPersistProposal(
            owner, schema, runtimeConnectionString, proposal1);
        const long long proposalId2 = InsertAndPersistProposal(
            owner, schema, runtimeConnectionString, proposal2);
        const long long proposalId3 = InsertAndPersistProposal(
            owner, schema, runtimeConnectionString, proposal3);
        const long long proposalId4 = InsertAndPersistProposal(
            owner, schema, runtimeConnectionString, proposal4);

        const std::string proposalSnapshotBefore = TableSnapshot(owner, schema,
            "experiment_recommendation_campaign_follow_up_proposal",
            "recommendation_campaign_follow_up_proposal_id");
        const std::string memberSnapshotBefore = TableSnapshot(owner, schema,
            "experiment_recommendation_campaign_follow_up_proposal_member",
            "recommendation_campaign_follow_up_proposal_member_id");
        const std::string experimentSnapshotBefore = TableSnapshot(
            owner, schema, "experiment", "placeholder");
        const std::string schedulerSnapshotBefore = TableSnapshot(
            owner, schema, "experiment_scheduler", "placeholder");

        assert(RecommendationCampaignFollowUpProposalReviewSchemaExists(owner));
        const Review approved = MakeReview(proposalId1, proposal1,
            Decision::approved, "operator.one", "Approved administratively.");

        PersistOutcome identicalFirst = PersistOutcome::recorded;
        PersistOutcome identicalSecond = PersistOutcome::recorded;
        long long identicalFirstId = 0;
        long long identicalSecondId = 0;
        std::exception_ptr identicalFirstError;
        std::exception_ptr identicalSecondError;
        std::atomic<bool> startIdentical{false};
        const auto persistIdentical = [&](PersistOutcome& outcome,
                                          long long& eventId,
                                          std::exception_ptr& error)
        {
            try
            {
                pqxx::connection connection{runtimeConnectionString};
                pqxx::work transaction{connection};
                while (!startIdentical.load(std::memory_order_acquire))
                    std::this_thread::yield();
                const auto result =
                    PersistRecommendationCampaignFollowUpProposalReview(
                        transaction, approved);
                outcome = result.outcome;
                eventId = result.persisted.reviewEventId;
                assert(result.persisted.review == approved);
                transaction.commit();
            }
            catch (...)
            {
                error = std::current_exception();
            }
        };
        std::thread identicalThread1(persistIdentical,
            std::ref(identicalFirst), std::ref(identicalFirstId),
            std::ref(identicalFirstError));
        std::thread identicalThread2(persistIdentical,
            std::ref(identicalSecond), std::ref(identicalSecondId),
            std::ref(identicalSecondError));
        startIdentical.store(true, std::memory_order_release);
        identicalThread1.join();
        identicalThread2.join();
        if (identicalFirstError) std::rethrow_exception(identicalFirstError);
        if (identicalSecondError) std::rethrow_exception(identicalSecondError);
        assert((identicalFirst == PersistOutcome::recorded &&
                   identicalSecond == PersistOutcome::existingIdentical) ||
            (identicalSecond == PersistOutcome::recorded &&
                identicalFirst == PersistOutcome::existingIdentical));
        assert(identicalFirstId == identicalSecondId);
        const long long approvedEventId = identicalFirstId;

        const auto approvedReload =
            FindRecommendationCampaignFollowUpProposalReview(
                owner, approvedEventId);
        assert(approvedReload);
        assert(approvedReload->review == approved);
        assert(approvedReload->review.identity.canonicalText ==
            approved.identity.canonicalText);
        assert(approvedReload->review.identity.hash == approved.identity.hash);
        assert(approvedReload->review.proposalCanonicalText ==
            proposal1.identity.canonicalText);
        assert(approvedReload->review.proposalIdentityHash ==
            proposal1.identity.hash);

        {
            pqxx::connection runtime{runtimeConnectionString};
            pqxx::work transaction{runtime};
            const auto replay =
                PersistRecommendationCampaignFollowUpProposalReview(
                    transaction, approved);
            assert(replay.outcome == PersistOutcome::existingIdentical);
            assert(replay.persisted.reviewEventId == approvedEventId);
            transaction.commit();
        }

        const Review rejected = MakeReview(proposalId2, proposal2,
            Decision::rejected, "operator.two", "Rejected administratively.");
        long long rejectedEventId = 0;
        {
            pqxx::connection runtime{runtimeConnectionString};
            pqxx::work transaction{runtime};
            const auto result =
                PersistRecommendationCampaignFollowUpProposalReview(
                    transaction, rejected);
            assert(result.outcome == PersistOutcome::recorded);
            assert(result.persisted.review.decision == Decision::rejected);
            rejectedEventId = result.persisted.reviewEventId;
            transaction.commit();
        }
        const auto rejectedReload =
            FindRecommendationCampaignFollowUpProposalReview(
                owner, rejectedEventId);
        assert(rejectedReload &&
            rejectedReload->review.decision == Decision::rejected);

        const auto assertConflict = [&](const Review& conflicting)
        {
            bool rejectedAsConflict = false;
            try
            {
                pqxx::connection runtime{runtimeConnectionString};
                pqxx::work transaction{runtime};
                (void)PersistRecommendationCampaignFollowUpProposalReview(
                    transaction, conflicting);
            }
            catch (const std::runtime_error& error)
            {
                rejectedAsConflict = std::string(error.what()) ==
                    "recommendation_campaign_follow_up_proposal_review_conflict";
            }
            assert(rejectedAsConflict);
        };
        assertConflict(MakeReview(proposalId1, proposal1,
            Decision::rejected, "operator.one", "Approved administratively."));
        assertConflict(MakeReview(proposalId1, proposal1,
            Decision::approved, "operator.other", "Approved administratively."));
        assertConflict(MakeReview(proposalId1, proposal1,
            Decision::approved, "operator.one", "Different reason."));

        const Review concurrentApproved = MakeReview(proposalId3, proposal3,
            Decision::approved, "operator.concurrent",
            "Concurrent approval.");
        const Review concurrentRejected = MakeReview(proposalId3, proposal3,
            Decision::rejected, "operator.concurrent",
            "Concurrent rejection.");
        bool concurrent1Recorded = false;
        bool concurrent2Recorded = false;
        bool concurrent1Conflict = false;
        bool concurrent2Conflict = false;
        std::exception_ptr concurrent1Error;
        std::exception_ptr concurrent2Error;
        std::atomic<bool> startConflict{false};
        const auto persistConflict = [&](const Review& review,
                                         bool& recorded,
                                         bool& conflict,
                                         std::exception_ptr& error)
        {
            try
            {
                pqxx::connection connection{runtimeConnectionString};
                pqxx::work transaction{connection};
                while (!startConflict.load(std::memory_order_acquire))
                    std::this_thread::yield();
                const auto result =
                    PersistRecommendationCampaignFollowUpProposalReview(
                        transaction, review);
                recorded = result.outcome == PersistOutcome::recorded;
                transaction.commit();
            }
            catch (const std::runtime_error& exception)
            {
                conflict = std::string(exception.what()) ==
                    "recommendation_campaign_follow_up_proposal_review_conflict";
                if (!conflict) error = std::current_exception();
            }
            catch (...)
            {
                error = std::current_exception();
            }
        };
        std::thread conflictThread1(persistConflict,
            std::cref(concurrentApproved), std::ref(concurrent1Recorded),
            std::ref(concurrent1Conflict), std::ref(concurrent1Error));
        std::thread conflictThread2(persistConflict,
            std::cref(concurrentRejected), std::ref(concurrent2Recorded),
            std::ref(concurrent2Conflict), std::ref(concurrent2Error));
        startConflict.store(true, std::memory_order_release);
        conflictThread1.join();
        conflictThread2.join();
        if (concurrent1Error) std::rethrow_exception(concurrent1Error);
        if (concurrent2Error) std::rethrow_exception(concurrent2Error);
        assert(concurrent1Recorded != concurrent2Recorded);
        assert(concurrent1Conflict != concurrent2Conflict);
        assert(concurrent1Recorded == concurrent2Conflict);
        assert(concurrent2Recorded == concurrent1Conflict);

        const Review fourth = MakeReview(proposalId4, proposal4,
            Decision::approved, "operator.four", "Fourth review.");
        long long fourthEventId = 0;
        {
            pqxx::connection runtime{runtimeConnectionString};
            pqxx::work transaction{runtime};
            const auto result =
                PersistRecommendationCampaignFollowUpProposalReview(
                    transaction, fourth);
            fourthEventId = result.persisted.reviewEventId;
            transaction.commit();
        }

        const long long reviewCountBeforeFailedPersistence = [&]
        {
            pqxx::read_transaction transaction{owner};
            SetSearchPath(transaction, schema);
            return transaction.exec(
                "SELECT count(*) FROM experiment_recommendation_campaign_"
                "follow_up_proposal_review_event;")
                .one_row()[0].as<long long>();
        }();
        assertConflict(MakeReview(proposalId2, proposal2,
            Decision::approved, "operator.two", "Opposite decision."));
        {
            pqxx::read_transaction transaction{owner};
            SetSearchPath(transaction, schema);
            assert(transaction.exec(
                       "SELECT count(*) FROM experiment_recommendation_"
                       "campaign_follow_up_proposal_review_event;")
                       .one_row()[0].as<long long>() ==
                reviewCountBeforeFailedPersistence);
        }

        AssertMalformedReloadRejected(owner, schema, approvedEventId,
            [&](pqxx::transaction_base& transaction)
            {
                transaction.exec(
                    "UPDATE experiment_recommendation_campaign_follow_up_"
                    "proposal_review_event SET review_identity_hash="
                    "'fnv1a64:0000000000000000' WHERE "
                    "recommendation_campaign_follow_up_proposal_review_event_id=$1;",
                    pqxx::params{approvedEventId});
            },
            "review_identity_mismatch");
        AssertMalformedReloadRejected(owner, schema, approvedEventId,
            [&](pqxx::transaction_base& transaction)
            {
                transaction.exec(
                    "UPDATE experiment_recommendation_campaign_follow_up_"
                    "proposal_review_event SET review_identity_canonical="
                    "review_identity_canonical||';activated=true' WHERE "
                    "recommendation_campaign_follow_up_proposal_review_event_id=$1;",
                    pqxx::params{approvedEventId});
            },
            "review_identity_mismatch");
        AssertMalformedReloadRejected(owner, schema, approvedEventId,
            [&](pqxx::transaction_base& transaction)
            {
                transaction.exec(R"SQL(
DO $drop_check$
DECLARE check_name name;
BEGIN
    SELECT conname INTO check_name
    FROM pg_constraint
    WHERE conrelid = to_regclass(
        'experiment_recommendation_campaign_follow_up_proposal_review_event')
      AND contype = 'c'
      AND pg_get_constraintdef(oid) LIKE '%review_decision%';
    EXECUTE format(
        'ALTER TABLE experiment_recommendation_campaign_follow_up_proposal_review_event DROP CONSTRAINT %I',
        check_name);
END
$drop_check$;
)SQL");
                transaction.exec(
                    "UPDATE experiment_recommendation_campaign_follow_up_"
                    "proposal_review_event SET review_decision='pending' "
                    "WHERE recommendation_campaign_follow_up_proposal_review_"
                    "event_id=$1;",
                    pqxx::params{approvedEventId});
            },
            "review_decision_invalid");
        AssertMalformedReloadRejected(owner, schema, approvedEventId,
            [&](pqxx::transaction_base& transaction)
            {
                transaction.exec(
                    "UPDATE experiment_recommendation_campaign_follow_up_"
                    "proposal_review_event SET proposal_identity_hash="
                    "'fnv1a64:0000000000000000' WHERE "
                    "recommendation_campaign_follow_up_proposal_review_event_id=$1;",
                    pqxx::params{approvedEventId});
            },
            "proposal_identity_invalid");

        const std::string forgedProposalCanonical =
            proposal1.identity.canonicalText + ";forged=true";
        const std::string forgedProposalHash =
            RecommendationCanonicalHash(forgedProposalCanonical);
        const Review forged =
            BuildRecommendationCampaignFollowUpProposalReview(1,
                proposalId1, 1, forgedProposalCanonical, forgedProposalHash,
                approved.decision, approved.reviewerIdentity,
                approved.reasonText);
        AssertMalformedReloadRejected(owner, schema, approvedEventId,
            [&](pqxx::transaction_base& transaction)
            {
                transaction.exec(
                    "UPDATE experiment_recommendation_campaign_follow_up_"
                    "proposal_review_event SET proposal_identity_canonical=$1,"
                    "proposal_identity_hash=$2,review_identity_canonical=$3,"
                    "review_identity_hash=$4 WHERE "
                    "recommendation_campaign_follow_up_proposal_review_event_id=$5;",
                    pqxx::params{forged.proposalCanonicalText,
                        forged.proposalIdentityHash,
                        forged.identity.canonicalText, forged.identity.hash,
                        approvedEventId});
            },
            "proposal_mismatch");

        bool provenanceRejected = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec(
                "INSERT INTO experiment_recommendation_campaign_follow_up_"
                "proposal_review_event (review_contract_version,"
                "recommendation_campaign_follow_up_proposal_id,"
                "proposal_contract_version,proposal_identity_canonical,"
                "proposal_identity_hash,review_decision,reviewer_identity,"
                "reason_text,review_identity_canonical,review_identity_hash) "
                "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10);",
                pqxx::params{fourth.identity.contractVersion, proposalId4,
                    fourth.proposalContractVersion,
                    forged.proposalCanonicalText, forged.proposalIdentityHash,
                    "approved", "operator.provenance", "Mismatch.",
                    forged.identity.canonicalText, forged.identity.hash});
        }
        catch (const pqxx::sql_error& error)
        {
            provenanceRejected = error.sqlstate() == "23514";
        }
        assert(provenanceRejected);

        bool idInsertDenied = false;
        try
        {
            pqxx::connection runtime{runtimeConnectionString};
            pqxx::work transaction{runtime};
            transaction.exec(
                "INSERT INTO experiment_recommendation_campaign_follow_up_"
                "proposal_review_event (recommendation_campaign_follow_up_"
                "proposal_review_event_id) VALUES(9999);");
        }
        catch (const pqxx::sql_error& error)
        {
            idInsertDenied = PermissionDenied(error);
        }
        assert(idInsertDenied);
        bool timestampInsertDenied = false;
        try
        {
            pqxx::connection runtime{runtimeConnectionString};
            pqxx::work transaction{runtime};
            transaction.exec(
                "INSERT INTO experiment_recommendation_campaign_follow_up_"
                "proposal_review_event (created_at) VALUES(now());");
        }
        catch (const pqxx::sql_error& error)
        {
            timestampInsertDenied = PermissionDenied(error);
        }
        assert(timestampInsertDenied);

        bool restrictiveDelete = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec(
                "DELETE FROM experiment_recommendation_campaign_follow_up_"
                "proposal WHERE recommendation_campaign_follow_up_proposal_id=$1;",
                pqxx::params{proposalId1});
        }
        catch (const pqxx::foreign_key_violation&)
        {
            restrictiveDelete = true;
        }
        assert(restrictiveDelete);

        const auto byProposal =
            FindRecommendationCampaignFollowUpProposalReviewByProposalId(
                owner, proposalId1);
        assert(byProposal && byProposal->reviewEventId == approvedEventId);
        assert(!FindRecommendationCampaignFollowUpProposalReview(
            owner, 999999));
        assert(!FindRecommendationCampaignFollowUpProposalReviewByProposalId(
            owner, 999999));

        const auto listed =
            ListRecommendationCampaignFollowUpProposalReviews(owner, 2);
        assert(listed.size() == 2);
        assert(listed[0].reviewEventId > listed[1].reviewEventId);
        assert(listed[0].reviewEventId == fourthEventId);
        AssertInvalidArgument([&]
        {
            (void)ListRecommendationCampaignFollowUpProposalReviews(owner, 0);
        }, "recommendation_campaign_follow_up_proposal_review_list_limit_invalid");
        AssertInvalidArgument([&]
        {
            (void)ListRecommendationCampaignFollowUpProposalReviews(
                owner,
                kMaximumRecommendationCampaignFollowUpProposalReviewListLimit +
                    1);
        }, "recommendation_campaign_follow_up_proposal_review_list_limit_invalid");

        const auto sequenceBeforePresentation =
            ReviewSequenceState(owner, schema);
        std::ostringstream showOutput;
        std::ostringstream showErrors;
        assert(RunShowRecommendationCampaignFollowUpProposalReview(
                   runtimeConnectionString, approvedEventId, showOutput,
                   showErrors) == 0);
        assert(showErrors.str().empty());
        const std::string shown = showOutput.str();
        for (const std::string& required : std::vector<std::string>{
                 "review_event_id=" + std::to_string(approvedEventId),
                 "follow_up_proposal_id=" + std::to_string(proposalId1),
                 "proposal_identity_hash=" + proposal1.identity.hash,
                 "review_contract_version=1",
                 "review_identity_hash=" + approved.identity.hash,
                 "decision=approved", "reviewer=operator.one",
                 "reason=Approved%20administratively.", "created_at=",
                 "read_only=true", "persisted=true",
                 "administrative_review=true", "activated=false",
                 "execution_authorized=false", "follow_up_authorized=false",
                 "queued=false", "scheduled=false",
                 "scheduler_started=false", "scheduler_signaled=false",
                 "workers_started=false", "experiments_created=false",
                 "experiments_modified=false",
                 "campaign_success_declared=false"})
            assert(shown.find(required) != std::string::npos);
        assert(shown.find("activated=true") == std::string::npos);
        assert(shown.find("execution_authorized=true") == std::string::npos);

        showOutput.str({});
        showOutput.clear();
        assert(RunShowRecommendationCampaignFollowUpProposalReview(
                   runtimeConnectionString, 999999, showOutput,
                   showErrors) == 1);
        assert(showErrors.str().find(
                   "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_REVIEW_NOT_FOUND") !=
            std::string::npos);
        std::ostringstream listOutput;
        std::ostringstream listErrors;
        assert(RunListRecommendationCampaignFollowUpProposalReviews(
                   runtimeConnectionString, 2, listOutput, listErrors) == 0);
        assert(listErrors.str().empty());
        const std::string listText = listOutput.str();
        const std::string newest =
            "review_event_id=" + std::to_string(listed[0].reviewEventId);
        const std::string next =
            "review_event_id=" + std::to_string(listed[1].reviewEventId);
        assert(listText.find(newest) < listText.find(next));
        assert(std::count(listText.begin(), listText.end(), '\n') == 2);
        std::ostringstream invalidListOutput;
        std::ostringstream invalidListErrors;
        assert(RunListRecommendationCampaignFollowUpProposalReviews(
                   runtimeConnectionString, 0, invalidListOutput,
                   invalidListErrors) == 1);
        assert(invalidListErrors.str().find(
                   "review_list_limit_invalid") != std::string::npos);
        assert(ReviewSequenceState(owner, schema) ==
            sequenceBeforePresentation);

        {
            pqxx::connection runtime{runtimeConnectionString};
            pqxx::read_transaction transaction{runtime};
            transaction.exec(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
            (void)ListRecommendationCampaignFollowUpProposalReviews(
                transaction, 100);
            assert(transaction.exec(
                       "SELECT count(*) FROM pg_locks WHERE "
                       "pid=pg_backend_pid() AND locktype IN "
                       "('advisory','tuple');")
                       .one_row()[0].as<int>() == 0);
        }

        assert(TableSnapshot(owner, schema,
                   "experiment_recommendation_campaign_follow_up_proposal",
                   "recommendation_campaign_follow_up_proposal_id") ==
            proposalSnapshotBefore);
        assert(TableSnapshot(owner, schema,
                   "experiment_recommendation_campaign_follow_up_proposal_member",
                   "recommendation_campaign_follow_up_proposal_member_id") ==
            memberSnapshotBefore);
        assert(TableSnapshot(owner, schema, "experiment", "placeholder") ==
            experimentSnapshotBefore);
        assert(TableSnapshot(
                   owner, schema, "experiment_scheduler", "placeholder") ==
            schedulerSnapshotBefore);

        {
            pqxx::work cleanup{owner};
            cleanup.exec("DROP SCHEMA " + cleanup.quote_name(schema) +
                " CASCADE;");
            cleanup.commit();
        }
    }
    catch (...)
    {
        try
        {
            pqxx::work cleanup{owner};
            cleanup.exec("DROP SCHEMA IF EXISTS " +
                cleanup.quote_name(schema) + " CASCADE;");
            cleanup.commit();
        }
        catch (...)
        {
        }
        throw;
    }

    return 0;
}
