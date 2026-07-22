#include "../Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationService.hpp"

#include "../Sources/ExperimentRecommendation.hpp"
#include "../Sources/ExperimentRecommendationCampaignFollowUpProposalRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignFollowUpProposalReviewRepository.hpp"

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

using Ratification = RecommendationCampaignFollowUpProposalRatification;
using RatificationOutcome =
    RecommendationCampaignFollowUpProposalRatificationPersistOutcome;
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
using Review = RecommendationCampaignFollowUpProposalReview;
using ReviewDecision =
    RecommendationCampaignFollowUpProposalReviewDecision;
using ResultEvidence = RecommendationCampaignOutcomeAssessmentResultEvidence;
using ResultIdentity = RecommendationCampaignOutcomeAssessmentResultIdentity;
using SourceEvidence = RecommendationCampaignOutcomeAssessmentSourceEvidence;

static_assert(!std::is_copy_assignable_v<
    PersistedRecommendationCampaignFollowUpProposalRatification>);
static_assert(!std::is_move_assignable_v<
    PersistedRecommendationCampaignFollowUpProposalRatification>);

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
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char character)
        { return static_cast<char>(std::tolower(character)); });
    return value;
}

bool SafeDisposableDatabaseName(const std::string& value)
{
    if (value.empty()) return false;
    const std::string lower = Lowercase(value);
    if (lower == "lstm" ||
        (lower.find("test") == std::string::npos &&
            lower.find("tmp") == std::string::npos &&
            lower.find("disposable") == std::string::npos))
        return false;
    return std::all_of(value.begin(), value.end(),
        [](unsigned char character)
        {
            return std::isalnum(character) || character == '_' ||
                character == '-';
        });
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

void CreateBaseSchema(
    pqxx::connection& owner,
    const std::string& schema)
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
    setup.commit();
}

void ApplyMigration(
    pqxx::connection& owner,
    const std::string& schema,
    const std::string& path)
{
    pqxx::work migration{owner};
    SetSearchPath(migration, schema);
    migration.exec(ReadFile(path));
    migration.commit();
}

void AssertMigration(
    pqxx::connection& owner,
    const std::string& schema)
{
    pqxx::work test{owner};
    SetSearchPath(test, schema);
    test.exec(ReadFile(
        "Tests/ExperimentRecommendationCampaignFollowUpProposalRatificationMigrationTests.sql"));
    test.commit();
}

void DropSchema(
    pqxx::connection& owner,
    const std::string& schema)
{
    pqxx::work cleanup{owner};
    cleanup.exec("DROP SCHEMA IF EXISTS " + cleanup.quote_name(schema) +
        " CASCADE;");
    cleanup.commit();
}

Context ComparisonContext(int seed)
{
    return {"eurusd", 12, 0.01, 24,
        "phase6d-label-" + std::to_string(seed), "2026-06-01",
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
        "phase6d-campaign-" + std::to_string(seed);
    CampaignIdentity campaign(100 + seed, campaignCanonical,
        RecommendationCanonicalHash(campaignCanonical));
    const std::string materializationCanonical =
        "phase6d-materialization-" + std::to_string(seed);
    MaterializationIdentity materialization(200 + seed,
        campaign.campaignApprovalId, campaign.identityHash, 1, 1,
        materializationCanonical,
        RecommendationCanonicalHash(materializationCanonical));
    const Assessment assessment =
        BuildRecommendationCampaignOutcomeAssessment(campaign,
            materialization, "2026-07-22 12:00:00+00",
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

PersistedRecommendationCampaignFollowUpProposalReview PersistReview(
    const std::string& runtimeConnectionString,
    long long proposalId,
    const Proposal& proposal,
    ReviewDecision decision,
    const std::string& reviewer,
    const std::string& basis)
{
    const Review review = BuildRecommendationCampaignFollowUpProposalReview(
        proposalId, proposal, decision, reviewer, basis);
    pqxx::connection runtime{runtimeConnectionString};
    pqxx::work transaction{runtime};
    const auto result =
        PersistRecommendationCampaignFollowUpProposalReview(
            transaction, review);
    assert(result.outcome ==
        RecommendationCampaignFollowUpProposalReviewPersistOutcome::recorded);
    transaction.commit();
    return result.persisted;
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

long long TableCount(
    pqxx::connection& owner,
    const std::string& schema,
    const std::string& table)
{
    pqxx::read_transaction transaction{owner};
    SetSearchPath(transaction, schema);
    return transaction.exec(
        "SELECT count(*) FROM " + transaction.quote_name(table) + ";")
        .one_row()[0].as<long long>();
}

std::pair<long long, bool> RatificationSequenceState(
    pqxx::connection& owner,
    const std::string& schema)
{
    pqxx::read_transaction transaction{owner};
    SetSearchPath(transaction, schema);
    const std::string sequence = transaction.exec(
        "SELECT relation.relname FROM pg_class relation WHERE relation.oid="
        "pg_get_serial_sequence('experiment_recommendation_campaign_follow_"
        "up_ratification_event','recommendation_campaign_follow_up_"
        "ratification_event_id')::regclass;")
                                     .one_row()[0].as<std::string>();
    const auto row = transaction.exec(
        "SELECT last_value,is_called FROM " +
        transaction.quote_name(sequence) + ";")
                         .one_row();
    return {row[0].as<long long>(), row[1].as<bool>()};
}

template <typename Function>
void AssertRuntimeError(Function&& function, const std::string& expected)
{
    bool rejected = false;
    try
    {
        function();
    }
    catch (const std::runtime_error& error)
    {
        rejected = error.what() == expected;
    }
    assert(rejected);
}

template <typename Function>
void AssertInvalidArgument(Function&& function, const std::string& expected)
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

template <typename Corruption>
void AssertMalformedReloadRejected(
    pqxx::connection& owner,
    const std::string& schema,
    long long ratificationEventId,
    Corruption&& corruption,
    const std::string& expectedFragment)
{
    bool rejected = false;
    std::string actualMessage;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        corruption(transaction);
        try
        {
            (void)FindRecommendationCampaignFollowUpProposalRatification(
                transaction, ratificationEventId);
        }
        catch (const std::runtime_error& error)
        {
            actualMessage = error.what();
            rejected = actualMessage.starts_with(
                           "invalid_persisted_recommendation_campaign_"
                           "follow_up_proposal_ratification:") &&
                actualMessage.find(expectedFragment) != std::string::npos;
        }
    }
    if (!rejected)
        std::cerr << "malformed_reload_expected=" << expectedFragment
                  << " actual=" << actualMessage << '\n';
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
        std::cerr << "clearly_disposable_non_LSTM_database_required\n";
        return 2;
    }
    const std::string host = EnvironmentOr("LSTM_TEST_DB_HOST", "127.0.0.1");
    const std::string port = EnvironmentOr("LSTM_TEST_DB_PORT", "5432");
    const std::string ownerUser = EnvironmentOr(
        "LSTM_TEST_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string suffix = std::to_string(getpid());
    const std::string cleanSchema = "phase6d_ratification_clean_" + suffix;
    const std::string upgradeSchema = "phase6d_ratification_upgrade_" + suffix;
    const std::string ownerConnectionString =
        "host=" + host + " port=" + port + " user=" + ownerUser +
        " dbname=" + database;
    const auto runtimeConnectionString = [&](const std::string& schema)
    {
        return "host=" + host + " port=" + port +
            " user=pqxx dbname=" + database +
            " options='-c search_path=" + schema + "'";
    };
    pqxx::connection owner{ownerConnectionString};

    try
    {
        {
            pqxx::read_transaction verify{owner};
            const std::string connectedDatabase =
                verify.exec("SELECT current_database();")
                    .one_row()[0].as<std::string>();
            if (connectedDatabase != database ||
                Lowercase(connectedDatabase) == "lstm")
                throw std::runtime_error("test_database_target_mismatch");
        }

        // Clean install: 042 -> 043 -> 044 on an empty isolated schema.
        CreateBaseSchema(owner, cleanSchema);
        ApplyMigration(owner, cleanSchema,
            "Database/migrations/042_experiment_recommendation_campaign_follow_up_proposal.sql");
        ApplyMigration(owner, cleanSchema,
            "Database/migrations/043_experiment_recommendation_campaign_follow_up_proposal_review.sql");
        ApplyMigration(owner, cleanSchema,
            "Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql");
        ApplyMigration(owner, cleanSchema,
            "Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql");
        AssertMigration(owner, cleanSchema);
        assert(TableCount(owner, cleanSchema,
                   "experiment_recommendation_campaign_follow_up_"
                   "ratification_event") == 0);
        DropSchema(owner, cleanSchema);

        // Upgrade path: create authoritative Phase 6B/6C records before 044.
        CreateBaseSchema(owner, upgradeSchema);
        ApplyMigration(owner, upgradeSchema,
            "Database/migrations/042_experiment_recommendation_campaign_follow_up_proposal.sql");
        ApplyMigration(owner, upgradeSchema,
            "Database/migrations/043_experiment_recommendation_campaign_follow_up_proposal_review.sql");
        const std::string runtime = runtimeConnectionString(upgradeSchema);

        const Proposal proposal1 = BuildProposal(1);
        const Proposal proposal2 = BuildProposal(2);
        const Proposal proposal3 = BuildProposal(3);
        const Proposal proposal4 = BuildProposal(4);
        const long long proposalId1 = InsertAndPersistProposal(
            owner, upgradeSchema, runtime, proposal1);
        const long long proposalId2 = InsertAndPersistProposal(
            owner, upgradeSchema, runtime, proposal2);
        const long long proposalId3 = InsertAndPersistProposal(
            owner, upgradeSchema, runtime, proposal3);
        const long long proposalId4 = InsertAndPersistProposal(
            owner, upgradeSchema, runtime, proposal4);
        const auto review1 = PersistReview(runtime, proposalId1, proposal1,
            ReviewDecision::approved, "reviewer.one",
            "Approved on merits for possible governance advancement.");
        const auto review2 = PersistReview(runtime, proposalId2, proposal2,
            ReviewDecision::rejected, "reviewer.two",
            "Rejected during administrative review.");
        const auto review3 = PersistReview(runtime, proposalId3, proposal3,
            ReviewDecision::approved, "reviewer.three",
            "Approved on merits for independent ratification.");
        const auto review4 = PersistReview(runtime, proposalId4, proposal4,
            ReviewDecision::approved, "reviewer.four",
            "Approved on merits for rollback verification.");

        const std::string proposalSnapshotBefore = TableSnapshot(owner,
            upgradeSchema,
            "experiment_recommendation_campaign_follow_up_proposal",
            "recommendation_campaign_follow_up_proposal_id");
        const std::string memberSnapshotBefore = TableSnapshot(owner,
            upgradeSchema,
            "experiment_recommendation_campaign_follow_up_proposal_member",
            "recommendation_campaign_follow_up_proposal_member_id");
        const std::string reviewSnapshotBefore = TableSnapshot(owner,
            upgradeSchema,
            "experiment_recommendation_campaign_follow_up_proposal_review_event",
            "recommendation_campaign_follow_up_proposal_review_event_id");
        const std::string experimentSnapshotBefore = TableSnapshot(
            owner, upgradeSchema, "experiment", "placeholder");
        const std::string schedulerSnapshotBefore = TableSnapshot(
            owner, upgradeSchema, "experiment_scheduler", "placeholder");

        ApplyMigration(owner, upgradeSchema,
            "Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql");
        ApplyMigration(owner, upgradeSchema,
            "Database/migrations/044_experiment_recommendation_campaign_follow_up_proposal_ratification.sql");
        AssertMigration(owner, upgradeSchema);

        RecommendationCampaignFollowUpProposalRatificationRequest request1{
            review1.reviewEventId, review1.review.identity.hash,
            "ratifier.one", "Independent governance ratification basis."};
        RatificationOutcome firstOutcome = RatificationOutcome::recorded;
        RatificationOutcome secondOutcome = RatificationOutcome::recorded;
        long long firstId = 0;
        long long secondId = 0;
        std::exception_ptr firstError;
        std::exception_ptr secondError;
        std::atomic<bool> startIdentical{false};
        const auto ratifyIdentical = [&](RatificationOutcome& outcome,
                                          long long& ratificationEventId,
                                          std::exception_ptr& error)
        {
            try
            {
                pqxx::connection connection{runtime};
                while (!startIdentical.load(std::memory_order_acquire))
                    std::this_thread::yield();
                const auto result =
                    RatifyRecommendationCampaignFollowUpProposal(
                        connection, request1);
                outcome = result.outcome;
                ratificationEventId = result.persisted.ratificationEventId;
            }
            catch (...)
            {
                error = std::current_exception();
            }
        };
        std::thread identical1(ratifyIdentical, std::ref(firstOutcome),
            std::ref(firstId), std::ref(firstError));
        std::thread identical2(ratifyIdentical, std::ref(secondOutcome),
            std::ref(secondId), std::ref(secondError));
        startIdentical.store(true, std::memory_order_release);
        identical1.join();
        identical2.join();
        if (firstError) std::rethrow_exception(firstError);
        if (secondError) std::rethrow_exception(secondError);
        assert((firstOutcome == RatificationOutcome::recorded &&
                   secondOutcome == RatificationOutcome::existingIdentical) ||
            (secondOutcome == RatificationOutcome::recorded &&
                firstOutcome == RatificationOutcome::existingIdentical));
        assert(firstId == secondId);
        const long long ratificationEventId1 = firstId;

        {
            pqxx::connection connection{runtime};
            const auto replay =
                RatifyRecommendationCampaignFollowUpProposal(
                    connection, request1);
            assert(replay.outcome == RatificationOutcome::existingIdentical);
            assert(replay.persisted.ratificationEventId == ratificationEventId1);
            assert(replay.persisted.ratification.reviewEventId ==
                review1.reviewEventId);
            assert(replay.persisted.ratification.reviewCanonicalText ==
                review1.review.identity.canonicalText);
            assert(replay.persisted.ratification.proposalCanonicalText ==
                proposal1.identity.canonicalText);
            assert(replay.persisted.ratification.reviewerIdentity ==
                review1.review.reviewerIdentity);
            assert(replay.persisted.ratification.ratificationAuthorityRole ==
                kRecommendationCampaignFollowUpProposalRatificationAuthorityRole);
            assert(RecommendationCampaignFollowUpProposalRatificationDecisionText(
                       replay.persisted.ratification.decision) == "ratified");
        }

        AssertRuntimeError([&]
        {
            pqxx::connection connection{runtime};
            auto conflicting = request1;
            conflicting.ratifierIdentity = "ratifier.changed";
            (void)RatifyRecommendationCampaignFollowUpProposal(
                connection, conflicting);
        }, "recommendation_campaign_follow_up_proposal_ratification_conflict");
        AssertRuntimeError([&]
        {
            pqxx::connection connection{runtime};
            auto conflicting = request1;
            conflicting.ratificationBasis = "Changed governance basis.";
            (void)RatifyRecommendationCampaignFollowUpProposal(
                connection, conflicting);
        }, "recommendation_campaign_follow_up_proposal_ratification_conflict");
        AssertRuntimeError([&]
        {
            pqxx::connection connection{runtime};
            auto selfRatification = request1;
            selfRatification.ratifierIdentity =
                review1.review.reviewerIdentity;
            (void)RatifyRecommendationCampaignFollowUpProposal(
                connection, selfRatification);
        }, "recommendation_campaign_follow_up_proposal_ratification_separation_of_duties_violation");
        AssertRuntimeError([&]
        {
            pqxx::connection connection{runtime};
            auto rejectedRequest = request1;
            rejectedRequest.reviewEventId = review2.reviewEventId;
            rejectedRequest.expectedReviewIdentityHash =
                review2.review.identity.hash;
            (void)RatifyRecommendationCampaignFollowUpProposal(
                connection, rejectedRequest);
        }, "recommendation_campaign_follow_up_proposal_ratification_review_not_eligible");
        AssertRuntimeError([&]
        {
            pqxx::connection connection{runtime};
            auto missing = request1;
            missing.reviewEventId = 999999;
            (void)RatifyRecommendationCampaignFollowUpProposal(
                connection, missing);
        }, "recommendation_campaign_follow_up_proposal_ratification_review_not_found");
        AssertRuntimeError([&]
        {
            pqxx::connection connection{runtime};
            auto stale = request1;
            stale.expectedReviewIdentityHash =
                "fnv1a64:0000000000000000";
            (void)RatifyRecommendationCampaignFollowUpProposal(
                connection, stale);
        }, "recommendation_campaign_follow_up_proposal_ratification_review_identity_mismatch");
        AssertInvalidArgument([&]
        {
            auto invalid = request1;
            invalid.expectedReviewIdentityHash = "not-a-hash";
            (void)ValidateRecommendationCampaignFollowUpProposalRatificationRequest(
                invalid);
        }, "recommendation_campaign_follow_up_proposal_ratification_expected_review_hash_invalid");
        AssertInvalidArgument([&]
        {
            auto invalid = request1;
            invalid.ratifierIdentity.clear();
            (void)ValidateRecommendationCampaignFollowUpProposalRatificationRequest(
                invalid);
        }, "ratifier_identity_invalid");
        AssertInvalidArgument([&]
        {
            auto invalid = request1;
            invalid.ratificationBasis = std::string("bad\x7f" "basis", 9);
            (void)ValidateRecommendationCampaignFollowUpProposalRatificationRequest(
                invalid);
        }, "ratification_basis_invalid");

        RecommendationCampaignFollowUpProposalRatificationRequest request3a{
            review3.reviewEventId, review3.review.identity.hash,
            "ratifier.concurrent.one", "Concurrent basis one."};
        auto request3b = request3a;
        request3b.ratifierIdentity = "ratifier.concurrent.two";
        request3b.ratificationBasis = "Concurrent basis two.";
        bool conflictRecorded1 = false;
        bool conflictRecorded2 = false;
        bool conflictRejected1 = false;
        bool conflictRejected2 = false;
        std::exception_ptr conflictError1;
        std::exception_ptr conflictError2;
        std::atomic<bool> startConflict{false};
        const auto ratifyConflict = [&](const auto& request,
                                         bool& recorded,
                                         bool& rejected,
                                         std::exception_ptr& error)
        {
            try
            {
                pqxx::connection connection{runtime};
                while (!startConflict.load(std::memory_order_acquire))
                    std::this_thread::yield();
                const auto result =
                    RatifyRecommendationCampaignFollowUpProposal(
                        connection, request);
                recorded = result.outcome == RatificationOutcome::recorded;
            }
            catch (const std::runtime_error& exception)
            {
                rejected = std::string(exception.what()) ==
                    "recommendation_campaign_follow_up_proposal_ratification_conflict";
                if (!rejected) error = std::current_exception();
            }
            catch (...)
            {
                error = std::current_exception();
            }
        };
        std::thread conflict1(ratifyConflict, std::cref(request3a),
            std::ref(conflictRecorded1), std::ref(conflictRejected1),
            std::ref(conflictError1));
        std::thread conflict2(ratifyConflict, std::cref(request3b),
            std::ref(conflictRecorded2), std::ref(conflictRejected2),
            std::ref(conflictError2));
        startConflict.store(true, std::memory_order_release);
        conflict1.join();
        conflict2.join();
        if (conflictError1) std::rethrow_exception(conflictError1);
        if (conflictError2) std::rethrow_exception(conflictError2);
        assert(conflictRecorded1 != conflictRecorded2);
        assert(conflictRejected1 != conflictRejected2);
        assert(conflictRecorded1 == conflictRejected2);
        assert(conflictRecorded2 == conflictRejected1);

        const Ratification rollbackRatification =
            BuildRecommendationCampaignFollowUpProposalRatification(
                review4.reviewEventId, review4.review, "ratifier.rollback",
                "This transaction\nmust\troll back.");
        const auto assertRatificationConstraint = [&]
            (const std::string& basis,
             const std::string& reviewerIdentity,
             const std::string& authorityRole,
             const std::string& ratifierIdentity)
        {
            bool rejected = false;
            try
            {
                pqxx::work transaction{owner};
                SetSearchPath(transaction, upgradeSchema);
                const auto& value = rollbackRatification;
                transaction.exec(
                    "INSERT INTO experiment_recommendation_campaign_follow_up_"
                    "ratification_event (ratification_contract_version,"
                    "recommendation_campaign_follow_up_proposal_review_event_id,"
                    "review_contract_version,review_identity_canonical,"
                    "review_identity_hash,review_decision,reviewer_identity,"
                    "recommendation_campaign_"
                    "follow_up_proposal_id,proposal_contract_version,"
                    "proposal_identity_canonical,proposal_identity_hash,"
                    "ratification_authority_role,ratification_decision,"
                    "ratifier_identity,ratification_basis,"
                    "ratification_identity_canonical,ratification_identity_hash) VALUES "
                    "($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17);",
                    pqxx::params{value.identity.contractVersion,
                        value.reviewEventId, value.reviewContractVersion,
                        value.reviewCanonicalText, value.reviewIdentityHash,
                        "approved", reviewerIdentity,
                        value.followUpProposalId,
                        value.proposalContractVersion,
                        value.proposalCanonicalText, value.proposalIdentityHash,
                        authorityRole, "ratified", ratifierIdentity, basis,
                        value.identity.canonicalText, value.identity.hash});
            }
            catch (const pqxx::sql_error& error)
            {
                rejected = error.sqlstate() == "23514";
            }
            assert(rejected);
        };
        assertRatificationConstraint(std::string("bad\x01" "basis", 9),
            rollbackRatification.reviewerIdentity,
            rollbackRatification.ratificationAuthorityRole,
            rollbackRatification.ratifierIdentity);
        assertRatificationConstraint(" \t\r\n",
            rollbackRatification.reviewerIdentity,
            rollbackRatification.ratificationAuthorityRole,
            rollbackRatification.ratifierIdentity);
        assertRatificationConstraint("Independent governance basis.",
            rollbackRatification.reviewerIdentity,
            rollbackRatification.ratificationAuthorityRole,
            rollbackRatification.reviewerIdentity);
        bool runtimeSelfRatificationRejected = false;
        try
        {
            pqxx::connection connection{runtime};
            pqxx::work transaction{connection};
            const auto& value = rollbackRatification;
            transaction.exec(
                "INSERT INTO experiment_recommendation_campaign_follow_up_"
                "ratification_event (ratification_contract_version,"
                "recommendation_campaign_follow_up_proposal_review_event_id,"
                "review_contract_version,review_identity_canonical,"
                "review_identity_hash,review_decision,reviewer_identity,"
                "recommendation_campaign_"
                "follow_up_proposal_id,proposal_contract_version,"
                "proposal_identity_canonical,proposal_identity_hash,"
                "ratification_authority_role,ratification_decision,"
                "ratifier_identity,ratification_basis,"
                "ratification_identity_canonical,ratification_identity_hash) VALUES "
                "($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17);",
                pqxx::params{value.identity.contractVersion,
                    value.reviewEventId, value.reviewContractVersion,
                    value.reviewCanonicalText, value.reviewIdentityHash,
                    "approved", value.reviewerIdentity,
                    value.followUpProposalId,
                    value.proposalContractVersion,
                    value.proposalCanonicalText, value.proposalIdentityHash,
                    value.ratificationAuthorityRole, "ratified",
                    value.reviewerIdentity,
                    "Independent governance basis.",
                    value.identity.canonicalText, value.identity.hash});
        }
        catch (const pqxx::sql_error& error)
        {
            runtimeSelfRatificationRejected = error.sqlstate() == "23514";
        }
        assert(runtimeSelfRatificationRejected);
        assertRatificationConstraint("Independent governance basis.",
            rollbackRatification.reviewerIdentity, "caller_injected_role",
            rollbackRatification.ratifierIdentity);
        assertRatificationConstraint("Independent governance basis.",
            "forged.reviewer",
            rollbackRatification.ratificationAuthorityRole,
            rollbackRatification.ratifierIdentity);
        const long long countBeforeRollback = TableCount(owner, upgradeSchema,
            "experiment_recommendation_campaign_follow_up_ratification_event");
        {
            pqxx::connection connection{runtime};
            pqxx::work transaction{connection};
            const auto result =
                PersistRecommendationCampaignFollowUpProposalRatification(
                    transaction, rollbackRatification);
            assert(result.outcome == RatificationOutcome::recorded);
        }
        assert(TableCount(owner, upgradeSchema,
                   "experiment_recommendation_campaign_follow_up_"
                   "ratification_event") == countBeforeRollback);
        {
            pqxx::connection connection{runtime};
            const auto result =
                RatifyRecommendationCampaignFollowUpProposal(connection,
                    {review4.reviewEventId, review4.review.identity.hash,
                        "ratifier.rollback",
                        "This transaction\nmust\troll back."});
            assert(result.outcome == RatificationOutcome::recorded);
        }

        const auto ratification1 = [&]
        {
            pqxx::read_transaction transaction{owner};
            SetSearchPath(transaction, upgradeSchema);
            return FindRecommendationCampaignFollowUpProposalRatification(
                transaction, ratificationEventId1);
        }();
        assert(ratification1);
        assert(ratification1->ratification.reviewEventId == review1.reviewEventId);
        assert(ratification1->ratification.reviewIdentityHash ==
            review1.review.identity.hash);
        assert(ratification1->ratification.followUpProposalId == proposalId1);
        assert(ratification1->ratification.proposalIdentityHash ==
            proposal1.identity.hash);

        AssertMalformedReloadRejected(owner, upgradeSchema, ratificationEventId1,
            [&](pqxx::transaction_base& transaction)
            {
                transaction.exec(
                    "UPDATE experiment_recommendation_campaign_follow_up_"
                    "ratification_event SET ratification_identity_hash="
                    "'fnv1a64:0000000000000000' WHERE "
                    "recommendation_campaign_follow_up_ratification_event_id=$1;",
                    pqxx::params{ratificationEventId1});
            }, "ratification_identity_mismatch");
        AssertMalformedReloadRejected(owner, upgradeSchema, ratificationEventId1,
            [&](pqxx::transaction_base& transaction)
            {
                transaction.exec(
                    "UPDATE experiment_recommendation_campaign_follow_up_"
                    "ratification_event SET review_identity_hash="
                    "'fnv1a64:0000000000000000' WHERE "
                    "recommendation_campaign_follow_up_ratification_event_id=$1;",
                    pqxx::params{ratificationEventId1});
            }, "review_identity_invalid");
        AssertMalformedReloadRejected(owner, upgradeSchema, ratificationEventId1,
            [&](pqxx::transaction_base& transaction)
            {
                transaction.exec(
                    "UPDATE experiment_recommendation_campaign_follow_up_"
                    "proposal_review_event SET review_identity_hash="
                    "'fnv1a64:0000000000000000' WHERE "
                    "recommendation_campaign_follow_up_proposal_review_event_id=$1;",
                    pqxx::params{review1.reviewEventId});
            }, "invalid_persisted_recommendation_campaign_follow_up_proposal_"
               "review:review_identity_mismatch");
        AssertMalformedReloadRejected(owner, upgradeSchema, ratificationEventId1,
            [&](pqxx::transaction_base& transaction)
            {
                const std::string forgedProposal =
                    proposal1.identity.canonicalText + ";forged=true";
                const auto forgedReview =
                    BuildRecommendationCampaignFollowUpProposalReview(
                        review1.review.identity.contractVersion, proposalId1,
                        review1.review.proposalContractVersion,
                        forgedProposal,
                        RecommendationCanonicalHash(forgedProposal),
                        ReviewDecision::approved,
                        review1.review.reviewerIdentity,
                        review1.review.reasonText);
                const auto forgedRatification =
                    BuildRecommendationCampaignFollowUpProposalRatification(1,
                        review1.reviewEventId, 1,
                        forgedReview.identity.canonicalText,
                        forgedReview.identity.hash, proposalId1, 1,
                        forgedProposal,
                        RecommendationCanonicalHash(forgedProposal),
                        ReviewDecision::approved,
                        review1.review.reviewerIdentity,
                        kRecommendationCampaignFollowUpProposalRatificationAuthorityRole,
                        RecommendationCampaignFollowUpProposalRatificationDecision::ratified,
                        "ratifier.one",
                        "Independent governance ratification basis.");
                transaction.exec(
                    "UPDATE experiment_recommendation_campaign_follow_up_"
                    "ratification_event SET review_identity_canonical=$1,"
                    "review_identity_hash=$2,proposal_identity_canonical=$3,"
                    "proposal_identity_hash=$4,ratification_identity_canonical=$5,"
                    "ratification_identity_hash=$6 WHERE recommendation_campaign_"
                    "follow_up_ratification_event_id=$7;",
                    pqxx::params{forgedReview.identity.canonicalText,
                        forgedReview.identity.hash, forgedProposal,
                        RecommendationCanonicalHash(forgedProposal),
                        forgedRatification.identity.canonicalText,
                        forgedRatification.identity.hash,
                        ratificationEventId1});
            }, "review_mismatch");

        // A colliding stored accelerator cannot merge or authenticate a
        // different canonical ratification.
        const auto byReview3 = [&]
        {
            pqxx::read_transaction transaction{owner};
            SetSearchPath(transaction, upgradeSchema);
            return FindRecommendationCampaignFollowUpProposalRatificationByReviewEventId(
                transaction, review3.reviewEventId);
        }();
        assert(byReview3);
        AssertMalformedReloadRejected(owner, upgradeSchema,
            byReview3->ratificationEventId,
            [&](pqxx::transaction_base& transaction)
            {
                transaction.exec(
                    "UPDATE experiment_recommendation_campaign_follow_up_"
                    "ratification_event SET ratification_identity_hash=$1 "
                    "WHERE recommendation_campaign_follow_up_"
                    "ratification_event_id=$2;",
                    pqxx::params{ratification1->ratification.identity.hash,
                        byReview3->ratificationEventId});
            }, "ratification_identity_mismatch");

        bool provenanceRejected = false;
        try
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, upgradeSchema);
            const auto& value = ratification1->ratification;
            transaction.exec(
                "INSERT INTO experiment_recommendation_campaign_follow_up_"
                "ratification_event (ratification_contract_version,"
                "recommendation_campaign_follow_up_proposal_review_event_id,"
                "review_contract_version,review_identity_canonical,"
                "review_identity_hash,review_decision,reviewer_identity,"
                "recommendation_campaign_"
                "follow_up_proposal_id,proposal_contract_version,"
                "proposal_identity_canonical,proposal_identity_hash,"
                "ratification_authority_role,ratification_decision,"
                "ratifier_identity,ratification_basis,"
                "ratification_identity_canonical,ratification_identity_hash) VALUES "
                "($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17);",
                pqxx::params{value.identity.contractVersion,
                    review2.reviewEventId, value.reviewContractVersion,
                    value.reviewCanonicalText, value.reviewIdentityHash,
                    "approved", value.reviewerIdentity, proposalId2,
                    value.proposalContractVersion,
                    value.proposalCanonicalText, value.proposalIdentityHash,
                    value.ratificationAuthorityRole, "ratified",
                    "ratifier.provenance", "Mismatch.",
                    value.identity.canonicalText, value.identity.hash});
        }
        catch (const pqxx::sql_error& error)
        {
            provenanceRejected = error.sqlstate() == "23514";
        }
        assert(provenanceRejected);

        bool updateDenied = false;
        try
        {
            pqxx::connection connection{runtime};
            pqxx::work transaction{connection};
            transaction.exec(
                "UPDATE experiment_recommendation_campaign_follow_up_"
                "ratification_event SET ratification_basis='changed';");
        }
        catch (const pqxx::sql_error& error)
        {
            updateDenied = PermissionDenied(error);
        }
        assert(updateDenied);
        bool deleteDenied = false;
        try
        {
            pqxx::connection connection{runtime};
            pqxx::work transaction{connection};
            transaction.exec(
                "DELETE FROM experiment_recommendation_campaign_follow_up_"
                "ratification_event;");
        }
        catch (const pqxx::sql_error& error)
        {
            deleteDenied = PermissionDenied(error);
        }
        assert(deleteDenied);
        bool idInsertDenied = false;
        try
        {
            pqxx::connection connection{runtime};
            pqxx::work transaction{connection};
            transaction.exec(
                "INSERT INTO experiment_recommendation_campaign_follow_up_"
                "ratification_event (recommendation_campaign_follow_up_"
                "ratification_event_id) VALUES(9999);");
        }
        catch (const pqxx::sql_error& error)
        {
            idInsertDenied = PermissionDenied(error);
        }
        assert(idInsertDenied);
        bool timestampInsertDenied = false;
        try
        {
            pqxx::connection connection{runtime};
            pqxx::work transaction{connection};
            transaction.exec(
                "INSERT INTO experiment_recommendation_campaign_follow_up_"
                "ratification_event (created_at) VALUES(now());");
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
            SetSearchPath(transaction, upgradeSchema);
            transaction.exec(
                "DELETE FROM experiment_recommendation_campaign_follow_up_"
                "proposal_review_event WHERE recommendation_campaign_follow_"
                "up_proposal_review_event_id=$1;",
                pqxx::params{review1.reviewEventId});
        }
        catch (const pqxx::foreign_key_violation&)
        {
            restrictiveDelete = true;
        }
        assert(restrictiveDelete);

        const auto sequenceBeforeReads =
            RatificationSequenceState(owner, upgradeSchema);
        {
            pqxx::connection connection{runtime};
            pqxx::read_transaction transaction{connection};
            transaction.exec(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
            const auto found =
                FindRecommendationCampaignFollowUpProposalRatificationByReviewEventId(
                    transaction, review1.reviewEventId);
            assert(found && found->ratificationEventId == ratificationEventId1);
            const auto byProposal =
                FindRecommendationCampaignFollowUpProposalRatificationByProposalId(
                    transaction, proposalId1);
            assert(byProposal && byProposal->ratificationEventId == ratificationEventId1);
            assert(!FindRecommendationCampaignFollowUpProposalRatification(
                transaction, 999999));
            assert(!FindRecommendationCampaignFollowUpProposalRatificationByReviewEventId(
                transaction, 999999));
            assert(!FindRecommendationCampaignFollowUpProposalRatificationByProposalId(
                transaction, 999999));
            const auto listed =
                ListRecommendationCampaignFollowUpProposalRatifications(
                    transaction, 2);
            assert(listed.size() == 2);
            assert(listed[0].ratificationEventId > listed[1].ratificationEventId);
            assert(transaction.exec(
                       "SELECT count(*) FROM pg_locks WHERE "
                       "pid=pg_backend_pid() AND locktype IN "
                       "('advisory','tuple');")
                       .one_row()[0].as<int>() == 0);
        }
        assert(RatificationSequenceState(owner, upgradeSchema) ==
            sequenceBeforeReads);
        AssertInvalidArgument([&]
        {
            pqxx::read_transaction transaction{owner};
            SetSearchPath(transaction, upgradeSchema);
            (void)ListRecommendationCampaignFollowUpProposalRatifications(
                transaction, 0);
        }, "recommendation_campaign_follow_up_proposal_ratification_list_limit_invalid");
        AssertInvalidArgument([&]
        {
            pqxx::read_transaction transaction{owner};
            SetSearchPath(transaction, upgradeSchema);
            (void)ListRecommendationCampaignFollowUpProposalRatifications(
                transaction,
                kMaximumRecommendationCampaignFollowUpProposalRatificationListLimit +
                    1);
        }, "recommendation_campaign_follow_up_proposal_ratification_list_limit_invalid");

        assert(TableSnapshot(owner, upgradeSchema,
                   "experiment_recommendation_campaign_follow_up_proposal",
                   "recommendation_campaign_follow_up_proposal_id") ==
            proposalSnapshotBefore);
        assert(TableSnapshot(owner, upgradeSchema,
                   "experiment_recommendation_campaign_follow_up_proposal_member",
                   "recommendation_campaign_follow_up_proposal_member_id") ==
            memberSnapshotBefore);
        assert(TableSnapshot(owner, upgradeSchema,
                   "experiment_recommendation_campaign_follow_up_proposal_review_event",
                   "recommendation_campaign_follow_up_proposal_review_event_id") ==
            reviewSnapshotBefore);
        assert(TableSnapshot(owner, upgradeSchema, "experiment", "placeholder") ==
            experimentSnapshotBefore);
        assert(TableSnapshot(
                   owner, upgradeSchema, "experiment_scheduler", "placeholder") ==
            schedulerSnapshotBefore);

        DropSchema(owner, upgradeSchema);
    }
    catch (...)
    {
        try
        {
            DropSchema(owner, cleanSchema);
        }
        catch (...)
        {
        }
        try
        {
            DropSchema(owner, upgradeSchema);
        }
        catch (...)
        {
        }
        throw;
    }

    return 0;
}
