#include "../Sources/ExperimentRecommendationConversionRepository.hpp"
#include <exception>

#include <cassert>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

#include <pqxx/pqxx>

using namespace EA::ExperimentRecommendation;

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

ExperimentInvocationConfiguration SourceInvocation()
{
    ExperimentInvocationConfiguration invocation;
    invocation.configuration.symbol = "eurusd";
    invocation.configuration.predictionHorizon = 12;
    invocation.configuration.labelThreshold = 0.001;
    invocation.configuration.coreLrMult = 1.0;
    invocation.configuration.headLrMult = 5.0;
    invocation.configuration.targetEpochs = 120;
    invocation.configuration.trainStartDate = "2010-01-01";
    invocation.configuration.trainEndDate = "2025-01-01";
    invocation.configuration.inferStartDate = "2025-01-01";
    invocation.configuration.inferEndDate = "2026-01-01";
    invocation.checkpointInterval = 15;
    invocation.resumeModelId = 77;
    return invocation;
}

void SetRecommendationIdentity(RecommendationConversionRequest& request)
{
    ExperimentInvocationConfiguration proposed = request.sourceInvocation;
    const RecommendationConversionMutation& mutation = request.mutations.front();
    if (mutation.family == kCoreLrMult)
        proposed.configuration.coreLrMult =
            std::stod(mutation.proposedValueCanonical);
    else if (mutation.family == kHeadLrMult)
        proposed.configuration.headLrMult =
            std::stod(mutation.proposedValueCanonical);
    else if (mutation.family == kLabelThreshold)
        proposed.configuration.labelThreshold =
            std::stod(mutation.proposedValueCanonical);
    else if (mutation.family == kPredictionHorizon)
        proposed.configuration.predictionHorizon =
            std::stoi(mutation.proposedValueCanonical);
    const auto semantic = BuildRecommendationCandidateIdentity(
        proposed.configuration);
    const auto invocation = BuildRecommendationInvocationIdentity(proposed);
    request.recommendationSemanticCanonical = semantic.canonicalText;
    request.recommendationSemanticHash = semantic.hash;
    request.recommendationInvocationCanonical = invocation.canonicalText;
    request.recommendationInvocationHash = invocation.hash;
}

RecommendationConversionRequest ValidRequest(const std::string& proposed = "1.25")
{
    RecommendationConversionRequest request;
    request.recommendationExists = true;
    request.recommendationId = 42;
    request.recommendationStatus = RecommendationStatus::approved;
    request.sourceExperimentId = 17;
    request.recommendationSourceExperimentId = 17;
    request.sourceInvocation = SourceInvocation();
    request.mutations.push_back({kCoreLrMult, "1", proposed});

    request.reviewAuthorization.present = true;
    request.reviewAuthorization.recommendationId = 42;
    request.reviewAuthorization.latestAction = RecommendationReviewAction::approve;
    request.reviewAuthorization.resultingStatus = RecommendationStatus::approved;
    request.reviewAuthorization.latestActionEffective = true;
    request.reviewAuthorization.authorizationCanonical =
        "review_authorization_v1;operator=8:reviewer";
    request.reviewAuthorization.authorizationHash = RecommendationCanonicalHash(
        request.reviewAuthorization.authorizationCanonical);

    request.evaluation.state = RecommendationConversionEvidenceState::completed;
    request.evaluation.valid = true;
    request.evaluation.recommendationId = 42;
    request.evaluation.sourceExperimentId = 17;
    request.evaluation.eligibility = RecommendationEligibility::eligible;
    request.evaluation.disposition =
        RecommendationEvaluationDisposition::advisoryReady;
    request.evaluation.evaluationIdentityCanonical =
        "evaluation_identity_v1;result=42";
    request.evaluation.evaluationIdentityHash = RecommendationCanonicalHash(
        request.evaluation.evaluationIdentityCanonical);
    request.evaluation.evaluationPolicyCanonical =
        "evaluation_policy_v1;policy=stable";
    request.evaluation.evaluationPolicyHash = RecommendationCanonicalHash(
        request.evaluation.evaluationPolicyCanonical);

    request.score.state = RecommendationConversionEvidenceState::completed;
    request.score.valid = true;
    request.score.recommendationId = 42;
    request.score.finalScore = 0.75;
    request.score.scoringPolicyCanonical = "scoring_policy_v1;policy=stable";
    request.score.scoringPolicyHash = RecommendationCanonicalHash(
        request.score.scoringPolicyCanonical);
    request.evaluation.scoringPolicyHash = request.score.scoringPolicyHash;
    SetRecommendationIdentity(request);
    return request;
}

ProposedExperimentSpecification Proposal(
    const std::string& proposed = "1.25",
    const std::optional<std::string>& rankingHash = std::nullopt)
{
    RecommendationConversionRequest request = ValidRequest(proposed);
    if (rankingHash)
    {
        RecommendationConversionRankingProvenance ranking;
        ranking.snapshotIdentityCanonical = "ranking_snapshot_v1;scope=global";
        ranking.snapshotIdentityHash = *rankingHash;
        ranking.memberIdentityCanonical = "ranking_member_v1;recommendation=42";
        ranking.memberIdentityHash = RecommendationCanonicalHash(
            ranking.memberIdentityCanonical);
        ranking.bucket = RecommendationRankingBucket::advisoryReady;
        ranking.bucketRank = 1;
        // The pure contract requires canonical/hash agreement.
        ranking.snapshotIdentityCanonical = "ranking_snapshot_v1;hash_input=" +
            *rankingHash;
        ranking.snapshotIdentityHash = RecommendationCanonicalHash(
            ranking.snapshotIdentityCanonical);
        request.ranking = ranking;
    }
    const RecommendationConversionResult result =
        BuildProposedExperimentSpecification(request);
    assert(result.eligibility.eligible && result.proposal);
    return *result.proposal;
}

ProposedExperimentSpecification NullableProposal()
{
    RecommendationConversionRequest request = ValidRequest();
    request.sourceInvocation.configuration.coreLrMult.reset();
    request.sourceInvocation.configuration.headLrMult.reset();
    request.sourceInvocation.configuration.inferStartDate.reset();
    request.sourceInvocation.configuration.inferEndDate.reset();
    request.sourceInvocation.resumeModelId.reset();
    request.mutations.clear();
    request.mutations.push_back({kLabelThreshold, "0.001", "0.0011"});
    SetRecommendationIdentity(request);
    const RecommendationConversionResult result =
        BuildProposedExperimentSpecification(request);
    assert(result.eligibility.eligible && result.proposal);
    return *result.proposal;
}

bool SameInvocation(const ExperimentInvocationConfiguration& left,
                    const ExperimentInvocationConfiguration& right)
{
    return BuildRecommendationInvocationIdentity(left).canonicalText ==
           BuildRecommendationInvocationIdentity(right).canonicalText;
}

void AssertRoundTrip(const ProposedExperimentSpecification& expected,
                     const PersistedRecommendationConversionProposal& actual)
{
    assert(actual.proposalId > 0);
    assert(actual.conversionContractVersion ==
           kRecommendationConversionContractVersion);
    assert(actual.hashCollisionOrdinal >= 0);
    assert(!actual.createdAt.empty());
    const ProposedExperimentSpecification& value = actual.proposal;
    assert(value.sourceExperimentId == expected.sourceExperimentId);
    assert(value.recommendationId == expected.recommendationId);
    assert(SameInvocation(value.proposedInvocation, expected.proposedInvocation));
    assert(value.sourceInvocationCanonical == expected.sourceInvocationCanonical);
    assert(value.proposedInvocationCanonical ==
           expected.proposedInvocationCanonical);
    assert(value.changedParameter == expected.changedParameter);
    assert(value.sourceValueCanonical == expected.sourceValueCanonical);
    assert(value.proposedValueCanonical == expected.proposedValueCanonical);
    assert(value.recommendationSemanticHash ==
           expected.recommendationSemanticHash);
    assert(value.evaluationIdentityHash == expected.evaluationIdentityHash);
    assert(value.evaluationPolicyHash == expected.evaluationPolicyHash);
    assert(value.scoringPolicyHash == expected.scoringPolicyHash);
    assert(value.reviewAuthorizationHash == expected.reviewAuthorizationHash);
    assert(value.rankingSnapshotIdentityHash ==
           expected.rankingSnapshotIdentityHash);
    assert(value.conversionIdentityCanonical ==
           expected.conversionIdentityCanonical);
    assert(value.conversionIdentityHash == expected.conversionIdentityHash);
}

void InsertSyntheticHashCollision(
    pqxx::connection& owner,
    const std::string& schema,
    long long sourceProposalId,
    const std::string& canonical,
    const std::string& collidingHash,
    int collisionOrdinal)
{
    pqxx::work transaction{owner};
    transaction.exec("SET LOCAL search_path TO " +
                     transaction.quote_name(schema) + ";");
    transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_proposal ("
        "recommendation_id,source_experiment_id,conversion_contract_version,"
        "changed_parameter,source_value_canonical,proposed_value_canonical,"
        "recommendation_semantic_hash,evaluation_identity_hash,"
        "evaluation_policy_hash,scoring_policy_hash,review_authorization_hash,"
        "ranking_snapshot_identity_hash,proposed_symbol,"
        "proposed_prediction_horizon,proposed_label_threshold,"
        "proposed_core_lr_mult,proposed_head_lr_mult,proposed_target_epochs,"
        "proposed_train_start_date,proposed_train_end_date,"
        "proposed_infer_start_date,proposed_infer_end_date,"
        "proposed_checkpoint_interval,proposed_resume_model_id,"
        "source_invocation_canonical,proposed_invocation_canonical,"
        "conversion_identity_canonical,conversion_identity_hash,"
        "conversion_hash_collision_ordinal) SELECT recommendation_id,"
        "source_experiment_id,conversion_contract_version,changed_parameter,"
        "source_value_canonical,proposed_value_canonical,"
        "recommendation_semantic_hash,evaluation_identity_hash,"
        "evaluation_policy_hash,scoring_policy_hash,review_authorization_hash,"
        "ranking_snapshot_identity_hash,proposed_symbol,"
        "proposed_prediction_horizon,proposed_label_threshold,"
        "proposed_core_lr_mult,proposed_head_lr_mult,proposed_target_epochs,"
        "proposed_train_start_date,proposed_train_end_date,"
        "proposed_infer_start_date,proposed_infer_end_date,"
        "proposed_checkpoint_interval,proposed_resume_model_id,"
        "source_invocation_canonical,proposed_invocation_canonical,"
        "$2,$3,$4 FROM "
        "experiment_recommendation_conversion_proposal WHERE "
        "recommendation_conversion_proposal_id=$1;",
        pqxx::params{sourceProposalId, canonical, collidingHash,
                     collisionOrdinal});
    transaction.commit();
}

void InsertSyntheticHashCollisionWithRepositoryProtocol(
    const std::string& ownerConnectionString,
    const std::string& schema,
    long long sourceProposalId,
    const std::string& canonical,
    const std::string& collidingHash,
    std::atomic<bool>& start)
{
    while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
    pqxx::connection owner{ownerConnectionString};
    pqxx::work transaction{owner};
    transaction.exec("SET LOCAL search_path TO " +
                     transaction.quote_name(schema) + ";");
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, "
        "7046029254386353131));",
        pqxx::params{collidingHash});
    const int ordinal = transaction.exec(
        "SELECT coalesce(max(conversion_hash_collision_ordinal),-1)+1 "
        "FROM experiment_recommendation_conversion_proposal WHERE "
        "conversion_identity_hash=$1;",
        pqxx::params{collidingHash}).one_row()[0].as<int>();
    transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_proposal ("
        "recommendation_id,source_experiment_id,conversion_contract_version,"
        "changed_parameter,source_value_canonical,proposed_value_canonical,"
        "recommendation_semantic_hash,evaluation_identity_hash,"
        "evaluation_policy_hash,scoring_policy_hash,review_authorization_hash,"
        "ranking_snapshot_identity_hash,proposed_symbol,"
        "proposed_prediction_horizon,proposed_label_threshold,"
        "proposed_core_lr_mult,proposed_head_lr_mult,proposed_target_epochs,"
        "proposed_train_start_date,proposed_train_end_date,"
        "proposed_infer_start_date,proposed_infer_end_date,"
        "proposed_checkpoint_interval,proposed_resume_model_id,"
        "source_invocation_canonical,proposed_invocation_canonical,"
        "conversion_identity_canonical,conversion_identity_hash,"
        "conversion_hash_collision_ordinal) SELECT recommendation_id,"
        "source_experiment_id,conversion_contract_version,changed_parameter,"
        "source_value_canonical,proposed_value_canonical,"
        "recommendation_semantic_hash,evaluation_identity_hash,"
        "evaluation_policy_hash,scoring_policy_hash,review_authorization_hash,"
        "ranking_snapshot_identity_hash,proposed_symbol,"
        "proposed_prediction_horizon,proposed_label_threshold,"
        "proposed_core_lr_mult,proposed_head_lr_mult,proposed_target_epochs,"
        "proposed_train_start_date,proposed_train_end_date,"
        "proposed_infer_start_date,proposed_infer_end_date,"
        "proposed_checkpoint_interval,proposed_resume_model_id,"
        "source_invocation_canonical,proposed_invocation_canonical,$2,$3,$4 "
        "FROM experiment_recommendation_conversion_proposal WHERE "
        "recommendation_conversion_proposal_id=$1;",
        pqxx::params{sourceProposalId, canonical, collidingHash, ordinal});
    transaction.commit();
}

ProposedExperimentSpecification WithCanonicalPadding(
    ProposedExperimentSpecification proposal,
    std::size_t paddingBytes)
{
    const std::string sourceMarker = ";source_invocation=";
    const std::size_t insertion =
        proposal.conversionIdentityCanonical.rfind(sourceMarker);
    assert(insertion != std::string::npos);
    const std::string padding =
        ";persistence_test_padding=" + std::to_string(paddingBytes) + ":" +
        std::string(paddingBytes, 'x');
    proposal.conversionIdentityCanonical.insert(insertion, padding);
    proposal.conversionIdentityHash = RecommendationCanonicalHash(
        proposal.conversionIdentityCanonical);
    return proposal;
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
    if (database == "LSTM")
    {
        std::cerr << "active_LSTM_database_forbidden\n";
        return 2;
    }
    const std::string host = EnvironmentOr("LSTM_TEST_DB_HOST", "127.0.0.1");
    const std::string port = EnvironmentOr("LSTM_TEST_DB_PORT", "5432");
    const std::string ownerUser = EnvironmentOr(
        "LSTM_TEST_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string schema = "phase4c_conversion_" + std::to_string(getpid());
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
            setup.exec("SET LOCAL search_path TO " + setup.quote_name(schema) + ";");
            setup.exec("CREATE TABLE experiment(experiment_id bigint PRIMARY KEY,"
                       "status text NOT NULL,marker text NOT NULL);");
            setup.exec("CREATE TABLE model(model_id bigint PRIMARY KEY,"
                       "marker text NOT NULL);");
            setup.exec("CREATE TABLE experiment_recommendation("
                       "recommendation_id bigint PRIMARY KEY,"
                       "source_experiment_id bigint NOT NULL REFERENCES "
                       "experiment(experiment_id),status text NOT NULL);");
            setup.exec("INSERT INTO experiment VALUES (17,'paused','unchanged');"
                       "INSERT INTO model VALUES (77,'unchanged');"
                       "INSERT INTO experiment_recommendation VALUES "
                       "(42,17,'approved');");
            const std::string migration = ReadFile(
                "Database/migrations/036_experiment_recommendation_conversion_proposal.sql");
            setup.exec(migration);
            setup.exec(migration);
            setup.exec(ReadFile(
                "Tests/ExperimentRecommendationConversionMigrationTests.sql"));
            setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                       " TO pqxx;");
            setup.commit();
        }

        pqxx::connection runtime{runtimeConnectionString};
        assert(RecommendationConversionProposalSchemaExists(runtime));
        const ProposedExperimentSpecification proposal =
            Proposal("1.25", "first_ranking");
        const auto created = PersistRecommendationConversionProposal(
            runtime, proposal);
        assert(created.outcome ==
               RecommendationConversionProposalPersistOutcome::created);
        AssertRoundTrip(proposal, created.persisted);

        const auto byId = FindRecommendationConversionProposal(
            runtime, created.persisted.proposalId);
        assert(byId);
        AssertRoundTrip(proposal, *byId);
        const auto byIdentity = FindRecommendationConversionProposalByIdentity(
            runtime, proposal.conversionIdentityCanonical,
            proposal.conversionIdentityHash);
        assert(byIdentity && byIdentity->proposalId == created.persisted.proposalId);
        assert(ListRecommendationConversionProposalsByRecommendation(
            runtime, 42).size() == 1);
        assert(ListRecommendationConversionProposalsBySourceExperiment(
            runtime, 17).size() == 1);

        const ProposedExperimentSpecification nullableProposal =
            NullableProposal();
        const auto nullableCreated = PersistRecommendationConversionProposal(
            runtime, nullableProposal);
        assert(nullableCreated.outcome ==
               RecommendationConversionProposalPersistOutcome::created);
        AssertRoundTrip(nullableProposal, nullableCreated.persisted);
        assert(!nullableCreated.persisted.proposal.proposedInvocation
                    .configuration.coreLrMult);
        assert(!nullableCreated.persisted.proposal.proposedInvocation
                    .configuration.headLrMult);
        assert(!nullableCreated.persisted.proposal.proposedInvocation
                    .configuration.inferStartDate);
        assert(!nullableCreated.persisted.proposal.proposedInvocation
                    .configuration.inferEndDate);
        assert(!nullableCreated.persisted.proposal.proposedInvocation.resumeModelId);

        const auto duplicate = PersistRecommendationConversionProposal(
            runtime, proposal);
        assert(duplicate.outcome ==
               RecommendationConversionProposalPersistOutcome::existingIdentical);
        assert(duplicate.persisted.proposalId == created.persisted.proposalId);

        ProposedExperimentSpecification changedRanking = proposal;
        changedRanking.rankingSnapshotIdentityHash = "different_advisory_ranking";
        const auto rankingRetry = PersistRecommendationConversionProposal(
            runtime, changedRanking);
        assert(rankingRetry.outcome ==
               RecommendationConversionProposalPersistOutcome::existingIdentical);
        assert(rankingRetry.persisted.proposalId == created.persisted.proposalId);
        assert(rankingRetry.persisted.proposal.rankingSnapshotIdentityHash ==
               proposal.rankingSnapshotIdentityHash);

        const ProposedExperimentSpecification concurrentProposal = Proposal("1.5");
        RecommendationConversionProposalPersistOutcome firstOutcome =
            RecommendationConversionProposalPersistOutcome::createdWithHashCollision;
        RecommendationConversionProposalPersistOutcome secondOutcome = firstOutcome;
        long long firstId = -1;
        long long secondId = -1;
        std::exception_ptr firstError;
        std::exception_ptr secondError;
        std::thread first([&] {
            try
            {
                pqxx::connection connection{runtimeConnectionString};
                const auto result = PersistRecommendationConversionProposal(
                    connection, concurrentProposal);
                firstOutcome = result.outcome;
                firstId = result.persisted.proposalId;
            }
            catch (...) { firstError = std::current_exception(); }
        });
        std::thread second([&] {
            try
            {
                pqxx::connection connection{runtimeConnectionString};
                const auto result = PersistRecommendationConversionProposal(
                    connection, concurrentProposal);
                secondOutcome = result.outcome;
                secondId = result.persisted.proposalId;
            }
            catch (...) { secondError = std::current_exception(); }
        });
        first.join();
        second.join();
        if (firstError) std::rethrow_exception(firstError);
        if (secondError) std::rethrow_exception(secondError);
        assert(firstId == secondId);
        assert((firstOutcome ==
                    RecommendationConversionProposalPersistOutcome::created &&
                secondOutcome == RecommendationConversionProposalPersistOutcome::
                    existingIdentical) ||
               (secondOutcome ==
                    RecommendationConversionProposalPersistOutcome::created &&
                firstOutcome == RecommendationConversionProposalPersistOutcome::
                    existingIdentical));
        {
            pqxx::read_transaction check{runtime};
            assert(check.exec(
                "SELECT count(*) FROM "
                "experiment_recommendation_conversion_proposal WHERE "
                "conversion_identity_canonical=$1;",
                pqxx::params{concurrentProposal.conversionIdentityCanonical})
                .one_row()[0].as<int>() == 1);
        }

        // A rolled-back ordinal allocation leaves neither a row nor a lock.
        const ProposedExperimentSpecification rollbackProposal = Proposal("1.625");
        {
            pqxx::work rolledBack{owner};
            rolledBack.exec("SET LOCAL search_path TO " +
                            rolledBack.quote_name(schema) + ";");
            rolledBack.exec(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, "
                "7046029254386353131));",
                pqxx::params{rollbackProposal.conversionIdentityHash});
            rolledBack.exec(
                "INSERT INTO experiment_recommendation_conversion_proposal ("
                "recommendation_id,source_experiment_id,conversion_contract_version,"
                "changed_parameter,source_value_canonical,proposed_value_canonical,"
                "recommendation_semantic_hash,evaluation_identity_hash,"
                "evaluation_policy_hash,scoring_policy_hash,review_authorization_hash,"
                "proposed_symbol,proposed_prediction_horizon,"
                "proposed_label_threshold,proposed_target_epochs,"
                "proposed_train_start_date,proposed_train_end_date,"
                "proposed_checkpoint_interval,source_invocation_canonical,"
                "proposed_invocation_canonical,conversion_identity_canonical,"
                "conversion_identity_hash,conversion_hash_collision_ordinal) "
                "SELECT recommendation_id,source_experiment_id,"
                "conversion_contract_version,changed_parameter,"
                "source_value_canonical,proposed_value_canonical,"
                "recommendation_semantic_hash,evaluation_identity_hash,"
                "evaluation_policy_hash,scoring_policy_hash,"
                "review_authorization_hash,proposed_symbol,"
                "proposed_prediction_horizon,proposed_label_threshold,"
                "proposed_target_epochs,proposed_train_start_date,"
                "proposed_train_end_date,proposed_checkpoint_interval,"
                "source_invocation_canonical,proposed_invocation_canonical,"
                "'rolled_back_synthetic_canonical',$2,0 FROM "
                "experiment_recommendation_conversion_proposal WHERE "
                "recommendation_conversion_proposal_id=$1;",
                pqxx::params{created.persisted.proposalId,
                             rollbackProposal.conversionIdentityHash});
            rolledBack.abort();
        }
        const auto afterRollback = PersistRecommendationConversionProposal(
            runtime, rollbackProposal);
        assert(afterRollback.outcome ==
               RecommendationConversionProposalPersistOutcome::created);
        assert(afterRollback.persisted.hashCollisionOrdinal == 0);

        // Multiple distinct canonical texts can share one accelerator hash.
        // Synthetic owner rows deliberately model a real hash collision; the
        // authoritative canonical text remains distinct.
        const ProposedExperimentSpecification collisionProposal = Proposal("1.75");
        InsertSyntheticHashCollision(
            owner, schema, created.persisted.proposalId,
            "synthetic_distinct_canonical_for_collision_0",
            collisionProposal.conversionIdentityHash, 0);
        InsertSyntheticHashCollision(
            owner, schema, created.persisted.proposalId,
            "synthetic_distinct_canonical_for_collision_1",
            collisionProposal.conversionIdentityHash, 1);
        const auto collision = PersistRecommendationConversionProposal(
            runtime, collisionProposal);
        assert(collision.outcome == RecommendationConversionProposalPersistOutcome::
            createdWithHashCollision);
        assert(collision.persisted.hashCollisionOrdinal == 2);
        assert(collision.persisted.proposal.conversionIdentityCanonical ==
               collisionProposal.conversionIdentityCanonical);

        // A direct synthetic collision and a valid repository insertion use
        // the same bucket lock and allocate distinct ordinals under overlap.
        const ProposedExperimentSpecification concurrentCollision =
            Proposal("1.875");
        std::atomic<bool> startCollision{false};
        std::exception_ptr syntheticCollisionError;
        std::exception_ptr repositoryCollisionError;
        RecommendationConversionProposalPersistResult concurrentCollisionResult;
        std::thread syntheticCollision([&] {
            try
            {
                InsertSyntheticHashCollisionWithRepositoryProtocol(
                    ownerConnectionString, schema, created.persisted.proposalId,
                    "synthetic_concurrent_distinct_canonical",
                    concurrentCollision.conversionIdentityHash, startCollision);
            }
            catch (...) { syntheticCollisionError = std::current_exception(); }
        });
        std::thread repositoryCollision([&] {
            try
            {
                while (!startCollision.load(std::memory_order_acquire))
                    std::this_thread::yield();
                pqxx::connection connection{runtimeConnectionString};
                concurrentCollisionResult =
                    PersistRecommendationConversionProposal(
                        connection, concurrentCollision);
            }
            catch (...) { repositoryCollisionError = std::current_exception(); }
        });
        startCollision.store(true, std::memory_order_release);
        syntheticCollision.join();
        repositoryCollision.join();
        if (syntheticCollisionError)
            std::rethrow_exception(syntheticCollisionError);
        if (repositoryCollisionError)
            std::rethrow_exception(repositoryCollisionError);
        assert(concurrentCollisionResult.outcome ==
                   RecommendationConversionProposalPersistOutcome::created ||
               concurrentCollisionResult.outcome ==
                   RecommendationConversionProposalPersistOutcome::
                       createdWithHashCollision);
        {
            pqxx::read_transaction check{runtime};
            const pqxx::row bucket = check.exec(
                "SELECT count(*) AS row_count,min("
                "conversion_hash_collision_ordinal) AS min_ordinal,max("
                "conversion_hash_collision_ordinal) AS max_ordinal FROM "
                "experiment_recommendation_conversion_proposal WHERE "
                "conversion_identity_hash=$1;",
                pqxx::params{concurrentCollision.conversionIdentityHash})
                .one_row();
            assert(bucket["row_count"].as<int>() == 2);
            assert(bucket["min_ordinal"].as<int>() == 0);
            assert(bucket["max_ordinal"].as<int>() == 1);
        }

        // Exact canonical equality never hides inconsistent persisted fields.
        const ProposedExperimentSpecification mismatchProposal = Proposal("1.9375");
        const auto mismatchCreated = PersistRecommendationConversionProposal(
            runtime, mismatchProposal);
        {
            pqxx::work corrupt{owner};
            corrupt.exec("SET LOCAL search_path TO " +
                         corrupt.quote_name(schema) + ";");
            corrupt.exec(
                "UPDATE experiment_recommendation_conversion_proposal SET "
                "evaluation_policy_hash='synthetic_inconsistent_hash' WHERE "
                "recommendation_conversion_proposal_id=$1;",
                pqxx::params{mismatchCreated.persisted.proposalId});
            corrupt.commit();
        }
        bool mismatchRejected = false;
        try
        {
            (void)PersistRecommendationConversionProposal(
                runtime, mismatchProposal);
        }
        catch (const std::runtime_error& error)
        {
            mismatchRejected = std::string{error.what()} ==
                "inconsistent_persisted_recommendation_conversion_identity";
        }
        assert(mismatchRejected);

        // PostgreSQL hash indexes support long equality keys without a B-tree
        // tuple-size dependency. The repository enforces the 1 MiB byte bound.
        ProposedExperimentSpecification longProposal = WithCanonicalPadding(
            Proposal("2.125"), 1040000);
        assert(longProposal.conversionIdentityCanonical.size() <= 1048576);
        const auto longCreated = PersistRecommendationConversionProposal(
            runtime, longProposal);
        AssertRoundTrip(longProposal, longCreated.persisted);
        ProposedExperimentSpecification oversizedProposal = WithCanonicalPadding(
            Proposal("2.25"), 1048577);
        bool oversizedRejected = false;
        try
        {
            (void)PersistRecommendationConversionProposal(
                runtime, oversizedProposal);
        }
        catch (const std::invalid_argument&) { oversizedRejected = true; }
        assert(oversizedRejected);

        ProposedExperimentSpecification invalid = proposal;
        invalid.conversionIdentityHash = "fnv1a64:0000000000000000";
        bool invalidRejected = false;
        try { (void)PersistRecommendationConversionProposal(runtime, invalid); }
        catch (const std::invalid_argument&) { invalidRejected = true; }
        assert(invalidRejected);
        invalid = proposal;
        invalid.proposedInvocationCanonical.clear();
        invalidRejected = false;
        try { (void)PersistRecommendationConversionProposal(runtime, invalid); }
        catch (const std::invalid_argument&) { invalidRejected = true; }
        assert(invalidRejected);
        invalid = proposal;
        invalid.reviewAuthorizationHash = "   ";
        invalidRejected = false;
        try { (void)PersistRecommendationConversionProposal(runtime, invalid); }
        catch (const std::invalid_argument&) { invalidRejected = true; }
        assert(invalidRejected);
        bool boundedRejected = false;
        try
        {
            (void)ListRecommendationConversionProposalsByRecommendation(
                runtime, 42,
                kMaximumRecommendationConversionProposalListLimit + 1);
        }
        catch (const std::invalid_argument&) { boundedRejected = true; }
        assert(boundedRejected);

        for (const std::string& statement : {
                 std::string{"UPDATE experiment_recommendation_conversion_proposal "
                             "SET source_value_canonical='changed' WHERE "
                             "recommendation_conversion_proposal_id="} +
                     std::to_string(created.persisted.proposalId),
                 std::string{"DELETE FROM "
                             "experiment_recommendation_conversion_proposal WHERE "
                             "recommendation_conversion_proposal_id="} +
                     std::to_string(created.persisted.proposalId)})
        {
            bool permissionRejected = false;
            try
            {
                pqxx::work forbidden{runtime};
                forbidden.exec(statement);
                forbidden.commit();
            }
catch (const pqxx::sql_error& error)
{
    const std::string state = error.sqlstate();
    const std::string message = error.what();

    // Some libpqxx builds do not populate sqlstate() for permission errors.
    // Accept either the standard SQLSTATE or the server error text.
    permissionRejected =
        state == "42501" ||
        message.find("permission denied") != std::string::npos;
}
            assert(permissionRejected);
        }

        pqxx::read_transaction unchanged{owner};
        unchanged.exec("SET LOCAL search_path TO " +
                       unchanged.quote_name(schema) + ";");
        const pqxx::row state = unchanged.exec(
            "SELECT count(*) AS experiment_count,min(status) AS status,"
            "min(marker) AS marker FROM experiment;").one_row();
        assert(state["experiment_count"].as<int>() == 1);
        assert(state["status"].as<std::string>() == "paused");
        assert(state["marker"].as<std::string>() == "unchanged");

        assert(RecommendationConversionProposalPersistOutcomeText(
            RecommendationConversionProposalPersistOutcome::created) == "created");
        assert(RecommendationConversionProposalPersistOutcomeText(
            RecommendationConversionProposalPersistOutcome::existingIdentical) ==
            "existing_identical");
        assert(RecommendationConversionProposalPersistOutcomeText(
            RecommendationConversionProposalPersistOutcome::
                createdWithHashCollision) == "created_with_hash_collision");
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
    std::cout << "ExperimentRecommendationConversionRepositoryTests passed\n";
}
