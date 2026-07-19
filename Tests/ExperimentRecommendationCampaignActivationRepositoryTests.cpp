#include "../Sources/ExperimentRecommendationCampaignActivationRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignActivationService.hpp"
#include "../Sources/ExperimentRecommendationCampaignHandoffRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionExecutionRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionProposalReviewRepository.hpp"

#include <algorithm>
#include <cassert>
#include <barrier>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>

#include <pqxx/pqxx>

using namespace EA::ExperimentRecommendation;

namespace
{

bool IsApprovedIsolatedTestDatabase(const std::string& database)
{
    constexpr std::string_view prefixes[]{
        "expertadvisor_phase5_step2_test_",
        "expertadvisor_phase5_step2_final_"};
    for (const auto prefix : prefixes)
    {
        if (!database.starts_with(prefix) || database.size() == prefix.size())
            continue;
        return std::all_of(
            database.begin() + static_cast<std::ptrdiff_t>(prefix.size()),
            database.end(),
            [](unsigned char value)
            {
                return std::islower(value) || std::isdigit(value) ||
                    value == '_';
            });
    }
    return false;
}

std::optional<std::string> RequestedDatabaseDiagnostic(const char* database)
{
    if (database == nullptr || *database == '\0')
        return "LSTM_TEST_DB_NAME_required";
    if (!IsApprovedIsolatedTestDatabase(database))
        return "isolated_test_database_required";
    return std::nullopt;
}

std::optional<std::string> ConnectedDatabaseDiagnostic(
    const std::string& requested,
    const std::string& connected)
{
    if (connected != requested)
        return "connected_database_name_mismatch";
    if (!IsApprovedIsolatedTestDatabase(connected))
        return "isolated_test_database_required";
    return std::nullopt;
}

bool PermissionDenied(const pqxx::sql_error& error)
{
    if (error.sqlstate() == "42501") return true;
    return error.sqlstate().empty() &&
        std::string{error.what()}.find("permission denied") != std::string::npos;
}

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

void SetSearchPath(pqxx::transaction_base& transaction, const std::string& schema)
{
    transaction.exec(
        "SET LOCAL search_path TO " + transaction.quote_name(schema) + ";");
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

ProposedExperimentSpecification Proposal(double proposed, int variant)
{
    RecommendationConversionRequest request;
    request.recommendationExists = true;
    request.recommendationId = 42;
    request.recommendationStatus = RecommendationStatus::approved;
    request.sourceExperimentId = 17;
    request.recommendationSourceExperimentId = 17;
    request.sourceInvocation = SourceInvocation();
    request.mutations.push_back(
        {kCoreLrMult, "1", CanonicalRecommendationDouble(proposed)});
    request.reviewAuthorization.present = true;
    request.reviewAuthorization.recommendationId = 42;
    request.reviewAuthorization.latestAction = RecommendationReviewAction::approve;
    request.reviewAuthorization.resultingStatus = RecommendationStatus::approved;
    request.reviewAuthorization.latestActionEffective = true;
    request.reviewAuthorization.authorizationCanonical =
        "review_authorization_v1;variant=" + std::to_string(variant);
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
        "evaluation_identity_v1;variant=" + std::to_string(variant);
    request.evaluation.evaluationIdentityHash = RecommendationCanonicalHash(
        request.evaluation.evaluationIdentityCanonical);
    request.evaluation.evaluationPolicyCanonical = "evaluation_policy_v1";
    request.evaluation.evaluationPolicyHash = RecommendationCanonicalHash(
        request.evaluation.evaluationPolicyCanonical);
    request.score.state = RecommendationConversionEvidenceState::completed;
    request.score.valid = true;
    request.score.recommendationId = 42;
    request.score.finalScore = 0.75;
    request.score.scoringPolicyCanonical = "scoring_policy_v1";
    request.score.scoringPolicyHash = RecommendationCanonicalHash(
        request.score.scoringPolicyCanonical);
    request.evaluation.scoringPolicyHash = request.score.scoringPolicyHash;
    auto proposedInvocation = request.sourceInvocation;
    proposedInvocation.configuration.coreLrMult = proposed;
    const auto semantic = BuildRecommendationCandidateIdentity(
        proposedInvocation.configuration);
    const auto invocation = BuildRecommendationInvocationIdentity(
        proposedInvocation);
    request.recommendationSemanticCanonical = semantic.canonicalText;
    request.recommendationSemanticHash = semantic.hash;
    request.recommendationInvocationCanonical = invocation.canonicalText;
    request.recommendationInvocationHash = invocation.hash;
    const auto result = BuildProposedExperimentSpecification(request);
    assert(result.eligibility.eligible && result.proposal);
    return *result.proposal;
}

RecommendationConversionProposalReviewRequest Review(
    long long proposalId,
    const std::string& requestId,
    RecommendationConversionProposalReviewDecision decision =
        RecommendationConversionProposalReviewDecision::approve)
{
    RecommendationConversionProposalReviewRequest request;
    request.proposalId = proposalId;
    request.decision = decision;
    request.requestId = requestId;
    request.operatorIdentity = "phase5 step2 test operator";
    request.reasonText = "Explicit isolated Phase 5 activation test.";
    return request;
}

struct ExecutedProposal
{
    PersistedRecommendationConversionProposal proposal;
    PersistedRecommendationConversionExecution execution;
};

ExecutedProposal CreateExecution(
    pqxx::connection& runtime,
    double proposed,
    int variant)
{
    ExecutedProposal result;
    result.proposal = PersistRecommendationConversionProposal(
        runtime, Proposal(proposed, variant)).persisted;
    const auto review = RecordRecommendationConversionProposalReviewDecision(
        runtime,
        Review(result.proposal.proposalId,
               "phase5-activation-" + std::to_string(variant)));
    assert(review.decision);
    const auto execution = ExecuteApprovedRecommendationConversionProposal(
        runtime, result.proposal.proposalId);
    assert(execution.outcome == RecommendationConversionExecutionOutcome::created);
    assert(execution.execution);
    result.execution = *execution.execution;
    return result;
}

struct MaterializationFixture
{
    long long approvalId = -1;
    long long rankingSnapshotId = -1;
    RecommendationCampaignPlan plan;
    RecommendationCampaignApprovalEvidence approval;
    RecommendationCampaignMaterializationEvidence materialization;
};

MaterializationFixture BuildMaterializationFixture(
    long long materializationId,
    const std::vector<ExecutedProposal>& values)
{
    assert(materializationId > 0 && !values.empty());
    MaterializationFixture fixture;
    fixture.approvalId = 1000 + materializationId;
    fixture.rankingSnapshotId = 2000 + materializationId;

    RecommendationCampaignPlanInput input;
    input.policy.enabled = true;
    input.policy.maximumSelectedRecommendations =
        static_cast<int>(values.size());
    input.policy.maximumCandidatesConsidered =
        static_cast<int>(values.size());
    input.policy.maximumPerSourceExperiment.reset();
    input.scope.rankingSnapshotId = fixture.rankingSnapshotId;
    input.rankingSnapshotIdentityCanonical =
        "phase5-step2-ranking-snapshot-" +
        std::to_string(materializationId);
    input.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        input.rankingSnapshotIdentityCanonical);
    input.generatedAt = "display-only";
    const std::string rankingPolicy = "phase5-step2-ranking-policy-v1";
    const std::string rankingPolicyHash = RecommendationCanonicalHash(
        rankingPolicy);
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        const auto& proposal = values[index].proposal.proposal;
        RecommendationCampaignCandidateInput candidate;
        candidate.rankingSnapshotId = fixture.rankingSnapshotId;
        candidate.rankingSnapshotIdentityCanonical =
            input.rankingSnapshotIdentityCanonical;
        candidate.rankingSnapshotIdentityHash =
            input.rankingSnapshotIdentityHash;
        candidate.rankingPolicyCanonical = rankingPolicy;
        candidate.rankingPolicyHash = rankingPolicyHash;
        candidate.rankingVersion = 1;
        candidate.rankingMemberId =
            300000 + materializationId * 100 +
            static_cast<long long>(index) + 1;
        candidate.rankingPosition = static_cast<int>(index) + 1;
        candidate.rankingBucket = "advisory_ready";
        candidate.rankingScore = 0.9;
        candidate.recommendationId = proposal.recommendationId;
        candidate.sourceExperimentId = proposal.sourceExperimentId;
        candidate.symbol = proposal.proposedInvocation.configuration.symbol;
        candidate.predictionHorizon =
            proposal.proposedInvocation.configuration.predictionHorizon;
        candidate.family = "core_lr_mult";
        candidate.targetEpochs =
            proposal.proposedInvocation.configuration.targetEpochs;
        candidate.leaderScore = 0.9;
        candidate.inferenceAccuracy = 0.8;
        candidate.predictedNeutralProportion = 0.2;
        candidate.recommendationSemanticCanonical =
            "phase5-step2-semantic-" + std::to_string(materializationId) +
            "-" + std::to_string(index + 1);
        candidate.recommendationSemanticHash = RecommendationCanonicalHash(
            candidate.recommendationSemanticCanonical);
        candidate.recommendationInvocationCanonical =
            proposal.proposedInvocationCanonical;
        candidate.recommendationInvocationHash = RecommendationCanonicalHash(
            candidate.recommendationInvocationCanonical);
        input.candidates.push_back(std::move(candidate));
    }
    fixture.plan = PlanRecommendationCampaign(input);
    assert(fixture.plan.summary.selectedCount ==
           static_cast<int>(values.size()));
    const auto review = ReviewRecommendationCampaignPlan(fixture.plan);
    RecommendationCampaignApprovalRequest approvalRequest;
    approvalRequest.expectedCampaignReviewIdentityHash = review.identityHash;
    approvalRequest.reviewerIdentity = "phase5 step2 test reviewer";
    approvalRequest.reasonText = "Approve exact activation test campaign.";
    fixture.approval = BuildRecommendationCampaignApprovalEvidence(
        fixture.rankingSnapshotId,
        input.rankingSnapshotIdentityCanonical,
        input.rankingSnapshotIdentityHash,
        fixture.plan,
        review,
        approvalRequest);
    std::vector<ProposedExperimentSpecification> proposals;
    proposals.reserve(values.size());
    for (const auto& value : values)
        proposals.push_back(value.proposal.proposal);
    fixture.materialization = BuildRecommendationCampaignMaterializationEvidence(
        {fixture.approvalId, "phase5 step2 materializer",
         "Materialize exact activation test campaign."},
        fixture.approval,
        fixture.plan,
        proposals);
    return fixture;
}

void InsertMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId,
    const std::vector<ExecutedProposal>& values)
{
    const auto fixture = BuildMaterializationFixture(materializationId, values);
    const auto& approval = fixture.approval;
    const auto& summary = approval.summary;
    transaction.exec(
        "INSERT INTO experiment_recommendation_ranking_snapshot VALUES($1);",
        pqxx::params{fixture.rankingSnapshotId});
    for (const auto& member : fixture.materialization.members)
        transaction.exec(
            "INSERT INTO experiment_recommendation_ranking_member("
            "recommendation_ranking_member_id,"
            "recommendation_ranking_snapshot_id,recommendation_id,"
            "source_experiment_id,global_ordinal) VALUES($1,$2,$3,$4,$5);",
            pqxx::params{
                member.rankingMemberId, fixture.rankingSnapshotId,
                member.recommendationId, member.sourceExperimentId,
                member.rankingPosition});
    transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_approval("
        "recommendation_campaign_approval_id,approval_contract_version,"
        "recommendation_ranking_snapshot_id,ranking_snapshot_identity_canonical,"
        "ranking_snapshot_identity_hash,planning_policy_canonical,"
        "planning_policy_hash,planning_scope_canonical,"
        "campaign_plan_identity_canonical,campaign_plan_identity_hash,"
        "review_contract_version,campaign_review_identity_canonical,"
        "campaign_review_identity_hash,campaign_review_hash_collision_ordinal,"
        "candidate_count,selected_count,excluded_count,duplicate_group_count,"
        "duplicate_candidate_count,considered_family_count,selected_family_count,"
        "considered_symbol_count,selected_symbol_count,considered_horizon_count,"
        "selected_horizon_count,deterministic_ordering_verified,decision,"
        "reviewer_identity,reason_text,approval_identity_canonical,"
        "approval_identity_hash) VALUES($1,1,$2,$3,$4,$5,$6,$7,$8,$9,1,$10,"
        "$11,0,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,true,'approved',"
        "$23,$24,$25,$26);",
        pqxx::params{
            fixture.approvalId, approval.rankingSnapshotId,
            approval.rankingSnapshotIdentityCanonical,
            approval.rankingSnapshotIdentityHash,
            approval.planningPolicyCanonical, approval.planningPolicyHash,
            approval.planningScopeCanonical,
            approval.campaignPlanIdentityCanonical,
            approval.campaignPlanIdentityHash,
            approval.campaignReviewIdentityCanonical,
            approval.campaignReviewIdentityHash, summary.candidateCount,
            summary.selectedCount, summary.excludedCount,
            summary.duplicateGroupCount, summary.duplicateCandidateCount,
            summary.consideredFamilyCount, summary.selectedFamilyCount,
            summary.consideredSymbolCount, summary.selectedSymbolCount,
            summary.consideredHorizonCount, summary.selectedHorizonCount,
            approval.reviewerIdentity, approval.reasonText,
            approval.approvalIdentityCanonical, approval.approvalIdentityHash});
    const auto& evidence = fixture.materialization;
    transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_materialization("
        "recommendation_campaign_materialization_id,"
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
        "VALUES($1,$2,1,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,'approved',"
        "$15,$16,$17,$18,$19,0,$19,$20,$21);",
        pqxx::params{
            materializationId, fixture.approvalId,
            approval.approvalIdentityCanonical, approval.approvalIdentityHash,
            approval.rankingSnapshotId,
            approval.rankingSnapshotIdentityCanonical,
            approval.rankingSnapshotIdentityHash,
            approval.planningPolicyCanonical, approval.planningPolicyHash,
            approval.planningScopeCanonical,
            approval.campaignPlanIdentityCanonical,
            approval.campaignPlanIdentityHash,
            approval.campaignReviewIdentityCanonical,
            approval.campaignReviewIdentityHash, approval.reviewerIdentity,
            approval.reasonText, evidence.operatorIdentity, evidence.reasonText,
            evidence.selectedMemberCount,
            evidence.materializationIdentityCanonical,
            evidence.materializationIdentityHash});
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        const auto& proposal = values[index].proposal;
        const auto& member = evidence.members[index];
        transaction.exec(
            "INSERT INTO experiment_recommendation_campaign_materialization_member("
            "recommendation_campaign_materialization_id,member_ordinal,"
            "recommendation_ranking_member_id,recommendation_id,"
            "source_experiment_id,ranking_position,"
            "selected_member_identity_canonical,selected_member_identity_hash,"
            "recommendation_conversion_proposal_id,proposal_identity_canonical,"
            "proposal_identity_hash) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11);",
            pqxx::params{
                materializationId, member.memberOrdinal,
                member.rankingMemberId, proposal.proposal.recommendationId,
                proposal.proposal.sourceExperimentId, member.rankingPosition,
                member.selectedMemberIdentityCanonical,
                member.selectedMemberIdentityHash, proposal.proposalId,
                proposal.proposal.conversionIdentityCanonical,
                proposal.proposal.conversionIdentityHash});
    }
}

struct RunResult
{
    RecommendationCampaignActivationPlan plan;
    std::vector<PersistedRecommendationConversionActivation> activations;
};

RunResult Run(const std::string& connectionString, long long materializationId)
{
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};
    const auto materialization = LoadRecommendationCampaignActivationMaterialization(
        transaction, materializationId);
    assert(materialization);
    LockRecommendationCampaignActivations(transaction, *materialization);
    const auto input = LoadRecommendationCampaignActivationInput(
        transaction, *materialization);
    RunResult result;
    result.plan = BuildRecommendationCampaignActivationPlan(
        {materializationId, false}, input);
    if (result.plan.state == RecommendationCampaignActivationPlanState::ready)
        result.activations = PersistRecommendationCampaignActivations(
            transaction, result.plan);
    transaction.commit();
    return result;
}

long long Scalar(
    pqxx::connection& connection,
    const std::string& schema,
    const std::string& sql)
{
    pqxx::read_transaction transaction{connection};
    SetSearchPath(transaction, schema);
    return transaction.exec(sql).one_row()[0].as<long long>();
}

std::string Text(
    pqxx::connection& connection,
    const std::string& schema,
    const std::string& sql)
{
    pqxx::read_transaction transaction{connection};
    SetSearchPath(transaction, schema);
    return transaction.exec(sql).one_row()[0].as<std::string>();
}

long long SequenceValue(
    pqxx::connection& connection,
    const std::string& schema,
    const std::string& table,
    const std::string& column)
{
    pqxx::read_transaction transaction{connection};
    SetSearchPath(transaction, schema);
    const std::string sequence = transaction.exec(
        "SELECT pg_get_serial_sequence($1,$2);",
        pqxx::params{table, column}).one_row()[0].as<std::string>();
    return transaction.exec("SELECT last_value FROM " + sequence + ";")
        .one_row()[0].as<long long>();
}

} // namespace

int main()
{
    assert(RequestedDatabaseDiagnostic(nullptr) ==
           std::optional<std::string>{"LSTM_TEST_DB_NAME_required"});
    assert(RequestedDatabaseDiagnostic("") ==
           std::optional<std::string>{"LSTM_TEST_DB_NAME_required"});
    for (const char* unsafe : {"LSTM", "lstm", "Lstm", "production",
                               "postgres", "unrelated_database"})
        assert(RequestedDatabaseDiagnostic(unsafe) ==
               std::optional<std::string>{"isolated_test_database_required"});
    assert(!RequestedDatabaseDiagnostic(
        "expertadvisor_phase5_step2_test_20260719_01"));
    assert(!RequestedDatabaseDiagnostic(
        "expertadvisor_phase5_step2_final_20260719_01"));
    assert(ConnectedDatabaseDiagnostic(
               "expertadvisor_phase5_step2_test_requested",
               "expertadvisor_phase5_step2_test_connected") ==
           std::optional<std::string>{"connected_database_name_mismatch"});

    const char* testDatabase = std::getenv("LSTM_TEST_DB_NAME");
    if (const auto diagnostic = RequestedDatabaseDiagnostic(testDatabase))
    {
        std::cerr << *diagnostic << '\n';
        return 2;
    }
    const std::string database = testDatabase;
    {
        std::ostringstream output;
        std::ostringstream errors;
        assert(RunRecommendationCampaignActivationCommand(
            "host=invalid.invalid dbname=must_not_connect", {0, false},
            output, errors) == 1);
        assert(output.str().empty());
        assert(errors.str().find(
            "campaign_activation_materialization_id_invalid") !=
            std::string::npos);
    }

    const std::string host = EnvironmentOr("LSTM_TEST_DB_HOST", "127.0.0.1");
    const std::string port = EnvironmentOr("LSTM_TEST_DB_PORT", "5432");
    const std::string ownerUser = EnvironmentOr(
        "LSTM_TEST_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string schema =
        "phase5_campaign_activation_" + std::to_string(getpid());
    const std::string ownerConnectionString =
        "host=" + host + " port=" + port + " user=" + ownerUser +
        " dbname=" + database;
    const std::string runtimeConnectionString =
        "host=" + host + " port=" + port + " user=pqxx dbname=" + database +
        " options='-c search_path=" + schema + "'";
    pqxx::connection owner{ownerConnectionString};
    {
        pqxx::read_transaction identity{owner};
        const std::string connected = identity.exec(
            "SELECT current_database();").one_row()[0].as<std::string>();
        if (const auto diagnostic = ConnectedDatabaseDiagnostic(
                database, connected))
        {
            std::cerr << *diagnostic << '\n';
            return 2;
        }
    }

    try
    {
        {
            pqxx::work setup{owner};
            setup.exec("CREATE SCHEMA " + setup.quote_name(schema) + ";");
            SetSearchPath(setup, schema);
            setup.exec(R"SQL(
CREATE TABLE experiment(
 experiment_id bigserial PRIMARY KEY,symbol text NOT NULL,
 prediction_horizon integer NOT NULL,c_next_threshold double precision NOT NULL,
 core_lr_mult double precision,head_lr_mult double precision,
 target_epochs integer NOT NULL,checkpoint_interval integer NOT NULL,
 train_start timestamptz NOT NULL,train_end timestamptz NOT NULL,
 infer_start timestamptz,infer_end timestamptz,resume_model_id bigint,
 duplicate_nonce bigint NOT NULL DEFAULT 0,status text NOT NULL,
 phase text NOT NULL,invocation_mode text,updated_at timestamptz NOT NULL DEFAULT now(),
 worker_pid integer,current_operation text,current_epoch integer,
 worker_started_at timestamptz,started_at timestamptz,completed_at timestamptz,
 exit_code integer,error_message text,last_model_id bigint,marker text);
CREATE UNIQUE INDEX experiment_unique_identity_uidx ON experiment(
 symbol,prediction_horizon,c_next_threshold,
 coalesce(core_lr_mult,'-infinity'::double precision),
 coalesce(head_lr_mult,'-infinity'::double precision),target_epochs,
 checkpoint_interval,train_start,train_end,
 coalesce(infer_start,'-infinity'::timestamptz),
 coalesce(infer_end,'-infinity'::timestamptz),
 coalesce(resume_model_id,-1),duplicate_nonce) WHERE status<>'cancelled';
CREATE TABLE model(model_id bigint PRIMARY KEY,marker text NOT NULL);
CREATE TABLE experiment_recommendation(
 recommendation_id bigint PRIMARY KEY,source_experiment_id bigint NOT NULL
 REFERENCES experiment(experiment_id),status text NOT NULL);
CREATE TABLE experiment_recommendation_ranking_snapshot(
 recommendation_ranking_snapshot_id bigint PRIMARY KEY);
CREATE TABLE experiment_recommendation_ranking_member(
 recommendation_ranking_member_id bigint PRIMARY KEY,
 recommendation_ranking_snapshot_id bigint NOT NULL REFERENCES
  experiment_recommendation_ranking_snapshot(recommendation_ranking_snapshot_id),
 recommendation_id bigint NOT NULL REFERENCES
  experiment_recommendation(recommendation_id),
 source_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
 global_ordinal integer NOT NULL CHECK(global_ordinal > 0));
INSERT INTO experiment(experiment_id,symbol,prediction_horizon,c_next_threshold,
 core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,train_start,train_end,
 infer_start,infer_end,resume_model_id,status,phase,marker,updated_at)
VALUES (17,'eurusd',12,0.001,1,5,120,15,
 '2010-01-01 America/Chicago','2025-01-01 America/Chicago',
 '2025-01-01 America/Chicago','2026-01-01 America/Chicago',77,
 'paused','train','unchanged','2026-07-19 12:34:56+00');
INSERT INTO model VALUES (77,'unchanged');
INSERT INTO experiment_recommendation VALUES (42,17,'approved');
)SQL");
            for (const char* migration : {
                     "Database/migrations/036_experiment_recommendation_conversion_proposal.sql",
                     "Database/migrations/037_experiment_recommendation_conversion_review.sql",
                     "Database/migrations/038_experiment_recommendation_conversion_execution.sql",
                     "Database/migrations/039_experiment_recommendation_conversion_activation.sql",
                     "Database/migrations/040_experiment_recommendation_campaign_approval.sql",
                     "Database/migrations/041_experiment_recommendation_campaign_materialization.sql"})
            {
                const std::string sql = ReadFile(migration);
                setup.exec(sql);
                setup.exec(sql);
            }
            setup.exec(ReadFile(
                "Tests/ExperimentRecommendationCampaignMaterializationMigrationTests.sql"));
            setup.exec(
                "GRANT SELECT,INSERT,UPDATE ON experiment TO pqxx;"
                "GRANT USAGE ON SEQUENCE experiment_experiment_id_seq TO pqxx;"
                "GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                " TO pqxx; GRANT SELECT ON model,experiment_recommendation,"
                "experiment_recommendation_ranking_snapshot,"
                "experiment_recommendation_ranking_member TO pqxx;");
            setup.commit();
        }

        pqxx::connection runtime{runtimeConnectionString};
        const std::string sourceBefore = Text(
            owner, schema,
            "SELECT row_to_json(e)::text FROM experiment e WHERE experiment_id=17;");

        std::vector<ExecutedProposal> success{
            CreateExecution(runtime, 1.25, 1),
            CreateExecution(runtime, 1.5, 2)};
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 1, success);
            setup.commit();
        }
        const auto persistedMaterialization =
            FindRecommendationCampaignMaterialization(runtime, 1);
        assert(persistedMaterialization);
        assert(persistedMaterialization->selectedMemberCount == 2);
        assert(persistedMaterialization->members.size() == 2);
        assert(persistedMaterialization->members[0].memberOrdinal == 1);
        assert(persistedMaterialization->members[1].memberOrdinal == 2);
        assert(persistedMaterialization->members[0].conversionProposalId ==
               success[0].proposal.proposalId);
        assert(persistedMaterialization->members[1].conversionProposalId ==
               success[1].proposal.proposalId);
        {
            bool updateDenied = false;
            try
            {
                pqxx::work forbidden{runtime};
                forbidden.exec(
                    "UPDATE experiment_recommendation_campaign_materialization "
                    "SET materialized_by='forged' WHERE "
                    "recommendation_campaign_materialization_id=1;");
            }
            catch (const pqxx::sql_error& error)
            {
                updateDenied = PermissionDenied(error);
            }
            assert(updateDenied);
        }
        {
            bool deleteDenied = false;
            try
            {
                pqxx::work forbidden{runtime};
                forbidden.exec(
                    "DELETE FROM "
                    "experiment_recommendation_campaign_materialization_member "
                    "WHERE recommendation_campaign_materialization_id=1;");
            }
            catch (const pqxx::sql_error& error)
            {
                deleteDenied = PermissionDenied(error);
            }
            assert(deleteDenied);
        }
        {
            bool malformedRejected = false;
            const auto& member = persistedMaterialization->members[0];
            try
            {
                pqxx::work malformed{runtime};
                malformed.exec(
                    "INSERT INTO "
                    "experiment_recommendation_campaign_materialization_member("
                    "recommendation_campaign_materialization_id,member_ordinal,"
                    "recommendation_ranking_member_id,recommendation_id,"
                    "source_experiment_id,ranking_position,"
                    "selected_member_identity_canonical,"
                    "selected_member_identity_hash,"
                    "recommendation_conversion_proposal_id,"
                    "proposal_identity_canonical,proposal_identity_hash) "
                    "VALUES(1,3,$1,$2,$3,3,$4,$5,$6,$7,$8);",
                    pqxx::params{
                        member.rankingMemberId, member.recommendationId,
                        member.sourceExperimentId,
                        member.selectedMemberIdentityCanonical,
                        member.selectedMemberIdentityHash,
                        member.conversionProposalId,
                        member.proposalIdentityCanonical,
                        member.proposalIdentityHash});
                malformed.commit();
            }
            catch (const pqxx::check_violation&)
            {
                malformedRejected = true;
            }
            assert(malformedRejected);
        }
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM information_schema.tables WHERE "
            "table_schema=current_schema() AND "
            "table_name LIKE 'experiment_recommendation_campaign_activation%';") == 0);
        const auto first = Run(runtimeConnectionString, 1);
        assert(first.activations.size() == 2);
        assert(first.activations[0].executionId == success[0].execution.executionId);
        assert(first.activations[1].executionId == success[1].execution.executionId);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment_recommendation_conversion_activation;") == 2);
        for (const auto& value : success)
        {
            assert(Text(owner, schema,
                "SELECT status||'/'||phase FROM experiment WHERE experiment_id=" +
                std::to_string(value.execution.experimentId) + ";") ==
                "pending/train");
            assert(FindRecommendationConversionActivationByExecution(
                runtime, value.execution.executionId));
        }
        const auto retry = Run(runtimeConnectionString, 1);
        assert(retry.plan.state ==
               RecommendationCampaignActivationPlanState::alreadySatisfied);
        assert(retry.activations.empty());
        const long long satisfiedDryRunSequenceBefore = SequenceValue(
            owner, schema, "experiment_recommendation_conversion_activation",
            "recommendation_conversion_activation_id");
        {
            std::ostringstream output;
            std::ostringstream errors;
            assert(RunRecommendationCampaignActivationCommand(
                runtimeConnectionString, {1, true}, output, errors) == 0);
            assert(errors.str().empty());
            assert(output.str().find("result=already_satisfied") !=
                   std::string::npos);
            assert(output.str().find("read_only=true") != std::string::npos);
            const auto firstOrdinal = output.str().find("member_ordinal=1");
            const auto secondOrdinal = output.str().find("member_ordinal=2");
            assert(firstOrdinal != std::string::npos);
            assert(secondOrdinal != std::string::npos);
            assert(firstOrdinal < secondOrdinal);
        }
        assert(SequenceValue(
            owner, schema, "experiment_recommendation_conversion_activation",
            "recommendation_conversion_activation_id") ==
            satisfiedDryRunSequenceBefore);
        {
            pqxx::read_transaction read{owner};
            SetSearchPath(read, schema);
            const auto handoff = FindRecommendationCampaignHandoff(read, 1);
            assert(handoff && handoff->state ==
                RecommendationCampaignHandoffState::fullyActivated);
        }

        const auto reversal = RecordRecommendationConversionProposalReviewDecision(
            runtime,
            Review(success[0].proposal.proposalId, "phase5-activation-reversal",
                   RecommendationConversionProposalReviewDecision::reject));
        assert(reversal.decision);
        {
            std::ostringstream output;
            std::ostringstream errors;
            assert(RunRecommendationCampaignActivationCommand(
                runtimeConnectionString, {1, false}, output, errors) == 0);
            assert(errors.str().empty());
            assert(output.str().find("result=already_satisfied") !=
                   std::string::npos);
            assert(output.str().find(
                "authorization_review_decision_id=" +
                std::to_string(success[0].execution.reviewDecisionId)) !=
                std::string::npos);
            assert(output.str().find(
                "authorization_review_decision_id=" +
                std::to_string(reversal.decision->reviewDecisionId)) ==
                std::string::npos);
        }

        std::vector<ExecutedProposal> dry{CreateExecution(runtime, 1.75, 3)};
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 2, dry);
            setup.commit();
        }
        const long long activationSequenceBefore = SequenceValue(
            owner, schema, "experiment_recommendation_conversion_activation",
            "recommendation_conversion_activation_id");
        const std::string dryExperimentBefore = Text(
            owner, schema,
            "SELECT row_to_json(e)::text FROM experiment e WHERE experiment_id=" +
            std::to_string(dry[0].execution.experimentId) + ";");
        {
            std::ostringstream output;
            std::ostringstream errors;
            assert(RunRecommendationCampaignActivationCommand(
                runtimeConnectionString, {2, true}, output, errors) == 0);
            assert(errors.str().empty());
            assert(output.str().find("result=dry_run_ready") !=
                   std::string::npos);
            assert(output.str().find("read_only=true") != std::string::npos);
        }
        assert(SequenceValue(
            owner, schema, "experiment_recommendation_conversion_activation",
            "recommendation_conversion_activation_id") ==
            activationSequenceBefore);
        assert(Text(owner, schema,
            "SELECT row_to_json(e)::text FROM experiment e WHERE experiment_id=" +
            std::to_string(dry[0].execution.experimentId) + ";") ==
            dryExperimentBefore);
        assert(!FindRecommendationConversionActivationByExecution(
            runtime, dry[0].execution.executionId));

        std::vector<ExecutedProposal> invalid{
            CreateExecution(runtime, 2.0, 4),
            CreateExecution(runtime, 2.25, 5)};
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 3, invalid);
            setup.exec(
                "UPDATE experiment SET current_operation='unexpected' WHERE "
                "experiment_id=$1;",
                pqxx::params{invalid[1].execution.experimentId});
            setup.commit();
        }
        bool invalidFailed = false;
        try
        {
            (void)Run(runtimeConnectionString, 3);
        }
        catch (const std::invalid_argument&)
        {
            invalidFailed = true;
        }
        assert(invalidFailed);
        assert(!FindRecommendationConversionActivationByExecution(
            runtime, invalid[0].execution.executionId));
        assert(Text(owner, schema,
            "SELECT status FROM experiment WHERE experiment_id=" +
            std::to_string(invalid[0].execution.experimentId) + ";") ==
            "paused");
        {
            pqxx::work reset{owner};
            SetSearchPath(reset, schema);
            reset.exec(
                "UPDATE experiment SET current_operation=NULL WHERE "
                "experiment_id=$1;",
                pqxx::params{invalid[1].execution.experimentId});
            reset.commit();
        }

        std::vector<ExecutedProposal> rollback{
            CreateExecution(runtime, 2.5, 6),
            CreateExecution(runtime, 2.75, 7)};
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 4, rollback);
            setup.exec(R"SQL(
CREATE FUNCTION fail_phase5_second_activation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.recommendation_conversion_execution_id = TG_ARGV[0]::bigint THEN
  RAISE EXCEPTION 'injected phase5 activation failure';
 END IF;
 RETURN NEW;
END $$;
)SQL");
            setup.exec(
                "CREATE TRIGGER fail_phase5_second_activation_trigger BEFORE INSERT "
                "ON experiment_recommendation_conversion_activation FOR EACH ROW "
                "EXECUTE FUNCTION fail_phase5_second_activation('" +
                std::to_string(rollback[1].execution.executionId) + "');");
            setup.commit();
        }
        bool rolledBack = false;
        try
        {
            (void)Run(runtimeConnectionString, 4);
        }
        catch (const std::exception&)
        {
            rolledBack = true;
        }
        assert(rolledBack);
        for (const auto& value : rollback)
        {
            assert(!FindRecommendationConversionActivationByExecution(
                runtime, value.execution.executionId));
            assert(Text(owner, schema,
                "SELECT status FROM experiment WHERE experiment_id=" +
                std::to_string(value.execution.experimentId) + ";") == "paused");
        }
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            setup.exec("DROP TRIGGER fail_phase5_second_activation_trigger ON "
                       "experiment_recommendation_conversion_activation;"
                       "DROP FUNCTION fail_phase5_second_activation();");
            setup.commit();
        }

        std::vector<ExecutedProposal> mixed{
            CreateExecution(runtime, 3.0, 8),
            CreateExecution(runtime, 3.25, 9)};
        const auto directMixed = ActivateRecommendationConversionExecution(
            runtime, mixed[0].execution.executionId);
        assert(directMixed.outcome ==
               RecommendationConversionActivationOutcome::activated);
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 5, mixed);
            setup.commit();
        }
        bool mixedConflict = false;
        try
        {
            (void)Run(runtimeConnectionString, 5);
        }
        catch (const std::invalid_argument& error)
        {
            mixedConflict = std::string{error.what()} ==
                "campaign_activation_partial_operation_conflict";
        }
        assert(mixedConflict);
        assert(!FindRecommendationConversionActivationByExecution(
            runtime, mixed[1].execution.executionId));

        std::vector<ExecutedProposal> identical{
            CreateExecution(runtime, 3.5, 10),
            CreateExecution(runtime, 3.75, 11)};
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 6, identical);
            setup.commit();
        }
        RunResult identicalLeft;
        RunResult identicalRight;
        std::exception_ptr identicalLeftError;
        std::exception_ptr identicalRightError;
        std::barrier identicalStart{3};
        auto invokeIdentical = [&](RunResult& result, std::exception_ptr& error) {
            try
            {
                identicalStart.arrive_and_wait();
                result = Run(runtimeConnectionString, 6);
            }
            catch (...)
            {
                error = std::current_exception();
            }
        };
        std::thread identicalLeftThread{
            invokeIdentical, std::ref(identicalLeft),
            std::ref(identicalLeftError)};
        std::thread identicalRightThread{
            invokeIdentical, std::ref(identicalRight),
            std::ref(identicalRightError)};
        identicalStart.arrive_and_wait();
        identicalLeftThread.join();
        identicalRightThread.join();
        if (identicalLeftError) std::rethrow_exception(identicalLeftError);
        if (identicalRightError) std::rethrow_exception(identicalRightError);
        assert((identicalLeft.activations.size() == 2 &&
                identicalRight.activations.empty()) ||
               (identicalRight.activations.size() == 2 &&
                identicalLeft.activations.empty()));

        std::vector<ExecutedProposal> overlap{
            CreateExecution(runtime, 4.0, 12),
            CreateExecution(runtime, 4.25, 13),
            CreateExecution(runtime, 4.5, 14)};
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 7, {overlap[0], overlap[1]});
            InsertMaterialization(setup, 8, {overlap[2], overlap[1]});
            setup.commit();
        }
        RunResult overlapLeft;
        RunResult overlapRight;
        std::exception_ptr overlapLeftError;
        std::exception_ptr overlapRightError;
        std::barrier overlapStart{3};
        auto invokeOverlap = [&](long long id, RunResult& result,
                                 std::exception_ptr& error) {
            try
            {
                overlapStart.arrive_and_wait();
                result = Run(runtimeConnectionString, id);
            }
            catch (...)
            {
                error = std::current_exception();
            }
        };
        std::thread overlapLeftThread{
            invokeOverlap, 7, std::ref(overlapLeft),
            std::ref(overlapLeftError)};
        std::thread overlapRightThread{
            invokeOverlap, 8, std::ref(overlapRight),
            std::ref(overlapRightError)};
        overlapStart.arrive_and_wait();
        overlapLeftThread.join();
        overlapRightThread.join();
        assert(static_cast<bool>(overlapLeftError) !=
               static_cast<bool>(overlapRightError));
        bool overlapConflict = false;
        try
        {
            std::rethrow_exception(
                overlapLeftError ? overlapLeftError : overlapRightError);
        }
        catch (const std::invalid_argument& error)
        {
            overlapConflict = std::string{error.what()} ==
                "campaign_activation_partial_operation_conflict";
        }
        assert(overlapConflict);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment_recommendation_conversion_activation "
            "WHERE recommendation_conversion_execution_id IN (" +
            std::to_string(overlap[0].execution.executionId) + "," +
            std::to_string(overlap[1].execution.executionId) + "," +
            std::to_string(overlap[2].execution.executionId) + ");") == 2);

        std::vector<ExecutedProposal> directRace{
            CreateExecution(runtime, 4.75, 15)};
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 9, directRace);
            setup.commit();
        }
        RunResult campaignRace;
        RecommendationConversionActivationResult directRaceResult;
        std::exception_ptr campaignRaceError;
        std::exception_ptr directRaceError;
        std::barrier directStart{3};
        std::thread campaignRaceThread{[&] {
            try
            {
                directStart.arrive_and_wait();
                campaignRace = Run(runtimeConnectionString, 9);
            }
            catch (...)
            {
                campaignRaceError = std::current_exception();
            }
        }};
        std::thread directRaceThread{[&] {
            try
            {
                pqxx::connection direct{runtimeConnectionString};
                directStart.arrive_and_wait();
                directRaceResult = ActivateRecommendationConversionExecution(
                    direct, directRace[0].execution.executionId);
            }
            catch (...)
            {
                directRaceError = std::current_exception();
            }
        }};
        directStart.arrive_and_wait();
        campaignRaceThread.join();
        directRaceThread.join();
        if (campaignRaceError) std::rethrow_exception(campaignRaceError);
        if (directRaceError) std::rethrow_exception(directRaceError);
        assert(directRaceResult.activation);
        assert(campaignRace.plan.state ==
                   RecommendationCampaignActivationPlanState::ready ||
               campaignRace.plan.state ==
                   RecommendationCampaignActivationPlanState::alreadySatisfied);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment_recommendation_conversion_activation "
            "WHERE recommendation_conversion_execution_id=" +
            std::to_string(directRace[0].execution.executionId) + ";") == 1);

        assert(Text(owner, schema,
            "SELECT row_to_json(e)::text FROM experiment e WHERE experiment_id=17;") ==
            sourceBefore);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment WHERE worker_pid IS NOT NULL OR "
            "current_operation IS NOT NULL OR current_epoch IS NOT NULL OR "
            "worker_started_at IS NOT NULL;") == 0);
    }
    catch (...)
    {
        pqxx::work cleanup{owner};
        cleanup.exec(
            "DROP SCHEMA IF EXISTS " + cleanup.quote_name(schema) + " CASCADE;");
        cleanup.commit();
        throw;
    }

    pqxx::work cleanup{owner};
    cleanup.exec("DROP SCHEMA " + cleanup.quote_name(schema) + " CASCADE;");
    cleanup.commit();
    std::cout <<
        "Experiment recommendation campaign activation repository tests passed\n";
    return 0;
}
