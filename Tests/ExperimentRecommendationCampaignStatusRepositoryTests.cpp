#include "../Sources/ExperimentRecommendationCampaignStatusRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignStatusService.hpp"
#include "../Sources/ExperimentRecommendationCampaignPlanning.hpp"
#include "../Sources/ExperimentRecommendationCampaignReview.hpp"
#include "../Sources/ExperimentRecommendationCampaignApproval.hpp"
#include "../Sources/ExperimentRecommendationCampaignMaterialization.hpp"
#include "../Sources/ExperimentRecommendationConversionRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionProposalReviewRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionExecutionRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionActivationRepository.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>

#include <pqxx/pqxx>

using namespace EA::ExperimentRecommendation;

namespace
{

void Require(bool condition, const char* expression, int line)
{
    if (!condition)
        throw std::runtime_error(
            "assertion_failed,line=" + std::to_string(line) +
            ",expression=" + expression);
}

#undef assert
#define assert(expression) Require(static_cast<bool>(expression), #expression, __LINE__)

bool IsApprovedDatabase(const std::string& database)
{
    constexpr std::string_view prefixes[]{
        "expertadvisor_phase5_step4_test_",
        "expertadvisor_phase5_step4_final_"};
    for (const auto prefix : prefixes)
    {
        if (!database.starts_with(prefix) || database.size() == prefix.size())
            continue;
        return std::all_of(
            database.begin() + static_cast<std::ptrdiff_t>(prefix.size()),
            database.end(), [](unsigned char value)
            {
                return std::islower(value) || std::isdigit(value) || value == '_';
            });
    }
    return false;
}

std::optional<std::string> RequestedDatabaseDiagnostic(const char* database)
{
    if (!database || *database == '\0') return "LSTM_TEST_DB_NAME_required";
    if (!IsApprovedDatabase(database)) return "isolated_test_database_required";
    return std::nullopt;
}

std::optional<std::string> ConnectedDatabaseDiagnostic(
    const std::string& requested,
    const std::string& connected)
{
    if (requested != connected) return "connected_database_name_mismatch";
    if (!IsApprovedDatabase(connected)) return "isolated_test_database_required";
    return std::nullopt;
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
    return {std::istreambuf_iterator<char>{input}, {}};
}

void SetSearchPath(pqxx::transaction_base& transaction, const std::string& schema)
{
    transaction.exec(
        "SET LOCAL search_path TO " + transaction.quote_name(schema) + ";");
}

bool PermissionDenied(const pqxx::sql_error& error)
{
    return error.sqlstate() == "42501" ||
        std::string{error.what()}.find("permission denied") != std::string::npos;
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
        "status-review-authorization-" + std::to_string(variant);
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
        "status-evaluation-" + std::to_string(variant);
    request.evaluation.evaluationIdentityHash = RecommendationCanonicalHash(
        request.evaluation.evaluationIdentityCanonical);
    request.evaluation.evaluationPolicyCanonical = "status-evaluation-policy";
    request.evaluation.evaluationPolicyHash = RecommendationCanonicalHash(
        request.evaluation.evaluationPolicyCanonical);
    request.score.state = RecommendationConversionEvidenceState::completed;
    request.score.valid = true;
    request.score.recommendationId = 42;
    request.score.finalScore = 0.75;
    request.score.scoringPolicyCanonical = "status-scoring-policy";
    request.score.scoringPolicyHash = RecommendationCanonicalHash(
        request.score.scoringPolicyCanonical);
    request.evaluation.scoringPolicyHash = request.score.scoringPolicyHash;
    auto invocation = request.sourceInvocation;
    invocation.configuration.coreLrMult = proposed;
    const auto semantic = BuildRecommendationCandidateIdentity(
        invocation.configuration);
    const auto identity = BuildRecommendationInvocationIdentity(invocation);
    request.recommendationSemanticCanonical = semantic.canonicalText;
    request.recommendationSemanticHash = semantic.hash;
    request.recommendationInvocationCanonical = identity.canonicalText;
    request.recommendationInvocationHash = identity.hash;
    const auto result = BuildProposedExperimentSpecification(request);
    assert(result.eligibility.eligible && result.proposal);
    return *result.proposal;
}

struct CampaignProposal
{
    PersistedRecommendationConversionProposal proposal;
    int variant = 0;
};

CampaignProposal CreateProposal(pqxx::connection& runtime, double value, int variant)
{
    return {PersistRecommendationConversionProposal(
                runtime, Proposal(value, variant)).persisted,
            variant};
}

void Approve(pqxx::connection& runtime, const CampaignProposal& proposal)
{
    const auto result = RecordRecommendationConversionProposalReviewDecision(
        runtime,
        {proposal.proposal.proposalId,
         RecommendationConversionProposalReviewDecision::approve,
         "phase5-status-approve-" + std::to_string(proposal.variant),
         "phase5 status test operator", "Explicit isolated status fixture."});
    assert(result.decision);
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
    const std::vector<CampaignProposal>& values)
{
    MaterializationFixture fixture;
    fixture.approvalId = 1000 + materializationId;
    fixture.rankingSnapshotId = 2000 + materializationId;
    RecommendationCampaignPlanInput input;
    input.policy.enabled = true;
    input.policy.maximumSelectedRecommendations = static_cast<int>(values.size());
    input.policy.maximumCandidatesConsidered = static_cast<int>(values.size());
    input.policy.maximumPerSourceExperiment.reset();
    input.scope.rankingSnapshotId = fixture.rankingSnapshotId;
    input.rankingSnapshotIdentityCanonical =
        "phase5-step4-ranking-snapshot-" + std::to_string(materializationId);
    input.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        input.rankingSnapshotIdentityCanonical);
    input.generatedAt = "display-only";
    const std::string rankingPolicy = "phase5-step4-ranking-policy-v1";
    const std::string rankingPolicyHash = RecommendationCanonicalHash(rankingPolicy);
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        const auto& proposal = values[index].proposal.proposal;
        RecommendationCampaignCandidateInput candidate;
        candidate.rankingSnapshotId = fixture.rankingSnapshotId;
        candidate.rankingSnapshotIdentityCanonical =
            input.rankingSnapshotIdentityCanonical;
        candidate.rankingSnapshotIdentityHash = input.rankingSnapshotIdentityHash;
        candidate.rankingPolicyCanonical = rankingPolicy;
        candidate.rankingPolicyHash = rankingPolicyHash;
        candidate.rankingVersion = 1;
        candidate.rankingMemberId = 300000 + materializationId * 100 +
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
        candidate.targetEpochs = proposal.proposedInvocation.configuration.targetEpochs;
        candidate.leaderScore = 0.9;
        candidate.inferenceAccuracy = 0.8;
        candidate.predictedNeutralProportion = 0.2;
        candidate.recommendationSemanticCanonical =
            "phase5-step4-semantic-" + std::to_string(materializationId) +
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
    const auto review = ReviewRecommendationCampaignPlan(fixture.plan);
    fixture.approval = BuildRecommendationCampaignApprovalEvidence(
        fixture.rankingSnapshotId, input.rankingSnapshotIdentityCanonical,
        input.rankingSnapshotIdentityHash, fixture.plan, review,
        {RecommendationCampaignApprovalDecision::approved, review.identityHash,
         "phase5 step4 reviewer", "Approve exact status campaign."});
    std::vector<ProposedExperimentSpecification> proposals;
    for (const auto& value : values) proposals.push_back(value.proposal.proposal);
    fixture.materialization = BuildRecommendationCampaignMaterializationEvidence(
        {fixture.approvalId, "phase5 step4 materializer",
         "Materialize exact status campaign."},
        fixture.approval, fixture.plan, proposals);
    return fixture;
}

void InsertMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId,
    const std::vector<CampaignProposal>& values)
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
            "recommendation_ranking_member_id,recommendation_ranking_snapshot_id,"
            "recommendation_id,source_experiment_id,global_ordinal) "
            "VALUES($1,$2,$3,$4,$5);",
            pqxx::params{member.rankingMemberId, fixture.rankingSnapshotId,
                         member.recommendationId, member.sourceExperimentId,
                         member.rankingPosition});
    transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_approval("
        "recommendation_campaign_approval_id,approval_contract_version,"
        "recommendation_ranking_snapshot_id,ranking_snapshot_identity_canonical,"
        "ranking_snapshot_identity_hash,planning_policy_canonical,planning_policy_hash,"
        "planning_scope_canonical,campaign_plan_identity_canonical,"
        "campaign_plan_identity_hash,review_contract_version,"
        "campaign_review_identity_canonical,campaign_review_identity_hash,"
        "campaign_review_hash_collision_ordinal,candidate_count,selected_count,"
        "excluded_count,duplicate_group_count,duplicate_candidate_count,"
        "considered_family_count,selected_family_count,considered_symbol_count,"
        "selected_symbol_count,considered_horizon_count,selected_horizon_count,"
        "deterministic_ordering_verified,decision,reviewer_identity,reason_text,"
        "approval_identity_canonical,approval_identity_hash) "
        "VALUES($1,1,$2,$3,$4,$5,$6,$7,$8,$9,1,$10,$11,0,$12,$13,$14,$15,"
        "$16,$17,$18,$19,$20,$21,$22,true,'approved',$23,$24,$25,$26);",
        pqxx::params{fixture.approvalId, approval.rankingSnapshotId,
            approval.rankingSnapshotIdentityCanonical,
            approval.rankingSnapshotIdentityHash, approval.planningPolicyCanonical,
            approval.planningPolicyHash, approval.planningScopeCanonical,
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
        "recommendation_campaign_materialization_id,recommendation_campaign_approval_id,"
        "materialization_contract_version,approval_identity_canonical,"
        "approval_identity_hash,recommendation_ranking_snapshot_id,"
        "ranking_snapshot_identity_canonical,ranking_snapshot_identity_hash,"
        "planning_policy_canonical,planning_policy_hash,planning_scope_canonical,"
        "campaign_plan_identity_canonical,campaign_plan_identity_hash,"
        "campaign_review_identity_canonical,campaign_review_identity_hash,"
        "approval_decision,approval_reviewer_identity,approval_reason_text,"
        "materialized_by,materialization_reason_text,selected_member_count,"
        "initially_created_proposal_count,initially_reused_proposal_count,"
        "materialization_identity_canonical,materialization_identity_hash) "
        "VALUES($1,$2,1,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,'approved',"
        "$15,$16,$17,$18,$19,0,$19,$20,$21);",
        pqxx::params{materializationId, fixture.approvalId,
            approval.approvalIdentityCanonical, approval.approvalIdentityHash,
            approval.rankingSnapshotId,
            approval.rankingSnapshotIdentityCanonical,
            approval.rankingSnapshotIdentityHash, approval.planningPolicyCanonical,
            approval.planningPolicyHash, approval.planningScopeCanonical,
            approval.campaignPlanIdentityCanonical,
            approval.campaignPlanIdentityHash,
            approval.campaignReviewIdentityCanonical,
            approval.campaignReviewIdentityHash, approval.reviewerIdentity,
            approval.reasonText, evidence.operatorIdentity, evidence.reasonText,
            evidence.selectedMemberCount, evidence.materializationIdentityCanonical,
            evidence.materializationIdentityHash});
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        const auto& proposal = values[index].proposal;
        const auto& member = evidence.members[index];
        transaction.exec(
            "INSERT INTO experiment_recommendation_campaign_materialization_member("
            "recommendation_campaign_materialization_id,member_ordinal,"
            "recommendation_ranking_member_id,recommendation_id,source_experiment_id,"
            "ranking_position,selected_member_identity_canonical,"
            "selected_member_identity_hash,recommendation_conversion_proposal_id,"
            "proposal_identity_canonical,proposal_identity_hash) "
            "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11);",
            pqxx::params{materializationId, member.memberOrdinal,
                member.rankingMemberId, proposal.proposal.recommendationId,
                proposal.proposal.sourceExperimentId, member.rankingPosition,
                member.selectedMemberIdentityCanonical,
                member.selectedMemberIdentityHash, proposal.proposalId,
                proposal.proposal.conversionIdentityCanonical,
                proposal.proposal.conversionIdentityHash});
    }
}

std::string SequenceState(
    pqxx::connection& owner,
    const std::string& schema,
    const std::string& table,
    const std::string& column)
{
    pqxx::read_transaction transaction{owner};
    SetSearchPath(transaction, schema);
    const std::string sequence = transaction.exec(
        "SELECT pg_get_serial_sequence($1,$2);", pqxx::params{table, column})
        .one_row()[0].as<std::string>();
    return transaction.exec(
        "SELECT last_value::text||':'||is_called::text FROM " + sequence + ";")
        .one_row()[0].as<std::string>();
}

int RunReleaseStatusCli(
    const std::string& database,
    const std::string& host,
    const std::string& port,
    const std::string& schema,
    long long materializationId,
    bool separatedForm)
{
    const char* binary = std::getenv("LSTM_STEP4_RELEASE_BINARY");
    if (!binary || *binary == '\0')
        throw std::runtime_error("LSTM_STEP4_RELEASE_BINARY_required");
    if (port != "5432")
        throw std::runtime_error("phase5_step4_release_cli_requires_default_port");
    const std::string option = "--recommendation-campaign-status=" +
        std::to_string(materializationId);
    const std::string pgOptions = "-c search_path=" + schema +
        " -c lock_timeout=5s -c statement_timeout=15s";
    const pid_t child = fork();
    if (child < 0) throw std::runtime_error("phase5_step4_cli_fork_failed");
    if (child == 0)
    {
        if (setenv("LSTM_DB_NAME", database.c_str(), 1) != 0 ||
            setenv("LSTM_DB_HOST", host.c_str(), 1) != 0 ||
            setenv("PGPORT", port.c_str(), 1) != 0 ||
            setenv("PGCONNECT_TIMEOUT", "5", 1) != 0 ||
            setenv("PGOPTIONS", pgOptions.c_str(), 1) != 0)
            _exit(126);
        if (separatedForm)
        {
            const std::string id = std::to_string(materializationId);
            execl(binary, binary, "--recommendation-campaign-status", id.c_str(),
                  static_cast<char*>(nullptr));
        }
        else
            execl(binary, binary, option.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    int status = 0;
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds{30};
    for (;;)
    {
        const pid_t waited = waitpid(child, &status, WNOHANG);
        if (waited == child) break;
        if (waited < 0 && errno != EINTR)
        {
            (void)kill(child, SIGKILL);
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
            throw std::runtime_error("phase5_step4_cli_wait_failed");
        }
        if (std::chrono::steady_clock::now() >= deadline)
        {
            (void)kill(child, SIGKILL);
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
            throw std::runtime_error("phase5_step4_cli_timeout");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }
    if (!WIFEXITED(status)) return 128;
    return WEXITSTATUS(status);
}

} // namespace

int main()
{
    assert(RequestedDatabaseDiagnostic(nullptr) ==
           std::optional<std::string>{"LSTM_TEST_DB_NAME_required"});
    for (const char* unsafe : {"", "LSTM", "lstm", "Lstm", "postgres",
                               "production", "unrelated"})
        assert(RequestedDatabaseDiagnostic(unsafe));
    assert(!RequestedDatabaseDiagnostic(
        "expertadvisor_phase5_step4_test_20260719_01"));
    assert(ConnectedDatabaseDiagnostic(
        "expertadvisor_phase5_step4_test_requested",
        "expertadvisor_phase5_step4_test_connected") ==
        std::optional<std::string>{"connected_database_name_mismatch"});

    const char* configured = std::getenv("LSTM_TEST_DB_NAME");
    if (const auto diagnostic = RequestedDatabaseDiagnostic(configured))
    {
        std::cerr << *diagnostic << '\n';
        return 2;
    }
    const std::string database = configured;
    const char* configuredHost = std::getenv("LSTM_TEST_DB_HOST");
    if (!configuredHost || *configuredHost == '\0')
    {
        std::cerr << "LSTM_TEST_DB_HOST_required\n";
        return 2;
    }
    const char* configuredPort = std::getenv("PGPORT");
    if (!configuredPort || *configuredPort == '\0')
    {
        std::cerr << "PGPORT_required\n";
        return 2;
    }
    const std::string host = configuredHost;
    const std::string port = configuredPort;
    const std::string ownerUser = EnvironmentOr("LSTM_TEST_DB_OWNER", "vjp");
    const std::string schema = "phase5_campaign_status_" + std::to_string(getpid());
    const std::string ownerString = "host=" + host + " port=" + port +
        " user=" + ownerUser + " dbname=" + database + " connect_timeout=5";
    const std::string runtimeString = "host=" + host + " port=" + port +
        " user=pqxx dbname=" + database + " connect_timeout=5 options='-c "
        "search_path=" + schema + " -c statement_timeout=15s'";
    pqxx::connection owner{ownerString};
    {
        pqxx::read_transaction identity{owner};
        const std::string connected = identity.exec(
            "SELECT current_database();").one_row()[0].as<std::string>();
        if (const auto diagnostic = ConnectedDatabaseDiagnostic(database, connected))
        {
            std::cerr << *diagnostic << '\n';
            return 2;
        }
    }

    std::string stage = "setup";
    try
    {
        {
            pqxx::work setup{owner};
            setup.exec("CREATE SCHEMA " + setup.quote_name(schema) + ";");
            SetSearchPath(setup, schema);
            setup.exec("CREATE TABLE model(model_id bigint PRIMARY KEY,name text);");
            for (const char* migration : {
                     "Database/migrations/004_inference_eval_result.sql",
                     "Database/migrations/005_experiment_scheduler.sql",
                     "Database/migrations/006_experiment_analysis.sql",
                     "Database/migrations/008_experiment_paused_status.sql",
                     "Database/migrations/009_experiment_run_metadata.sql",
                     "Database/migrations/011_experiment_live_progress.sql",
                     "Database/migrations/012_model_experiment_id.sql",
                     "Database/migrations/014_checkpoint_stop_and_eval.sql",
                     "Database/migrations/015_checkpoint_infer_phase1.sql",
                     "Database/migrations/016_checkpoint_analysis_scope.sql",
                     "Database/migrations/017_checkpoint_policy.sql",
                     "Database/migrations/018_checkpoint_inference_result_scope.sql",
                     "Database/migrations/019_checkpoint_policy_hardening.sql"})
                setup.exec(ReadFile(migration));
            setup.exec("ALTER TABLE experiment ADD COLUMN worker_started_at timestamptz;");
            setup.exec(R"SQL(
CREATE TABLE experiment_recommendation(
 recommendation_id bigint PRIMARY KEY,
 source_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
 status text NOT NULL);
CREATE TABLE experiment_recommendation_ranking_snapshot(
 recommendation_ranking_snapshot_id bigint PRIMARY KEY);
CREATE TABLE experiment_recommendation_ranking_member(
 recommendation_ranking_member_id bigint PRIMARY KEY,
 recommendation_ranking_snapshot_id bigint NOT NULL REFERENCES
  experiment_recommendation_ranking_snapshot(recommendation_ranking_snapshot_id),
 recommendation_id bigint NOT NULL REFERENCES experiment_recommendation(recommendation_id),
 source_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
 global_ordinal integer NOT NULL CHECK(global_ordinal > 0));
INSERT INTO experiment(experiment_id,symbol,prediction_horizon,c_next_threshold,
 core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,train_start,train_end,
 infer_start,infer_end,resume_model_id,status,phase,invocation_mode,updated_at)
VALUES(17,'eurusd',12,0.001,1,5,120,15,
 '2010-01-01 America/Chicago','2025-01-01 America/Chicago',
 '2025-01-01 America/Chicago','2026-01-01 America/Chicago',77,
 'paused','train','source','2026-07-19 12:34:56+00');
INSERT INTO model(model_id,name,experiment_id) VALUES(77,'source',17);
INSERT INTO experiment_recommendation VALUES(42,17,'approved');
SELECT setval(pg_get_serial_sequence('experiment','experiment_id'),17,true);
)SQL");
            for (const char* migration : {
                     "Database/migrations/036_experiment_recommendation_conversion_proposal.sql",
                     "Database/migrations/037_experiment_recommendation_conversion_review.sql",
                     "Database/migrations/038_experiment_recommendation_conversion_execution.sql",
                     "Database/migrations/039_experiment_recommendation_conversion_activation.sql",
                     "Database/migrations/040_experiment_recommendation_campaign_approval.sql",
                     "Database/migrations/041_experiment_recommendation_campaign_materialization.sql"})
            {
                setup.exec(ReadFile(migration));
                setup.exec(ReadFile(migration));
            }
            setup.exec(ReadFile(
                "Tests/ExperimentRecommendationCampaignMaterializationMigrationTests.sql"));
            setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                " TO pqxx; GRANT SELECT ON experiment_recommendation,"
                "experiment_recommendation_ranking_snapshot,"
                "experiment_recommendation_ranking_member TO pqxx;");
            setup.commit();
        }

        pqxx::connection runtime{runtimeString};
        const CampaignProposal first = CreateProposal(runtime, 1.11, 1);
        const CampaignProposal unrelated = CreateProposal(runtime, 1.99, 99);
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 1, {first});
            fixture.commit();
        }

        stage = "proposal_only";
        auto snapshot = ReadRecommendationCampaignStatusSnapshot(runtime, {1});
        assert(snapshot.proposalOnlyCount == 1);
        assert(snapshot.members[0].proposalId == first.proposal.proposalId);
        assert(snapshot.members[0].proposalId != unrelated.proposal.proposalId);

        stage = "executed_paused";
        Approve(runtime, first);
        const auto executionResult = ExecuteApprovedRecommendationConversionProposal(
            runtime, first.proposal.proposalId);
        assert(executionResult.execution);
        snapshot = ReadRecommendationCampaignStatusSnapshot(runtime, {1});
        assert(snapshot.pausedCount == 1 && snapshot.executedCount == 1);

        stage = "activated_pending";
        const auto activationResult = ActivateRecommendationConversionExecution(
            runtime, executionResult.execution->executionId);
        assert(activationResult.activation);
        snapshot = ReadRecommendationCampaignStatusSnapshot(runtime, {1});
        assert(snapshot.pendingCount == 1 && snapshot.activatedCount == 1);

        const std::string experimentSequenceBefore = SequenceState(
            owner, schema, "experiment", "experiment_id");
        const std::string executionSequenceBefore = SequenceState(
            owner, schema, "experiment_recommendation_conversion_execution",
            "recommendation_conversion_execution_id");
        const std::string activationSequenceBefore = SequenceState(
            owner, schema, "experiment_recommendation_conversion_activation",
            "recommendation_conversion_activation_id");
        std::ostringstream output;
        std::ostringstream errors;
        assert(RunRecommendationCampaignStatusCommand(
            runtimeString, {1}, output, errors) == 0);
        assert(errors.str().empty());
        assert(output.str().find("RECOMMENDATION_CAMPAIGN_STATUS") == 0);
        assert(output.str().find("transaction_read_only=true") != std::string::npos);
        assert(output.str().find("scheduler_polled=false") != std::string::npos);
        assert(RunReleaseStatusCli(database, host, port, schema, 1, false) == 0);
        assert(RunReleaseStatusCli(database, host, port, schema, 1, true) == 0);
        assert(SequenceState(owner, schema, "experiment", "experiment_id") ==
               experimentSequenceBefore);
        assert(SequenceState(owner, schema,
            "experiment_recommendation_conversion_execution",
            "recommendation_conversion_execution_id") == executionSequenceBefore);
        assert(SequenceState(owner, schema,
            "experiment_recommendation_conversion_activation",
            "recommendation_conversion_activation_id") == activationSequenceBefore);

        stage = "running_train";
        {
            pqxx::work mutate{owner};
            SetSearchPath(mutate, schema);
            mutate.exec(
                "UPDATE experiment SET status='running',phase='train',worker_pid=4321,"
                "current_operation='train',worker_started_at=now(),started_at=now(),"
                "current_epoch=20 WHERE experiment_id=$1;",
                pqxx::params{executionResult.execution->experimentId});
            mutate.commit();
        }
        snapshot = ReadRecommendationCampaignStatusSnapshot(runtime, {1});
        assert(snapshot.runningTrainCount == 1 && snapshot.activeWorkerCount == 1);

        stage = "snapshot_consistency";
        pqxx::read_transaction stable{runtime};
        stable.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
        const auto before = LoadRecommendationCampaignStatusSnapshot(stable, {1});
        {
            pqxx::work mutate{owner};
            SetSearchPath(mutate, schema);
            mutate.exec("UPDATE experiment SET current_epoch=21 WHERE experiment_id=$1;",
                pqxx::params{executionResult.execution->experimentId});
            mutate.commit();
        }
        const auto withinSameSnapshot = LoadRecommendationCampaignStatusSnapshot(stable, {1});
        assert(before.snapshotIdentityHash == withinSameSnapshot.snapshotIdentityHash);
        stable.abort();
        const auto after = ReadRecommendationCampaignStatusSnapshot(runtime, {1});
        assert(before.snapshotIdentityHash != after.snapshotIdentityHash);

        stage = "review_reversal";
        (void)RecordRecommendationConversionProposalReviewDecision(
            runtime,
            {first.proposal.proposalId,
             RecommendationConversionProposalReviewDecision::reject,
             "phase5-status-reverse", "phase5 status test operator",
             "Reverse current disposition after execution."});
        snapshot = ReadRecommendationCampaignStatusSnapshot(runtime, {1});
        assert(snapshot.members[0].currentReviewDecision ==
               std::optional<std::string>{"reject"});
        assert(snapshot.members[0].authorizationReviewDecisionId ==
               std::optional<long long>{executionResult.execution->reviewDecisionId});

        stage = "completed";
        {
            pqxx::work mutate{owner};
            SetSearchPath(mutate, schema);
            const long long experimentId = executionResult.execution->experimentId;
            mutate.exec("INSERT INTO model(model_id,name,experiment_id) VALUES(1001,'final',$1);",
                pqxx::params{experimentId});
            mutate.exec(R"SQL(
INSERT INTO inference_eval_result(model_id,symbol,prediction_horizon,
 threshold_logret,window_size,label_rule_id,target_type,from_date,to_date,
 status,inference_scope) VALUES(1001,'eurusd',12,0.001,128,1,1,
 '2025-01-01','2026-01-01','completed','final');
)SQL");
            mutate.exec(
                "INSERT INTO experiment_analysis_result(experiment_id,model_id,"
                "analysis_status,analysis_scope) VALUES($1,1001,'completed','final');",
                pqxx::params{experimentId});
            mutate.exec(
                "UPDATE experiment SET status='completed',phase='done',last_model_id=1001,"
                "worker_pid=NULL,current_operation='analyze',current_epoch=120,"
                "exit_code=0,error_message=NULL,completed_at=now() WHERE experiment_id=$1;",
                pqxx::params{experimentId});
            mutate.commit();
        }
        snapshot = ReadRecommendationCampaignStatusSnapshot(runtime, {1});
        assert(snapshot.completedCount == 1);
        assert(snapshot.members[0].trainComplete && snapshot.members[0].inferComplete &&
               snapshot.members[0].analyzeComplete);
        assert(snapshot.members[0].failedFinalAnalysisResultCount == 0);

        const std::string completedIdentity = snapshot.snapshotIdentityHash;
        stage = "unrelated_result_evidence";
        {
            pqxx::work mutate{owner};
            SetSearchPath(mutate, schema);
            mutate.exec(R"SQL(
INSERT INTO inference_eval_result(model_id,symbol,prediction_horizon,
 threshold_logret,window_size,label_rule_id,target_type,from_date,to_date,
 status,inference_scope) VALUES(1001,'gbpusd',12,0.001,128,1,1,
 '2025-01-01','2026-01-01','completed','final');
)SQL");
            mutate.commit();
        }
        assert(ReadRecommendationCampaignStatusSnapshot(runtime, {1}).snapshotIdentityHash ==
               completedIdentity);
        {
            pqxx::work mutate{owner};
            SetSearchPath(mutate, schema);
            mutate.exec("DELETE FROM inference_eval_result WHERE model_id=1001 "
                        "AND symbol='gbpusd';");
            mutate.commit();
        }

        stage = "wrong_exact_inference_provenance";
        {
            pqxx::work mutate{owner};
            SetSearchPath(mutate, schema);
            mutate.exec("DELETE FROM inference_eval_result WHERE model_id=1001 "
                        "AND symbol='eurusd';");
            mutate.exec(R"SQL(
INSERT INTO inference_eval_result(model_id,symbol,prediction_horizon,
 threshold_logret,window_size,label_rule_id,target_type,from_date,to_date,
 status,inference_scope) VALUES(1001,'eurusd',12,0.002,128,1,1,
 '2025-01-01','2026-01-01','completed','final');
)SQL");
            mutate.commit();
        }
        snapshot = ReadRecommendationCampaignStatusSnapshot(runtime, {1});
        assert(snapshot.inconsistentCount == 1);
        assert(snapshot.members[0].completedFinalInferenceResultCount == 0);
        {
            pqxx::work restore{owner};
            SetSearchPath(restore, schema);
            restore.exec("DELETE FROM inference_eval_result WHERE model_id=1001;");
            restore.exec(R"SQL(
INSERT INTO inference_eval_result(model_id,symbol,prediction_horizon,
 threshold_logret,window_size,label_rule_id,target_type,from_date,to_date,
 status,inference_scope) VALUES(1001,'eurusd',12,0.001,128,1,1,
 '2025-01-01','2026-01-01','completed','final');
)SQL");
            restore.commit();
        }

        stage = "historical_analysis_ignored";
        {
            pqxx::work mutate{owner};
            SetSearchPath(mutate, schema);
            mutate.exec("INSERT INTO model(model_id,name,experiment_id) "
                        "VALUES(1002,'historical',$1);",
                pqxx::params{executionResult.execution->experimentId});
            mutate.exec("INSERT INTO experiment_analysis_result(experiment_id,model_id,"
                        "analysis_status,analysis_scope) "
                        "VALUES($1,1002,'completed','final');",
                pqxx::params{executionResult.execution->experimentId});
            mutate.commit();
        }
        assert(ReadRecommendationCampaignStatusSnapshot(runtime, {1}).snapshotIdentityHash ==
               completedIdentity);

        stage = "member_inconsistent";
        {
            pqxx::work mutate{owner};
            SetSearchPath(mutate, schema);
            mutate.exec("UPDATE experiment SET status='pending',phase='train',"
                        "worker_pid=9999 WHERE experiment_id=$1;",
                pqxx::params{executionResult.execution->experimentId});
            mutate.commit();
        }
        snapshot = ReadRecommendationCampaignStatusSnapshot(runtime, {1});
        assert(snapshot.inconsistentCount == 1);
        assert(snapshot.members.size() == 1);

        stage = "read_only_enforced";
        bool readOnlyDenied = false;
        try
        {
            pqxx::read_transaction readOnly{runtime};
            readOnly.exec("UPDATE experiment SET status='failed' WHERE experiment_id=" +
                          std::to_string(executionResult.execution->experimentId));
        }
        catch (const pqxx::sql_error& error)
        {
            readOnlyDenied = error.sqlstate() == "25006";
        }
        assert(readOnlyDenied);

        bool materializationUpdateDenied = false;
        try
        {
            pqxx::work forbidden{runtime};
            forbidden.exec(
                "UPDATE experiment_recommendation_campaign_materialization "
                "SET materialized_by='forged';");
        }
        catch (const pqxx::sql_error& error)
        {
            materializationUpdateDenied = PermissionDenied(error);
        }
        assert(materializationUpdateDenied);

        {
            pqxx::read_transaction verify{owner};
            SetSearchPath(verify, schema);
            assert(verify.exec(
                "SELECT count(*) FROM information_schema.tables WHERE "
                "table_schema=current_schema() AND table_name LIKE "
                "'experiment_recommendation_campaign_status%';")
                .one_row()[0].as<int>() == 0);
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "phase5_step4_repository_test_failed,stage=" << stage
                  << ",error=" << error.what() << '\n';
        pqxx::work cleanup{owner};
        cleanup.exec("DROP SCHEMA IF EXISTS " + cleanup.quote_name(schema) +
                     " CASCADE;");
        cleanup.commit();
        return 1;
    }

    pqxx::work cleanup{owner};
    cleanup.exec("DROP SCHEMA " + cleanup.quote_name(schema) + " CASCADE;");
    cleanup.commit();
    return 0;
}
