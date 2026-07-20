#include "../Sources/ExperimentRecommendationCampaignLaunchRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignLaunchService.hpp"
#include "../Sources/ExperimentRecommendationCampaignExecutionRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignActivationRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionExecutionRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionProposalReviewRepository.hpp"

#include <algorithm>
#include <barrier>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <exception>
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

bool IsApprovedDatabase(const std::string& database)
{
    constexpr std::string_view prefixes[]{
        "expertadvisor_phase5_step3_test_",
        "expertadvisor_phase5_step3_final_"};
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
        (error.sqlstate().empty() &&
         std::string{error.what()}.find("permission denied") !=
             std::string::npos);
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
        "launch-review-authorization-" + std::to_string(variant);
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
        "launch-evaluation-" + std::to_string(variant);
    request.evaluation.evaluationIdentityHash = RecommendationCanonicalHash(
        request.evaluation.evaluationIdentityCanonical);
    request.evaluation.evaluationPolicyCanonical = "launch-evaluation-policy";
    request.evaluation.evaluationPolicyHash = RecommendationCanonicalHash(
        request.evaluation.evaluationPolicyCanonical);
    request.score.state = RecommendationConversionEvidenceState::completed;
    request.score.valid = true;
    request.score.recommendationId = 42;
    request.score.finalScore = 0.75;
    request.score.scoringPolicyCanonical = "launch-scoring-policy";
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
    return {proposalId, decision, requestId, "phase5 step3 test operator",
            "Explicit isolated Phase 5 launch test."};
}

struct CampaignProposal
{
    PersistedRecommendationConversionProposal proposal;
    int variant = 0;
};

CampaignProposal CreateApprovedProposal(
    pqxx::connection& runtime,
    double proposed,
    int variant)
{
    CampaignProposal result;
    result.variant = variant;
    result.proposal = PersistRecommendationConversionProposal(
        runtime, Proposal(proposed, variant)).persisted;
    const auto review = RecordRecommendationConversionProposalReviewDecision(
        runtime, Review(result.proposal.proposalId,
                        "phase5-launch-approve-" + std::to_string(variant)));
    assert(review.decision);
    return result;
}

PersistedRecommendationConversionExecution Execute(
    pqxx::connection& runtime,
    const CampaignProposal& proposal)
{
    const auto result = ExecuteApprovedRecommendationConversionProposal(
        runtime, proposal.proposal.proposalId);
    assert(result.execution);
    return *result.execution;
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
    input.policy.maximumSelectedRecommendations =
        static_cast<int>(values.size());
    input.policy.maximumCandidatesConsidered =
        static_cast<int>(values.size());
    input.policy.maximumPerSourceExperiment.reset();
    input.scope.rankingSnapshotId = fixture.rankingSnapshotId;
    input.rankingSnapshotIdentityCanonical =
        "phase5-step3-ranking-snapshot-" +
        std::to_string(materializationId);
    input.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        input.rankingSnapshotIdentityCanonical);
    input.generatedAt = "display-only";
    const std::string rankingPolicy = "phase5-step3-ranking-policy-v1";
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
            "phase5-step3-semantic-" + std::to_string(materializationId) +
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
    const auto campaignReview = ReviewRecommendationCampaignPlan(fixture.plan);
    fixture.approval = BuildRecommendationCampaignApprovalEvidence(
        fixture.rankingSnapshotId,
        input.rankingSnapshotIdentityCanonical,
        input.rankingSnapshotIdentityHash,
        fixture.plan,
        campaignReview,
        {RecommendationCampaignApprovalDecision::approved,
         campaignReview.identityHash, "phase5 step3 reviewer",
         "Approve exact launch campaign."});
    std::vector<ProposedExperimentSpecification> proposals;
    for (const auto& value : values) proposals.push_back(value.proposal.proposal);
    fixture.materialization = BuildRecommendationCampaignMaterializationEvidence(
        {fixture.approvalId, "phase5 step3 materializer",
         "Materialize exact launch campaign."},
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
                materializationId, member.memberOrdinal, member.rankingMemberId,
                proposal.proposal.recommendationId,
                proposal.proposal.sourceExperimentId, member.rankingPosition,
                member.selectedMemberIdentityCanonical,
                member.selectedMemberIdentityHash, proposal.proposalId,
                proposal.proposal.conversionIdentityCanonical,
                proposal.proposal.conversionIdentityHash});
    }
}

RecommendationCampaignLaunchPersistResult RunLaunch(
    const std::string& connectionString,
    long long materializationId)
{
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};
    transaction.exec("SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='15s';");
    const auto materialization = LoadRecommendationCampaignLaunchMaterialization(
        transaction, materializationId);
    assert(materialization);
    auto result = LaunchRecommendationCampaignInTransaction(
        transaction, {materializationId, false}, *materialization);
    transaction.commit();
    return result;
}

std::vector<PersistedRecommendationConversionExecution> RunStep1(
    const std::string& connectionString,
    long long materializationId)
{
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};
    const auto materialization = LoadRecommendationCampaignExecutionMaterialization(
        transaction, materializationId);
    assert(materialization);
    LockRecommendationCampaignExecutions(transaction, *materialization);
    const auto input = LoadRecommendationCampaignExecutionInput(
        transaction, *materialization);
    const auto plan = BuildRecommendationCampaignExecutionPlan(
        {materializationId, false}, input);
    std::vector<PersistedRecommendationConversionExecution> result;
    if (plan.state == RecommendationCampaignExecutionPlanState::ready)
        result = PersistRecommendationCampaignExecutions(transaction, plan);
    transaction.commit();
    return result;
}

std::vector<PersistedRecommendationConversionActivation> RunStep2(
    const std::string& connectionString,
    long long materializationId)
{
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};
    const auto materialization = LoadRecommendationCampaignActivationMaterialization(
        transaction, materializationId);
    assert(materialization);
    LockRecommendationCampaignActivations(transaction, *materialization);
    const auto input = LoadRecommendationCampaignActivationInput(
        transaction, *materialization);
    const auto plan = BuildRecommendationCampaignActivationPlan(
        {materializationId, false}, input);
    std::vector<PersistedRecommendationConversionActivation> result;
    if (plan.state == RecommendationCampaignActivationPlanState::ready)
        result = PersistRecommendationCampaignActivations(transaction, plan);
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

std::string SequenceState(
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
    return transaction.exec(
        "SELECT last_value::text||':'||is_called::text FROM " + sequence + ";")
        .one_row()[0].as<std::string>();
}

void AssertPending(
    pqxx::connection& runtime,
    pqxx::connection& owner,
    const std::string& schema,
    const CampaignProposal& proposal)
{
    const auto execution = FindRecommendationConversionExecutionByProposal(
        runtime, proposal.proposal.proposalId);
    assert(execution);
    const auto activation = FindRecommendationConversionActivationByExecution(
        runtime, execution->executionId);
    assert(activation);
    assert(Text(owner, schema,
        "SELECT status||'/'||phase FROM experiment WHERE experiment_id=" +
        std::to_string(execution->experimentId) + ";") == "pending/train");
}

int RunReleaseLaunchCli(
    const std::string& database,
    const std::string& host,
    const std::string& port,
    const std::string& schema,
    long long materializationId,
    bool dryRun)
{
    const char* binary = std::getenv("LSTM_STEP3_RELEASE_BINARY");
    if (!binary || *binary == '\0')
        throw std::runtime_error("LSTM_STEP3_RELEASE_BINARY_required");
    const std::string option =
        "--launch-recommendation-campaign-materialization=" +
        std::to_string(materializationId);
    const std::string pgOptions = "-c search_path=" + schema +
        " -c lock_timeout=5s -c statement_timeout=15s";
    const pid_t child = fork();
    if (child < 0) throw std::runtime_error("phase5_step3_cli_fork_failed");
    if (child == 0)
    {
        if (setenv("LSTM_DB_NAME", database.c_str(), 1) != 0 ||
            setenv("LSTM_DB_HOST", host.c_str(), 1) != 0 ||
            setenv("PGPORT", port.c_str(), 1) != 0 ||
            setenv("PGCONNECT_TIMEOUT", "5", 1) != 0 ||
            setenv("PGOPTIONS", pgOptions.c_str(), 1) != 0)
            _exit(126);
        execl(
            binary, binary, option.c_str(), dryRun ? "--dry-run" : "--yes",
            static_cast<char*>(nullptr));
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
            throw std::runtime_error("phase5_step3_cli_wait_failed");
        }
        if (std::chrono::steady_clock::now() >= deadline)
        {
            (void)kill(child, SIGKILL);
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
            throw std::runtime_error("phase5_step3_cli_timeout");
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
    assert(RequestedDatabaseDiagnostic("") ==
           std::optional<std::string>{"LSTM_TEST_DB_NAME_required"});
    for (const char* unsafe : {"LSTM", "lstm", "Lstm", "production",
                               "postgres", "unrelated"})
        assert(RequestedDatabaseDiagnostic(unsafe) ==
               std::optional<std::string>{"isolated_test_database_required"});
    assert(!RequestedDatabaseDiagnostic(
        "expertadvisor_phase5_step3_test_20260719_01"));
    assert(ConnectedDatabaseDiagnostic(
               "expertadvisor_phase5_step3_test_requested",
               "expertadvisor_phase5_step3_test_connected") ==
           std::optional<std::string>{"connected_database_name_mismatch"});

    const char* configured = std::getenv("LSTM_TEST_DB_NAME");
    if (const auto diagnostic = RequestedDatabaseDiagnostic(configured))
    {
        std::cerr << *diagnostic << '\n';
        return 2;
    }
    const std::string database = configured;
    const std::string host = EnvironmentOr("LSTM_TEST_DB_HOST", "127.0.0.1");
    const std::string port = EnvironmentOr("LSTM_TEST_DB_PORT", "5432");
    const std::string ownerUser = EnvironmentOr(
        "LSTM_TEST_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string schema =
        "phase5_campaign_launch_" + std::to_string(getpid());
    const std::string ownerString =
        "host=" + host + " port=" + port + " user=" + ownerUser +
        " dbname=" + database + " connect_timeout=5";
    const std::string runtimeString =
        "host=" + host + " port=" + port + " user=pqxx dbname=" + database +
        " connect_timeout=5" +
        " options='-c search_path=" + schema +
        " -c lock_timeout=5s -c statement_timeout=15s'";
    pqxx::connection owner{ownerString};
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

    std::string currentStage = "setup";
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
VALUES(17,'eurusd',12,0.001,1,5,120,15,
 '2010-01-01 America/Chicago','2025-01-01 America/Chicago',
 '2025-01-01 America/Chicago','2026-01-01 America/Chicago',77,
 'paused','train','source-unchanged','2026-07-19 12:34:56+00');
INSERT INTO model VALUES(77,'unchanged');
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
                const std::string sql = ReadFile(migration);
                setup.exec(sql);
                setup.exec(sql);
            }
            setup.exec(ReadFile(
                "Tests/ExperimentRecommendationCampaignMaterializationMigrationTests.sql"));
            setup.exec(
                "GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                " TO pqxx; GRANT SELECT ON model,experiment_recommendation,"
                "experiment_recommendation_ranking_snapshot,"
                "experiment_recommendation_ranking_member TO pqxx;"
                "GRANT SELECT,INSERT,UPDATE ON experiment TO pqxx;"
                "GRANT USAGE ON SEQUENCE experiment_experiment_id_seq TO pqxx;");
            setup.commit();
        }

        pqxx::connection runtime{runtimeString};
        {
            pqxx::work schemaBoundary{owner};
            SetSearchPath(schemaBoundary, schema);
            assert(RecommendationCampaignLaunchSchemasExist(schemaBoundary));
            schemaBoundary.exec(
                "DROP TABLE experiment_recommendation_conversion_activation;");
            assert(!RecommendationCampaignLaunchSchemasExist(schemaBoundary));
            // No commit: preserve the installed production schema after proving
            // that a missing activation object fails the Step 3 schema check.
        }
        const std::string sourceBefore = Text(
            owner, schema,
            "SELECT row_to_json(e)::text FROM experiment e WHERE experiment_id=17;");

        const std::vector<CampaignProposal> first{
            CreateApprovedProposal(runtime, 1.11, 1),
            CreateApprovedProposal(runtime, 1.12, 2)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 1, first);
            fixture.commit();
        }
        currentStage = "first_launch";
        const auto firstResult = RunLaunch(runtimeString, 1);
        assert(firstResult.createdExecutions.size() == 2);
        assert(firstResult.createdActivations.size() == 2);
        for (const auto& value : first)
            AssertPending(runtime, owner, schema, value);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment_recommendation_conversion_execution;") == 2);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment_recommendation_conversion_activation;") == 2);

        const std::string executionSequenceBeforeRetry = SequenceState(
            owner, schema, "experiment_recommendation_conversion_execution",
            "recommendation_conversion_execution_id");
        const std::string activationSequenceBeforeRetry = SequenceState(
            owner, schema, "experiment_recommendation_conversion_activation",
            "recommendation_conversion_activation_id");
        currentStage = "exact_retry";
        const auto retry = RunLaunch(runtimeString, 1);
        assert(retry.plan.state ==
               RecommendationCampaignLaunchPlanState::alreadySatisfied);
        assert(retry.createdExecutions.empty());
        assert(retry.createdActivations.empty());
        assert(SequenceState(owner, schema,
            "experiment_recommendation_conversion_execution",
            "recommendation_conversion_execution_id") ==
            executionSequenceBeforeRetry);
        assert(SequenceState(owner, schema,
            "experiment_recommendation_conversion_activation",
            "recommendation_conversion_activation_id") ==
            activationSequenceBeforeRetry);

        {
            pqxx::work progress{owner};
            SetSearchPath(progress, schema);
            progress.exec(
                "UPDATE experiment SET status='running' WHERE experiment_id=$1;",
                pqxx::params{retry.plan.members[0].experimentId});
            progress.commit();
        }
        bool progressedStateConflict = false;
        try { (void)RunLaunch(runtimeString, 1); }
        catch (const std::invalid_argument& error)
        {
            progressedStateConflict = std::string{error.what()} ==
                "campaign_activation_existing_state_invalid";
        }
        assert(progressedStateConflict);
        {
            pqxx::work restore{owner};
            SetSearchPath(restore, schema);
            restore.exec(
                "UPDATE experiment SET status='pending' WHERE experiment_id=$1;",
                pqxx::params{retry.plan.members[0].experimentId});
            restore.commit();
        }

        const std::vector<CampaignProposal> existing{
            CreateApprovedProposal(runtime, 1.13, 3),
            CreateApprovedProposal(runtime, 1.14, 4)};
        const auto existingExecution0 = Execute(runtime, existing[0]);
        const auto existingExecution1 = Execute(runtime, existing[1]);
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 2, existing);
            fixture.commit();
        }
        currentStage = "existing_executions";
        const auto existingResult = RunLaunch(runtimeString, 2);
        assert(existingResult.createdExecutions.empty());
        assert(existingResult.createdActivations.size() == 2);
        assert(existingResult.plan.state == RecommendationCampaignLaunchPlanState::
               readyToActivateExistingExecutions);
        assert(existingResult.plan.members[0].executionId ==
               existingExecution0.executionId);
        assert(existingResult.plan.members[1].executionId ==
               existingExecution1.executionId);
        for (const auto& value : existing)
            AssertPending(runtime, owner, schema, value);
        const auto reversedReview =
            RecordRecommendationConversionProposalReviewDecision(
                runtime,
                Review(existing[0].proposal.proposalId,
                       "phase5-launch-review-reversal",
                       RecommendationConversionProposalReviewDecision::reject));
        assert(reversedReview.decision);
        const auto reversalRetry = RunLaunch(runtimeString, 2);
        assert(reversalRetry.plan.state ==
               RecommendationCampaignLaunchPlanState::alreadySatisfied);
        assert(reversalRetry.plan.members[0].authorizationReviewDecisionId ==
               existingExecution0.reviewDecisionId);
        assert(reversalRetry.plan.members[0].authorizationReviewDecisionId !=
               reversedReview.decision->reviewDecisionId);

        const std::vector<CampaignProposal> partialExecution{
            CreateApprovedProposal(runtime, 1.15, 5),
            CreateApprovedProposal(runtime, 1.16, 6)};
        (void)Execute(runtime, partialExecution[0]);
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 3, partialExecution);
            fixture.commit();
        }
        bool partialExecutionConflict = false;
        currentStage = "partial_execution";
        try { (void)RunLaunch(runtimeString, 3); }
        catch (const std::invalid_argument& error)
        {
            partialExecutionConflict = std::string{error.what()} ==
                "campaign_execution_partial_operation_conflict";
        }
        assert(partialExecutionConflict);
        assert(!FindRecommendationConversionExecutionByProposal(
            runtime, partialExecution[1].proposal.proposalId));

        const std::vector<CampaignProposal> partialActivation{
            CreateApprovedProposal(runtime, 1.17, 7),
            CreateApprovedProposal(runtime, 1.18, 8)};
        const auto partialActivationExecution0 = Execute(
            runtime, partialActivation[0]);
        const auto partialActivationExecution1 = Execute(
            runtime, partialActivation[1]);
        (void)partialActivationExecution1;
        assert(ActivateRecommendationConversionExecution(
            runtime, partialActivationExecution0.executionId).activation);
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 4, partialActivation);
            fixture.commit();
        }
        bool partialActivationConflict = false;
        currentStage = "partial_activation";
        try { (void)RunLaunch(runtimeString, 4); }
        catch (const std::invalid_argument& error)
        {
            partialActivationConflict = std::string{error.what()} ==
                "campaign_activation_partial_operation_conflict";
        }
        assert(partialActivationConflict);
        assert(!FindRecommendationConversionActivationByExecution(
            runtime, partialActivationExecution1.executionId));

        const std::vector<CampaignProposal> activationRollback{
            CreateApprovedProposal(runtime, 1.19, 9),
            CreateApprovedProposal(runtime, 1.20, 10)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 5, activationRollback);
            fixture.exec(R"SQL(
CREATE FUNCTION fail_phase5_launch_activation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.recommendation_conversion_proposal_id = TG_ARGV[0]::bigint THEN
  RAISE EXCEPTION 'injected phase5 launch activation failure';
 END IF;
 RETURN NEW;
END $$;
)SQL");
            fixture.exec(
                "CREATE TRIGGER fail_phase5_launch_activation_trigger BEFORE "
                "INSERT ON experiment_recommendation_conversion_activation "
                "FOR EACH ROW EXECUTE FUNCTION fail_phase5_launch_activation('" +
                std::to_string(activationRollback[1].proposal.proposalId) +
                "');");
            fixture.commit();
        }
        const long long experimentCountBeforeActivationFailure = Scalar(
            owner, schema, "SELECT count(*) FROM experiment;");
        bool activationFailure = false;
        currentStage = "activation_rollback";
        try { (void)RunLaunch(runtimeString, 5); }
        catch (const std::exception&) { activationFailure = true; }
        assert(activationFailure);
        assert(Scalar(owner, schema, "SELECT count(*) FROM experiment;") ==
               experimentCountBeforeActivationFailure);
        for (const auto& value : activationRollback)
        {
            assert(!FindRecommendationConversionExecutionByProposal(
                runtime, value.proposal.proposalId));
        }
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            fixture.exec(
                "DROP TRIGGER fail_phase5_launch_activation_trigger ON "
                "experiment_recommendation_conversion_activation;"
                "DROP FUNCTION fail_phase5_launch_activation();");
            fixture.commit();
        }

        const std::vector<CampaignProposal> executionRollback{
            CreateApprovedProposal(runtime, 1.21, 11),
            CreateApprovedProposal(runtime, 1.22, 12)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 6, executionRollback);
            fixture.exec(R"SQL(
CREATE FUNCTION fail_phase5_launch_execution() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.recommendation_conversion_proposal_id = TG_ARGV[0]::bigint THEN
  RAISE EXCEPTION 'injected phase5 launch execution failure';
 END IF;
 RETURN NEW;
END $$;
)SQL");
            fixture.exec(
                "CREATE TRIGGER fail_phase5_launch_execution_trigger BEFORE "
                "INSERT ON experiment_recommendation_conversion_execution "
                "FOR EACH ROW EXECUTE FUNCTION fail_phase5_launch_execution('" +
                std::to_string(executionRollback[1].proposal.proposalId) +
                "');");
            fixture.commit();
        }
        const long long experimentCountBeforeExecutionFailure = Scalar(
            owner, schema, "SELECT count(*) FROM experiment;");
        bool executionFailure = false;
        currentStage = "execution_rollback";
        try { (void)RunLaunch(runtimeString, 6); }
        catch (const std::exception&) { executionFailure = true; }
        assert(executionFailure);
        assert(Scalar(owner, schema, "SELECT count(*) FROM experiment;") ==
               experimentCountBeforeExecutionFailure);
        for (const auto& value : executionRollback)
            assert(!FindRecommendationConversionExecutionByProposal(
                runtime, value.proposal.proposalId));
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            fixture.exec(
                "DROP TRIGGER fail_phase5_launch_execution_trigger ON "
                "experiment_recommendation_conversion_execution;"
                "DROP FUNCTION fail_phase5_launch_execution();");
            fixture.commit();
        }

        currentStage = "dry_run";
        const std::vector<CampaignProposal> dry{
            CreateApprovedProposal(runtime, 1.23, 13)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 7, dry);
            fixture.commit();
        }
        const std::string experimentSequenceBeforeDry = SequenceState(
            owner, schema, "experiment", "experiment_id");
        const std::string executionSequenceBeforeDry = SequenceState(
            owner, schema, "experiment_recommendation_conversion_execution",
            "recommendation_conversion_execution_id");
        const std::string activationSequenceBeforeDry = SequenceState(
            owner, schema, "experiment_recommendation_conversion_activation",
            "recommendation_conversion_activation_id");
        {
            std::ostringstream output;
            std::ostringstream errors;
            assert(RunRecommendationCampaignLaunchCommand(
                runtimeString, {7, true}, output, errors) == 0);
            assert(errors.str().empty());
            assert(output.str().find("status=dry_run_ready_to_launch") !=
                   std::string::npos);
            assert(output.str().find("execution_id=null") != std::string::npos);
            assert(output.str().find("experiment_id=null") != std::string::npos);
            assert(output.str().find("activation_id=null") != std::string::npos);
            assert(output.str().find("scheduler_polled=false") !=
                   std::string::npos);
        }
        assert(RunReleaseLaunchCli(database, host, port, schema, 7, true) == 0);
        assert(SequenceState(owner, schema, "experiment", "experiment_id") ==
               experimentSequenceBeforeDry);
        assert(SequenceState(owner, schema,
            "experiment_recommendation_conversion_execution",
            "recommendation_conversion_execution_id") ==
            executionSequenceBeforeDry);
        assert(SequenceState(owner, schema,
            "experiment_recommendation_conversion_activation",
            "recommendation_conversion_activation_id") ==
            activationSequenceBeforeDry);

        currentStage = "identical_concurrency";
        const std::vector<CampaignProposal> identical{
            CreateApprovedProposal(runtime, 1.24, 14),
            CreateApprovedProposal(runtime, 1.25, 15)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 8, identical);
            fixture.commit();
        }
        RecommendationCampaignLaunchPersistResult identicalLeft;
        RecommendationCampaignLaunchPersistResult identicalRight;
        std::exception_ptr identicalLeftError;
        std::exception_ptr identicalRightError;
        std::barrier identicalStart{3};
        auto invokeLaunch = [&](long long id,
                                RecommendationCampaignLaunchPersistResult& result,
                                std::exception_ptr& error) {
            try
            {
                identicalStart.arrive_and_wait();
                result = RunLaunch(runtimeString, id);
            }
            catch (...) { error = std::current_exception(); }
        };
        std::thread identicalLeftThread{
            invokeLaunch, 8, std::ref(identicalLeft),
            std::ref(identicalLeftError)};
        std::thread identicalRightThread{
            invokeLaunch, 8, std::ref(identicalRight),
            std::ref(identicalRightError)};
        identicalStart.arrive_and_wait();
        identicalLeftThread.join();
        identicalRightThread.join();
        if (identicalLeftError) std::rethrow_exception(identicalLeftError);
        if (identicalRightError) std::rethrow_exception(identicalRightError);
        assert((identicalLeft.createdExecutions.size() == 2 &&
                identicalRight.plan.state ==
                    RecommendationCampaignLaunchPlanState::alreadySatisfied) ||
               (identicalRight.createdExecutions.size() == 2 &&
                identicalLeft.plan.state ==
                    RecommendationCampaignLaunchPlanState::alreadySatisfied));

        currentStage = "overlap_concurrency";
        const std::vector<CampaignProposal> overlap{
            CreateApprovedProposal(runtime, 1.26, 16),
            CreateApprovedProposal(runtime, 1.27, 17),
            CreateApprovedProposal(runtime, 1.28, 18)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 9, {overlap[0], overlap[1]});
            InsertMaterialization(fixture, 10, {overlap[2], overlap[1]});
            fixture.commit();
        }
        RecommendationCampaignLaunchPersistResult overlapLeft;
        RecommendationCampaignLaunchPersistResult overlapRight;
        std::exception_ptr overlapLeftError;
        std::exception_ptr overlapRightError;
        std::barrier overlapStart{3};
        auto invokeOverlap = [&](long long id,
                                 RecommendationCampaignLaunchPersistResult& result,
                                 std::exception_ptr& error) {
            try
            {
                overlapStart.arrive_and_wait();
                result = RunLaunch(runtimeString, id);
            }
            catch (...) { error = std::current_exception(); }
        };
        std::thread overlapLeftThread{
            invokeOverlap, 9, std::ref(overlapLeft), std::ref(overlapLeftError)};
        std::thread overlapRightThread{
            invokeOverlap, 10, std::ref(overlapRight), std::ref(overlapRightError)};
        overlapStart.arrive_and_wait();
        overlapLeftThread.join();
        overlapRightThread.join();
        assert(static_cast<bool>(overlapLeftError) !=
               static_cast<bool>(overlapRightError));
        try
        {
            std::rethrow_exception(
                overlapLeftError ? overlapLeftError : overlapRightError);
        }
        catch (const std::invalid_argument& error)
        {
            assert(std::string{error.what()} ==
                   "campaign_execution_partial_operation_conflict");
        }

        currentStage = "direct_execution_race";
        const std::vector<CampaignProposal> directExecutionRace{
            CreateApprovedProposal(runtime, 1.29, 19)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 11, directExecutionRace);
            fixture.commit();
        }
        std::exception_ptr launchRaceError;
        std::exception_ptr directExecutionError;
        std::barrier directExecutionStart{3};
        std::thread launchRaceThread{[&] {
            try
            {
                directExecutionStart.arrive_and_wait();
                (void)RunLaunch(runtimeString, 11);
            }
            catch (...) { launchRaceError = std::current_exception(); }
        }};
        std::thread directExecutionThread{[&] {
            try
            {
                directExecutionStart.arrive_and_wait();
                pqxx::connection direct{runtimeString};
                (void)ExecuteApprovedRecommendationConversionProposal(
                    direct, directExecutionRace[0].proposal.proposalId);
            }
            catch (...) { directExecutionError = std::current_exception(); }
        }};
        directExecutionStart.arrive_and_wait();
        launchRaceThread.join();
        directExecutionThread.join();
        if (launchRaceError) std::rethrow_exception(launchRaceError);
        if (directExecutionError) std::rethrow_exception(directExecutionError);
        AssertPending(runtime, owner, schema, directExecutionRace[0]);

        currentStage = "step1_race";
        const std::vector<CampaignProposal> step1Race{
            CreateApprovedProposal(runtime, 1.30, 20)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 12, step1Race);
            fixture.commit();
        }
        std::exception_ptr launchStep1Error;
        std::exception_ptr step1Error;
        std::barrier step1Start{3};
        std::thread launchStep1Thread{[&] {
            try
            {
                step1Start.arrive_and_wait();
                (void)RunLaunch(runtimeString, 12);
            }
            catch (...) { launchStep1Error = std::current_exception(); }
        }};
        std::thread step1Thread{[&] {
            try
            {
                step1Start.arrive_and_wait();
                (void)RunStep1(runtimeString, 12);
            }
            catch (...) { step1Error = std::current_exception(); }
        }};
        step1Start.arrive_and_wait();
        launchStep1Thread.join();
        step1Thread.join();
        if (launchStep1Error) std::rethrow_exception(launchStep1Error);
        if (step1Error) std::rethrow_exception(step1Error);
        AssertPending(runtime, owner, schema, step1Race[0]);

        currentStage = "direct_activation_race";
        const std::vector<CampaignProposal> activationRace{
            CreateApprovedProposal(runtime, 1.31, 21)};
        const auto activationRaceExecution = Execute(runtime, activationRace[0]);
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 13, activationRace);
            fixture.commit();
        }
        std::exception_ptr launchActivationError;
        std::exception_ptr directActivationError;
        std::barrier activationStart{3};
        std::thread launchActivationThread{[&] {
            try
            {
                activationStart.arrive_and_wait();
                (void)RunLaunch(runtimeString, 13);
            }
            catch (...) { launchActivationError = std::current_exception(); }
        }};
        std::thread directActivationThread{[&] {
            try
            {
                activationStart.arrive_and_wait();
                pqxx::connection direct{runtimeString};
                (void)ActivateRecommendationConversionExecution(
                    direct, activationRaceExecution.executionId);
            }
            catch (...) { directActivationError = std::current_exception(); }
        }};
        activationStart.arrive_and_wait();
        launchActivationThread.join();
        directActivationThread.join();
        if (launchActivationError) std::rethrow_exception(launchActivationError);
        if (directActivationError) std::rethrow_exception(directActivationError);
        AssertPending(runtime, owner, schema, activationRace[0]);

        currentStage = "step2_race";
        const std::vector<CampaignProposal> step2Race{
            CreateApprovedProposal(runtime, 1.32, 22)};
        (void)Execute(runtime, step2Race[0]);
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 14, step2Race);
            fixture.commit();
        }
        std::exception_ptr launchStep2Error;
        std::exception_ptr step2Error;
        std::barrier step2Start{3};
        std::thread launchStep2Thread{[&] {
            try
            {
                step2Start.arrive_and_wait();
                (void)RunLaunch(runtimeString, 14);
            }
            catch (...) { launchStep2Error = std::current_exception(); }
        }};
        std::thread step2Thread{[&] {
            try
            {
                step2Start.arrive_and_wait();
                (void)RunStep2(runtimeString, 14);
            }
            catch (...) { step2Error = std::current_exception(); }
        }};
        step2Start.arrive_and_wait();
        launchStep2Thread.join();
        step2Thread.join();
        if (launchStep2Error) std::rethrow_exception(launchStep2Error);
        if (step2Error) std::rethrow_exception(step2Error);
        AssertPending(runtime, owner, schema, step2Race[0]);

        currentStage = "review_race";
        const std::vector<CampaignProposal> reviewRace{
            CreateApprovedProposal(runtime, 1.33, 23)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 15, reviewRace);
            fixture.commit();
        }
        std::exception_ptr launchReviewError;
        std::exception_ptr reviewError;
        std::barrier reviewStart{3};
        std::thread launchReviewThread{[&] {
            try
            {
                reviewStart.arrive_and_wait();
                (void)RunLaunch(runtimeString, 15);
            }
            catch (...) { launchReviewError = std::current_exception(); }
        }};
        std::thread reviewThread{[&] {
            try
            {
                reviewStart.arrive_and_wait();
                pqxx::connection direct{runtimeString};
                (void)RecordRecommendationConversionProposalReviewDecision(
                    direct, Review(
                        reviewRace[0].proposal.proposalId,
                        "phase5-launch-review-race-reject",
                        RecommendationConversionProposalReviewDecision::reject));
            }
            catch (...) { reviewError = std::current_exception(); }
        }};
        reviewStart.arrive_and_wait();
        launchReviewThread.join();
        reviewThread.join();
        if (reviewError) std::rethrow_exception(reviewError);
        const auto reviewRaceExecution =
            FindRecommendationConversionExecutionByProposal(
                runtime, reviewRace[0].proposal.proposalId);
        if (launchReviewError)
        {
            try { std::rethrow_exception(launchReviewError); }
            catch (const std::invalid_argument& error)
            {
                assert(std::string{error.what()} ==
                       "campaign_execution_proposal_not_approved");
            }
            assert(!reviewRaceExecution);
        }
        else
        {
            assert(reviewRaceExecution);
            AssertPending(runtime, owner, schema, reviewRace[0]);
        }

        currentStage = "confirmed_service_write";
        const std::vector<CampaignProposal> confirmedService{
            CreateApprovedProposal(runtime, 1.34, 24)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 16, confirmedService);
            fixture.commit();
        }
        {
            std::ostringstream output;
            std::ostringstream errors;
            assert(RunRecommendationCampaignLaunchCommand(
                runtimeString, {16, false}, output, errors) == 0);
            assert(errors.str().empty());
            assert(output.str().find("status=newly_launched") !=
                   std::string::npos);
            assert(output.str().find("scheduler_started=false") !=
                   std::string::npos);
            assert(output.str().find("workers_launched=false") !=
                   std::string::npos);
        }
        AssertPending(runtime, owner, schema, confirmedService[0]);

        currentStage = "confirmed_release_cli_write";
        const std::vector<CampaignProposal> confirmedReleaseCli{
            CreateApprovedProposal(runtime, 1.35, 25)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 17, confirmedReleaseCli);
            fixture.commit();
        }
        assert(RunReleaseLaunchCli(
            database, host, port, schema, 17, false) == 0);
        AssertPending(runtime, owner, schema, confirmedReleaseCli[0]);

        bool updateDenied = false;
        try
        {
            pqxx::work forbidden{runtime};
            forbidden.exec(
                "UPDATE experiment_recommendation_campaign_materialization "
                "SET materialized_by='forged';");
        }
        catch (const pqxx::sql_error& error)
        {
            updateDenied = PermissionDenied(error);
        }
        assert(updateDenied);
        bool memberDeleteDenied = false;
        try
        {
            pqxx::work forbidden{runtime};
            forbidden.exec(
                "DELETE FROM "
                "experiment_recommendation_campaign_materialization_member;");
        }
        catch (const pqxx::sql_error& error)
        {
            memberDeleteDenied = PermissionDenied(error);
        }
        assert(memberDeleteDenied);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM information_schema.tables WHERE "
            "table_schema=current_schema() AND ("
            "table_name LIKE 'experiment_recommendation_campaign_launch%' OR "
            "table_name LIKE 'experiment_recommendation_campaign_execution%' OR "
            "table_name LIKE 'experiment_recommendation_campaign_activation%');") == 0);
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
        std::cerr << "phase5_step3_test_stage=" << currentStage << '\n';
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
        "Experiment recommendation campaign launch repository tests passed\n";
    return 0;
}
