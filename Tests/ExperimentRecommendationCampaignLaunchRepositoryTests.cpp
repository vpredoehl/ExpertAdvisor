#include "../Sources/ExperimentRecommendationCampaignLaunchRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignLaunchService.hpp"
#include "../Sources/ExperimentRecommendationCampaignExecutionRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignActivationRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionExecutionRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionProposalReviewRepository.hpp"
#include "../Sources/CampaignOperationsDispatchService.hpp"
#include "../Sources/CampaignOperationsControlService.hpp"
#include "../Sources/CampaignOperationsRepository.hpp"
#include "../Sources/CampaignOperationsService.hpp"

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
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{
std::string RecommendationMachineText(const std::string& value)
{
    std::ostringstream escaped;
    escaped << std::uppercase << std::hex;
    for (const unsigned char ch : value)
    {
        const bool alphanumeric =
            (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9');
        if (alphanumeric || ch == '-' || ch == '_' || ch == '.' ||
            ch == ':' || ch == '/' || ch == ';')
            escaped << static_cast<char>(ch);
        else
            escaped << '%' << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(ch);
    }
    return escaped.str() == "NULL" ? "%4E%55%4C%4C" : escaped.str();
}
} // namespace EA::ExperimentRecommendation

using namespace EA::ExperimentRecommendation;

namespace
{

bool IsApprovedDatabase(const std::string& database)
{
    constexpr std::string_view prefixes[]{
        "expertadvisor_phase5_step3_test_",
        "expertadvisor_phase5_step3_final_",
        "expertadvisor_campaign_operations_phase3_test_"};
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

EA::CampaignOperations::AcceptedOperationalRequest
CreatePhase3Request(pqxx::connection& owner, const std::string& schema,
    long long materializationId,
    std::optional<EA::CampaignOperations::UtcTimestamp>
        authorizationExpiresAt = std::nullopt)
{
    using namespace EA::CampaignOperations;
    EA::ExperimentRecommendation::
        PersistedRecommendationCampaignMaterialization materialization;
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        const auto loaded =
            LoadRecommendationCampaignLaunchMaterialization(
                transaction, materializationId);
        assert(loaded);
        materialization = *loaded;
        transaction.commit();
    }
    const OperationalCampaign campaign = BuildOperationalCampaign(
        materialization.materializationId, materialization.contractVersion,
        materialization.identityCanonical, materialization.identityHash,
        materialization.selectedMemberCount);
    const PersistedOperationalCampaign persistedCampaign = [&]
    {
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_campaign_creator;");
        auto result = PersistOperationalCampaign(
            transaction, campaign,
            ActorIdentity("phase3.creator@example.test"),
            Reason("Create isolated Phase 3 handoff fixture."));
        transaction.commit();
        return result.persisted;
    }();
    const PersistedOperationalAuthorizationEvent authorization = [&]
    {
        const auto grant = BuildOperationalAuthorizationEvent(
            persistedCampaign.campaignId,
            campaign.identity.canonicalText(), std::nullopt, std::nullopt,
            std::nullopt, 1, AuthorizationEventKind::granted,
            campaign.actionKind, campaign.actionContractVersion,
            campaign.scopeKind, campaign.scopeContractVersion,
            PrerequisitePolicy::phase4dMaterializationOnlyV1,
            std::nullopt, std::nullopt, std::nullopt,
            kCampaignOperationsAuthorizationRole,
            ActorIdentity("phase3.authorizer@example.test"),
            Reason("Authorize isolated complete materialization dispatch."),
            UtcTimestamp("2020-01-01T00:00:00.000000Z"),
            std::move(authorizationExpiresAt));
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_authorizer;");
        auto result = PersistOperationalAuthorizationEvent(
            transaction, grant);
        transaction.commit();
        return result.persisted;
    }();
    (void)authorization;
    {
        const auto budget = BuildBudgetLedgerEntry(
            persistedCampaign.campaignId,
            campaign.identity.canonicalText(), std::nullopt, std::nullopt,
            std::nullopt, 1, BudgetLedgerEntryKind::grant,
            BudgetLedgerStatus::active,
            BudgetUnit::materializedMemberDispatch,
            materialization.selectedMemberCount, 0,
            materialization.selectedMemberCount,
            ActorIdentity("phase3.budget@example.test"),
            Reason("Fund isolated complete materialization dispatch."));
        pqxx::work transaction{owner};
        SetSearchPath(transaction, schema);
        transaction.exec(
            "SET LOCAL ROLE campaign_operations_budget_administrator;");
        (void)PersistBudgetLedgerEntry(transaction, budget);
        transaction.commit();
    }
    pqxx::work transaction{owner};
    SetSearchPath(transaction, schema);
    transaction.exec(
        "SET LOCAL ROLE campaign_operations_request_acceptor;");
    auto accepted = PersistAcceptedOperationalRequest(transaction,
        persistedCampaign.campaignId,
        ActorIdentity("phase3.requester@example.test"),
        Reason("Accept isolated complete materialization dispatch."),
        std::nullopt);
    transaction.commit();
    return accepted.persisted;
}

EA::CampaignOperations::PersistedOperationalAuthorizationEvent
GrantPhase3Adoption(pqxx::connection& owner,
    const std::string& schema,
    const EA::CampaignOperations::AcceptedOperationalRequest& accepted,
    std::optional<EA::CampaignOperations::UtcTimestamp> expiresAt =
        std::nullopt)
{
    using namespace EA::CampaignOperations;
    const auto& operation = accepted.request.request.logicalOperation;
    const auto grant = BuildOperationalAuthorizationEvent(
        operation.campaignId, operation.campaignCanonicalText,
        std::nullopt, std::nullopt, std::nullopt, 1,
        AuthorizationEventKind::granted,
        OperationalActionKind::adoptExistingPendingAndControl, 1,
        operation.scopeKind, operation.scopeContractVersion,
        PrerequisitePolicy::phase4dMaterializationOnlyV1,
        std::nullopt, std::nullopt, std::nullopt,
        kCampaignOperationsAuthorizationRole,
        ActorIdentity("phase3.adoption.authorizer@example.test"),
        Reason("Authorize exact existing pending work adoption."),
        UtcTimestamp("2020-01-01T00:00:00.000000Z"),
        std::move(expiresAt));
    pqxx::work transaction{owner};
    SetSearchPath(transaction, schema);
    transaction.exec(
        "SET LOCAL ROLE campaign_operations_authorizer;");
    auto persisted =
        PersistOperationalAuthorizationEvent(transaction, grant).persisted;
    transaction.commit();
    return persisted;
}

void RecordPhase3AdoptionAuthorizationSuccessor(
    pqxx::connection& owner, const std::string& schema,
    const EA::CampaignOperations::AcceptedOperationalRequest& accepted,
    const EA::CampaignOperations::PersistedOperationalAuthorizationEvent&
        prior,
    EA::CampaignOperations::AuthorizationEventKind kind)
{
    using namespace EA::CampaignOperations;
    const auto& operation = accepted.request.request.logicalOperation;
    const auto successor = BuildOperationalAuthorizationEvent(
        operation.campaignId, operation.campaignCanonicalText,
        prior.authorizationEventId, prior.event.identity.canonicalText(),
        prior.event.identity.hash(), prior.event.chainVersion + 1, kind,
        OperationalActionKind::adoptExistingPendingAndControl, 1,
        operation.scopeKind, operation.scopeContractVersion,
        PrerequisitePolicy::phase4dMaterializationOnlyV1,
        std::nullopt, std::nullopt, std::nullopt,
        kCampaignOperationsAuthorizationRole,
        ActorIdentity("phase3.adoption.successor@example.test"),
        Reason(kind == AuthorizationEventKind::revoked
            ? "Revoke exact adoption authority for verification."
            : "Supersede exact adoption authority for verification."),
        UtcTimestamp("2020-01-01T00:00:00.000000Z"), std::nullopt);
    pqxx::work transaction{owner};
    SetSearchPath(transaction, schema);
    transaction.exec(
        "SET LOCAL ROLE campaign_operations_authorizer;");
    (void)PersistOperationalAuthorizationEvent(transaction, successor);
    transaction.commit();
}

void RecordPhase3DispatchAuthorizationSuccessor(
    pqxx::connection& owner, const std::string& schema,
    const EA::CampaignOperations::AcceptedOperationalRequest& accepted,
    EA::CampaignOperations::AuthorizationEventKind kind)
{
    using namespace EA::CampaignOperations;
    const auto& request = accepted.request.request;
    const auto& operation = request.logicalOperation;
    const auto successor = BuildOperationalAuthorizationEvent(
        operation.campaignId, operation.campaignCanonicalText,
        request.acceptingAuthorizationEventId,
        request.acceptingAuthorizationCanonicalText,
        request.acceptingAuthorizationIdentityHash, 2, kind,
        operation.actionKind, operation.actionContractVersion,
        operation.scopeKind, operation.scopeContractVersion,
        request.prerequisitePolicy, std::nullopt,
        request.provenanceCanonicalText, request.provenanceIdentityHash,
        kCampaignOperationsAuthorizationRole,
        ActorIdentity("phase3.authorization.successor@example.test"),
        Reason(kind == AuthorizationEventKind::revoked
            ? "Revoke exact dispatch authority during race verification."
            : "Supersede exact dispatch authority during race verification."),
        UtcTimestamp("2020-01-01T00:00:00.000000Z"), std::nullopt);
    pqxx::work transaction{owner};
    SetSearchPath(transaction, schema);
    transaction.exec(
        "SET LOCAL ROLE campaign_operations_authorizer;");
    (void)PersistOperationalAuthorizationEvent(transaction, successor);
    transaction.commit();
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
            setup.exec(R"SQL(
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
            for (const char* migration : {
                     "Database/migrations/045_campaign_operations_foundation.sql",
                     "Database/migrations/047_campaign_operations_budget_request_acceptance.sql",
                     "Database/migrations/048_campaign_operations_durable_dispatch_handoff.sql",
                     "Database/migrations/049_campaign_operations_controls_cancellation_reconciliation.sql"})
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
                "DROP TABLE experiment_recommendation_conversion_activation "
                "CASCADE;");
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

        currentStage = "campaign_operations_phase3_atomic_handoff";
        const std::vector<CampaignProposal> phase3Proposals{
            CreateApprovedProposal(runtime, 1.36, 26),
            CreateApprovedProposal(runtime, 1.37, 27)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 18, phase3Proposals);
            fixture.commit();
        }
        const auto phase3Request =
            CreatePhase3Request(owner, schema, 18);
        const std::string phase3ConnectionString = ownerString +
            " options='-c search_path=" + schema +
            " -c lock_timeout=5s -c statement_timeout=15s'";
        const EA::CampaignOperations::IsolatedDispatchSafetyGate phase3Gate{
            database,
            EA::CampaignOperations::
                kCampaignOperationsPhase3TestAcknowledgement};
        const auto phase3Result =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString, phase3Request.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.dispatcher@example.test"),
                phase3Gate);
        assert(phase3Result.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                createdAndBound);
        assert(!phase3Result.bindingSetIdentityHash.empty());
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM campaign_operations_request_binding "
            "WHERE operational_request_id=" +
            std::to_string(phase3Request.request.requestId.value())) == 2);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM "
            "campaign_operations_downstream_control_owner "
            "WHERE operational_request_id=" +
            std::to_string(phase3Request.request.requestId.value())) == 2);
        assert(Text(owner, schema,
            "SELECT request_state||':'||state_version::text||':'||"
            "coalesce(lease_token_hash,'NULL') FROM "
            "campaign_operations_operational_request "
            "WHERE operational_request_id=" +
            std::to_string(phase3Request.request.requestId.value())) ==
            "bound:3:NULL");
        assert(Text(owner, schema,
            "SELECT reservation_state||':'||state_version::text FROM "
            "campaign_operations_reservation WHERE reservation_id=" +
            std::to_string(phase3Request.reservation.reservationId.value())) ==
            "committed:2");
        const auto replay =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString, phase3Request.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.dispatcher@example.test"),
                phase3Gate);
        assert(replay.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                existingIdentical);
        assert(replay.bindingSetIdentityHash ==
            phase3Result.bindingSetIdentityHash);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM campaign_operations_dispatch_attempt "
            "WHERE operational_request_id=" +
            std::to_string(phase3Request.request.requestId.value())) == 1);

        currentStage =
            "campaign_operations_phase4_completed_bound_cancellation";
        const std::vector<CampaignProposal> phase4CompletedProposals{
            CreateApprovedProposal(runtime, 1.375, 250),
            CreateApprovedProposal(runtime, 1.376, 251)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(
                fixture, 250, phase4CompletedProposals);
            fixture.commit();
        }
        const auto phase4CompletedRequest =
            CreatePhase3Request(owner, schema, 250);
        const auto phase4CompletedDispatch =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                phase4CompletedRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase4.dispatcher@example.test"),
                phase3Gate);
        assert(phase4CompletedDispatch.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                createdAndBound);
        {
            pqxx::work completed{owner};
            SetSearchPath(completed, schema);
            completed.exec(
                "UPDATE experiment SET status='completed',phase='done',"
                "completed_at=transaction_timestamp(),"
                "updated_at=transaction_timestamp() "
                "WHERE experiment_id IN ("
                "SELECT experiment_id FROM "
                "campaign_operations_request_binding "
                "WHERE operational_request_id=$1);",
                pqxx::params{
                    phase4CompletedRequest.request.requestId.value()});
            completed.commit();
        }
        EA::CampaignOperations::CampaignCancellationCommandRequest
            completedCancellation;
        completedCancellation.campaignId =
            phase4CompletedRequest.request.request.logicalOperation
                .campaignId.value();
        completedCancellation.requestId =
            phase4CompletedRequest.request.requestId.value();
        completedCancellation.expectedRequestVersion = 3;
        completedCancellation.operationKey =
            "phase4-completed-bound-cancellation";
        completedCancellation.actorIdentity =
            "phase4.operator@example.test";
        completedCancellation.reason =
            "Record that all bound lifecycle work is already terminal.";
        const auto completedCancellationResult =
            EA::CampaignOperations::CancelCampaign(
                phase3ConnectionString, completedCancellation);
        assert(completedCancellationResult.progress ==
            EA::CampaignOperations::CancellationProgress::settled);
        assert(completedCancellationResult.settlement);
        assert(completedCancellationResult.settlement->settlement
                   .disposition ==
            EA::CampaignOperations::CancellationSettlementDisposition::
                alreadyTerminal);
        const auto completedCancellationReplay =
            EA::CampaignOperations::CancelCampaign(
                phase3ConnectionString, completedCancellation);
        assert(completedCancellationReplay.replay ==
            EA::CampaignOperations::ControlReplayDisposition::
                existingIdentical);
        assert(completedCancellationReplay.settlement);
        assert(completedCancellationReplay.settlement
                   ->cancellationSettlementId ==
            completedCancellationResult.settlement
                ->cancellationSettlementId);
        assert(Text(owner, schema,
            "SELECT request_state||':'||state_version::text "
            "FROM campaign_operations_operational_request "
            "WHERE operational_request_id=" +
            std::to_string(
                phase4CompletedRequest.request.requestId.value())) ==
            "bound:3");
        assert(Text(owner, schema,
            "SELECT reservation_state||':'||state_version::text "
            "FROM campaign_operations_reservation "
            "WHERE reservation_id=" +
            std::to_string(
                phase4CompletedRequest.reservation.reservationId.value())) ==
            "committed:2");
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment_lifecycle_cancellation_event "
            "WHERE cancellation_request_id=("
            "SELECT cancellation_request_id FROM "
            "campaign_operations_cancellation_request "
            "WHERE operational_request_id=" +
            std::to_string(
                phase4CompletedRequest.request.requestId.value()) +
            ") AND disposition='already_terminal'") == 2);

        currentStage =
            "campaign_operations_phase4_interrupted_bound_cancellation";
        const std::vector<CampaignProposal> phase4InterruptedProposals{
            CreateApprovedProposal(runtime, 1.377, 252),
            CreateApprovedProposal(runtime, 1.378, 253)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(
                fixture, 252, phase4InterruptedProposals);
            fixture.commit();
        }
        const auto phase4InterruptedRequest =
            CreatePhase3Request(owner, schema, 252);
        assert(EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                   phase3ConnectionString,
                   phase4InterruptedRequest.request.requestId, 1,
                   EA::CampaignOperations::ActorIdentity(
                       "phase4.dispatcher@example.test"),
                   phase3Gate)
                   .classification ==
            EA::CampaignOperations::DispatchResultClassification::
                createdAndBound);
        EA::CampaignOperations::CampaignCancellationCommandRequest
            interruptedCancellation;
        interruptedCancellation.campaignId =
            phase4InterruptedRequest.request.request.logicalOperation
                .campaignId.value();
        interruptedCancellation.requestId =
            phase4InterruptedRequest.request.requestId.value();
        interruptedCancellation.expectedRequestVersion = 3;
        interruptedCancellation.operationKey =
            "phase4-interrupted-bound-cancellation";
        interruptedCancellation.actorIdentity =
            "phase4.operator@example.test";
        interruptedCancellation.reason =
            "Resume cancellation after one lifecycle member commits.";
        EA::CampaignOperations::PersistedCampaignCancellationRequest
            interruptedIntent = [&]
        {
            pqxx::connection connection{phase3ConnectionString};
            pqxx::work transaction{connection};
            transaction.exec(
                "SET LOCAL ROLE "
                "campaign_operations_cancellation_coordinator;");
            const auto campaign =
                EA::CampaignOperations::FindOperationalCampaign(
                    transaction,
                    phase4InterruptedRequest.request.request.logicalOperation
                        .campaignId);
            assert(campaign);
            const auto request =
                EA::CampaignOperations::BuildCampaignCancellationRequest(
                    campaign->campaignId,
                    campaign->campaign.identity.canonicalText(),
                    phase4InterruptedRequest.request.requestId,
                    phase4InterruptedRequest.request.request.identity
                        .canonicalText(),
                    EA::CampaignOperations::RequestState::bound, 3,
                    interruptedCancellation.operationKey,
                    EA::CampaignOperations::ActorIdentity(
                        interruptedCancellation.actorIdentity),
                    EA::CampaignOperations::Reason(
                        interruptedCancellation.reason));
            auto persisted =
                EA::CampaignOperations::PersistCampaignCancellationRequest(
                    transaction, request);
            transaction.commit();
            return persisted;
        }();
        EA::CampaignOperations::ReconciliationObserveRequest
            interruptedCancellationObservation;
        interruptedCancellationObservation.runKey =
            "phase4-interrupted-bound-cancellation-observation";
        interruptedCancellationObservation.afterRequestId =
            phase4InterruptedRequest.request.requestId.value() - 1;
        interruptedCancellationObservation.limit = 1;
        const auto interruptedObservationResult =
            EA::CampaignOperations::ObserveAndRecoverCampaignOperations(
                phase3ConnectionString,
                interruptedCancellationObservation);
        assert(interruptedObservationResult.selectedCount == 1);
        assert(interruptedObservationResult.observationCount == 1);
        std::vector<std::pair<
            EA::CampaignOperations::DownstreamControlOwnerId, long long>>
            interruptedOwners;
        {
            pqxx::connection connection{phase3ConnectionString};
            pqxx::read_transaction transaction{connection};
            transaction.exec(
                "SET LOCAL ROLE "
                "campaign_operations_cancellation_coordinator;");
            interruptedOwners =
                EA::CampaignOperations::LoadDownstreamControlOwners(
                    transaction,
                    phase4InterruptedRequest.request.requestId);
        }
        assert(interruptedOwners.size() == 2);
        {
            pqxx::connection connection{phase3ConnectionString};
            pqxx::work transaction{connection};
            transaction.exec(
                "SET LOCAL ROLE experiment_lifecycle_cancellation;");
            (void)EA::CampaignOperations::
                ApplyLifecycleCancellationInTransaction(
                    transaction, interruptedIntent,
                    interruptedOwners.front().first,
                    interruptedOwners.front().second);
            transaction.commit();
        }
        const auto interruptedResult =
            EA::CampaignOperations::CancelCampaign(
                phase3ConnectionString, interruptedCancellation);
        assert(interruptedResult.replay ==
            EA::CampaignOperations::ControlReplayDisposition::
                existingIdentical);
        assert(interruptedResult.progress ==
            EA::CampaignOperations::CancellationProgress::settled);
        assert(interruptedResult.settlement);
        assert(interruptedResult.settlement->settlement.disposition ==
            EA::CampaignOperations::CancellationSettlementDisposition::
                lifecycleRequestAccepted);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment_lifecycle_cancellation_event "
            "WHERE cancellation_request_id=" +
            std::to_string(
                interruptedIntent.cancellationRequestId.value())) == 2);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment "
            "WHERE status='cancelled' AND experiment_id IN ("
            "SELECT experiment_id FROM "
            "campaign_operations_request_binding "
            "WHERE operational_request_id=" +
            std::to_string(
                phase4InterruptedRequest.request.requestId.value()) +
            ")") == 2);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM "
            "campaign_operations_reconciliation_observation observation "
            "LEFT JOIN campaign_operations_reconciliation_resolution resolution "
            "USING(reconciliation_observation_id) "
            "WHERE observation.operational_request_id=" +
            std::to_string(
                phase4InterruptedRequest.request.requestId.value()) +
            " AND observation.reason_code="
            "'cancellation_settlement_pending' "
            "AND resolution.reconciliation_resolution_id IS NULL") == 0);
        assert(Text(owner, schema,
            "SELECT resolution.owning_capability||':'||"
            "resolution.resolution_disposition||':'||"
            "(resolution.transition_identity_hash="
            "settlement.settlement_identity_hash)::text "
            "FROM campaign_operations_reconciliation_observation observation "
            "JOIN campaign_operations_reconciliation_resolution resolution "
            "USING(reconciliation_observation_id) "
            "JOIN campaign_operations_cancellation_request cancellation "
            "ON cancellation.operational_request_id="
            "observation.operational_request_id "
            "JOIN campaign_operations_cancellation_settlement settlement "
            "USING(cancellation_request_id) "
            "WHERE observation.operational_request_id=" +
            std::to_string(
                phase4InterruptedRequest.request.requestId.value())) ==
            "campaign_operations_cancellation_coordinator:"
            "cancellation_settled:true");

        currentStage =
            "campaign_operations_phase4_running_cancellation_refusal";
        const std::vector<CampaignProposal> phase4RunningProposals{
            CreateApprovedProposal(runtime, 1.379, 254)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 254, phase4RunningProposals);
            fixture.commit();
        }
        const auto phase4RunningRequest =
            CreatePhase3Request(owner, schema, 254);
        assert(EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                   phase3ConnectionString,
                   phase4RunningRequest.request.requestId, 1,
                   EA::CampaignOperations::ActorIdentity(
                       "phase4.dispatcher@example.test"),
                   phase3Gate)
                   .classification ==
            EA::CampaignOperations::DispatchResultClassification::
                createdAndBound);
        {
            pqxx::work running{owner};
            SetSearchPath(running, schema);
            running.exec(
                "UPDATE experiment SET status='running',phase='train',"
                "updated_at=transaction_timestamp() "
                "WHERE experiment_id IN ("
                "SELECT experiment_id FROM "
                "campaign_operations_request_binding "
                "WHERE operational_request_id=$1);",
                pqxx::params{
                    phase4RunningRequest.request.requestId.value()});
            running.commit();
        }
        EA::CampaignOperations::CampaignCancellationCommandRequest
            runningCancellation;
        runningCancellation.campaignId =
            phase4RunningRequest.request.request.logicalOperation
                .campaignId.value();
        runningCancellation.requestId =
            phase4RunningRequest.request.requestId.value();
        runningCancellation.expectedRequestVersion = 3;
        runningCancellation.operationKey =
            "phase4-running-cancellation-refusal";
        runningCancellation.actorIdentity =
            "phase4.operator@example.test";
        runningCancellation.reason =
            "Record lifecycle refusal without signaling running work.";
        const auto runningCancellationResult =
            EA::CampaignOperations::CancelCampaign(
                phase3ConnectionString, runningCancellation);
        assert(runningCancellationResult.progress ==
            EA::CampaignOperations::CancellationProgress::settled);
        assert(runningCancellationResult.settlement);
        assert(runningCancellationResult.settlement->settlement.disposition ==
            EA::CampaignOperations::CancellationSettlementDisposition::
                runningCancellationNotSupported);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment "
            "WHERE status='running' AND experiment_id IN ("
            "SELECT experiment_id FROM "
            "campaign_operations_request_binding "
            "WHERE operational_request_id=" +
            std::to_string(
                phase4RunningRequest.request.requestId.value()) +
            ")") == 1);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment_lifecycle_cancellation_event "
            "WHERE cancellation_request_id=" +
            std::to_string(runningCancellationResult.request
                .cancellationRequestId.value()) +
            " AND disposition='running_not_supported'") == 1);

        {
            pqxx::work progress{owner};
            SetSearchPath(progress, schema);
            progress.exec(
                "UPDATE experiment SET status='running',phase='train',"
                "updated_at=transaction_timestamp() WHERE experiment_id IN ("
                "SELECT experiment_id FROM "
                "campaign_operations_request_binding WHERE "
                "operational_request_id=$1);",
                pqxx::params{
                    phase3Request.request.requestId.value()});
            progress.commit();
        }
        int progressedReplayPhase5Invocations = 0;
        const auto progressedReplay =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                phase3Request.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.progressed.replay@example.test"),
                phase3Gate,
                [&](EA::CampaignOperations::
                        DispatchTestInjectionPoint point)
                {
                    if (point == EA::CampaignOperations::
                            DispatchTestInjectionPoint::
                                beforePhase5Invocation)
                        ++progressedReplayPhase5Invocations;
                });
        assert(progressedReplay.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                existingIdentical);
        assert(progressedReplayPhase5Invocations == 0);

        currentStage =
            "campaign_operations_phase3_create_with_adoption_grant";
        const std::vector<CampaignProposal> createWithAdoptionProposals{
            CreateApprovedProposal(runtime, 4.10, 400)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(
                fixture, 230, createWithAdoptionProposals);
            fixture.commit();
        }
        const auto createWithAdoptionRequest =
            CreatePhase3Request(owner, schema, 230);
        GrantPhase3Adoption(owner, schema, createWithAdoptionRequest);
        const auto createdWithAdoption =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                createWithAdoptionRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.create.with.adoption@example.test"),
                phase3Gate);
        assert(createdWithAdoption.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                createdAndBound);
        assert(Text(owner, schema,
            "SELECT binding.binding_disposition||':'||owner.control_mode||':'||"
            "coalesce(owner.adoption_authorization_event_id::text,'NULL') "
            "FROM campaign_operations_request_binding binding "
            "JOIN campaign_operations_downstream_control_owner owner USING "
            "(operational_request_id) WHERE binding.request_binding_id="
            "owner.request_binding_id AND binding.operational_request_id=" +
            std::to_string(
                createWithAdoptionRequest.request.requestId.value())) ==
            "created:created_control:NULL");

        currentStage =
            "campaign_operations_phase3_direct_phase5_overlap";
        const std::vector<CampaignProposal> directOverlapProposals{
            CreateApprovedProposal(runtime, 4.11, 401)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 231, directOverlapProposals);
            fixture.commit();
        }
        const auto directOverlapRequest =
            CreatePhase3Request(owner, schema, 231);
        std::barrier handoffAtPhase5{2};
        std::barrier allowHandoffPhase5{2};
        std::optional<EA::CampaignOperations::DispatchServiceResult>
            directOverlapHandoffResult;
        std::exception_ptr directOverlapHandoffError;
        std::thread directOverlapHandoff([&]
        {
            try
            {
                directOverlapHandoffResult.emplace(
                    EA::CampaignOperations::
                        DispatchOneRequestForIsolatedTest(
                            phase3ConnectionString,
                            directOverlapRequest.request.requestId, 1,
                            EA::CampaignOperations::ActorIdentity(
                                "phase3.direct.overlap@example.test"),
                            phase3Gate,
                            [&](EA::CampaignOperations::
                                    DispatchTestInjectionPoint point)
                            {
                                if (point == EA::CampaignOperations::
                                        DispatchTestInjectionPoint::
                                            beforePhase5Invocation)
                                {
                                    handoffAtPhase5.arrive_and_wait();
                                    allowHandoffPhase5.arrive_and_wait();
                                }
                            }));
            }
            catch (...)
            {
                directOverlapHandoffError = std::current_exception();
            }
        });
        handoffAtPhase5.arrive_and_wait();
        std::atomic<bool> directOverlapFinished{false};
        std::optional<RecommendationCampaignLaunchPersistResult>
            directOverlapLaunchResult;
        std::exception_ptr directOverlapLaunchError;
        std::thread directOverlapLaunch([&]
        {
            try
            {
                directOverlapLaunchResult.emplace(
                    RunLaunch(runtimeString, 231));
                directOverlapFinished = true;
            }
            catch (...)
            {
                directOverlapLaunchError = std::current_exception();
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        const bool directPhase5BypassedHandoffLock =
            directOverlapFinished.load();
        allowHandoffPhase5.arrive_and_wait();
        directOverlapHandoff.join();
        directOverlapLaunch.join();
        if (directOverlapHandoffError)
            std::rethrow_exception(directOverlapHandoffError);
        if (directOverlapLaunchError)
            std::rethrow_exception(directOverlapLaunchError);
        assert(!directPhase5BypassedHandoffLock);
        assert(directOverlapHandoffResult);
        assert(directOverlapHandoffResult->classification ==
            EA::CampaignOperations::DispatchResultClassification::
                createdAndBound);
        assert(directOverlapLaunchResult);
        assert(directOverlapLaunchResult->createdExecutions.empty());
        assert(directOverlapLaunchResult->createdActivations.empty());
        assert(Text(owner, schema,
            "SELECT binding_disposition FROM "
            "campaign_operations_request_binding WHERE "
            "operational_request_id=" +
            std::to_string(
                directOverlapRequest.request.requestId.value())) ==
            "created");

        currentStage = "campaign_operations_phase3_rollback_restart";
        const std::vector<CampaignProposal> rollbackProposals{
            CreateApprovedProposal(runtime, 1.38, 28),
            CreateApprovedProposal(runtime, 1.39, 29)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 19, rollbackProposals);
            fixture.commit();
        }
        const auto rollbackRequest =
            CreatePhase3Request(owner, schema, 19);
        bool injectedRollback = false;
        try
        {
            (void)EA::CampaignOperations::
                DispatchOneRequestForIsolatedTest(
                    phase3ConnectionString,
                    rollbackRequest.request.requestId, 1,
                    EA::CampaignOperations::ActorIdentity(
                        "phase3.dispatcher@example.test"),
                    phase3Gate,
                    [](EA::CampaignOperations::
                           DispatchTestInjectionPoint point)
                    {
                        if (point == EA::CampaignOperations::
                                DispatchTestInjectionPoint::
                                    afterPhase5ExecutionMutation)
                            throw std::runtime_error(
                                "phase3_injected_rollback");
                    });
        }
        catch (const std::runtime_error& error)
        {
            injectedRollback =
                std::string(error.what()) == "phase3_injected_rollback";
        }
        assert(injectedRollback);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM "
            "experiment_recommendation_conversion_execution "
            "WHERE recommendation_conversion_proposal_id IN (" +
            std::to_string(
                rollbackProposals[0].proposal.proposalId) + "," +
            std::to_string(
                rollbackProposals[1].proposal.proposalId) + ")") == 0);
        assert(Text(owner, schema,
            "SELECT request_state FROM "
            "campaign_operations_operational_request "
            "WHERE operational_request_id=" +
            std::to_string(
                rollbackRequest.request.requestId.value())) ==
            "dispatching");
        const auto resumed =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                rollbackRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.dispatcher@example.test"),
                phase3Gate);
        assert(resumed.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                createdAndBound);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM campaign_operations_dispatch_attempt "
            "WHERE operational_request_id=" +
            std::to_string(
                rollbackRequest.request.requestId.value())) == 1);

        currentStage = "campaign_operations_phase3_lost_response";
        const std::vector<CampaignProposal> lostResponseProposals{
            CreateApprovedProposal(runtime, 1.40, 30)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 20, lostResponseProposals);
            fixture.commit();
        }
        const auto lostResponseRequest =
            CreatePhase3Request(owner, schema, 20);
        bool responseLost = false;
        try
        {
            (void)EA::CampaignOperations::
                DispatchOneRequestForIsolatedTest(
                    phase3ConnectionString,
                    lostResponseRequest.request.requestId, 1,
                    EA::CampaignOperations::ActorIdentity(
                        "phase3.dispatcher@example.test"),
                    phase3Gate,
                    [](EA::CampaignOperations::
                           DispatchTestInjectionPoint point)
                    {
                        if (point == EA::CampaignOperations::
                                DispatchTestInjectionPoint::
                                    afterSuccessfulCommitBeforeResponse)
                            throw std::runtime_error(
                                "phase3_response_lost");
                    });
        }
        catch (const std::runtime_error& error)
        {
            responseLost =
                std::string(error.what()) == "phase3_response_lost";
        }
        assert(responseLost);
        const auto recovered =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                lostResponseRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.dispatcher@example.test"),
                phase3Gate);
        assert(recovered.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                existingIdentical);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM campaign_operations_dispatch_attempt "
            "WHERE operational_request_id=" +
            std::to_string(
                lostResponseRequest.request.requestId.value())) == 1);

        currentStage = "campaign_operations_phase3_adoption";
        const std::vector<CampaignProposal> adoptionProposals{
            CreateApprovedProposal(runtime, 1.41, 31)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 21, adoptionProposals);
            fixture.commit();
        }
        (void)RunLaunch(runtimeString, 21);
        const auto adoptionRequest =
            CreatePhase3Request(owner, schema, 21);
        GrantPhase3Adoption(owner, schema, adoptionRequest);
        const auto adopted =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                adoptionRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.dispatcher@example.test"),
                phase3Gate);
        assert(adopted.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                adoptedExistingPendingAndBound);
        assert(Text(owner, schema,
            "SELECT binding_disposition FROM "
            "campaign_operations_request_binding "
            "WHERE operational_request_id=" +
            std::to_string(
                adoptionRequest.request.requestId.value())) ==
            "adopted_existing_pending");

        currentStage = "campaign_operations_phase3_adoption_denied";
        const std::vector<CampaignProposal> unauthorizedAdoptionProposals{
            CreateApprovedProposal(runtime, 1.42, 32)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(
                fixture, 22, unauthorizedAdoptionProposals);
            fixture.commit();
        }
        (void)RunLaunch(runtimeString, 22);
        const auto unauthorizedAdoptionRequest =
            CreatePhase3Request(owner, schema, 22);
        int unauthorizedAdoptionPhase5Invocations = 0;
        const auto adoptionDenied =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                unauthorizedAdoptionRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.dispatcher@example.test"),
                phase3Gate,
                [&](EA::CampaignOperations::
                        DispatchTestInjectionPoint point)
                {
                    if (point == EA::CampaignOperations::
                            DispatchTestInjectionPoint::
                                beforePhase5Invocation)
                        ++unauthorizedAdoptionPhase5Invocations;
                });
        assert(unauthorizedAdoptionPhase5Invocations == 0);
        assert(adoptionDenied.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                reconciliationRequired);
        assert(adoptionDenied.downstreamEvidence ==
            EA::CampaignOperations::DownstreamEvidenceClassification::
                exactCompletePendingTrain);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM campaign_operations_request_binding "
            "WHERE operational_request_id=" +
            std::to_string(
                unauthorizedAdoptionRequest.request.requestId.value())) == 0);
        const auto deniedReplay =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                unauthorizedAdoptionRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.dispatcher@example.test"),
                phase3Gate);
        assert(deniedReplay.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                reconciliationRequired);
        assert(deniedReplay.replayDisposition ==
            EA::CampaignOperations::ExactReplayDisposition::
                authoritativeExisting);

        currentStage =
            "campaign_operations_phase3_adoption_authority_matrix";
        const auto verifyInactiveAdoption =
            [&](long long materializationId, double threshold,
                std::optional<EA::CampaignOperations::
                    AuthorizationEventKind> successor,
                std::optional<EA::CampaignOperations::UtcTimestamp>
                    expiresAt)
        {
            const std::vector<CampaignProposal> proposals{
                CreateApprovedProposal(
                    runtime, threshold, materializationId + 100)};
            {
                pqxx::work fixture{owner};
                SetSearchPath(fixture, schema);
                InsertMaterialization(
                    fixture, materializationId, proposals);
                fixture.commit();
            }
            (void)RunLaunch(runtimeString, materializationId);
            const auto request =
                CreatePhase3Request(owner, schema, materializationId);
            const auto grant = GrantPhase3Adoption(
                owner, schema, request, std::move(expiresAt));
            if (successor)
                RecordPhase3AdoptionAuthorizationSuccessor(
                    owner, schema, request, grant, *successor);
            const auto result =
                EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                    phase3ConnectionString,
                    request.request.requestId, 1,
                    EA::CampaignOperations::ActorIdentity(
                        "phase3.inactive.adoption@example.test"),
                    phase3Gate);
            assert(result.classification ==
                EA::CampaignOperations::DispatchResultClassification::
                    reconciliationRequired);
            assert(result.downstreamEvidence ==
                EA::CampaignOperations::DownstreamEvidenceClassification::
                    exactCompletePendingTrain);
            assert(result.diagnosticCode ==
                "dispatch_adoption_authorization_required");
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM "
                "campaign_operations_request_binding WHERE "
                "operational_request_id=" +
                std::to_string(request.request.requestId.value())) == 0);
            assert(Text(owner, schema,
                "SELECT reservation_state FROM "
                "campaign_operations_reservation WHERE reservation_id=" +
                std::to_string(
                    request.reservation.reservationId.value())) == "held");
        };
        verifyInactiveAdoption(
            217, 3.82,
            EA::CampaignOperations::AuthorizationEventKind::revoked,
            std::nullopt);
        verifyInactiveAdoption(
            218, 3.83,
            EA::CampaignOperations::AuthorizationEventKind::expiryObserved,
            std::nullopt);
        verifyInactiveAdoption(
            219, 3.84, std::nullopt,
            EA::CampaignOperations::UtcTimestamp(
                "2021-01-01T00:00:00.000000Z"));

        currentStage = "campaign_operations_phase3_paused_refusal";
        const std::vector<CampaignProposal> pausedOnlyProposals{
            CreateApprovedProposal(runtime, 1.43, 33)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 23, pausedOnlyProposals);
            fixture.commit();
        }
        (void)Execute(runtime, pausedOnlyProposals.front());
        const auto pausedOnlyRequest =
            CreatePhase3Request(owner, schema, 23);
        const auto pausedRefused =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                pausedOnlyRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.dispatcher@example.test"),
                phase3Gate);
        assert(pausedRefused.downstreamEvidence ==
            EA::CampaignOperations::DownstreamEvidenceClassification::
                pausedOnlyEvidence);
        assert(pausedRefused.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                reconciliationRequired);

        currentStage = "campaign_operations_phase3_partial_refusal";
        const std::vector<CampaignProposal> partialProposals{
            CreateApprovedProposal(runtime, 1.44, 34),
            CreateApprovedProposal(runtime, 1.45, 35)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 24, partialProposals);
            fixture.commit();
        }
        (void)Execute(runtime, partialProposals.front());
        const auto partialRequest =
            CreatePhase3Request(owner, schema, 24);
        const auto partialRefused =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                partialRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.dispatcher@example.test"),
                phase3Gate);
        assert(partialRefused.downstreamEvidence ==
            EA::CampaignOperations::DownstreamEvidenceClassification::
                partialPhase5Evidence);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM "
            "experiment_recommendation_conversion_execution "
            "WHERE recommendation_conversion_proposal_id IN (" +
            std::to_string(partialProposals[0].proposal.proposalId) + "," +
            std::to_string(partialProposals[1].proposal.proposalId) + ")") ==
            1);

        currentStage = "campaign_operations_phase3_acquisition_rollback_matrix";
        const std::vector<EA::CampaignOperations::DispatchTestInjectionPoint>
            acquisitionRollbackPoints{
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    beforeRequestLeaseTransition,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    afterRequestLeaseTransitionBeforeAttempt,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    afterAttemptBeforeAcquisitionAudit,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    beforeAcquisitionCommit};
        for (std::size_t index = 0;
             index < acquisitionRollbackPoints.size(); ++index)
        {
            const std::vector<CampaignProposal> proposals{
                CreateApprovedProposal(runtime, 1.50 + index / 100.0,
                    40 + static_cast<int>(index))};
            const long long materializationId =
                100 + static_cast<long long>(index);
            {
                pqxx::work fixture{owner};
                SetSearchPath(fixture, schema);
                InsertMaterialization(
                    fixture, materializationId, proposals);
                fixture.commit();
            }
            const auto request =
                CreatePhase3Request(owner, schema, materializationId);
            bool injected = false;
            try
            {
                (void)EA::CampaignOperations::
                    DispatchOneRequestForIsolatedTest(
                        phase3ConnectionString, request.request.requestId, 1,
                        EA::CampaignOperations::ActorIdentity(
                            "phase3.rollback.dispatcher@example.test"),
                        phase3Gate,
                        [target = acquisitionRollbackPoints[index]](
                            EA::CampaignOperations::
                                DispatchTestInjectionPoint point)
                        {
                            if (point == target)
                                throw std::runtime_error(
                                    "phase3_acquisition_rollback");
                        });
            }
            catch (const std::runtime_error& error)
            {
                injected = std::string(error.what()) ==
                    "phase3_acquisition_rollback";
            }
            assert(injected);
            const auto requestId =
                std::to_string(request.request.requestId.value());
            assert(Text(owner, schema,
                "SELECT request_state||':'||state_version::text||':'||"
                "coalesce(lease_token_hash,'NULL') FROM "
                "campaign_operations_operational_request WHERE "
                "operational_request_id=" + requestId) ==
                "ready:1:NULL");
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM campaign_operations_dispatch_attempt "
                "WHERE operational_request_id=" + requestId) == 0);
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM "
                "campaign_operations_dispatch_audit_reference_event "
                "WHERE operational_request_id=" + requestId) == 0);
            assert(Text(owner, schema,
                "SELECT reservation_state||':'||state_version::text FROM "
                "campaign_operations_reservation WHERE reservation_id=" +
                std::to_string(request.reservation.reservationId.value())) ==
                "held:1");
        }

        currentStage = "campaign_operations_phase3_handoff_rollback_matrix";
        const std::vector<EA::CampaignOperations::DispatchTestInjectionPoint>
            handoffRollbackPoints{
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    beforeDownstreamEvidenceClassification,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    beforePhase5Invocation,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    afterPhase5ExecutionMutation,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    afterPhase5ActivationMutation,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    afterExperimentCreationOrReuseMutation,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    duringBindingInsertion,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    afterBindingSubset,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    duringControlOwnerInsertion,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    afterControlOwnerSubset,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    beforeReservationCommitmentEventInsertion,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    afterReservationCommitmentEventBeforeProjection,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    afterReservationProjectionBeforeRequestTransition,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    afterRequestTransitionBeforeAttemptOutcome,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    afterAttemptOutcomeBeforeAudit,
                EA::CampaignOperations::DispatchTestInjectionPoint::
                    beforeHandoffCommit};
        for (std::size_t index = 0;
             index < handoffRollbackPoints.size(); ++index)
        {
            const std::vector<CampaignProposal> proposals{
                CreateApprovedProposal(runtime, 1.60 + index / 100.0,
                    50 + static_cast<int>(index)),
                CreateApprovedProposal(runtime, 1.80 + index / 100.0,
                    80 + static_cast<int>(index))};
            const long long materializationId =
                120 + static_cast<long long>(index);
            {
                pqxx::work fixture{owner};
                SetSearchPath(fixture, schema);
                InsertMaterialization(
                    fixture, materializationId, proposals);
                fixture.commit();
            }
            const auto request =
                CreatePhase3Request(owner, schema, materializationId);
            bool injected = false;
            try
            {
                (void)EA::CampaignOperations::
                    DispatchOneRequestForIsolatedTest(
                        phase3ConnectionString, request.request.requestId, 1,
                        EA::CampaignOperations::ActorIdentity(
                            "phase3.rollback.dispatcher@example.test"),
                        phase3Gate,
                        [target = handoffRollbackPoints[index]](
                            EA::CampaignOperations::
                                DispatchTestInjectionPoint point)
                        {
                            if (point == target)
                                throw std::runtime_error(
                                    "phase3_handoff_rollback");
                        });
            }
            catch (const std::runtime_error& error)
            {
                injected = std::string(error.what()) ==
                    "phase3_handoff_rollback";
            }
            assert(injected);
            const auto requestId =
                std::to_string(request.request.requestId.value());
            assert(Text(owner, schema,
                "SELECT request_state||':'||state_version::text||':'||"
                "(lease_token_hash IS NOT NULL)::text FROM "
                "campaign_operations_operational_request WHERE "
                "operational_request_id=" + requestId) ==
                "dispatching:2:true");
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM campaign_operations_dispatch_attempt "
                "WHERE operational_request_id=" + requestId) == 1);
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM "
                "campaign_operations_dispatch_audit_reference_event "
                "WHERE operational_request_id=" + requestId) == 1);
            for (const char* table : {
                     "campaign_operations_request_binding",
                     "campaign_operations_downstream_control_owner",
                     "campaign_operations_reservation_commitment"})
                assert(Scalar(owner, schema,
                    "SELECT count(*) FROM " + std::string(table) +
                    " WHERE operational_request_id=" + requestId) == 0);
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM "
                "campaign_operations_dispatch_attempt_outcome outcome "
                "JOIN campaign_operations_dispatch_attempt attempt USING "
                "(dispatch_attempt_id) WHERE "
                "attempt.operational_request_id=" + requestId) == 0);
            assert(Text(owner, schema,
                "SELECT reservation_state||':'||state_version::text FROM "
                "campaign_operations_reservation WHERE reservation_id=" +
                std::to_string(request.reservation.reservationId.value())) ==
                "held:1");
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM "
                "experiment_recommendation_conversion_execution WHERE "
                "recommendation_conversion_proposal_id IN (" +
                std::to_string(proposals[0].proposal.proposalId) + "," +
                std::to_string(proposals[1].proposal.proposalId) + ")") == 0);
        }

        currentStage =
            "campaign_operations_phase3_acquisition_retry";
        const std::vector<CampaignProposal> acquisitionRetryProposals{
            CreateApprovedProposal(runtime, 4.12, 402)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(
                fixture, 232, acquisitionRetryProposals);
            fixture.commit();
        }
        const auto acquisitionRetryRequest =
            CreatePhase3Request(owner, schema, 232);
        int acquisitionRetryInvocations = 0;
        const auto acquisitionRetried =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                acquisitionRetryRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.acquisition.retry@example.test"),
                phase3Gate,
                [&](EA::CampaignOperations::
                        DispatchTestInjectionPoint point)
                {
                    if (point == EA::CampaignOperations::
                            DispatchTestInjectionPoint::
                                afterRequestLeaseTransitionBeforeAttempt &&
                        ++acquisitionRetryInvocations < 3)
                        throw EA::CampaignOperations::
                            DispatchTestSqlState("40P01");
                });
        assert(acquisitionRetryInvocations == 3);
        assert(acquisitionRetried.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                createdAndBound);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM campaign_operations_dispatch_attempt "
            "WHERE operational_request_id=" +
            std::to_string(
                acquisitionRetryRequest.request.requestId.value())) == 1);

        currentStage =
            "campaign_operations_phase3_acquisition_retry_exhaustion";
        const std::vector<CampaignProposal> acquisitionExhaustedProposals{
            CreateApprovedProposal(runtime, 4.13, 403)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(
                fixture, 233, acquisitionExhaustedProposals);
            fixture.commit();
        }
        const auto acquisitionExhaustedRequest =
            CreatePhase3Request(owner, schema, 233);
        int acquisitionExhaustedInvocations = 0;
        const auto acquisitionExhausted =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                acquisitionExhaustedRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.acquisition.exhausted@example.test"),
                phase3Gate,
                [&](EA::CampaignOperations::
                        DispatchTestInjectionPoint point)
                {
                    if (point == EA::CampaignOperations::
                            DispatchTestInjectionPoint::
                                afterRequestLeaseTransitionBeforeAttempt)
                    {
                        ++acquisitionExhaustedInvocations;
                        throw EA::CampaignOperations::
                            DispatchTestSqlState("40001");
                    }
                });
        assert(acquisitionExhaustedInvocations == 3);
        assert(acquisitionExhausted.failure ==
            EA::CampaignOperations::DispatchServiceFailureClassification::
                transientDatabaseRetryExhausted);
        assert(acquisitionExhausted.transactionAttempts == 3);
        assert(acquisitionExhausted.diagnosticCode ==
            "dispatch_retry_exhausted_40001");
        assert(Text(owner, schema,
            "SELECT request_state||':'||state_version::text FROM "
            "campaign_operations_operational_request WHERE "
            "operational_request_id=" +
            std::to_string(
                acquisitionExhaustedRequest.request.requestId.value())) ==
            "ready:1");
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM campaign_operations_dispatch_attempt "
            "WHERE operational_request_id=" +
            std::to_string(
                acquisitionExhaustedRequest.request.requestId.value())) == 0);

        currentStage = "campaign_operations_phase3_whole_operation_retry";
        for (const std::string sqlState : {"40001", "40P01"})
        {
            static long long retryMaterializationId = 160;
            const std::vector<CampaignProposal> proposals{
                CreateApprovedProposal(runtime,
                    2.10 + retryMaterializationId / 1000.0,
                    static_cast<int>(retryMaterializationId)),
                CreateApprovedProposal(runtime,
                    2.40 + retryMaterializationId / 1000.0,
                    static_cast<int>(retryMaterializationId + 1))};
            {
                pqxx::work fixture{owner};
                SetSearchPath(fixture, schema);
                InsertMaterialization(
                    fixture, retryMaterializationId, proposals);
                fixture.commit();
            }
            const auto request = CreatePhase3Request(
                owner, schema, retryMaterializationId++);
            int phase5Invocations = 0;
            int authoritativeReloads = 0;
            const auto result =
                EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                    phase3ConnectionString, request.request.requestId, 1,
                    EA::CampaignOperations::ActorIdentity(
                        "phase3.retry.dispatcher@example.test"),
                    phase3Gate,
                    [&](EA::CampaignOperations::
                            DispatchTestInjectionPoint point)
                    {
                        if (point == EA::CampaignOperations::
                                DispatchTestInjectionPoint::
                                    beforeDownstreamEvidenceClassification)
                            ++authoritativeReloads;
                        if (point == EA::CampaignOperations::
                                DispatchTestInjectionPoint::
                                    afterPhase5ExecutionMutation)
                        {
                            ++phase5Invocations;
                            if (phase5Invocations < 3)
                                throw EA::CampaignOperations::
                                    DispatchTestSqlState(sqlState);
                        }
                    });
            assert(result.classification ==
                EA::CampaignOperations::DispatchResultClassification::
                    createdAndBound);
            assert(result.transactionAttempts == 3);
            assert(phase5Invocations == 3);
            assert(authoritativeReloads == 3);
            const auto requestId =
                std::to_string(request.request.requestId.value());
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM campaign_operations_dispatch_attempt "
                "WHERE operational_request_id=" + requestId) == 1);
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM campaign_operations_request_binding "
                "WHERE operational_request_id=" + requestId) == 2);
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM "
                "experiment_recommendation_conversion_execution WHERE "
                "recommendation_conversion_proposal_id IN (" +
                std::to_string(proposals[0].proposal.proposalId) + "," +
                std::to_string(proposals[1].proposal.proposalId) + ")") == 2);
        }

        currentStage =
            "campaign_operations_phase3_unknown_commit_proven_absent_retry";
        const std::vector<CampaignProposal> unknownAbsentProposals{
            CreateApprovedProposal(runtime, 2.79, 320)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(
                fixture, 220, unknownAbsentProposals);
            fixture.commit();
        }
        const auto unknownAbsentRequest =
            CreatePhase3Request(owner, schema, 220);
        int unknownAbsentPhase5Invocations = 0;
        int unknownAbsentFailures = 0;
        const auto unknownAbsentRecovered =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                unknownAbsentRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.unknown.absent@example.test"),
                phase3Gate,
                [&](EA::CampaignOperations::
                        DispatchTestInjectionPoint point)
                {
                    if (point == EA::CampaignOperations::
                            DispatchTestInjectionPoint::
                                beforePhase5Invocation)
                        ++unknownAbsentPhase5Invocations;
                    if (point == EA::CampaignOperations::
                            DispatchTestInjectionPoint::
                                beforeHandoffCommit &&
                        unknownAbsentFailures++ < 2)
                        throw pqxx::broken_connection(
                            "phase3 deterministic lost connection");
                });
        assert(unknownAbsentRecovered.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                createdAndBound);
        assert(unknownAbsentRecovered.transactionAttempts == 3);
        assert(unknownAbsentPhase5Invocations == 3);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM "
            "experiment_recommendation_conversion_execution WHERE "
            "recommendation_conversion_proposal_id=" +
            std::to_string(
                unknownAbsentProposals.front().proposal.proposalId)) == 1);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM campaign_operations_request_binding "
            "WHERE operational_request_id=" +
            std::to_string(
                unknownAbsentRequest.request.requestId.value())) == 1);

        currentStage = "campaign_operations_phase3_retry_exhaustion";
        const std::vector<CampaignProposal> exhaustedProposals{
            CreateApprovedProposal(runtime, 2.81, 181)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 180, exhaustedProposals);
            fixture.commit();
        }
        const auto exhaustedRequest =
            CreatePhase3Request(owner, schema, 180);
        int exhaustedInvocations = 0;
        const auto exhausted =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                exhaustedRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.retry.dispatcher@example.test"),
                phase3Gate,
                [&](EA::CampaignOperations::
                        DispatchTestInjectionPoint point)
                {
                    if (point == EA::CampaignOperations::
                            DispatchTestInjectionPoint::
                                afterPhase5ExecutionMutation)
                    {
                        ++exhaustedInvocations;
                        throw EA::CampaignOperations::
                            DispatchTestSqlState("40001");
                    }
                });
        assert(exhausted.failure == EA::CampaignOperations::
            DispatchServiceFailureClassification::
                transientDatabaseRetryExhausted);
        assert(exhausted.transactionAttempts == 3);
        assert(exhaustedInvocations == 3);
        assert(exhausted.diagnosticCode ==
            "dispatch_retry_exhausted_40001");
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM "
            "experiment_recommendation_conversion_execution WHERE "
            "recommendation_conversion_proposal_id=" +
            std::to_string(
                exhaustedProposals.front().proposal.proposalId)) == 0);

        currentStage = "campaign_operations_phase3_nonretryable_sql";
        const std::vector<CampaignProposal> nonretryableProposals{
            CreateApprovedProposal(runtime, 2.82, 182)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 181, nonretryableProposals);
            fixture.commit();
        }
        const auto nonretryableRequest =
            CreatePhase3Request(owner, schema, 181);
        bool nonretryableObserved = false;
        try
        {
            (void)EA::CampaignOperations::
                DispatchOneRequestForIsolatedTest(
                    phase3ConnectionString,
                    nonretryableRequest.request.requestId, 1,
                    EA::CampaignOperations::ActorIdentity(
                        "phase3.retry.dispatcher@example.test"),
                    phase3Gate,
                    [](EA::CampaignOperations::
                           DispatchTestInjectionPoint point)
                    {
                        if (point == EA::CampaignOperations::
                                DispatchTestInjectionPoint::
                                    afterPhase5ExecutionMutation)
                            throw EA::CampaignOperations::
                                DispatchTestSqlState("22000");
                    });
        }
        catch (const pqxx::sql_error& error)
        {
            nonretryableObserved = error.sqlstate() == "22000";
        }
        assert(nonretryableObserved);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM "
            "experiment_recommendation_conversion_execution WHERE "
            "recommendation_conversion_proposal_id=" +
            std::to_string(
                nonretryableProposals.front().proposal.proposalId)) == 0);

        currentStage = "campaign_operations_phase3_authorization_races";
        const std::vector<CampaignProposal> authorizationRaceProposals{
            CreateApprovedProposal(runtime, 2.90, 190)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(
                fixture, 190, authorizationRaceProposals);
            fixture.commit();
        }
        const auto authorizationRaceRequest =
            CreatePhase3Request(owner, schema, 190);
        std::barrier authorizationLocked{2};
        std::barrier allowAuthorizationCommit{2};
        std::optional<EA::CampaignOperations::DispatchServiceResult>
            authorizationRaceResult;
        std::exception_ptr authorizationRaceError;
        std::thread authorizationHandoff([&]
        {
            try
            {
                authorizationRaceResult.emplace(
                    EA::CampaignOperations::
                        DispatchOneRequestForIsolatedTest(
                            phase3ConnectionString,
                            authorizationRaceRequest.request.requestId, 1,
                            EA::CampaignOperations::ActorIdentity(
                                "phase3.authorization.race@example.test"),
                            phase3Gate,
                            [&](EA::CampaignOperations::
                                    DispatchTestInjectionPoint point)
                            {
                                if (point == EA::CampaignOperations::
                                        DispatchTestInjectionPoint::
                                            beforeDownstreamEvidenceClassification)
                                {
                                    authorizationLocked.arrive_and_wait();
                                    allowAuthorizationCommit.arrive_and_wait();
                                }
                            }));
            }
            catch (...)
            {
                authorizationRaceError = std::current_exception();
            }
        });
        authorizationLocked.arrive_and_wait();
        std::atomic<bool> revocationFinished{false};
        std::exception_ptr revocationError;
        std::thread revocation([&]
        {
            try
            {
                pqxx::connection connection{ownerString};
                RecordPhase3DispatchAuthorizationSuccessor(
                    connection, schema, authorizationRaceRequest,
                    EA::CampaignOperations::AuthorizationEventKind::revoked);
                revocationFinished = true;
            }
            catch (...)
            {
                revocationError = std::current_exception();
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        assert(!revocationFinished);
        allowAuthorizationCommit.arrive_and_wait();
        authorizationHandoff.join();
        revocation.join();
        if (authorizationRaceError)
            std::rethrow_exception(authorizationRaceError);
        if (revocationError) std::rethrow_exception(revocationError);
        assert(authorizationRaceResult);
        assert(authorizationRaceResult->classification ==
            EA::CampaignOperations::DispatchResultClassification::
                createdAndBound);
        assert(revocationFinished);
        assert(Text(owner, schema,
            "SELECT request_state FROM "
            "campaign_operations_operational_request WHERE "
            "operational_request_id=" +
            std::to_string(
                authorizationRaceRequest.request.requestId.value())) ==
            "bound");

        for (const auto [materializationId, kind] :
             std::array<std::pair<long long,
                 EA::CampaignOperations::AuthorizationEventKind>, 2>{
                 std::pair{191LL, EA::CampaignOperations::
                     AuthorizationEventKind::revoked},
                 std::pair{192LL, EA::CampaignOperations::
                     AuthorizationEventKind::granted}})
        {
            const std::vector<CampaignProposal> proposals{
                CreateApprovedProposal(runtime,
                    2.90 + materializationId / 1000.0,
                    static_cast<int>(materializationId))};
            {
                pqxx::work fixture{owner};
                SetSearchPath(fixture, schema);
                InsertMaterialization(
                    fixture, materializationId, proposals);
                fixture.commit();
            }
            const auto request =
                CreatePhase3Request(owner, schema, materializationId);
            RecordPhase3DispatchAuthorizationSuccessor(
                owner, schema, request, kind);
            bool denied = false;
            try
            {
                (void)EA::CampaignOperations::
                    DispatchOneRequestForIsolatedTest(
                        phase3ConnectionString, request.request.requestId, 1,
                        EA::CampaignOperations::ActorIdentity(
                            "phase3.authorization.loser@example.test"),
                        phase3Gate);
            }
            catch (const std::runtime_error& error)
            {
                denied = std::string(error.what()).find(
                    "dispatch_authorization_not_head") !=
                    std::string::npos;
            }
            assert(denied);
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM "
                "campaign_operations_dispatch_attempt WHERE "
                "operational_request_id=" +
                std::to_string(request.request.requestId.value())) == 0);
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM "
                "experiment_recommendation_conversion_execution WHERE "
                "recommendation_conversion_proposal_id=" +
                std::to_string(proposals.front().proposal.proposalId)) == 0);
        }

        const auto expiringAt = [&]
        {
            pqxx::read_transaction transaction{owner};
            SetSearchPath(transaction, schema);
            return EA::CampaignOperations::UtcTimestamp(
                transaction.exec(
                    "SELECT to_char((clock_timestamp()+"
                    "interval '1.5 seconds') AT TIME ZONE 'UTC',"
                    "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"');")
                    .one_row()[0].as<std::string>());
        }();
        const std::vector<CampaignProposal> expiryHandoffProposals{
            CreateApprovedProposal(runtime, 2.99, 199)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(
                fixture, 193, expiryHandoffProposals);
            fixture.commit();
        }
        const auto expiryHandoffRequest =
            CreatePhase3Request(owner, schema, 193, expiringAt);
        std::barrier expiryValidated{2};
        std::barrier allowExpiredTransactionCommit{2};
        std::optional<EA::CampaignOperations::DispatchServiceResult>
            expiryHandoffResult;
        std::exception_ptr expiryHandoffError;
        std::thread expiryHandoff([&]
        {
            try
            {
                expiryHandoffResult.emplace(
                    EA::CampaignOperations::
                        DispatchOneRequestForIsolatedTest(
                            phase3ConnectionString,
                            expiryHandoffRequest.request.requestId, 1,
                            EA::CampaignOperations::ActorIdentity(
                                "phase3.expiry.race@example.test"),
                            phase3Gate,
                            [&](EA::CampaignOperations::
                                    DispatchTestInjectionPoint point)
                            {
                                if (point == EA::CampaignOperations::
                                        DispatchTestInjectionPoint::
                                            beforeDownstreamEvidenceClassification)
                                {
                                    expiryValidated.arrive_and_wait();
                                    allowExpiredTransactionCommit.
                                        arrive_and_wait();
                                }
                            }));
            }
            catch (...)
            {
                expiryHandoffError = std::current_exception();
            }
        });
        expiryValidated.arrive_and_wait();
        std::this_thread::sleep_for(std::chrono::milliseconds{1700});
        allowExpiredTransactionCommit.arrive_and_wait();
        expiryHandoff.join();
        if (expiryHandoffError)
            std::rethrow_exception(expiryHandoffError);
        assert(expiryHandoffResult);
        assert(expiryHandoffResult->classification ==
            EA::CampaignOperations::DispatchResultClassification::
                createdAndBound);

        const auto alreadyExpiredAt = [&]
        {
            pqxx::read_transaction transaction{owner};
            SetSearchPath(transaction, schema);
            return EA::CampaignOperations::UtcTimestamp(
                transaction.exec(
                    "SELECT to_char((clock_timestamp()+"
                    "interval '1 second') AT TIME ZONE 'UTC',"
                    "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"');")
                    .one_row()[0].as<std::string>());
        }();
        const std::vector<CampaignProposal> expiryDeniedProposals{
            CreateApprovedProposal(runtime, 3.00, 200)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 194, expiryDeniedProposals);
            fixture.commit();
        }
        const auto expiryDeniedRequest =
            CreatePhase3Request(owner, schema, 194, alreadyExpiredAt);
        std::this_thread::sleep_for(std::chrono::milliseconds{1200});
        bool expiryDenied = false;
        try
        {
            (void)EA::CampaignOperations::
                DispatchOneRequestForIsolatedTest(
                    phase3ConnectionString,
                    expiryDeniedRequest.request.requestId, 1,
                    EA::CampaignOperations::ActorIdentity(
                        "phase3.expiry.loser@example.test"),
                    phase3Gate);
        }
        catch (const std::runtime_error& error)
        {
            expiryDenied = std::string(error.what()).find(
                "dispatch_authorization_inactive") != std::string::npos;
        }
        assert(expiryDenied);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM "
            "experiment_recommendation_conversion_execution WHERE "
            "recommendation_conversion_proposal_id=" +
            std::to_string(
                expiryDeniedProposals.front().proposal.proposalId)) == 0);

        currentStage = "campaign_operations_phase3_budget_races";
        const std::string budgetAdministratorConnectionString =
            ownerString + " options='-c search_path=" + schema +
            " -c role=campaign_operations_budget_administrator'";
        const std::vector<CampaignProposal> budgetRaceProposals{
            CreateApprovedProposal(runtime, 3.01, 201)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 200, budgetRaceProposals);
            fixture.commit();
        }
        const auto budgetRaceRequest =
            CreatePhase3Request(owner, schema, 200);
        std::barrier budgetLocked{2};
        std::barrier allowBudgetCommit{2};
        std::optional<EA::CampaignOperations::DispatchServiceResult>
            budgetRaceResult;
        std::exception_ptr budgetRaceError;
        std::thread budgetHandoff([&]
        {
            try
            {
                budgetRaceResult.emplace(
                    EA::CampaignOperations::
                        DispatchOneRequestForIsolatedTest(
                            phase3ConnectionString,
                            budgetRaceRequest.request.requestId, 1,
                            EA::CampaignOperations::ActorIdentity(
                                "phase3.budget.race@example.test"),
                            phase3Gate,
                            [&](EA::CampaignOperations::
                                    DispatchTestInjectionPoint point)
                            {
                                if (point == EA::CampaignOperations::
                                        DispatchTestInjectionPoint::
                                            beforeDownstreamEvidenceClassification)
                                {
                                    budgetLocked.arrive_and_wait();
                                    allowBudgetCommit.arrive_and_wait();
                                }
                            }));
            }
            catch (...)
            {
                budgetRaceError = std::current_exception();
            }
        });
        budgetLocked.arrive_and_wait();
        std::atomic<bool> budgetAmendFinished{false};
        std::exception_ptr budgetAmendError;
        std::thread budgetAmend([&]
        {
            try
            {
                pqxx::connection connection{
                    budgetAdministratorConnectionString};
                (void)EA::CampaignOperations::AdministerCampaignBudget(
                    connection,
                    {budgetRaceRequest.request.request.logicalOperation.
                         campaignId.value(),
                     1, EA::CampaignOperations::
                            BudgetLedgerEntryKind::amend,
                     1, "phase3.budget.race@example.test",
                     "Amend budget after atomic handoff winner."});
                budgetAmendFinished = true;
            }
            catch (...)
            {
                budgetAmendError = std::current_exception();
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        assert(!budgetAmendFinished);
        allowBudgetCommit.arrive_and_wait();
        budgetHandoff.join();
        budgetAmend.join();
        if (budgetRaceError) std::rethrow_exception(budgetRaceError);
        if (budgetAmendError) std::rethrow_exception(budgetAmendError);
        assert(budgetRaceResult);
        assert(budgetRaceResult->classification ==
            EA::CampaignOperations::DispatchResultClassification::
                createdAndBound);
        assert(budgetAmendFinished);
        {
            pqxx::connection connection{
                budgetAdministratorConnectionString};
            (void)EA::CampaignOperations::AdministerCampaignBudget(
                connection,
                {budgetRaceRequest.request.request.logicalOperation.
                     campaignId.value(),
                 2, EA::CampaignOperations::BudgetLedgerEntryKind::revoke,
                 std::nullopt,
                 "phase3.budget.race@example.test",
                 "Revoke budget after committed handoff."});
        }
        assert(Text(owner, schema,
            "SELECT reservation_state FROM "
            "campaign_operations_reservation WHERE reservation_id=" +
            std::to_string(
                budgetRaceRequest.reservation.reservationId.value())) ==
            "committed");

        for (long long materializationId : {201LL, 202LL})
        {
            const std::vector<CampaignProposal> proposals{
                CreateApprovedProposal(runtime,
                    3.10 + materializationId / 1000.0,
                    static_cast<int>(materializationId))};
            {
                pqxx::work fixture{owner};
                SetSearchPath(fixture, schema);
                InsertMaterialization(
                    fixture, materializationId, proposals);
                fixture.commit();
            }
            const auto request =
                CreatePhase3Request(owner, schema, materializationId);
            pqxx::connection connection{
                budgetAdministratorConnectionString};
            if (materializationId == 201)
            {
                (void)EA::CampaignOperations::AdministerCampaignBudget(
                    connection,
                    {request.request.request.logicalOperation.campaignId.
                         value(),
                     1, EA::CampaignOperations::
                            BudgetLedgerEntryKind::amend,
                     1, "phase3.budget.mutation@example.test",
                     "Amend budget before handoff."});
            }
            else
            {
                (void)EA::CampaignOperations::AdministerCampaignBudget(
                    connection,
                    {request.request.request.logicalOperation.campaignId.
                         value(),
                     1, EA::CampaignOperations::
                            BudgetLedgerEntryKind::revoke,
                     std::nullopt,
                     "phase3.budget.mutation@example.test",
                     "Revoke budget before handoff."});
                (void)EA::CampaignOperations::AdministerCampaignBudget(
                    connection,
                    {request.request.request.logicalOperation.campaignId.
                         value(),
                     2, EA::CampaignOperations::
                            BudgetLedgerEntryKind::supersede,
                     request.reservation.reservation.amount,
                     "phase3.budget.mutation@example.test",
                     "Supersede revoked budget without lending authority."});
            }
            bool denied = false;
            try
            {
                (void)EA::CampaignOperations::
                    DispatchOneRequestForIsolatedTest(
                        phase3ConnectionString, request.request.requestId, 1,
                        EA::CampaignOperations::ActorIdentity(
                            "phase3.budget.loser@example.test"),
                        phase3Gate);
            }
            catch (const std::runtime_error& error)
            {
                denied = std::string(error.what()).find(
                    "dispatch_budget_inactive") != std::string::npos;
            }
            assert(denied);
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM "
                "experiment_recommendation_conversion_execution WHERE "
                "recommendation_conversion_proposal_id=" +
                std::to_string(proposals.front().proposal.proposalId)) == 0);
        }

        currentStage = "campaign_operations_phase3_progressed_evidence";
        for (const auto [materializationId, status] :
             std::array<std::pair<long long, const char*>, 2>{
                 std::pair{210LL, "running"},
                 std::pair{211LL, "completed"}})
        {
            const std::vector<CampaignProposal> proposals{
                CreateApprovedProposal(runtime,
                    3.40 + materializationId / 1000.0,
                    static_cast<int>(materializationId))};
            {
                pqxx::work fixture{owner};
                SetSearchPath(fixture, schema);
                InsertMaterialization(
                    fixture, materializationId, proposals);
                fixture.commit();
            }
            const auto direct = RunLaunch(
                runtimeString, materializationId);
            assert(direct.plan.members.size() == 1);
            const auto experimentId =
                direct.plan.members.front().experimentId;
            assert(experimentId);
            {
                pqxx::work fixture{owner};
                SetSearchPath(fixture, schema);
                fixture.exec(
                    "UPDATE experiment SET status=$1,phase='train' "
                    "WHERE experiment_id=$2;",
                    pqxx::params{status, *experimentId});
                fixture.commit();
            }
            const auto request =
                CreatePhase3Request(owner, schema, materializationId);
            GrantPhase3Adoption(owner, schema, request);
            const auto refused =
                EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                    phase3ConnectionString, request.request.requestId, 1,
                    EA::CampaignOperations::ActorIdentity(
                        "phase3.progressed@example.test"),
                    phase3Gate);
            assert(refused.classification ==
                EA::CampaignOperations::DispatchResultClassification::
                    reconciliationRequired);
            assert(refused.downstreamEvidence ==
                EA::CampaignOperations::DownstreamEvidenceClassification::
                    progressedUnboundEvidence);
            assert(Scalar(owner, schema,
                "SELECT count(*) FROM campaign_operations_request_binding "
                "WHERE operational_request_id=" +
                std::to_string(request.request.requestId.value())) == 0);
            assert(Text(owner, schema,
                "SELECT reservation_state FROM "
                "campaign_operations_reservation WHERE reservation_id=" +
                std::to_string(request.reservation.reservationId.value())) ==
                "held");
        }

        currentStage = "campaign_operations_phase3_mixed_evidence";
        const std::vector<CampaignProposal> mixedEvidenceProposals{
            CreateApprovedProposal(runtime, 3.62, 212),
            CreateApprovedProposal(runtime, 3.63, 213)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(
                fixture, 212, mixedEvidenceProposals);
            fixture.commit();
        }
        const auto mixedExecutions = RunStep1(runtimeString, 212);
        assert(mixedExecutions.size() == 2);
        {
            pqxx::work activation{runtime};
            (void)ActivateRecommendationConversionExecutionInTransaction(
                activation, mixedExecutions.front().executionId);
            activation.commit();
        }
        const auto mixedRequest =
            CreatePhase3Request(owner, schema, 212);
        GrantPhase3Adoption(owner, schema, mixedRequest);
        const auto mixedRefused =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString, mixedRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.mixed@example.test"),
                phase3Gate);
        assert(mixedRefused.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                reconciliationRequired);
        assert(mixedRefused.downstreamEvidence ==
            EA::CampaignOperations::DownstreamEvidenceClassification::
                partialPhase5Evidence);

        currentStage = "campaign_operations_phase3_control_owner_collision";
        const std::vector<CampaignProposal> ownershipProposals{
            CreateApprovedProposal(runtime, 3.70, 214)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(fixture, 214, ownershipProposals);
            InsertMaterialization(fixture, 215, ownershipProposals);
            fixture.commit();
        }
        const auto owningRequest =
            CreatePhase3Request(owner, schema, 214);
        const auto owningResult =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString, owningRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.owner@example.test"),
                phase3Gate);
        assert(owningResult.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                createdAndBound);
        const auto collidingRequest =
            CreatePhase3Request(owner, schema, 215);
        GrantPhase3Adoption(owner, schema, collidingRequest);
        const auto collision =
            EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
                phase3ConnectionString,
                collidingRequest.request.requestId, 1,
                EA::CampaignOperations::ActorIdentity(
                    "phase3.collision@example.test"),
                phase3Gate);
        assert(collision.classification ==
            EA::CampaignOperations::DispatchResultClassification::
                reconciliationRequired);
        assert(collision.diagnosticCode ==
            "dispatch_control_owner_collision");
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM campaign_operations_request_binding "
            "WHERE operational_request_id=" +
            std::to_string(
                collidingRequest.request.requestId.value())) == 0);

        currentStage = "campaign_operations_phase3_corruption_matrix";
        const auto assertBindingCorruptionRejected =
            [&](EA::CampaignOperations::OperationalRequestId requestId,
                const std::string& corruption)
        {
            pqxx::work transaction{owner};
            SetSearchPath(transaction, schema);
            transaction.exec(corruption);
            bool rejected = false;
            try
            {
                (void)EA::CampaignOperations::
                    FindAndValidateCompleteDispatchBinding(
                        transaction, requestId);
            }
            catch (const std::exception&)
            {
                rejected = true;
            }
            assert(rejected);
            transaction.abort();
        };
        const auto phase3RequestId =
            phase3Request.request.requestId.value();
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "UPDATE campaign_operations_dispatch_attempt SET "
            "attempt_identity_canonical=attempt_identity_canonical||"
            "';corrupt=1' WHERE operational_request_id=" +
                std::to_string(phase3RequestId));
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "UPDATE campaign_operations_dispatch_attempt SET "
            "attempt_ordinal=attempt_ordinal+1 WHERE "
            "operational_request_id=" + std::to_string(phase3RequestId));
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "UPDATE campaign_operations_dispatch_attempt SET "
            "lease_token_digest='fnv1a64:0000000000000000' WHERE "
            "operational_request_id=" + std::to_string(phase3RequestId));
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "DELETE FROM campaign_operations_dispatch_audit_reference_event "
            "WHERE dispatch_attempt_outcome_id IS NOT NULL AND "
            "operational_request_id=" + std::to_string(phase3RequestId) +
            "; DELETE FROM campaign_operations_dispatch_attempt_outcome "
            "WHERE dispatch_attempt_id IN (SELECT dispatch_attempt_id FROM "
            "campaign_operations_dispatch_attempt WHERE "
            "operational_request_id=" + std::to_string(phase3RequestId) +
            ");");
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "UPDATE campaign_operations_dispatch_attempt_outcome SET "
            "resulting_request_version=resulting_request_version+1 WHERE "
            "dispatch_attempt_id IN (SELECT dispatch_attempt_id FROM "
            "campaign_operations_dispatch_attempt WHERE "
            "operational_request_id=" + std::to_string(phase3RequestId) +
            ");");
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "DELETE FROM campaign_operations_downstream_control_owner "
            "WHERE request_binding_id=(SELECT request_binding_id FROM "
            "campaign_operations_request_binding WHERE "
            "operational_request_id=" + std::to_string(phase3RequestId) +
            " ORDER BY member_ordinal DESC LIMIT 1); DELETE FROM "
            "campaign_operations_request_binding WHERE request_binding_id=("
            "SELECT request_binding_id FROM "
            "campaign_operations_request_binding WHERE "
            "operational_request_id=" + std::to_string(phase3RequestId) +
            " ORDER BY member_ordinal DESC LIMIT 1);");
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "UPDATE campaign_operations_request_binding SET "
            "proposal_identity_canonical=proposal_identity_canonical||"
            "';corrupt=1' WHERE "
            "operational_request_id=" + std::to_string(phase3RequestId) +
            " AND member_ordinal=1;");
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "UPDATE campaign_operations_request_binding SET "
            "binding_disposition='adopted_existing_pending',"
            "execution_disposition='reused',"
            "activation_disposition='reused' WHERE "
            "operational_request_id=" + std::to_string(phase3RequestId) +
            " AND member_ordinal=1;");
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "DELETE FROM campaign_operations_downstream_control_owner "
            "WHERE operational_request_id=" +
                std::to_string(phase3RequestId) +
            " AND request_binding_id=(SELECT request_binding_id FROM "
            "campaign_operations_request_binding WHERE "
            "operational_request_id=" + std::to_string(phase3RequestId) +
            " ORDER BY member_ordinal LIMIT 1);");
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "UPDATE campaign_operations_downstream_control_owner SET "
            "owner_identity_canonical=owner_identity_canonical||"
            "';corrupt=1' WHERE operational_request_id=" +
                std::to_string(phase3RequestId));
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "UPDATE campaign_operations_reservation SET "
            "reservation_state='held' WHERE reservation_id=" +
                std::to_string(
                    phase3Request.reservation.reservationId.value()));
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "UPDATE campaign_operations_operational_request SET "
            "state_version=state_version+1 WHERE operational_request_id=" +
                std::to_string(phase3RequestId));
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "UPDATE campaign_operations_dispatch_audit_reference_event SET "
            "diagnostic_code='corrupt_audit_reference' WHERE "
            "dispatch_attempt_outcome_id IS NOT NULL AND "
            "operational_request_id=" + std::to_string(phase3RequestId));
        assertBindingCorruptionRejected(
            phase3Request.request.requestId,
            "UPDATE campaign_operations_request_binding SET "
            "binding_identity_canonical=binding_identity_canonical||"
            "';hash_collision_bytes=1' WHERE operational_request_id=" +
                std::to_string(phase3RequestId) +
            " AND member_ordinal=1;");

        currentStage =
            "campaign_operations_phase3_unknown_commit_partial_binding";
        const std::vector<CampaignProposal> corruptReplayProposals{
            CreateApprovedProposal(runtime, 3.80, 216),
            CreateApprovedProposal(runtime, 3.81, 217)};
        {
            pqxx::work fixture{owner};
            SetSearchPath(fixture, schema);
            InsertMaterialization(
                fixture, 216, corruptReplayProposals);
            fixture.commit();
        }
        const auto corruptReplayRequest =
            CreatePhase3Request(owner, schema, 216);
        (void)EA::CampaignOperations::DispatchOneRequestForIsolatedTest(
            phase3ConnectionString,
            corruptReplayRequest.request.requestId, 1,
            EA::CampaignOperations::ActorIdentity(
                "phase3.corruption.fixture@example.test"),
            phase3Gate);
        {
            pqxx::work corruption{owner};
            SetSearchPath(corruption, schema);
            corruption.exec(
                "DELETE FROM "
                "campaign_operations_downstream_control_owner "
                "WHERE request_binding_id=(SELECT request_binding_id FROM "
                "campaign_operations_request_binding WHERE "
                "operational_request_id=$1 ORDER BY member_ordinal DESC "
                "LIMIT 1);",
                pqxx::params{
                    corruptReplayRequest.request.requestId.value()});
            corruption.exec(
                "DELETE FROM campaign_operations_request_binding "
                "WHERE request_binding_id=(SELECT request_binding_id FROM "
                "campaign_operations_request_binding WHERE "
                "operational_request_id=$1 ORDER BY member_ordinal DESC "
                "LIMIT 1);",
                pqxx::params{
                    corruptReplayRequest.request.requestId.value()});
            corruption.commit();
        }
        int corruptReplayPhase5Invocations = 0;
        bool corruptReplayRejected = false;
        try
        {
            (void)EA::CampaignOperations::
                DispatchOneRequestForIsolatedTest(
                    phase3ConnectionString,
                    corruptReplayRequest.request.requestId, 1,
                    EA::CampaignOperations::ActorIdentity(
                        "phase3.corruption.replay@example.test"),
                    phase3Gate,
                    [&](EA::CampaignOperations::
                            DispatchTestInjectionPoint point)
                    {
                        if (point == EA::CampaignOperations::
                                DispatchTestInjectionPoint::
                                    beforePhase5Invocation)
                            ++corruptReplayPhase5Invocations;
                    });
        }
        catch (const std::exception&)
        {
            corruptReplayRejected = true;
        }
        assert(corruptReplayRejected);
        assert(corruptReplayPhase5Invocations == 0);
        {
            pqxx::work corruption{owner};
            SetSearchPath(corruption, schema);
            corruption.exec(
                "DELETE FROM campaign_operations_downstream_control_owner "
                "WHERE operational_request_id=$1;",
                pqxx::params{
                    corruptReplayRequest.request.requestId.value()});
            corruption.exec(
                "DELETE FROM campaign_operations_request_binding "
                "WHERE operational_request_id=$1;",
                pqxx::params{
                    corruptReplayRequest.request.requestId.value()});
            corruption.commit();
        }
        int missingBindingPhase5Invocations = 0;
        bool missingBindingRejected = false;
        try
        {
            (void)EA::CampaignOperations::
                DispatchOneRequestForIsolatedTest(
                    phase3ConnectionString,
                    corruptReplayRequest.request.requestId, 1,
                    EA::CampaignOperations::ActorIdentity(
                        "phase3.missing.binding@example.test"),
                    phase3Gate,
                    [&](EA::CampaignOperations::
                            DispatchTestInjectionPoint point)
                    {
                        if (point == EA::CampaignOperations::
                                DispatchTestInjectionPoint::
                                    beforePhase5Invocation)
                            ++missingBindingPhase5Invocations;
                    });
        }
        catch (const std::exception&)
        {
            missingBindingRejected = true;
        }
        assert(missingBindingRejected);
        assert(missingBindingPhase5Invocations == 0);

        currentStage =
            "campaign_operations_phase3_transactional_role_scope";
        const long long alreadyBoundExperimentId = Scalar(owner, schema,
            "SELECT experiment_id FROM campaign_operations_request_binding "
            "WHERE operational_request_id=" +
            std::to_string(
                directOverlapRequest.request.requestId.value()));
        {
            pqxx::work pause{owner};
            SetSearchPath(pause, schema);
            pause.exec(
                "UPDATE experiment SET status='paused',phase='train',"
                "updated_at=transaction_timestamp() WHERE experiment_id=$1;",
                pqxx::params{alreadyBoundExperimentId});
            pause.commit();
        }
        bool alreadyBoundReactivationDenied = false;
        try
        {
            pqxx::work forbidden{owner};
            SetSearchPath(forbidden, schema);
            forbidden.exec(
                "SET LOCAL ROLE "
                "campaign_operations_phase5_transactional;");
            forbidden.exec(
                "UPDATE experiment SET status='pending',phase='train',"
                "updated_at=transaction_timestamp() WHERE experiment_id=$1;",
                pqxx::params{alreadyBoundExperimentId});
            forbidden.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            alreadyBoundReactivationDenied = PermissionDenied(error) ||
                std::string{error.what()}.find(
                    "Campaign Operations Phase 5 experiment update denied") !=
                    std::string::npos;
        }
        assert(alreadyBoundReactivationDenied);

        bool arbitraryExperimentInsertDenied = false;
        try
        {
            pqxx::work forbidden{owner};
            SetSearchPath(forbidden, schema);
            forbidden.exec(
                "SET LOCAL ROLE "
                "campaign_operations_phase5_transactional;");
            forbidden.exec(
                "INSERT INTO experiment (symbol, prediction_horizon, "
                "c_next_threshold, core_lr_mult, head_lr_mult, "
                "target_epochs, checkpoint_interval, train_start, "
                "train_end, infer_start, infer_end, status, phase, "
                "resume_model_id, duplicate_nonce, invocation_mode, "
                "updated_at) SELECT symbol, prediction_horizon, "
                "c_next_threshold, core_lr_mult, head_lr_mult, "
                "target_epochs, checkpoint_interval, train_start, "
                "train_end, infer_start, infer_end, 'pending', 'train', "
                "resume_model_id, duplicate_nonce + 900000, "
                "invocation_mode, transaction_timestamp() "
                "FROM experiment WHERE experiment_id=17;");
            forbidden.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            arbitraryExperimentInsertDenied = PermissionDenied(error) ||
                std::string{error.what()}.find(
                    "Campaign Operations Phase 5 mutation lacks atomic "
                    "binding") != std::string::npos;
        }
        assert(arbitraryExperimentInsertDenied);

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
