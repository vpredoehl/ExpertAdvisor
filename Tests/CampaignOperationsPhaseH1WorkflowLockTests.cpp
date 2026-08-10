#include "CampaignOperationsCompletionService.hpp"
#include "CampaignOperationsControlService.hpp"
#include "CampaignOperationsRepository.hpp"
#include "CampaignOperationsService.hpp"
#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationConversion.hpp"
#include "ExperimentRecommendationConversionProposalReviewRepository.hpp"
#include "ExperimentRecommendationConversionRepository.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <pqxx/pqxx>

namespace CO = EA::CampaignOperations;

namespace
{

constexpr long long kPrimaryCampaignId = 71;

std::string FixtureSymbol(long long campaignId)
{
    if (campaignId == 71) return "eurusd";
    if (campaignId == 72) return "gbpusd";
    return "gbp" + std::to_string(campaignId);
}

EA::ExperimentRecommendation::ProposedExperimentSpecification
BuildH2FixtureProposal(
    long long recommendationId,
    long long sourceExperimentId,
    const std::string& symbol)
{
    using namespace EA::ExperimentRecommendation;
    RecommendationConversionRequest request;
    request.recommendationExists = true;
    request.recommendationId = recommendationId;
    request.recommendationStatus = RecommendationStatus::approved;
    request.sourceExperimentId = sourceExperimentId;
    request.recommendationSourceExperimentId = sourceExperimentId;

    request.sourceInvocation.configuration.symbol = symbol;
    request.sourceInvocation.configuration.predictionHorizon = 12;
    request.sourceInvocation.configuration.labelThreshold = 0.001;
    request.sourceInvocation.configuration.coreLrMult = 1.0;
    request.sourceInvocation.configuration.headLrMult = 5.0;
    request.sourceInvocation.configuration.targetEpochs = 120;
    request.sourceInvocation.configuration.trainStartDate = "2010-01-01";
    request.sourceInvocation.configuration.trainEndDate = "2025-01-01";
    request.sourceInvocation.configuration.inferStartDate = "2025-01-01";
    request.sourceInvocation.configuration.inferEndDate = "2026-01-01";
    request.sourceInvocation.checkpointInterval = 15;

    request.mutations.push_back({
        kCoreLrMult, "1", CanonicalRecommendationDouble(1.25)});
    request.reviewAuthorization.present = true;
    request.reviewAuthorization.recommendationId = recommendationId;
    request.reviewAuthorization.latestAction = RecommendationReviewAction::approve;
    request.reviewAuthorization.resultingStatus = RecommendationStatus::approved;
    request.reviewAuthorization.latestActionEffective = true;
    request.reviewAuthorization.authorizationCanonical =
        "h2-fixture-review-authorization";
    request.reviewAuthorization.authorizationHash = RecommendationCanonicalHash(
        request.reviewAuthorization.authorizationCanonical);
    request.evaluation.state = RecommendationConversionEvidenceState::completed;
    request.evaluation.valid = true;
    request.evaluation.recommendationId = recommendationId;
    request.evaluation.sourceExperimentId = sourceExperimentId;
    request.evaluation.eligibility = RecommendationEligibility::eligible;
    request.evaluation.disposition =
        RecommendationEvaluationDisposition::advisoryReady;
    request.evaluation.evaluationIdentityCanonical =
        "h2-fixture-evaluation-identity";
    request.evaluation.evaluationIdentityHash = RecommendationCanonicalHash(
        request.evaluation.evaluationIdentityCanonical);
    request.evaluation.evaluationPolicyCanonical = "h2-fixture-evaluation-policy";
    request.evaluation.evaluationPolicyHash = RecommendationCanonicalHash(
        request.evaluation.evaluationPolicyCanonical);
    request.score.state = RecommendationConversionEvidenceState::completed;
    request.score.valid = true;
    request.score.recommendationId = recommendationId;
    request.score.finalScore = 0.75;
    request.score.scoringPolicyCanonical = "h2-fixture-scoring-policy";
    request.score.scoringPolicyHash = RecommendationCanonicalHash(
        request.score.scoringPolicyCanonical);
    request.evaluation.scoringPolicyHash = request.score.scoringPolicyHash;

    const auto proposed = request.sourceInvocation;
    auto proposedConfiguration = proposed.configuration;
    proposedConfiguration.coreLrMult = 1.25;
    const auto semantic = BuildRecommendationCandidateIdentity(
        proposedConfiguration);
    const auto invocation = BuildRecommendationInvocationIdentity(
        {proposedConfiguration, proposed.checkpointInterval,
         proposed.resumeModelId});
    request.recommendationSemanticCanonical = semantic.canonicalText;
    request.recommendationSemanticHash = semantic.hash;
    request.recommendationInvocationCanonical = invocation.canonicalText;
    request.recommendationInvocationHash = invocation.hash;

    const auto result = BuildProposedExperimentSpecification(request);
    if (!result.eligibility.eligible || !result.proposal)
        throw std::runtime_error("h2_fixture_proposal_not_eligible");
    return *result.proposal;
}

void SeedH2PhaseEUpstream(
    pqxx::transaction_base& transaction,
    long long campaignId,
    long long& recommendationId,
    long long& sourceExperimentId,
    long long& sourceAnalysisId)
{
    recommendationId = campaignId * 2000 + 1;
    sourceExperimentId = campaignId * 3000 + 1;
    sourceAnalysisId = campaignId * 6000 + 1;
    const std::string symbol = FixtureSymbol(campaignId);
    transaction.exec(
        "INSERT INTO experiment (experiment_id,symbol,prediction_horizon,"
        "c_next_threshold,core_lr_mult,head_lr_mult,target_epochs,"
        "checkpoint_interval,train_start,train_end,infer_start,infer_end,"
        "status,phase,invocation_mode) VALUES ($1,$2,12,0.001,1.0,"
        "5.0,120,15,'2010-01-01','2025-01-01','2025-01-01','2026-01-01',"
        "'paused','train','h2_fixture_source');",
        pqxx::params{sourceExperimentId, symbol});
    transaction.exec(
        "INSERT INTO experiment_analysis_result(analysis_id,experiment_id) "
        "VALUES ($1,$2);",
        pqxx::params{sourceAnalysisId, sourceExperimentId});
    transaction.exec(
        "INSERT INTO experiment_recommendation (recommendation_id,status,"
        "source_experiment_id,source_analysis_id,source_symbol,"
        "source_prediction_horizon,source_leader_score,source_infer_accuracy,"
        "reason,approved_at) VALUES ($1,'approved',$2,$3,$4,12,0.9,"
        "0.9,'h2 fixture approved recommendation',now());",
        pqxx::params{recommendationId, sourceExperimentId, sourceAnalysisId,
            symbol});
}

void SeedH2CompletePhaseEFixtureOne(
    pqxx::transaction_base& transaction,
    long long id)
{
    using namespace EA::ExperimentRecommendation;
    long long recommendationId = 0;
    long long sourceExperimentId = 0;
    long long sourceAnalysisId = 0;
    SeedH2PhaseEUpstream(transaction, id, recommendationId,
        sourceExperimentId, sourceAnalysisId);
    const auto symbol = FixtureSymbol(id);
    const auto proposal = BuildH2FixtureProposal(
        recommendationId, sourceExperimentId, symbol);
    const auto persistedProposal = PersistRecommendationConversionProposal(
        transaction, proposal).persisted;
    RecommendationConversionProposalReviewRequest review;
    review.proposalId = persistedProposal.proposalId;
    review.decision = RecommendationConversionProposalReviewDecision::approve;
    review.requestId = "h2-fixture-review-" + std::to_string(id);
    review.operatorIdentity = "h2.fixture.reviewer@example.test";
    review.reasonText = "H2 fixture approved Phase E conversion proposal.";
    const auto persistedReview =
        PersistRecommendationConversionProposalReviewDecision(
            transaction, review);
    if (!persistedReview.decision ||
        persistedReview.decision->decision !=
            RecommendationConversionProposalReviewDecision::approve)
        throw std::runtime_error("h2_fixture_review_not_approved");

    const std::string materializationCanonical =
        "h1-lock-materialization-" + std::to_string(id);
    const std::string materializationHash = RecommendationCanonicalHash(
        materializationCanonical);
    const auto hash = [](const std::string& value)
    {
        return RecommendationCanonicalHash(value);
    };
    const std::string approvalCanonical = "h1-lock-approval-" + std::to_string(id);
    const std::string rankingCanonical = "h1-lock-ranking-" + std::to_string(id);
    const std::string planningCanonical = "h1-lock-planning-" + std::to_string(id);
    const std::string planCanonical = "h1-lock-plan-" + std::to_string(id);
    const std::string reviewCanonical = "h1-lock-review-" + std::to_string(id);
    transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_materialization ("
        "recommendation_campaign_materialization_id,recommendation_campaign_approval_id,"
        "materialization_contract_version,approval_identity_canonical,approval_identity_hash,"
        "recommendation_ranking_snapshot_id,ranking_snapshot_identity_canonical,"
        "ranking_snapshot_identity_hash,planning_policy_canonical,planning_policy_hash,"
        "planning_scope_canonical,campaign_plan_identity_canonical,campaign_plan_identity_hash,"
        "campaign_review_identity_canonical,campaign_review_identity_hash,approval_decision,"
        "approval_reviewer_identity,approval_reason_text,materialized_by,"
        "materialization_reason_text,selected_member_count,initially_created_proposal_count,"
        "initially_reused_proposal_count,materialization_identity_canonical,"
        "materialization_identity_hash) VALUES($1,$2,1,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,"
        "'approved','h2.fixture.approver','H2 fixture approval.',"
        "'h2.fixture.materializer','H2 fixture materialization.',1,1,0,$15,$16);",
        pqxx::params{id, id + 1000, approvalCanonical, hash(approvalCanonical),
            id + 2000, rankingCanonical, hash(rankingCanonical), planningCanonical,
            hash(planningCanonical), "h1-lock-planning-scope-" + std::to_string(id),
            planCanonical, hash(planCanonical), reviewCanonical, hash(reviewCanonical),
            materializationCanonical, materializationHash});
    const std::string selectedCanonical = "h1-lock-selected-" + std::to_string(id);
    transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_materialization_member ("
        "recommendation_campaign_materialization_member_id,recommendation_campaign_materialization_id,"
        "member_ordinal,recommendation_ranking_member_id,recommendation_id,source_experiment_id,"
        "ranking_position,selected_member_identity_canonical,selected_member_identity_hash,"
        "recommendation_conversion_proposal_id,proposal_identity_canonical,proposal_identity_hash) "
        "VALUES($1,$2,1,$3,$4,$5,1,$6,$7,$8,$9,$10);",
        pqxx::params{id * 100 + 1, id, id * 1000 + 1, recommendationId,
            sourceExperimentId, selectedCanonical, hash(selectedCanonical),
            persistedProposal.proposalId,
            persistedProposal.proposal.conversionIdentityCanonical,
            persistedProposal.proposal.conversionIdentityHash});

    const auto workflowCounts = transaction.exec(
        "SELECT (SELECT count(*) FROM experiment_recommendation_conversion_proposal "
        "WHERE recommendation_conversion_proposal_id=$1),"
        "(SELECT count(*) FROM experiment_recommendation_conversion_review_decision "
        "WHERE recommendation_conversion_proposal_id=$1 AND decision='approve'),"
        "(SELECT count(*) FROM experiment_recommendation_conversion_execution "
        "WHERE recommendation_conversion_proposal_id=$1),"
        "(SELECT count(*) FROM experiment_recommendation_conversion_activation "
        "WHERE recommendation_conversion_proposal_id=$1),"
        "(SELECT count(*) FROM experiment_recommendation_campaign_materialization_member "
        "WHERE recommendation_conversion_proposal_id=$1 AND proposal_identity_hash=$2);",
        pqxx::params{persistedProposal.proposalId,
            persistedProposal.proposal.conversionIdentityHash}).one_row();
    if (workflowCounts[0].as<int>() != 1 || workflowCounts[1].as<int>() != 1 ||
        workflowCounts[2].as<int>() != 0 || workflowCounts[3].as<int>() != 0 ||
        workflowCounts[4].as<int>() != 1)
        throw std::runtime_error("h2_fixture_phase_e_chain_incomplete");
    std::cout << "H2_PHASE_E_FIXTURE_ASSERTIONS campaign_id=" << id
              << " proposal_id=" << persistedProposal.proposalId
              << " review_id=" << persistedReview.decision->reviewDecisionId
              << " proposal=1 approved_review=1 execution=0 activation=0 materialization_member=1\n";
}

void SeedOneFixture(
    pqxx::transaction_base& transaction,
    long long id,
    bool completePhaseE)
{
    const std::string suffix = std::to_string(id);
    const std::string materializationCanonical =
        "h1-lock-materialization-" + suffix;
    const std::string materializationHash =
        EA::ExperimentRecommendation::RecommendationCanonicalHash(
            materializationCanonical);
    const auto hash = [](const std::string& value)
    {
        return EA::ExperimentRecommendation::RecommendationCanonicalHash(value);
    };
    const std::string approvalCanonical = "h1-lock-approval-" + suffix;
    const std::string rankingCanonical = "h1-lock-ranking-" + suffix;
    const std::string planningCanonical = "h1-lock-planning-" + suffix;
    const std::string planCanonical = "h1-lock-plan-" + suffix;
    const std::string reviewCanonical = "h1-lock-review-" + suffix;
    if (completePhaseE)
    {
        SeedH2CompletePhaseEFixtureOne(transaction, id);
    }
    else
    {
      transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_materialization ("
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
        "VALUES($1,$2,1,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,"
        "'approved','assurance@example.test','H1 lock fixture approval.',"
        "'assurance@example.test','H1 lock fixture materialization.',1,1,0,"
        "$15,$16);",
        pqxx::params{id, id + 1000, approvalCanonical, hash(approvalCanonical),
            id + 2000, rankingCanonical, hash(rankingCanonical),
            planningCanonical, hash(planningCanonical),
            "h1-lock-planning-scope-" + suffix, planCanonical,
            hash(planCanonical), reviewCanonical, hash(reviewCanonical),
            materializationCanonical, materializationHash});
    const std::string selectedCanonical = "h1-lock-selected-" + suffix;
    const std::string proposalCanonical = "h1-lock-proposal-" + suffix;
      transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_materialization_member ("
        "recommendation_campaign_materialization_member_id,"
        "recommendation_campaign_materialization_id,member_ordinal,"
        "recommendation_ranking_member_id,recommendation_id,"
        "source_experiment_id,ranking_position,"
        "selected_member_identity_canonical,selected_member_identity_hash,"
        "recommendation_conversion_proposal_id,proposal_identity_canonical,"
        "proposal_identity_hash) VALUES($1,$2,1,$3,$4,$5,1,$6,$7,$8,$9,$10);",
        pqxx::params{id * 100 + 1, id, id * 1000 + 1, id * 2000 + 1,
            id * 3000 + 1, selectedCanonical, hash(selectedCanonical),
            id * 4000 + 1, proposalCanonical, hash(proposalCanonical)});
    }

    const CO::OperationalCampaign campaign = CO::BuildOperationalCampaign(
        id, 1, materializationCanonical, materializationHash, 1);
    const CO::OperationalCampaignId campaignId(id);
    const CO::OperationalAuthorizationEvent authorization =
        CO::BuildOperationalAuthorizationEvent(campaignId,
            campaign.identity.canonicalText(), std::nullopt, std::nullopt,
            std::nullopt, 1, CO::AuthorizationEventKind::granted,
            CO::OperationalActionKind::dispatchFullMaterialization, 1,
            CO::ScopeKind::completeMaterialization, 1,
            CO::PrerequisitePolicy::phase4dMaterializationOnlyV1,
            std::nullopt, std::nullopt, std::nullopt,
            CO::kCampaignOperationsAuthorizationRole,
            CO::ActorIdentity("assurance@example.test"),
            CO::Reason("H1 lock fixture authorization."),
            CO::UtcTimestamp("2026-08-01T00:00:00.000000Z"), std::nullopt);
    const CO::BudgetLedgerEntry budget = CO::BuildBudgetLedgerEntry(campaignId,
        campaign.identity.canonicalText(), std::nullopt, std::nullopt,
        std::nullopt, 1, CO::BudgetLedgerEntryKind::grant,
        CO::BudgetLedgerStatus::active,
        CO::BudgetUnit::materializedMemberDispatch, 1, 0, 1,
        CO::ActorIdentity("assurance@example.test"),
        CO::Reason("H1 lock fixture budget."));
    const CO::LogicalOperation operation =
        CO::BuildLogicalOperation(campaignId, campaign);
    const CO::Reservation reservation = CO::BuildReservation(operation,
        CO::AuthorizationEventId(id), authorization.identity.canonicalText(),
        authorization.identity.hash(), CO::BudgetLedgerEntryId(id), 1,
        budget.identity.canonicalText(), budget.identity.hash(), 1, 1,
        CO::BudgetUnit::materializedMemberDispatch, std::nullopt);
    const CO::OperationalRequest request = CO::BuildOperationalRequest(
        operation, CO::AuthorizationEventId(id),
        authorization.identity.canonicalText(), authorization.identity.hash(),
        CO::ReservationId(id), reservation.identity.canonicalText(),
        reservation.identity.hash(), 1, materializationHash,
        CO::ActorIdentity("assurance@example.test"),
        CO::Reason("H1 lock fixture request."),
        CO::PrerequisitePolicy::phase4dMaterializationOnlyV1,
        std::nullopt, std::nullopt);
    const CO::ReservationEvent acquisition =
        CO::BuildReservationAcquisitionEvent(CO::ReservationId(id),
            reservation.identity.canonicalText(), CO::OperationalRequestId(id),
            request.identity.canonicalText(), 1);

    transaction.exec(
        "INSERT INTO campaign_operations_campaign (operational_campaign_id,"
        "recommendation_campaign_materialization_id,"
        "materialization_contract_version,materialization_identity_canonical,"
        "materialization_identity_hash,materialization_member_count,origin_kind,"
        "action_kind,action_contract_version,scope_kind,scope_contract_version,"
        "campaign_contract_version,campaign_identity_canonical,"
        "campaign_identity_hash) VALUES($1,$1,1,$2,$3,1,"
        "'phase4d_materialization_v1','dispatch_full_materialization',1,"
        "'complete_materialization',1,1,$4,$5);",
        pqxx::params{id, materializationCanonical, materializationHash,
            campaign.identity.canonicalText(), campaign.identity.hash()});
    transaction.exec(
        "INSERT INTO campaign_operations_authorization_event ("
        "authorization_event_id,operational_campaign_id,"
        "campaign_identity_canonical,chain_version,event_kind,action_kind,"
        "action_contract_version,scope_kind,scope_contract_version,"
        "prerequisite_policy,authorization_role,actor_identity,reason,"
        "not_before,authorization_contract_version,"
        "authorization_identity_canonical,authorization_identity_hash) "
        "VALUES($1,$1,$2,1,'granted','dispatch_full_materialization',1,"
        "'complete_materialization',1,'phase4d_materialization_only_v1',"
        "$3,'assurance@example.test','H1 lock fixture authorization.',"
        "'2026-08-01T00:00:00.000000Z',1,$4,$5);",
        pqxx::params{id, campaign.identity.canonicalText(),
            CO::kCampaignOperationsAuthorizationRole,
            authorization.identity.canonicalText(), authorization.identity.hash()});
    transaction.exec(
        "INSERT INTO campaign_operations_budget_ledger_entry ("
        "budget_ledger_entry_id,operational_campaign_id,"
        "campaign_identity_canonical,ledger_version,entry_kind,ledger_status,"
        "budget_unit,delta,prior_total,resulting_total,administrator_identity,"
        "reason,budget_contract_version,budget_identity_canonical,"
        "budget_identity_hash) VALUES($1,$1,$2,1,'grant','active',"
        "'materialized_member_dispatch',1,0,1,'assurance@example.test',"
        "'H1 lock fixture budget.',1,$3,$4);",
        pqxx::params{id, campaign.identity.canonicalText(),
            budget.identity.canonicalText(), budget.identity.hash()});
    transaction.exec(
        "INSERT INTO campaign_operations_reservation (reservation_id,"
        "operational_campaign_id,campaign_identity_canonical,"
        "logical_operation_contract_version,logical_operation_canonical,"
        "logical_operation_hash,authorization_event_id,"
        "authorization_identity_canonical,authorization_identity_hash,"
        "budget_ledger_entry_id,budget_ledger_version,budget_identity_canonical,"
        "budget_identity_hash,action_kind,action_contract_version,"
        "recommendation_campaign_materialization_id,"
        "materialization_contract_version,materialization_identity_canonical,"
        "materialization_identity_hash,scope_kind,scope_contract_version,"
        "materialization_member_count,amount,budget_unit,"
        "reservation_contract_version,reservation_identity_canonical,"
        "reservation_identity_hash,reservation_state,state_version) "
        "VALUES($1,$1,$2,1,$3,$4,$1,$5,$6,$1,1,$7,$8,"
        "'dispatch_full_materialization',1,$1,1,$9,$10,"
        "'complete_materialization',1,1,1,'materialized_member_dispatch',"
        "1,$11,$12,'held',1);",
        pqxx::params{id, campaign.identity.canonicalText(),
            operation.identity.canonicalText(), operation.identity.hash(),
            authorization.identity.canonicalText(), authorization.identity.hash(),
            budget.identity.canonicalText(), budget.identity.hash(),
            materializationCanonical, materializationHash,
            reservation.identity.canonicalText(), reservation.identity.hash()});
    transaction.exec(
        "INSERT INTO campaign_operations_operational_request ("
        "operational_request_id,operational_campaign_id,"
        "campaign_identity_canonical,logical_operation_contract_version,"
        "logical_operation_canonical,logical_operation_hash,authorization_event_id,"
        "authorization_identity_canonical,authorization_identity_hash,"
        "reservation_id,reservation_identity_canonical,reservation_identity_hash,"
        "action_kind,action_contract_version,"
        "recommendation_campaign_materialization_id,"
        "materialization_contract_version,materialization_identity_canonical,"
        "materialization_identity_hash,ordered_scope_digest,"
        "materialization_member_count,accepting_actor_identity,reason,"
        "prerequisite_policy,request_contract_version,request_identity_canonical,"
        "request_identity_hash,request_state,state_version,"
        "production_dispatch_enabled) VALUES($1,$1,$2,1,$3,$4,$1,$5,$6,$1,"
        "$7,$8,'dispatch_full_materialization',1,$1,1,$9,$10,$10,1,"
        "'assurance@example.test','H1 lock fixture request.',"
        "'phase4d_materialization_only_v1',1,$11,$12,'ready',3,false);",
        pqxx::params{id, campaign.identity.canonicalText(),
            operation.identity.canonicalText(), operation.identity.hash(),
            authorization.identity.canonicalText(), authorization.identity.hash(),
            reservation.identity.canonicalText(), reservation.identity.hash(),
            materializationCanonical, materializationHash,
            request.identity.canonicalText(), request.identity.hash()});
    transaction.exec(
        "INSERT INTO campaign_operations_reservation_event ("
        "reservation_event_id,reservation_id,reservation_identity_canonical,"
        "transition_kind,expected_state,resulting_state,expected_version,"
        "resulting_version,operational_request_id,request_identity_canonical,"
        "amount,reservation_event_contract_version,"
        "reservation_event_identity_canonical,reservation_event_identity_hash) "
        "VALUES($1,$1,$2,'acquired',NULL,'held',0,1,$1,$3,1,1,$4,$5);",
        pqxx::params{id, reservation.identity.canonicalText(),
            request.identity.canonicalText(),
            acquisition.identity.canonicalText(), acquisition.identity.hash()});
}

void SeedFixtures(const std::string& connectionString, bool completePhaseE)
{
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};
    transaction.exec("SET LOCAL session_replication_role=replica;");
    SeedOneFixture(transaction, kPrimaryCampaignId, completePhaseE);
    SeedOneFixture(transaction, 72, completePhaseE);
    if (const char* extra = std::getenv("H3_EXTRA_FIXTURE_IDS");
        extra && *extra)
    {
        std::stringstream ids(extra);
        std::string idText;
        while (std::getline(ids, idText, ','))
        {
            if (idText.empty())
                throw std::invalid_argument("h3_fixture_id_empty");
            const long long id = std::stoll(idText);
            if (id <= 72)
                throw std::invalid_argument("h3_fixture_id_not_extra");
            SeedOneFixture(transaction, id, completePhaseE);
        }
    }
    transaction.commit();
}

void Pause(pqxx::transaction_base& transaction)
{
    transaction.exec("SELECT pg_sleep(8);");
}

void RunCompletion(const std::string& connectionString)
{
    const CO::CompleteIfSettledRequest request{
        kPrimaryCampaignId, "h1-lock-completion", "assurance@example.test",
        "H1 complete-workflow lock assurance."};
    const auto factory = [&connectionString]
    {
        return std::make_unique<pqxx::connection>(connectionString);
    };
    (void)CO::CompleteCampaignIfSettled(factory, request,
        [](CO::CompletionTestInjectionPoint point)
        {
            if (point == CO::CompletionTestInjectionPoint::
                    afterLocksBeforeEvidence)
                std::this_thread::sleep_for(std::chrono::seconds(4));
        });
}

void RunCancellation(const std::string& connectionString,
    const std::string& operationKey)
{
    const CO::CampaignCancellationCommandRequest request{
        kPrimaryCampaignId, kPrimaryCampaignId, 3, operationKey,
        "assurance@example.test",
        "H1 complete-workflow lock assurance."};
    (void)CO::CancelCampaign(connectionString, request,
        [](CO::CampaignOperationsControlTestInjectionPoint point,
           pqxx::transaction_base& transaction)
        {
            if (point == CO::CampaignOperationsControlTestInjectionPoint::
                    afterCancellationIntentInsertion)
                Pause(transaction);
        });
}

void RunReconciliation(const std::string& connectionString, bool recover)
{
    const CO::ReconciliationObserveRequest request{
        recover ? "h1-lock-recovery" : "h1-lock-reconciliation",
        0, 100, recover};
    (void)CO::ObserveAndRecoverCampaignOperations(connectionString, request,
        [recover](CO::CampaignOperationsControlTestInjectionPoint point,
                  pqxx::transaction_base& transaction)
        {
            if ((!recover && point == CO::CampaignOperationsControlTestInjectionPoint::
                    afterReconciliationCampaignLocksBeforeRequestLocks) ||
                (recover && point == CO::CampaignOperationsControlTestInjectionPoint::
                    beforeReconciliationRecoveryCommit))
                Pause(transaction);
        });
}

void RunBudgetMutation(const std::string& connectionString)
{
    pqxx::connection connection{connectionString};
    const CO::BudgetAdministrationRequest request{
        kPrimaryCampaignId, 1, CO::BudgetLedgerEntryKind::amend, 1,
        "assurance@example.test", "H1 complete-workflow lock assurance."};
    (void)CO::AdministerCampaignBudget(connection, request,
        [](CO::BudgetAdministrationTestInjectionPoint point)
        {
            if (point == CO::BudgetAdministrationTestInjectionPoint::
                    afterDomainLocksBeforePersistence)
                std::this_thread::sleep_for(std::chrono::seconds(4));
        });
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: CampaignOperationsPhaseH1WorkflowLockTests "
                     "CONNECTION MODE\n";
        return 64;
    }
    try
    {
        const std::string connectionString = argv[1];
        const std::string mode = argv[2];
        if (mode == "fixture-setup") SeedFixtures(connectionString, false);
        else if (mode == "phase-e-fixture-setup")
            SeedFixtures(connectionString, true);
        else if (mode == "completion") RunCompletion(connectionString);
        else if (mode == "cancellation")
            RunCancellation(connectionString, "h1-lock-cancellation");
        else if (mode == "reservation-release")
            RunCancellation(connectionString, "h1-lock-reservation-release");
        else if (mode == "reconciliation")
            RunReconciliation(connectionString, false);
        else if (mode == "recovery") RunReconciliation(connectionString, true);
        else if (mode == "budget-mutation") RunBudgetMutation(connectionString);
        else throw std::invalid_argument("unknown workflow mode");
        std::cout << "workflow=" << mode << ",outcome=completed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "workflow_error=" << error.what() << '\n';
        return 1;
    }
}
