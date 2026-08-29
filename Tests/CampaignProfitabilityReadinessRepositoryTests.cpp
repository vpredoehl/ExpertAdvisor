#include "../Sources/ExperimentRecommendation.hpp"
#include "../Sources/ExperimentRecommendationRanking.hpp"
#include "../Sources/InferenceProfitabilityRepository.hpp"
#include "../Sources/ProfitabilityVerificationRepository.hpp"
#include "../Sources/ProfitabilityVerificationService.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>

#include <pqxx/pqxx>

namespace Profitability = EA::InferenceProfitability;
namespace Verification = EA::ProfitabilityVerification;
namespace Recommendation = EA::ExperimentRecommendation;

namespace
{

std::string Environment(const char* name)
{
    const char* value = std::getenv(name);
    assert(value && *value);
    return value;
}

std::string ConnectionString(const std::string& user)
{
    return "hostaddr=" + Environment("LSTM_DB_HOST") +
        " user=" + user + " dbname=" + Environment("LSTM_DB_NAME");
}

void InsertContext(pqxx::transaction_base& transaction,
                   long long experimentId,
                   long long modelId,
                   long long resultId)
{
    transaction.exec(
        "INSERT INTO experiment("
        "experiment_id,symbol,prediction_horizon,c_next_threshold,"
        "infer_start,infer_end,last_model_id,target_epochs,status,phase,"
        "updated_at) VALUES($1,'EURUSDRMP',5,0.001,'2025-01-01',"
        "'2025-12-31',$2,100,'completed','analyze',now());",
        pqxx::params{experimentId, modelId});
    transaction.exec(
        "INSERT INTO model(model_id,experiment_id) VALUES($1,$2);",
        pqxx::params{modelId, experimentId});
    transaction.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "SELECT $1,'train_config_meta',0,ordinality-1,value "
        "FROM unnest(ARRAY[1,5,0.001,64,1,1,1,1,1,1,100,1,1,1]"
        "::double precision[]) WITH ORDINALITY AS config(value,ordinality);",
        pqxx::params{modelId});
    transaction.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "VALUES($1,'target_meta',0,0,1);",
        pqxx::params{modelId});
    transaction.exec(
        "INSERT INTO inference_eval_result("
        "id,model_id,status,inference_scope,checkpoint_eval_id,"
        "parent_experiment_id,symbol,prediction_horizon,threshold_logret,"
        "window_size,label_rule_id,target_type,from_date,to_date,"
        "completed_epochs,accept_model) VALUES("
        "$1,$2,'completed','final',NULL,NULL,'EURUSDRMP',5,0.001,64,1,1,"
        "'2025-01-01','2025-12-31',100,true);",
        pqxx::params{resultId, modelId});
}

Profitability::PersistResult Persist(pqxx::transaction_base& transaction,
                                     long long experimentId,
                                     long long modelId,
                                     long long resultId,
                                     bool positive)
{
    Profitability::Accumulator values;
    values.Observe(
        positive ? Profitability::kUpClass : Profitability::kDownClass,
        100.0f,
        110.0f);
    Profitability::ObservationRequest request;
    request.provenance.experimentId = experimentId;
    request.provenance.modelId = modelId;
    request.provenance.inferenceEvalResultId = resultId;
    request.provenance.scope = Profitability::Scope::finalInference;
    request.provenance.inferenceStart = "2025-01-01";
    request.provenance.inferenceEnd = "2025-12-31";
    request.statistics = values.statistics();
    request.sourceContentHash = values.SourceContentHash();
    return Profitability::PersistObservationIdempotently(transaction, request);
}

void InsertSnapshot(pqxx::transaction_base& transaction,
                    long long snapshotId,
                    int memberCount)
{
    const std::string identity =
        "profitability-readiness-snapshot-" + std::to_string(snapshotId);
    const std::string policy = "profitability-readiness-ranking-policy";
    const std::string scoring = "profitability-readiness-scoring-semantic";
    const std::string evaluation =
        "profitability-readiness-evaluation-semantic";
    transaction.exec(
        "INSERT INTO experiment_recommendation_ranking_snapshot("
        "recommendation_ranking_snapshot_id,status,"
        "ranking_snapshot_identity_canonical,ranking_snapshot_identity_hash,"
        "ranking_policy_canonical,ranking_policy_hash,ranking_version,"
        "population_semantic_state,scoring_semantic_canonical,"
        "scoring_semantic_hash,scoring_semantic_version,"
        "evaluation_semantic_canonical,evaluation_semantic_hash,"
        "evaluation_semantic_version,distinct_scoring_semantic_count,"
        "distinct_evaluation_semantic_count,homogeneity_validation_result,"
        "member_count,advisory_ready_count) VALUES("
        "$1,'completed',$2,$3,$4,$5,1,'verified_homogeneous',$6,$7,1,"
        "$8,$9,1,1,1,'verified_homogeneous',$10,$10);",
        pqxx::params{
            snapshotId,
            identity, Recommendation::RecommendationCanonicalHash(identity),
            policy, Recommendation::RecommendationCanonicalHash(policy),
            scoring, Recommendation::RecommendationCanonicalHash(scoring),
            evaluation, Recommendation::RecommendationCanonicalHash(evaluation),
            memberCount});
}

void InsertRecommendation(
    pqxx::transaction_base& transaction,
    long long recommendationId,
    long long experimentId,
    long long modelId,
    long long resultId,
    const std::optional<Profitability::Observation>& observation,
    const std::string& unavailableReason,
    bool frozenMismatch = false)
{
    const std::string semantic =
        "profitability-readiness-semantic-" +
        std::to_string(recommendationId);
    const std::string invocation =
        "profitability-readiness-invocation-" +
        std::to_string(recommendationId);
    transaction.exec(
        "INSERT INTO experiment_recommendation("
        "recommendation_id,source_experiment_id,source_model_id,source_symbol,"
        "source_prediction_horizon,changed_parameter,source_leader_score,"
        "source_infer_accuracy,source_predicted_neutral_proportion,"
        "semantic_configuration_canonical,semantic_hash,"
        "invocation_configuration_canonical,invocation_hash,"
        "final_profitability_provenance_version,"
        "source_final_inference_eval_result_id,"
        "source_final_profitability_observation_id,"
        "source_final_profitability_unavailable_reason,"
        "source_final_profitability_inference_scope,"
        "source_final_profitability_inference_start,"
        "source_final_profitability_inference_end,"
        "source_final_profitability_actionable_count,"
        "source_final_profitability_aggregate_return,"
        "source_final_profitability_average_return,"
        "source_final_profitability_metric_definition_hash,"
        "source_final_profitability_source_content_hash,"
        "source_final_profitability_observation_identity_hash) VALUES("
        "$1,$2,$3,'EURUSDRMP',5,'core_lr_mult',0.9,0.8,0.1,$4,$5,$6,$7,"
        "1,$8,$9,$10,'final',$11,$12,$13,$14,$15,$16,$17,$18);",
        pqxx::params{
            recommendationId, experimentId, modelId,
            semantic, Recommendation::RecommendationCanonicalHash(semantic),
            invocation, Recommendation::RecommendationCanonicalHash(invocation),
            resultId,
            observation
                ? std::optional<long long>{observation->observationId}
                : std::nullopt,
            unavailableReason.empty()
                ? std::optional<std::string>{}
                : std::optional<std::string>{unavailableReason},
            observation
                ? std::optional<std::string>{
                      observation->provenance.inferenceStart}
                : std::nullopt,
            observation
                ? std::optional<std::string>{observation->provenance.inferenceEnd}
                : std::nullopt,
            observation
                ? std::optional<long long>{static_cast<long long>(
                      observation->statistics.actionableCount)}
                : std::nullopt,
            observation
                ? std::optional<double>{observation->statistics
                      .aggregateTerminalHorizonLogReturnSum}
                : std::nullopt,
            observation
                ? observation
                      ->averageTerminalHorizonLogReturnPerActionablePrediction
                : std::nullopt,
            observation
                ? std::optional<std::string>{observation->metricDefinitionHash}
                : std::nullopt,
            observation
                ? std::optional<std::string>{observation->sourceContentHash}
                : std::nullopt,
            observation
                ? std::optional<std::string>{
                      frozenMismatch
                          ? Profitability::DeterministicHash("mismatch")
                          : observation->observationIdentityHash}
                : std::nullopt});
}

void InsertMember(pqxx::transaction_base& transaction,
                  long long memberId,
                  long long snapshotId,
                  long long recommendationId,
                  long long experimentId,
                  long long modelId,
                  int rank,
                  double score,
                  std::optional<long long> evaluationResultId = std::nullopt)
{
    transaction.exec(
        "INSERT INTO experiment_recommendation_ranking_member("
        "recommendation_ranking_member_id,recommendation_ranking_snapshot_id,"
        "recommendation_evaluation_result_id,recommendation_id,"
        "source_experiment_id,source_model_id,global_ordinal,bucket,"
        "final_score,symbol,horizon,family) VALUES("
        "$1,$2,$3,$4,$5,$6,$7,'advisory_ready',$8,'EURUSDRMP',5,"
        "'core_lr_mult');",
        pqxx::params{memberId, snapshotId, evaluationResultId,
                     recommendationId, experimentId, modelId, rank, score});
}

void InsertEvaluationResult(pqxx::transaction_base& transaction,
                            long long evaluationResultId,
                            long long recommendationId,
                            long long experimentId,
                            long long modelId,
                            double score)
{
    const std::string identity =
        "profitability-shadow-evaluation-" +
        std::to_string(evaluationResultId);
    transaction.exec(
        "INSERT INTO experiment_recommendation_evaluation_result("
        "recommendation_evaluation_result_id,recommendation_evaluation_run_id,"
        "recommendation_id,evaluation_identity_canonical,"
        "evaluation_identity_hash,source_experiment_id,source_model_id,"
        "final_score,eligibility,disposition,result_status,"
        "final_profitability_provenance_version,"
        "source_final_inference_eval_result_id,"
        "source_final_profitability_observation_id,"
        "source_final_profitability_unavailable_reason,"
        "source_final_profitability_inference_scope,"
        "source_final_profitability_inference_start,"
        "source_final_profitability_inference_end,"
        "source_final_profitability_actionable_count,"
        "source_final_profitability_aggregate_return,"
        "source_final_profitability_average_return,"
        "source_final_profitability_metric_definition_hash,"
        "source_final_profitability_source_content_hash,"
        "source_final_profitability_observation_identity_hash) "
        "SELECT $1,6,recommendation_id,$2,$3,source_experiment_id,"
        "source_model_id,$4,'eligible','advisory_ready','evaluated',"
        "final_profitability_provenance_version,"
        "source_final_inference_eval_result_id,"
        "source_final_profitability_observation_id,"
        "source_final_profitability_unavailable_reason,"
        "source_final_profitability_inference_scope,"
        "source_final_profitability_inference_start,"
        "source_final_profitability_inference_end,"
        "source_final_profitability_actionable_count,"
        "source_final_profitability_aggregate_return,"
        "source_final_profitability_average_return,"
        "source_final_profitability_metric_definition_hash,"
        "source_final_profitability_source_content_hash,"
        "source_final_profitability_observation_identity_hash "
        "FROM experiment_recommendation WHERE recommendation_id=$5 "
        "AND source_experiment_id=$6 AND source_model_id=$7;",
        pqxx::params{
            evaluationResultId, identity,
            Recommendation::RecommendationCanonicalHash(identity), score,
            recommendationId, experimentId, modelId});
}

void InsertAuthoritativeShadowSnapshot(pqxx::transaction_base& transaction)
{
    const std::string runIdentity = "profitability-shadow-evaluation-run-6";
    transaction.exec(
        "INSERT INTO experiment_recommendation_evaluation_run VALUES("
        "6,'completed',$1,$2,3,3,0,0);",
        pqxx::params{
            runIdentity,
            Recommendation::RecommendationCanonicalHash(runIdentity)});
    InsertEvaluationResult(transaction, 9001, 1002, 2, 20, 0.99);
    InsertEvaluationResult(transaction, 9002, 1003, 3, 30, 0.75);
    InsertEvaluationResult(transaction, 9003, 1001, 1, 10, 0.50);

    const std::string membership =
        "experiment_recommendation_ranking_membership_v1;count=3;"
        "member[0].evaluation_identity=36:profitability-shadow-evaluation-9001;"
        "member[0].evaluation_result_id=9001;"
        "member[1].evaluation_identity=36:profitability-shadow-evaluation-9002;"
        "member[1].evaluation_result_id=9002;"
        "member[2].evaluation_identity=36:profitability-shadow-evaluation-9003;"
        "member[2].evaluation_result_id=9003";
    const Recommendation::RecommendationRankingPolicy policy;
    Recommendation::RecommendationRankingScope scope;
    scope.type = Recommendation::RecommendationRankingScopeType::evaluationRun;
    scope.evaluationRunId = 6;
    Recommendation::RecommendationRankingPopulationSemanticValidation semantics;
    semantics.state = Recommendation::
        RecommendationRankingPopulationSemanticState::verifiedHomogeneous;
    const std::string scoring = "profitability-shadow-scoring-semantic-v1";
    const std::string evaluation =
        "profitability-shadow-evaluation-semantic-v1";
    semantics.scoringIdentity = Recommendation::
        RecommendationScoringSemanticIdentity{
            scoring, Recommendation::RecommendationCanonicalHash(scoring), 1};
    semantics.evaluationIdentity = Recommendation::
        RecommendationEvaluationSemanticIdentity{
            evaluation,
            Recommendation::RecommendationCanonicalHash(evaluation), 1};
    semantics.distinctScoringIdentityCount = 1;
    semantics.distinctEvaluationIdentityCount = 1;
    const std::string policyCanonical =
        Recommendation::RecommendationRankingPolicyCanonicalText(policy);
    const std::string scopeCanonical =
        Recommendation::RecommendationRankingScopeCanonicalText(scope);
    const std::string identity = Recommendation::
        RecommendationRankingSnapshotIdentityCanonicalTextFromMembership(
            policy, scope, 1000, membership, semantics);
    transaction.exec(
        "INSERT INTO experiment_recommendation_ranking_snapshot("
        "recommendation_ranking_snapshot_id,status,"
        "ranking_snapshot_identity_canonical,ranking_snapshot_identity_hash,"
        "ranking_policy_canonical,ranking_policy_hash,ranking_version,"
        "scope_type,scope_canonical,scope_hash,evaluation_run_filter,"
        "requested_limit,source_membership_canonical,source_membership_hash,"
        "ranking_snapshot_identity_version,population_semantic_state,"
        "scoring_semantic_canonical,scoring_semantic_hash,"
        "scoring_semantic_version,evaluation_semantic_canonical,"
        "evaluation_semantic_hash,evaluation_semantic_version,"
        "distinct_scoring_semantic_count,distinct_evaluation_semantic_count,"
        "homogeneity_validation_result,member_count,advisory_ready_count,"
        "blocked_count,non_actionable_count) VALUES("
        "9,'completed',$1,$2,$3,$4,1,'evaluation_run',$5,$6,6,1000,"
        "$7,$8,2,'verified_homogeneous',$9,$10,1,$11,$12,1,1,1,"
        "'verified_homogeneous',3,3,0,0);",
        pqxx::params{
            identity, Recommendation::RecommendationCanonicalHash(identity),
            policyCanonical,
            Recommendation::RecommendationCanonicalHash(policyCanonical),
            scopeCanonical,
            Recommendation::RecommendationCanonicalHash(scopeCanonical),
            membership, Recommendation::RecommendationCanonicalHash(membership),
            scoring, Recommendation::RecommendationCanonicalHash(scoring),
            evaluation,
            Recommendation::RecommendationCanonicalHash(evaluation)});
    InsertMember(transaction, 90001, 9, 1002, 2, 20, 1, 0.99, 9001);
    InsertMember(transaction, 90002, 9, 1003, 3, 30, 2, 0.75, 9002);
    InsertMember(transaction, 90003, 9, 1001, 1, 10, 3, 0.50, 9003);
}

void Setup()
{
    pqxx::connection connection{
        ConnectionString(Environment("LSTM_DB_ADMIN_USER"))};
    pqxx::work transaction{connection};
    InsertContext(transaction, 1, 10, 101);
    InsertContext(transaction, 2, 20, 201);
    InsertContext(transaction, 3, 30, 301);
    const auto positive = Persist(transaction, 1, 10, 101, true).observation;
    const auto negative = Persist(transaction, 2, 20, 201, false).observation;

    InsertSnapshot(transaction, 7, 3);
    InsertRecommendation(transaction, 1001, 1, 10, 101, positive, "");
    InsertRecommendation(transaction, 1002, 2, 20, 201, negative, "");
    InsertRecommendation(
        transaction, 1003, 3, 30, 301, std::nullopt,
        "no_profitability_observation");
    InsertMember(transaction, 7001, 7, 1002, 2, 20, 1, 0.99);
    InsertMember(transaction, 7002, 7, 1001, 1, 10, 2, 0.50);
    InsertMember(transaction, 7003, 7, 1003, 3, 30, 3, 0.75);

    InsertSnapshot(transaction, 8, 1);
    InsertRecommendation(
        transaction, 1004, 1, 10, 101, positive, "", true);
    InsertMember(transaction, 8001, 8, 1004, 1, 10, 1, 0.90);
    InsertAuthoritativeShadowSnapshot(transaction);
    transaction.commit();
}

void Verify()
{
    const std::string connectionString =
        ConnectionString(Environment("LSTM_DB_USER"));
    std::ostringstream output;
    std::ostringstream errors;
    assert(Verification::RunCampaignReadinessCommand(
               connectionString, 7, output, errors) == 4);
    const std::string rendered = output.str();
    assert(errors.str().empty());
    assert(rendered.find(
        "CAMPAIGN_PROFITABILITY_COVERAGE,ranking_snapshot_id=7,"
        "candidate_count=3,exact_final_profitability_count=2,"
        "missing_profitability_count=1") != std::string::npos);
    assert(rendered.find("profitability_sign=positive") != std::string::npos);
    assert(rendered.find("profitability_sign=negative") != std::string::npos);
    assert(rendered.find("profitability_sign=unavailable") != std::string::npos);
    assert(rendered.find(
        "ranking_member_id=7002,recommendation_id=1001") != std::string::npos);
    assert(rendered.find(
        "current_rank=2,current_score=0.5") != std::string::npos);
    assert(rendered.find("profitability_shadow_rank=1,rank_delta=1,"
                         "live_rank_authoritative=true") != std::string::npos);
    assert(rendered.find("live_profitability_weight=0") != std::string::npos);
    assert(rendered.find(
        "live_profitability_score_contribution=0") != std::string::npos);
    assert(rendered.find("current_rank_changed=false") != std::string::npos);
    assert(rendered.find(
        "campaign_profitability_contract_ready=true") != std::string::npos);
    assert(rendered.find(
        "profitability_shadow_ranking_ready=true") != std::string::npos);
    assert(rendered.find(
        "campaign_profitability_activation_performed=false") !=
           std::string::npos);
    assert(rendered.find("live_ranking_changed=false,read_only=true") !=
           std::string::npos);

    output.str({});
    output.clear();
    assert(Verification::RunCampaignReadinessCommand(
               connectionString, 8, output, errors) == 3);
    assert(output.str().find(
        "profitability_evidence_status=invalid_provenance") !=
           std::string::npos);
    assert(output.str().find(
        "campaign_profitability_contract_ready=false") !=
           std::string::npos);
    assert(output.str().find(
        "campaign_profitability_action=blocked_evidence_contract") !=
           std::string::npos);
    assert(output.str().find(
        "campaign_profitability_activation_performed=false") !=
           std::string::npos);

    output.str({});
    output.clear();
    bool calibrationRejectedInvalidProvenance = false;
    try
    {
        (void)Verification::RunCampaignProfitabilityCalibrationCommand(
            connectionString, 8, output, errors);
    }
    catch (const std::runtime_error&)
    {
        calibrationRejectedInvalidProvenance = true;
    }
    assert(calibrationRejectedInvalidProvenance);
    assert(output.str().empty());

    {
        pqxx::connection sourceConnection{connectionString};
        pqxx::read_transaction sourceTransaction{sourceConnection};
        const auto source = Verification::LoadCampaignProfitabilityShadowSource(
            sourceTransaction, 9);
        assert(source.controlSnapshotId == 9);
        assert(source.sourceEvaluationRunId == 6);
        assert(source.persistedMemberCount == 3);
        assert(source.candidates.size() == 3);
        assert(source.candidates[0].recommendationId == 1002);
        assert(source.candidates[1].recommendationId == 1003);
        assert(source.candidates[2].recommendationId == 1001);
        assert(source.candidates[0].profitability.state ==
               Verification::EvidenceState::valid);
        assert(source.candidates[1].profitability.state ==
               Verification::EvidenceState::unavailable);
        assert(source.candidates[2].profitability.state ==
               Verification::EvidenceState::valid);
        const auto coverage =
            Verification::LoadCampaignProfitabilityCoverageAudit(
                sourceTransaction, 9);
        assert(coverage.members.size() == 3);
        assert(coverage.reasonCounts.at("valid_profitability_observation") == 2);
        assert(coverage.reasonCounts.at("no_profitability_observation") == 1);
        assert(coverage.recoveryClassCounts.at(
                   "recoverable_historical_absence") == 1);
        const auto& missing = coverage.members[1];
        assert(missing.recommendationId == 1003);
        assert(missing.exactFinalInferenceResultExists);
        assert(missing.anyFinalInferenceResultExists);
        assert(!missing.exactFinalProfitabilityObservationExists);
        assert(!missing.anyInferenceProfitabilityObservationExists);
        assert(missing.recoveryClass == Verification::CoverageRecoveryClass::
                   recoverableHistoricalAbsence);
        assert(!missing.frozenSnapshotBackfillPermitted);
        assert(!coverage.hash.empty());
    }

    output.str({});
    output.clear();
    assert(Verification::RunCampaignShadowRankingCommand(
               connectionString, 9, {0.05, 0.01, 0.025}, output, errors) == 0);
    const std::string shadow = output.str();
    assert(shadow.find(
        "shadow_only=true,control_snapshot_id=9,source_evaluation_run_id=6,"
        "source_member_count=3,live_rank_authoritative=true,"
        "live_profitability_weight=0,live_profitability_score_contribution=0,"
        "activation=false,database_write=false,experiment_created=false,"
        "experiment_queued=false,scheduler_modified=false") !=
        std::string::npos);
    assert(shadow.find("shadow_weight=0.01") != std::string::npos);
    assert(shadow.find("shadow_weight=0.025") != std::string::npos);
    assert(shadow.find("shadow_weight=0.050000000000000003") !=
           std::string::npos);
    assert(shadow.find("profitability_state=unavailable") !=
           std::string::npos);
    assert(shadow.find("normalization_state=explicitly_unavailable") !=
           std::string::npos);
    assert(shadow.find("profitability_contribution=NULL") !=
               std::string::npos);

    output.str({});
    output.clear();
    assert(Verification::RunCampaignProfitabilityCalibrationCommand(
               connectionString, 9, output, errors) == 0);
    const std::string calibration = output.str();
    assert(calibration.find(
        "CAMPAIGN_PROFITABILITY_CALIBRATION_START,control_snapshot_id=9") !=
        std::string::npos);
    assert(calibration.find("grid_point_count=21") != std::string::npos);
    assert(calibration.find(
        "valid_profitability_observation=2,no_profitability_observation=1") !=
        std::string::npos);
    assert(calibration.find(
        "recovery_class=recoverable_historical_absence") !=
        std::string::npos);
    assert(calibration.find(
        "exact_final_inference_result_exists=true") != std::string::npos);
    assert(calibration.find(
        "frozen_snapshot_backfill_permitted=false") != std::string::npos);
    assert(calibration.find(
        "CAMPAIGN_PROFITABILITY_CALIBRATION_SWEEP_POINT") !=
        std::string::npos);
    assert(calibration.find("weight=0.025000000000000001") !=
        std::string::npos);
    assert(calibration.find(
        "CAMPAIGN_PROFITABILITY_CALIBRATION_PAIRWISE") != std::string::npos);
    assert(calibration.find(
        "CAMPAIGN_PROFITABILITY_CALIBRATION_RESPONSE_CURVE") !=
        std::string::npos);
    assert(calibration.find(
        "CAMPAIGN_PROFITABILITY_PRODUCTION_READINESS") != std::string::npos);
    assert(calibration.find("activation_ready=false") != std::string::npos);
    assert(calibration.find("database_write=false") != std::string::npos);
    std::ostringstream repeatedOutput;
    assert(Verification::RunCampaignProfitabilityCalibrationCommand(
               connectionString, 9, repeatedOutput, errors) == 0);
    assert(repeatedOutput.str() == calibration);

    {
        pqxx::connection auditConnection{connectionString};
        pqxx::read_transaction auditTransaction{auditConnection};
        const auto temporalAudit = Verification::
            LoadCampaignProfitabilityTemporalFeasibilityAudit(
                auditTransaction);
        assert(temporalAudit.cohorts.size() == 3);
        assert(!temporalAudit.hash.empty());
        const auto snapshot9 = std::find_if(
            temporalAudit.cohorts.begin(), temporalAudit.cohorts.end(),
            [](const auto& cohort) { return cohort.rankingSnapshotId == 9; });
        assert(snapshot9 != temporalAudit.cohorts.end());
        assert(snapshot9->rankingPopulationReconstructable);
        assert(snapshot9->exactControlReconstruction);
        assert(snapshot9->pointInTimeProvenanceViolationCount == 0);
        assert(snapshot9->validRankingTimeProfitabilityEvidenceCount == 2);
        assert(snapshot9->unavailableRankingTimeEvidenceCount == 1);
        assert(snapshot9->legitimateSubsequentOutcomeCount == 0);
        assert(snapshot9->classification == Verification::
            TemporalCohortClassification::insufficientSubsequentOutcome);
    }

    output.str({});
    output.clear();
    assert(Verification::RunCampaignProfitabilityTemporalValidationCommand(
               connectionString, output, errors) == 0);
    const std::string temporal = output.str();
    assert(temporal.find(
        "CAMPAIGN_PROFITABILITY_TEMPORAL_VALIDATION_START") !=
           std::string::npos);
    assert(temporal.find("precommitted_candidate_weight=0.025") !=
           std::string::npos);
    assert(temporal.find("phase10_sweep_repeated=false") !=
           std::string::npos);
    assert(temporal.find(
        "classification=insufficient_subsequent_outcome") !=
           std::string::npos);
    assert(temporal.find(
        "assessment=HISTORICAL_HOLDOUT_UNAVAILABLE_FORWARD_VALIDATION_REQUIRED") !=
           std::string::npos);
    assert(temporal.find("historical_top_5_result=unavailable") !=
           std::string::npos);
    assert(temporal.find("live_profitability_weight=0") !=
           std::string::npos);
    assert(temporal.find("recommendation_modified=false") !=
           std::string::npos);
    assert(temporal.find("ranking_snapshot_modified=false") !=
           std::string::npos);
    assert(temporal.find("worker_modified=false") != std::string::npos);
    std::ostringstream repeatedTemporal;
    assert(Verification::RunCampaignProfitabilityTemporalValidationCommand(
               connectionString, repeatedTemporal, errors) == 0);
    assert(repeatedTemporal.str() == temporal);

    output.str({});
    output.clear();
    assert(Verification::
        RunCampaignProfitabilityForwardValidationPrecommitCommand(
            connectionString, 9, "2026-09-02", "2027-09-02",
            output, errors) == 0);
    const std::string forward = output.str();
    assert(forward.find(
        "CAMPAIGN_PROFITABILITY_FORWARD_VALIDATION_PRECOMMIT") !=
           std::string::npos);
    assert(forward.find("control_weight=0,precommitted_candidate_weight=0.025") !=
           std::string::npos);
    assert(forward.find("weight_selected_from_future_outcome=false") !=
           std::string::npos);
    assert(forward.find("CAMPAIGN_PROFITABILITY_ASOF_MEMBER") !=
           std::string::npos);
    assert(forward.find("CAMPAIGN_PROFITABILITY_TEMPORAL_TOP_N") !=
           std::string::npos);
    assert(forward.find("subsequent_outcome_identity=PENDING") !=
           std::string::npos);
    assert(forward.find("experiment_created=false,experiment_queued=false") !=
           std::string::npos);
    assert(forward.find("scheduler_modified=false,worker_modified=false") !=
           std::string::npos);
    std::ostringstream repeatedForward;
    assert(Verification::
        RunCampaignProfitabilityForwardValidationPrecommitCommand(
            connectionString, 9, "2026-09-02", "2027-09-02",
            repeatedForward, errors) == 0);
    assert(repeatedForward.str() == forward);
    bool overlapRejected = false;
    try
    {
        std::ostringstream rejected;
        (void)Verification::
            RunCampaignProfitabilityForwardValidationPrecommitCommand(
                connectionString, 9, "2026-09-01", "2027-09-02",
                rejected, errors);
    }
    catch (const std::invalid_argument&)
    {
        overlapRejected = true;
    }
    assert(overlapRejected);

    pqxx::connection verifyConnection{connectionString};
    pqxx::read_transaction transaction{verifyConnection};
    assert(transaction.exec("SHOW transaction_read_only;")
               .one_row()[0].as<std::string>() == "on");
    assert(transaction.exec(
        "SELECT count(*) FROM experiment_recommendation;")
               .one_row()[0].as<int>() == 4);
    assert(transaction.exec(
        "SELECT count(*) FROM experiment_recommendation_ranking_member;")
               .one_row()[0].as<int>() == 7);
    assert(transaction.exec(
        "SELECT count(*) FROM experiment_recommendation_evaluation_result;")
               .one_row()[0].as<int>() == 3);
    assert(transaction.exec("SELECT count(*) FROM experiment;")
               .one_row()[0].as<int>() == 3);
    const pqxx::row sentinel = transaction.exec(
        "SELECT last_value,is_called FROM campaign_read_sentinel;").one_row();
    assert(sentinel["last_value"].as<long long>() == 1);
    assert(!sentinel["is_called"].as<bool>());
}

} // namespace

int main(int argc, char** argv)
{
    assert(argc == 2);
    const std::string mode = argv[1];
    if (mode == "setup")
        Setup();
    else
    {
        assert(mode == "verify");
        Verify();
    }
}
